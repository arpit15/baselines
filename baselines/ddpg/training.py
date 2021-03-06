import os
import time
from collections import deque
import pickle
import math 

from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import os.path as osp
from ipdb import set_trace

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    
    if "dologging" in kwargs: 
        dologging = kwargs["dologging"]
    else:
        dologging = True

    if "tf_sum_logging" in kwargs: 
        tf_sum_logging = kwargs["tf_sum_logging"]
    else:
        tf_sum_logging = False
        
    if "invert_grad" in kwargs: 
        invert_grad = kwargs["invert_grad"]
    else:
        invert_grad = False

    if dologging: logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale,
        inverting_grad = invert_grad
        )
    if dologging: logger.info('Using agent with the following configuration:')
    if dologging: logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=5)
        save_freq = kwargs["save_freq"]
    else:
        saver = None
    

    # step = 0
    global_t = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    
   
    with U.single_threaded_session() as sess:
        
        # Set summary saver
        if dologging and tf_sum_logging: 
            tf.summary.histogram("actor_grads", agent.actor_grads)
            tf.summary.histogram("critic_grads", agent.critic_grads)
            actor_trainable_vars = actor.trainable_vars
            for var in actor_trainable_vars:
                tf.summary.histogram(var.name, var)
            critic_trainable_vars = critic.trainable_vars
            for var in critic_trainable_vars:
                tf.summary.histogram(var.name, var)

            summary_var = tf.summary.merge_all()
            writer_t = tf.summary.FileWriter(osp.join(logger.get_dir(), 'train'), sess.graph)
        else:
            summary_var = tf.no_op()

        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        ## restore
        if kwargs["restore_dir"] is not None:
            restore_dir = osp.join(kwargs["restore_dir"], "model")
            if (restore_dir is not None) and rank==0:
                print('Restore path : ',restore_dir)
                checkpoint = tf.train.get_checkpoint_state(restore_dir)
                if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(U.get_session(), checkpoint.model_checkpoint_path)
                    print( "checkpoint loaded:" , checkpoint.model_checkpoint_path)
                    logger.info("checkpoint loaded:" + str(checkpoint.model_checkpoint_path))
                    tokens = checkpoint.model_checkpoint_path.split("-")[-1]
                    # set global step
                    global_t = int(tokens)
                    print( ">>> global step set:", global_t)
        
        agent.reset()
        obs = env.reset()
        
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        print("Training!")
        for epoch in range(global_t, nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    #if((t+1)%100) == 0:
                    #    print(max_action*action, new_obs, r)
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()
                        #print(obs)

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al, current_summary = agent.train(summary_var)
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                    if dologging and tf_sum_logging:
                        
                        writer_t.add_summary(current_summary, epoch*nb_epoch_cycles*nb_train_steps + cycle*nb_train_steps + t_train)

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                
                if eval_env is not None:
                    eval_episode_reward = 0.
                    eval_obs = eval_env.reset()
                    eval_done = False
                    while(not eval_done):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            print("Render!")
                            
                            eval_env.render()
                            print("rendered!")
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        
                    eval_episode_rewards.append(eval_episode_reward)
                    eval_episode_rewards_history.append(eval_episode_reward)
                    
                    # for t_rollout in range(nb_eval_steps):
                    #     eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                    #     eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    #     if render_eval:
                    #         print("Render!")
                            
                    #         eval_env.render()
                    #         print("rendered!")
                    #     eval_episode_reward += eval_r

                    #     eval_qs.append(eval_q)
                    #     if eval_done:
                    #         eval_obs = eval_env.reset()
                    #         eval_episode_rewards.append(eval_episode_reward)
                    #         eval_episode_rewards_history.append(eval_episode_reward)
                    #         eval_episode_reward = 0.

                    # if not eval_done:
                    #     # eval_obs = eval_env.reset()
                    #     eval_episode_rewards.append(eval_episode_reward)
                    #     eval_episode_rewards_history.append(eval_episode_reward)
                    #     eval_episode_reward = 0.

            if dologging: 
                # Log stats.
                epoch_train_duration = time.time() - epoch_start_time
                duration = time.time() - start_time
                stats = agent.get_stats()
                combined_stats = {}
                for key in sorted(stats.keys()):
                    combined_stats[key] = mpi_mean(stats[key])

                # Rollout statistics.
                combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
                if len(episode_rewards_history)>0:
                    combined_stats['rollout/return_history'] = mpi_mean( np.mean(episode_rewards_history))
                else:
                    combined_stats['rollout/return_history'] = 0.
                combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
                combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
                combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
                combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
                combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)
        
                # Train statistics.
                combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
                combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
                combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

                # Evaluation statistics.
                if eval_env is not None:
                    combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
                    if len(eval_episode_rewards_history) > 0:
                        combined_stats['eval/return_history'] = mpi_mean( np.mean(eval_episode_rewards_history) )
                    else:
                        combined_stats['eval/return_history'] = 0.
                    combined_stats['eval/Q'] = mpi_mean(eval_qs)
                    #combined_stats['eval/episodes'] = mpi_mean(len(eval_episode_rewards))

                # Total statistics.
                combined_stats['total/duration'] = mpi_mean(duration)
                combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
                combined_stats['total/episodes'] = mpi_mean(episodes)
                combined_stats['total/epochs'] = epoch + 1
                combined_stats['total/steps'] = t
                
                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                logger.dump_tabular()
                logger.info('')
                logdir = logger.get_dir()
                if rank == 0 and logdir:
                    if hasattr(env, 'get_state'):
                        with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                            pickle.dump(env.get_state(), f)
                    if eval_env and hasattr(eval_env, 'get_state'):
                        with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                            pickle.dump(eval_env.get_state(), f)


                ## save tf model
                if rank==0 and (epoch+1)%save_freq == 0:
                    os.makedirs(osp.join(logdir, "model"), exist_ok=True)
                    saver.save(U.get_session(), logdir+"/model/ddpg", global_step = epoch)





