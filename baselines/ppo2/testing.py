import baselines.ppo2.ppo2 as ppo2_main_file
import os.path as osp
import numpy as np
from time import sleep
import baselines.common.tf_util as U
import tensorflow as tf
import os

def test(*, policy, env, nsteps, total_timesteps, ent_coef, lr, 
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            nminibatches=4, noptepochs=4, cliprange=0.2,
            restore_dir=None, render_eval = True):

    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # saver = tf.train.Saver()

    make_model = lambda : ppo2_main_file.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    
    if restore_dir:
        import cloudpickle
        with open(osp.join(restore_dir, 'make_model.pkl'), 'rb') as fh:
            make_model = cloudpickle.loads(fh.read())
    model = make_model()        ## calls tf init

    ## restore
    restore_dir_ckpt = osp.join(restore_dir, "checkpoints")
    numfiles = len(os.listdir(restore_dir_ckpt))
    model.load(osp.join(restore_dir_ckpt, '%.5i'%((numfiles-1)*10)))
    
    # Evaluate.
    eval_episode_rewards = []
    eval_episode_rewards_history = []

    from ipdb import set_trace

    for i in range(10):
        print("Evaluating:%d"%(i+1))
        eval_episode_reward = 0.
        eval_done = False
        states = None
        eval_obs = np.zeros(env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        eval_obs = env.reset()
        while(not eval_done):
            eval_actions, _, states, _ = model.step(eval_obs[np.newaxis,:])
            eval_obs, rewards, eval_done, infos = env.step(eval_actions[0])
            
            if render_eval:
                env.render()
                sleep(0.01)

            eval_episode_reward += rewards

        print("episode reward::%f"%eval_episode_reward)
        eval_episode_rewards.append(eval_episode_reward)
        eval_episode_rewards_history.append(eval_episode_reward)
        eval_episode_reward = 0.

    print("episode reward - mean:%.4f, var:%.4f"%(np.mean(eval_episode_rewards), np.var(eval_episode_rewards)))     
        
    env.close()