import argparse
from baselines import bench, logger
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

def test(env_id, num_timesteps, seed, restore_dir, render_eval):
    from baselines.common import set_global_seeds
    # from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import testing
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    # from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        # env = bench.Monitor(env, logger.get_dir())
        return env

    env = make_env()
    
    set_global_seeds(seed)
    policy = MlpPolicy
    testing.test(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        restore_dir=restore_dir, render_eval = render_eval)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--restore-dir', type=str, default='/home/arpit/new_RL3/baseline_results/Hopper-v1/run4')
    boolean_flag(parser, 'render-eval', default=True)
    args = parser.parse_args()
    # logger.configure()
    test(args.env, num_timesteps=args.num_timesteps, seed=args.seed, restore_dir=args.restore_dir, render_eval = args.render_eval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting!")

