"""
Modified from OpenAI Retro contest baseline

Source link: https://github.com/openai/retro-baselines
"""

import os
import argparse

import skimage
import tensorflow as tf
from baselines.common.mpi_util import setup_mpi_gpus, sync_from_root
from baselines.common.vec_env import VecExtractDictObs, VecMonitor
from baselines import logger
from mpi4py import MPI

from procgen import ProcgenEnv

from anyrl.rollouts import PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer

from .dqn_dist import rainbow_models
from .players import VecPlayer
from .dqn import MpiDQN
from .utils import REWARD_RANGE_FOR_C51

LOG_DIR = '~/procgen_exp/rainbow'

def main():
    """Run DQN until the environment throws an exception."""
    # Hyperparameters
    num_envs = 64
    learning_rate = 2.5e-4
    gamma = 0.99
    nstep_return = 3
    timesteps_per_proc = 25_000_000
    train_interval = 64
    target_interval = 8192
    batch_size = 512
    min_buffer_size = 20000

    # Parse arguments
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='hard',
                        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--gpus_id', type=str, default='')
    parser.add_argument('--level_setup', type=str, default='procgen',
                        choices=["procgen", "oracle"])
    parser.add_argument('--mix_mode', type=str, default='nomix',
                        choices=['nomix', 'mixreg'])
    parser.add_argument('--mix_alpha', type=float, default=0.2)
    parser.add_argument('--use_l2reg', action='store_true')
    parser.add_argument('--data_aug', type=str, default='no_aug',
                        choices=['no_aug', 'cutout_color', 'crop'])
    args = parser.parse_args()

    # Setup test worker
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    test_worker_interval = args.test_worker_interval
    is_test_worker = False
    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)
    mpi_rank_weight = 0 if is_test_worker else 1

    # Setup env specs
    if args.level_setup == "procgen":
        env_name = args.env_name
        num_levels = 0 if is_test_worker else args.num_levels
        start_level = args.start_level
    elif args.level_setup == "oracle":
        env_name = args.env_name
        num_levels = 0
        start_level = args.start_level

    # Setup logger
    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(
        dir=LOG_DIR + f'/{args.level_setup}/{args.mix_mode}/{env_name}/run_{args.run_id}',
        format_strs=format_strs
    )

    # Create env
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels,
                      start_level=start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)

    # Setup Tensorflow
    logger.info("creating tf session")
    if args.gpus_id:
        gpus_id = [x.strip() for x in args.gpus_id.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_id[rank % len(gpus_id)]
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    # Setup Rainbow models
    logger.info("building models")
    online_net, target_net = rainbow_models(sess,
                                            venv.action_space.n,
                                            gym_space_vectorizer(venv.observation_space),
                                            min_val=REWARD_RANGE_FOR_C51[env_name][0],
                                            max_val=REWARD_RANGE_FOR_C51[env_name][1])
    dqn = MpiDQN(online_net, target_net, discount=gamma,
                 comm=comm, mpi_rank_weight=mpi_rank_weight,
                 mix_mode=args.mix_mode, mix_alpha=args.mix_alpha,
                 use_l2reg=args.use_l2reg, data_aug=args.data_aug)
    player = NStepPlayer(VecPlayer(venv, dqn.online_net), nstep_return)
    optimize = dqn.optimize(learning_rate=learning_rate)

    # Initialize and sync variables
    sess.run(tf.global_variables_initializer())
    global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
    if comm.Get_size() > 1:
        sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E110

    # Training
    logger.info("training")
    dqn.train(num_steps=timesteps_per_proc,
              player=player,
              replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
              optimize_op=optimize,
              train_interval=train_interval,
              target_interval=target_interval,
              batch_size=batch_size,
              min_buffer_size=min_buffer_size)

if __name__ == '__main__':
    main()
