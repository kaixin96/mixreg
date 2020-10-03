import os
import argparse

import skimage
import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize
from baselines import logger
from procgen import ProcgenEnv
from mpi4py import MPI

from .model import get_mixreg_model
from .ppo2 import learn
from .network import build_impala_cnn

LOG_DIR = '~/procgen_exp/ppo'


def main():
    # Hyperparameters
    num_envs = 128
    learning_rate = 5e-4
    ent_coef = .01
    vf_coef = 0.5
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    max_grad_norm = 0.5
    timesteps_per_proc = 100_000_000
    use_vf_clipping = True

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--distribution_mode', type=str, default='hard',
                        choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--gpus_id', type=str, default='')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--use_l2reg', action='store_true')
    parser.add_argument('--l2reg_coeff', type=float, default=1e-4)
    parser.add_argument('--data_aug', type=str, default='no_aug', choices=["no_aug", "cutout_color", "crop"])
    parser.add_argument('--use_rand_conv', action='store_true')
    parser.add_argument('--model_width', type=str, default='1x', choices=["1x", "2x", "4x"])
    parser.add_argument('--level_setup', type=str, default='procgen',
                        choices=["procgen", "oracle"])
    parser.add_argument('--mix_mode', type=str, default='nomix',
                        choices=['nomix', 'mixreg', 'mixobs'])
    parser.add_argument('--mix_alpha', type=float, default=0.2)
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
        dir=LOG_DIR +
        f'/{args.level_setup}/{args.mix_mode}/{env_name}/run_{args.run_id}',
        format_strs=format_strs
    )

    # Create env
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels,
                      start_level=start_level, distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)

    # Setup Tensorflow
    logger.info("creating tf session")
    if args.gpus_id:
        gpus_id = [x.strip() for x in args.gpus_id.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_id[rank]
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    # Setup model
    if args.model_width == '1x':
        depths = [16, 32, 32]
    elif args.model_width == '2x':
        depths = [32, 64, 64]
    elif args.model_width == '4x':
        depths = [64, 128, 128]
    conv_fn = lambda x: build_impala_cnn(x, depths=depths, use_bn=args.use_bn,
                                            randcnn=args.use_rand_conv and not is_test_worker)

    # Training
    logger.info("training")
    ppo2.learn = learn  # use customized "learn" function
    model = ppo2.learn(
        env=venv,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=0,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        data_aug=args.data_aug,
        use_rand_conv=args.use_rand_conv,
        model_fn=get_mixreg_model(mix_mode=args.mix_mode, mix_alpha=args.mix_alpha,
                                  use_l2reg=args.use_l2reg, l2reg_coeff=args.l2reg_coeff),
    )

    # Saving
    logger.info("saving final model")
    if rank == 0:
        checkdir = os.path.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        model.save(os.path.join(checkdir, 'final_model.ckpt'))


if __name__ == '__main__':
    main()
