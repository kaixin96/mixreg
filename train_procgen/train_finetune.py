import os
import argparse
from pathlib import Path

import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize
from baselines import logger
from procgen import ProcgenEnv
from mpi4py import MPI


from .model import get_mixreg_model
from .ppo2 import learn

LOG_DIR = '~/procgen_exp/finetune'

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
    timesteps_per_proc = 30_000_000
    use_vf_clipping = True

    # Parse arguments
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--gpus_id', type=str, default='')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--finetune_full', action='store_true')
    args = parser.parse_args()
    if args.finetune_full:
        timesteps_per_proc = 100_000_000

    # Setup test worker
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    test_worker_interval = args.test_worker_interval
    is_test_worker = False
    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)
    mpi_rank_weight = 0 if is_test_worker else 1

    # Setup env specs from the path of trained model
    load_path = Path(args.load_path)
    env_name = load_path.parents[2].name
    mix_mode = load_path.parents[3].name
    level_setup = load_path.parents[4].name
    num_levels = 0
    start_level = args.start_level

    # Setup logger
    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=LOG_DIR + f'/{level_setup}/{mix_mode}/{env_name}/run_{args.run_id}', format_strs=format_strs)

    # Create env
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=args.distribution_mode,
                      num_threads=16)
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
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    # Setup model
    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32])

    # Training
    logger.info("training")
    ppo2.learn = learn
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
        model_fn=get_mixreg_model(fix_representation=not args.finetune_full),
        load_path=load_path,
    )

    # Saving
    logger.info("saving final model")
    if rank == 0:
        checkdir = os.path.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        model.save(os.path.join(checkdir, 'final_model.ckpt'))

if __name__ == '__main__':
    main()
