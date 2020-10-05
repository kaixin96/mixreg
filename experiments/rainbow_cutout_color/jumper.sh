mpiexec -np 9 python -m train_procgen.train_dqn --env_name jumper --num_levels 500 --data_aug cutout_color --test_worker_interval 9 --run_id 321 --gpus_id 0,1,2
mpiexec -np 9 python -m train_procgen.train_dqn --env_name jumper --num_levels 500 --data_aug cutout_color --test_worker_interval 9 --run_id 322 --gpus_id 0,1,2
mpiexec -np 9 python -m train_procgen.train_dqn --env_name jumper --num_levels 500 --data_aug cutout_color --test_worker_interval 9 --run_id 323 --gpus_id 0,1,2
