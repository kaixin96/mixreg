mpiexec -np 9 python -m train_procgen.train_dqn --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 9 --run_id 1 --gpus_id 0,1,2
mpiexec -np 9 python -m train_procgen.train_dqn --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 9 --run_id 2 --gpus_id 0,1,2
mpiexec -np 9 python -m train_procgen.train_dqn --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 9 --run_id 3 --gpus_id 0,1,2
