mpiexec -np 3 python -m train_procgen.train_netrand --env_name climber --num_levels 500 --test_worker_interval 3 --run_id 101 --gpus_id 0,1,2
mpiexec -np 3 python -m train_procgen.train_netrand --env_name climber --num_levels 500 --test_worker_interval 3 --run_id 102 --gpus_id 0,1,2
mpiexec -np 3 python -m train_procgen.train_netrand --env_name climber --num_levels 500 --test_worker_interval 3 --run_id 103 --gpus_id 0,1,2
