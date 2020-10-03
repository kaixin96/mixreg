mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --test_worker_interval 3 --run_id 1 --gpus_id 0,1,2
mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --test_worker_interval 3 --run_id 2 --gpus_id 0,1,2
mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --test_worker_interval 3 --run_id 3 --gpus_id 0,1,2
