mpiexec -np 3 python -m train_procgen.train --env_name caveflyer --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 401 --gpus_id 0,1,2 --data_aug crop
mpiexec -np 3 python -m train_procgen.train --env_name caveflyer --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 402 --gpus_id 0,1,2 --data_aug crop
mpiexec -np 3 python -m train_procgen.train --env_name caveflyer --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 403 --gpus_id 0,1,2 --data_aug crop
