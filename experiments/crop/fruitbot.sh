mpiexec -np 3 python -m train_procgen.train --env_name fruitbot --num_levels 500 --test_worker_interval 3 --run_id 301 --gpus_id 0,1,2 --data_aug crop
mpiexec -np 3 python -m train_procgen.train --env_name fruitbot --num_levels 500 --test_worker_interval 3 --run_id 302 --gpus_id 0,1,2 --data_aug crop
mpiexec -np 3 python -m train_procgen.train --env_name fruitbot --num_levels 500 --test_worker_interval 3 --run_id 303 --gpus_id 0,1,2 --data_aug crop
