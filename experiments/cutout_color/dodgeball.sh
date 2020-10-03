mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --test_worker_interval 3 --run_id 191 --gpus_id 0,1,2 --data_aug cutout_color
mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --test_worker_interval 3 --run_id 192 --gpus_id 0,1,2 --data_aug cutout_color
mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --test_worker_interval 3 --run_id 193 --gpus_id 0,1,2 --data_aug cutout_color
