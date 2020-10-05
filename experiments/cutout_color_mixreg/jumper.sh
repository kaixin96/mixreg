mpiexec -np 3 python -m train_procgen.train --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 391 --gpus_id 0,1,2 --data_aug cutout_color
mpiexec -np 3 python -m train_procgen.train --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 392 --gpus_id 0,1,2 --data_aug cutout_color
mpiexec -np 3 python -m train_procgen.train --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 393 --gpus_id 0,1,2 --data_aug cutout_color
