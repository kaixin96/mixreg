mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 164 --gpus_id 0,1,2 --model_width 4x
mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 165 --gpus_id 0,1,2 --model_width 4x
mpiexec -np 3 python -m train_procgen.train --env_name dodgeball --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 166 --gpus_id 0,1,2 --model_width 4x
