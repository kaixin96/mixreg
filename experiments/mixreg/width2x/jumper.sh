mpiexec -np 3 python -m train_procgen.train --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 161 --gpus_id 0,1,2 --model_width 2x
mpiexec -np 3 python -m train_procgen.train --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 162 --gpus_id 0,1,2 --model_width 2x
mpiexec -np 3 python -m train_procgen.train --env_name jumper --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 163 --gpus_id 0,1,2 --model_width 2x
