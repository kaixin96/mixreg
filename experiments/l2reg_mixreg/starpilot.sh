mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 201 --gpus_id 0,1,2 --use_l2reg
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 202 --gpus_id 0,1,2 --use_l2reg
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --run_id 203 --gpus_id 0,1,2 --use_l2reg
