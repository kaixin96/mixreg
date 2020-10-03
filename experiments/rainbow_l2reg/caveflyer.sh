mpiexec -np 9 python -m train_procgen.train_dqn --env_name caveflyer --num_levels 500 --use_l2reg --test_worker_interval 9 --run_id 311 --gpus_id 0,1,2
mpiexec -np 9 python -m train_procgen.train_dqn --env_name caveflyer --num_levels 500 --use_l2reg --test_worker_interval 9 --run_id 312 --gpus_id 0,1,2
mpiexec -np 9 python -m train_procgen.train_dqn --env_name caveflyer --num_levels 500 --use_l2reg --test_worker_interval 9 --run_id 313 --gpus_id 0,1,2
