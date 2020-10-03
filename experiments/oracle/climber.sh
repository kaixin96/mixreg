mpiexec -np 2 python -m train_procgen.train --env_name climber --level_setup oracle --run_id 1 --gpus_id 0,1
mpiexec -np 2 python -m train_procgen.train --env_name climber --level_setup oracle --run_id 2 --gpus_id 0,1
mpiexec -np 2 python -m train_procgen.train --env_name climber --level_setup oracle --run_id 3 --gpus_id 0,1
