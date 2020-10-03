mpiexec -np 2 python -m train_procgen.train_finetune --run_id 1 --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/mixreg/dodgeball/run_1/checkpoints/final_model.ckpt"
mpiexec -np 2 python -m train_procgen.train_finetune --run_id 2 --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/mixreg/dodgeball/run_1/checkpoints/final_model.ckpt"
mpiexec -np 2 python -m train_procgen.train_finetune --run_id 3 --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/mixreg/dodgeball/run_1/checkpoints/final_model.ckpt"
