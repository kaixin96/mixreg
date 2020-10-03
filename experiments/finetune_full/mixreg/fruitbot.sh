mpiexec -np 2 python -m train_procgen.train_finetune --finetune_full --run_id 101 --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/mixreg/fruitbot/run_1/checkpoints/final_model.ckpt"
mpiexec -np 2 python -m train_procgen.train_finetune --finetune_full --run_id 102 --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/mixreg/fruitbot/run_1/checkpoints/final_model.ckpt"
mpiexec -np 2 python -m train_procgen.train_finetune --finetune_full --run_id 103 --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/mixreg/fruitbot/run_1/checkpoints/final_model.ckpt"
