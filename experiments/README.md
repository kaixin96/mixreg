## Reproduce procgen baseline results

> Training progress files for procgen baseline results are [available](https://github.com/openai/train-procgen/tree/master/train_procgen/results) now.

500-level generalization
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --test_worker_interval 3 --gpus_id 0,1,2
```

Varying number of training levels
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 100 --test_worker_interval 3 --gpus_id 0,1,2
```

Varying number of convolutional channels
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --test_worker_interval 3 --gpus_id 0,1,2 --model_width 2x
```

## Apply mixreg on PPO

500-level generalization
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --gpus_id 0,1,2
```

Vary the number of training levels
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 100 --mix_mode mixreg --test_worker_interval 3 --gpus_id 0,1,2
```

Vary the number of convolutional channels
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --gpus_id 0,1,2 --model_width 2x
```

Vary Beta distribution parameter $\alpha$ (default is 0.2)
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --mix_alpha 0.1 --test_worker_interval 3 --gpus_id 0,1,2
```

## Other methods used for comparison

[Network randomization](https://arxiv.org/abs/1910.05396) (without test time MC approximation)
```bash
mpiexec -np 3 python -m train_procgen.train_netrand --env_name starpilot --num_levels 500 --test_worker_interval 3 --gpus_id 0,1,2
```

Random convolution (similar to network randomization but simpler to implement) described in [RAD](https://arxiv.org/abs/2004.14990)
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --test_worker_interval 3 --gpus_id 0,1,2 --use_rand_conv
```

L2 regularization
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --test_worker_interval 3 --gpus_id 0,1,2 --use_l2reg
```

Batch normalization
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --test_worker_interval 3 --gpus_id 0,1,2 --use_bn
```

Data augmentation: cutout color
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --test_worker_interval 3 --gpus_id 0,1,2 --data_aug cutout_color
```

Data augmentation: random crop
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --test_worker_interval 3 --gpus_id 0,1,2 --data_aug crop
```

## Rainbow experiments

500-level generalization baseline
```bash
mpiexec -np 9 python -m train_procgen.train_dqn --env_name starpilot --num_levels 500 --test_worker_interval 9 --gpus_id 0,1,2
```

Apply mixreg
```bash
mpiexec -np 9 python -m train_procgen.train_dqn --env_name starpilot --num_levels 500 --mix_mode mixreg --test_worker_interval 9 --gpus_id 0,1,2
```

Use L2 regularization
```bash
mpiexec -np 9 python -m train_procgen.train_dqn --env_name starpilot --num_levels 500 --use_l2reg --test_worker_interval 9 --gpus_id 0,1,2
```

## Combine mixreg with other regularization / data augmentation methods

Mixreg + L2 regularization
```bash
mpiexec -np 3 python -m train_procgen.train --env_name starpilot --num_levels 500 --mix_mode mixreg --test_worker_interval 3 --gpus_id 0,1,2 --use_l2reg
```

## Finetuning experiments

Finetune baseline model with representation fixed
```bash
mpiexec -np 2 python -m train_procgen.train_finetune --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/no_mix/starpilot/run_1/checkpoints/final_model.ckpt"
```

Finetune mixreg model with representation fixed
```bash
mpiexec -np 2 python -m train_procgen.train_finetune --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/mixreg/starpilot/run_1/checkpoints/final_model.ckpt"
```

Finetune baseline model
```bash
mpiexec -np 2 python -m train_procgen.train_finetune --finetune_full --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/no_mix/starpilot/run_1/checkpoints/final_model.ckpt"
```

Finetune mixreg model
```bash
mpiexec -np 2 python -m train_procgen.train_finetune --finetune_full --gpus_id 0,1 --load_path "~/procgen_exp/ppo/procgen/mixreg/starpilot/run_1/checkpoints/final_model.ckpt"
```

Training baseline model on full distribution of levels (i.e. oracle in the paper)
```bash
mpiexec -np 2 python -m train_procgen.train --env_name starpilot --level_setup oracle --gpus_id 0,1
```