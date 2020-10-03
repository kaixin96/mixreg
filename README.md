# Improving Generalization in Reinforcement Learning with Mixture Regularization

[site]() [paper]()

This repo contains code for our NeurIPS 2020 paper Improving Generalization in Reinforcement Learning with Mixture Regularization. Code for PPO is based on [train-procgen](https://github.com/openai/train-procgen). Code for Rainbow is based on [retro-baselines](https://github.com/openai/retro-baselines) and [anyrl-py](https://github.com/unixpickle/anyrl-py).

## 🍜 Set up conda env and install OpenAI Baselines

```bash
conda env create --file py36_cu9_tf112.yml
conda activate py36_cu9_tf112

git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

## 🌭 Experiments & results

Check out experiments README for running different experiments. You may also use the scripts in `experiments` folder to start training.

## Citation