# Improving Generalization in Reinforcement Learning with Mixture Regularization

[[site]](https://policy.fit/projects/mixreg/index.html) [[paper]](https://arxiv.org/abs/2010.10814)

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

Check out [experiments README](https://github.com/kaixin96/mixreg/blob/master/experiments/README.md) for running different experiments. You may also use the scripts in `experiments` folder to start training. All results are available at [Google Drive](https://drive.google.com/drive/folders/1wTURCswt6IfTDbEkBqMaIZhBlO7n8qDb?usp=sharing).

## Citation
@misc{wang2020improving,
      title={Improving Generalization in Reinforcement Learning with Mixture Regularization}, 
      author={Kaixin Wang and Bingyi Kang and Jie Shao and Jiashi Feng},
      year={2020},
      eprint={2010.10814},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}