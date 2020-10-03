"""
Runner wrapper to add augmentation
"""

import itertools

import numpy as np
from baselines.ppo2.runner import Runner, sf01

from .data_augs import Cutout_Color, Rand_Crop

class RunnerWithAugs(Runner):
    def __init__(self, *, env, model, nsteps, gamma, lam, data_aug="no_aug", is_train=True):
        super().__init__(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
        self.data_aug = data_aug
        self.is_train = is_train
        if self.is_train and self.data_aug != "no_aug":
            if self.data_aug == "cutout_color":
                self.aug_func = Cutout_Color(batch_size=self.obs.shape[0])
            elif self.data_aug == "crop":
                self.aug_func = Rand_Crop(batch_size=self.obs.shape[0], sess=model.sess)
            else:
                raise ValueError("Invalid value for argument data_aug.")
            self.obs = self.aug_func.do_augmentation(self.obs)

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]

        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs, rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

            if self.data_aug != 'no_aug' and self.is_train:
                self.obs[:] = self.aug_func.do_augmentation(obs)
            else:
                self.obs[:] = obs

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        if self.data_aug != 'no_aug' and self.is_train:
            self.aug_func.change_randomization_params_all()
            self.obs = self.aug_func.do_augmentation(obs)

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
