"""
Modified from anyrl-py

Source link: https://github.com/unixpickle/anyrl-py
"""

import numpy as np

from anyrl.rollouts.players import Player

class VecPlayer(Player):
    """
    A Player that uses a VecEnv to gather transitions.
    """
    def __init__(self, venv, model, num_timesteps=1):
        self.venv = venv
        self.model = model
        self.num_timesteps = num_timesteps
        self._last_obses = None
        self._episode_ids = np.arange(venv.num_envs)
        self._episode_steps = np.zeros(venv.num_envs, dtype='int')
        self._total_rewards = np.zeros(venv.num_envs, dtype='float')

    def play(self):
        if self._last_obses is None:
            self._last_obses = self.venv.reset()
        results = []
        for _ in range(self.num_timesteps):
            results.extend(self._step())
        return results

    def _step(self):
        model_outs = self.model.step(self._last_obses, None)
        obs, rew, done, info = self.venv.step(model_outs['actions'])
        self._total_rewards += rew
        transitions = [{
            'obs': self._last_obses[i],
            'model_outs': {
                'actions': model_outs['actions'][i],
                'action_values': model_outs['action_values'][i],
                'action_dists': model_outs['action_dists'][i],
            },
            'rewards': [rew[i]],
            'new_obs': (obs[i] if not done[i] else None),
            'episode_id': self._episode_ids[i],
            'episode_step': self._episode_steps[i],
            'is_last': done[i],
            'total_reward': self._total_rewards[i]
        } for i in range(self.venv.num_envs)]

        self._episode_ids[done] += self.venv.num_envs
        self._episode_steps[done] = 0
        self._episode_steps[~done] += 1
        self._total_rewards[done] = 0.0
        self._last_obses = obs

        return transitions
