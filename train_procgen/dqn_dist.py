"""
Modified from anyrl-py

Source link: https://github.com/unixpickle/anyrl-py
"""

from functools import partial
from math import log

import tensorflow as tf
from anyrl.models.dqn_dist import DistQNetwork, _kl_divergence
from anyrl.models.dqn_scalar import noisy_net_dense
from anyrl.models.util import take_vector_elems

from .network import build_impala_cnn

def rainbow_models(session,
                   num_actions,
                   obs_vectorizer,
                   num_atoms=51,
                   min_val=-10,
                   max_val=10,
                   sigma0=0.5):
    """
    Create the models used for Rainbow
    (https://arxiv.org/abs/1710.02298).

    Args:
      session: the TF session.
      num_actions: size of action space.
      obs_vectorizer: observation vectorizer.
      num_atoms: number of distribution atoms.
      min_val: minimum atom value.
      max_val: maximum atom value.
      sigma0: initial Noisy Net noise.

    Returns:
      A tuple (online, target).
    """
    def maker(name):
        return ImpalaDistQNetwork(session, num_actions, obs_vectorizer, name,
                                  num_atoms, min_val, max_val, dueling=True,
                                  dense=partial(noisy_net_dense, sigma0=sigma0))
    return maker('online'), maker('target')

class ImpalaDistQNetwork(DistQNetwork):
    """
    A distributional Q-network model based on the IMPALA network
    """
    def __init__(self,
                 session,
                 num_actions,
                 obs_vectorizer,
                 name,
                 num_atoms,
                 min_val,
                 max_val,
                 dueling=False,
                 dense=tf.layers.dense,
                 input_dtype=tf.uint8,
                 input_scale=1 / 0xff):
        self._input_dtype = input_dtype
        self.input_scale = input_scale
        super().__init__(session, num_actions, obs_vectorizer, name,
                         num_atoms, min_val, max_val,
                         dueling=dueling, dense=dense)

    @property
    def input_dtype(self):
        return self._input_dtype

    def base(self, obs_batch):
        return build_impala_cnn(obs_batch, dense=self.dense)

    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts,
                        coeff=None, indices=None, other_indices=None):
        discounts = tf.where(terminals, tf.zeros_like(discounts), discounts)
        with tf.variable_scope(self.name, reuse=True):
            max_actions = tf.argmax(self.dist.mean(self.value_func(self.base(new_obses))),
                                    axis=1, output_type=tf.int32)
        with tf.variable_scope(target_net.name, reuse=True):
            target_preds = target_net.value_func(target_net.base(new_obses))
            target_preds = tf.where(terminals,
                                    tf.zeros_like(target_preds) - log(self.dist.num_atoms),
                                    target_preds)
            if coeff is not None and indices is not None and other_indices is not None:
                # mix Q'
                target_preds = tf.exp(target_preds)
                coeff_tgt = tf.reshape(coeff, (-1, 1, 1))
                target_preds = coeff_tgt * tf.gather(target_preds, indices, axis=0) + (1 - coeff_tgt) * tf.gather(target_preds, other_indices, axis=0)
                # mix reward
                rews = coeff * tf.gather(rews, indices, axis=0) + (1 - coeff) * tf.gather(rews, other_indices, axis=0)
                # gather action
                max_actions = tf.gather(max_actions, indices, axis=0)
                # gather discounts
                discounts = tf.gather(discounts, indices, axis=0)
                target_dists = self.dist.add_rewards(take_vector_elems(target_preds, max_actions),
                                                    rews, discounts)
            else:
                target_dists = self.dist.add_rewards(tf.exp(take_vector_elems(target_preds, max_actions)),
                                                    rews, discounts)
        with tf.variable_scope(self.name, reuse=True):
            if coeff is not None and indices is not None and other_indices is not None:
                # mix obs
                coeff_obs = tf.reshape(coeff, (-1, 1, 1, 1))
                obses = tf.cast(obses, tf.float32)
                obses = coeff_obs * tf.gather(obses, indices, axis=0) + (1 - coeff_obs) * tf.gather(obses, other_indices, axis=0)
                obses = tf.cast(obses, tf.uint8)
                actions = tf.gather(actions, indices, axis=0)
            online_preds = self.value_func(self.base(obses))
            onlines = take_vector_elems(online_preds, actions)
            return _kl_divergence(tf.stop_gradient(target_dists), onlines)
