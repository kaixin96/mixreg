"""
Modified from anyrl-py

Source link: https://github.com/unixpickle/anyrl-py
"""

import time
from collections import deque

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from baselines import logger
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
from baselines.common.mpi_util import sync_from_root

from .data_augs import Cutout_Color, Rand_Crop

class MpiDQN:
    def __init__(self, online_net, target_net, discount=0.99,
                 comm=None, mpi_rank_weight=1., log_interval=100,
                 mix_mode='nomix', mix_alpha=0.2, use_l2reg=False,
                 data_aug='no_aug'):
        """
        Create a Q-learning session.

        Args:
          online_net: the online TFQNetwork.
          target_net: the target TFQNetwork.
          discount: the per-step discount factor.
        """
        self.online_net = online_net
        self.target_net = target_net
        self.discount = discount

        self.comm = comm
        self.mpi_rank_weight = mpi_rank_weight
        self.log_interval = log_interval
        self.mix_mode = mix_mode
        self.mix_alpha = mix_alpha
        self.use_l2reg = use_l2reg
        self.data_aug = data_aug

        obs_shape = (None,) + online_net.obs_vectorizer.out_shape
        self.obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))
        self.rews_ph = tf.placeholder(tf.float32, shape=(None,))
        self.new_obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.terminals_ph = tf.placeholder(tf.bool, shape=(None,))
        self.discounts_ph = tf.placeholder(tf.float32, shape=(None,))
        self.weights_ph = tf.placeholder(tf.float32, shape=(None,))
        # Indices and coefficients for mixreg
        if mix_mode == 'mixreg':
            self.coeff_ph = tf.placeholder(tf.float32, [None])
            self.indices_ph = tf.placeholder(tf.int32, [None])
            self.other_indices_ph = tf.placeholder(tf.int32, [None])
            losses = online_net.transition_loss(
                target_net, self.obses_ph, self.actions_ph, self.rews_ph,
                self.new_obses_ph, self.terminals_ph, self.discounts_ph,
                self.coeff_ph, self.indices_ph, self.other_indices_ph
            )
        elif mix_mode == 'nomix':
            losses = online_net.transition_loss(
                target_net, self.obses_ph, self.actions_ph, self.rews_ph,
                self.new_obses_ph, self.terminals_ph, self.discounts_ph,
            )
        else:
            raise ValueError(f"Unknown mixing mode: {mix_mode} !")

        self.losses = self.weights_ph * losses
        self.loss = tf.reduce_mean(self.losses)

        assigns = []
        for dst, src in zip(target_net.variables, online_net.variables):
            assigns.append(tf.assign(dst, src))
        self.update_target = tf.group(*assigns)

    def feed_dict(self, transitions):
        """
        Generate a feed_dict that feeds the batch of
        transitions to the DQN loss terms.

        Args:
          transition: a sequence of transition dicts, as
            defined in anyrl.rollouts.ReplayBuffer.

        Returns:
          A dict which can be fed to tf.Session.run().
        """
        obs_vect = self.online_net.obs_vectorizer
        res = {
            self.obses_ph: obs_vect.to_vecs([t['obs'] for t in transitions]),
            self.actions_ph: [t['model_outs']['actions'] for t in transitions],
            self.rews_ph: [self._discounted_rewards(t['rewards']) for t in transitions],
            self.terminals_ph: [t['new_obs'] is None for t in transitions],
            self.discounts_ph: [(self.discount ** len(t['rewards'])) for t in transitions],
            self.weights_ph: [t['weight'] for t in transitions]
        }
        new_obses = []
        for trans in transitions:
            if trans['new_obs'] is None:
                new_obses.append(trans['obs'])
            else:
                new_obses.append(trans['new_obs'])
        res[self.new_obses_ph] = obs_vect.to_vecs(new_obses)

        # Feeds for mixreg
        if self.mix_mode == 'mixreg':
            # Generate mix coefficients and indices
            coeff = np.random.beta(self.mix_alpha, self.mix_alpha, size=(len(transitions),))
            seq_indices = np.arange(len(transitions))
            rand_indices = np.random.permutation(len(transitions))
            indices = np.where(coeff > 0.5, seq_indices, rand_indices)
            other_indices = np.where(coeff > 0.5, rand_indices, seq_indices)
            coeff = np.where(coeff > 0.5, coeff, 1 - coeff)
            # Add into feed dict
            res[self.coeff_ph] = coeff
            res[self.indices_ph] = indices
            res[self.other_indices_ph] = other_indices
        if self.data_aug != 'no_aug' and self.mpi_rank_weight > 0:
            res[self.obses_ph] = self.aug_func.do_augmentation(res[self.obses_ph])
            self.aug_func.change_randomization_params_all()

        return res

    def optimize(self, learning_rate=6.25e-5, epsilon=1.5e-4, **adam_kwargs):
        """
        Create a TF Op that optimizes the objective.

        Args:
          learning_rate: the Adam learning rate.
          epsilon: the Adam epsilon.
        """
        if self.comm is not None and self.comm.Get_size() > 1:
            optim = MpiAdamOptimizer(self.comm, learning_rate=learning_rate,
                                     mpi_rank_weight=self.mpi_rank_weight, epsilon=epsilon,
                                     **adam_kwargs)
        else:
            optim = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon,
                                           **adam_kwargs)
        if self.use_l2reg:
            params = tf.trainable_variables('online')
            weight_params = [v for v in params if '/bias' not in v.name]
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])
            self.loss = self.loss + l2_loss * 1e-4
        return optim.minimize(self.loss)

    def train(self,
              num_steps,
              player,
              replay_buffer,
              optimize_op,
              train_interval=1,
              target_interval=8192,
              batch_size=32,
              min_buffer_size=20000,
              tf_schedules=(),
              handle_ep=lambda steps, rew: None,
              timeout=None):
        """
        Run an automated training loop.

        This is meant to provide a convenient way to run a
        standard training loop without any modifications.
        You may get more flexibility by writing your own
        training loop.

        Args:
          num_steps: the number of timesteps to run.
          player: the Player for gathering experience.
          replay_buffer: the ReplayBuffer for experience.
          optimize_op: a TF Op to optimize the model.
          train_interval: timesteps per training step.
          target_interval: number of timesteps between
            target network updates.
          batch_size: the size of experience mini-batches.
          min_buffer_size: minimum replay buffer size
            before training is performed.
          tf_schedules: a sequence of TFSchedules that are
            updated with the number of steps taken.
          handle_ep: called with information about every
            completed episode.
          timeout: if set, this is a number of seconds
            after which the training loop should exit.
        """
        sess = self.online_net.session
        sess.run(self.update_target)
        steps_taken = 0
        next_target_update = target_interval
        next_train_step = train_interval
        start_time = time.time()

        eprew_buf = deque(maxlen=100)
        eplen_buf = deque(maxlen=100)
        loss_buf = deque(maxlen=self.log_interval)
        n_updates = 0

        if self.data_aug != 'no_aug' and self.mpi_rank_weight > 0:
            if self.data_aug == "cutout_color":
                self.aug_func = Cutout_Color(batch_size=batch_size)
            elif self.data_aug == "crop":
                self.aug_func = Rand_Crop(batch_size=batch_size, sess=sess)
            else:
                raise ValueError("Invalid value for argument data_aug.")

        while steps_taken < num_steps:
            if timeout is not None and time.time() - start_time > timeout:
                return
            transitions = player.play()
            for trans in transitions:
                if trans['is_last']:
                    eprew_buf.append(trans['total_reward'])
                    eplen_buf.append(trans['episode_step'] + 1)
                    # handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                replay_buffer.add_sample(trans)
                steps_taken += 1
                for sched in tf_schedules:
                    sched.add_time(sess, 1)
                if replay_buffer.size >= min_buffer_size and steps_taken >= next_train_step:
                    next_train_step = steps_taken + train_interval
                    batch = replay_buffer.sample(batch_size)
                    feed_dict = self.feed_dict(batch)
                    _, losses = sess.run((optimize_op, self.losses),
                                         feed_dict=feed_dict)
                    # gather batch
                    if self.mix_mode == 'mixreg':
                        batch = [batch[i] for i in feed_dict[self.indices_ph]]
                    replay_buffer.update_weights(batch, losses)
                    loss_buf.append(np.mean(losses))
                    n_updates += 1
                    # logging
                    if n_updates % self.log_interval == 0:
                        logger.logkv('misc/is_test_work', self.mpi_rank_weight == 0)
                        logger.logkv('eprewmean', np.mean(eprew_buf))
                        logger.logkv('eplenmean', np.mean(eplen_buf))
                        logger.logkv('loss', np.mean(loss_buf))
                        logger.logkv('misc/time_elapsed', time.time() - start_time)
                        logger.logkv('misc/steps_taken', steps_taken)
                        logger.dumpkvs()
                if steps_taken >= next_target_update:
                    next_target_update = steps_taken + target_interval
                    sess.run(self.update_target)

    def _discounted_rewards(self, rews):
        res = 0
        for i, rew in enumerate(rews):
            res += rew * (self.discount ** i)
        return res
