import tensorflow as tf
from baselines.common.policies import _normalize_clip_observation, PolicyWithValue
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.models import get_network_builder

from .utils import reduce_std

def build_policy(env, policy_network, value_network=None, normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None, mix_mode='nomix'):
        ob_space = env.observation_space

        extra_tensors = {}

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=None)

        if mix_mode in ['mixreg', 'mixobs']:
            COEFF = tf.placeholder(tf.float32, [None])
            INDICES = tf.placeholder(tf.int32, [None])
            OTHER_INDICES = tf.placeholder(tf.int32, [None])
            coeff = tf.reshape(COEFF, (-1, 1, 1, 1))
            encoded_x = tf.cast(X, tf.float32)
            encoded_x = coeff * tf.gather(encoded_x, INDICES, axis=0) + (1 - coeff) * tf.gather(encoded_x, OTHER_INDICES, axis=0)
            encoded_x = tf.cast(encoded_x, tf.uint8)
            extra_tensors['coeff'] = COEFF
            extra_tensors['indices'] = INDICES
            extra_tensors['other_indices'] = OTHER_INDICES
        elif mix_mode == 'nomix':
            encoded_x = X
        else:
            raise ValueError(f"Unknown mixing mode: {mix_mode} !")

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn
