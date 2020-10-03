import tensorflow as tf
from baselines.common.policies import _normalize_clip_observation, PolicyWithValue
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.models import get_network_builder

from .utils import reduce_std

def build_policy(env, policy_network, value_network=None, normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None, randomization=True):
        ob_space = env.observation_space

        extra_tensors = {}

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=None)

        encoded_x = encode_observation(ob_space, X)

        # Randomization
        if randomization:
            encoded_x = tf.layers.conv2d(encoded_x / 255., 3, 3, padding='same',
                                         kernel_initializer=tf.initializers.glorot_normal(),
                                         trainable=False, name='randcnn') * 255.
            randcnn_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ppo2_model/randcnn")
            extra_tensors['randcnn_param'] = randcnn_param

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            extra_tensors['latent_fts'] = policy_latent
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
