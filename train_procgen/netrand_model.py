import numpy as np
import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    import baselines.common.mpi_adam_optimizer as mpi_adam
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

from .utils import reduce_std

# Disable sync check because each worker has different rand conv weights
mpi_adam.check_synced = lambda: None

class NetRandModel:
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None,
                fm_coeff=0.002):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)
            act_model_clean = policy(nbatch_act, 1, sess, randomization=False)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
                train_model_clean = policy(nbatch_train, nsteps, sess, randomization=False)
            else:
                train_model = policy(microbatch_size, nsteps, sess)
                train_model_clean = policy(microbatch_size, nsteps, sess, randomization=False)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        # Normalizing advantage
        ADV = (ADV - tf.reduce_mean(ADV)) / (reduce_std(ADV) + 1e-8)

        ############ Training with Randomized Obs ############
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Calculate the entropy
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Calculate value loss
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate policy gradient loss
        neglogpac = train_model.pd.neglogp(A)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        # Record some information
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        ############################################


        ############ Training with Clean Obs ############
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Calculate the entropy
        entropy_clean = tf.reduce_mean(train_model_clean.pd.entropy())

        # Calculate value loss
        vpred_clean = train_model_clean.vf
        vpredclipped_clean = OLDVPRED + tf.clip_by_value(train_model_clean.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1_clean = tf.square(vpred_clean - R)
        vf_losses2_clean = tf.square(vpredclipped_clean - R)
        vf_loss_clean = .5 * tf.reduce_mean(tf.maximum(vf_losses1_clean, vf_losses2_clean))

        # Calculate policy gradient loss
        neglogpac_clean = train_model_clean.pd.neglogp(A)
        ratio_clean = tf.exp(OLDNEGLOGPAC - neglogpac_clean)
        pg_losses_clean = -ADV * ratio_clean
        pg_losses2_clean = -ADV * tf.clip_by_value(ratio_clean, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss_clean = tf.reduce_mean(tf.maximum(pg_losses_clean, pg_losses2_clean))

        # Record some information
        approxkl_clean = .5 * tf.reduce_mean(tf.square(neglogpac_clean - self.OLDNEGLOGPAC))
        clipfrac_clean = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio_clean - 1.0), self.CLIPRANGE)))
        ############################################


        ############ Calculate the total loss ############
        fm_loss = tf.losses.mean_squared_error(labels=tf.stop_gradient(train_model_clean.latent_fts), predictions=train_model.latent_fts)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + fm_loss * fm_coeff
        loss_clean = pg_loss_clean - entropy_clean * ent_coef + vf_loss_clean * vf_coef + fm_loss * fm_coeff
        self.stats_list = [loss, fm_loss, pg_loss, vf_loss, entropy, approxkl, clipfrac]
        self.stats_list_clean = [loss_clean, fm_loss, pg_loss_clean, vf_loss_clean, entropy_clean, approxkl_clean, clipfrac_clean]
        ##################################################


        ############ UPDATE THE PARAMETERS ############
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = mpi_adam.MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads_and_var_clean = self.trainer.compute_gradients(loss_clean, params)
        grads, var = zip(*grads_and_var)
        grads_clean, var_clean = zip(*grads_and_var_clean)
        # 4. Clip the gradient if required
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads_clean, _grad_norm = tf.clip_by_global_norm(grads_clean, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        grads_and_var_clean = list(zip(grads_clean, var_clean))
        ###############################################


        self.loss_names = ['total_loss', 'fm_loss', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self._train_clean_op = self.trainer.apply_gradients(grads_and_var_clean)
        self.fm_coeff = fm_coeff
        self.clean_flag = False
        self._init_randcnn = tf.variables_initializer(act_model.randcnn_param)

        self.train_model = train_model
        self.train_model_clean = train_model_clean
        self.act_model = act_model
        self.act_model_clean = act_model_clean
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def step(self, *args, **kwargs):
        if self.clean_flag:
            return self.act_model_clean.step(*args, **kwargs)
        else:
            return self.act_model.step(*args, **kwargs)

    def value(self, *args, **kwargs):
        if self.clean_flag:
            return self.act_model_clean.value(*args, **kwargs)
        else:
            return self.act_model.value(*args, **kwargs)

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        td_map = {
            self.train_model.X : obs,
            self.train_model_clean.X: obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }

        if self.clean_flag:
            return self.sess.run(self.stats_list_clean + [self._train_clean_op], td_map)[:-1]
        else:
            return self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]
