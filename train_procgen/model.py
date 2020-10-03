import numpy as np
import tensorflow as tf
import functools
from scipy.spatial.distance import cdist

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

from .utils import reduce_std

def get_mixreg_model(mix_mode='nomix', mix_alpha=0.2, use_l2reg=False, l2reg_coeff=1e-4,
                     fix_representation=False):
    def model_fn(*args, **kwargs):
        kwargs['mix_mode'] = mix_mode
        kwargs['mix_alpha'] = mix_alpha
        kwargs['use_l2reg'] = use_l2reg
        kwargs['l2reg_coeff'] = l2reg_coeff
        kwargs['fix_representation'] = fix_representation
        return MixregModel(**kwargs)
    return model_fn

class MixregModel:
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
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None,
                microbatch_size=None, mix_mode='nomix', mix_alpha=0.2,
                fix_representation=False, use_l2reg=False, l2reg_coeff=1e-4):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess, mix_mode=mix_mode)
            else:
                train_model = policy(microbatch_size, nsteps, sess, mix_mode=mix_mode)

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

        # Interpolating the supervision
        if mix_mode == 'mixreg':
            # get coeff and indices
            coeff = train_model.coeff
            indices = train_model.indices
            other_indices = train_model.other_indices
            # mixup
            OLDNEGLOGPAC = coeff * tf.gather(OLDNEGLOGPAC, indices, axis=0) + (1 - coeff) * tf.gather(OLDNEGLOGPAC, other_indices, axis=0)
            OLDVPRED = coeff * tf.gather(OLDVPRED, indices, axis=0) + (1 - coeff) * tf.gather(OLDVPRED, other_indices, axis=0)
            R = coeff * tf.gather(R, indices, axis=0) + (1 - coeff) * tf.gather(R, other_indices, axis=0)
            ADV = coeff * tf.gather(ADV, indices, axis=0) + (1 - coeff) * tf.gather(ADV, other_indices, axis=0)
            A = tf.gather(A, indices, axis=0)
        elif mix_mode == 'mixobs':
            # get indices
            indices = train_model.indices
            # gather
            OLDNEGLOGPAC = tf.gather(OLDNEGLOGPAC, train_model.indices, axis=0)
            OLDVPRED = tf.gather(OLDVPRED, train_model.indices, axis=0)
            R = tf.gather(R, train_model.indices, axis=0)
            ADV = tf.gather(ADV, train_model.indices, axis=0)
            A = tf.gather(A, train_model.indices, axis=0)
        elif mix_mode == 'nomix':
            pass
        else:
            raise ValueError(f"Unknown mixing mode: {mix_mode} !")

        # Store the nodes to be recorded
        self.loss_names = []
        self.stats_list = []


        ############ CALCULATE LOSS ############
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Normalizing advantage
        ADV = (ADV - tf.reduce_mean(ADV)) / (reduce_std(ADV) + 1e-8)

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

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # Record some information
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        self.loss_names.extend([
            'total_loss',
            'policy_loss',
            'value_loss',
            'policy_entropy',
            'approxkl',
            'clipfrac',
        ])
        self.stats_list.extend([
            loss,
            pg_loss,
            vf_loss,
            entropy,
            approxkl,
            clipfrac,
        ])
        ############################################


        ############ UPDATE THE PARAMETERS ############
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        if use_l2reg:
            weight_params = [v for v in params if '/b' not in v.name]
            l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])
            self.loss_names.append('l2_loss')
            self.stats_list.append(l2_loss)
            loss = loss + l2_loss * l2reg_coeff
        if fix_representation:
            params = params[-4:]
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)
        # 4. Clip the gradient if required
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        ###############################################

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self._init_op = tf.variables_initializer(params)
        self._sync_param = lambda: sync_from_root(sess, params, comm=comm)

        self.mix_mode = mix_mode
        self.mix_alpha = mix_alpha
        self.fix_representation = fix_representation
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        # Exclude the random convolution layer from syncing
        global_variables = [v for v in global_variables if 'randcnn' not in v.name]
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }

        batchsize = len(obs)
        if self.mix_mode in ['mixreg', 'mixobs']:
            # Generate mix coefficients and indices
            coeff = np.random.beta(self.mix_alpha, self.mix_alpha, size=(batchsize,))
            seq_indices = np.arange(batchsize)
            rand_indices = np.random.permutation(batchsize)
            indices = np.where(coeff > 0.5, seq_indices, rand_indices)
            other_indices = np.where(coeff > 0.5, rand_indices, seq_indices)
            coeff = np.where(coeff > 0.5, coeff, 1 - coeff)
            # Add into feed dict
            td_map[self.train_model.coeff] = coeff
            td_map[self.train_model.indices] = indices
            td_map[self.train_model.other_indices] = other_indices
        elif self.mix_mode == 'nomix':
            pass
        else:
            raise ValueError(f"Unknown mixing mode: {self.mix_mode} !")

        return self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]
