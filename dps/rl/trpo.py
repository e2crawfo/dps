import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
import numpy as np

from dps import cfg
from dps.updater import Param
from dps.rl import (
    ReinforcementLearner, episodic_mean, policy_gradient_objective,
    GeneralizedAdvantageEstimator, BasicValueEstimator)
from dps.utils import build_scheduled_value, lst_to_vec, vec_to_lst


class KLCell(RNNCell):
    def __init__(self, policy, prev_policy):
        self.policy, self.prev_policy = policy, prev_policy

    def __call__(self, obs, state, scope=None):
        with tf.name_scope(scope or 'trpo_cell'):
            policy_state, prev_policy_state = state

            utils, new_policy_state = self.policy.build_update(obs, policy_state)
            prev_utils, new_prev_policy_state = self.prev_policy.build_update(obs, prev_policy_state)
            kl = self.policy.build_kl(prev_utils, utils)

            return kl, (new_policy_state, new_prev_policy_state)

    @property
    def state_size(self):
        return (self.policy.state_size, self.prev_policy.state_size)

    @property
    def output_size(self):
        return 1

    def zero_state(self, batch_size, dtype):
        initial_state = (
            self.policy.zero_state(batch_size, dtype),
            self.prev_policy.zero_state(batch_size, dtype))
        return initial_state


def mean_kl(p, q, obs):
    """ `p` and `q` are instances of `Policy`. """
    # from tensorflow.python.ops.rnn import dynamic_rnn
    # kl_cell = KLCell(policy, prev_policy)
    # batch_size = tf.shape(obs)[1]
    # initial_state = kl_cell.zero_state(batch_size, tf.float32)

    # kl, _ = dynamic_rnn(
    #     kl_cell, obs, initial_state=initial_state,
    #     parallel_iterations=1, swap_memory=False,
    #     time_major=True)

    # return tf.reduce_mean(kl)

    # Code below requires that we know T at graph build time; need to do this in a python
    # while loop, because taking hessian_vector_products when the thing we are taking
    # the hessian of includes a tensorflow while loop is not currently supported
    batch_size = tf.shape(obs)[1]
    dtype = tf.float32
    p_state, q_state = p.zero_state(batch_size, dtype), q.zero_state(batch_size, dtype)

    kl = None
    T = int(obs.shape[0])
    for t in range(T):
        p_utils, p_state = p.build_update(obs[t, :, :], p_state)
        q_utils, q_state = q.build_update(obs[t, :, :], q_state)

        if kl is None:
            kl = p.build_kl(p_utils, q_utils)
        else:
            kl += p.build_kl(p_utils, q_utils)
    return tf.reduce_mean(kl) / T


class TRPO(ReinforcementLearner):
    delta_schedule = Param()
    entropy_schedule = Param()
    max_cg_steps = Param()

    def __init__(self, policy, advantage_estimator=None, **kwargs):
        if not advantage_estimator:
            advantage_estimator = GeneralizedAdvantageEstimator(BasicValueEstimator())
        self.advantage_estimator = advantage_estimator

        self.prev_policy = policy.deepcopy("prev_policy")

        self.policy = policy

        self.T = cfg.T

        super(TRPO, self).__init__(**kwargs)

    def build_graph(self, is_training, exploration):
        with tf.name_scope("updater"):
            self.delta = build_scheduled_value(self.delta_schedule, 'delta')

            self.obs = tf.placeholder(tf.float32, shape=(self.T, None)+self.policy.obs_shape, name="_obs")
            self.actions = tf.placeholder(tf.float32, shape=(self.T, None, self.policy.n_actions), name="_actions")
            self.advantage = tf.placeholder(tf.float32, shape=(self.T, None, 1), name="_advantage")
            self.rewards = tf.placeholder(tf.float32, shape=(self.T, None, 1), name="_rewards")
            self.reward_per_ep = episodic_mean(self.rewards, name="_reward_per_ep")

            self.grad_norm_pure = tf.placeholder(tf.float32, shape=(), name="_grad_norm_pure")
            self.grad_norm_natural = tf.placeholder(tf.float32, shape=(), name="_grad_norm_natural")
            self.step_norm = tf.placeholder(tf.float32, shape=(), name="_step_norm")

            self.policy.set_exploration(exploration)
            self.prev_policy.set_exploration(exploration)

            self.pg_objective, _, self.mean_entropy = policy_gradient_objective(
                self.policy, self.obs, self.actions, self.advantage)

            self.objective = self.pg_objective

            if self.entropy_schedule:
                entropy_param = build_scheduled_value(self.entropy_schedule, 'entropy_param')
                self.objective += entropy_param * self.mean_entropy

            g = tf.get_default_graph()
            tvars = g.get_collection('trainable_variables', scope=self.policy.scope.name)
            self.policy_gradient = tf.gradients(self.objective, tvars)

            self.mean_kl = mean_kl(self.prev_policy, self.policy, self.obs)
            self.fv_product = HessianVectorProduct(self.mean_kl, tvars)

            self.train_summary_op = tf.summary.merge([
                tf.summary.scalar("grad_norm_pure", self.grad_norm_pure),
                tf.summary.scalar("grad_norm_natural", self.grad_norm_natural),
                tf.summary.scalar("step_norm", self.step_norm),
            ])

        with tf.name_scope("eval"):
            self.eval_summary_op = tf.summary.merge([
                tf.summary.scalar("pg_objective", self.pg_objective),
                tf.summary.scalar("objective", self.objective),
                tf.summary.scalar("reward_per_ep", self.reward_per_ep),
                tf.summary.scalar("mean_entropy", self.mean_entropy),
                tf.summary.scalar("mean_kl", self.mean_kl)
            ])

    def compute_advantage(self, rollouts):
        advantage = self.advantage_estimator.estimate(rollouts)

        # Standardize advantage
        advantage = advantage - advantage.mean()
        adv_std = advantage.std()
        if adv_std > 1e-6:
            advantage /= adv_std
        return advantage

    def update(self, rollouts, collect_summaries):
        # Compute standard policy gradient
        # --------------------------------
        advantage = self.compute_advantage(rollouts)

        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.advantage: advantage,
        }

        sess = tf.get_default_session()
        policy_gradient = sess.run(self.policy_gradient, feed_dict=feed_dict)
        policy_gradient = lst_to_vec(policy_gradient)

        grad_norm_pure = np.linalg.norm(policy_gradient)
        grad_norm_natural = 0.0
        step_norm = 0.0

        if np.isclose(0, grad_norm_pure):
            print("Got zero policy gradient, not updating.")
        else:
            # Compute natural gradient direction
            # ----------------------------------
            self.prev_policy.set_params_flat(self.policy.get_params_flat())

            self.fv_product.update_feed_dict(feed_dict)
            step_dir = cg(self.fv_product, policy_gradient, max_steps=self.max_cg_steps)

            grad_norm_natural = np.linalg.norm(step_dir)

            if grad_norm_natural < 1e-6:
                print("Step dir has norm 0, not updating.")
            else:
                # Perform line search in natural gradient direction
                # -------------------------------------------------
                delta = sess.run(self.delta)
                denom = step_dir.dot(self.fv_product(step_dir))
                beta = np.sqrt(2 * delta / denom)
                full_step = beta * step_dir

                def objective(_params):
                    self.policy.set_params_flat(_params)
                    sess = tf.get_default_session()
                    return sess.run(self.objective, feed_dict=feed_dict)

                grad_dot_step_dir = policy_gradient.dot(step_dir)

                params = self.policy.get_params_flat()

                expected_imp = beta * grad_dot_step_dir
                success, new_params = maximizing_line_search(
                    objective, params, full_step, expected_imp,
                    max_backtracks=cfg.max_line_search_steps, verbose=cfg.verbose)

                self.policy.set_params_flat(new_params)

                step_norm = np.linalg.norm(new_params - params)

        if cfg.verbose:
            print("Gradient norm: ", grad_norm_pure)
            print("Natural Gradient norm: ", grad_norm_natural)
            print("Step norm: ", step_norm)

        if collect_summaries:
            feed_dict = {
                self.grad_norm_pure: grad_norm_pure,
                self.grad_norm_natural: grad_norm_natural,
                self.step_norm: step_norm,
            }
            sess = tf.get_default_session()
            train_summaries = sess.run(self.train_summary_op, feed_dict=feed_dict)
            return train_summaries
        else:
            return b''

    def evaluate(self, rollouts):
        advantage = self.compute_advantage(rollouts)

        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.advantage: advantage
        }

        sess = tf.get_default_session()

        eval_summaries, pg_objective, reward_per_ep, mean_entropy, mean_kl = (
            sess.run(
                [self.eval_summary_op, self.pg_objective, self.reward_per_ep, self.mean_entropy, self.mean_kl],
                feed_dict=feed_dict))

        record = dict(
            pg_objective=pg_objective,
            reward_per_ep=reward_per_ep,
            mean_entropy=mean_entropy,
            mean_kl=mean_kl)

        return eval_summaries, record


class HessianVectorProduct(object):
    def __init__(self, h, xs):
        self.h = h
        self.xs = xs
        self.v = tf.placeholder(tf.float32, lst_to_vec(xs).shape[0])

        # The `v` arg to _hessian_vector_product is a list of tensors with same structure as `xs`,
        # but the cg alg will give us a vector. Thus need to wrangle `v` into correct form.
        vec_as_list = vec_to_lst(self.v, xs)
        self.fv_product = lst_to_vec(_hessian_vector_product(h, xs, vec_as_list))
        self.feed_dict = None

    def update_feed_dict(self, fd):
        self.feed_dict = fd

    def __call__(self, v):
        if self.feed_dict is None:
            raise Exception("HessianVectorProduct instance has not yet been provided with a feed_dict.")
        sess = tf.get_default_session()
        fd = self.feed_dict.copy()
        fd[self.v] = v
        prod = sess.run(self.fv_product, feed_dict=fd)
        return prod


class MatrixVectorProduct(object):
    def __init__(self, A):
        self.A = A

    def __call__(self, v):
        return self.A.dot(v)


def cg(A, b, x=None, tol=1e-10, verbose=0, f=10, max_steps=None):
    """
    Parameters
    ----------
    A: A matrix, or a function capable of carrying out matrix-vector products.

    """
    n = b.size
    b = b.reshape(n)
    if x is None:
        x = np.zeros(n)
    else:
        x = x.reshape(n)
    if isinstance(A, np.ndarray):
        A = MatrixVectorProduct(A)

    max_steps = max_steps or n

    alpha = None

    r = b - A(x)
    d = r.copy()
    A_dot_d = A(d)
    r_dot_r = r.dot(r)

    for i in range(min(n, max_steps)):
        if i != 0:
            if f > 0 and i % f == 0:
                r = b - A(x)
            else:
                r -= alpha * A_dot_d

            old_r_dot_r = r_dot_r
            r_dot_r = r.dot(r)

            beta = r_dot_r / old_r_dot_r

            d = r + beta * d
            A_dot_d = A(d)

        if verbose:
            print("Step {}".format(i))
            print("Drift: {}.".format(np.linalg.norm(r - b + A(x))))
            print("R norm: {}.".format(np.linalg.norm(r)))

        d_energy_norm = d.dot(A_dot_d)
        if d_energy_norm < tol:
            break
        alpha = r_dot_r / d_energy_norm
        x += alpha * d

    if verbose:
        r = b - A(x)
        print("Final residual norm: {}.".format(np.linalg.norm(r)))

    return x


def maximizing_line_search(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1, verbose=False):
    """ Backtracking line search, where expected_improve_rate is the slope dy/dx at the initial point """
    fval = f(x)
    if verbose:
        print("Line search - fval before", fval)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = newfval - fval
        expected_improve = max(1e-6, expected_improve_rate*stepfrac)
        ratio = actual_improve/expected_improve
        if verbose:
            print("Line search - actual/expected/ratio", actual_improve, expected_improve, ratio)
        if ratio > accept_ratio and actual_improve > 0:
            if verbose:
                print("Line search succeeded - fval after", newfval)
            return True, xnew
    if verbose:
        print("Line search failed")

    return False, x
