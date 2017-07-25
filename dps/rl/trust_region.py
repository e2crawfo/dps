import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
import numpy as np

from dps.utils import lst_to_vec, vec_to_lst, masked_mean


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


def mean_kl(p, q, obs, mask):
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

    kl = []
    T = int(obs.shape[0])
    for t in range(T):
        p_utils, p_state = p.build_update(obs[t, :, :], p_state)
        q_utils, q_state = q.build_update(obs[t, :, :], q_state)
        kl.append(p.build_kl(p_utils, q_utils))
    kl = tf.stack(kl)
    return masked_mean(kl, mask)


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


def line_search(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1, verbose=False):
    """ Backtracking line search, where expected_improve_rate is the slope dy/dx at the initial point """
    fval = f(x)
    if verbose:
        print("Line search - fval before", fval)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = newfval - fval
        if expected_improve_rate > 0:
            expected_improve = max(1e-6, expected_improve_rate*stepfrac)
        else:
            expected_improve = min(-1e-6, expected_improve_rate*stepfrac)
        ratio = actual_improve/expected_improve
        if verbose:
            print("Line search - actual/expected/ratio", actual_improve, expected_improve, ratio)
        if ratio > accept_ratio and expected_improve_rate * actual_improve > 0:
            if verbose:
                print("Line search succeeded - fval after", newfval)
            return True, xnew
    if verbose:
        print("Line search failed")

    return False, x
