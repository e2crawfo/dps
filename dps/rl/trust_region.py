import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product
import numpy as np

from dps import cfg
from dps.rl.optimizer import Optimizer
from dps.utils.tf import lst_to_vec, vec_to_lst, masked_mean, build_scheduled_value


class TrustRegionOptimizer(Optimizer):
    def __init__(self, agents, policy, delta_schedule, max_cg_steps, max_line_search_steps):
        super(TrustRegionOptimizer, self).__init__(agents)
        self.policy = policy
        self.delta_schedule = delta_schedule
        self.max_cg_steps = max_cg_steps
        self.max_line_search_steps = max_line_search_steps

    def build_update(self, context):
        self.delta = build_scheduled_value(self.delta_schedule, "delta")

        tvars = self.trainable_variables(for_opt=True)
        self.gradient = tf.gradients(context.objective, tvars)

        mask = context.get_signal('mask')
        kl = context.get_signal('kl', self.policy)

        mean_kl = masked_mean(kl, mask)
        self.fv_product = HessianVectorProduct(mean_kl, tvars)

        self.grad_norm_pure = tf.placeholder(tf.float32, shape=(), name="_grad_norm_pure")
        self.grad_norm_natural = tf.placeholder(tf.float32, shape=(), name="_grad_norm_natural")
        self.step_norm = tf.placeholder(tf.float32, shape=(), name="_step_norm")

        for s in [
                tf.summary.scalar("grad_norm_pure", self.grad_norm_pure),
                tf.summary.scalar("grad_norm_natural", self.grad_norm_natural),
                tf.summary.scalar("step_norm", self.step_norm)]:
            context.add_train_summary(s)

    def update(self, feed_dict):
        # Compute gradient of objective
        # -----------------------------
        sess = tf.get_default_session()
        gradient = sess.run(self.gradient, feed_dict=feed_dict)
        gradient = lst_to_vec(gradient)

        grad_norm_pure = np.linalg.norm(gradient)
        grad_norm_natural = 0.0
        step_norm = 0.0

        if np.isclose(0, grad_norm_pure):
            print("Got zero policy gradient, not updating.")
        else:
            # Compute natural gradient direction
            # ----------------------------------
            self.prev_policy.set_params_flat(self.policy.get_params_flat())

            self.fv_product.update_feed_dict(feed_dict)
            step_dir = cg(self.fv_product, gradient, max_steps=self.max_cg_steps)

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

                grad_dot_step_dir = gradient.dot(step_dir)

                params = self.policy.get_params_flat()

                expected_imp = beta * grad_dot_step_dir
                success, new_params = line_search(
                    objective, params, full_step, expected_imp,
                    max_backtracks=self.max_line_search_steps, verbose=cfg.verbose)

                self.policy.set_params_flat(new_params)

                step_norm = np.linalg.norm(new_params - params)

        if cfg.verbose:
            print("Gradient norm: ", grad_norm_pure)
            print("Natural Gradient norm: ", grad_norm_natural)
            print("Step norm: ", step_norm)

        feed_dict.update({
            self.grad_norm_pure: grad_norm_pure,
            self.grad_norm_natural: grad_norm_natural,
            self.step_norm: step_norm,
        })


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
