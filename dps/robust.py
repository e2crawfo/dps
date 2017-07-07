import tensorflow as tf
import numpy as np

from dps import cfg
from dps.reinforce import policy_gradient_surrogate
from dps.trpo import TRPO, mean_kl, maximizing_line_search
from dps.utils import build_scheduled_value, lst_to_vec


class RobustREINFORCE(TRPO):
    """ Combine the KL-divergence line search of TRPO with REINFORCE's use of vanilla policy gradient as search direction. """

    def __init__(self, env, policy, **kwargs):
        self.prev_policy = policy.deepcopy("prev_policy")
        self.prev_policy.capture_scope()

        self.T = cfg.T

        super(TRPO, self).__init__(env, policy, **kwargs)

        self.clear_buffers()

    def build_graph(self):
        with tf.name_scope("updater"):
            self.delta = build_scheduled_value(self.delta_schedule, 'delta')

            self.obs = tf.placeholder(tf.float32, shape=(self.T, None, self.obs_dim), name="_obs")
            self.actions = tf.placeholder(tf.float32, shape=(self.T, None, self.n_actions), name="_actions")
            self.advantage = tf.placeholder(tf.float32, shape=(self.T, None, 1), name="_advantage")
            self.rewards = tf.placeholder(tf.float32, shape=(self.T, None, 1), name="_rewards")
            self.reward_per_ep = tf.reduce_mean(tf.reduce_sum(self.rewards, axis=0), name="_reward_per_ep")

            self.grad_norm_pure = tf.placeholder(tf.float32, shape=(), name="_grad_norm_pure")
            self.step_norm = tf.placeholder(tf.float32, shape=(), name="_step_norm")

            self.surrogate_objective, _, self.mean_entropy = policy_gradient_surrogate(
                self.policy, self.obs, self.actions, self.advantage)

            g = tf.get_default_graph()
            tvars = g.get_collection('trainable_variables', scope=self.policy.scope.name)
            self.policy_gradient = tf.gradients(self.surrogate_objective, tvars)

            self.mean_kl = mean_kl(self.prev_policy, self.policy, self.obs)

            kl_penalty = tf.cond(self.mean_kl > self.delta, lambda: tf.constant(np.inf), lambda: tf.constant(0.0))
            self.line_search_objective = self.surrogate_objective - kl_penalty

            tf.summary.scalar("grad_norm_pure", self.grad_norm_pure)
            tf.summary.scalar("step_norm", self.step_norm)

        with tf.name_scope("performance"):
            tf.summary.scalar("surrogate_loss", -self.surrogate_objective)
            tf.summary.scalar("mean_entropy", self.mean_entropy)
            tf.summary.scalar("mean_kl", self.mean_kl)
            tf.summary.scalar("reward_per_ep", self.reward_per_ep)

    def _update(self, batch_size, summary_op=None):
        self.clear_buffers()
        self.env.do_rollouts(self, self.policy, batch_size, mode='train')

        feed_dict = self._build_feeddict()
        feed_dict[self.is_training] = True

        sess = tf.get_default_session()
        policy_gradient = sess.run(self.policy_gradient, feed_dict=feed_dict)
        policy_gradient = lst_to_vec(policy_gradient)

        grad_norm_pure = 0.0
        step_norm = 0.0

        if np.allclose(policy_gradient, 0):
            if cfg.verbose:
                print("Got zero gradient, not updating.")
        else:
            grad_norm_pure = np.linalg.norm(policy_gradient)
            step_dir = policy_gradient

            if cfg.verbose:
                print("Gradient norm: ", grad_norm_pure)

            delta = sess.run(self.delta)

            denom = step_dir.dot(step_dir)
            beta = np.sqrt(2 * delta / denom)
            full_step = beta * step_dir

            def objective(_params):
                self.policy.set_params_flat(_params)
                sess = tf.get_default_session()
                return sess.run(self.line_search_objective, feed_dict=feed_dict)

            params = self.policy.get_params_flat()
            self.prev_policy.set_params_flat(params)

            expected_imp = beta * denom
            success, new_params = maximizing_line_search(
                objective, params, full_step, expected_imp,
                max_backtracks=cfg.max_line_search_steps, verbose=cfg.verbose)

            step_norm = np.linalg.norm(new_params - params)

            if cfg.verbose:
                print("Step norm: ", step_norm)

            self.policy.set_params_flat(new_params)

        feed_dict[self.grad_norm_pure] = grad_norm_pure
        feed_dict[self.step_norm] = step_norm

        if summary_op is not None:
            train_summary, train_reward = sess.run([summary_op, self.reward_per_ep], feed_dict=feed_dict)

            # Run some validation rollouts
            self.clear_buffers()
            self.env.do_rollouts(self, self.policy, mode='val')
            feed_dict = self._build_feeddict()
            feed_dict[self.is_training] = False

            feed_dict[self.grad_norm_pure] = 0.0
            feed_dict[self.step_norm] = 0.0

            val_summary, val_reward = sess.run([summary_op, self.reward_per_ep], feed_dict=feed_dict)

            return_value = train_summary, -train_reward, val_summary, -val_reward
        else:
            train_reward = sess.run(self.reward_per_ep, feed_dict=feed_dict)
            return_value = -train_reward

        return return_value
