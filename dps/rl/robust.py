import tensorflow as tf
import numpy as np

from dps import cfg
from dps.utils import build_scheduled_value, lst_to_vec
from dps.rl import policy_gradient_objective
from dps.rl.trpo import TRPO, mean_kl, line_search


class RobustREINFORCE(TRPO):
    """ Combine the KL-divergence line search of TRPO with REINFORCE's use of vanilla policy gradient as search direction. """

    def _build_graph(self, is_training, exploration):
        self.delta = build_scheduled_value(self.delta_schedule, 'delta')
        self.build_placeholders()
        self.advantage_estimator.build_graph()

        self.policy.set_exploration(exploration)
        self.prev_policy.set_exploration(exploration)

        self.pg_objective, _, self.mean_entropy = policy_gradient_objective(
            self.policy, self.obs, self.actions, self.advantage)

        self.objective = self.pg_objective

        if self.entropy_schedule:
            entropy_param = build_scheduled_value(self.entropy_schedule, 'entropy_param')
            self.objective += entropy_param * self.mean_entropy

        tvars = self.policy.trainable_variables()
        self.policy_gradient = tf.gradients(self.objective, tvars)

        self.mean_kl = mean_kl(self.prev_policy, self.policy, self.obs)

        kl_penalty = tf.cond(self.mean_kl > self.delta, lambda: tf.constant(np.inf), lambda: tf.constant(0.0))
        self.line_search_objective = self.objective - kl_penalty

        self.grad_norm_pure = tf.placeholder(tf.float32, shape=(), name="_grad_norm_pure")
        self.step_norm = tf.placeholder(tf.float32, shape=(), name="_step_norm")

        self.train_summary_op = tf.summary.merge([
            tf.summary.scalar("grad_norm_pure", self.grad_norm_pure),
            tf.summary.scalar("step_norm", self.step_norm),
        ])

        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("pg_objective", self.pg_objective),
            tf.summary.scalar("objective", self.objective),
            tf.summary.scalar("reward_per_ep", self.reward_per_ep),
            tf.summary.scalar("mean_entropy", self.mean_entropy),
            tf.summary.scalar("mean_kl", self.mean_kl)
        ])

        self.recorded_values = [
            ('loss', -self.reward_per_ep),
            ('reward_per_ep', self.reward_per_ep),
            ('pg_objective', self.pg_objective),
            ('mean_entropy', self.mean_entropy),
            ('mean_kl', self.mean_kl)
        ]

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
        step_dir = policy_gradient

        grad_norm_pure = np.linalg.norm(policy_gradient)
        step_norm = 0.0

        if np.isclose(0, grad_norm_pure):
            print("Got zero policy gradient, not updating.")
        else:
            # Perform line search in policy gradient direction
            # -------------------------------------------------
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
            success, new_params = line_search(
                objective, params, full_step, expected_imp,
                max_backtracks=cfg.max_line_search_steps, verbose=cfg.verbose)

            self.policy.set_params_flat(new_params)

            step_norm = np.linalg.norm(new_params - params)

        if cfg.verbose:
            print("Gradient norm: ", grad_norm_pure)
            print("Step norm: ", step_norm)

        if collect_summaries:
            feed_dict = {
                self.grad_norm_pure: grad_norm_pure,
                self.step_norm: step_norm,
            }
            sess = tf.get_default_session()
            train_summaries = sess.run(self.train_summary_op, feed_dict=feed_dict)
            return train_summaries
        else:
            return b''
