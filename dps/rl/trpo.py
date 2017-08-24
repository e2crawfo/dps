import tensorflow as tf
import numpy as np

from dps import cfg
from dps.updater import Param
from dps.rl import (
    PolicyOptimization, policy_gradient_objective,
    GeneralizedAdvantageEstimator, BasicValueEstimator)
from dps.rl.trust_region import mean_kl, HessianVectorProduct, cg, line_search
from dps.utils import build_scheduled_value, lst_to_vec


class TRPO(PolicyOptimization):
    delta_schedule = Param()
    entropy_schedule = Param()
    max_cg_steps = Param()
    max_line_search_steps = Param()

    def __init__(self, policy, advantage_estimator=None, **kwargs):
        if not advantage_estimator:
            advantage_estimator = GeneralizedAdvantageEstimator(BasicValueEstimator())
        self.advantage_estimator = advantage_estimator

        self.policy = policy
        self.prev_policy = policy.deepcopy("prev_policy")

        super(TRPO, self).__init__(**kwargs)

    def _build_graph(self, is_training, exploration):
        self.delta = build_scheduled_value(self.delta_schedule, 'delta')
        self.build_placeholders()

        self.policy.set_exploration(exploration)
        self.prev_policy.set_exploration(exploration)

        self.pg_objective, _, self.mean_entropy = policy_gradient_objective(
            self.policy, self.obs, self.actions, self.advantage, self.mask)

        self.objective = self.pg_objective

        if self.entropy_schedule:
            entropy_param = build_scheduled_value(self.entropy_schedule, 'entropy_param')
            self.objective += entropy_param * self.mean_entropy

        tvars = self.policy.trainable_variables()
        self.policy_gradient = tf.gradients(self.objective, tvars)

        self.mean_kl = mean_kl(self.prev_policy, self.policy, self.obs, self.mask)
        self.fv_product = HessianVectorProduct(self.mean_kl, tvars)

        self.grad_norm_pure = tf.placeholder(tf.float32, shape=(), name="_grad_norm_pure")
        self.grad_norm_natural = tf.placeholder(tf.float32, shape=(), name="_grad_norm_natural")
        self.step_norm = tf.placeholder(tf.float32, shape=(), name="_step_norm")

        self.train_summary_op = tf.summary.merge([
            tf.summary.scalar("grad_norm_pure", self.grad_norm_pure),
            tf.summary.scalar("grad_norm_natural", self.grad_norm_natural),
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
        feed_dict = self.build_feed_dict(rollouts)

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
                success, new_params = line_search(
                    objective, params, full_step, expected_imp,
                    max_backtracks=self.max_line_search_steps, verbose=cfg.verbose)

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
