import numpy as np
import tensorflow as tf

from dps import cfg
from dps.rl.qlearning import QLearning
from dps.rl.trust_region import mean_kl, HessianVectorProduct, cg, line_search
from dps.rl.policy import NormalWithExploration, ProductDist, Policy
from dps.utils import Param, build_scheduled_value, lst_to_vec, masked_mean, build_gradient_train_op


class TrustRegionQLearning(QLearning):
    delta_schedule = Param()
    max_cg_steps = Param()
    max_line_search_steps = Param()

    def __init__(self, q_network, **kwargs):
        action_selection = ProductDist(*[NormalWithExploration() for a in range(q_network.n_actions)])
        self.surrogate_policy = Policy(
            q_network.controller, action_selection,
            q_network.obs_shape, name="surrogate_policy")
        self.prev_surrogate_policy = self.surrogate_policy.deepcopy("prev_surrogate_policy")

        super(TrustRegionQLearning, self).__init__(q_network, **kwargs)

    def build_update_ops(self):
        tvars = self.q_network.trainable_variables()
        self.gradient = tf.gradients(self.q_loss, tvars)
        self.q_loss_gradient = tf.gradients(self.q_loss, tvars)
        self.init_q_loss_gradient = tf.gradients(self.init_q_loss, tvars)

        self.mean_kl = mean_kl(self.prev_surrogate_policy, self.surrogate_policy, self.obs, self.mask)
        self.fv_product = HessianVectorProduct(self.mean_kl, tvars)

        self.grad_norm_pure = tf.placeholder(tf.float32, shape=(), name="_grad_norm_pure")
        self.grad_norm_natural = tf.placeholder(tf.float32, shape=(), name="_grad_norm_natural")
        self.step_norm = tf.placeholder(tf.float32, shape=(), name="_step_norm")

        self.train_summary_op = tf.summary.merge([
            tf.summary.scalar("grad_norm_pure", self.grad_norm_pure),
            tf.summary.scalar("grad_norm_natural", self.grad_norm_natural),
            tf.summary.scalar("step_norm", self.step_norm),
        ])

    def _build_graph(self, is_training, exploration):
        self.delta = build_scheduled_value(self.delta_schedule, 'delta')
        self.std_dev = tf.placeholder(tf.float32, ())
        self.surrogate_policy.set_exploration(self.std_dev)
        self.prev_surrogate_policy.set_exploration(self.std_dev)
        return super(TrustRegionQLearning, self)._build_graph(is_training, exploration)

    def update_params(self, rollouts, collect_summaries, feed_dict, init):
        if init:
            tf_loss, tf_gradient = self.init_q_loss, self.init_q_loss_gradient
        else:
            tf_loss, tf_gradient = self.q_loss, self.q_loss_gradient

        sess = tf.get_default_session()
        gradient, loss = sess.run([tf_gradient, tf_loss], feed_dict=feed_dict)
        gradient = lst_to_vec(gradient)

        grad_norm_pure = np.linalg.norm(gradient)
        grad_norm_natural = 0.0
        step_norm = 0.0

        if np.isclose(0, grad_norm_pure):
            print("Got zero policy gradient, not updating.")
        else:
            # Compute natural gradient direction
            # ----------------------------------
            self.prev_surrogate_policy.set_params_flat(self.surrogate_policy.get_params_flat())

            feed_dict[self.std_dev] = np.sqrt(loss)
            self.fv_product.update_feed_dict(feed_dict)
            step_dir = -cg(self.fv_product, gradient, max_steps=self.max_cg_steps)

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
                    self.surrogate_policy.set_params_flat(_params)
                    sess = tf.get_default_session()
                    return sess.run(tf_loss, feed_dict=feed_dict)

                grad_dot_step_dir = gradient.dot(step_dir)

                params = self.surrogate_policy.get_params_flat()

                expected_imp = beta * grad_dot_step_dir
                success, new_params = line_search(
                    objective, params, full_step, expected_imp,
                    max_backtracks=self.max_line_search_steps, verbose=cfg.verbose)

                self.surrogate_policy.set_params_flat(new_params)

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


class ProximalQLearning(QLearning):
    optimizer_spec = Param()
    lr_schedule = Param()
    epsilon = Param()
    S = Param()  # number of samples from normal distribution

    def build_cpi_objective(self, targets):
        batch_size = tf.shape(self.obs)[1]
        alpha = (self.prev_q_values_selected_actions - self.q_values_selected_actions) / self.std_dev
        samples_shape = (tf.shape(self.obs)[0], batch_size, self.S)
        samples = tf.random_normal(samples_shape)

        ratio = tf.exp(-alpha * (samples + 0.5 * alpha))

        noisy_value_estimates = tf.stop_gradient(self.q_values_selected_actions + samples * self.std_dev)

        # MSE = Variance + Bias^2
        mse = self.std_dev**2 + (self.q_values_selected_actions - targets)**2
        baseline = -mse
        baseline = tf.stop_gradient(baseline)

        reward = -(noisy_value_estimates - targets)**2

        advantage = reward - baseline

        ratio_times_adv = tf.minimum(
            advantage * ratio,
            advantage * tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
        )

        ratio_times_adv = tf.reduce_mean(ratio_times_adv, axis=2, keep_dims=True)

        cpi_objective = masked_mean(ratio_times_adv, self.mask)
        return cpi_objective

    def build_update_ops(self):
        tvars = self.q_network.trainable_variables()

        self.cpi_objective = self.build_cpi_objective(self.targets)
        self.train_op, train_summaries = build_gradient_train_op(
            -self.cpi_objective, tvars, self.optimizer_spec, self.lr_schedule)
        self.train_summary_op = tf.summary.merge(train_summaries)

        masked_rewards = self.rewards * self.mask
        self.init_cpi_objective = self.build_cpi_objective(tf.cumsum(masked_rewards, axis=0, reverse=True))
        self.init_train_op, train_summaries = build_gradient_train_op(
            -self.init_cpi_objective, tvars, self.optimizer_spec, self.lr_schedule)
        self.init_train_summary_op = tf.summary.merge(train_summaries)

    def _build_graph(self, is_training, exploration):
        self.std_dev = tf.placeholder(tf.float32, (), name='std_dev')
        self.prev_q_values_selected_actions = tf.placeholder(tf.float32, (cfg.T, None, 1))

        return super(ProximalQLearning, self)._build_graph(is_training, exploration)

    def update_params(self, rollouts, collect_summaries, feed_dict, init):
        sess = tf.get_default_session()

        prev_q_values_selected_actions = sess.run(self.q_values_selected_actions, feed_dict=feed_dict)
        feed_dict[self.prev_q_values_selected_actions] = prev_q_values_selected_actions

        if init:
            std_dev = sess.run(self.init_q_loss, feed_dict=feed_dict)
        else:
            std_dev = sess.run(self.q_loss, feed_dict=feed_dict)
        feed_dict[self.std_dev] = std_dev

        return super(ProximalQLearning, self).update_params(rollouts, collect_summaries, feed_dict, init)
