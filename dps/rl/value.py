import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from dps import cfg
from dps.utils import (
    build_gradient_train_op, Param, lst_to_vec, masked_mean,
    build_scheduled_value, trainable_variables, Config)
from dps.rl import ReinforcementLearner, RLUpdater
from dps.rl.policy import Policy, NormalWithExploration
from dps.rl.trust_region import mean_kl, HessianVectorProduct, cg, line_search


def compute_td_error(value_estimates, gamma, reward):
    if isinstance(value_estimates, np.ndarray):
        shifted_value_estimates = np.concatenate(
            [value_estimates[1:, :, :], np.zeros_like(value_estimates[0:1, :, :])],
            axis=0)
        return reward + gamma * shifted_value_estimates - value_estimates
    elif isinstance(value_estimates, tf.Tensor):
        shifted_value_estimates = tf.concat(
            [value_estimates[1:, :, :], tf.zeros_like(value_estimates[0:1, :, :])],
            axis=0)
        return reward + gamma * shifted_value_estimates - value_estimates
    else:
        raise Exception("Not implemented.")


def symmetricize(a):
    """ Turn a 1D array `a` into a matrix where the entries on the i-th diagonals are equal to `a[i]`. """
    ID = np.arange(a.size)
    return a[np.abs(ID - ID[:, None])]


class AbstractPolicyEvaluation(ReinforcementLearner):
    opt_steps_per_batch = Param(1)

    def __init__(self, estimator, gamma=1.0, **kwargs):
        self.estimator = estimator
        self.gamma = gamma

        super(AbstractPolicyEvaluation, self).__init__(**kwargs)

    def build_placeholders(self):
        self.obs = tf.placeholder(
            tf.float32, shape=(cfg.T, None) + self.estimator.obs_shape, name="_obs")
        self.rewards = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_rewards")
        self.targets = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_targets")
        self.mask = tf.placeholder(tf.float32, shape=(cfg.T, None, 1), name="_mask")

    def _build_graph(self, is_training, exploration):
        self.build_placeholders()

        batch_size = tf.shape(self.obs)[1]
        initial_state = self.estimator.controller.zero_state(batch_size, tf.float32)
        self.value_estimates, _ = dynamic_rnn(
            self.estimator.controller, self.obs, initial_state=initial_state,
            parallel_iterations=1, swap_memory=False, time_major=True)

        self.squared_error = (self.value_estimates - self.targets)**2
        self.mean_squared_error = masked_mean(self.squared_error, self.mask)
        self.mean_estimated_value = masked_mean(self.value_estimates, self.mask)

        self.td_error = compute_td_error(self.value_estimates, self.gamma, self.rewards)
        self.approx_bellman_error = masked_mean(self.td_error, self.mask)
        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("mean_squared_error", self.mean_squared_error),
            tf.summary.scalar("approx_bellman_error", self.approx_bellman_error),
            tf.summary.scalar("mean_estimated_value", self.mean_estimated_value)
        ])

    def build_feed_dict(self, rollouts):
        masked_rewards = rollouts.r * rollouts.mask
        cumsum_rewards = np.flipud(np.cumsum(np.flipud(masked_rewards), axis=0))
        feed_dict = {
            self.obs: rollouts.o,
            self.targets: cumsum_rewards,
            self.rewards: rollouts.r,
            self.mask: rollouts.mask
        }
        return feed_dict

    def update(self, rollouts, collect_summaries):
        feed_dict = self.build_feed_dict(rollouts)

        sess = tf.get_default_session()
        for k in range(self.opt_steps_per_batch):
            if collect_summaries:
                train_summaries, _ = sess.run(
                    [self.train_summary_op, self.train_op], feed_dict=feed_dict)
            else:
                sess.run(self.train_op, feed_dict=feed_dict)
                train_summaries = b''

        return train_summaries

    def evaluate(self, rollouts):
        feed_dict = self.build_feed_dict(rollouts)

        sess = tf.get_default_session()
        eval_summaries, mean_squared_error, approx_bellman_error, mean_estimated_value = (
            sess.run(
                [self.eval_summary_op,
                 self.mean_squared_error,
                 self.approx_bellman_error,
                 self.mean_estimated_value],
                feed_dict=feed_dict))

        record = {
            'mean_estimated_value': mean_estimated_value,
            'mean_squared_error': mean_squared_error,
            'loss': mean_squared_error,
            'approx_bellman_error': approx_bellman_error}

        return eval_summaries, record


class PolicyEvaluation(AbstractPolicyEvaluation):
    """ Evaluate a policy by performing batch regression. """
    optimizer_spec = Param()
    lr_schedule = Param()

    def _build_graph(self, is_training, exploration):
        super(PolicyEvaluation, self)._build_graph(is_training, exploration)

        tvars = self.estimator.trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            self.mean_squared_error, tvars, self.optimizer_spec, self.lr_schedule)

        self.train_summary_op = tf.summary.merge(train_summaries)


class ProximalPolicyEvaluation(AbstractPolicyEvaluation):
    """ Evaluate a policy by performing batch regression, but stick close to previous value function.

    Allows us to take big steps, but not so big that catastrophic steps are likely.

    To implement this, we basically treat the value function as parameterizing a gaussian policy with
    standard deviation given by self.std_dev. The goal of said policy is to accurately predict the
    return given that we are currently in state s, and following policy pi. The reward received by
    such a return-prediction policy is thus the negative squared error between the estimated return
    and the observed return. Then we basically perform straightforward PPO on this surrogate policy/MDP
    combination.

    """
    optimizer_spec = Param()
    lr_schedule = Param()
    epsilon = Param()
    S = Param()  # number of samples from normal distribution

    def __init__(self, estimator, gamma=1.0, **kwargs):
        super(ProximalPolicyEvaluation, self).__init__(estimator, gamma, **kwargs)

    def build_feed_dict(self, rollouts):
        feed_dict = super(ProximalPolicyEvaluation, self).build_feed_dict(rollouts)

        sess = tf.get_default_session()

        prev_value_estimates, mean_squared_error = sess.run(
            [self.value_estimates, self.mean_squared_error], feed_dict=feed_dict)

        feed_dict[self.prev_value_estimates] = prev_value_estimates
        feed_dict[self.std_dev] = np.sqrt(mean_squared_error)

        return feed_dict

    def _build_graph(self, is_training, exploration):
        super(ProximalPolicyEvaluation, self)._build_graph(is_training, exploration)

        self.prev_value_estimates = tf.placeholder(tf.float32, (cfg.T, None, 1), name='prev_value_estimates')
        self.std_dev = tf.placeholder(tf.float32, ())

        alpha = (self.prev_value_estimates - self.value_estimates) / self.std_dev

        obs_shape = tf.shape(self.obs)
        samples_shape = (obs_shape[0], obs_shape[1], self.S)
        samples = tf.random_normal(samples_shape)

        ratio = tf.exp(-alpha * (samples + 0.5 * alpha))

        noisy_value_estimates = tf.stop_gradient(self.value_estimates + samples * self.std_dev)

        # MSE = Variance + Bias^2
        mse = self.std_dev**2 + self.squared_error
        baseline = -mse
        baseline = tf.stop_gradient(baseline)

        reward = -(noisy_value_estimates - self.targets)**2

        advantage = reward - baseline

        ratio_times_adv = tf.minimum(
            advantage * ratio,
            advantage * tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
        )

        ratio_times_adv = tf.reduce_mean(ratio_times_adv, axis=2, keep_dims=True)

        self.cpi_objective = masked_mean(ratio_times_adv, self.mask)

        tvars = self.estimator.trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            -self.cpi_objective, tvars, self.optimizer_spec, self.lr_schedule)

        self.train_summary_op = tf.summary.merge(train_summaries)


class TrustRegionPolicyEvaluation(AbstractPolicyEvaluation):
    delta_schedule = Param()
    max_cg_steps = Param()
    max_line_search_steps = Param()

    def __init__(self, estimator, gamma=1.0, **kwargs):
        super(TrustRegionPolicyEvaluation, self).__init__(estimator, gamma, **kwargs)

        self.policy = Policy(
            estimator.controller, NormalWithExploration(),
            estimator.obs_shape, name="policy_eval")
        self.prev_policy = self.policy.deepcopy("prev_policy")

    def _build_graph(self, is_training, exploration):
        super(TrustRegionPolicyEvaluation, self)._build_graph(is_training, exploration)

        self.delta = build_scheduled_value(self.delta_schedule, 'delta')

        self.std_dev = tf.placeholder(tf.float32, ())
        self.policy.set_exploration(self.std_dev)
        self.prev_policy.set_exploration(self.std_dev)

        tvars = self.policy.trainable_variables()
        self.gradient = tf.gradients(self.mean_squared_error, tvars)

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

    def update(self, rollouts, collect_summaries):
        feed_dict = self.build_feed_dict(rollouts)

        sess = tf.get_default_session()
        gradient, mean_squared_error = sess.run([self.gradient, self.mean_squared_error], feed_dict=feed_dict)
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

            feed_dict[self.std_dev] = np.sqrt(mean_squared_error)
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
                    self.policy.set_params_flat(_params)
                    sess = tf.get_default_session()
                    return sess.run(self.mean_squared_error, feed_dict=feed_dict)

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


class BasicValueEstimator(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def build_graph(self):
        pass

    def estimate(self, rollouts):
        obs, rewards = rollouts.o, rollouts.r * rollouts.mask
        T = len(obs)
        discounts = np.logspace(0, T-1, T, base=self.gamma).reshape(-1, 1, 1)
        discounted_rewards = rewards * discounts
        sum_discounted_rewards = np.flipud(
            np.cumsum(np.flipud(discounted_rewards), axis=0))
        value_t = sum_discounted_rewards.mean(axis=1, keepdims=True)
        value_t = np.tile(value_t, (1, obs.shape[1], 1))
        return value_t

    def trainable_variables(self):
        return []


class NeuralValueEstimator(object):
    def __init__(self, controller, obs_shape, **kwargs):
        self.controller = controller
        self.obs_shape = obs_shape
        self.estimate_is_built = False
        super(NeuralValueEstimator, self).__init__(**kwargs)

    def build_estimate(self):
        if not self.estimate_is_built:
            self.obs = tf.placeholder(
                tf.float32, shape=(None, None) + self.obs_shape, name="obs")
            batch_size = tf.shape(self.obs)[1]
            initial_state = self.controller.zero_state(batch_size, tf.float32)
            self.value_estimates, _ = dynamic_rnn(
                self.controller, self.obs, initial_state=initial_state,
                parallel_iterations=1, swap_memory=False, time_major=True)
            self.estimate_is_built = True

    def estimate(self, rollouts):
        self.build_estimate()

        sess = tf.get_default_session()
        value_estimates = sess.run(
            self.value_estimates, feed_dict={self.obs: rollouts.o})
        return value_estimates

    def trainable_variables(self):
        return trainable_variables(self.controller.scope.name)


class GeneralizedAdvantageEstimator(object):
    def __init__(self, value_estimator, gamma=1.0, lmbda=1.0):
        self.value_estimator = value_estimator
        self.gamma = gamma
        self.lmbda = lmbda

    def estimate(self, rollouts):
        value_estimates = self.value_estimator.estimate(rollouts)
        T = len(value_estimates)
        discounts = np.logspace(0, T-1, T, base=self.gamma*self.lmbda)
        discount_matrix = np.triu(symmetricize(discounts))
        td_error = compute_td_error(value_estimates, self.gamma, rollouts.r) * rollouts.mask
        advantage_estimates = np.tensordot(discount_matrix, td_error, axes=1)
        return advantage_estimates


def actor_critic(
        env, policy_controller, action_selection, critic_controller,
        actor_config=None, critic_config=None):

    policy = Policy(policy_controller, action_selection, env.obs_shape)

    if critic_config is None:
        critic_config = Config()

    with critic_config:
        value_estimator = NeuralValueEstimator(critic_controller, env.obs_shape)
        critic = cfg.alg(value_estimator, name="critic")

    if actor_config is None:
        actor_config = Config()

    with actor_config:
        advantage_estimator = GeneralizedAdvantageEstimator(
            value_estimator, lmbda=cfg.lmbda, gamma=cfg.gamma)
        actor = cfg.alg(policy, advantage_estimator, name="actor")

    # Make sure we update critic before actor, so that the critic
    # is really giving us a more accurate view of the performance
    # of the current actor
    learners = [critic, actor]
    loss_func = lambda records: records[1]['loss']

    return RLUpdater(env, policy, learners=learners, loss_func=loss_func)
