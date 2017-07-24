import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from dps import cfg
from dps.utils import (
    build_gradient_train_op, Param, lst_to_vec,
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


class PolicyEvaluation(ReinforcementLearner):
    """ Evaluate a policy by performing regression. """
    optimizer_spec = Param()
    lr_schedule = Param()

    def __init__(self, estimator, gamma=1.0, **kwargs):
        self.estimator = estimator
        self.gamma = gamma

        super(PolicyEvaluation, self).__init__(**kwargs)

    def _build_graph(self, is_training, exploration):
        self.estimator.build_graph()
        self.rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="_rewards")
        self.targets = tf.placeholder(tf.float32, shape=(None, None, 1), name="_targets")
        error = self.estimator.value_estimates - self.targets
        self.loss = tf.reduce_mean(error**2)
        self.mean_estimated_value = tf.reduce_mean(self.estimator.value_estimates)

        tvars = self.estimator.trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule)

        td_error = compute_td_error(self.estimator.value_estimates, self.gamma, self.rewards)

        self.train_summary_op = tf.summary.merge(train_summaries)

        self.approx_bellman_error = tf.reduce_mean(td_error)
        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("squared_error", self.loss),
            tf.summary.scalar("approx_bellman_error", self.approx_bellman_error),
            tf.summary.scalar("mean_estimated_value", self.mean_estimated_value)
        ])

    def update(self, rollouts, collect_summaries):
        cumsum_rewards = np.flipud(np.cumsum(np.flipud(rollouts.r), axis=0))
        feed_dict = {
            self.estimator.obs: rollouts.o,
            self.targets: cumsum_rewards,
        }

        sess = tf.get_default_session()

        if collect_summaries:
            train_summaries, _ = sess.run([self.train_summary_op, self.train_op], feed_dict=feed_dict)
            return train_summaries
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            return b''

    def evaluate(self, rollouts):
        cumsum_rewards = np.flipud(np.cumsum(np.flipud(rollouts.r), axis=0))
        feed_dict = {
            self.estimator.obs: rollouts.o,
            self.targets: cumsum_rewards,
            self.rewards: rollouts.r
        }

        sess = tf.get_default_session()

        eval_summaries, loss, approx_bellman_error, mean_estimated_value = (
            sess.run(
                [self.eval_summary_op,
                 self.loss,
                 self.approx_bellman_error,
                 self.mean_estimated_value],
                feed_dict=feed_dict))

        record = {
            'mean_estimated_value': mean_estimated_value,
            'squared_error': loss,
            'loss': loss,
            'approx_bellman_error': approx_bellman_error}

        return eval_summaries, record


class TrustRegionPolicyEvaluation(ReinforcementLearner):
    delta_schedule = Param()
    max_cg_steps = Param()
    max_line_search_steps = Param()

    def __init__(self, estimator, gamma=1.0, **kwargs):
        self.estimator = estimator
        self.gamma = gamma

        self.policy = Policy(estimator.controller, NormalWithExploration(), estimator.obs_shape, name="policy_eval")
        self.prev_policy = self.policy.deepcopy("prev_policy")

        self.T = cfg.T

        super(TrustRegionPolicyEvaluation, self).__init__(**kwargs)

    def _build_graph(self, is_training, exploration):
        self.delta = build_scheduled_value(self.delta_schedule, 'delta')

        self.obs = tf.placeholder(tf.float32, shape=(self.T, None) + self.estimator.obs_shape, name="_obs")
        self.rewards = tf.placeholder(tf.float32, shape=(self.T, None, 1), name="_rewards")
        self.targets = tf.placeholder(tf.float32, shape=(self.T, None, 1), name="_targets")

        self.std_dev = tf.placeholder(tf.float32, ())

        self.policy.set_exploration(self.std_dev)
        self.prev_policy.set_exploration(self.std_dev)

        batch_size = tf.shape(self.obs)[1]
        initial_state = self.policy.zero_state(batch_size, tf.float32)
        (_, self.value_estimates), _ = dynamic_rnn(
            self.policy, self.obs, initial_state=initial_state,
            parallel_iterations=1, swap_memory=False, time_major=True)

        error = self.value_estimates - self.targets
        self.loss = tf.reduce_mean(error**2)
        self.mean_estimated_value = tf.reduce_mean(self.value_estimates)

        tvars = self.policy.trainable_variables()
        self.gradient = tf.gradients(self.loss, tvars)

        self.mean_kl = mean_kl(self.prev_policy, self.policy, self.obs)
        self.fv_product = HessianVectorProduct(self.mean_kl, tvars)

        self.grad_norm_pure = tf.placeholder(tf.float32, shape=(), name="_grad_norm_pure")
        self.grad_norm_natural = tf.placeholder(tf.float32, shape=(), name="_grad_norm_natural")
        self.step_norm = tf.placeholder(tf.float32, shape=(), name="_step_norm")

        self.train_summary_op = tf.summary.merge([
            tf.summary.scalar("grad_norm_pure", self.grad_norm_pure),
            tf.summary.scalar("grad_norm_natural", self.grad_norm_natural),
            tf.summary.scalar("step_norm", self.step_norm),
        ])

        td_error = compute_td_error(self.value_estimates, self.gamma, self.rewards)
        self.approx_bellman_error = tf.reduce_mean(td_error)
        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("squared_error", self.loss),
            tf.summary.scalar("approx_bellman_error", self.approx_bellman_error),
            tf.summary.scalar("mean_estimated_value", self.mean_estimated_value)
        ])

    def update(self, rollouts, collect_summaries):
        cumsum_rewards = np.flipud(np.cumsum(np.flipud(rollouts.r), axis=0))
        feed_dict = {
            self.obs: rollouts.o,
            self.targets: cumsum_rewards,
        }

        sess = tf.get_default_session()
        gradient, loss = sess.run([self.gradient, self.loss], feed_dict=feed_dict)
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
                    self.policy.set_params_flat(_params)
                    sess = tf.get_default_session()
                    return sess.run(self.loss, feed_dict=feed_dict)

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

    def evaluate(self, rollouts):
        cumsum_rewards = np.flipud(np.cumsum(np.flipud(rollouts.r), axis=0))
        feed_dict = {
            self.obs: rollouts.o,
            self.targets: cumsum_rewards,
            self.rewards: rollouts.r
        }

        sess = tf.get_default_session()

        eval_summaries, loss, approx_bellman_error, mean_estimated_value = (
            sess.run(
                [self.eval_summary_op,
                 self.loss,
                 self.approx_bellman_error,
                 self.mean_estimated_value],
                feed_dict=feed_dict))

        record = {
            'mean_estimated_value': mean_estimated_value,
            'squared_error': loss,
            'loss': loss,
            'approx_bellman_error': approx_bellman_error}

        return eval_summaries, record


class BasicValueEstimator(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def build_graph(self):
        pass

    def estimate(self, rollouts):
        obs, rewards = rollouts.o, rollouts.r
        T = len(obs)
        discounts = np.logspace(0, T-1, T, base=self.gamma).reshape(-1, 1, 1)
        discounted_rewards = rewards * discounts
        sum_discounted_rewards = np.flipud(np.cumsum(np.flipud(discounted_rewards), axis=0))
        value_t = sum_discounted_rewards.mean(axis=1, keepdims=True)
        value_t = np.tile(value_t, (1, obs.shape[1], 1))
        return value_t

    def trainable_variables(self):
        return []


class NeuralValueEstimator(object):
    def __init__(self, controller, obs_shape, **kwargs):
        self.controller = controller
        self.obs_shape = obs_shape
        super(NeuralValueEstimator, self).__init__(**kwargs)

    def build_graph(self):
        self.obs = tf.placeholder(tf.float32, shape=(None, None) + self.obs_shape, name="_obs")
        batch_size = tf.shape(self.obs)[1]
        initial_state = self.controller.zero_state(batch_size, tf.float32)
        self.value_estimates, _ = dynamic_rnn(
            self.controller, self.obs, initial_state=initial_state,
            parallel_iterations=1, swap_memory=False, time_major=True)

    def estimate(self, rollouts):
        sess = tf.get_default_session()
        value_estimates = sess.run(self.value_estimates, feed_dict={self.obs: rollouts.o})
        return value_estimates

    def trainable_variables(self):
        return trainable_variables(self.controller.scope.name)


class GeneralizedAdvantageEstimator(object):
    def __init__(self, value_estimator, gamma=1.0, lmbda=1.0):
        self.value_estimator = value_estimator
        self.gamma = gamma
        self.lmbda = lmbda

    def build_graph(self):
        self.value_estimator.build_graph()

    def estimate(self, rollouts):
        value_estimates = self.value_estimator.estimate(rollouts)
        T = len(value_estimates)
        discounts = np.logspace(0, T-1, T, base=self.gamma*self.lmbda)
        discount_matrix = np.triu(symmetricize(discounts))
        td_error = compute_td_error(value_estimates, self.gamma, rollouts.r)
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
        critic = cfg.alg(value_estimator)

    if actor_config is None:
        actor_config = Config()

    with actor_config:
        advantage_estimator = GeneralizedAdvantageEstimator(
            value_estimator, lmbda=cfg.lmbda, gamma=cfg.gamma)
        actor = cfg.alg(policy, advantage_estimator)

    learners = [actor, critic]

    return RLUpdater(env, policy, learners)
