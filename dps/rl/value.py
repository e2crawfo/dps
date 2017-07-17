import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn

from dps.utils import build_gradient_train_op, Param
from dps.rl import ReinforcementLearner


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

    def build_graph(self, is_training, exploration):
        self.estimator.build_graph()
        self.rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="_rewards")
        self.targets = tf.placeholder(tf.float32, shape=(None, None, 1), name="_targets")
        error = self.estimator.value_estimates - self.targets
        self.loss = tf.reduce_mean(error**2)
        self.mean_estimated_value = tf.reduce_mean(self.estimator.value_estimates)

        scope = None
        self.train_op, train_summaries = build_gradient_train_op(
            self.loss, scope, self.optimizer_spec, self.lr_schedule)

        td_error = compute_td_error(self.estimator.value_estimates, self.gamma, self.rewards)

        self.train_summary_op = tf.summary.merge(train_summaries)

        with tf.name_scope("eval"):
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
            'approx_bellman_error': approx_bellman_error}

        return loss, eval_summaries, record


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


class GeneralizedAdvantageEstimator(object):
    def __init__(self, value_estimator, gamma=1.0, lmbda=1.0):
        self.value_estimator = value_estimator
        self.gamma = gamma
        self.lmbda = lmbda

    def build_graph(self):
        pass

    def estimate(self, rollouts):
        value_estimates = self.value_estimator.estimate(rollouts)
        T = len(value_estimates)
        discounts = np.logspace(0, T-1, T, base=self.gamma*self.lmbda)
        discount_matrix = np.triu(symmetricize(discounts))
        td_error = compute_td_error(value_estimates, self.gamma, rollouts.r)
        advantage_estimates = np.tensordot(discount_matrix, td_error, axes=1)
        return advantage_estimates
