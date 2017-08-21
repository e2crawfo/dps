import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from dps import cfg
from dps.rl.qlearning import QLearning, PrioritizedReplayBuffer, clipped_error
from dps.rl.policy import EpsilonGreedy
from dps.utils import Param, build_scheduled_value, masked_mean


class RetraceCell(RNNCell):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, inp, state, scope=None):
        rho, r, v, q = inp
        prev_retrace = state

        one_step_estimate = r + self.gamma * v
        adjustment = self.gamma * rho * (prev_retrace - q)

        new_retrace = one_step_estimate + adjustment
        return (new_retrace, one_step_estimate, adjustment), new_retrace

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return (1, 1, 1)

    def zero_state(self, batch_size, dtype):
        return tf.fill((batch_size, 1), 0.0)


class Retrace(QLearning):
    """ Off-policy learning of action-value function from returns. """

    lmbda = Param(1.0)
    greedy_factor = Param(2.0)  # Ratio between greediness of policy we learn about and that of the exploration policy.

    def build_placeholders(self):
        self.mu_log_probs = tf.placeholder(tf.float32, shape=(cfg.T, None, 1), name="_mu_log_probs")
        super(Retrace, self).build_placeholders()

    def _build_graph(self, is_training, exploration):
        self.beta = build_scheduled_value(self.beta_schedule, 'beta')
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_max_size, self.n_partitions, self.alpha, self.beta)

        self.build_placeholders()

        self.q_network.set_exploration(exploration)
        self.target_network.set_exploration(exploration)

        batch_size = tf.shape(self.obs)[1]

        # Q values
        (_, q_values), _ = dynamic_rnn(
            self.q_network, self.obs, initial_state=self.q_network.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=True)
        self.q_values_selected_actions = tf.reduce_sum(q_values * self.actions, axis=-1, keep_dims=True)
        mean_q_value = masked_mean(self.q_values_selected_actions, self.mask)

        # Bootstrap values
        (_, bootstrap_values), _ = dynamic_rnn(
            self.target_network, self.obs, initial_state=self.target_network.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=True)
        bootstrap_values_selected_actions = tf.reduce_sum(bootstrap_values * self.actions, axis=-1, keep_dims=True)
        mean_q_value_target_network = masked_mean(bootstrap_values_selected_actions, self.mask)

        # Compute retrace target - policy we are learning about is the one that is epsilon-greedy wrt to the target network
        epsilon_greedy = EpsilonGreedy(self.q_network.n_actions)

        if self.double:
            pi_log_probs = epsilon_greedy.log_prob_all(
                q_values, exploration/self.greedy_factor)
        else:
            pi_log_probs = epsilon_greedy.log_prob_all(
                bootstrap_values, exploration/self.greedy_factor)

        self.pi_probs = pi_probs = tf.exp(pi_log_probs)
        prob_sums = tf.reduce_sum(pi_probs, axis=-1, keep_dims=True)
        check = masked_mean(tf.abs(prob_sums - tf.ones_like(prob_sums)), self.mask)
        values = tf.reduce_sum(pi_probs * bootstrap_values, axis=-1, keep_dims=True)

        retrace_cell = RetraceCell(self.gamma)

        pi_log_probs_selected_actions = tf.reduce_sum(pi_log_probs * self.actions, axis=-1, keep_dims=True)

        importance_ratio = tf.exp(pi_log_probs_selected_actions - self.mu_log_probs)
        rho = self.lmbda * tf.minimum(importance_ratio, 1)
        rho = tf.concat([rho[1:, :, :], tf.ones_like(rho[:1, :, :])], axis=0)

        retrace_q = tf.concat(
            [bootstrap_values_selected_actions[1:, :, :],
             tf.zeros_like(bootstrap_values_selected_actions[:1, :, :])],
            axis=0)

        retrace_v = tf.concat(
            [values[1:, :, :], tf.zeros_like(values[:1, :, :])],
            axis=0)

        retrace_input = (
            tf.reverse(rho, axis=[0]),
            tf.reverse(self.rewards, axis=[0]),
            tf.reverse(retrace_v, axis=[0]),
            tf.reverse(retrace_q, axis=[0])
        )

        (retrace, one_step_estimate, adjustment), _ = dynamic_rnn(
            retrace_cell, retrace_input,
            initial_state=retrace_cell.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=True)

        one_step_estimate = tf.reverse(one_step_estimate, axis=[0])
        one_step_loss = masked_mean(
            (self.rewards + self.gamma * retrace_v - one_step_estimate)**2,
            self.mask
        )

        adjustment = tf.reverse(adjustment, axis=[0])

        retrace = tf.reverse(retrace, axis=[0])
        self.targets = tf.stop_gradient(retrace)

        self.td_error = (self.targets - self.q_values_selected_actions) * self.mask
        self.weighted_td_error = self.td_error * self.weights
        self.q_loss = masked_mean(clipped_error(self.weighted_td_error), self.mask)
        self.unweighted_q_loss = masked_mean(clipped_error(self.td_error), self.mask)

        masked_rewards = self.rewards * self.mask
        self.monte_carlo_error_unweighted = (
            tf.cumsum(masked_rewards, axis=0, reverse=True) - self.q_values_selected_actions) * self.mask
        self.monte_carlo_error = self.monte_carlo_error_unweighted * self.weights
        self.monte_carlo_loss_unweighted = masked_mean(clipped_error(self.monte_carlo_error_unweighted), self.mask)
        self.monte_carlo_loss = masked_mean(clipped_error(self.monte_carlo_error), self.mask)

        self.build_update_ops()

        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("1_step_td_loss", self.q_loss),
            tf.summary.scalar("1_step_td_loss_unweighted", self.unweighted_q_loss),
            tf.summary.scalar("monte_carlo_td_loss", self.monte_carlo_loss),
            tf.summary.scalar("monte_carlo_td_loss_unweighted", self.monte_carlo_loss_unweighted),
            tf.summary.scalar("reward_per_ep", self.reward_per_ep),
            tf.summary.scalar("mean_q_value", mean_q_value),
            tf.summary.scalar("mean_q_value_target_network", mean_q_value_target_network),
            tf.summary.scalar("abs_adjustment", masked_mean(tf.abs(adjustment), self.mask)),
            tf.summary.scalar("importance_ratio", masked_mean(importance_ratio, self.mask)),
            tf.summary.scalar("rho", masked_mean(rho, self.mask)),
            tf.summary.scalar("prob_check", check)
        ])

        self.recorded_values = [
            ('loss', -self.reward_per_ep),
            ('reward_per_ep', self.reward_per_ep),
            ('q_loss', self.q_loss),
            ('one_step_loss', one_step_loss),
        ]

        self.build_target_network_update()

    def update_params(self, rollouts, collect_summaries, feed_dict, init):
        feed_dict[self.mu_log_probs] = rollouts.log_probs
        return super(Retrace, self).update_params(rollouts, collect_summaries, feed_dict, init)

    def evaluate(self, rollouts):
        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.mask: rollouts.mask,
            self.weights: np.ones_like(rollouts.mask),
            self.mu_log_probs: rollouts.log_probs
        }

        sess = tf.get_default_session()

        eval_summaries, *values = (
            sess.run(
                [self.eval_summary_op] + [v for _, v in self.recorded_values],
                feed_dict=feed_dict))

        record = {k: v for v, (k, _) in zip(values, self.recorded_values)}
        return eval_summaries, record
