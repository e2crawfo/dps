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


def q_retrace_estimate(
        q_values, pi_log_probs, mu_log_probs,
        actions, rewards, mask, gamma, lmbda):
    """
    Compute the the return-based Retrace estimate of the state-action value
    function of a target policy pi based on trajectories generated from
    a behaviour policy mu, and as well as an existing estimate of the
    state-action value function given by `q_values`.

    """
    pi_probs = tf.exp(pi_log_probs)

    prob_sums = tf.reduce_sum(pi_probs, axis=-1, keep_dims=True)
    prob_check = masked_mean(tf.abs(prob_sums - tf.ones_like(prob_sums)), mask)

    values = tf.reduce_sum(pi_probs * q_values, axis=-1, keep_dims=True)

    retrace_cell = RetraceCell(gamma)

    pi_log_probs_selected_actions = tf.reduce_sum(pi_log_probs * actions, axis=-1, keep_dims=True)

    importance_ratio = tf.exp(pi_log_probs_selected_actions - mu_log_probs)
    rho = lmbda * tf.minimum(importance_ratio, 1)
    rho = tf.concat([rho[1:, :, :], tf.ones_like(rho[:1, :, :])], axis=0)

    q_values_selected_actions = tf.reduce_sum(q_values * actions, axis=-1, keep_dims=True)

    retrace_q = tf.concat(
        [q_values_selected_actions[1:, :, :], tf.zeros_like(q_values_selected_actions[:1, :, :])],
        axis=0)

    retrace_v = tf.concat(
        [values[1:, :, :], tf.zeros_like(values[:1, :, :])],
        axis=0)

    retrace_input = (
        tf.reverse(rho, axis=[0]),
        tf.reverse(rewards, axis=[0]),
        tf.reverse(retrace_v, axis=[0]),
        tf.reverse(retrace_q, axis=[0])
    )

    batch_size = tf.shape(actions)[1]
    (retrace, one_step_estimate, adjustment), _ = dynamic_rnn(
        retrace_cell, retrace_input,
        initial_state=retrace_cell.zero_state(batch_size, tf.float32),
        parallel_iterations=1, swap_memory=False, time_major=True)

    one_step_estimate = tf.reverse(one_step_estimate, axis=[0])
    one_step_loss = masked_mean(
        (rewards + gamma * retrace_v - one_step_estimate)**2, mask)

    adjustment = tf.reverse(adjustment, axis=[0])

    retrace = tf.reverse(retrace, axis=[0])

    return retrace, one_step_loss, adjustment, importance_ratio, rho, prob_check


class Retrace(QLearning):
    """ Off-policy learning of action-value functions from returns. """

    lmbda = Param(1.0)
    greedy_factor = Param(2.0)  # Ratio between greediness of policy we learn about and that of the exploration policy.

    def __init__(self, q_network, target_policy=None, **kwargs):
        """
        When `target_policy` is None, we are learning the value function for the policy
        that is (epsilon/greedy_factor)-greedy with respect to current `target_network` function,
        so we are learning control. When it is not None, we are learning about the supplied target_policy
        policy, which corresponds to doing policy evaluation.

        """
        self.target_policy = target_policy
        super(Retrace, self).__init__(q_network, **kwargs)

    def build_placeholders(self):
        self.mu_log_probs = tf.placeholder(tf.float32, shape=(cfg.T, None, 1), name="_mu_log_probs")
        super(Retrace, self).build_placeholders()

    def build_feed_dict(self, rollouts, weights=None):
        feed_dict = super(Retrace, self).build_feed_dict(rollouts, weights=weights)
        feed_dict[self.mu_log_probs] = rollouts.log_probs
        return feed_dict

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

        if self.target_policy is not None:
            (_, target_policy_utils), _ = dynamic_rnn(
                self.target_network, self.obs, initial_state=self.target_network.zero_state(batch_size, tf.float32),
                parallel_iterations=1, swap_memory=False, time_major=True)

            pi_log_probs = self.target_policy.build_log_prob_all(target_policy_utils)
        else:
            # Compute retrace target - policy we are learning about is the one that is epsilon-greedy wrt to the target network
            epsilon_greedy = EpsilonGreedy(self.q_network.n_actions)

            if self.double:
                pi_log_probs = epsilon_greedy.log_prob_all(
                    q_values, exploration/self.greedy_factor)
            else:
                pi_log_probs = epsilon_greedy.log_prob_all(
                    bootstrap_values, exploration/self.greedy_factor)

        retrace, one_step_loss, adjustment, importance_ratio, rho, prob_check = q_retrace_estimate(
            bootstrap_values, pi_log_probs, self.mu_log_probs,
            self.actions, self.rewards, self.mask, self.gamma, self.lmbda)

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
            tf.summary.scalar("prob_check", prob_check)
        ])

        self.recorded_values = [
            ('loss', -self.reward_per_ep),
            ('reward_per_ep', self.reward_per_ep),
            ('q_loss', self.q_loss),
            ('one_step_loss', one_step_loss),
        ]

        self.build_target_network_update()
