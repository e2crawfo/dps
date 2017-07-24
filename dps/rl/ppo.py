import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from dps.updater import Param
from dps.utils import build_scheduled_value
from dps.rl import (
    PolicyOptimization, GeneralizedAdvantageEstimator, BasicValueEstimator)
from dps.utils import build_gradient_train_op


class LogProbCell(RNNCell):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, inp, policy_state, scope=None):
        with tf.name_scope(scope or 'log_prob_cell'):
            obs, actions = inp
            utils, new_policy_state = self.policy.build_update(obs, policy_state)
            log_prob = self.policy.build_log_prob(utils, actions)
            entropy = self.policy.build_entropy(utils)
            return (log_prob, entropy), new_policy_state

    @property
    def state_size(self):
        return self.policy.state_size

    @property
    def output_size(self):
        return (1, 1)

    def zero_state(self, batch_size, dtype):
        return self.policy.zero_state(batch_size, dtype)


def cpi_surrogate_objective(prev_policy, policy, obs, actions, advantage, epsilon=None):
    """ If `epsilon` is not None, compute PPO objective instead of CPI. """
    batch_size = tf.shape(obs)[1]

    inp = obs, actions

    prev_lp_cell = LogProbCell(prev_policy)
    initial_state = prev_lp_cell.zero_state(batch_size, tf.float32)

    (prev_log_probs, _), _ = dynamic_rnn(
        prev_lp_cell, inp, initial_state=initial_state, parallel_iterations=1,
        swap_memory=False, time_major=True)

    lp_cell = LogProbCell(policy)
    initial_state = lp_cell.zero_state(batch_size, tf.float32)

    (log_probs, entropy), _ = dynamic_rnn(
        lp_cell, inp, initial_state=initial_state, parallel_iterations=1,
        swap_memory=False, time_major=True)

    if epsilon is None:
        log_prob_times_adv = tf.exp(log_probs - prev_log_probs) * advantage
    else:
        ratio = tf.exp(log_probs - prev_log_probs)
        log_prob_times_adv = tf.minimum(
            advantage * ratio,
            advantage * tf.clip_by_value(ratio, 1-epsilon, 1+epsilon))

    surrogate_objective = tf.reduce_sum(tf.reduce_mean(log_prob_times_adv, axis=0))
    mean_entropy = tf.reduce_mean(entropy)

    return surrogate_objective, mean_entropy


class PPO(PolicyOptimization):
    entropy_schedule = Param()
    optimizer_spec = Param()
    lr_schedule = Param()
    epsilon = Param()
    K = Param()  # number of optimization steps per batch

    def __init__(self, policy, advantage_estimator=None, **kwargs):
        if not advantage_estimator:
            advantage_estimator = GeneralizedAdvantageEstimator(BasicValueEstimator())
        self.advantage_estimator = advantage_estimator

        self.policy = policy
        self.prev_policy = policy.deepcopy("prev_policy")

        super(PPO, self).__init__(**kwargs)

    def _build_graph(self, is_training, exploration):
        self.build_placeholders()
        self.advantage_estimator.build_graph()

        self.prev_policy.set_exploration(exploration)
        self.policy.set_exploration(exploration)

        self.cpi_objective, self.mean_entropy = cpi_surrogate_objective(
            self.prev_policy, self.policy,
            self.obs, self.actions, self.advantage, self.epsilon)

        self.loss = -self.cpi_objective

        if self.entropy_schedule:
            entropy_param = build_scheduled_value(self.entropy_schedule, 'entropy_param')
            self.loss += entropy_param * -self.mean_entropy

        tvars = self.policy.trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule)

        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("cpi_objective", self.cpi_objective),
            tf.summary.scalar("objective", -self.loss),
            tf.summary.scalar("reward_per_ep", self.reward_per_ep),
            tf.summary.scalar("mean_entropy", self.mean_entropy),
        ])

        self.recorded_values = [
            ('loss', -self.reward_per_ep),
            ('reward_per_ep', self.reward_per_ep),
            ('cpi_objective', self.cpi_objective),
            ('mean_entropy', self.mean_entropy)
        ]

    def update(self, rollouts, collect_summaries):
        self.prev_policy.set_params_flat(self.policy.get_params_flat())

        advantage = self.compute_advantage(rollouts)

        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.advantage: advantage,
        }

        sess = tf.get_default_session()
        for k in range(self.K):
            sess.run(self.train_op, feed_dict=feed_dict)
        return b''
