import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from dps.updater import Param
from dps.utils import build_scheduled_value
from dps.rl import (
    ReinforcementLearner, episodic_mean,
    GeneralizedAdvantageEstimator, BasicValueEstimator)
from dps.utils import build_gradient_train_op


class PolicyGradientCell(RNNCell):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, inp, state, scope=None):
        with tf.name_scope(scope or 'pg_cell'):
            obs, actions, advantage = inp
            policy_state = state

            utils, new_policy_state = self.policy.build_update(obs, policy_state)
            log_prob = self.policy.build_log_prob(utils, actions)
            entropy = self.policy.build_entropy(utils)
            log_prob_times_adv = log_prob * advantage

            return (log_prob_times_adv, log_prob, entropy), new_policy_state

    @property
    def state_size(self):
        return self.policy.state_size

    @property
    def output_size(self):
        return (1, 1, 1)

    def zero_state(self, batch_size, dtype):
        return self.policy.zero_state(batch_size, dtype)


def policy_gradient_objective(policy, obs, actions, advantage):
    inp = obs, actions, advantage

    pg_cell = PolicyGradientCell(policy)
    batch_size = tf.shape(obs)[1]
    initial_state = pg_cell.zero_state(batch_size, tf.float32)

    (log_prob_times_adv, log_prob, entropy), _ = dynamic_rnn(
        pg_cell, inp, initial_state=initial_state, parallel_iterations=1,
        swap_memory=False, time_major=True)

    surrogate_objective = tf.reduce_mean(tf.reduce_sum(log_prob_times_adv, axis=0))
    mean_entropy = tf.reduce_mean(entropy)
    return surrogate_objective, log_prob, mean_entropy


class REINFORCE(ReinforcementLearner):
    entropy_schedule = Param()
    optimizer_spec = Param()
    lr_schedule = Param()

    def __init__(self, policy, advantage_estimator=None, **kwargs):
        if not advantage_estimator:
            advantage_estimator = GeneralizedAdvantageEstimator(BasicValueEstimator())
        self.advantage_estimator = advantage_estimator
        self.policy = policy
        super(REINFORCE, self).__init__(**kwargs)

    def build_graph(self, is_training, exploration):
        with tf.name_scope("update"):
            self.obs = tf.placeholder(tf.float32, shape=(None, None) + self.policy.obs_shape, name="_obs")
            self.actions = tf.placeholder(tf.float32, shape=(None, None, self.policy.n_actions), name="_actions")
            self.advantage = tf.placeholder(tf.float32, shape=(None, None, 1), name="_advantage")
            self.rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="_rewards")
            self.reward_per_ep = episodic_mean(self.rewards, name="_reward_per_ep")

            self.policy.set_exploration(exploration)

            self.pg_objective, _, self.mean_entropy = policy_gradient_objective(
                self.policy, self.obs, self.actions, self.advantage)

            self.loss = -self.pg_objective

            if self.entropy_schedule:
                entropy_param = build_scheduled_value(self.entropy_schedule, 'entropy_param')
                self.loss += entropy_param * -self.mean_entropy

            scope = None
            self.train_op, train_summaries = build_gradient_train_op(
                self.loss, scope, self.optimizer_spec, self.lr_schedule)

            self.train_summary_op = tf.summary.merge(train_summaries)

        with tf.name_scope("eval"):
            self.eval_summary_op = tf.summary.merge([
                tf.summary.scalar("pg_objective", self.pg_objective),
                tf.summary.scalar("objective", -self.loss),
                tf.summary.scalar("reward_per_ep", self.reward_per_ep),
                tf.summary.scalar("mean_entropy", self.mean_entropy),
            ])

    def compute_advantage(self, rollouts):
        advantage = self.advantage_estimator.estimate(rollouts)

        # Standardize advantage
        advantage = advantage - advantage.mean()
        adv_std = advantage.std()
        if adv_std > 1e-6:
            advantage /= adv_std
        return advantage

    def update(self, rollouts, collect_summaries):
        advantage = self.compute_advantage(rollouts)

        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.advantage: advantage,
        }

        sess = tf.get_default_session()
        if collect_summaries:
            train_summaries, _ = sess.run([self.train_summary_op, self.train_op], feed_dict=feed_dict)
            return train_summaries
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            return b''

    def evaluate(self, rollouts):
        advantage = self.compute_advantage(rollouts)

        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.advantage: advantage
        }

        sess = tf.get_default_session()

        eval_summaries, pg_objective, reward_per_ep, mean_entropy = (
            sess.run(
                [self.eval_summary_op, self.pg_objective, self.reward_per_ep, self.mean_entropy],
                feed_dict=feed_dict))

        record = dict(
            pg_objective=pg_objective,
            reward_per_ep=reward_per_ep,
            mean_entropy=mean_entropy)

        return eval_summaries, record