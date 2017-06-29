import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import numpy as np

from dps.updater import ReinforcementLearningUpdater, Param
from dps.utils import build_scheduled_value


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


def policy_gradient_surrogate(policy, obs, actions, advantage):
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


class REINFORCE(ReinforcementLearningUpdater):
    entropy_schedule = Param()

    def __init__(self, env, policy, **kwargs):
        super(REINFORCE, self).__init__(env, policy, **kwargs)
        self.clear_buffers()

    def build_graph(self):
        with tf.name_scope("updater"):
            self.obs = tf.placeholder(tf.float32, shape=(None, None, self.obs_dim), name="_obs")
            self.actions = tf.placeholder(tf.float32, shape=(None, None, self.n_actions), name="_actions")
            self.advantage = tf.placeholder(tf.float32, shape=(None, None, 1), name="_advantage")
            self.rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="_rewards")
            self.reward_per_ep = tf.reduce_mean(tf.reduce_sum(self.rewards, axis=0), name="_reward_per_ep")

            self.surrogate_objective, log_action_probs, mean_entropy = policy_gradient_surrogate(
                self.policy, self.obs, self.actions, self.advantage)

            loss = -self.surrogate_objective

            if self.entropy_schedule:
                self.entropy_loss = -mean_entropy
                entropy_param = build_scheduled_value(self.entropy_schedule, 'entropy_param')
                loss += entropy_param * self.entropy_loss

            self.loss = loss

            self._build_optimizer()

        with tf.name_scope("performance"):
            tf.summary.scalar("surrogate_loss", -self.surrogate_objective)
            tf.summary.scalar("mean_entropy", mean_entropy)
            tf.summary.scalar("reward_per_ep", self.reward_per_ep)

    def _build_feeddict(self):
        obs = np.array(self.obs_buffer)
        actions = np.array(self.action_buffer)

        T = len(self.reward_buffer)
        discounts = np.logspace(0, T-1, T, base=self.gamma).reshape(-1, 1, 1)
        rewards = np.array(self.reward_buffer)
        discounted_rewards = rewards * discounts
        sum_discounted_rewards = np.flipud(np.cumsum(np.flipud(discounted_rewards), axis=0))
        baselines = sum_discounted_rewards.mean(1, keepdims=True)
        advantage = sum_discounted_rewards - baselines
        advantage = advantage - advantage.mean()
        adv_std = advantage.std()
        if adv_std > 1e-6:
            advantage /= adv_std

        return {
            self.obs: obs,
            self.actions: actions,
            self.advantage: advantage,
            self.rewards: rewards
        }

    def _update(self, batch_size, summary_op=None):
        self.clear_buffers()
        self.env.do_rollouts(self, self.policy, batch_size, mode='train')
        feed_dict = self._build_feeddict()
        feed_dict[self.is_training] = True
        sess = tf.get_default_session()

        if summary_op is not None:
            train_summary, train_loss, train_reward, _ = sess.run(
                [summary_op, self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)

            # Run some validation rollouts
            self.clear_buffers()
            self.env.do_rollouts(self, self.policy, mode='val')
            feed_dict = self._build_feeddict()
            feed_dict[self.is_training] = False

            val_summary, val_loss, val_reward = sess.run(
                [summary_op, self.loss, self.reward_per_ep], feed_dict=feed_dict)

            return_value = train_summary, -train_reward, val_summary, -val_reward
        else:
            train_loss, train_reward, _ = sess.run(
                [self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)
            return_value = -train_reward
        return return_value
