import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import numpy as np

from dps.updater import ReinforcementLearningUpdater


class REINFORCE(ReinforcementLearningUpdater):
    def __init__(self, *args, **kwargs):
        super(REINFORCE, self).__init__(*args, **kwargs)
        self.clear_buffers()

    def _update(self, batch_size, summary_op=None):
        # reduce gradient variance by normalization
        # self.all_rewards += discounted_rewards.tolist()
        # self.all_rewards = self.all_rewards[:self.max_reward_length]
        # discounted_rewards -= np.mean(self.all_rewards)
        # discounted_rewards /= np.std(self.all_rewards)
        self.clear_buffers()
        self.env.do_rollouts(self, self.policy, 'train', batch_size)
        feed_dict = self.build_feeddict()
        sess = tf.get_default_session()

        if summary_op is not None:
            train_summary, train_loss, train_reward, _ = sess.run(
                [summary_op, self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)

            # Run some validation rollouts
            self.clear_buffers()
            self.env.do_rollouts(self, self.policy, 'val')
            feed_dict = self.build_feeddict()

            val_summary, val_loss, val_reward = sess.run(
                [summary_op, self.loss, self.reward_per_ep], feed_dict=feed_dict)

            return_value = train_summary, -train_reward, val_summary, -val_reward
        else:
            train_loss, train_reward, _ = sess.run([self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)
            return_value = -train_reward
        return return_value

    def build_feeddict(self):
        obs = np.array(self.obs_buffer)
        actions = np.array(self.action_buffer)

        T = len(self.reward_buffer)
        discounts = np.logspace(0, T-1, T, base=self.gamma).reshape(-1, 1)
        true_rewards = np.array(self.reward_buffer)
        discounted_rewards = true_rewards * discounts
        sum_discounted_rewards = np.flipud(np.cumsum(np.flipud(discounted_rewards), axis=0))
        baselines = sum_discounted_rewards.mean(1, keepdims=True)
        rewards = sum_discounted_rewards - baselines
        rewards = np.expand_dims(rewards, -1)

        feed_dict = {
            self.obs: obs,
            self.actions: actions,
            self.cumulative_rewards: rewards,
            self.true_rewards: np.expand_dims(true_rewards, -1)}
        return feed_dict

    def start_episode(self):
        pass

    def clear_buffers(self):
        self.obs_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

    def remember(self, obs, action, reward, behaviour_policy=None):
        super(REINFORCE, self).remember(obs, action, reward, behaviour_policy)

        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def _build_graph(self):
        with tf.name_scope("loss"):
            self.obs = tf.placeholder(tf.float32, shape=(None, None, self.obs_dim), name="obs")
            self.actions = tf.placeholder(tf.float32, shape=(None, None, self.n_actions), name="actions")
            self.cumulative_rewards = tf.placeholder(
                tf.float32, shape=(None, None, 1), name="cumulative_rewards")
            self.true_rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="true_rewards")
            self.reward_per_ep = tf.squeeze(
                tf.reduce_sum(tf.reduce_mean(self.true_rewards, axis=1), axis=0, name="reward_per_ep"))

            inp = (self.obs, self.actions, self.cumulative_rewards)

            reinforce_cell = ReinforceCell(self.policy)
            batch_size = tf.shape(self.obs)[1]
            initial_state = reinforce_cell.zero_state(batch_size, tf.float32)

            _, (surrogate_objective, controller_state) = dynamic_rnn(
                reinforce_cell, inp, initial_state=initial_state,
                parallel_iterations=1, swap_memory=False,
                time_major=True)

            self.q_loss = -tf.reduce_mean(surrogate_objective)

            if self.l2_norm_param > 0:
                policy_network_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.policy.scope)
                self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
                self.loss = self.q_loss + self.l2_norm_param * self.reg_loss
            else:
                self.reg_loss = None
                self.loss = self.q_loss

            tf.summary.scalar("policy_loss", self.q_loss)
            tf.summary.scalar("reward_per_ep", self.reward_per_ep)
            if self.l2_norm_param > 0:
                tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.loss)

        self._build_train()


class ReinforceCell(RNNCell):
    """ Used in defining the surrogate loss function that we differentiate to perform REINFORCE. """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, inp, state, scope=None):
        with tf.name_scope(scope or 'reinforce_cell'):
            observations, actions, cumulative_rewards = inp
            accumulator, policy_state = state

            _, action_probs, _, new_policy_state = self.policy.build(observations, policy_state)

            new_term = tf.log(tf.reduce_sum(action_probs * actions, axis=-1, keep_dims=True)) * cumulative_rewards
            new_accumulator = accumulator + new_term
            new_state = (new_accumulator, new_policy_state)

            return new_accumulator, new_state

    @property
    def state_size(self):
        return (1, self.policy.state_size)

    @property
    def output_size(self):
        return 1

    def zero_state(self, batch_size, dtype):
        initial_state = (
            tf.fill((batch_size, 1), 0.0),
            self.policy.zero_state(batch_size, dtype))
        return initial_state
