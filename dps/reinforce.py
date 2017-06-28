import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import numpy as np

from dps.updater import ReinforcementLearningUpdater, Param
from dps.utils import build_scheduled_value


class REINFORCE(ReinforcementLearningUpdater):
    entropy_schedule = Param()

    def __init__(self,
                 env,
                 policy,
                 **kwargs):

        super(REINFORCE, self).__init__(env, policy, **kwargs)
        self.clear_buffers()

    def _update(self, batch_size, summary_op=None):
        self.clear_buffers()
        self.env.do_rollouts(self, self.policy, batch_size, mode='train')
        feed_dict = self.build_feeddict()
        feed_dict[self.is_training] = True
        sess = tf.get_default_session()

        if summary_op is not None:
            train_summary, train_loss, train_reward, _ = sess.run(
                [summary_op, self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)

            # Run some validation rollouts
            self.clear_buffers()
            self.env.do_rollouts(self, self.policy, mode='val')
            feed_dict = self.build_feeddict()
            feed_dict[self.is_training] = False

            val_summary, val_loss, val_reward = sess.run(
                [summary_op, self.loss, self.reward_per_ep], feed_dict=feed_dict)

            return_value = train_summary, -train_reward, val_summary, -val_reward
        else:
            train_loss, train_reward, _ = sess.run(
                [self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)
            return_value = -train_reward
        return return_value

    def build_feeddict(self):
        obs = np.array(self.obs_buffer)
        actions = np.array(self.action_buffer)

        T = len(self.reward_buffer)
        discounts = np.logspace(0, T-1, T, base=self.gamma).reshape(-1, 1, 1)
        rewards = np.array(self.reward_buffer)
        discounted_rewards = rewards * discounts
        sum_discounted_rewards = np.flipud(np.cumsum(np.flipud(discounted_rewards), axis=0))
        baselines = sum_discounted_rewards.mean(1, keepdims=True)
        advantage = sum_discounted_rewards - baselines
        advantage = (advantage - advantage.mean()) / advantage.std()

        return {
            self.obs: obs,
            self.actions: actions,
            self.advantage: advantage,
            self.rewards: rewards
        }

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
        with tf.name_scope("train"):
            self.obs = tf.placeholder(tf.float32, shape=(None, None, self.obs_dim), name="obs")
            self.actions = tf.placeholder(tf.float32, shape=(None, None, self.n_actions), name="actions")
            self.advantage = tf.placeholder(
                tf.float32, shape=(None, None, 1), name="advantage")
            self.rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="rewards")
            self.reward_per_ep = tf.reduce_mean(tf.reduce_sum(self.rewards, axis=0), name="reward_per_ep")

            inp = (self.obs, self.actions, self.advantage)

            reinforce_cell = ReinforceCell(self.policy)
            batch_size = tf.shape(self.obs)[1]
            initial_state = reinforce_cell.zero_state(batch_size, tf.float32)

            (_, log_action_probs, entropy), (surrogate_objective, controller_state) = dynamic_rnn(
                reinforce_cell, inp, initial_state=initial_state,
                parallel_iterations=1, swap_memory=False,
                time_major=True)

            tf.summary.scalar("mean_action_prob", tf.reduce_mean(tf.exp(log_action_probs)))

            self.policy_loss = -tf.reduce_mean(surrogate_objective)
            tf.summary.scalar("loss_policy", self.policy_loss)

            loss = self.policy_loss

            self.reg_loss = None
            if self.l2_norm_penalty > 0:
                policy_network_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.policy.scope)
                self.reg_loss = tf.reduce_sum(
                    [tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
                tf.summary.scalar("loss_reg", self.reg_loss)

                loss += self.l2_norm_penalty * self.reg_loss

            policy_entropy = tf.reduce_mean(entropy)
            tf.summary.scalar("policy_entropy", policy_entropy)

            if self.entropy_schedule:
                self.entropy_loss = -policy_entropy
                entropy_bonus = build_scheduled_value(self.entropy_schedule, 'entropy_bonus')
                loss += entropy_bonus * self.entropy_loss

            self.loss = loss

            tf.summary.scalar("reward_per_ep", self.reward_per_ep)

            self._build_train()


class ReinforceCell(RNNCell):
    """ Used in defining the surrogate loss function that we differentiate to perform REINFORCE. """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, inp, state, scope=None):
        with tf.name_scope(scope or 'reinforce_cell'):
            observations, actions, advantage = inp
            accumulator, policy_state = state

            log_prob, entropy, new_policy_state = self.policy.build_log_prob_and_entropy(
                observations, policy_state, actions)

            new_acc = accumulator + (log_prob * advantage)

            return (new_acc, log_prob, entropy), (new_acc, new_policy_state)

    @property
    def state_size(self):
        return (1, self.policy.state_size)

    @property
    def output_size(self):
        return (1, 1, 1)

    def zero_state(self, batch_size, dtype):
        initial_state = (
            tf.fill((batch_size, 1), 0.0),
            self.policy.zero_state(batch_size, dtype))
        return initial_state
