import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import numpy as np

from dps.updater import ReinforcementLearningUpdater
from dps.utils import build_decaying_value


class REINFORCE(ReinforcementLearningUpdater):
    def __init__(self,
                 env,
                 policy,
                 optimizer_class,
                 lr_schedule,
                 noise_schedule,
                 max_grad_norm,
                 gamma,
                 l2_norm_penalty,
                 entropy_schedule):
        self.entropy_schedule = entropy_schedule
        super(REINFORCE, self).__init__(
            env, policy, optimizer_class, lr_schedule,
            noise_schedule, max_grad_norm, gamma, l2_norm_penalty)
        self.clear_buffers()

    def _update(self, batch_size, summary_op=None):
        self.clear_buffers()
        self.env.do_rollouts(self, self.policy, 'train', batch_size)
        feed_dict = self.build_feeddict()
        feed_dict[self.is_training] = True
        sess = tf.get_default_session()

        if summary_op is not None:
            train_summary, train_loss, train_reward, _ = sess.run(
                [summary_op, self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)

            # Run some validation rollouts
            self.clear_buffers()
            self.env.do_rollouts(self, self.policy, 'val')
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
        true_rewards = np.array(self.reward_buffer)
        discounted_rewards = true_rewards * discounts
        sum_discounted_rewards = np.flipud(np.cumsum(np.flipud(discounted_rewards), axis=0))
        baselines = sum_discounted_rewards.mean(1, keepdims=True)
        rewards = sum_discounted_rewards - baselines

        return {
            self.obs: obs,
            self.actions: actions,
            self.cumulative_rewards: rewards,
            self.true_rewards: true_rewards
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
            self.cumulative_rewards = tf.placeholder(
                tf.float32, shape=(None, None, 1), name="cumulative_rewards")
            self.true_rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="true_rewards")
            self.reward_per_ep = tf.squeeze(
                tf.reduce_sum(
                    tf.reduce_mean(self.true_rewards, axis=1),
                    axis=0,
                    name="reward_per_ep"))

            inp = (self.obs, self.actions, self.cumulative_rewards)

            reinforce_cell = ReinforceCell(self.policy)
            batch_size = tf.shape(self.obs)[1]
            initial_state = reinforce_cell.zero_state(batch_size, tf.float32)

            (_, action_probs), (surrogate_objective, controller_state) = dynamic_rnn(
                reinforce_cell, inp, initial_state=initial_state,
                parallel_iterations=1, swap_memory=False,
                time_major=True)

            self.policy_loss = -tf.reduce_mean(surrogate_objective)
            tf.summary.scalar("loss_policy", self.policy_loss)

            loss = self.policy_loss

            self.reg_loss = None
            if self.l2_norm_penalty > 0:
                policy_network_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.policy.scope)
                self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
                tf.summary.scalar("loss_reg", self.reg_loss)

                loss += self.l2_norm_penalty * self.reg_loss

            policy_mean_entropy = tf.reduce_mean(tf.reduce_sum(-action_probs * tf.log(action_probs + 1e-6), axis=-1))
            tf.summary.scalar("policy_mean_entropy", policy_mean_entropy)

            if self.entropy_schedule:
                self.entropy_loss = -policy_mean_entropy
                entropy_bonus = build_decaying_value(self.entropy_schedule, 'entropy_schedule')
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
            observations, actions, cumulative_rewards = inp
            accumulator, policy_state = state

            _, action_probs, _, new_policy_state = self.policy.build(observations, policy_state)

            new_term = tf.log(tf.reduce_sum(action_probs * actions, axis=-1, keep_dims=True)) * cumulative_rewards
            new_accumulator = accumulator + new_term

            return (new_accumulator, action_probs), (new_accumulator, new_policy_state)

    @property
    def state_size(self):
        return (1, self.policy.state_size)

    @property
    def output_size(self):
        return (1, self.policy.n_actions)

    def zero_state(self, batch_size, dtype):
        initial_state = (
            tf.fill((batch_size, 1), 0.0),
            self.policy.zero_state(batch_size, dtype))
        return initial_state
