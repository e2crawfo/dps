import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from collections import deque

from dps.updater import ReinforcementLearningUpdater


class QLearning(ReinforcementLearningUpdater):
    def __init__(self,
                 env,
                 q_network,
                 target_network,
                 double,
                 replay_max_size,
                 target_update_rate,
                 recurrent,
                 optimizer_class,
                 lr_schedule,
                 noise_schedule,
                 max_grad_norm,
                 gamma,
                 l2_norm_param):

        self.q_network = q_network
        self.target_network = target_network
        self.double = double
        self.replay_buffer = ReplayBuffer(max_size=replay_max_size)
        self.target_update_rate = target_update_rate
        self.recurrent = recurrent
        if not self.recurrent:
            raise NotImplementedError("Non-recurrent learning not available.")

        super(QLearning, self).__init__(
            env,
            q_network,
            optimizer_class,
            lr_schedule,
            noise_schedule,
            max_grad_norm,
            gamma,
            l2_norm_param)
        self.clear_buffers()

    def _update(self, batch_size, summary_op=None):
        self.clear_buffers()
        self.env.do_rollouts(self, self.q_network, 'train', batch_size)

        self.replay_buffer.add(self.obs_buffer, self.action_buffer, self.reward_buffer)
        self.obs_buffer, self.action_buffer, self.reward_buffer = self.replay_buffer.get_batch(batch_size)

        feed_dict = self.build_feeddict()
        feed_dict[self.is_training] = True
        sess = tf.get_default_session()

        if summary_op is not None:
            train_summary, train_loss, train_reward, _ = sess.run(
                [summary_op, self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)

            # Run some validation rollouts
            self.clear_buffers()
            self.env.do_rollouts(self, self.q_network, 'val')
            feed_dict = self.build_feeddict()
            feed_dict[self.is_training] = False

            val_summary, val_loss, val_reward = sess.run(
                [summary_op, self.loss, self.reward_per_ep], feed_dict=feed_dict)

            return_value = train_summary, -train_reward, val_summary, -val_reward
        else:
            train_loss, train_reward, _ = sess.run([self.loss, self.reward_per_ep, self.train_op], feed_dict=feed_dict)
            return_value = -train_reward

        sess.run(self.target_network_update)
        return return_value

    def build_feeddict(self):
        obs = np.array(self.obs_buffer)
        actions = np.array(self.action_buffer)
        rewards = np.expand_dims(np.array(self.reward_buffer), -1)

        feed_dict = {
            self.obs: obs,
            self.actions: actions,
            self.rewards: rewards}
        return feed_dict

    def start_episode(self):
        pass

    def clear_buffers(self):
        self.obs_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

    def remember(self, obs, action, reward, behaviour_policy=None):
        super(QLearning, self).remember(obs, action, reward, behaviour_policy)

        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def _build_graph(self):
        with tf.name_scope("loss"):
            self.obs = tf.placeholder(tf.float32, shape=(None, None, self.obs_dim), name="obs")
            self.actions = tf.placeholder(tf.float32, shape=(None, None, self.n_actions), name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="rewards")
            self.reward_per_ep = tf.squeeze(
                tf.reduce_sum(tf.reduce_mean(self.rewards, axis=1), axis=0, name="reward_per_ep"))

            first_obs = self.obs[0, :, :]
            rest_obs = self.obs[1:, :, :]

            first_actions = self.actions[0, :, :]
            rest_actions = self.actions[1:, :, :]

            rest_rewards = self.rewards[:-1, :, :]
            final_rewards = self.rewards[-1, :, :]

            inp = (rest_obs, rest_actions, rest_rewards)

            q_learning_cell = QLearningCell(self.q_network, self.target_network, self.double, self.gamma)
            batch_size = tf.shape(self.obs)[1]
            initial_state = q_learning_cell.initial_state(first_obs, first_actions, batch_size, tf.float32)

            _, (squared_error, prev_q_value, _, _) = dynamic_rnn(
                q_learning_cell, inp, initial_state=initial_state,
                parallel_iterations=1, swap_memory=False,
                time_major=True)

            final_error = tf.reduce_sum(tf.square(final_rewards - prev_q_value), axis=-1, keep_dims=True)
            squared_error = squared_error + final_error

            self.q_loss = tf.reduce_sum(squared_error)

            if self.l2_norm_param > 0:
                policy_network_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.q_network.scope.name)
                self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
                self.loss = self.q_loss + self.l2_norm_param * self.reg_loss
            else:
                self.reg_loss = None
                self.loss = self.q_loss

            tf.summary.scalar("reward_per_ep", self.reward_per_ep)

            tf.summary.scalar("q_loss", self.q_loss)
            if self.l2_norm_param > 0:
                tf.summary.scalar("reg_loss", self.reg_loss)

            self._build_train()

        with tf.name_scope("update_target_network"):
            q_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.q_network.scope.name)
            target_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.target_network.scope.name)
            target_network_update = []

            for v_source, v_target in zip(q_network_variables, target_network_variables):
                update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                target_network_update.append(update_op)
            self.target_network_update = tf.group(*target_network_update)


class QLearningCell(RNNCell):
    """ Used in defining the loss function that we differentiate to perform QLearning.

    Reward needs to be offset by one time step, always want to be working with reward caused by previous state and action.

    """
    def __init__(self, q_network, target_network, double, gamma):
        self.q_network = q_network
        self.target_network = target_network
        self.double = double
        self.gamma = gamma

    def __call__(self, inp, state, scope=None):
        with tf.name_scope(scope or 'q_learning_cell'):
            observations, actions, rewards = inp
            accumulator, prev_q_value, qn_state, tn_state = state

            _, _, target_values, new_tn_state = self.target_network.build(observations, tn_state)
            _, _, q_values, new_qn_state = self.q_network.build(observations, qn_state)

            if self.double:
                action_selection = tf.argmax(tf.stop_gradient(target_values), 1, name="action_selection")
                action_selection_mask = tf.cast(tf.one_hot(action_selection, self.q_network.n_actions, 1, 0), tf.float32)
                bootstrap = tf.reduce_sum(action_selection_mask * q_values, axis=-1, keep_dims=True)
            else:
                bootstrap = tf.reduce_max(tf.stop_gradient(target_values), axis=-1, keep_dims=True)

            target = rewards + self.gamma * bootstrap
            new_accumulator = accumulator + tf.reduce_sum(tf.square(target - prev_q_value), axis=-1, keep_dims=True)

            q_value_selected_action = tf.reduce_sum(actions * q_values, axis=1)

            new_state = (new_accumulator, q_value_selected_action, new_qn_state, new_tn_state)
            return new_accumulator, new_state

    @property
    def state_size(self):
        return (1, 1, self.q_network.state_size, self.target_network.state_size)

    @property
    def output_size(self):
        return 1

    def zero_state(self, batch_size, dtype):
        initial_state = (
            tf.fill((batch_size, 1), 0.0),
            tf.fill((batch_size, 1), 0.0),
            self.q_network.zero_state(batch_size, dtype),
            self.target_network.zero_state(batch_size, dtype))
        return initial_state

    def initial_state(self, first_obs, first_actions, batch_size, dtype):
        acc, _, qn_state, tn_state = self.zero_state(batch_size, dtype)

        _, _, _, new_tn_state = self.target_network.build(first_obs, tn_state)
        _, _, q_values, new_qn_state = self.q_network.build(first_obs, qn_state)
        q_value_selected_action = tf.reduce_sum(first_actions * q_values, axis=1)

        return acc, q_value_selected_action, new_qn_state, new_tn_state


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.n_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        replace = batch_size > len(self.buffer)
        idx = np.random.choice(len(self.buffer), batch_size, replace=replace)
        obs, actions, rewards = [], [], []
        for i in idx:
            o, a, r = self.buffer[i]
            obs.append(o)
            actions.append(a)
            rewards.append(r)

        # TODO: need to pad returned array with zeros if trajectories are different lengths
        obs = np.transpose(obs, (1, 0, 2))
        actions = np.transpose(actions, (1, 0, 2))
        rewards = np.transpose(rewards)
        assert rewards.ndim == 2

        return obs, actions, rewards

    def add(self, obs, actions, rewards):
        # Assumes shape is (n_timsesteps, batch_size, dim)
        obs = list(np.transpose(obs, (1, 0, 2)))
        actions = list(np.transpose(actions, (1, 0, 2)))
        rewards = list(np.transpose(rewards))

        for o, a, r in zip(obs, actions, rewards):
            self.buffer.append((o, a, r))
            while len(self.buffer) > self.max_size:
                self.buffer.popleft()

    def erase(self):
        self.buffer = deque()
        self.n_experiences = 0
