import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from collections import deque

from dps.updater import ReinforcementLearningUpdater, Param
from dps.utils import trainable_variables


def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class QLearning(ReinforcementLearningUpdater):
    double = Param()
    replay_max_size = Param()
    replay_threshold = Param()
    replay_proportion = Param()
    target_update_rate = Param()
    steps_per_target_update = Param()
    samples_per_update = Param()
    update_batch_size = Param()

    def __init__(self, env, q_network, target_network, **kwargs):
        self.q_network = q_network
        self.target_network = target_network

        super(QLearning, self).__init__(env, q_network, **kwargs)

        self.n_rollouts_since_target_update = 0

        self.replay_buffer = PrioritizedReplayBuffer(
            self.replay_max_size, self.replay_proportion, self.replay_threshold)

        self.clear_buffers()

    def build_graph(self):
        with tf.name_scope("train"):
            self.obs = tf.placeholder(tf.float32, shape=(None, None)+self.obs_shape, name="_obs")
            self.actions = tf.placeholder(tf.float32, shape=(None, None, self.n_actions), name="_actions")
            self.rewards = tf.placeholder(tf.float32, shape=(None, None, 1), name="_rewards")
            self.reward_per_ep = tf.squeeze(tf.reduce_sum(tf.reduce_mean(self.rewards, axis=1), axis=0, name="_reward_per_ep"))

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

            td_error, (prev_q_value, _, _) = dynamic_rnn(
                q_learning_cell, inp, initial_state=initial_state,
                parallel_iterations=1, swap_memory=False,
                time_major=True)

            final_td_error = final_rewards - prev_q_value
            td_error = tf.concat((td_error, tf.expand_dims(final_td_error, 0)), axis=0)
            self.q_loss = tf.reduce_mean(clipped_error(td_error))
            tf.summary.scalar("loss_q", self.q_loss)

            loss = self.q_loss

            if self.l2_norm_penalty > 0:
                policy_network_variables = trainable_variables(self.q_network.scope.name)
                self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
                tf.summary.scalar("loss_reg", self.reg_loss)
                loss += self.l2_norm_penalty * self.reg_loss

            self.loss = loss

            tf.summary.scalar("reward_per_ep", self.reward_per_ep)

            self._build_train()

        if self.steps_per_target_update is None:
            assert self.target_update_rate is not None

            with tf.name_scope("update_target_network"):
                q_network_variables = trainable_variables(self.q_network.scope.name)
                target_network_variables = trainable_variables(self.target_network.scope.name)
                target_network_update = []

                for v_source, v_target in zip(q_network_variables, target_network_variables):
                    update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                    target_network_update.append(update_op)
                self.target_network_update = tf.group(*target_network_update)
        else:
            with tf.name_scope("update_target_network"):
                q_network_variables = trainable_variables(self.q_network.scope.name)
                target_network_variables = trainable_variables(self.target_network.scope.name)
                target_network_update = []

                for v_source, v_target in zip(q_network_variables, target_network_variables):
                    update_op = v_target.assign(v_source)
                    target_network_update.append(update_op)
                self.target_network_update = tf.group(*target_network_update)

    def _build_feeddict(self):
        obs = np.array(self.obs_buffer)
        actions = np.array(self.action_buffer)
        rewards = np.array(self.reward_buffer)

        feed_dict = {
            self.obs: obs,
            self.actions: actions,
            self.rewards: rewards}
        return feed_dict

    def _update(self, batch_size, summary_op=None):
        rollouts_remaining = batch_size
        sess = tf.get_default_session()

        while rollouts_remaining > 0:
            n_rollouts = min(rollouts_remaining, self.samples_per_update)
            rollouts_remaining -= n_rollouts

            # Collect experiences
            self.clear_buffers()
            self.env.do_rollouts(self, self.q_network, n_rollouts, mode='train')
            self.replay_buffer.add(self.obs_buffer, self.action_buffer, self.reward_buffer)

            # Get a batch from replay buffer for performing update
            (self.obs_buffer,
             self.action_buffer,
             self.reward_buffer) = self.replay_buffer.get_batch(self.update_batch_size)

            # Perform the update
            feed_dict = self._build_feeddict()
            sess.run([self.train_op], feed_dict=feed_dict)

            if (self.steps_per_target_update is None) or (self.n_rollouts_since_target_update > self.steps_per_target_update):
                # Update target network
                sess.run(self.target_network_update)
                self.n_rollouts_since_target_update = 0
            else:
                self.n_rollouts_since_target_update += n_rollouts

        # Run some evaluation rollouts
        if summary_op is not None:
            self.clear_buffers()
            self.env.do_rollouts(self, self.q_network, batch_size, mode='train')
            feed_dict = self._build_feeddict()

            train_summary, train_loss, train_reward = sess.run(
                [summary_op, self.loss, self.reward_per_ep], feed_dict=feed_dict)

            self.clear_buffers()
            self.env.do_rollouts(self, self.q_network, batch_size, mode='val')
            feed_dict = self._build_feeddict()

            val_summary, val_loss, val_reward = sess.run(
                [summary_op, self.loss, self.reward_per_ep], feed_dict=feed_dict)

            return_value = train_summary, -train_reward, val_summary, -val_reward
        else:
            self.clear_buffers()
            self.env.do_rollouts(self, self.q_network, batch_size, mode='train')
            feed_dict = self._build_feeddict()

            train_reward, = sess.run([self.reward_per_ep], feed_dict=feed_dict)

            return_value = -train_reward

        return return_value


class QLearningCell(RNNCell):
    """ Used in defining the loss function that we differentiate to perform QLearning.

    Reward needs to be offset by one time step; we always want to be working with
    reward caused by previous state and action.

    """
    def __init__(self, q_network, target_network, double, gamma):
        self.q_network = q_network
        self.target_network = target_network
        self.double = double
        self.gamma = gamma

    def __call__(self, inp, state, scope=None):
        with tf.name_scope(scope or 'q_learning_cell'):
            observations, actions, rewards = inp
            prev_q_value, qn_state, tn_state = state

            _, target_values, new_tn_state = self.target_network.build(observations, tn_state)
            _, q_values, new_qn_state = self.q_network.build(observations, qn_state)

            if self.double:
                action_selection = tf.argmax(tf.stop_gradient(target_values), 1, name="action_selection")
                action_selection_mask = tf.cast(tf.one_hot(action_selection, self.q_network.n_actions, 1, 0), tf.float32)
                bootstrap = tf.reduce_sum(action_selection_mask * q_values, axis=-1, keep_dims=True)
            else:
                bootstrap = tf.reduce_max(tf.stop_gradient(target_values), axis=-1, keep_dims=True)

            td_error = rewards + self.gamma * bootstrap - prev_q_value
            q_value_selected_action = tf.reduce_sum(actions * q_values, axis=1, keep_dims=True)

            new_state = (q_value_selected_action, new_qn_state, new_tn_state)
            return td_error, new_state

    @property
    def state_size(self):
        return (1, 1, self.q_network.state_size, self.target_network.state_size)

    @property
    def output_size(self):
        return 1

    def zero_state(self, batch_size, dtype):
        initial_state = (
            tf.fill((batch_size, 1), 0.0),  # q-value of action selected on previous time step
            self.q_network.zero_state(batch_size, dtype),  # hidden state for q network
            self.target_network.zero_state(batch_size, dtype))  # hidden state for target network
        return initial_state

    def initial_state(self, first_obs, first_actions, batch_size, dtype):
        _, qn_state, tn_state = self.zero_state(batch_size, dtype)

        _, _, new_tn_state = self.target_network.build(first_obs, tn_state)
        _, q_values, new_qn_state = self.q_network.build(first_obs, qn_state)
        q_value_selected_action = tf.reduce_sum(first_actions * q_values, axis=1, keep_dims=True)

        return q_value_selected_action, new_qn_state, new_tn_state


class PrioritizedReplayBuffer(object):
    def __init__(self, max_size, ro=None, threshold=0.0):
        """
        A heuristic from:
        Language Understanding for Text-based Games using Deep Reinforcement Learning.

        Parameters
        ----------
        ro: float in [0, 1]
            Proportion of each batch that comes from the high-priority subset.
        threshold: float
            All episodes that recieve reward above this threshold are given high-priority.

        """
        self.max_size = max_size
        self.threshold = threshold
        self.ro = ro
        self.low_buffer = deque()
        self.high_buffer = deque()

    @staticmethod
    def _get_batch(batch_size, buff):
        replace = batch_size > len(buff)
        idx = np.random.choice(len(buff), batch_size, replace=replace)
        obs, actions, rewards = [], [], []
        for i in idx:
            o, a, r = buff[i]
            obs.append(o)
            actions.append(a)
            rewards.append(r)

        # TODO: need to pad returned array with zeros if trajectories are different lengths
        obs = np.transpose(obs, (1, 0, 2))
        actions = np.transpose(actions, (1, 0, 2))
        rewards = np.transpose(rewards, (1, 0, 2))

        return obs, actions, rewards

    def get_batch(self, batch_size):
        if self.ro > 0 and len(self.high_buffer) > 0:
            n_high_priority = int(self.ro * batch_size)
            n_low_priority = batch_size - n_high_priority

            low_priority = self._get_batch(n_low_priority, self.low_buffer)
            high_priority = self._get_batch(n_high_priority, self.high_buffer)

            obs = np.concatenate((low_priority[0], high_priority[0]), axis=1)
            actions = np.concatenate((low_priority[1], high_priority[1]), axis=1)
            rewards = np.concatenate((low_priority[2], high_priority[2]), axis=1)
            return obs, actions, rewards
        else:
            return self._get_batch(batch_size, self.low_buffer)

    def add(self, obs, actions, rewards):
        # Assumes shape is (n_timesteps, batch_size, dim)
        obs = list(np.transpose(obs, (1, 0, 2)))
        actions = list(np.transpose(actions, (1, 0, 2)))
        rewards = list(np.transpose(rewards, (1, 0, 2)))

        for o, a, r in zip(obs, actions, rewards):
            buff = self.high_buffer if (r.sum() > self.threshold and self.ro > 0) else self.low_buffer
            buff.append((o, a, r))
            while len(buff) > self.max_size:
                buff.popleft()
