import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from collections import deque

from dps.rl import PolicyOptimization, RolloutBatch
from dps.utils import Param, build_gradient_train_op


def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class QLearning(PolicyOptimization):
    double = Param()
    replay_max_size = Param()
    replay_threshold = Param()
    replay_proportion = Param()
    target_update_rate = Param()
    steps_per_target_update = Param()
    update_batch_size = Param()
    optimizer_spec = Param()
    lr_schedule = Param()
    gamma = Param()

    def __init__(self, q_network, **kwargs):
        self.policy = self.q_network = q_network
        self.target_network = q_network.deepcopy("target_network")

        super(QLearning, self).__init__(**kwargs)

        self.n_steps_since_target_update = 0

        self.replay_buffer = PrioritizedReplayBuffer(
            self.replay_max_size, self.replay_proportion, self.replay_threshold)

    def _build_graph(self, is_training, exploration):
        self.build_placeholders()

        self.q_network.set_exploration(exploration)
        self.target_network.set_exploration(exploration)

        batch_size = tf.shape(self.obs)[1]

        # Q values
        (_, q_values), _ = dynamic_rnn(
            self.q_network, self.obs, initial_state=self.q_network.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=True)
        q_values_selected_actions = q_values * self.actions

        # Bootstrap values
        (_, bootstrap_values), _ = dynamic_rnn(
            self.target_network, self.obs, initial_state=self.target_network.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=True)

        bootstrap_values = tf.concat(
            [bootstrap_values[1:, :, :], tf.zeros_like(bootstrap_values[0:1, :, :])],
            axis=0)
        bootstrap_values = tf.reduce_max(bootstrap_values, axis=-1, keep_dims=True)
        bootstrap_values = tf.stop_gradient(bootstrap_values)

        if self.double:
            # Select maximum action using policy network
            action_selection = tf.argmax(tf.stop_gradient(q_values), 1, name="action_selection")
            action_selection_mask = tf.cast(tf.one_hot(action_selection, self.q_network.n_actions, 1, 0), tf.float32)

            # Evaluate selected action using target network
            bootstrap_values = tf.reduce_sum(action_selection_mask * bootstrap_values, axis=-1, keep_dims=True)

        td_error = self.rewards + self.gamma * bootstrap_values - q_values_selected_actions

        self.q_loss = tf.reduce_mean(clipped_error(td_error))

        tvars = self.q_network.trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            self.q_loss, tvars, self.optimizer_spec, self.lr_schedule)

        self.train_summary_op = tf.summary.merge(train_summaries)

        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("q_loss", self.q_loss),
            tf.summary.scalar("reward_per_ep", self.reward_per_ep),
        ])

        self.recorded_values = [
            ('loss', -self.reward_per_ep),
            ('reward_per_ep', self.reward_per_ep),
            ('q_loss', self.q_loss),
        ]

        q_network_variables = self.q_network.trainable_variables()
        target_network_variables = self.target_network.trainable_variables()

        with tf.name_scope("update_target_network"):
            target_network_update = []

            if self.steps_per_target_update is not None:
                for v_source, v_target in zip(q_network_variables, target_network_variables):
                    update_op = v_target.assign(v_source)
                    target_network_update.append(update_op)
            else:
                assert self.target_update_rate is not None
                for v_source, v_target in zip(q_network_variables, target_network_variables):
                    update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                    target_network_update.append(update_op)

            self.target_network_update = tf.group(*target_network_update)

    def update(self, rollouts, collect_summaries):
        self.replay_buffer.add(rollouts)
        rollouts = self.replay_buffer.get_batch(self.update_batch_size)

        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.mask: rollouts.mask
        }

        # Perform the update
        sess = tf.get_default_session()
        if collect_summaries:
            train_summaries, _ = sess.run([self.train_summary_op, self.train_op], feed_dict=feed_dict)
            summaries = train_summaries
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            summaries = b''

        if (self.steps_per_target_update is None) or (self.n_steps_since_target_update > self.steps_per_target_update):
            sess.run(self.target_network_update)
            self.n_steps_since_target_update = 0
        else:
            self.n_steps_since_target_update += 1

        return summaries

    def evaluate(self, rollouts):
        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.mask: rollouts.mask
        }

        sess = tf.get_default_session()

        eval_summaries, *values = (
            sess.run(
                [self.eval_summary_op] + [v for _, v in self.recorded_values],
                feed_dict=feed_dict))

        record = {k: v for v, (k, _) in zip(values, self.recorded_values)}
        return eval_summaries, record


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
        return RolloutBatch.join([buff[i] for i in idx])

    def get_batch(self, batch_size):
        if self.ro > 0 and len(self.high_buffer) > 0:
            n_high_priority = int(self.ro * batch_size)
            n_low_priority = batch_size - n_high_priority

            low_priority = self._get_batch(n_low_priority, self.low_buffer)
            high_priority = self._get_batch(n_high_priority, self.high_buffer)
            return RolloutBatch.join([low_priority, high_priority])
        else:
            return self._get_batch(batch_size, self.low_buffer)

    def add(self, rollouts):
        assert isinstance(rollouts, RolloutBatch)
        for r in rollouts.split():
            buff = self.high_buffer if (r.rewards.sum() > self.threshold and self.ro > 0) else self.low_buffer
            buff.append(r)
            while len(buff) > self.max_size:
                buff.popleft()
