import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from collections import deque
from itertools import cycle, islice
from pyskiplist import SkipList

from dps import cfg
from dps.rl import PolicyOptimization, RolloutBatch
from dps.utils import Param, build_gradient_train_op, masked_mean, build_scheduled_value


class DuelingHead(object):
    def __init__(self, value_fn, adv_fn, kind="mean"):
        self.value_fn, self.adv_fn = value_fn, adv_fn

        if kind not in "mean max".split():
            raise Exception('`kind` must be one of "mean", "max".')
        self.kind = kind

    def __call__(self, inp, n_actions):
        value = self.value_fn(inp, 1)
        advantage = self.adv_fn(inp, n_actions)

        if self.kind == "mean":
            normed_advantage = advantage - tf.reduce_mean(advantage, axis=-1, keep_dims=True)
        else:
            normed_advantage = advantage - tf.reduce_max(advantage, axis=-1, keep_dims=True)
        return value + normed_advantage


def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class QLearning(PolicyOptimization):
    double = Param()
    target_update_rate = Param()
    steps_per_target_update = Param()
    update_batch_size = Param()
    gamma = Param()
    opt_steps_per_batch = Param()
    init_steps = Param(0)
    pure_exploration_steps = Param(1000)

    optimizer_spec = Param()
    lr_schedule = Param()

    replay_max_size = Param()
    alpha = Param()
    beta_schedule = Param()
    n_partitions = Param(100)

    def __init__(self, q_network, **kwargs):
        self.policy = self.q_network = q_network
        self.target_network = q_network.deepcopy("target_network")

        super(QLearning, self).__init__(**kwargs)

        self.n_steps_since_target_update = 0

    def build_placeholders(self):
        self.weights = tf.placeholder(tf.float32, shape=(cfg.T, None, 1), name="_weights")
        super(QLearning, self).build_placeholders()

    def build_feed_dict(self, rollouts, weights=None):
        if weights is None:
            weights = np.ones_like(rollouts.mask)

        return {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.mask: rollouts.mask,
            self.weights: weights
        }

    def build_update_ops(self):
        tvars = self.q_network.trainable_variables()

        self.train_op, train_summaries = build_gradient_train_op(
            self.q_loss, tvars, self.optimizer_spec, self.lr_schedule)
        self.train_summary_op = tf.summary.merge(train_summaries)

        self.init_train_op, train_summaries = build_gradient_train_op(
            self.monte_carlo_loss, tvars, self.optimizer_spec, self.lr_schedule)
        self.init_train_summary_op = tf.summary.merge(train_summaries)

    def build_target_network_update(self):
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

        if self.double:
            # Select maximum action using policy network
            action_selection = tf.argmax(q_values, -1, name="action_selection")
            action_selection_mask = tf.cast(tf.one_hot(action_selection, self.q_network.n_actions, 1, 0), tf.float32)

            # Evaluate selected action using target network
            bootstrap_values = tf.reduce_sum(action_selection_mask * bootstrap_values, axis=-1, keep_dims=True)
        else:
            bootstrap_values = tf.reduce_max(bootstrap_values, axis=-1, keep_dims=True)

        bootstrap_values = tf.concat(
            [bootstrap_values[1:, :, :], tf.zeros_like(bootstrap_values[:1, :, :])],
            axis=0)
        bootstrap_values = tf.stop_gradient(bootstrap_values)

        self.targets = self.rewards + self.gamma * bootstrap_values

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
        ])

        self.recorded_values = [
            ('loss', -self.reward_per_ep),
            ('reward_per_ep', self.reward_per_ep),
            ('q_loss', self.q_loss),
        ]

        self.build_target_network_update()

    def update_params(self, rollouts, collect_summaries, feed_dict, init):
        if init:
            train_op, train_summary_op = self.init_train_op, self.init_train_summary_op
        else:
            train_op, train_summary_op = self.train_op, self.train_summary_op

        sess = tf.get_default_session()
        if collect_summaries:
            train_summaries, _ = sess.run([train_summary_op, train_op], feed_dict=feed_dict)
            summaries = train_summaries
        else:
            sess.run(train_op, feed_dict=feed_dict)
            summaries = b''
        return summaries

    def update(self, rollouts, collect_summaries):
        # Store sampled rollouts
        self.replay_buffer.add(rollouts)

        sess = tf.get_default_session()

        global_step = sess.run(tf.contrib.framework.get_or_create_global_step())

        pure_exploration = global_step < self.pure_exploration_steps
        init = global_step < self.init_steps

        if pure_exploration:
            return b''

        for i in range(self.opt_steps_per_batch):
            # Sample rollouts from replay buffer
            rollouts, weights = self.replay_buffer.get_batch(self.update_batch_size)

            weights = np.tile(weights.reshape(1, -1, 1), (rollouts.T, 1, 1))

            feed_dict = self.build_feed_dict(rollouts, weights=weights)

            # Perform the update
            _collect_summaries = collect_summaries and i == self.opt_steps_per_batch-1
            summaries = self.update_params(rollouts, _collect_summaries, feed_dict, init)

            # Update rollout priorities in replay buffer.
            if init:
                td_error = sess.run(
                    self.monte_carlo_error_unweighted, feed_dict=feed_dict)
            else:
                td_error = sess.run(self.td_error, feed_dict=feed_dict)
            priority = np.abs(td_error).sum(axis=0).reshape(-1)
            self.replay_buffer.update_priority(priority)

        if init:
            self.target_network.set_params_flat(self.q_network.get_params_flat())
        else:
            # Update target network if applicable
            if (self.steps_per_target_update is None) or (self.n_steps_since_target_update > self.steps_per_target_update):
                sess.run(self.target_network_update)
                self.n_steps_since_target_update = 0
            else:
                self.n_steps_since_target_update += 1

        return summaries

    def evaluate(self, rollouts):
        feed_dict = self.build_feed_dict(rollouts)

        sess = tf.get_default_session()

        eval_summaries, *values = (
            sess.run(
                [self.eval_summary_op] + [v for _, v in self.recorded_values],
                feed_dict=feed_dict))

        record = {k: v for v, (k, _) in zip(values, self.recorded_values)}
        return eval_summaries, record


class HeuristicReplayBuffer(object):
    def __init__(self, max_size, ro=0.0, threshold=0.0):
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
            rollouts = RolloutBatch.join([low_priority, high_priority])
        else:
            rollouts = self._get_batch(batch_size, self.low_buffer)
        return rollouts, np.ones(rollouts.batch_size)

    def update_priority(self, priorities):
        pass

    def add(self, rollouts):
        assert isinstance(rollouts, RolloutBatch)
        for r in rollouts.split():
            buff = self.high_buffer if (r.rewards.sum() > self.threshold and self.ro > 0) else self.low_buffer
            buff.append(r)
            while len(buff) > self.max_size:
                buff.popleft()


class PrioritizedReplayBuffer(object):
    """
    Implements rank-based version of Prioritized Experience Replay.

    Parameters
    ----------
    size: int
        Maximum number of experiences to store.
    n_partitions: int
        Number of partitions to use for the sampling approximation.
    alpha: float > 0
        Degree of prioritization (similar to a softmax temperature); 0 corresponds to no prioritization
        (uniform distribution), inf corresponds to degenerate distribution which always picks element
        with highest priority.
    beta: 1 > float > 0
        Degree of importance sampling correction, 0 corresponds to no correction, 1 corresponds to
        full correction. Usually anneal linearly from an initial value beta_0 to a value of 1 by the
        end of learning.

    """
    def __init__(self, size, n_partitions, alpha, beta):
        self.size = size
        self.n_partitions = n_partitions
        self.alpha = alpha
        self.beta = beta

        self.index = 0

        self._experiences = {}

        # Note this is actually a MIN priority queue, so to make it act like a MAX priority
        # queue, we use the negative of the provided priorities.
        self.skip_list = SkipList()
        self.distributions = self.build_distribution()

        self._active_set = None

    @property
    def n_experiences(self):
        return len(self._experiences)

    def build_distribution(self):
        pdf = np.arange(1, self.size+1)**-self.alpha
        pdf /= pdf.sum()

        cdf = np.cumsum(pdf)

        # Whenever the CDF crosses one of the discretization bucket boundaries,
        # we assign the index where the crossing occurred to the next bucket rather
        # than the current one.
        strata_starts, strata_ends = [], []
        start_idx, end_idx = 0, 0
        for s in range(self.n_partitions):
            strata_starts.append(start_idx)
            if s == self.n_partitions-1:
                strata_ends.append(len(cdf))
            else:
                while cdf[end_idx] < (s+1) / self.n_partitions:
                    end_idx += 1
                if start_idx == end_idx:
                    strata_ends.append(end_idx + 1)
                else:
                    strata_ends.append(end_idx)
            start_idx = end_idx

        self.strata_starts = strata_starts
        self.strata_ends = strata_ends
        self.pdf = pdf

    def add(self, rollouts):
        assert isinstance(rollouts, RolloutBatch)
        for r in rollouts.split():
            # If there was already an experience at location `self.index`, it is effectively ejected.
            self._experiences[self.index] = r

            # Insert with minimum priority initially.
            if self.skip_list:
                priority = self.skip_list[0][0]
            else:
                priority = 0.0
            self.skip_list.insert(priority, self.index)

            self.index = (self.index + 1) % self.size

    def update_priority(self, priorities):
        """ update priority after calling `get_batch` """
        if self._active_set is None:
            raise Exception("``update_priority`` should only called after calling ``get_batch``.")

        for (p_idx, e_idx, old_priority), new_priority in zip(self._active_set, priorities):
            # negate `new_priority` because SkipList puts lowest first.
            if old_priority != -new_priority:
                del self.skip_list[p_idx]
                self.skip_list.insert(-new_priority, e_idx)

    def get_batch(self, batch_size):
        priority_indices = []
        start = 0
        permutation = np.random.permutation(self.n_partitions)

        # batch_size = min(batch_size, self.n_experiences)

        for i in islice(cycle(permutation), batch_size):
            start, end = self.strata_starts[i], self.strata_ends[i]
            priority_indices.append(np.random.randint(start, end))

        p_x = [self.pdf[idx] for idx in priority_indices]

        beta = self.beta
        if isinstance(beta, tf.Tensor):
            beta = tf.get_default_session().run(beta)

        w = (np.array(p_x) * self.size)**-beta
        w /= w.max()

        # When we aren't full, map priority_indices (which are in range(self.size)) down to range(self.n_experiences)
        if self.n_experiences < self.size:
            priority_indices = [int(np.floor(self.n_experiences * (i / self.size))) for i in priority_indices]

        self._active_set = []
        for p_idx in priority_indices:
            priority, e_idx = self.skip_list[p_idx]
            self._active_set.append((p_idx, e_idx, priority))

        experiences = RolloutBatch.join([self._experiences[e_idx] for _, e_idx, _ in self._active_set])

        return experiences, w
