import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from collections import deque
import sys
from itertools import cycle, islice

from dps import cfg
from dps.rl import PolicyOptimization, RolloutBatch
from dps.utils import Param, build_gradient_train_op, masked_mean


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
    replay_max_size = Param()
    replay_threshold = Param()
    replay_proportion = Param()
    target_update_rate = Param()
    steps_per_target_update = Param()
    update_batch_size = Param()
    optimizer_spec = Param()
    lr_schedule = Param()
    gamma = Param()
    opt_steps_per_batch = Param()

    def __init__(self, q_network, **kwargs):
        self.policy = self.q_network = q_network
        self.target_network = q_network.deepcopy("target_network")

        super(QLearning, self).__init__(**kwargs)

        self.n_steps_since_target_update = 0

        self.replay_buffer = HeuristicReplayBuffer(
            self.replay_max_size, self.replay_proportion, self.replay_threshold)

    def build_placeholders(self):
        self.weights = tf.placeholder(tf.float32, shape=(cfg.T, None, 1), name="_weights")
        super(QLearning, self).build_placeholders()

    def _build_graph(self, is_training, exploration):
        self.build_placeholders()

        self.q_network.set_exploration(exploration)
        self.target_network.set_exploration(exploration)

        batch_size = tf.shape(self.obs)[1]

        # Q values
        (_, q_values), _ = dynamic_rnn(
            self.q_network, self.obs, initial_state=self.q_network.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=True)
        q_values_selected_actions = tf.reduce_sum(q_values * self.actions, axis=-1, keep_dims=True)
        mean_q_value = masked_mean(q_values_selected_actions, self.mask)

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
            bootstrap_values = tf.stop_gradient(bootstrap_values)

        bootstrap_values = tf.concat(
            [bootstrap_values[1:, :, :], tf.zeros_like(bootstrap_values[0:1, :, :])],
            axis=0)
        bootstrap_values = tf.stop_gradient(bootstrap_values)

        mean_bootstrap_value = masked_mean(bootstrap_values, self.mask)

        self.td_error = (self.rewards + self.gamma * bootstrap_values - q_values_selected_actions) * self.mask
        self.weighted_td_error = self.td_error * self.weights
        self.q_loss = masked_mean(clipped_error(self.td_error), self.mask)

        tvars = self.q_network.trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            self.q_loss, tvars, self.optimizer_spec, self.lr_schedule)

        self.train_summary_op = tf.summary.merge(train_summaries)

        self.eval_summary_op = tf.summary.merge([
            tf.summary.scalar("q_loss", self.q_loss),
            tf.summary.scalar("reward_per_ep", self.reward_per_ep),
            tf.summary.scalar("mean_q_value", mean_q_value),
            tf.summary.scalar("mean_q_value_target_network", mean_q_value_target_network),
            tf.summary.scalar("bootstrap_value", mean_bootstrap_value),
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
        # Store sampled rollouts
        self.replay_buffer.add(rollouts)

        # Sample rollouts from replay buffer
        rollouts, weights = self.replay_buffer.get_batch(self.update_batch_size)

        weights = np.tile(weights.reshape(1, -1, 1), (rollouts.T, 1, 1))

        # Perform the update
        feed_dict = {
            self.obs: rollouts.o,
            self.actions: rollouts.a,
            self.rewards: rollouts.r,
            self.mask: rollouts.mask,
            self.weights: weights
        }

        sess = tf.get_default_session()
        for i in range(self.opt_steps_per_batch-1):
            sess.run(self.train_op, feed_dict=feed_dict)

        if collect_summaries:
            train_summaries, _ = sess.run([self.train_summary_op, self.train_op], feed_dict=feed_dict)
            summaries = train_summaries
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            summaries = b''

        # Update rollout priorities in replay buffer.
        td_error = sess.run(self.td_error, feed_dict=feed_dict)
        priority = np.abs(td_error).sum(axis=0).reshape(-1)
        self.replay_buffer.update_priority(priority)

        # Update target network if applicable
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
            self.mask: rollouts.mask,
            self.weights: np.ones_like(rollouts.mask)
        }

        sess = tf.get_default_session()

        eval_summaries, *values = (
            sess.run(
                [self.eval_summary_op] + [v for _, v in self.recorded_values],
                feed_dict=feed_dict))

        record = {k: v for v, (k, _) in zip(values, self.recorded_values)}
        return eval_summaries, record


class HeuristicReplayBuffer(object):
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
    def __init__(self, size, n_partitions, alpha, beta):
        self.size = size
        self.n_partitions = n_partitions
        self.alpha = alpha
        self.beta = beta

        self.index = 0

        self._experiences = {}
        self.priority_queue = BinaryHeap(self.size)
        self.distributions = self.build_distribution()

        self._active_indices = None

    @property
    def n_experiences(self):
        return len(self._experiences)

    def build_distribution(self):
        pdf = np.arange(1, self.size+1)**-self.alpha
        pdf /= pdf.sum()

        cdf = np.cumsum(pdf)

        strata_ends = []
        i = 0
        for s in range(self.n_partitions):
            while i < len(cdf) and (cdf[i] < (s+1) / self.n_partitions):
                i += 1
            strata_ends[s] = i

        self.strata_ends = strata_ends
        self.pdf = pdf

    def add(self, rollouts):
        assert isinstance(rollouts, RolloutBatch)
        for r in rollouts.split():
            self._experience[self.index] = r

            # Insert with maximum priority initially.
            priority = self.priority_queue.get_max_priority()
            self.priority_queue.update(priority, self.index)

            self.index = (self.index + 1) % self.size

    def rebalance(self):
        """ rebalance priority queue """
        self.priority_queue.balance_tree()

    def update_priority(self, priorities):
        """ update priority after calling `get_batch` """
        if self._active_indices is None:
            raise Exception("``update_priority`` should only called after calling ``get_batch``.")

        for idx, p in zip(self._active_indices, priorities):
            self.priority_queue.update(p, idx)

    def get_batch(self, batch_size):
        indices = []
        start = 0
        for end in islice(cycle(self.strata_ends), batch_size):
            indices.append(np.random.randint(start, end))
            start = end

        p_x = [self.pdf[idx] for idx in indices]

        w = (np.array(p_x) * self.size)**-self.beta
        w /= w.max()

        # When we aren't full, map indices (which are in range(self.size)) down to range(self.n_experiences)
        if self.n_experiences < self.size:
            indices = [int(np.floor(self.n_experiences * (i / self.size))) for i in indices]

        experience_indices = self.priority_queue.priority_to_experience(indices)
        experiences = RolloutBatch.join([self._experiences[i] for i in experience_indices])

        self._active_indices = experience_indices

        return experiences, w


def list_to_dict(in_list):
    return dict((i, in_list[i]) for i in range(0, len(in_list)))


def exchange_key_value(in_dict):
    return dict((in_dict[i], i) for i in in_dict)


class BinaryHeap(object):
    def __init__(self, priority_size=100, priority_init=None, replace=True):
        self.e2p = {}
        self.p2e = {}
        self.replace = replace

        if priority_init is None:
            self.priority_queue = {}
            self.size = 0
            self.max_size = priority_size
        else:
            # not yet test
            self.priority_queue = priority_init
            self.size = len(self.priority_queue)
            self.max_size = None or self.size

            experience_list = list(map(lambda x: self.priority_queue[x], self.priority_queue))
            self.p2e = list_to_dict(experience_list)
            self.e2p = exchange_key_value(self.p2e)
            for i in range(int(self.size / 2), -1, -1):
                self.down_heap(i)

    def __repr__(self):
        """
        :return: string of the priority queue, with level info
        """
        if self.size == 0:
            return 'No element in heap!'
        to_string = ''
        level = -1
        max_level = np.floor(np.log(self.size, 2))

        for i in range(1, self.size + 1):
            now_level = np.floor(np.log(i, 2))
            if level != now_level:
                to_string = to_string + ('\n' if level != -1 else '') + '    ' * (max_level - now_level)
                level = now_level
            to_string = to_string + '%.2f ' % self.priority_queue[i][1] + '    ' * (max_level - now_level)

        return to_string

    def check_full(self):
        return self.size > self.max_size

    def _insert(self, priority, e_id):
        """
        insert new experience id with priority
        (maybe don't need get_max_priority and implement it in this function)
        :param priority: priority value
        :param e_id: experience id
        :return: bool
        """
        self.size += 1

        if self.check_full() and not self.replace:
            sys.stderr.write('Error: no space left to add experience id %d with priority value %f\n' % (e_id, priority))
            return False
        else:
            self.size = min(self.size, self.max_size)

        self.priority_queue[self.size] = (priority, e_id)
        self.p2e[self.size] = e_id
        self.e2p[e_id] = self.size

        self.up_heap(self.size)
        return True

    def update(self, priority, e_id):
        """
        update priority value according its experience id
        :param priority: new priority value
        :param e_id: experience id
        :return: bool
        """
        if e_id in self.e2p:
            p_id = self.e2p[e_id]
            self.priority_queue[p_id] = (priority, e_id)
            self.p2e[p_id] = e_id

            self.down_heap(p_id)
            self.up_heap(p_id)
            return True
        else:
            # this e id is new, do insert
            return self._insert(priority, e_id)

    def get_max_priority(self):
        """
        get max priority, if no experience, return 1
        :return: max priority if size > 0 else 1
        """
        if self.size > 0:
            return self.priority_queue[1][0]
        else:
            return 1

    def pop(self):
        """
        pop out the max priority value with its experience id
        :return: priority value & experience id
        """
        if self.size == 0:
            sys.stderr.write('Error: no value in heap, pop failed\n')
            return False, False

        pop_priority, pop_e_id = self.priority_queue[1]
        self.e2p[pop_e_id] = -1
        # replace first
        last_priority, last_e_id = self.priority_queue[self.size]
        self.priority_queue[1] = (last_priority, last_e_id)
        self.size -= 1
        self.e2p[last_e_id] = 1
        self.p2e[1] = last_e_id

        self.down_heap(1)

        return pop_priority, pop_e_id

    def up_heap(self, i):
        """
        upward balance
        :param i: tree node i
        :return: None
        """
        if i > 1:
            parent = np.floor(i / 2)
            if self.priority_queue[parent][0] < self.priority_queue[i][0]:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[parent]
                self.priority_queue[parent] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[parent][1]] = parent
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[parent] = self.priority_queue[parent][1]
                # up heap parent
                self.up_heap(parent)

    def down_heap(self, i):
        """
        downward balance
        :param i: tree node i
        :return: None
        """
        if i < self.size:
            greatest = i
            left, right = i * 2, i * 2 + 1
            if left < self.size and self.priority_queue[left][0] > self.priority_queue[greatest][0]:
                greatest = left
            if right < self.size and self.priority_queue[right][0] > self.priority_queue[greatest][0]:
                greatest = right

            if greatest != i:
                tmp = self.priority_queue[i]
                self.priority_queue[i] = self.priority_queue[greatest]
                self.priority_queue[greatest] = tmp
                # change e2p & p2e
                self.e2p[self.priority_queue[i][1]] = i
                self.e2p[self.priority_queue[greatest][1]] = greatest
                self.p2e[i] = self.priority_queue[i][1]
                self.p2e[greatest] = self.priority_queue[greatest][1]
                # down heap greatest
                self.down_heap(greatest)

    def get_priority(self):
        """
        get all priority value
        :return: list of priority
        """
        return list(map(lambda x: x[0], self.priority_queue.values()))[0:self.size]

    def get_e_id(self):
        """
        get all experience id in priority queue
        :return: list of experience ids order by their priority
        """
        return list(map(lambda x: x[1], self.priority_queue.values()))[0:self.size]

    def balance_tree(self):
        """
        rebalance priority queue
        :return: None
        """
        sort_array = sorted(self.priority_queue.values(), key=lambda x: x[0], reverse=True)
        # reconstruct priority_queue
        self.priority_queue.clear()
        self.p2e.clear()
        self.e2p.clear()
        cnt = 1
        while cnt <= self.size:
            priority, e_id = sort_array[cnt - 1]
            self.priority_queue[cnt] = (priority, e_id)
            self.p2e[cnt] = e_id
            self.e2p[e_id] = cnt
            cnt += 1
        # sort the heap
        for i in range(np.floor(self.size / 2), 1, -1):
            self.down_heap(i)

    def priority_to_experience(self, priority_ids):
        """
        retrieve experience ids by priority ids
        :param priority_ids: list of priority id
        :return: list of experience id
        """
        return [self.p2e[i] for i in priority_ids]
