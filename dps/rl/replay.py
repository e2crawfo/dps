import numpy as np
import tensorflow as tf
from itertools import cycle, islice
from pyskiplist import SkipList

from dps.rl import RolloutBatch, RLObject
from dps.utils.tf import build_scheduled_value


class ReplayBuffer(RLObject):
    """
    Basic Experience Replay.

    Parameters
    ----------
    size: int
        Maximum number of experiences to store.
    min_experiences: int > 0
        Minimum number of experiences that must be stored in the replay buffer before it will return
        a valid batch when `get_batch` is called. Before this point, it returns None, indicating that
        whatever is making use of this replay memory should not make an update.

    """
    def __init__(self, size, min_experiences=None, name=None):
        self.size = size
        self.min_experiences = min_experiences
        self.index = 0
        self._experiences = {}

        super(ReplayBuffer, self).__init__(name)

    @property
    def n_experiences(self):
        return len(self._experiences)

    def add_rollouts(self, rollouts):
        assert isinstance(rollouts, RolloutBatch)
        for r in rollouts.split():
            # If there was already an experience at location `self.index`, it is effectively ejected.
            self._experiences[self.index] = r
            self.index = (self.index + 1) % self.size

    def get_batch(self, batch_size):
        no_sample = (
            (self.min_experiences is not None and
             self.n_experiences < self.min_experiences) or
            self.n_experiences < batch_size)
        if no_sample:
            return None, None

        indices = np.random.randint(self.n_experiences, size=batch_size)
        experiences = RolloutBatch.join([self._experiences[i] for i in indices])

        weights = np.ones_like(indices).astype('f')

        return experiences, weights


class PrioritizedReplayBuffer(RLObject):
    """ Implements rank-based version of Prioritized Experience Replay.

    Parameters
    ----------
    size: int
        Maximum number of experiences to store.
    n_partitions: int
        Number of partitions to use for the sampling approximation.
    priority_func: callable
        Maps from an RLContext object to a signal to use as the priorities for the replay buffer.
    alpha: float > 0
        Degree of prioritization (similar to a softmax temperature); 0 corresponds to no prioritization
        (uniform distribution), inf corresponds to degenerate distribution which always picks element
        with highest priority.
    beta_schedule: 1 > float > 0
        Degree of importance sampling correction, 0 corresponds to no correction, 1 corresponds to
        full correction. Usually anneal linearly from an initial value beta_0 to a value of 1 by the
        end of learning.
    min_experiences: int > 0
        Minimum number of experiences that must be stored in the replay buffer before it will return
        a valid batch when `get_batch` is called. Before this point, it returns None, indicating that
        whatever is making use of this replay memory should not make an update.

    """
    def __init__(self, size, n_partitions, priority_func, alpha, beta_schedule, min_experiences=None, name=None):
        self.size = size
        self.n_partitions = n_partitions
        self.priority_func = priority_func
        self.alpha = alpha
        self.beta_schedule = beta_schedule
        self.min_experiences = min_experiences

        self.index = 0

        self._experiences = {}

        # Note this is actually a MIN priority queue, so to make it act like a MAX priority
        # queue, we use the negative of the provided priorities.
        self.skip_list = SkipList()
        self.distributions = self.build_distribution()

        self._active_set = None

        super(PrioritizedReplayBuffer, self).__init__(name)

    def build_core_signals(self, context):
        self.beta = context.get_signal("beta", self)
        self.priority_signal = tf.reshape(self.priority_func(context), [-1])

    def generate_signal(self, signal_key, context):
        if signal_key == "beta":
            return build_scheduled_value(self.beta_schedule, '{}-beta'.format(self.name))
        else:
            raise Exception("NotImplemented")

    def post_update(self, feed_dict, context):
        if self._active_set is not None:
            priority = tf.get_default_session().run(self.priority_signal, feed_dict=feed_dict)
            self.update_priority(priority)

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
        strata_starts, strata_ends, strata_sizes = [], [], []
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
            strata_sizes.append(strata_ends[-1] - strata_starts[-1])
            start_idx = end_idx

        self.strata_starts = strata_starts
        self.strata_ends = strata_ends
        self.strata_sizes = strata_sizes
        self.pdf = pdf

    def add_rollouts(self, rollouts):
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

        self._active_set = None

    def get_batch(self, batch_size):
        no_sample = (
            (self.min_experiences is not None and
             self.n_experiences < self.min_experiences) or
            self.n_experiences < batch_size)
        if no_sample:
            return None, None

        priority_indices = []
        start = 0
        permutation = np.random.permutation(self.n_partitions)

        selected_sizes = []
        for i in islice(cycle(permutation), batch_size):
            start, end = self.strata_starts[i], self.strata_ends[i]
            priority_indices.append(np.random.randint(start, end))
            selected_sizes.append(self.strata_sizes[i])

        # We set p_x to be the actual probability that we sampled with,
        # namely 1 / (n_partitions * size_of_partition), rather than
        # the pdf that our sampling method approximates, namely `self.pdf`.
        # This is both more faithful, and works better when the memory is not full.
        p_x = (self.n_partitions * np.array(selected_sizes))**-1.

        beta = tf.get_default_session().run(self.beta)

        weights = (p_x * self.size)**-beta
        weights /= weights.max()

        # When we aren't full, map priority_indices (which are in range(self.size)) down to range(self.n_experiences)
        if self.n_experiences < self.size:
            priority_indices = [int(np.floor(self.n_experiences * (i / self.size))) for i in priority_indices]

        self._active_set = []
        for p_idx in priority_indices:
            priority, e_idx = self.skip_list[p_idx]
            self._active_set.append((p_idx, e_idx, priority))

        experiences = RolloutBatch.join([self._experiences[e_idx] for _, e_idx, _ in self._active_set])

        return experiences, weights
