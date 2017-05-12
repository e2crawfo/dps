import abc
from future.utils import with_metaclass
import time

import numpy as np
import tensorflow as tf

import gym
from gym import Env as GymEnv
from gym.utils import seeding
from gym.spaces import prng


class BatchBox(gym.Space):
    """ A box that allows some dimensions to be unspecified at instance-creation time.

    Example usage:
    self.action_space = BatchBox(low=-10, high=10, shape=(None, 1))

    """
    def __init__(self, low, high, shape=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            shape = [1 if s is None else s for s in shape]
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)

    def sample(self):
        return prng.np_random.uniform(low=self.low, high=self.high, size=self.low.shape)

    def contains(self, x):
        return True
        # return (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    @property
    def shape(self):
        return self.low.shape

    def __repr__(self):
        return "<BatchBox {}>".format(self.shape)

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


class Env(with_metaclass(abc.ABCMeta, GymEnv)):
    def set_mode(self, kind, batch_size):
        # It would be preferable to have these be passed to ``reset``, but
        # the gym interface do not allow extra args to ``reset``.
        assert kind in ['train', 'val', 'test'], "Unknown kind {}.".format(kind)
        self._kind = kind
        self._batch_size = batch_size

    def do_rollouts(self, alg, policy, mode, n_rollouts=None):
        start_time = time.time()
        self.set_mode(mode, n_rollouts)
        obs = self.reset()
        batch_size = obs.shape[0]

        alg.start_episode()
        policy_state = policy.zero_state(batch_size, tf.float32)
        policy_state = tf.get_default_session().run(policy_state)

        done = False
        while not done:
            action, policy_state = policy.act(obs, policy_state, sample=True)
            new_obs, reward, done, info = self.step(action)

            alg.remember(obs, action, reward)
            obs = new_obs

        alg.end_episode()

        print("Took {} seconds to do {} rollouts.".format(time.time() - start_time, n_rollouts))


class DifferentiableEnv(with_metaclass(abc.ABCMeta, Env)):
    """ An environment which, when provided with a differentiable policy,
        has a loss that is a differentiable function of its input. """

    @abc.abstractmethod
    def build_loss(self, actions):
        """

        Parameters
        ----------
        actions: Tensor (batch_size, n_actions)

        Returns
        -------
        loss: Tensor(batch_size, 1)

        """
        raise NotImplementedError("Abstract method.")


class RegressionDataset(object):
    def __init__(self, x, y, for_eval=False, shuffle=True):
        self.x = x
        self.y = y
        self.for_eval = for_eval
        self.shuffle = shuffle

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def n_examples(self):
        return self.x.shape[0]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def completion(self):
        return self.epochs_completed + self.index_in_epoch / self.n_examples

    def next_batch(self, batch_size=None):
        """ Return the next ``batch_size`` examples from this data set.

        If ``batch_size`` not specified, return rest of the examples in the current epoch.

        """
        start = self._index_in_epoch

        if batch_size is None:
            batch_size = self.n_examples - start

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and self.shuffle:
            perm0 = np.arange(self.n_examples)
            np.random.shuffle(perm0)
            self._x = self.x[perm0]
            self._y = self.y[perm0]

        # Go to the next epoch
        if start + batch_size >= self.n_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_n_examples = self.n_examples - start
            x_rest_part = self._x[start:self.n_examples]
            y_rest_part = self._y[start:self.n_examples]

            if self.for_eval:
                self._index_in_epoch = 0
                return x_rest_part, y_rest_part
            else:
                # Shuffle the data
                if self.shuffle:
                    perm = np.arange(self.n_examples)
                    np.random.shuffle(perm)
                    self._x = self.x[perm]
                    self._y = self.y[perm]

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size - rest_n_examples
                end = self._index_in_epoch
                x_new_part = self._x[start:end]
                y_new_part = self._y[start:end]
                x = np.concatenate((x_rest_part, x_new_part), axis=0)
                y = np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            x, y = self._x[start:end], self._y[start:end]

        return x, y


class RegressionEnv(DifferentiableEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, train, val, test):
        self.train, self.val, self.test = train, val, test

        self.n_actions = self.train.y.shape[1]
        self.action_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, self.n_actions))

        self.obs_dim = self.train.x.shape[1]
        self.observation_space = BatchBox(
            low=-np.inf, high=np.inf, shape=(None, self.obs_dim))

        self.reward_range = (-np.inf, 0)

        self._kind = 'train'
        self._batch_size = None

        self.action_ph, self.loss, self.target_ph = None, None, None

        self.reset()

    def __str__(self):
        return "<RegressionEnv train={} val={} test={}>".format(self.train, self.val, self.test)

    @property
    def completion(self):
        return self.train.completion

    def build_loss(self, actions):
        target_ph = tf.placeholder(tf.float32, shape=actions.shape, name='target')
        loss = tf.reduce_mean((actions - target_ph)**2, axis=-1, keep_dims=True)
        return loss, target_ph

    def build_rl_loss(self, actions):
        """ A separate loss used in the RL setting where things are not required to be differentiable. """
        target_ph = tf.placeholder(tf.float32, shape=actions.shape, name='target')
        error = tf.reduce_sum(tf.abs(actions - target_ph), axis=-1, keep_dims=True)
        loss = tf.cast(error > 0.1, tf.float32)
        return loss, target_ph

    def _step(self, action):
        assert self.action_space.contains(action), (
            "{} ({}) is not a valid action for env {}.".format(action, type(action), self))
        self.t += 1

        assert self.y.shape == action.shape
        obs = np.zeros(self.x.shape)

        if self.action_ph is None:
            self.action_ph = tf.placeholder(tf.float32, (None, self.n_actions))
            build_loss = getattr(self, 'build_rl_loss', self.build_loss)
            self.loss, self.target_ph = build_loss(self.action_ph)

        sess = tf.get_default_session()
        reward = -sess.run(self.loss, {self.action_ph: action, self.target_ph: self.y})

        done = True
        info = {"y": self.y}
        return obs, reward, done, info

    def _reset(self):
        self.t = 0

        dataset = getattr(self, self._kind)
        self.x, self.y = dataset.next_batch(self._batch_size)
        return self.x

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
