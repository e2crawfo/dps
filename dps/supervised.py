import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp

from gym.utils import seeding
from dps.utils import Parameterized, Param, one_hot
from dps.environment import Env


class SupervisedDataset(Parameterized):
    n_examples = Param()

    def __init__(self, x, y, shuffle=True, **kwargs):
        self.x = x
        self.y = y
        self.n_examples = self.x.shape[0]
        self.shuffle = shuffle

        self._epochs_completed = 0
        self._index_in_epoch = 0

        super(SupervisedDataset, self).__init__(**kwargs)

    @property
    def obs_shape(self):
        return self.x.shape[1:]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def completion(self):
        return self.epochs_completed + self.index_in_epoch / self.n_examples

    def next_batch(self, batch_size=None, advance=True):
        """ Return the next ``batch_size`` examples from this data set.

        If ``batch_size`` not specified, return rest of the examples in the current epoch.

        """
        start = self._index_in_epoch

        if batch_size is None:
            batch_size = self.n_examples - start
        elif batch_size > self.n_examples:
            raise Exception("Too few examples ({}) to satisfy batch size of {}.".format(self.n_examples, batch_size))

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and self.shuffle:
            perm0 = np.arange(self.n_examples)
            np.random.shuffle(perm0)
            self._x = self.x[perm0]
            self._y = self.y[perm0]

        if start + batch_size >= self.n_examples:
            # Finished epoch

            # Get the remaining examples in this epoch
            x_rest_part = self._x[start:]
            y_rest_part = self._y[start:]

            # Shuffle the data
            if self.shuffle and advance:
                perm = np.arange(self.n_examples)
                np.random.shuffle(perm)
                self._x = self.x[perm]
                self._y = self.y[perm]

            # Start next epoch
            end = batch_size - len(x_rest_part)
            x_new_part = self._x[:end]
            y_new_part = self._y[:end]
            x = np.concatenate((x_rest_part, x_new_part), axis=0)
            y = np.concatenate((y_rest_part, y_new_part), axis=0)

            if advance:
                self._index_in_epoch = end
                self._epochs_completed += 1
        else:
            # Middle of epoch
            end = start + batch_size
            x, y = self._x[start:end], self._y[start:end]

            if advance:
                self._index_in_epoch = end

        return x, y


class SupervisedEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, train, val, test=None, **kwargs):
        self.train, self.val, self.test = train, val, test
        self.datasets = {
            'train': self.train,
            'val': self.val,
            'test': self.test,
        }

        self.obs_shape = self.train.x.shape[1:]
        self.actions_dim = self.train.y.shape[1]

        self.mode = 'train'
        self.batch_size = None
        self.t = 0

    def __str__(self):
        return "<{} - train={}, val={}, test={}>".format(
            self.__class__.__name__, self.train, self.val, self.test)

    def next_batch(self, batch_size, mode):
        advance = mode == 'train'
        return self.datasets[mode].next_batch(batch_size=batch_size, advance=advance)

    def build_loss(self, actions, targets):
        raise Exception("NotImplemented")

    @property
    def completion(self):
        return self.train.completion

    def _step(self, action):
        self.t += 1

        obs = np.zeros(self.x.shape)

        reward = self.get_reward(action, self.y)

        done = True
        info = {"y": self.y}
        for name in self.recorded_names:
            func = getattr(self, "get_{}".format(name))
            info[name] = np.mean(func(action, self.y))

        return obs, reward, done, info

    def _reset(self):
        self.t = 0
        advance = self.mode == 'train'
        self.x, self.y = self.datasets[self.mode].next_batch(self.batch_size, advance=advance)
        return self.x

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class ClassificationEnv(SupervisedEnv):
    reward_range = (-np.inf, 0)
    recorded_names = ["xent_loss", "01_loss"]

    def __init__(self, *args, one_hot=True, **kwargs):
        """
        Parameters
        ----------
        one_hot: bool
            Whether targets are presented as one-hot vectors or as integers.

        """
        self.one_hot = one_hot
        super(ClassificationEnv, self).__init__(*args, **kwargs)

    def build_loss(self, actions, targets):
        return self.build_xent_loss(actions, targets)

    def build_xent_loss(self, actions, targets):
        if not self.one_hot:
            targets = tf.one_hot(tf.squeeze(tf.cast(targets, tf.int32), axis=-1), depth=tf.shape(actions)[-1])
        return tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=actions)[..., None]

    def build_01_loss(self, actions, targets):
        action_argmax = tf.argmax(actions, axis=-1)
        if self.one_hot:
            targets_argmax = tf.argmax(targets, axis=-1)
        else:
            targets_argmax = tf.reshape(targets, tf.shape(action_argmax))
        return tf.reduce_mean(
            1 - tf.to_float(tf.equal(action_argmax, targets_argmax)),
            axis=-1, keep_dims=True)

    def get_reward(self, actions, targets):
        return -self.get_xent_loss(actions, targets)

    def get_xent_loss(self, logits, targets):
        """ Assumes `targets` is one-hot. """
        if not self.one_hot:
            targets = one_hot(np.squeeze(targets, axis=-1), logits.shape[-1])
        log_numer = np.sum(logits * targets, axis=-1, keepdims=True)
        log_denom = logsumexp(logits, axis=-1, keepdims=True)
        return -(log_numer - log_denom)

    def get_01_loss(self, actions, targets):
        action_argmax = np.argmax(actions, axis=-1)[..., None]
        if self.one_hot:
            targets_argmax = np.argmax(targets, axis=-1)
        else:
            targets_argmax = targets.reshape(action_argmax.shape)
        return 1 - (action_argmax == targets_argmax).astype('f')


class RegressionEnv(SupervisedEnv):
    reward_range = (-np.inf, 0)
    recorded_names = ["2norm_loss"]

    def build_loss(self, actions, targets):
        return self.build_2norm_loss(actions, targets)

    def build_2norm_loss(self, actions, targets):
        return tf.reduce_mean((actions - targets)**2, keep_dims=True)

    def get_reward(self, actions, targets):
        return -self.get_2norm_loss(actions, targets)

    def get_2norm_loss(self, actions, targets):
        return np.mean((actions - targets)**2, axis=-1, keepdims=True)


class IntegerRegressionEnv(RegressionEnv):
    reward_range = (-1, 0)
    recorded_names = ["2norm_loss", "01_loss"]

    def build_01_loss(self, actions, targets):
        return tf.reduce_mean(
            tf.to_float(tf.abs(actions - targets) >= 0.5),
            axis=-1, keep_dims=True)

    def get_reward(self, actions, targets):
        return -self.get_01_loss(actions, targets)

    def get_01_loss(self, actions, targets):
        return np.mean(np.abs(actions - targets) >= 0.5, axis=-1, keepdims=True)
