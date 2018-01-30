""" Datasets and environments for supervised learning. """

import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp
from gym.utils import seeding

from dps.utils import one_hot
from dps.env import Env


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
        self.action_shape = self.train.y.shape[1:]

        self._mode = 'train'
        self._batch_size = None
        self.t = 0

    def __str__(self):
        return "<{} - train={}, val={}, test={}>".format(
            self.__class__.__name__, self.train, self.val, self.test)

    def next_batch(self, batch_size, mode):
        advance = mode == 'train'
        return self.datasets[mode].next_batch(batch_size=batch_size, advance=advance)

    def build_loss(self, actions, targets):
        raise Exception("NotImplemented")

    def get_reward(self, actions, targets):
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
        advance = self._mode == 'train'
        self.x, self.y = self.datasets[self._mode].next_batch(self._batch_size, advance=advance)
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
            targets = tf.one_hot(
                tf.squeeze(tf.cast(targets, tf.int32), axis=-1),
                depth=tf.shape(actions)[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=targets, logits=actions)[..., None]

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


class BernoulliSigmoid(SupervisedEnv):
    """ Assumes that `targets` is in th range [0, 1]. """
    reward_range = (-np.inf, 0)
    recorded_names = ["xent_loss", "2norm_loss", "1norm_loss"]

    def build_loss(self, logits, targets):
        return self.build_xent_loss(logits, targets)

    def build_xent_loss(self, logits, targets):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits),
            keep_dims=True, axis=-1
        )

    def build_2norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)
        return tf.reduce_mean((actions - targets)**2, keep_dims=True)

    def build_1norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)
        return tf.reduce_mean(tf.abs(actions - targets), keep_dims=True)

    def get_reward(self, logits, targets):
        return -self.get_xent_loss(logits, targets)

    def get_xent_loss(self, logits, targets):
        loss = (
            np.max(logits, 0) -
            logits * targets +
            np.log(1 + np.exp(-np.abs(logits)))
        )
        return np.mean(loss, axis=-1, keepdims=True)

    def get_2norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)
        return np.mean((actions - targets)**2, axis=-1, keepdims=True)

    def get_1norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)
        return np.mean(np.abs(actions - targets), axis=-1, keepdims=True)


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
