""" Datasets and environments for supervised learning. """

import numpy as np
import tensorflow as tf
from gym.utils import seeding

from dps import cfg
from dps.env import Env
from dps.updater import DataManager


class SupervisedEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, train, val, **kwargs):
        self.train, self.val = train, val
        self.datasets = dict(
            train=self.train,
            val=self.val,
        )

        if getattr(self, 'obs_shape', None) is None:
            self.obs_shape = self.train.x[0].shape

        if getattr(self, 'action_shape', None) is None:
            self.action_shape = self.train.y[0].shape

        self._mode = 'train'
        self._batch_size = None
        self.t = 0

    def __str__(self):
        return "<{} - train={}, val={}>".format(
            self.__class__.__name__, self.train, self.val)

    @property
    def recorded_names(self):
        """ Returns a list of names to be recorded. First one is used as the main loss.

        For each entry `s` in this list, a method called `build_s` is called to get the
        tensor value. Finally, the mean of the tensor is taken to get the final value
        that gets recorded.

        """
        raise Exception("NotImplemented")

    def _build(self):
        """ Should be over-ridden in sub-classes. Must return a dictionary of tensors to record.
            At least one of those tensors must have the key `loss`.

        """
        recorded_tensors = {
            name: tf.reduce_mean(getattr(self, 'build_' + name)(self.prediction, self.target))
            for name in self.recorded_names
        }

        assert recorded_tensors
        recorded_tensors['loss'] = recorded_tensors[self.recorded_names[0]]

        return recorded_tensors

    def build(self, f):
        self.data_manager = DataManager(self.datasets['train'],
                                        self.datasets['val'],
                                        cfg.batch_size)
        self.data_manager.build_graph()

        self._build_placeholders()
        self.f = f

        self.x, self.y = self.data_manager.iterator.get_next()
        self.target = self.y
        self.is_training = self.data_manager.is_training

        self.prediction = self.f(self.x, self.action_shape, self.is_training)

        recorded_tensors = self._build()

        return recorded_tensors

    def get_reward(self, actions, targets):
        raise Exception("NotImplemented")

    @property
    def completion(self):
        return self.train.completion

    def _step(self, action):
        self.t += 1

        obs = np.zeros(self.rl_x.shape)

        reward = self.get_reward(action, self.rl_target)

        done = True
        info = {"y": self.rl_target}
        for name in self.recorded_names:
            func = getattr(self, "get_{}".format(name))
            info[name] = np.mean(func(action, self.rl_target))

        return obs, reward, done, info

    def _reset(self):
        self.t = 0
        advance = self._mode == 'train'
        self.rl_x, self.rl_target = self.datasets[self._mode].next_batch(self._batch_size, advance=advance)
        self.rl_y = self.rl_target
        return self.rl_x

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
            axis=-1, keepdims=True)

    def get_reward(self, actions, targets):
        return -self.get_xent_loss(actions, targets)

    def get_xent_loss(self, logits, targets):
        """ Assumes `targets` is one-hot. """
        raise Exception("NotImplemented. logsumexp dep no longer satisfied since scipy was removed.")

        # if not self.one_hot:
        #     targets = one_hot(np.squeeze(targets, axis=-1), logits.shape[-1])
        # log_numer = np.sum(logits * targets, axis=-1, keepdims=True)
        # log_denom = logsumexp(logits, axis=-1, keepdims=True)
        # return -(log_numer - log_denom)

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

    def build_2norm_loss(self, actions, targets):
        return tf.reduce_mean((actions - targets)**2, keepdims=True)

    def get_reward(self, actions, targets):
        return -self.get_2norm_loss(actions, targets)

    def get_2norm_loss(self, actions, targets):
        return np.mean((actions - targets)**2, axis=-1, keepdims=True)


class BernoulliSigmoid(SupervisedEnv):
    """ Assumes that `targets` is in the range [0, 1]. """
    reward_range = (-np.inf, 0)
    recorded_names = ["xent_loss", "2norm_loss", "1norm_loss"]

    def build_xent_loss(self, logits, targets):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits),
            keepdims=True, axis=-1
        )

    def build_2norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)
        return tf.reduce_mean((actions - targets)**2, axis=-1, keepdims=True)

    def build_1norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)
        return tf.reduce_mean(tf.abs(actions - targets), axis=-1, keepdims=True)

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
    recorded_names = ["01_loss", "2norm_loss"]

    def build_01_loss(self, actions, targets):
        return tf.reduce_mean(
            tf.to_float(tf.abs(actions - targets) >= 0.5),
            axis=-1, keepdims=True)

    def get_reward(self, actions, targets):
        return -self.get_01_loss(actions, targets)

    def get_01_loss(self, actions, targets):
        return np.mean(np.abs(actions - targets) >= 0.5, axis=-1, keepdims=True)
