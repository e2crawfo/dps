
import tensorflow as tf
import numpy as np

from dps.register import RegisterBank
from dps.environment import TensorFlowEnv
from dps.utils import Param, Config


def build_env():
    return PathDiscovery()


config = Config(
    build_env=build_env,
    curriculum=[
        dict(shape=(2, 2), threshold=-6),
        dict(shape=(3, 3), threshold=-4),
        dict(shape=(4, 4), threshold=-2)
    ],
    n_controller_units=32,
    log_name='path_discovery',
    shape=(3, 3),
    T=10,
    threshold=-10,
)


class PathDiscovery(TensorFlowEnv):
    T = Param()
    shape = Param()
    n_val = Param()
    require_discovery = Param(True)

    def __init__(self, **kwargs):
        self.action_names = '^ > v < look'.split()
        self.actions_dim = len(self.action_names)
        self.rb = RegisterBank('PathDiscoveryRB', 'x y vision action', 'discovered', [0.0, 0.0, -1.0, 0.0, 0.0], 'x y')
        self.val = self._make_input(self.n_val)
        self.mode = 'train'

        super(PathDiscovery, self).__init__()

    @property
    def completion(self):
        return 0.0

    def _make_input(self, batch_size):
        start_x = np.random.randint(self.shape[0], size=(batch_size, 1))
        start_y = np.random.randint(self.shape[1], size=(batch_size, 1))
        grid = np.random.randint(3, size=(batch_size, np.product(self.shape)))
        return np.concatenate([start_x, start_y, grid], axis=1).astype('f')

    def start_episode(self, batch_size):
        if self.mode in 'train train_eval'.split():
            self.input = self._make_input(batch_size)
        elif self.mode == 'val':
            self.input = self.val
        else:
            raise Exception("Unknown mode: {}.".format(self.mode))
        return self.input.shape[0], {self.input_ph: self.input}

    def build_init(self, r):
        self.input_ph = tf.placeholder(tf.float32, (None, 2+np.product(self.shape)))
        return self.rb.wrap(
            x=self.input_ph[:, 0:1], y=self.input_ph[:, 1:2],
            vision=r[:, 2:3], action=r[:, 3:4], discovered=r[:, 4:5])

    def build_step(self, t, r, actions):
        x, y, vision, action, discovered = self.rb.as_tuple(r)
        up, right, down, left, look = tf.split(actions, 5, axis=1)

        new_y = (1 - down - up) * y + down * (y+1) + up * (y-1)
        new_x = (1 - right - left) * x + right * (x+1) + left * (x-1)

        new_y = tf.clip_by_value(new_y, 0.0, self.shape[0]-1)
        new_x = tf.clip_by_value(new_x, 0.0, self.shape[1]-1)

        idx = tf.cast(y * self.shape[1] + x, tf.int32)
        new_vision = tf.reduce_sum(
            tf.one_hot(tf.reshape(idx, (-1,)), np.product(self.shape)) * self.input_ph[:, 2:],
            axis=1, keep_dims=True)
        vision = (1 - look) * vision + look * new_vision
        action = tf.cast(tf.reshape(tf.argmax(actions, axis=1), (-1, 1)), tf.float32)

        top_left = tf.cast(tf.equal(idx, 0), tf.float32)

        discovered = discovered + look * top_left
        discovered = tf.minimum(discovered, 1.0)

        new_registers = self.rb.wrap(x=new_x, y=new_y, vision=vision, action=action, discovered=discovered)

        top_right = tf.cast(tf.equal(idx, self.shape[1]-1), tf.float32)
        bottom_left = tf.cast(tf.equal(idx, (self.shape[0]-1) * self.shape[1]), tf.float32)
        bottom_right = tf.cast(tf.equal(idx, self.shape[0] * self.shape[1] - 1), tf.float32)

        reward = (
            top_right * tf.cast(tf.equal(self.input_ph[:, 2:3], 0), tf.float32) +
            bottom_left * tf.cast(tf.equal(self.input_ph[:, 2:3], 1), tf.float32) +
            bottom_right * tf.cast(tf.equal(self.input_ph[:, 2:3], 2), tf.float32)
        )

        if self.require_discovery:
            reward = reward * discovered

        return tf.fill((tf.shape(r)[0], 1), 0.0), reward, new_registers
