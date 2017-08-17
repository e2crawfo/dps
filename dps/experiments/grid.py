import tensorflow as tf
import numpy as np

from dps.register import RegisterBank
from dps.environment import TensorFlowEnv
from dps.utils import Param


class Grid(TensorFlowEnv):
    action_names = '^ > v <'.split()

    rb = RegisterBank(
        'GridRB', 'x y r dx dy', 'goal_x goal_y',
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'x y')

    T = Param()
    shape = Param()
    restart_prob = Param()
    dense_reward = Param()
    l2l = Param()
    n_val = Param()

    def __init__(self, **kwargs):
        self.val = self._make_input(self.n_val)
        self.mode = 'train'

        if self.l2l and not self.dense_reward:
            raise Exception("When learning to learn, reward must be dense!")
        super(Grid, self).__init__()

    @property
    def completion(self):
        return 0.0

    def _make_input(self, batch_size):
        start_x = np.random.randint(self.shape[0], size=(batch_size, 1))
        start_y = np.random.randint(self.shape[1], size=(batch_size, 1))

        if self.l2l:
            goal_x = np.random.randint(self.shape[0], size=(batch_size, 1))
            goal_y = np.random.randint(self.shape[1], size=(batch_size, 1))
        else:
            goal_x = np.zeros((batch_size, 1))
            goal_y = np.zeros((batch_size, 1))

        return np.concatenate([start_x, start_y, goal_x, goal_y], axis=1).astype('f')

    def start_episode(self, batch_size):
        if self.mode in 'train train_eval'.split():
            self.input = self._make_input(batch_size)
        elif self.mode == 'val':
            self.input = self.val
        else:
            raise Exception("Unknown mode: {}.".format(self.mode))
        return self.input.shape[0], {self.input_ph: self.input}

    def build_init(self, r):
        batch_size = tf.shape(r)[0]
        self.input_ph = tf.placeholder(tf.float32, (None, 4))
        return self.rb.wrap(
            x=self.input_ph[:, 0:1], y=self.input_ph[:, 1:2],
            goal_x=self.input_ph[:, 2:3], goal_y=self.input_ph[:, 3:4],
            dx=tf.fill((batch_size, 1), 0.0),
            dy=tf.fill((batch_size, 1), 0.0),
            r=tf.fill((batch_size, 1), 0.0))

    def build_step(self, t, r, a):
        x, y, _, _, _, goal_x, goal_y = self.rb.as_tuple(r)
        up, right, down, left = tf.split(a, 4, axis=1)

        new_x = (1 - right - left) * x + right * (x+1) + left * (x-1)
        new_y = (1 - down - up) * y + down * (y+1) + up * (y-1)

        new_x = tf.clip_by_value(new_x, 0.0, self.shape[0]-1)
        new_y = tf.clip_by_value(new_y, 0.0, self.shape[1]-1)

        if self.restart_prob > 0:
            restart = tf.contrib.distributions.Bernoulli(self.restart_prob).sample(tf.shape(x))

            start_x = tf.random_uniform(tf.shape(x), maxval=self.shape[0], dtype=tf.int32)
            start_x = tf.cast(start_x, tf.float32)

            start_y = tf.random_uniform(tf.shape(y), maxval=self.shape[1], dtype=tf.int32)
            start_y = tf.cast(start_y, tf.float32)

            new_x = tf.where(tf.equal(restart, 1), start_x, new_x)
            new_y = tf.where(tf.equal(restart, 1), start_y, new_y)

        if self.dense_reward:
            reward = -tf.cast(tf.sqrt((new_x-goal_x)**2 + (new_y-goal_y)**2) > 0.1, tf.float32)
        else:
            reward = tf.cond(
                tf.equal(t[0, 0], tf.constant(self.T-1)),
                lambda: -tf.cast(tf.sqrt((new_x-goal_x)**2 + (new_y-goal_y)**2) > 0.1, tf.float32),
                lambda: tf.fill(tf.shape(x), 0.0))

        new_registers = self.rb.wrap(
            x=new_x, y=new_y, goal_x=goal_x, goal_y=goal_y,
            dx=new_x-x, dy=new_y-y, r=reward)

        return tf.fill((tf.shape(r)[0], 1), 0.0), reward, new_registers


def build_env():
    return Grid()
