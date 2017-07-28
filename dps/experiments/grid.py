import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import TensorFlowEnv


class Grid(TensorFlowEnv):
    action_names = '^ > v <'.split()
    make_input_available = False
    static_inp_width = 4

    def __init__(self, T, shape, restart_prob, dense_reward, l2l, n_val):

        self.rb = RegisterBank(
            'GridRB', 'x y r dx dy', 'goal_x goal_y',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'x y')
        self.T = T
        self.shape = shape
        self.restart_prob = restart_prob
        self.dense_reward = dense_reward
        self.l2l = l2l
        self.val = self._make_static_input(n_val)
        self.mode = 'train'

        if self.l2l and not self.dense_reward:
            raise Exception("When learning to learn, reward must be dense!")
        super(Grid, self).__init__()

    @property
    def completion(self):
        return 0.0

    def static_inp_type_and_shape(self):
        return (tf.float32, (self.static_inp_width,))

    def _make_static_input(self, batch_size):
        start_x = np.random.randint(self.shape[0], size=(batch_size, 1))
        start_y = np.random.randint(self.shape[1], size=(batch_size, 1))

        if self.l2l:
            goal_x = np.random.randint(self.shape[0], size=(batch_size, 1))
            goal_y = np.random.randint(self.shape[1], size=(batch_size, 1))
        else:
            goal_x = np.zeros((batch_size, 1))
            goal_y = np.zeros((batch_size, 1))

        return np.concatenate([start_x, start_y, goal_x, goal_y], axis=1).astype('f')

    def make_static_input(self, batch_size):
        if self.mode in 'train train_eval'.split():
            return self._make_static_input(batch_size)
        elif self.mode == 'val':
            return self.val
        else:
            raise Exception("Unknown mode: {}.".format(self.mode))

    def build_init(self, r, inp):
        batch_size = tf.shape(r)[0]
        return self.rb.wrap(
            x=inp[:, 0:1], y=inp[:, 1:2], goal_x=inp[:, 2:3], goal_y=inp[:, 3:4],
            dx=tf.fill((batch_size, 1), 0.0),
            dy=tf.fill((batch_size, 1), 0.0),
            r=tf.fill((batch_size, 1), 0.0))

    def build_step(self, t, r, a, static_inp):
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
    args = [cfg.T, cfg.shape, cfg.restart_prob, cfg.dense_reward, cfg.l2l, cfg.n_val]
    return Grid(*args)
