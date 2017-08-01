import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import TensorFlowEnv


class Room(TensorFlowEnv):
    action_names = ['delta_x', 'delta_y']

    def __init__(
            self, T, reward_radius, max_step, restart_prob,
            dense_reward, l2l, n_val):

        self.rb = RegisterBank(
            'RoomRB', 'x y r dx dy', 'goal_x goal_y',
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'x y')
        self.T = T
        self.reward_radius = reward_radius
        self.max_step = max_step
        self.restart_prob = restart_prob
        self.dense_reward = dense_reward
        self.l2l = l2l
        self.val_input = self._make_input(n_val)
        self.mode = 'train'

        if self.l2l and not self.dense_reward:
            raise Exception("When learning to learn, reward must be dense!")
        super(Room, self).__init__()

    @property
    def completion(self):
        return 0.0

    def _make_input(self, batch_size):
        if self.l2l:
            return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 4))
        else:
            return np.concatenate(
                [np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 2)),
                 np.zeros((batch_size, 2))],
                axis=1)

    def start_episode(self, n_rollouts):
        if self.mode in 'train train_eval'.split():
            self.input = self._make_input(n_rollouts)
        elif self.mode == 'val':
            self.input = self.val_input
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

    def process_actions(self, a):
        delta_x, delta_y = tf.split(a, 2, axis=1)
        return delta_x, delta_y

    def build_step(self, t, r, a):
        x, y, _, _, _, goal_x, goal_y = self.rb.as_tuple(r)
        delta_x, delta_y = self.process_actions(a)

        if self.max_step > 0:
            delta_x = tf.clip_by_value(delta_x, -self.max_step, self.max_step)
            delta_y = tf.clip_by_value(delta_y, -self.max_step, self.max_step)

        new_x = tf.clip_by_value(x + delta_x, -1.0, 1.0)
        new_y = tf.clip_by_value(y + delta_y, -1.0, 1.0)

        if self.restart_prob > 0:
            restart = tf.contrib.distributions.Bernoulli(self.restart_prob).sample(tf.shape(x))
            new_x = tf.where(
                tf.equal(restart, 1),
                tf.contrib.distributions.Uniform(-1., 1.).sample(tf.shape(x)),
                new_x)
            new_y = tf.where(
                tf.equal(restart, 1),
                tf.contrib.distributions.Uniform(-1., 1.).sample(tf.shape(x)),
                new_y)

        if self.dense_reward:
            reward = -tf.cast(tf.sqrt((new_x-goal_x)**2 + (new_y-goal_y)**2) > self.reward_radius, tf.float32)
        else:
            reward = tf.cond(
                tf.equal(t[0, 0], tf.constant(self.T-1)),
                lambda: -tf.cast(tf.sqrt((new_x-goal_x)**2 + (new_y-goal_y)**2) > self.reward_radius, tf.float32),
                lambda: tf.fill(tf.shape(x), 0.0))

        new_registers = self.rb.wrap(
            x=new_x, y=new_y, goal_x=goal_x, goal_y=goal_y,
            dx=delta_x, dy=delta_y, r=reward)

        return tf.fill((tf.shape(r)[0], 1), 0.0), reward, new_registers


class RoomAngular(Room):
    action_names = ['delta_x', 'delta_y', 'mag']

    def process_actions(self, a):
        delta_x, delta_y, mag = tf.split(a, 3, axis=1)
        norm = tf.sqrt(delta_x**2 + delta_y**2)
        norm = tf.where(norm > 1e-6, norm, tf.zeros_like(norm))
        delta_x = mag * delta_x / norm
        delta_y = mag * delta_y / norm
        return delta_x, delta_y


def build_env():
    args = [cfg.T, cfg.reward_radius, cfg.max_step, cfg.restart_prob, cfg.dense_reward, cfg.l2l, cfg.n_val]
    if cfg.room_angular:
        return RoomAngular(*args)
    else:
        return Room(*args)
