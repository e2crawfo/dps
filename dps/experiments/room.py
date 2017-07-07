import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import TensorFlowEnv, BatchBox


class Room(TensorFlowEnv):
    action_names = ['delta_x', 'delta_y', 'stop']
    make_input_available = False
    static_inp_width = 4

    def __init__(self, T, reward_std, max_step, restart_prob, dense_reward, l2l, n_val):
        self.rb = RegisterBank('RoomRB', 'x y r dx dy', 'goal_x goal_y', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'x y')
        self.observation_space = BatchBox(low=-1.0, high=1.0, shape=(None, self.rb.visible_width))
        self.action_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, self.n_actions))

        self.T = T
        self.reward_std = reward_std
        self.max_step = max_step
        self.restart_prob = restart_prob
        self.dense_reward = dense_reward
        self.l2l = l2l
        self.val = self._make_static_input(n_val)
        self._kind = 'train'
        super(Room, self).__init__()

    @property
    def completion(self):
        return 0.0

    def static_inp_type_and_shape(self):
        return (tf.float32, (self.static_inp_width,))

    def _make_static_input(self, batch_size):
        if self.l2l:
            return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, self.static_inp_width))
        else:
            return np.concatenate(
                [np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 2)),
                 np.zeros((batch_size, 2))],
                axis=1)

    def make_static_input(self, batch_size):
        if self._kind == 'train':
            return self._make_static_input(batch_size)
        elif self._kind == 'val':
            return self.val
        else:
            raise Exception()

    def build_init(self, r, inp):
        batch_size = tf.shape(r)[0]
        return self.rb.wrap(
            x=inp[:, 0:1], y=inp[:, 1:2], goal_x=inp[:, 2:3], goal_y=inp[:, 3:4],
            dx=tf.fill((batch_size, 1), 0.0),
            dy=tf.fill((batch_size, 1), 0.0),
            r=tf.fill((batch_size, 1), 0.0))

    def process_actions(self, a):
        delta_x, delta_y = tf.split(a, 2, axis=1)
        return delta_x, delta_y

    def build_step(self, t, r, a, static_inp):
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

        reward = -tf.cast(tf.sqrt(new_x**2 + new_y**2) > self.reward_std, tf.float32)

        # if self.dense_reward:
        #     reward = -1+tf.exp(-0.5 * (new_x**2 + new_y**2) / (self.reward_std**2))
        # else:
        #     reward = tf.cond(
        #         tf.equal(t[0, 0], tf.constant(self.T-1)),
        #         lambda: -1+tf.exp(-0.5 * (new_x**2 + new_y**2) / (self.reward_std**2)),
        #         lambda: tf.fill(tf.shape(x), 0.0))
        new_registers = self.rb.wrap(
            x=new_x, y=new_y, goal_x=goal_x, goal_y=goal_y,
            dx=delta_x, dy=delta_y, r=reward)

        return reward, new_registers


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
    args = [cfg.T, cfg.reward_std, cfg.max_step, cfg.restart_prob, cfg.dense_reward, cfg.l2l, cfg.n_val]
    if cfg.room_angular:
        return RoomAngular(*args)
    else:
        return Room(*args)
