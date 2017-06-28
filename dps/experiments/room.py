import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import TensorFlowEnv, BatchBox


class RoomAngular(TensorFlowEnv):
    action_names = ['delta_x', 'delta_y', 'mag']
    make_input_available = False

    def __init__(self, T, dense_reward, n_val):
        self.rb = RegisterBank('RoomRB', 'x y', None, [0.0, 0.0], 'x y')
        self.observation_space = BatchBox(low=-1.0, high=1.0, shape=(None, 2))
        self.action_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, 3))

        self.T = T
        self.dense_reward = dense_reward
        self.val = np.random.uniform(low=-1.0, high=1.0, size=(n_val, 2))
        self._kind = 'train'
        super(RoomGrid, self).__init__()

    @property
    def completion(self):
        return 0.0

    def static_inp_type_and_shape(self):
        return (tf.float32, (2,))

    def make_static_input(self, batch_size):
        if self._kind == 'train':
            return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 2))
        elif self._kind == 'val':
            return self.val
        else:
            raise Exception()

    def build_init(self, r, inp):
        return self.rb.wrap(x=inp[:, 0:1], y=inp[:, 1:2])

    def build_step(self, t, r, a, static_inp):
        x, y = self.rb.as_tuple(r)

        delta_x, delta_y, mag = tf.split(a, self.n_actions, axis=1)
        norm = tf.sqrt(delta_x**2 + delta_y**2)
        norm = tf.where(norm > 1e-6, norm, tf.zeros_like(norm))
        delta_x = mag * delta_x / norm
        delta_y = mag * delta_y / norm

        new_x = tf.clip_by_value(x + delta_x, -1.0, 1.0)
        new_y = tf.clip_by_value(y + delta_y, -1.0, 1.0)

        new_registers = self.rb.wrap(x=new_x, y=new_y)

        if self.dense_reward:
            reward = -(new_x**2 + new_y**2)
        else:
            reward = tf.cond(
                tf.equal(t[0, 0], tf.constant(self.T-1)),
                lambda: -(new_x**2 + new_y**2),
                lambda: tf.fill(tf.shape(x), 0.0))

        return reward, new_registers


class RoomGrid(TensorFlowEnv):
    action_names = ['delta_x', 'delta_y']
    make_input_available = False

    def __init__(self, T, max_step, restart_prob, dense_reward, n_val):
        self.rb = RegisterBank('RoomRB', 'x y', None, [0.0, 0.0], 'x y')
        self.observation_space = BatchBox(low=-1.0, high=1.0, shape=(None, 2))
        self.action_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, 2))

        self.T = T
        self.max_step = max_step
        self.restart_prob = restart_prob
        self.dense_reward = dense_reward
        self.val = np.random.uniform(low=-1.0, high=1.0, size=(n_val, 2))
        self._kind = 'train'
        super(RoomGrid, self).__init__()

    @property
    def completion(self):
        return 0.0

    def static_inp_type_and_shape(self):
        return (tf.float32, (2,))

    def make_static_input(self, batch_size):
        if self._kind == 'train':
            return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 2))
        elif self._kind == 'val':
            return self.val
        else:
            raise Exception()

    def build_init(self, r, inp):
        return self.rb.wrap(x=inp[:, 0:1], y=inp[:, 1:2])

    def build_step(self, t, r, a, static_inp):
        x, y = self.rb.as_tuple(r)

        delta_x, delta_y = tf.split(a, self.n_actions, axis=1)

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

        new_registers = self.rb.wrap(x=new_x, y=new_y)

        if self.dense_reward:
            reward = -(new_x**2 + new_y**2)
        else:
            reward = tf.cond(
                tf.equal(t[0, 0], tf.constant(self.T-1)),
                lambda: -(new_x**2 + new_y**2),
                lambda: tf.fill(tf.shape(x), 0.0))

        return reward, new_registers


class RoomGridL2L(TensorFlowEnv):
    action_names = ['delta_x', 'delta_y']
    make_input_available = False

    def __init__(self, T, max_step, restart_prob, n_val):
        self.rb = RegisterBank('RoomL2LRB', 'x y r dx dy', 'goal_x goal_y', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'x y')
        self.observation_space = BatchBox(low=-1.0, high=1.0, shape=(None, 5))
        self.action_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, 2))

        self.T = T
        self.max_step = max_step
        self.restart_prob = restart_prob
        self.val = np.random.uniform(low=-1.0, high=1.0, size=(n_val, 4))
        self._kind = 'train'
        super(RoomGridL2L, self).__init__()

    @property
    def completion(self):
        return 0.0

    def static_inp_type_and_shape(self):
        return (tf.float32, (4,))

    def make_static_input(self, batch_size):
        if self._kind == 'train':
            return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 4))
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

    def build_step(self, t, r, a, static_inp):
        x, y, _, _, _, goal_x, goal_y = self.rb.as_tuple(r)

        delta_x, delta_y = tf.split(a, self.n_actions, axis=1)

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

        reward = -((new_x-goal_x)**2 + (new_y-goal_y)**2)
        new_registers = self.rb.wrap(
            x=new_x, y=new_y, goal_x=goal_x, goal_y=goal_y,
            dx=delta_x, dy=delta_y, r=reward)

        return reward, new_registers


def build_env():
    if cfg.l2l:
        return RoomGridL2L(cfg.T, cfg.max_step, cfg.restart_prob, cfg.n_val)
    else:
        return RoomGrid(cfg.T, cfg.max_step, cfg.restart_prob, cfg.dense_reward, cfg.n_val)
