import tensorflow as tf
import numpy as np

from dps import cfg
from dps.envs import TensorFlowEnv
from dps.register import RegisterBank
from dps.utils import Param, Config
from dps.rl.policy import ProductDist, Normal, Gamma, Policy


def build_env():
    if cfg.room_angular:
        return RoomAngular()
    else:
        return Room()


def build_policy(env, **kwargs):
    if cfg.room_angular:
        action_selection = ProductDist(Normal(), Normal(), Gamma())
    else:
        action_selection = ProductDist(Normal(), Normal())
    return Policy(action_selection, env.obs_shape, **kwargs)


config = Config(
    build_env=build_env,
    n_controller_units=128,
    build_policy=build_policy,
    log_name='room',
    T=20,
    restart_prob=0.0,
    max_step=0.3,
    room_angular=False,
    l2l=False,
    reward_radius=0.5,
    n_val=100,
)


class Room(TensorFlowEnv):
    action_names = ['delta_x', 'delta_y']

    T = Param()
    reward_radius = Param()
    max_step = Param()
    restart_prob = Param()
    l2l = Param()
    n_val = Param()

    def __init__(self, **kwargs):
        self.val_input = self._make_input(self.n_val)
        self.test_input = self._make_input(self.n_val)

        self.rb = RegisterBank(
            'RoomRB', 'x y r dx dy', 'goal_x goal_y', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'x y'
        )

        super(Room, self).__init__()

    def _make_input(self, batch_size):
        if self.l2l:
            return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 4))
        else:
            return np.concatenate(
                [np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 2)),
                 np.zeros((batch_size, 2))], axis=1)

    def _build_placeholders(self):
        self.input = tf.placeholder(tf.float32, (None, 4))

    def _make_feed_dict(self, n_rollouts, T, mode):
        if mode == 'train':
            inp = self._make_input(n_rollouts)
        elif mode == 'val':
            inp = self.val_input
        elif mode == 'test':
            inp = self.test_input
        else:
            raise Exception("Unknown mode: {}.".format(mode))

        if n_rollouts is not None:
            inp = inp[:n_rollouts, :]

        return {self.input: inp}

    def build_init(self, r):
        batch_size = tf.shape(r)[0]
        return self.rb.wrap(
            x=self.input[:, 0:1], y=self.input[:, 1:2],
            goal_x=self.input[:, 2:3], goal_y=self.input[:, 3:4],
            dx=tf.fill((batch_size, 1), 0.0),
            dy=tf.fill((batch_size, 1), 0.0),
            r=tf.fill((batch_size, 1), 0.0))

    def _process_actions(self, a):
        delta_x, delta_y = tf.split(a, 2, axis=1)
        return delta_x, delta_y

    def build_step(self, t, r, a):
        x, y, _, _, _, goal_x, goal_y = self.rb.as_tuple(r)
        delta_x, delta_y = self._process_actions(a)

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

        reward = -tf.cast(tf.sqrt((new_x-goal_x)**2 + (new_y-goal_y)**2) > self.reward_radius, tf.float32)

        new_registers = self.rb.wrap(
            x=new_x, y=new_y, goal_x=goal_x, goal_y=goal_y,
            dx=delta_x, dy=delta_y, r=reward)

        return tf.fill((tf.shape(r)[0], 1), 0.0), reward, new_registers


class RoomAngular(Room):
    action_names = ['delta_x', 'delta_y', 'mag']

    def _process_actions(self, a):
        delta_x, delta_y, mag = tf.split(a, 3, axis=1)
        norm = tf.sqrt(delta_x**2 + delta_y**2)
        norm = tf.where(norm > 1e-6, norm, tf.zeros_like(norm))
        delta_x = mag * delta_x / norm
        delta_y = mag * delta_y / norm
        return delta_x, delta_y
