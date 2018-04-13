import tensorflow as tf
import numpy as np

from dps.register import RegisterBank
from dps.env import TensorFlowEnv
from dps.utils import Param, Config


def build_env():
    return GridBandit()


config = Config(
    build_env=build_env,
    curriculum=[dict()],
    log_name='grid_bandit',
    threshold=-5,
    T=5,
    shape=(2, 2),
    n_arms=10,
)


class GridBandit(TensorFlowEnv):
    """
    Agent starts off in random location. Agent can move around the grid, and can perform a `look`
    action to reveal an integer stored at its current location in the grid. It can also pull a number
    of arms determined by `n_arms` (these arms can be pulled anywhere, they have no spatial location).
    Also, one arm is pulled at all times; taking an action to pull one of the arms persistently
    changes the arm the agent is taken to be pulling. The integer stored in the top left location gives
    the identity of the correct arm. The optimal strategy for an episode is to move to the top-left corner,
    perform the `look` action, and then pull the correct arm thereafter.

    The agent receives a reward of 1 for every step that it pulls the correct arm, and 0 otherwise.

    """
    T = Param()
    shape = Param()
    n_val = Param()
    n_arms = Param(2)

    def __init__(self, **kwargs):
        self.action_names = '^ > v < look'.split() + ["arm_{}".format(i) for i in range(self.n_arms)]
        self.action_shape = (len(self.action_names),)
        self.rb = RegisterBank(
            'GridBanditRB', 'x y vision action arm', None, [0.0, 0.0, -1.0, 0.0, 0.0], 'x y',
        )
        self.val_input = self._make_input(self.n_val)
        self.test_input = self._make_input(self.n_val)

        super(GridBandit, self).__init__()

    def _make_input(self, batch_size):
        start_x = np.random.randint(self.shape[0], size=(batch_size, 1))
        start_y = np.random.randint(self.shape[1], size=(batch_size, 1))
        grid = np.random.randint(self.n_arms, size=(batch_size, np.product(self.shape)))
        return np.concatenate([start_x, start_y, grid], axis=1).astype('f')

    def _build_placeholders(self):
        self.input = tf.placeholder(tf.float32, (None, 2+np.product(self.shape)), name="input")

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
        return self.rb.wrap(
            x=self.input[:, 0:1], y=self.input[:, 1:2],
            vision=r[:, 2:3], action=r[:, 3:4], arm=r[:, 4:5])

    def build_step(self, t, r, actions):
        x, y, vision, action, current_arm = self.rb.as_tuple(r)
        up, right, down, left, look, *arms = tf.split(actions, 5+self.n_arms, axis=1)

        new_y = (1 - down - up) * y + down * (y+1) + up * (y-1)
        new_x = (1 - right - left) * x + right * (x+1) + left * (x-1)

        new_y = tf.clip_by_value(new_y, 0.0, self.shape[0]-1)
        new_x = tf.clip_by_value(new_x, 0.0, self.shape[1]-1)

        idx = tf.cast(y * self.shape[1] + x, tf.int32)
        new_vision = tf.reduce_sum(
            tf.one_hot(tf.reshape(idx, (-1,)), np.product(self.shape)) * self.input[:, 2:],
            axis=1, keepdims=True)
        vision = (1 - look) * vision + look * new_vision
        action = tf.cast(tf.reshape(tf.argmax(actions, axis=1), (-1, 1)), tf.float32)

        arm_chosen = tf.reduce_sum(tf.concat(arms, axis=1), axis=1, keepdims=True) > 0.5
        chosen_arm = tf.reshape(tf.argmax(arms), (-1, 1))
        current_arm = tf.cast(current_arm, tf.int64)
        new_current_arm = tf.where(arm_chosen, chosen_arm, current_arm)

        new_registers = self.rb.wrap(
            x=new_x, y=new_y, vision=vision, action=action, arm=tf.cast(new_current_arm, tf.float32))

        correct_arm = tf.equal(new_current_arm, tf.cast(self.input[:, 2:3], tf.int64))

        reward = tf.cast(correct_arm, tf.float32)

        return tf.fill((tf.shape(r)[0], 1), 0.0), reward, new_registers
