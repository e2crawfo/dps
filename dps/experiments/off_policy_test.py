import tensorflow as tf
import numpy as np

from dps.register import RegisterBank
from dps.environment import TensorFlowEnv
from dps.utils import Param


class OffPolicyTest(TensorFlowEnv):
    action_names = '^ > v <'.split()

    rb = RegisterBank(
        'OffPolicyTestRB', 'x y', None,
        [0.0, 0.0], 'x y')

    T = Param()
    shape = Param()

    def __init__(self, **kwargs):
        self.mode = 'train'
        super(OffPolicyTest, self).__init__()

    @property
    def completion(self):
        return 0.0

    def _make_input(self, batch_size):
        return np.zeros((batch_size, 2))

    def start_episode(self, batch_size):
        self.input = self._make_input(batch_size)
        return self.input.shape[0], {self.input_ph: self.input}

    def build_init(self, r):
        self.input_ph = tf.placeholder(tf.float32, (None, 2))
        return self.rb.wrap(x=self.input_ph[:, 0:1], y=self.input_ph[:, 1:2])

    def build_step(self, t, r, a):
        x, y = self.rb.as_tuple(r)
        up, right, down, left = tf.split(a, 4, axis=1)

        new_x = (1 - right - left) * x + right * (x+1) + left * (x-1)
        new_y = (1 - down - up) * y + down * (y+1) + up * (y-1)

        new_y = tf.clip_by_value(new_y, 0.0, self.shape[0]-1)
        new_x = tf.clip_by_value(new_x, 0.0, self.shape[1]-1)

        reward = -tf.cast(tf.sqrt((new_x-(self.shape[1]-1))**2 + (new_y-(self.shape[0]-1))**2) > 0.1, tf.float32)

        new_registers = self.rb.wrap(x=new_x, y=new_y)

        return tf.fill((tf.shape(r)[0], 1), 0.0), reward, new_registers


def build_env():
    return OffPolicyTest()
