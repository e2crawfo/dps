import tensorflow as tf
import numpy as np

from dps.register import RegisterBank
from dps.environment import TensorFlowEnv
from dps.utils import Param, Config


def build_cliff_walk():
    return CliffWalk()


config = Config(
    build_env=build_cliff_walk,
    curriculum=[dict()],
    log_name='cliff_walk',
    T=20,
    width=10,
    order=None,
    n_actions=2,
)

config.threshold = 0.01 - config.T / config.width


class CliffWalk(TensorFlowEnv):

    T = Param()
    n_actions = Param()
    width = Param()
    order = Param()

    def __init__(self, **kwargs):
        self.action_names = ["a_{}".format(i) for i in range(self.n_actions)]
        self.actions_dim = self.n_actions
        state_names = ["s{}".format(i) for i in range(self.width)]
        self.rb = RegisterBank('CliffWalkRB', state_names, None, [1.0] + [0.0]*(self.width-1), state_names)

        if self.order is None:
            self.order = np.random.randint(self.n_actions, size=self.width)

        super(CliffWalk, self).__init__()

    @property
    def completion(self):
        return 0.0

    def start_episode(self, batch_size):
        self.input = np.zeros((batch_size, 0))
        return batch_size, {self.input_ph: self.input}

    def build_init(self, r):
        self.input_ph = tf.placeholder(tf.float32, (None, 0))
        return r

    def build_step(self, t, r, a):
        current_state = tf.argmax(r, axis=-1)[:, None]
        correct_action = tf.gather(self.order, current_state)
        chosen_action = tf.argmax(a, axis=-1)[:, None]
        chose_correct_action = tf.equal(chosen_action, correct_action)
        new_state = tf.where(chose_correct_action, 1 + current_state, tf.zeros_like(current_state))
        new_state = tf.mod(new_state, self.width)
        reward = tf.cast(tf.logical_and(chose_correct_action, tf.equal(current_state, self.width-1)), tf.float32)
        new_state = tf.one_hot(new_state[:, 0], self.width)
        return tf.fill((tf.shape(r)[0], 1), 0.0), reward, new_state
