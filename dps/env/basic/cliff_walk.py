import tensorflow as tf
import numpy as np

from dps.register import RegisterBank
from dps.env import TensorFlowEnv
from dps.utils import Param, Config


def build_env():
    return CliffWalk()


config = Config(
    build_env=build_env,
    curriculum=[dict()],
    env_name='cliff_walk',
    T=20,
    n_states=10,
    order=None,
    n_actions=2,
    stopping_criteria="reward_per_ep,max",
    threshold=10.0,
)


class CliffWalk(TensorFlowEnv):
    """ An abstraction of the classic cliff walk domain. The agent needs to choose the exact correct sequence
        of actions. Any action move and it is sent back at the initial state, where it starts again. Choosing
        the correct action advances it to the next state. A reward is received for choosing the correct answer
        in the final state.

    """
    T = Param(help="Number of timesteps")
    n_actions = Param(help="Number of actions")
    n_states = Param(help="Number of states")
    order = Param(help="Optional list of integers specifying correct order of actions")

    def __init__(self, **kwargs):
        self.action_names = ["a_{}".format(i) for i in range(self.n_actions)]
        self.action_shape = (self.n_actions,)
        state_names = ["s{}".format(i) for i in range(self.n_states)]
        self.rb = RegisterBank('CliffWalkRB', state_names, None, [1.0] + [0.0]*(self.n_states-1), state_names)

        if self.order is None:
            self.order = np.random.randint(self.n_actions, size=self.n_states)

        super(CliffWalk, self).__init__()

    def _build_placeholders(self):
        pass

    def _make_feed_dict(self, *args, **kwargs):
        return {}

    def build_init(self, r):
        return r

    def build_step(self, t, r, a):
        current_state = tf.argmax(r, axis=-1)[:, None]
        correct_action = tf.gather(self.order, current_state)
        chosen_action = tf.argmax(a, axis=-1)[:, None]
        chose_correct_action = tf.equal(chosen_action, correct_action)
        new_state = tf.where(chose_correct_action, 1 + current_state, tf.zeros_like(current_state))
        new_state = tf.mod(new_state, self.n_states)
        reward = tf.cast(tf.logical_and(chose_correct_action, tf.equal(current_state, self.n_states-1)), tf.float32)
        new_state = tf.one_hot(new_state[:, 0], self.n_states)
        return tf.fill((tf.shape(r)[0], 1), 0.0), reward, new_state
