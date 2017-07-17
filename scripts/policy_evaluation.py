import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.experiments.room import Room
from dps.rl import RLUpdater
from dps.rl.value import PolicyEvaluation, NeuralValueEstimator
from dps.rl.policy import Policy, Deterministic
from dps.utils import FeedforwardCell, MLP


def build_env():
    return Room(cfg.T, cfg.reward_radius, cfg.max_step, cfg.restart_prob, cfg.dense_reward, cfg.l2l, cfg.n_val)


class GoToPoint(RNNCell):
    def __init__(self, point=None):
        if point is None:
            point = (0, 0)
        self.point = np.array(point).reshape(1, -1)

    def __call__(self, inp, state, scope=None):
        with tf.name_scope(scope or 'go_to_point'):
            batch_size = tf.shape(inp)[0]
            return (self.point - inp[:, :2]), tf.fill((batch_size, 1), 0.0)

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 2

    def zero_state(self, batch_size, dtype):
        return tf.fill((batch_size, 1), 0.0)


def get_updater(env):
    policy = Policy(GoToPoint(), Deterministic(2), env.obs_shape)
    controller = FeedforwardCell(lambda inp, output_size: MLP([128, 128])(inp, output_size), 1)
    # controller = FeedforwardCell(lambda inp, output_size: fully_connected(inp, output_size, activation_fn=None), 1)
    estimator = NeuralValueEstimator(controller, env.obs_shape)
    updater = RLUpdater(env, policy, PolicyEvaluation(estimator))
    return updater


config = DEFAULT_CONFIG.copy(
    get_updater=get_updater,
    build_env=build_env,
    log_name="policy_evaluation",
    max_steps=100000,

    display_step=100,

    T=4,
    reward_radius=0.2,
    max_step=0.1,
    restart_prob=0.0,
    dense_reward=False,
    l2l=False,
    n_val=200,

    optimizer_spec='rmsprop',
    lr_schedule='1e-5',
    threshold=1e-4
)

with config:
    cl_args = clify.wrap_object(cfg).parse()
    config.update(cl_args)

    val = training_loop()
