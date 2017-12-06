import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
import sys

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.train import training_loop
from dps.envs.room import Room
from dps.rl import RLUpdater
from dps.rl.value import (
    PolicyEvaluation, ProximalPolicyEvaluation, TrustRegionPolicyEvaluation,
    NeuralValueEstimator)
from dps.rl.policy import Policy, Deterministic
from dps.utils.tf import FeedforwardCell


def build_env():
    return Room()


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
    # controller = FeedforwardCell(lambda inp, output_size: MLP([128, 128])(inp, output_size), 1)
    controller = FeedforwardCell(lambda inp, output_size: fully_connected(inp, output_size, activation_fn=None), 1)
    estimator = NeuralValueEstimator(controller, env.obs_shape)
    alg = cfg.alg_class(estimator, name="critic")
    updater = RLUpdater(env, policy, alg)
    return updater


config = DEFAULT_CONFIG.copy(
    get_updater=get_updater,
    build_env=build_env,
    log_name="policy_evaluation",
    max_steps=100000,

    display_step=100,

    T=3,
    reward_radius=0.2,
    max_step=0.1,
    restart_prob=0.0,
    l2l=False,
    n_val=200,

    threshold=1e-4,

    verbose=False,
)

x = int(sys.argv[1])

if x == 0:
    print("TRPE")
    config.update(
        name="TRPE",
        delta_schedule='0.01',
        max_cg_steps=10,
        max_line_search_steps=10,
        alg_class=TrustRegionPolicyEvaluation
    )
elif x == 1:
    print("PPE")
    config.update(
        name="PPE",
        optimizer_spec="rmsprop",
        lr_schedule="1e-2",
        epsilon=0.2,
        opt_steps_per_update=100,
        S=1,
        alg_class=ProximalPolicyEvaluation
    )
else:
    print("PE")
    config.update(
        name="PolicyEvaluation",
        optimizer_spec='rmsprop',
        lr_schedule='1e-5',
        opt_steps_per_update=100,
        alg_class=PolicyEvaluation
    )


with config:
    cfg.update_from_command_line()
    list(training_loop())
