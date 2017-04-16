from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
import numpy as np

from dps import (
    ProductionSystem, ProductionSystemFunction, ProductionSystemEnv,
    CoreNetwork, RegisterSpec, DifferentiableUpdater)
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import Config, default_config, CompositeCell
from dps.train import training_loop
from dps.rl import REINFORCE
from dps.policy import Policy, ReluSelect, SoftmaxSelect, GumbelSoftmaxSelect


class AdditionDataset(RegressionDataset):
    def __init__(self, order, n_examples, for_eval=False, shuffle=True):
        self.order = order

        x = np.random.randn(n_examples, 2)
        x = np.concatenate((x.copy(), np.zeros((x.shape[0], 1))), axis=1)
        y = x.copy()
        for i in order:
            if i == 0:
                y[:, 0] = y[:, 0] + y[:, 1]
            else:
                y[:, 1] = y[:, 0] * y[:, 1]

        super(AdditionDataset, self).__init__(x, y, for_eval, shuffle)


class AdditionEnv(RegressionEnv):
    def __init__(self, order, n_train, n_val, n_test):
        super(AdditionEnv, self).__init__(
            train=AdditionDataset(order, n_train, for_eval=False),
            val=AdditionDataset(order, n_val, for_eval=True),
            test=AdditionDataset(order, n_test, for_eval=True))


# Define at top-level to enable pickling
addition_nt = namedtuple('_AdditionRegister', 'r0 r1 r2'.split())


class AdditionRegSpec(RegisterSpec):
    @property
    def visible(self):
        return [1, 1, 1]

    @property
    def initial_values(self):
        return [np.array([v], dtype='f') for v in [1.0, 0.0, 0.0]]

    @property
    def namedtuple(self):
        return addition_nt

    @property
    def input_names(self):
        return self.names

    @property
    def output_names(self):
        return self.names


class Addition(CoreNetwork):
    _register_spec = AdditionRegSpec()
    _n_actions = 3

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def register_spec(self):
        return self._register_spec

    def __call__(self, action_activations, r):
        """ Action 0: add the variables in the registers, store in r0.
            Action 1: multiply the variables in the registers, store in r1.
            Action 2: no-op """
        debug = default_config().debug
        if debug:
            action_activations = tf.Print(action_activations, [r], "registers", summarize=20)
            action_activations = tf.Print(
                action_activations, [action_activations], "action activations", summarize=20)

        a0, a1, a2 = tf.split(action_activations, self.n_actions, axis=1)
        r0 = a0 * (r.r0 + r.r1) + (1 - a0) * r.r0
        r1 = a1 * (r.r0 * r.r1) + (1 - a1) * r.r1

        if debug:
            r0 = tf.Print(r0, [r0], "r0", summarize=20)
            r1 = tf.Print(r1, [r1], "r1", summarize=20)
        new_registers = self.register_spec.wrap(r0=r0, r1=r1, r2=r.r2+0)

        return new_registers


def _build_psystem(global_step):
    config = default_config()

    adder = Addition()

    start, decay_steps, decay_rate, staircase = config.exploration_schedule
    exploration = tf.train.exponential_decay(
        start, global_step, decay_steps, decay_rate, staircase=staircase)
    tf.summary.scalar('exploration', exploration)

    policy = Policy(
        config.controller, config.action_selection, exploration,
        adder.n_actions+1, adder.obs_dim, name="addition_lstm")

    psystem = ProductionSystem(adder, policy, False, config.T)
    return psystem


class DefaultConfig(Config):
    seed = 10

    T = 4
    order = [0, 0, 0, 1]

    optimizer_class = tf.train.RMSPropOptimizer

    max_steps = 10000
    batch_size = 100
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-3
    patience = 100

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=64),
        fully_connected,
        Addition._n_actions+1)

    action_selection = staticmethod([
        SoftmaxSelect(),
        GumbelSoftmaxSelect(hard=0),
        ReluSelect()][0])

    use_rl = False

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.96, False)
    noise_schedule = (0.0, 10, 0.96, False)
    exploration_schedule = (10.0, 100, 0.96, False)

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


class DebugConfig(DefaultConfig):
    debug = True

    max_steps = 100
    n_train = 2
    batch_size = 2
    eval_step = 1
    display_step = 1
    checkpoint_step = 1
    exploration_schedule = (0.5, 100, 0.96, False)


class RLConfig(DefaultConfig):
    use_rl = True
    threshold = 1e-2


class RLDebugConfig(DebugConfig, RLConfig):
    pass


def get_config(name):
    try:
        return dict(
            default=DefaultConfig(),
            rl=RLConfig(),
            debug=DebugConfig(),
            rl_debug=RLDebugConfig(),
        )[name]
    except KeyError:
        raise KeyError("Unknown config name {}.".format(name))


def train_addition(log_dir, config="default"):
    config = get_config(config)
    np.random.seed(config.seed)

    env = AdditionEnv(config.order, config.n_train, config.n_val, config.n_test)

    def build_diff_updater():
        global_step = tf.contrib.framework.get_or_create_global_step()
        psystem = _build_psystem(global_step)
        ps_func = ProductionSystemFunction(psystem)
        return DifferentiableUpdater(
            env, ps_func, global_step, config.optimizer_class,
            config.lr_schedule, config.noise_schedule, config.max_grad_norm)

    def build_reinforce_updater():
        global_step = tf.contrib.framework.get_or_create_global_step()
        psystem = _build_psystem(global_step)
        ps_env = ProductionSystemEnv(psystem, env)
        return REINFORCE(
            ps_env, psystem.policy, global_step, config.optimizer_class,
            config.lr_schedule, config.noise_schedule, config.max_grad_norm,
            config.gamma, config.l2_norm_param)

    build_updater = build_reinforce_updater if config.use_rl else build_diff_updater
    training_loop(env, build_updater, log_dir, config)


def main(argv=None):
    from clify import command_line
    command_line(train_addition)(log_dir='/tmp/dps/addition')


if __name__ == '__main__':
    tf.app.run()
