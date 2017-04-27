from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import Config, default_config, CompositeCell, MLP
from dps.train import training_loop
from dps.policy import Policy, ReluSelect, SoftmaxSelect, GumbelSoftmaxSelect
from dps.production_system import ProductionSystemCurriculum
from dps.updater import DifferentiableUpdater


class ArithmeticDataset(RegressionDataset):
    def __init__(self, order, n_examples, for_eval=False, shuffle=True):
        self.order = order

        x = np.random.randn(n_examples, 2)
        x = np.concatenate((x.copy(), np.zeros((x.shape[0], 1))), axis=1)
        y = x[:, :2].copy()
        for i in order:
            if i == 0:
                y[:, 0] = y[:, 0] + y[:, 1]
            else:
                y[:, 1] = y[:, 0] * y[:, 1]

        super(ArithmeticDataset, self).__init__(x, y, for_eval, shuffle)


class ArithmeticEnv(RegressionEnv):
    def __init__(self, order, n_train, n_val, n_test):
        super(ArithmeticEnv, self).__init__(
            train=ArithmeticDataset(order, n_train, for_eval=False),
            val=ArithmeticDataset(order, n_val, for_eval=True),
            test=ArithmeticDataset(order, n_test, for_eval=True))


# Define at top-level to enable pickling
arithmetic_nt = namedtuple('_ArithmeticRegister', 'r0 r1 r2'.split())


class ArithmeticRegSpec(RegisterSpec):
    _visible = [1, 1, 1]
    _initial_values = [np.array([v], dtype='f') for v in [1.0, 0.0, 0.0]]
    _namedtuple = arithmetic_nt
    _input_names = arithmetic_nt._fields
    _output_names = 'r0 r1'.split()


class Arithmetic(CoreNetwork):
    _n_actions = 3
    _action_names = ['r0 = r0 + r1', 'r1 = r0 * r1', 'no-op/stop']
    _register_spec = ArithmeticRegSpec()

    def __init__(self, env):
        super(Arithmetic, self).__init__()

    def __call__(self, action_activations, r):
        """ Action 0: add the variables in the registers, store in r0.
            Action 1: multiply the variables in the registers, store in r1.
            Action 2: no-op """
        a0, a1, a2 = tf.split(action_activations, self.n_actions, axis=1)
        r0 = a0 * (r.r0 + r.r1) + (1 - a0) * r.r0
        r1 = a1 * (r.r0 * r.r1) + (1 - a1) * r.r1
        new_registers = self.register_spec.wrap(r0=r0, r1=r1, r2=r.r2+1)
        return new_registers


class ArithmeticConfig(Config):
    seed = 10

    T = 3
    curriculum = [dict(order=[0, 1, 0])]
    optimizer_class = tf.train.RMSPropOptimizer
    updater_class = DifferentiableUpdater

    max_steps = 10000
    batch_size = 100
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-2
    patience = 100

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=32),
        MLP(),
        Arithmetic._n_actions)

    action_selection = staticmethod([
        SoftmaxSelect(),
        GumbelSoftmaxSelect(hard=0),
        GumbelSoftmaxSelect(hard=1),
        ReluSelect()][0])

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.96, False)
    noise_schedule = (0.0, 10, 0.96, False)
    exploration_schedule = (10.0, 100, 0.96, False)

    test_time_explore = None

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


def train(log_dir, config, seed=-1):
    config.seed = config.seed if seed < 0 else seed
    np.random.seed(config.seed)

    base_kwargs = dict(n_train=config.n_train, n_val=config.n_val, n_test=config.n_test)

    def build_env(**kwargs):
        return ArithmeticEnv(**kwargs)

    def build_core_network(env):
        return Arithmetic(env)

    def build_policy(cn, exploration):
        config = default_config()
        return Policy(
            config.controller, config.action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="arithmetic_policy")

    curriculum = ProductionSystemCurriculum(
        base_kwargs, config.curriculum, config.updater_class, build_env, build_core_network, build_policy)

    exp_name = "selection={}_updater={}".format(
        config.action_selection.__class__.__name__, config.updater_class.__name__)
    training_loop(curriculum, log_dir, config, exp_name=exp_name)


if __name__ == '__main__':
    from clify import command_line
    command_line(train)(log_dir='/tmp/dps/arithmetic', config=ArithmeticConfig())
