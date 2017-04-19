from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import Config, default_config, CompositeCell, get_config
from dps.train import training_loop
from dps.policy import Policy, ReluSelect, SoftmaxSelect, GumbelSoftmaxSelect
from dps.production_system import ProductionSystemCurriculum


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
    _visible = [1, 1, 1]
    _initial_values = [np.array([v], dtype='f') for v in [1.0, 0.0, 0.0]]
    _namedtuple = addition_nt
    _input_names = addition_nt._fields
    _output_names = addition_nt._fields


class Addition(CoreNetwork):
    _register_spec = AdditionRegSpec()
    _n_actions = 3

    def __init__(self, env):
        super(Addition, self).__init__()

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


class DefaultConfig(Config):
    seed = 10

    T = 3
    curriculum = [dict(order=[0, 1, 0])]
    optimizer_class = tf.train.RMSPropOptimizer

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
        tf.contrib.rnn.LSTMCell(num_units=64),
        fully_connected,
        Addition._n_actions+1)

    action_selection = staticmethod([
        SoftmaxSelect(),
        GumbelSoftmaxSelect(hard=0),
        GumbelSoftmaxSelect(hard=1),
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


class CurriculumConfig(DefaultConfig):
    curriculum = [
        dict(order=[0], T=1),
        dict(order=[0, 1], T=2),
        dict(order=[0, 1, 0], T=3)]


class RLConfig(DefaultConfig):
    use_rl = True
    threshold = 1e-2
    action_selection = SoftmaxSelect()


class RLCurriculumConfig(RLConfig, CurriculumConfig):
    pass


def train_addition(log_dir, config="default", seed=-1):
    config = get_config(__file__, config)
    config.seed = config.seed if seed < 0 else seed
    np.random.seed(config.seed)

    base_kwargs = dict(n_train=config.n_train, n_val=config.n_val, n_test=config.n_test)

    def build_env(**kwargs):
        return AdditionEnv(**kwargs)

    def build_core_network(env):
        return Addition(env)

    def build_policy(cn, exploration):
        config = default_config()
        return Policy(
            config.controller, config.action_selection, exploration,
            cn.n_actions+1, cn.obs_dim, name="addition_policy")

    curriculum = ProductionSystemCurriculum(
        base_kwargs, config.curriculum, build_env, build_core_network, build_policy)

    exp_name = "selection={}_rl={}".format(
        config.action_selection.__class__.__name__, int(config.use_rl))
    training_loop(curriculum, log_dir, config, exp_name=exp_name)


if __name__ == '__main__':
    from clify import command_line
    command_line(train_addition)(log_dir='/tmp/dps/addition')
