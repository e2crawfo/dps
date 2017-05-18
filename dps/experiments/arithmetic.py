from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config
from dps.train import build_and_visualize
from dps.policy import Policy
from dps.production_system import ProductionSystemTrainer


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
arithmetic_nt = namedtuple('ArithmeticRegister', 'r0 r1 r2'.split())


class ArithmeticRegSpec(RegisterSpec):
    _visible = [1, 1, 1]
    _initial_values = [np.array([v], dtype='f') for v in [1.0, 0.0, 0.0]]
    _namedtuple = arithmetic_nt
    _input_names = arithmetic_nt._fields
    _output_names = 'r0 r1'.split()


class Arithmetic(CoreNetwork):
    action_names = ['r0 = r0 + r1', 'r1 = r0 * r1', 'no-op/stop']
    register_spec = ArithmeticRegSpec()

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


def visualize(config):
    from dps.production_system import ProductionSystem
    from dps.policy import IdentitySelect
    from dps.utils import build_decaying_value, FixedController

    def build_psystem():
        _config = default_config()
        env = ArithmeticEnv([0, 1, 0], 10, 10, 10)
        cn = Arithmetic(env)

        controller = FixedController([0, 1, 0], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.schedule(exploration), 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="arithmetic_policy")
        return ProductionSystem(env, cn, policy, False, 3)

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 1, False)


class ArithmeticTrainer(ProductionSystemTrainer):
    def build_env(self):
        config = default_config()
        return ArithmeticEnv(config.order, config.n_train, config.n_val, config.n_test)

    def build_core_network(self, env):
        return Arithmetic(env)
