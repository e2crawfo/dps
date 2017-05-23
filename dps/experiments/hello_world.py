import tensorflow as tf
import numpy as np

from dps import CoreNetwork
from dps.register import RegisterBank
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config
from dps.train import build_and_visualize
from dps.policy import Policy
from dps.production_system import ProductionSystemTrainer


class HelloWorldDataset(RegressionDataset):
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

        super(HelloWorldDataset, self).__init__(x, y, for_eval, shuffle)


class HelloWorldEnv(RegressionEnv):
    def __init__(self, order, n_train, n_val, n_test):
        super(HelloWorldEnv, self).__init__(
            train=HelloWorldDataset(order, n_train, for_eval=False),
            val=HelloWorldDataset(order, n_val, for_eval=True),
            test=HelloWorldDataset(order, n_test, for_eval=True))


class HelloWorld(CoreNetwork):
    action_names = ['r0 = r0 + r1', 'r1 = r0 * r1', 'no-op/stop']

    def __init__(self, env):
        self.register_bank = RegisterBank(
            'HelloWorldRB', 'r0 r1 r2', None, [1.0, 0.0, 0.0], 'r0 r1 r2', 'r0 r1')
        super(HelloWorld, self).__init__()

    def __call__(self, action_activations, r):
        """ Action 0: add the variables in the registers, store in r0.
            Action 1: multiply the variables in the registers, store in r1.
            Action 2: no-op

        """
        _r0, _r1, _r2 = self.register_bank.as_tuple(r)

        a0, a1, a2 = tf.split(action_activations, self.n_actions, axis=1)

        new_registers = self.register_bank.wrap(
            r0=a0 * (_r0 + _r1) + (1 - a0) * _r0,
            r1=a1 * (_r0 * _r1) + (1 - a1) * _r1,
            r2=_r2+1)
        return new_registers


def visualize(config):
    from dps.production_system import ProductionSystem
    from dps.policy import IdentitySelect
    from dps.utils import build_decaying_value, FixedController

    def build_psystem():
        _config = default_config()
        env = HelloWorldEnv([0, 1, 0], 10, 10, 10)
        cn = HelloWorld(env)

        controller = FixedController([0, 1, 0], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(
            _config.schedule('exploration'), 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="hello_world_policy")
        return ProductionSystem(env, cn, policy, False, 3)

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 1, False)


class HelloWorldTrainer(ProductionSystemTrainer):
    def build_env(self):
        config = default_config()
        return HelloWorldEnv(config.order, config.n_train, config.n_val, config.n_test)

    def build_core_network(self, env):
        return HelloWorld(env)
