import tensorflow as tf
import numpy as np

from dps import cfg
from dps.utils import Param
from dps.register import RegisterBank
from dps.environment import (
    RegressionDataset, RegressionEnv, CompositeEnv, InternalEnv)


class HelloWorldDataset(RegressionDataset):
    order = Param()

    def __init__(self, **kwargs):
        x = np.random.randn(self.n_examples, 2)
        x = np.concatenate((x.copy(), np.zeros((x.shape[0], 1))), axis=1)
        y = x[:, :2].copy()
        for i in self.order:
            if i == 0:
                y[:, 0] = y[:, 0] + y[:, 1]
            else:
                y[:, 1] = y[:, 0] * y[:, 1]

        super(HelloWorldDataset, self).__init__(x, y)


class HelloWorld(InternalEnv):
    action_names = ['r0 = r0 + r1', 'r1 = r0 * r1', 'no-op/stop']
    rb = RegisterBank('HelloWorldRB', 'r0 r1 r2', None, [1.0, 0.0, 0.0], 'r0 r1')

    @property
    def input_shape(self):
        return (3,)

    @property
    def target_shape(self):
        return (2,)

    def build_init(self, r):
        self.build_placeholders(r)

        r0 = self.input_ph[:, :1]
        r1 = self.input_ph[:, 1:2]
        r2 = self.input_ph[:, 2:]
        return self.rb.wrap(r0=r0, r1=r1, r2=r2)

    def build_step(self, t, r, a):
        """ Action 0: add the variables in the registers, store in r0.
            Action 1: multiply the variables in the registers, store in r1.
            Action 2: no-op

        """
        _r0, _r1, _r2 = self.rb.as_tuple(r)

        add, mult, noop = tf.split(a, self.n_actions, axis=1)

        new_registers = self.rb.wrap(
            r0=add * (_r0 + _r1) + (1 - add) * _r0,
            r1=mult * (_r0 * _r1) + (1 - mult) * _r1,
            r2=_r2+1)

        rewards = self.build_rewards(new_registers)

        return (
            tf.fill((tf.shape(r)[0], 1), 0.0),
            rewards,
            new_registers)


def build_env():
    train = HelloWorldDataset(n_examples=cfg.n_train)
    val = HelloWorldDataset(n_examples=cfg.n_val)
    external = RegressionEnv(train, val)
    internal = HelloWorld()
    return CompositeEnv(external, internal)
