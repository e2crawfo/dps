import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import (
    RegressionDataset, RegressionEnv, CompositeEnv, TensorFlowEnv)


class HelloWorldDataset(RegressionDataset):
    def __init__(self, order, n_examples):
        self.order = order

        x = np.random.randn(n_examples, 2)
        x = np.concatenate((x.copy(), np.zeros((x.shape[0], 1))), axis=1)
        y = x[:, :2].copy()
        for i in order:
            if i == 0:
                y[:, 0] = y[:, 0] + y[:, 1]
            else:
                y[:, 1] = y[:, 0] * y[:, 1]

        super(HelloWorldDataset, self).__init__(x, y)


class HelloWorldEnv(RegressionEnv):
    def __init__(self, order, n_train, n_val):
        super(HelloWorldEnv, self).__init__(
            train=HelloWorldDataset(order, n_train),
            val=HelloWorldDataset(order, n_val))


class HelloWorld(TensorFlowEnv):
    action_names = ['r0 = r0 + r1', 'r1 = r0 * r1', 'no-op/stop']
    input_shape = (3,)
    make_input_available = False

    def __init__(self, env):
        self.rb = RegisterBank(
            'HelloWorldRB', 'r0 r1 r2', None, [1.0, 0.0, 0.0], 'r0 r1')
        super(HelloWorld, self).__init__()

    def static_inp_type_and_shape(self):
        return (tf.float32, (3,))

    def build_init(self, r, inp):
        r0 = inp[:, :1]
        r1 = inp[:, 1:2]
        r2 = inp[:, 2:]
        return self.rb.wrap(r0=r0, r1=r1, r2=r2)

    def build_step(self, t, r, a, static_inp):
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
        return tf.fill((tf.shape(r)[0], 1), 0.0), new_registers


def build_env():
    external = HelloWorldEnv(cfg.order, cfg.n_train, cfg.n_val)
    internal = HelloWorld(external)
    return CompositeEnv(external, internal)
