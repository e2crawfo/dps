import tensorflow as tf
import numpy as np

from dps import CoreNetwork
from dps.register import RegisterBank
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config
from dps.attention import apply_gaussian_filter
from dps.production_system import ProductionSystemTrainer


class SimpleAdditionDataset(RegressionDataset):
    def __init__(self, width, n_digits, n_examples, for_eval=False, shuffle=True):
        self.width = width
        self.n_digits = n_digits

        x = np.random.randint(0, n_digits, size=(n_examples, 2*width+1))
        y = x[:, :1] + x[:, -1:]
        super(SimpleAdditionDataset, self).__init__(x, y, for_eval, shuffle)


class SimpleAdditionEnv(RegressionEnv):
    def __init__(self, width, n_digits, n_train, n_val, n_test):
        self.width = width
        self.n_digits = n_digits
        super(SimpleAdditionEnv, self).__init__(
            train=SimpleAdditionDataset(width, n_digits, n_train, for_eval=False),
            val=SimpleAdditionDataset(width, n_digits, n_val, for_eval=True),
            test=SimpleAdditionDataset(width, n_digits, n_test, for_eval=True))

    def __str__(self):
        return "<SimpleAdditionEnv width={} n_digits={}>".format(self.width, self.n_digits)


class SimpleAddition(CoreNetwork):
    action_names = ['fovea += 1', 'fovea -= 1', 'wm1 = vision', 'wm2 = vision',
                    'output = vision', 'output = wm1 + wm2', 'no-op/stop']

    @property
    def input_shape(self):
        return 2*self.width+1

    @property
    def make_input_available(self):
        return True

    def __init__(self, env):
        self.width = env.width
        self.register_bank = RegisterBank(
            'SimpleAdditionRB',
            'fovea vision wm1 wm2 output t', None,
            values=([0.] * 6),
            output_names='output')
        super(SimpleAddition, self).__init__()

    def init(self, r, inp):
        fovea, vision, wm1, wm2, output, t = self.register_bank.as_tuple(r)
        std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f'), dtype=tf.float32)
        vision = apply_gaussian_filter(fovea, std, locations, inp)

        new_registers = self.register_bank.wrap(
            fovea=fovea, vision=vision, wm1=wm1, wm2=wm2, output=output, t=t)

        return new_registers

    def __call__(self, action_activations, r, inp):
        _fovea, _vision, _wm1, _wm2, _output, _t = self.register_bank.as_tuple(r)

        inc_fovea, dec_fovea, vision_to_wm1, vision_to_wm2, vision_to_output, add, no_op = (
            tf.split(action_activations, self.n_actions, axis=1))

        fovea = (1 - inc_fovea - dec_fovea) * _fovea + inc_fovea * (_fovea + 1) + dec_fovea * (_fovea - 1)
        wm1 = (1 - vision_to_wm1) * _wm1 + vision_to_wm1 * _vision
        wm2 = (1 - vision_to_wm2) * _wm2 + vision_to_wm2 * _vision
        output = (1 - vision_to_output - add) * _output + vision_to_output * _vision + add * (_wm1 + _wm2)

        std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f'), dtype=tf.float32)
        vision = apply_gaussian_filter(fovea, std, locations, inp)

        t = _t + 1

        new_registers = self.register_bank.wrap(
            fovea=fovea, vision=vision, wm1=wm1, wm2=wm2, output=output, t=t)

        return new_registers


class SimpleAdditionTrainer(ProductionSystemTrainer):
    def build_env(self):
        config = default_config()
        return SimpleAdditionEnv(
            config.width, config.n_digits,
            config.n_train, config.n_val, config.n_test)

    def build_core_network(self, env):
        return SimpleAddition(env)
