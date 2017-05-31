import tensorflow as tf
import numpy as np

from dps import CoreNetwork
from dps.register import RegisterBank
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config
from dps.attention import apply_gaussian_filter
from dps.production_system import ProductionSystemTrainer


class PointerDataset(RegressionDataset):
    def __init__(self, width, n_digits, n_examples, for_eval=False, shuffle=True):
        self.width = width
        self.n_digits = n_digits

        x = np.random.randint(0, n_digits, size=(n_examples, 2*width+1))
        x[:, width] = np.random.randint(-width, width+1, size=n_examples)
        y = x[range(x.shape[0]), x[:, width]+width].reshape(-1, 1)
        super(PointerDataset, self).__init__(x, y, for_eval, shuffle)


class PointerEnv(RegressionEnv):
    def __init__(self, width, n_digits, n_train, n_val, n_test):
        self.width = width
        self.n_digits = n_digits
        super(PointerEnv, self).__init__(
            train=PointerDataset(width, n_digits, n_train, for_eval=False),
            val=PointerDataset(width, n_digits, n_val, for_eval=True),
            test=PointerDataset(width, n_digits, n_test, for_eval=True))

    def __str__(self):
        return "<PointerEnv width={} n_digits={}>".format(self.width, self.n_digits)


class Pointer(CoreNetwork):
    action_names = ['fovea += 1', 'fovea -= 1', 'wm = vision', 'no-op/stop']

    def __init__(self, env):
        self.width = env.width
        self.register_bank = RegisterBank(
            'PointerRB', 'fovea vision wm t', None,
            values=([0.0] * 4), output_names='wm')
        super(Pointer, self).__init__()

    @property
    def input_shape(self):
        return 2*self.width+1

    @property
    def make_input_available(self):
        return True

    def init(self, r, inp):
        fovea, vision, wm, t = self.register_bank.as_tuple(r)

        diag_std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f').reshape(-1, 1))
        vision = apply_gaussian_filter(fovea, diag_std, locations, inp)

        new_registers = self.register_bank.wrap(fovea=fovea, vision=vision, wm=wm, t=t)

        return new_registers

    def __call__(self, action_activations, r, inp):
        _fovea, _vision, _wm, _t = self.register_bank.as_tuple(r)

        inc_fovea, dec_fovea, vision_to_wm, no_op = tf.split(action_activations, self.n_actions, axis=1)

        fovea = (1 - inc_fovea - dec_fovea) * _fovea + inc_fovea * (_fovea + 1) + dec_fovea * (_fovea - 1)
        wm = (1 - vision_to_wm) * _wm + vision_to_wm * _vision
        t = _t + 1

        diag_std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f').reshape(-1, 1))
        vision = apply_gaussian_filter(fovea, diag_std, locations, inp)

        new_registers = self.register_bank.wrap(fovea=fovea, vision=vision, wm=wm, t=t)

        return new_registers


class PointerTrainer(ProductionSystemTrainer):
    def build_env(self):
        config = default_config()
        return PointerEnv(config.width, config.n_digits, config.n_train, config.n_val, config.n_test)

    def build_core_network(self, env):
        return Pointer(env)
