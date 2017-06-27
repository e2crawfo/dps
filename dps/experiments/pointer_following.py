import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import (
    RegressionDataset, RegressionEnv, CompositeEnv, TensorFlowEnv)
from dps.attention import apply_gaussian_filter


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


class Pointer(TensorFlowEnv):
    action_names = ['fovea += 1', 'fovea -= 1', 'wm = vision', 'no-op/stop']

    def __init__(self, env):
        self.width = env.width
        self.rb = RegisterBank(
            'PointerRB', 'fovea vision wm', None,
            values=([0.0] * 3), output_names='wm')
        super(Pointer, self).__init__()

    def static_inp_type_and_shape(self):
        return (tf.float32, (2*self.width+1,))

    make_input_available = True

    def build_init(self, r, static_inp):
        fovea, vision, wm = self.rb.as_tuple(r)

        diag_std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f').reshape(-1, 1))
        vision = apply_gaussian_filter(fovea, diag_std, locations, static_inp)

        new_registers = self.rb.wrap(fovea=fovea, vision=vision, wm=wm)

        return new_registers

    def build_step(self, t, r, a, static_inp):
        _fovea, _vision, _wm = self.rb.as_tuple(r)

        inc_fovea, dec_fovea, vision_to_wm, no_op = tf.split(a, self.n_actions, axis=1)

        fovea = (1 - inc_fovea - dec_fovea) * _fovea + inc_fovea * (_fovea + 1) + dec_fovea * (_fovea - 1)
        wm = (1 - vision_to_wm) * _wm + vision_to_wm * _vision

        diag_std = tf.fill(tf.shape(fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f').reshape(-1, 1))
        vision = apply_gaussian_filter(fovea, diag_std, locations, static_inp)

        new_registers = self.rb.wrap(fovea=fovea, vision=vision, wm=wm)

        return tf.fill((tf.shape(r)[0], 1), 0.0), new_registers


def build_env():
    external = PointerEnv(cfg.width, cfg.n_digits, cfg.n_train, cfg.n_val, cfg.n_test)
    internal = Pointer(external)
    return CompositeEnv(external, internal)
