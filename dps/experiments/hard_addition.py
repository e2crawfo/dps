import tensorflow as tf
import numpy as np

from dps import CoreNetwork
from dps.register import RegisterBank
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config
from dps.attention import gaussian_filter
from dps.production_system import ProductionSystemTrainer


def digits_to_numbers(digits, base=10, axis=-1, keepdims=False):
    """ Assumes little-endian (least-significant stored first). """
    mult = base ** np.arange(digits.shape[axis])
    shape = [1] * digits.ndim
    shape[axis] = mult.shape[axis]
    mult = mult.reshape(shape)
    return (digits * mult).sum(axis=axis, keepdims=keepdims)


def numbers_to_digits(numbers, n_digits, base=10):
    numbers = numbers.copy()
    digits = []
    for i in range(n_digits):
        digits.append(numbers % base)
        numbers //= base
    return np.stack(digits, -1)


class HardAdditionDataset(RegressionDataset):
    def __init__(
            self, height, width, n_digits, n_examples, for_eval=False, shuffle=True):
        self.height = height
        self.width = width
        self.n_digits = n_digits

        x = np.random.randint(0, n_digits, size=(n_examples, width*height))
        for h in range(height):
            x[:, (h+1)*width - 1] = 0
        y = digits_to_numbers(x[:, :width])
        offset = width
        for i in range(height-1):
            y += digits_to_numbers(x[:, offset:offset+width])
            offset += width
        y = numbers_to_digits(y, width)
        if height > 2:
            raise NotImplementedError("Need to specify greater number of digits when adding > 2 numbers.")

        super(HardAdditionDataset, self).__init__(x, y, for_eval, shuffle)


class HardAdditionEnv(RegressionEnv):
    def __init__(self, height, width, n_digits, n_train, n_val, n_test):
        self.height = height
        self.width = width
        self.n_digits = n_digits
        super(HardAdditionEnv, self).__init__(
            train=HardAdditionDataset(height, width, n_digits, n_train, for_eval=False),
            val=HardAdditionDataset(height, width, n_digits, n_val, for_eval=True),
            test=HardAdditionDataset(height, width, n_digits, n_test, for_eval=True))

    def __str__(self):
        return "<HardAdditionEnv height={} width={} n_digits={}>".format(self.height, self.width, self.n_digits)


class HardAddition(CoreNetwork):
    """ Top left is (x=0, y=0).

    For now, the location of the write head is the same as the x location of the read head.

    """
    action_names = ['fovea_x += 1', 'fovea_x -= 1', 'fovea_y += 1', 'fovea_y -= 1',
                    'wm1 = vision', 'wm2 = vision', 'add', 'write_digit', 'no-op/stop']

    @property
    def input_dim(self):
        return self.width * self.height

    @property
    def make_input_available(self):
        return True

    def __init__(self, env):
        self.height = env.height
        self.width = env.width

        values = (
            ([0.0] * 8) +
            [np.zeros(self.width, dtype='f')])

        self.register_bank = RegisterBank(
            'HardAdditionRB',
            'fovea_x fovea_y vision wm1 wm2 carry digit t', 'outp',
            values=values, output_names='outp')

        super(HardAddition, self).__init__()

    def init(self, r, inp):
        fovea_x, fovea_y, _vision, wm1, wm2, carry, digit, t, outp = self.register_bank.as_tuple(r)

        # Read input
        fovea = tf.concat([fovea_y, fovea_x], 1)
        std = tf.fill(tf.shape(fovea), 0.01)
        inp = tf.reshape(inp, (-1, self.height, self.width))
        x_filter = gaussian_filter(fovea[:, 1:], std[:, 1:], np.arange(self.width, dtype='f'))
        y_filter = gaussian_filter(fovea[:, :1], std[:, :1], np.arange(self.height, dtype='f'))
        vision = tf.matmul(y_filter, tf.matmul(inp, x_filter, adjoint_b=True))
        vision = tf.reshape(vision, (-1, 1))

        with tf.name_scope("HardAddition"):
            new_registers = self.register_bank.wrap(
                outp=tf.identity(outp, "outp"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                wm1=tf.identity(wm1, "wm1"),
                wm2=tf.identity(wm2, "wm2"),
                carry=tf.identity(carry, "carry"),
                digit=tf.identity(digit, "digit"),
                t=tf.identity(t, "t"))
        return new_registers

    def __call__(self, action_activations, r, inp):

        _fovea_x, _fovea_y, _vision, _wm1, _wm2, _carry, _digit, _t, _outp = self.register_bank.as_tuple(r)

        (inc_fovea_x, dec_fovea_x, inc_fovea_y, dec_fovea_y,
         vision_to_wm1, vision_to_wm2, add, write_digit, no_op) = (
            tf.split(action_activations, self.n_actions, axis=1))

        fovea_x = (1 - inc_fovea_x - dec_fovea_x) * _fovea_x + inc_fovea_x * (_fovea_x + 1) + dec_fovea_x * (_fovea_x - 1)
        fovea_y = (1 - inc_fovea_y - dec_fovea_y) * _fovea_y + inc_fovea_y * (_fovea_y + 1) + dec_fovea_y * (_fovea_y - 1)
        wm1 = (1 - vision_to_wm1) * _wm1 + vision_to_wm1 * _vision
        wm2 = (1 - vision_to_wm2) * _wm2 + vision_to_wm2 * _vision

        add_result = tf.round(_wm1 + _wm2 + _carry)

        carry = (1 - add) * _carry + add * (add_result // 10)
        digit = (1 - add) * _digit + add * tf.mod(add_result, 10)

        # Read input
        fovea = tf.concat([fovea_y, fovea_x], 1)
        std = tf.fill(tf.shape(fovea), 0.01)
        inp = tf.reshape(inp, (-1, self.height, self.width))
        x_filter = gaussian_filter(fovea[:, 1:], std[:, 1:], np.arange(self.width, dtype='f'))
        y_filter = gaussian_filter(fovea[:, :1], std[:, :1], np.arange(self.height, dtype='f'))
        vision = tf.matmul(y_filter, tf.matmul(inp, x_filter, adjoint_b=True))
        vision = tf.reshape(vision, (-1, 1))

        # Store output
        write_weighting = gaussian_filter(_fovea_x, tf.fill(tf.shape(_fovea_x), 0.01), np.arange(self.width, dtype='f'))
        write_weighting = tf.squeeze(write_weighting, axis=[1])
        outp = (1 - write_digit) * _outp + write_digit * ((1 - write_weighting) * _outp + write_weighting * _digit)

        t = _t + 1

        with tf.name_scope("HardAddition"):
            new_registers = self.register_bank.wrap(
                outp=tf.identity(outp, "outp"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                wm1=tf.identity(wm1, "wm1"),
                wm2=tf.identity(wm2, "wm2"),
                carry=tf.identity(carry, "carry"),
                digit=tf.identity(digit, "digit"),
                t=tf.identity(t, "t"))

        return new_registers


class HardAdditionTrainer(ProductionSystemTrainer):
    def build_env(self):
        config = default_config()
        return HardAdditionEnv(
            config.height, config.width, config.n_digits,
            config.n_train, config.n_val, config.n_test)

    def build_core_network(self, env):
        return HardAddition(env)
