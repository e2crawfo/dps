import tensorflow as tf
import numpy as np

from dps import cfg
from dps.register import RegisterBank
from dps.environment import (
    RegressionDataset, RegressionEnv, CompositeEnv, InternalEnv)
from dps.vision.attention import gaussian_filter
from dps.utils import Param, digits_to_numbers, numbers_to_digits, Config


def build_env():
    train = HardAdditionDataset(n_examples=cfg.n_train)
    val = HardAdditionDataset(n_examples=cfg.n_val)

    external = RegressionEnv(train, val)
    internal = HardAddition()
    return CompositeEnv(external, internal)


config = Config(
    build_env=build_env,
    T=40,
    curriculum=[
        dict(height=2, width=3, base=2, entropy_start=1.0),
        dict(height=2, width=3, base=2, entropy_start=0.0)],
    log_name='hard_addition',
)


class HardAdditionDataset(RegressionDataset):
    width = Param()
    height = Param()
    base = Param()

    def __init__(self, **kwargs):
        width, height = self.width, self.height
        x = np.random.randint(0, self.base, size=(self.n_examples, width*height))
        for h in range(height):
            x[:, (h+1)*width - 1] = 0
        y = digits_to_numbers(x[:, :width])
        offset = width
        for i in range(height-1):
            y += digits_to_numbers(x[:, offset:offset+width])
            offset += width
        y = numbers_to_digits(y, width)
        if height > 2:
            raise Exception(
                "Need to specify greater number of digits when adding > 2 numbers.")

        super(HardAdditionDataset, self).__init__(x, y)


class HardAddition(InternalEnv):
    """ Top left is (x=0, y=0).

    For now, the location of the write head is the same as the x location of
    the read head.

    """
    action_names = ['fovea_x += 1', 'fovea_x -= 1', 'fovea_y += 1', 'fovea_y -= 1',
                    'wm1 = vision', 'wm2 = vision', 'add', 'write_digit', 'no-op/stop']

    width = Param()
    height = Param()

    @property
    def input_shape(self):
        return (self.width*self.height,)

    @property
    def target_shape(self):
        return (self.width,)

    def __init__(self, **kwargs):
        values = (
            ([0.0] * 7) +
            [np.zeros(self.width, dtype='f')])

        self.rb = RegisterBank(
            'HardAdditionRB',
            'fovea_x fovea_y vision wm1 wm2 carry digit', 'outp',
            values=values, output_names='outp')

        super(HardAddition, self).__init__()

    def build_init(self, r):
        self.build_placeholders(r)

        fovea_x, fovea_y, _vision, wm1, wm2, carry, digit, outp = self.rb.as_tuple(r)

        # Read input
        fovea = tf.concat([fovea_y, fovea_x], 1)
        std = tf.fill(tf.shape(fovea), 0.01)
        static_inp = tf.reshape(self.input_ph, (-1, self.height, self.width))
        x_filter = gaussian_filter(fovea[:, 1:], std[:, 1:], np.arange(self.width, dtype='f'))
        y_filter = gaussian_filter(fovea[:, :1], std[:, :1], np.arange(self.height, dtype='f'))
        vision = tf.matmul(y_filter, tf.matmul(static_inp, x_filter, adjoint_b=True))
        vision = tf.reshape(vision, (-1, 1))

        with tf.name_scope("HardAddition"):
            new_registers = self.rb.wrap(
                outp=tf.identity(outp, "outp"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                wm1=tf.identity(wm1, "wm1"),
                wm2=tf.identity(wm2, "wm2"),
                carry=tf.identity(carry, "carry"),
                digit=tf.identity(digit, "digit"))
        return new_registers

    def build_step(self, t, r, a):
        _fovea_x, _fovea_y, _vision, _wm1, _wm2, _carry, _digit, _outp = self.rb.as_tuple(r)

        (inc_fovea_x, dec_fovea_x, inc_fovea_y, dec_fovea_y,
         vision_to_wm1, vision_to_wm2, add, write_digit, no_op) = self.unpack_actions(a)

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
        static_inp = tf.reshape(self.input_ph, (-1, self.height, self.width))
        x_filter = gaussian_filter(fovea[:, 1:], std[:, 1:], np.arange(self.width, dtype='f'))
        y_filter = gaussian_filter(fovea[:, :1], std[:, :1], np.arange(self.height, dtype='f'))
        vision = tf.matmul(y_filter, tf.matmul(static_inp, x_filter, adjoint_b=True))
        vision = tf.reshape(vision, (-1, 1))

        # Store output
        write_weighting = gaussian_filter(_fovea_x, tf.fill(tf.shape(_fovea_x), 0.01), np.arange(self.width, dtype='f'))
        write_weighting = tf.squeeze(write_weighting, axis=[1])
        outp = (1 - write_digit) * _outp + write_digit * ((1 - write_weighting) * _outp + write_weighting * _digit)

        with tf.name_scope("HardAddition"):
            new_registers = self.rb.wrap(
                outp=tf.identity(outp, "outp"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                wm1=tf.identity(wm1, "wm1"),
                wm2=tf.identity(wm2, "wm2"),
                carry=tf.identity(carry, "carry"),
                digit=tf.identity(digit, "digit"))

        rewards = self.build_rewards(new_registers)

        return (
            tf.fill((tf.shape(r)[0], 1), 0.0),
            rewards,
            new_registers)
