from collections import namedtuple
from itertools import product

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config, visual_filter
from dps.production_system import ProductionSystemCurriculum
from dps.train import training_loop, build_and_visualize
from dps.policy import Policy


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
            raise NotImplementedError("^^^ need to specify greater number of digits when adding > 2 numbers.")

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


# Define at top-level to enable pickling
addition_nt = namedtuple('HardAdditionRegister', 'inp outp fovea_x fovea_y vision wm1 wm2 carry digit t'.split())


class HardAdditionRegSpec(RegisterSpec):
    _visible = [0, 0] + [1] * 8
    _initial_values = None
    _namedtuple = addition_nt
    _input_names = ['inp']
    _output_names = ['outp']

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self._initial_values = (
            [np.zeros(height*width, dtype='f')] +
            [np.zeros(width, dtype='f')] +
            [np.array([v], dtype='f') for v in [0.0] * 8])

        super(HardAdditionRegSpec, self).__init__()


class HardAddition(CoreNetwork):
    """ Top left is (x=0, y=0).

    For now, the location of the write head is the same as the x location of the read head.

    """
    action_names = ['fovea_x += 1', 'fovea_x -= 1', 'fovea_y += 1', 'fovea_y -= 1',
                    'wm1 = vision', 'wm2 = vision', 'add', 'write_digit', 'no-op/stop']

    def __init__(self, env):
        self.height = env.height
        self.width = env.width
        self.register_spec = HardAdditionRegSpec(env.height, env.width)
        super(HardAddition, self).__init__()

    def __call__(self, action_activations, r):
        (inc_fovea_x, dec_fovea_x, inc_fovea_y, dec_fovea_y,
         vision_to_wm1, vision_to_wm2, add, write_digit, no_op) = (
            tf.split(action_activations, self.n_actions, axis=1))

        fovea_x = (1 - inc_fovea_x - dec_fovea_x) * r.fovea_x + inc_fovea_x * (r.fovea_x + 1) + dec_fovea_x * (r.fovea_x - 1)
        fovea_y = (1 - inc_fovea_y - dec_fovea_y) * r.fovea_y + inc_fovea_y * (r.fovea_y + 1) + dec_fovea_y * (r.fovea_y - 1)
        wm1 = (1 - vision_to_wm1) * r.wm1 + vision_to_wm1 * r.vision
        wm2 = (1 - vision_to_wm2) * r.wm2 + vision_to_wm2 * r.vision

        add_result = r.wm1 + r.wm2 + r.carry
        add_result = tf.round(add_result)
        _carry = add_result // 10
        _digit = tf.mod(add_result, 10)

        carry = (1 - add) * r.carry + add * _carry
        digit = (1 - add) * r.digit + add * _digit

        # Read input
        fovea = tf.concat([fovea_y, fovea_x], 1)
        diag_std = tf.fill(tf.shape(fovea), 0.01)
        locations = np.array(list(product(range(self.height), range(self.width))), dtype='f')
        vision = visual_filter(fovea, diag_std, locations, r.inp)

        # Store output
        locations = np.arange(self.width).astype('f').reshape(-1, 1)
        write_weighting = visual_filter(fovea_x, tf.fill(tf.shape(fovea_x), 0.01), locations)
        output = (1 - write_digit) * r.outp + write_digit * ((1 - write_weighting) * r.outp + write_weighting * digit)

        t = r.t + 1

        with tf.name_scope("HardAddition"):
            new_registers = self.register_spec.wrap(
                inp=tf.identity(r.inp, "inp"),
                outp=tf.identity(output, "outp"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                wm1=tf.identity(wm1, "wm1"),
                wm2=tf.identity(wm2, "wm2"),
                carry=tf.identity(carry, "carry"),
                digit=tf.identity(digit, "digit"),
                t=tf.identity(t, "t"))

        return new_registers


def train(log_dir, config, seed=-1):
    config.seed = config.seed if seed < 0 else seed
    np.random.seed(config.seed)

    base_kwargs = dict(n_train=config.n_train, n_val=config.n_val, n_test=config.n_test)

    def build_env(**kwargs):
        return HardAdditionEnv(**kwargs)

    def build_core_network(env):
        return HardAddition(env)

    def build_policy(cn, exploration):
        config = default_config()
        return Policy(
            config.controller_func(cn.n_actions), config.action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="addition_policy")

    curriculum = ProductionSystemCurriculum(
        base_kwargs, config.curriculum, config.updater_class, build_env, build_core_network, build_policy)

    exp_name = "selection={}_updater={}".format(
        config.action_selection.__class__.__name__, config.updater_class.__name__)
    training_loop(curriculum, log_dir, config, exp_name=exp_name)


def visualize(config):
    from dps.production_system import ProductionSystem
    from dps.policy import IdentitySelect
    from dps.utils import build_decaying_value, FixedController

    def build_psystem():
        _config = default_config()
        height = _config.curriculum[0]['height']
        width = _config.curriculum[0]['width']
        n_digits = _config.curriculum[0]['n_digits']
        env = HardAdditionEnv(height, width, n_digits, 10, 10, 10)
        cn = HardAddition(env)

        # controller = FixedController(list(range(cn.n_actions)), cn.n_actions)
        # This is a perfect execution of the algo for width == height == 2.
        controller = FixedController([4, 2, 5, 6, 7, 0, 4, 3, 5, 6, 7, 0], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.exploration_schedule, 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="addition_policy")
        return ProductionSystem(env, cn, policy, False, 12)
        # return ProductionSystem(env, cn, policy, False, cn.n_actions)

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 1, False)
