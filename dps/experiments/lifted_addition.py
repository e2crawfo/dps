from collections import namedtuple

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config
from dps.attention import gaussian_filter
from dps.production_system import ProductionSystemCurriculum
from dps.train import training_loop, build_and_visualize
from dps.policy import Policy
from dps.lift import lift_binary


def one_hot(inp, depth):
    """ numpy version of tf.one_hot. """
    new = np.zeros(inp.shape + (depth,))
    for idx in np.ndindex(inp.shape):
        new[idx + (inp[idx],)] = 1.0
    return new


class LiftedAdditionDataset(RegressionDataset):
    def __init__(
            self, width, n_digits, n_examples, for_eval=False, shuffle=True):
        if n_examples == 0:
            x = np.zeros((0, n_digits * (2*width+1)))
            y = np.zeros((0, 2*n_digits-1))
        else:
            self.width = width
            self.n_digits = n_digits

            _x = np.random.randint(0, n_digits, size=(n_examples, 2*width+1))
            _y = _x[:, :1] + _x[:, -1:]

            x = one_hot(_x, n_digits).reshape(_x.shape[0], -1)
            y = one_hot(_y, 2*n_digits-1).reshape(_y.shape[0], -1)

        super(LiftedAdditionDataset, self).__init__(x, y, for_eval, shuffle)


class LiftedAdditionEnv(RegressionEnv):
    def __init__(self, width, n_digits, n_train, n_val, n_test):
        self.width = width
        self.n_digits = n_digits
        super(LiftedAdditionEnv, self).__init__(
            train=LiftedAdditionDataset(width, n_digits, n_train, for_eval=False),
            val=LiftedAdditionDataset(width, n_digits, n_val, for_eval=True),
            test=LiftedAdditionDataset(width, n_digits, n_test, for_eval=True))

    def build_loss(self, policy_output):
        target_ph = tf.placeholder(tf.float32, shape=policy_output.shape, name='target')
        n_outputs = int(policy_output.shape[1])
        x, y = np.meshgrid(range(n_outputs), range(n_outputs), indexing='xy')
        distances = np.expand_dims((x - y)**2, 0)
        probs = tf.matmul(tf.expand_dims(policy_output, 2), tf.expand_dims(target_ph, 1))
        weighted_probs = distances * probs
        expected_distance = tf.reduce_sum(weighted_probs, [-2, -1])
        return expected_distance, target_ph

    def __str__(self):
        return "<LiftedAdditionEnv width={} n_digits={}>".format(self.width, self.n_digits)


# Define at top-level to enable pickling
_lifted_addition_nt = namedtuple('LiftedAdditionRegister', 'inp fovea t vision wm1 wm2 output'.split())


class lifted_addition_nt(_lifted_addition_nt):
    def __str__(self):
        s = [self.__class__.__name__ + "("]
        for f in self._fields:
            s.append("    {}={},".format(f, getattr(self, f)))
        s.append(")")
        return '\n'.join(s)

    def __repr__(self):
        return str(self)


class LiftedAdditionRegSpec(RegisterSpec):
    _visible = [0, 1, 1, 1, 1, 1, 1]
    _initial_values = None
    _namedtuple = lifted_addition_nt
    _input_names = ['inp']
    _output_names = ['output']

    def __init__(self, width, n_digits):
        self.width = width
        self.n_digits = n_digits

        zero = np.ones(self.n_digits)
        zero /= zero.sum()

        output_zero = np.ones(2*self.n_digits-1)
        output_zero /= output_zero.sum()

        self._initial_values = (
            [np.zeros((2*width+1)*self.n_digits, dtype='f')] +
            [np.array([v], dtype='f') for v in [0.0, 0.0]] +
            [zero.copy() for i in range(3)] +
            [output_zero])

        super(LiftedAdditionRegSpec, self).__init__()


class LiftedAddition(CoreNetwork):
    action_names = ['fovea += 1', 'fovea -= 1', 'wm1 = vision', 'wm2 = vision', 'output = wm1 + wm2', 'no-op/stop']

    def __init__(self, env):
        self.width = env.width
        self.n_digits = env.n_digits
        self.register_spec = LiftedAdditionRegSpec(env.width, env.n_digits)
        super(LiftedAddition, self).__init__()

    def __call__(self, action_activations, r):
        inc_fovea, dec_fovea, vision_to_wm1, vision_to_wm2, add, no_op = (
            tf.split(action_activations, self.n_actions, axis=1))

        fovea = (1 - inc_fovea - dec_fovea) * r.fovea + inc_fovea * (r.fovea + 1) + dec_fovea * (r.fovea - 1)
        wm1 = (1 - vision_to_wm1) * r.wm1 + vision_to_wm1 * r.vision
        wm2 = (1 - vision_to_wm2) * r.wm2 + vision_to_wm2 * r.vision

        std = tf.fill(tf.shape(fovea), 0.1)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f'))
        filt = gaussian_filter(fovea, std, locations)
        filt = tf.squeeze(filt, axis=[1])
        inp = tf.reshape(r.inp, (-1, 2*self.width+1, self.n_digits))
        vision = tf.reduce_sum(tf.expand_dims(filt, -1) * inp, 1)
        l1_norm = tf.reduce_sum(vision, axis=1, keep_dims=True)
        vision = l1_norm * vision + (1 - l1_norm) * (1 / self.n_digits) * tf.ones_like(vision)

        add_output, _ = lift_binary(
            lambda x, y: x + y, range(self.n_digits), range(self.n_digits),
            range(2*self.n_digits-1), r.wm1, r.wm2)

        output = (1 - add) * r.output + add * add_output
        t = r.t + 1

        new_registers = self.register_spec.wrap(inp=r.inp, fovea=fovea, t=t, vision=vision, wm1=wm1, wm2=wm2, output=output)

        return new_registers


def train(log_dir, config, seed=-1):
    config.seed = config.seed if seed < 0 else seed
    np.random.seed(config.seed)

    base_kwargs = dict(n_train=config.n_train, n_val=config.n_val, n_test=config.n_test)

    def build_env(**kwargs):
        return LiftedAdditionEnv(**kwargs)

    def build_core_network(env):
        return LiftedAddition(env)

    def build_policy(cn, exploration):
        config = default_config()
        return Policy(
            config.controller_func(cn.n_actions), config.action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="lifted_addition_policy")

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
        width = _config.curriculum[0]['width']
        n_digits = _config.curriculum[0]['n_digits']
        env = LiftedAdditionEnv(width, n_digits, 10, 10, 10)
        cn = LiftedAddition(env)

        # controller = FixedController([0, 2, 1, 1, 3, 4, 0], cn.n_actions)
        controller = FixedController([0, 1, 1, 1, 2, 4, 2], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.exploration_schedule, 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="lifted_addition_policy")
        return ProductionSystem(env, cn, policy, False, 7)

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 1, False)
