from collections import namedtuple

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config, visual_filter_one_d
from dps.production_system import ProductionSystemCurriculum
from dps.train import training_loop, build_and_visualize
from dps.policy import Policy


class AdditionDataset(RegressionDataset):
    def __init__(
            self, width, n_digits, n_examples, for_eval=False, shuffle=True):
        self.width = width
        self.n_digits = n_digits

        x = np.random.randint(0, n_digits, size=(n_examples, 2*width+1))
        y = x[:, :1] + x[:, -1:]
        super(AdditionDataset, self).__init__(x, y, for_eval, shuffle)


class AdditionEnv(RegressionEnv):
    def __init__(self, width, n_digits, n_train, n_val, n_test):
        self.width = width
        self.n_digits = n_digits
        super(AdditionEnv, self).__init__(
            train=AdditionDataset(width, n_digits, n_train, for_eval=False),
            val=AdditionDataset(width, n_digits, n_val, for_eval=True),
            test=AdditionDataset(width, n_digits, n_test, for_eval=True))

    def __str__(self):
        return "<AdditionEnv width={} n_digits={}>".format(self.width, self.n_digits)


# Define at top-level to enable pickling
addition_nt = namedtuple('AdditionRegister', 'inp fovea vision wm1 wm2 output t'.split())


class AdditionRegSpec(RegisterSpec):
    _visible = [0, 1, 1, 1, 1, 1, 1]
    _initial_values = None
    _namedtuple = addition_nt
    _input_names = ['inp']
    _output_names = ['output']

    def __init__(self, width):
        self.width = width
        self._initial_values = (
            [np.zeros(2*width+1, dtype='f')] +
            [np.array([v], dtype='f') for v in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


class Addition(CoreNetwork):
    action_names = ['fovea += 1', 'fovea -= 1', 'wm1 = vision', 'wm2 = vision',
                    'output = vision', 'output = wm1 + wm2', 'no-op/stop']

    def __init__(self, env):
        self.width = env.width
        self.register_spec = AdditionRegSpec(env.width)
        super(Addition, self).__init__()

    def __call__(self, action_activations, r):
        inc_fovea, dec_fovea, vision_to_wm1, vision_to_wm2, vision_to_output, add, no_op = (
            tf.split(action_activations, self.n_actions, axis=1))

        fovea = (1 - inc_fovea - dec_fovea) * r.fovea + inc_fovea * (r.fovea + 1) + dec_fovea * (r.fovea - 1)
        wm1 = (1 - vision_to_wm1) * r.wm1 + vision_to_wm1 * r.vision
        wm2 = (1 - vision_to_wm2) * r.wm2 + vision_to_wm2 * r.vision
        output = (1 - vision_to_output - add) * r.output + vision_to_output * r.vision + add * (r.wm1 + r.wm2)
        t = r.t + 1

        vision = visual_filter_one_d(self.width, r.inp, r.fovea, 0.01)
        new_registers = self.register_spec.wrap(inp=r.inp, fovea=fovea, vision=vision, wm1=wm1, wm2=wm2, output=output, t=t)

        return new_registers


def train(log_dir, config, seed=-1):
    config.seed = config.seed if seed < 0 else seed
    np.random.seed(config.seed)

    base_kwargs = dict(n_train=config.n_train, n_val=config.n_val, n_test=config.n_test)

    def build_env(**kwargs):
        return AdditionEnv(**kwargs)

    def build_core_network(env):
        return Addition(env)

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
        width = _config.curriculum[0]['width']
        n_digits = _config.curriculum[0]['n_digits']
        env = AdditionEnv(width, n_digits, 10, 10, 10)
        cn = Addition(env)

        controller = FixedController([0, 2, 1, 1, 3, 5, 0], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.exploration_schedule, 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="addition_policy")
        return ProductionSystem(env, cn, policy, False, 7)

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 1, False)
