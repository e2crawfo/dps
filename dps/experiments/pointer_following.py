from collections import namedtuple

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config
from dps.attention import apply_gaussian_filter
from dps.production_system import ProductionSystemTrainer
from dps.train import build_and_visualize
from dps.policy import Policy


class PointerDataset(RegressionDataset):
    def __init__(
            self, width, n_digits, n_examples, for_eval=False, shuffle=True):
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


# Define at top-level to enable pickling
pointer_nt = namedtuple('PointerRegister', 'inp fovea vision wm t'.split())


class PointerRegSpec(RegisterSpec):
    _visible = [0, 1, 1, 1, 1]
    _initial_values = None
    _namedtuple = pointer_nt
    _input_names = ['inp']
    _output_names = ['wm']

    def __init__(self, width):
        self.width = width
        self._initial_values = (
            [np.zeros(2*width+1, dtype='f')] +
            [np.array([v], dtype='f') for v in [0.0, 0.0, 0.0, 0.0]])
        super(PointerRegSpec, self).__init__()


class Pointer(CoreNetwork):
    action_names = ['fovea += 1', 'fovea -= 1', 'wm = vision', 'no-op/stop']

    def __init__(self, env):
        self.width = env.width
        self.register_spec = PointerRegSpec(env.width)
        super(Pointer, self).__init__()

    def __call__(self, action_activations, r):
        inc_fovea, dec_fovea, vision_to_wm, no_op = tf.split(action_activations, self.n_actions, axis=1)
        fovea = (1 - inc_fovea - dec_fovea) * r.fovea + inc_fovea * (r.fovea + 1) + dec_fovea * (r.fovea - 1)
        wm = (1 - vision_to_wm) * r.wm + vision_to_wm * r.vision
        t = r.t + 1

        diag_std = tf.fill(tf.shape(r.fovea), 0.01)
        locations = tf.constant(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f').reshape(-1, 1))
        vision = apply_gaussian_filter(r.fovea, diag_std, locations, r.inp)

        new_registers = self.register_spec.wrap(inp=r.inp, fovea=fovea, vision=vision, wm=wm, t=t)

        return new_registers


def visualize(config):
    from dps.production_system import ProductionSystem
    from dps.policy import IdentitySelect
    from dps.utils import build_decaying_value, FixedController

    def build_psystem():
        _config = default_config()
        width = _config.curriculum[0]['width']
        n_digits = _config.curriculum[0]['n_digits']
        env = PointerEnv(width, n_digits, 10, 10, 10)
        cn = Pointer(env)

        controller = FixedController([0, 2, 1], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.exploration_schedule, 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="pointer_policy")
        return ProductionSystem(env, cn, policy, False, 3)

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 1, False)


class PointerTrainer(ProductionSystemTrainer):
    def build_env(self, **kwargs):
        return PointerEnv(**kwargs)

    def build_core_network(self, env):
        return Pointer(env)
