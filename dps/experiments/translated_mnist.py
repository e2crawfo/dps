from collections import namedtuple
from itertools import product

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import default_config
from dps.production_system import ProductionSystemCurriculum
from dps.train import training_loop, build_and_visualize
from dps.policy import Policy
from dps.experiments.mnist import TranslatedMnistDataset


class TranslatedMnistEnv(RegressionEnv):
    def __init__(self, W, n_digits, n_train, n_val, n_test):
        self.W = W
        self.n_digits = n_digits
        super(TranslatedMnistEnv, self).__init__(
            train=TranslatedMnistDataset(W, n_digits, n_train, for_eval=False),
            val=TranslatedMnistDataset(W, n_digits, n_val, for_eval=True),
            test=TranslatedMnistDataset(W, n_digits, n_test, for_eval=True))

    def __str__(self):
        return "<TranslatedMnistEnv W={} n_digits={}>".format(self.W, self.n_digits)


# Define at top-level to enable pickling
translated_mnist_nt = namedtuple('TranslatedMnistRegister', 'inp outp fovea_x fovea_y vision t'.split())


class TranslatedMnistRegSpec(RegisterSpec):
    _visible = [0, 0] + [1] * 8
    _initial_values = None
    _namedtuple = translated_mnist_nt
    _input_names = ['inp']
    _output_names = ['outp']

    def __init__(self, W):
        self.W = W

        zero = np.ones(self.n_digits)
        zero /= zero.sum()

        self._initial_values = (
            [np.zeros(W*W, dtype='f'), zero.copy(), np.zeros(1), np.zeros(1), zero.copy(), np.zeros(1)])
        super(TranslatedMnistRegSpec, self).__init__()


class TranslatedMnist(CoreNetwork):
    """ Top left is (x=0, y=0). """
    action_names = ['fovea_x += 1', 'fovea_x -= 1', 'fovea_y += 1', 'fovea_y -= 1', 'store', 'no-op/stop']

    def __init__(self, env):
        self.W = env.W
        self.register_spec = TranslatedMnistRegSpec(env.W)
        super(TranslatedMnist, self).__init__()

    def __call__(self, action_activations, r):
        (inc_fovea_x, dec_fovea_x, inc_fovea_y, dec_fovea_y, store, no_op) = (
            tf.split(action_activations, self.n_actions, axis=1))

        fovea_x = (1 - inc_fovea_x - dec_fovea_x) * r.fovea_x + inc_fovea_x * (r.fovea_x + 1) + dec_fovea_x * (r.fovea_x - 1)
        fovea_y = (1 - inc_fovea_y - dec_fovea_y) * r.fovea_y + inc_fovea_y * (r.fovea_y + 1) + dec_fovea_y * (r.fovea_y - 1)
        outp = (1 - store) * r.outp + store * r.vision

        fovea = tf.concat([fovea_y, fovea_x], 1)
        locations = np.array(list(product(range(self.height), range(self.W))), dtype='f')
        vision = None

        t = r.t + 1

        with tf.name_scope("TranslatedMnist"):
            new_registers = self.register_spec.wrap(
                inp=tf.identity(r.inp, "inp"),
                outp=tf.identity(outp, "outp"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                t=tf.identity(t, "t"))

        return new_registers


def train(log_dir, config, seed=-1):
    config.seed = config.seed if seed < 0 else seed
    np.random.seed(config.seed)

    base_kwargs = dict(n_train=config.n_train, n_val=config.n_val, n_test=config.n_test)

    def build_env(**kwargs):
        return TranslatedMnistEnv(**kwargs)

    def build_core_network(env):
        return TranslatedMnist(env)

    def build_policy(cn, exploration):
        config = default_config()
        return Policy(
            config.controller_func(cn.n_actions), config.action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="translated_mnist_policy")

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
        W = _config.curriculum[0]['W']
        n_digits = _config.curriculum[0]['n_digits']
        env = TranslatedMnistEnv(height, W, n_digits, 10, 10, 10)
        cn = TranslatedMnist(env)

        # controller = FixedController(list(range(cn.n_actions)), cn.n_actions)
        # This is a perfect execution of the algo for W == height == 2.
        controller = FixedController([4, 2, 5, 6, 7, 0, 4, 3, 5, 6, 7, 0], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.exploration_schedule, 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="translated_mnist_policy")
        return ProductionSystem(env, cn, policy, False, 12)
        # return ProductionSystem(env, cn, policy, False, cn.n_actions)

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 1, False)
