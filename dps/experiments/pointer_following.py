from collections import namedtuple

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import (
    Config, default_config, CompositeCell, FeedforwardCell, get_config, MLP)
from dps.production_system import ProductionSystemCurriculum
from dps.train import training_loop, build_and_visualize
from dps.policy import Policy, ReluSelect, SigmoidSelect, SoftmaxSelect, GumbelSoftmaxSelect
from dps.updater import DifferentiableUpdater


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
pointer_nt = namedtuple('_PointerRegister', 'inp fovea vision wm t'.split())


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


class Pointer(CoreNetwork):
    _n_actions = 6
    _action_names = ['fovea += 1', 'fovea -= 1', 'wm += 1', 'wm -= 1', 'wm = vision', 'no-op/stop']
    _register_spec = None

    def __init__(self, env):
        super(Pointer, self).__init__()
        self.width = env.width
        self._register_spec = PointerRegSpec(env.width)

    def __call__(self, action_activations, r):
        """ Actions:
            * +/- fovea
            * +/- wm
            * store vision in wm
            * no-op

        """
        a0, a1, a2, a3, a4, a5 = tf.split(action_activations, self.n_actions, axis=1)
        fovea = (1 - a0 - a1) * r.fovea + a0 * (r.fovea + 1) + a1 * (r.fovea - 1)
        wm = (1 - a2 - a3 - a4) * r.wm + a2 * (r.wm + 1) + a3 * (r.wm - 1) + a4 * r.vision
        t = r.t + 1

        filt = tf.contrib.distributions.Normal(fovea, 0.1)
        filt = filt.pdf(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f'))
        l1_norm = tf.reduce_sum(filt, axis=1, keep_dims=True)
        filt = filt / l1_norm
        filt = tf.where(tf.is_nan(filt), tf.fill(tf.shape(filt), 1/(2*self.width+1)), filt)
        vision = tf.reduce_sum(r.inp * filt, axis=1, keep_dims=True)

        new_registers = self.register_spec.wrap(inp=r.inp, fovea=fovea, vision=vision, wm=wm, t=t)

        return new_registers


class PointerConfig(Config):
    seed = 12

    T = 8
    curriculum = [dict(width=1, n_digits=2)]

    optimizer_class = tf.train.RMSPropOptimizer
    updater_class = DifferentiableUpdater

    max_steps = 10000
    batch_size = 10
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-2
    patience = np.inf

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    controller = FeedforwardCell(MLP([100, 100]), Pointer._n_actions)
    controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=32),
        MLP(),
        Pointer._n_actions)

    action_selection = staticmethod([
        SoftmaxSelect(),
        GumbelSoftmaxSelect(hard=0),
        GumbelSoftmaxSelect(hard=1),
        SigmoidSelect(),
        ReluSelect()][0])

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.99, False)
    noise_schedule = (0.0, 1000, 0.96, False)
    exploration_schedule = (10.0, 1000, 0.96, False)

    test_time_explore = None

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


def train(log_dir, config, seed=-1):
    config.seed = config.seed if seed < 0 else seed
    np.random.seed(config.seed)

    base_kwargs = dict(n_train=config.n_train, n_val=config.n_val, n_test=config.n_test)

    def build_env(**kwargs):
        return PointerEnv(**kwargs)

    def build_core_network(env):
        return Pointer(env)

    def build_policy(cn, exploration):
        config = default_config()
        return Policy(
            config.controller, config.action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="pointer_policy")

    curriculum = ProductionSystemCurriculum(
        base_kwargs, config.curriculum, config.updater_class, build_env, build_core_network, build_policy)

    exp_name = "selection={}_updater={}".format(
        config.action_selection.__class__.__name__, config.updater_class.__name__)
    training_loop(curriculum, log_dir, config, exp_name=exp_name)


def visualize():
    from dps.production_system import ProductionSystem
    from dps.policy import IdentitySelect
    from dps.utils import build_decaying_value, FixedController

    def build_psystem():
        config = default_config()
        env = PointerEnv(1, 2, 10, 10, 10)
        cn = Pointer(env)

        controller = FixedController([4, 0, 4], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(config.exploration_schedule, 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="pointer_policy")
        return ProductionSystem(env, cn, policy, False, 3)

    with PointerConfig().as_default():
        build_and_visualize(build_psystem, 'train', 1, False)


if __name__ == '__main__':
    from clify import command_line
    command_line(train)(log_dir='/tmp/dps/pointer', config=PointerConfig())
