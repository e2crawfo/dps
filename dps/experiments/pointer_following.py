from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
import numpy as np

from dps import (
    ProductionSystem, ProductionSystemFunction, ProductionSystemEnv,
    CoreNetwork, RegisterSpec, DifferentiableUpdater)
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import Config, default_config, CompositeCell
from dps.train import training_loop
from dps.rl import REINFORCE
from dps.policy import Policy, ReluSelect, SoftmaxSelect, GumbelSoftmaxSelect


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
        super(PointerEnv, self).__init__(
            train=PointerDataset(width, n_digits, n_train, for_eval=False),
            val=PointerDataset(width, n_digits, n_val, for_eval=True),
            test=PointerDataset(width, n_digits, n_test, for_eval=True))


# Define at top-level to enable pickling
pointer_nt = namedtuple('_PointerRegister', 'inp fovea vision wm'.split())


class PointerRegSpec(RegisterSpec):
    _visible = [0, 1, 1, 1]
    _initial_values = None
    _namedtuple = pointer_nt
    _input_names = ['inp']
    _output_names = ['wm']

    def __init__(self, width):
        self.width = width
        self._initial_values = (
            [np.zeros(2*width+1, dtype='f')] +
            [np.array([v], dtype='f') for v in [width, 0.0, 0.0]])


class Pointer(CoreNetwork):
    _register_spec = None
    _n_actions = 5

    def __init__(self, width):
        super(Pointer, self).__init__()
        self.width = width
        self._register_spec = PointerRegSpec(width)

    def __call__(self, action_activations, r):
        """ Actions:
            * +/- fovea
            * +/- wm
            * store vision in wm

        """
        a0, a1, a2, a3, a4 = tf.split(action_activations, self.n_actions, axis=1)
        fovea = (1 - a0 - a1) * r.fovea + a0 * (r.fovea + 1) + a1 * (r.fovea - 1)
        wm = (1 - a2 - a3 - a4) * r.wm + a2 * (r.wm + 1) + a3 * (r.wm - 1) + a4 * r.vision

        filt = tf.contrib.distributions.Normal(fovea, 0.4)
        filt = filt.pdf(np.linspace(-self.width, self.width, 2*self.width+1, dtype='f'))
        l1_norm = tf.reduce_sum(tf.abs(filt), axis=1, keep_dims=True)
        filt = filt / l1_norm
        vision = tf.reduce_sum(r.inp * filt, axis=1, keep_dims=True)

        new_registers = self.register_spec.wrap(inp=r.inp, fovea=fovea, vision=vision, wm=wm)

        return new_registers


def _build_psystem(global_step):
    config = default_config()

    pointer = Pointer(config.width)

    start, decay_steps, decay_rate, staircase = config.exploration_schedule
    exploration = tf.train.exponential_decay(
        start, global_step, decay_steps, decay_rate, staircase=staircase)
    tf.summary.scalar('exploration', exploration)

    policy = Policy(
        config.controller, config.action_selection, exploration,
        pointer.n_actions+1, pointer.obs_dim, name="pointer_policy")

    psystem = ProductionSystem(pointer, policy, False, config.T)
    return psystem


class DefaultConfig(Config):
    seed = 10

    width = 1
    n_digits = 2
    T = 4

    optimizer_class = tf.train.RMSPropOptimizer

    max_steps = 10000
    batch_size = 100
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-3
    patience = 100

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    controller = CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=64),
        fully_connected,
        Pointer._n_actions+1)

    action_selection = staticmethod([
        SoftmaxSelect(),
        GumbelSoftmaxSelect(hard=0),
        GumbelSoftmaxSelect(hard=1),
        ReluSelect()][0])

    use_rl = False

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.96, False)
    noise_schedule = (0.0, 100, 0.96, False)
    exploration_schedule = (10.0, 100, 0.96, False)

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


class DebugConfig(DefaultConfig):
    debug = True

    max_steps = 100
    n_train = 2
    batch_size = 2
    eval_step = 1
    display_step = 1
    checkpoint_step = 1
    exploration_schedule = (0.5, 100, 0.96, False)


class RLConfig(DefaultConfig):
    use_rl = True
    threshold = 1e-2

    action_selection = SoftmaxSelect()

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.96, False)
    noise_schedule = (0.0, 100, 0.96, False)
    exploration_schedule = (10.0, 1000, 0.96, False)


class RLDebugConfig(DebugConfig, RLConfig):
    pass


def get_config(name):
    try:
        return dict(
            default=DefaultConfig(),
            rl=RLConfig(),
            debug=DebugConfig(),
            rl_debug=RLDebugConfig(),
        )[name]
    except KeyError:
        raise KeyError("Unknown config name {}.".format(name))


def train_pointer(log_dir, config="default", seed=-1):
    config = get_config(config)
    config.seed = config.seed if seed < 0 else seed
    np.random.seed(config.seed)

    env = PointerEnv(config.width, config.n_digits, config.n_train, config.n_val, config.n_test)

    def build_diff_updater():
        global_step = tf.contrib.framework.get_or_create_global_step()
        psystem = _build_psystem(global_step)
        ps_func = ProductionSystemFunction(psystem)
        return DifferentiableUpdater(
            env, ps_func, global_step, config.optimizer_class,
            config.lr_schedule, config.noise_schedule, config.max_grad_norm)

    def build_reinforce_updater():
        global_step = tf.contrib.framework.get_or_create_global_step()
        psystem = _build_psystem(global_step)
        ps_env = ProductionSystemEnv(psystem, env)
        return REINFORCE(
            ps_env, psystem.policy, global_step, config.optimizer_class,
            config.lr_schedule, config.noise_schedule, config.max_grad_norm,
            config.gamma, config.l2_norm_param)

    build_updater = build_reinforce_updater if config.use_rl else build_diff_updater
    training_loop(env, build_updater, log_dir, config)


if __name__ == '__main__':
    from clify import command_line
    command_line(train_pointer)(log_dir='/tmp/dps/pointer')
