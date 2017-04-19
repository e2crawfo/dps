from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionDataset, RegressionEnv
from dps.utils import Config, default_config, CompositeCell, get_config
from dps.production_system import ProductionSystemCurriculum
from dps.train import training_loop
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
        self.width = width
        self.n_digits = n_digits
        super(PointerEnv, self).__init__(
            train=PointerDataset(width, n_digits, n_train, for_eval=False),
            val=PointerDataset(width, n_digits, n_val, for_eval=True),
            test=PointerDataset(width, n_digits, n_test, for_eval=True))

    def __str__(self):
        return "<PointerEnv width={} n_digits={}>".format(self.width, self.n_digits)


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

    def __init__(self, env):
        super(Pointer, self).__init__()
        self.width = env.width
        self._register_spec = PointerRegSpec(env.width)

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


class DefaultConfig(Config):
    seed = 12

    T = 3

    optimizer_class = tf.train.RMSPropOptimizer

    max_steps = 10000
    batch_size = 100
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-2
    patience = np.inf

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
        ReluSelect()][1])

    use_rl = False

    curriculum = [dict(width=1, n_digits=2)]

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.96, False)
    noise_schedule = (0.0, 100, 0.96, False)
    exploration_schedule = (10.0, 100, 0.96, False)

    max_grad_norm = 0.0
    l2_norm_param = 0.0
    gamma = 1.0

    debug = False


class CurriculumConfig(DefaultConfig):
    curriculum = [
        dict(width=1, n_digits=2),
        dict(width=1, n_digits=3),
        dict(width=1, n_digits=4)]

    # curriculum = [
    #     dict(width=1, n_digits=4),
    #     dict(width=2, n_digits=4),
    #     dict(width=3, n_digits=4)]


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


def train_pointer(log_dir, config="default", seed=-1):
    config = get_config(__file__, config)
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
            cn.n_actions+1, cn.obs_dim, name="pointer_policy")

    curriculum = ProductionSystemCurriculum(
        base_kwargs, config.curriculum, build_env, build_core_network, build_policy)

    training_loop(curriculum, log_dir, config)


if __name__ == '__main__':
    from clify import command_line
    command_line(train_pointer)(log_dir='/tmp/dps/pointer')
