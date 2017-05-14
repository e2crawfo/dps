from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from pathlib import Path

import tensorflow as tf
import numpy as np

from dps import CoreNetwork, RegisterSpec
from dps.environment import RegressionEnv
from dps.utils import default_config, MLP
from dps.production_system import ProductionSystemTrainer
from dps.train import build_and_visualize
from dps.policy import Policy
from dps.experiments import mnist
from dps.experiments.translated_mnist import MnistDrawPretrained, render_rollouts


class MnistAdditionEnv(RegressionEnv):
    def __init__(self, n_digits, W, N, n_train, n_val, n_test, inc_delta, inc_x, inc_y):
        self.n_digits = n_digits
        self.W = W
        self.N = N
        self.inc_delta = inc_delta
        self.inc_x = inc_x
        self.inc_y = inc_y
        max_overlap = 200
        super(MnistAdditionEnv, self).__init__(
            train=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_train, for_eval=False),
            val=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_val, for_eval=True),
            test=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_test, for_eval=True))

    def __str__(self):
        return "<MnistAdditionEnv W={}>".format(self.W)

    def _render(self, mode='human', close=False):
        pass


def build_classifier(inp):
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(inp, 10)
    return tf.nn.softmax(logits)


# Define at top-level to enable pickling
mnist_addition_nt = namedtuple('MnistAdditionRegister', 'inp outp glimpse wm1 wm2 fovea_x fovea_y vision delta t'.split())


class MnistAdditionRegSpec(RegisterSpec):
    _visible = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    _initial_values = None
    _namedtuple = mnist_addition_nt
    _input_names = ['inp']
    _output_names = ['outp']
    omit = ['inp', 'glimpse']

    def __init__(self, W, N):
        self.W = W
        self.N = N

        self._initial_values = [
            np.zeros(W*W, dtype='f'), np.array([0.0]), np.zeros(N*N, dtype='f'),
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0])]
        super(MnistAdditionRegSpec, self).__init__()


class MnistAddition(CoreNetwork):
    """ Top left is (y=0, x=0). Corresponds to using origin='upper' in plt.imshow. """

    action_names = [
        'fovea_x += ', 'fovea_x -= ', 'fovea_x ++= ', 'fovea_x --= ',
        'fovea_y += ', 'fovea_y -= ', 'fovea_y ++= ', 'fovea_y --= ',
        'delta += ', 'delta -= ', 'delta ++= ', 'delta --= ',
        'store_wm1', 'store_wm2', 'add', 'no-op/stop']

    def __init__(self, env):
        self.W = env.W
        self.N = env.N
        self.inc_delta = env.inc_delta
        self.inc_x = env.inc_x
        self.inc_y = env.inc_y

        build_classifier = default_config().build_classifier
        classifier_str = default_config().classifier_str
        self.classifier = MnistDrawPretrained(build_classifier, self.N, name='{}_N={}.chk'.format(classifier_str, self.N))
        self.register_spec = MnistAdditionRegSpec(env.W, env.N)
        super(MnistAddition, self).__init__()

    def __call__(self, action_activations, r):
        (inc_fovea_x, dec_fovea_x, inc_fovea_x_big, dec_fovea_x_big,
         inc_fovea_y, dec_fovea_y, inc_fovea_y_big, dec_fovea_y_big,
         inc_delta, dec_delta, inc_delta_big, dec_delta_big,
         store_wm1, store_wm2, add, no_op) = (
            tf.split(action_activations, self.n_actions, axis=1))

        fovea_x = (1 - inc_fovea_x - dec_fovea_x - inc_fovea_x_big - dec_fovea_x_big) * r.fovea_x + \
            inc_fovea_x * (r.fovea_x + self.inc_x) + \
            inc_fovea_x_big * (r.fovea_x + 5 * self.inc_x) + \
            dec_fovea_x * (r.fovea_x - self.inc_x) + \
            dec_fovea_x_big * (r.fovea_x - 5 * self.inc_x)

        fovea_y = (1 - inc_fovea_y - dec_fovea_y - inc_fovea_y_big - dec_fovea_y_big) * r.fovea_y + \
            inc_fovea_y * (r.fovea_y + self.inc_y) + \
            inc_fovea_y_big * (r.fovea_y + 5 * self.inc_y) + \
            dec_fovea_y * (r.fovea_y - self.inc_y) + \
            dec_fovea_y_big * (r.fovea_y - 5 * self.inc_y)

        delta = (1 - inc_delta - dec_delta - inc_delta_big - dec_delta_big) * r.delta + \
            inc_delta * (r.delta + self.inc_delta) + \
            inc_delta_big * (r.delta + 5 * self.inc_delta) + \
            dec_delta * (r.delta - self.inc_delta) + \
            dec_delta_big * (r.delta - 5 * self.inc_delta)

        wm1 = (1 - store_wm1) * r.wm1 + store_wm1 * r.vision
        wm2 = (1 - store_wm2) * r.wm2 + store_wm2 * r.vision

        outp = (1 - add) * r.outp + add * (wm1 + wm2)

        inp = tf.reshape(r.inp, (-1, self.W, self.W))
        classification, glimpse = self.classifier.build_pretrained(
            inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta)

        # batch_size = tf.shape(classification)[0]
        # _vision = classification * tf.tile(tf.expand_dims(tf.range(10, dtype=tf.float32), 0), (batch_size, 1))
        # vision = tf.reduce_sum(_vision, 1, keep_dims=True)
        vision = tf.cast(tf.expand_dims(tf.argmax(classification, 1), 1), tf.float32)

        t = r.t + 1

        with tf.name_scope("MnistAddition"):
            new_registers = self.register_spec.wrap(
                inp=tf.identity(r.inp, "inp"),
                outp=tf.identity(outp, "outp"),
                glimpse=tf.reshape(glimpse, (-1, self.N*self.N), name="glimpse"),
                wm1=tf.identity(wm1, "wm1"),
                wm2=tf.identity(wm2, "wm2"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                delta=tf.identity(delta, "delta"),
                t=tf.identity(t, "t"))

        return new_registers


def visualize(config):
    from dps.production_system import ProductionSystem
    from dps.policy import IdentitySelect
    from dps.utils import build_decaying_value, FixedController

    def build_psystem():
        _config = default_config()
        W = 60
        N = 14
        n_digits = 2
        env = MnistAdditionEnv(n_digits, W, N, 10, 10, 10, inc_delta=0.1, inc_x=0.1, inc_y=0.1)
        cn = MnistAddition(env)

        controller = FixedController(list(range(cn.n_actions)), cn.n_actions)
        # controller = FixedController([4, 2, 5, 6, 7, 0, 4, 3, 5, 6, 7, 0], cn.n_actions)
        # controller = FixedController([8], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.exploration_schedule, 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="mnist_addition_policy")
        return ProductionSystem(env, cn, policy, False, len(controller))

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 16, False, render_rollouts=render_rollouts)


class MnistAdditionTrainer(ProductionSystemTrainer):
    def build_env(self, **kwargs):
        return MnistAdditionEnv(**kwargs)

    def build_core_network(self, env):
        return MnistAddition(env)
