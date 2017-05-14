from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import animation, patches
import os
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
from dps.attention import DRAW_attention_2D


class TranslatedMnistEnv(RegressionEnv):
    def __init__(self, W, N, n_train, n_val, n_test, inc_delta, inc_x, inc_y):
        self.W = W
        self.N = N
        self.inc_delta = inc_delta
        self.inc_x = inc_x
        self.inc_y = inc_y
        n_digits = 1
        max_overlap = 200
        super(TranslatedMnistEnv, self).__init__(
            train=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_train, for_eval=False),
            val=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_val, for_eval=True),
            test=mnist.TranslatedMnistDataset(W, n_digits, max_overlap, n_test, for_eval=True))

    def __str__(self):
        return "<TranslatedMnistEnv W={}>".format(self.W)

    def _render(self, mode='human', close=False):
        pass


def build_classifier(inp):
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(inp, 10)
    return tf.nn.softmax(logits)


class MnistDrawPretrained(object):
    """ A wrapper around a classifier that initializes it with values stored on disk. """
    def __init__(
            self, build_classifier, N, var_scope_name='mnist', freeze_weights=True,
            model_dir='/tmp/dps_mnist/', name='model.chk', config=None):

        self._build_classifier = build_classifier
        self.N = N
        self.var_scope_name = var_scope_name
        self.var_scope = None
        self.model_dir = model_dir
        self.name = name
        self.path = os.path.join(model_dir, name)
        self.config = config
        self.n_builds = 0
        self.freeze_weights = freeze_weights

    def build_classifier(self, inp):
        """ Returns class probabilities given a glimpse. """
        if len(inp.shape) == 3:
            inp = tf.reshape(inp, (tf.shape(inp)[0], int(inp.shape[1]) * int(inp.shape[2])))
        return self._build_classifier(inp)

    def build_draw_plus_classifier(self, inp, fovea_x=None, fovea_y=None, delta=None, sigma=None):
        """ Returns class probabilities and a glimpse given raw image, but doesn't load from disk. """
        if self.N:
            if len(inp.shape) == 2:
                s = int(np.sqrt(int(inp.shape[1])))
                inp = tf.reshape(inp, (-1, s, s))

            batch_size = tf.shape(inp)[0]
            if fovea_x is None:
                fovea_x = tf.zeros((batch_size, 1))
            if fovea_y is None:
                fovea_y = tf.zeros((batch_size, 1))
            if delta is None:
                delta = tf.ones((batch_size, 1))
            if sigma is None:
                sigma = tf.ones((batch_size, 1))

            glimpse = DRAW_attention_2D(
                inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta,
                std=tf.ones((batch_size, 1)), N=self.N)
        else:
            glimpse = inp

        return self.build_classifier(glimpse), glimpse

    def build_pretrained(self, inp, **build_kwargs):
        """ Adds a draw layer and a classifier pretrained from data. """
        if self.n_builds == 0:
            with tf.variable_scope(self.var_scope_name, reuse=False) as var_scope:
                outp, glimpse = self.build_draw_plus_classifier(inp, **build_kwargs)
                if self.freeze_weights:
                    outp = tf.stop_gradient(outp)

            self.var_scope = var_scope

            # Initializes its own variables by loading from a file or training separately
            mnist.load_or_train(
                tf.get_default_session(), self, var_scope, self.path, self.config)
            self.n_builds += 1
        else:
            with tf.variable_scope(self.var_scope, reuse=True) as var_scope:
                outp, glimpse = self.build_draw_plus_classifier(inp, **build_kwargs)
                if self.freeze_weights:
                    outp = tf.stop_gradient(outp)

        return outp, glimpse

    def __call__(self, inp):
        inference, glimpse = self.build_draw_plus_classifier(inp)
        return inference


# Define at top-level to enable pickling
translated_mnist_nt = namedtuple('TranslatedMnistRegister', 'inp outp glimpse fovea_x fovea_y vision delta t'.split())


class TranslatedMnistRegSpec(RegisterSpec):
    _visible = [0, 1, 1, 1, 1, 1, 1, 1]
    _initial_values = None
    _namedtuple = translated_mnist_nt
    _input_names = ['inp']
    _output_names = ['outp']
    omit = ['inp', 'glimpse']

    def __init__(self, W, N):
        self.W = W
        self.N = N

        self._initial_values = [
            np.zeros(W*W, dtype='f'), np.array([0.0]), np.zeros(N*N, dtype='f'), np.array([0.0]),
            np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0])]
        super(TranslatedMnistRegSpec, self).__init__()


class TranslatedMnist(CoreNetwork):
    """ Top left is (y=0, x=0). Corresponds to using origin='upper' in plt.imshow. """

    action_names = [
        'fovea_x += ', 'fovea_x -= ', 'fovea_x ++= ', 'fovea_x --= ',
        'fovea_y += ', 'fovea_y -= ', 'fovea_y ++= ', 'fovea_y --= ',
        'delta += ', 'delta -= ', 'delta ++= ', 'delta --= ',
        # 'sigma += ', 'sigma -= ',
        'store', 'no-op/stop']

    def __init__(self, env):
        self.W = env.W
        self.N = env.N
        self.inc_delta = env.inc_delta
        self.inc_x = env.inc_x
        self.inc_y = env.inc_y

        build_classifier = default_config().build_classifier
        classifier_str = default_config().classifier_str
        self.classifier = MnistDrawPretrained(build_classifier, self.N, name='{}_N={}.chk'.format(classifier_str, self.N))
        self.register_spec = TranslatedMnistRegSpec(env.W, env.N)
        super(TranslatedMnist, self).__init__()

    def __call__(self, action_activations, r):
        (inc_fovea_x, dec_fovea_x, inc_fovea_x_big, dec_fovea_x_big,
         inc_fovea_y, dec_fovea_y, inc_fovea_y_big, dec_fovea_y_big,
         inc_delta, dec_delta, inc_delta_big, dec_delta_big,
         # inc_sigma,dec_sigma,
         store, no_op) = (
            tf.split(action_activations, self.n_actions, axis=1))
        # (inc_fovea_x, dec_fovea_x, inc_fovea_y, dec_fovea_y,
        #  inc_delta, dec_delta, inc_sigma, dec_sigma, store, no_op) = (
        #     tf.split(action_activations, self.n_actions, axis=1))

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

        outp = (1 - store) * r.outp + store * r.vision

        inp = tf.reshape(r.inp, (-1, self.W, self.W))
        classification, glimpse = self.classifier.build_pretrained(
            inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, sigma=1.0)

        # batch_size = tf.shape(classification)[0]
        # _vision = classification * tf.tile(tf.expand_dims(tf.range(10, dtype=tf.float32), 0), (batch_size, 1))
        # vision = tf.reduce_sum(_vision, 1, keep_dims=True)
        vision = tf.cast(tf.expand_dims(tf.argmax(classification, 1), 1), tf.float32)

        t = r.t + 1

        with tf.name_scope("TranslatedMnist"):
            new_registers = self.register_spec.wrap(
                inp=tf.identity(r.inp, "inp"),
                outp=tf.identity(outp, "outp"),
                glimpse=tf.reshape(glimpse, (-1, self.N*self.N), name="glimpse"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                delta=tf.identity(delta, "delta"),
                # sigma=tf.identity(sigma, "sigma"),
                t=tf.identity(t, "t"))

        return new_registers


def render_rollouts(psystem, actions, registers, reward, external_step_lengths):
    """ Render rollouts from TranslatedMnist task. """
    n_timesteps, batch_size, n_actions = actions.shape
    s = int(np.ceil(np.sqrt(batch_size)))

    fig, subplots = plt.subplots(2*s, s)

    env_subplots = subplots[::2, :].flatten()
    glimpse_subplots = subplots[1::2, :].flatten()

    W = psystem.core_network.W
    N = psystem.core_network.N

    raw_images = registers.inp[0].reshape((-1, W, W))

    [ax.imshow(raw_img, cmap='gray', origin='upper') for raw_img, ax in zip(raw_images, env_subplots)]

    rectangles = [
        ax.add_patch(patches.Rectangle(
            (0.05, 0.05), 0.9, 0.9, alpha=0.6, transform=ax.transAxes))
        for ax in env_subplots]

    glimpses = [ax.imshow(np.random.randint(256, size=(N, N)), cmap='gray', origin='upper') for ax in glimpse_subplots]

    def animate(i):
        # Find locations of bottom-left in fovea co-ordinate system, then transform to axis co-ordinate system.
        delta = registers.delta[i, :, :]
        fx = registers.fovea_x[i, :, :] - delta
        fy = registers.fovea_y[i, :, :] + delta
        fx *= 0.5
        fy *= 0.5
        fy -= 0.5
        fx += 0.5
        fy *= -1

        # use delta and fovea to modify the rectangles
        for d, x, y, rect in zip(delta, fx, fy, rectangles):
            rect.set_x(x)
            rect.set_y(y)
            rect.set_width(d)
            rect.set_height(d)

        for g, gimg in zip(registers.glimpse[i, :, :], glimpses):
            gimg.set_data(g.reshape(N, N))

        return rectangles + glimpses

    _animation = animation.FuncAnimation(fig, animate, n_timesteps, blit=True, interval=1000, repeat=True)

    if default_config().save_display:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        _animation.save(str(Path(default_config().path) / 'animation.mp4'), writer=writer)

    if default_config().display:
        plt.show()


def visualize(config):
    from dps.production_system import ProductionSystem
    from dps.policy import IdentitySelect
    from dps.utils import build_decaying_value, FixedController

    def build_psystem():
        _config = default_config()
        W = 60
        N = 14
        env = TranslatedMnistEnv(W, N, 10, 10, 10, inc_delta=0.1, inc_x=0.1, inc_y=0.1)
        cn = TranslatedMnist(env)

        controller = FixedController(list(range(cn.n_actions)), cn.n_actions)
        # controller = FixedController([4, 2, 5, 6, 7, 0, 4, 3, 5, 6, 7, 0], cn.n_actions)
        # controller = FixedController([8], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.exploration_schedule, 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="translated_mnist_policy")
        return ProductionSystem(env, cn, policy, False, len(controller))

    with config.as_default():
        build_and_visualize(build_psystem, 'train', 16, False, render_rollouts=render_rollouts)


class TranslatedMnistTrainer(ProductionSystemTrainer):
    def build_env(self, **kwargs):
        return TranslatedMnistEnv(**kwargs)

    def build_core_network(self, env):
        return TranslatedMnist(env)
