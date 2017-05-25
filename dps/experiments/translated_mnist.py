import matplotlib.pyplot as plt
from matplotlib import animation, patches
from pathlib import Path

import tensorflow as tf
import numpy as np

from dps import CoreNetwork
from dps.register import RegisterBank
from dps.environment import RegressionEnv
from dps.utils import default_config
from dps.production_system import ProductionSystemTrainer
from dps.train import build_and_visualize
from dps.policy import Policy
from dps.mnist import TranslatedMnistDataset, DRAW, MnistPretrained, MnistConfig


class TranslatedMnistEnv(RegressionEnv):
    def __init__(self, scaled, W, N, n_train, n_val, n_test, inc_delta, inc_x, inc_y):
        self.scaled = scaled
        self.W = W
        self.N = N
        self.inc_delta = inc_delta
        self.inc_x = inc_x
        self.inc_y = inc_y
        n_digits = 1
        max_overlap = 200
        super(TranslatedMnistEnv, self).__init__(
            train=TranslatedMnistDataset(W, n_digits, max_overlap, n_train, for_eval=False),
            val=TranslatedMnistDataset(W, n_digits, max_overlap, n_val, for_eval=True),
            test=TranslatedMnistDataset(W, n_digits, max_overlap, n_test, for_eval=True))

    def __str__(self):
        return "<TranslatedMnistEnv W={}>".format(self.W)

    def _render(self, mode='human', close=False):
        pass


class TranslatedMnist(CoreNetwork):
    """ Top left is (y=0, x=0). Corresponds to using origin='upper' in plt.imshow. """

    action_names = [
        'fovea_x += ', 'fovea_x -= ', 'fovea_x ++= ', 'fovea_x --= ',
        'fovea_y += ', 'fovea_y -= ', 'fovea_y ++= ', 'fovea_y --= ',
        'delta += ', 'delta -= ', 'delta ++= ', 'delta --= ', 'store', 'no-op/stop']

    def __init__(self, env):
        self.W = env.W
        self.N = env.N
        self.scaled = env.scaled
        self.inc_delta = env.inc_delta
        self.inc_x = env.inc_x
        self.inc_y = env.inc_y

        build_classifier = default_config().build_classifier
        classifier_str = default_config().classifier_str

        self.build_attention = DRAW(self.N)

        config = MnistConfig()
        self.build_classifier = MnistPretrained(
            self.build_attention, build_classifier,
            var_scope_name='digit_classifier',
            name='{}_N={}.chk'.format(classifier_str, self.N),
            config=config)

        values = (
            [0., 0., 0., 0., 1., 0.] +
            [np.zeros(self.N * self.N, dtype='f')])

        self.register_bank = RegisterBank(
            'TranslatedMnistRB',
            'outp fovea_x fovea_y vision delta t glimpse', None, values=values,
            output_names='outp', no_display='glimpse')
        super(TranslatedMnist, self).__init__()

    @property
    def input_dim(self):
        return self.W * self.W

    @property
    def make_input_available(self):
        return True

    def init(self, r, inp):
        outp, fovea_x, fovea_y, vision, delta, t, glimpse = self.register_bank.as_tuple(r)

        glimpse = self.build_attention(inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, sigma=1.0)
        classification = tf.stop_gradient(self.build_classifier(glimpse, preprocess=False))
        vision = tf.cast(tf.expand_dims(tf.argmax(classification, 1), 1), tf.float32)

        with tf.name_scope("TranslatedMnist"):
            new_registers = self.register_bank.wrap(
                outp=tf.identity(outp, "outp"),
                glimpse=tf.identity(glimpse, "glimpse"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                delta=tf.identity(delta, "delta"),
                t=tf.identity(t, "t"))

        return new_registers

    def __call__(self, action_activations, r, inp):
        _outp, _fovea_x, _fovea_y, _vision, _delta, _t, _glimpse = self.register_bank.as_tuple(r)

        (inc_fovea_x, dec_fovea_x, inc_fovea_x_big, dec_fovea_x_big,
         inc_fovea_y, dec_fovea_y, inc_fovea_y_big, dec_fovea_y_big,
         inc_delta, dec_delta, inc_delta_big, dec_delta_big,
         store, no_op) = (
            tf.split(action_activations, self.n_actions, axis=1))

        if self.scaled:
            fovea_x = (1 - inc_fovea_x - dec_fovea_x) * _fovea_x + \
                inc_fovea_x * (_fovea_x + _delta) + dec_fovea_x * (_fovea_x - _delta)

            fovea_y = (1 - inc_fovea_y - dec_fovea_y) * _fovea_y + \
                inc_fovea_y * (_fovea_y + _delta) + dec_fovea_y * (_fovea_y - _delta)

            delta = (1 - inc_delta - dec_delta - inc_delta_big - dec_delta_big) * _delta + \
                inc_delta * (_delta + self.inc_delta) + \
                inc_delta_big * (_delta + 5 * self.inc_delta) + \
                dec_delta * (_delta - self.inc_delta) + \
                dec_delta_big * (_delta - 5 * self.inc_delta)
        else:
            fovea_x = (1 - inc_fovea_x - dec_fovea_x - inc_fovea_x_big - dec_fovea_x_big) * _fovea_x + \
                inc_fovea_x * (_fovea_x + self.inc_x) + \
                inc_fovea_x_big * (_fovea_x + 5 * self.inc_x) + \
                dec_fovea_x * (_fovea_x - self.inc_x) + \
                dec_fovea_x_big * (_fovea_x - 5 * self.inc_x)

            fovea_y = (1 - inc_fovea_y - dec_fovea_y - inc_fovea_y_big - dec_fovea_y_big) * _fovea_y + \
                inc_fovea_y * (_fovea_y + self.inc_y) + \
                inc_fovea_y_big * (_fovea_y + 5 * self.inc_y) + \
                dec_fovea_y * (_fovea_y - self.inc_y) + \
                dec_fovea_y_big * (_fovea_y - 5 * self.inc_y)

            delta = (1 - inc_delta - dec_delta - inc_delta_big - dec_delta_big) * _delta + \
                inc_delta * (_delta + self.inc_delta) + \
                inc_delta_big * (_delta + 5 * self.inc_delta) + \
                dec_delta * (_delta - self.inc_delta) + \
                dec_delta_big * (_delta - 5 * self.inc_delta)

        outp = (1 - store) * _outp + store * _vision

        glimpse = self.build_attention(inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, sigma=1.0)
        classification = self.build_classifier(glimpse, preprocess=False)
        vision = tf.cast(tf.expand_dims(tf.argmax(classification, 1), 1), tf.float32)

        t = _t + 1

        with tf.name_scope("TranslatedMnist"):
            new_registers = self.register_bank.wrap(
                outp=tf.identity(outp, "outp"),
                glimpse=tf.identity(glimpse, "glimpse"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                delta=tf.identity(delta, "delta"),
                t=tf.identity(t, "t"))

        return new_registers


def render_rollouts(psystem, actions, registers, reward, external_obs, external_step_lengths):
    """ Render rollouts from TranslatedMnist task. """
    config = default_config()
    if not config.save_display and not config.display:
        print("Skipping rendering.")
        return

    n_timesteps, batch_size, n_actions = actions.shape
    s = int(np.ceil(np.sqrt(batch_size)))

    fig, subplots = plt.subplots(2*s, s)

    env_subplots = subplots[::2, :].flatten()
    glimpse_subplots = subplots[1::2, :].flatten()

    W = psystem.core_network.W
    N = psystem.core_network.N

    raw_images = external_obs[0].reshape((-1, W, W))

    [ax.imshow(raw_img, cmap='gray', origin='upper') for raw_img, ax in zip(raw_images, env_subplots)]

    rectangles = [
        ax.add_patch(patches.Rectangle(
            (0.05, 0.05), 0.9, 0.9, alpha=0.6, transform=ax.transAxes))
        for ax in env_subplots]

    glimpses = [ax.imshow(np.random.randint(256, size=(N, N)), cmap='gray', origin='upper') for ax in glimpse_subplots]

    fovea_x = psystem.rb.get('fovea_x', registers)
    fovea_y = psystem.rb.get('fovea_y', registers)
    delta = psystem.rb.get('delta', registers)
    glimpse = psystem.rb.get('glimpse', registers)

    def animate(i):
        # Find locations of bottom-left in fovea co-ordinate system, then transform to axis co-ordinate system.
        fx = fovea_x[i, :, :] - delta[i, :, :]
        fy = fovea_y[i, :, :] + delta[i, :, :]
        fx *= 0.5
        fy *= 0.5
        fy -= 0.5
        fx += 0.5
        fy *= -1

        # use delta and fovea to modify the rectangles
        for d, x, y, rect in zip(delta[i, :, :], fx, fy, rectangles):
            rect.set_x(x)
            rect.set_y(y)
            rect.set_width(d)
            rect.set_height(d)

        for g, gimg in zip(glimpse[i, :, :], glimpses):
            gimg.set_data(g.reshape(N, N))

        return rectangles + glimpses

    _animation = animation.FuncAnimation(fig, animate, n_timesteps, blit=True, interval=1000, repeat=True)

    if default_config().save_display and 0:
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
        env = TranslatedMnistEnv(True, W, N, 10, 10, 10, inc_delta=0.1, inc_x=0.1, inc_y=0.1)
        cn = TranslatedMnist(env)

        # controller = FixedController(list(range(cn.n_actions)), cn.n_actions)
        controller = FixedController([0, 1, 11, 0, 1], cn.n_actions)
        # controller = FixedController([8], cn.n_actions)
        action_selection = IdentitySelect()

        exploration = build_decaying_value(_config.schedule('exploration'), 'exploration')
        policy = Policy(
            controller, action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="translated_mnist_policy")
        return ProductionSystem(env, cn, policy, False, len(controller))

    with config.as_default():
        build_and_visualize(
            build_psystem, 'train', 16, False, render_rollouts=render_rollouts)


class TranslatedMnistTrainer(ProductionSystemTrainer):
    def build_env(self, **kwargs):
        config = default_config()
        return TranslatedMnistEnv(
            config.scaled, config.W, config.N, config.n_train, config.n_val, config.n_test,
            config.inc_delta, config.inc_x, config.inc_y)

    def build_core_network(self, env):
        return TranslatedMnist(env)
