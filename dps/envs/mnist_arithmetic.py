import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation, patches
from pathlib import Path

from dps import cfg
from dps.register import RegisterBank
from dps.environment import (
    RegressionEnv, CompositeEnv, InternalEnv)
from dps.vision import (
    MnistArithmeticDataset, DRAW,
    MnistPretrained, MNIST_CONFIG, ClassifierFunc)
from dps.utils import Param, Config, MLP


def build_env():
    train = MnistArithmeticDataset(n_examples=cfg.n_train)
    val = MnistArithmeticDataset(n_examples=cfg.n_val)
    test = MnistArithmeticDataset(n_examples=cfg.n_val)

    external = RegressionEnv(train, val, test)
    internal = MnistArithmetic()
    return CompositeEnv(external, internal)


config = Config(
    build_env=build_env,

    curriculum=[
        dict(image_width=100, N=16, T=20, min_digits=2, max_digits=3, base=10),
    ],
    threshold=0.15,

    classifier_str="MLP_30_30",
    build_classifier=lambda inp, outp_size, is_training=False: tf.nn.softmax(
        MLP([30, 30], activation_fn=tf.nn.sigmoid)(inp, outp_size)),

    reward_window=0.5,

    log_name='mnist_arithmetic',

    inc_delta=0.1,
    inc_x=0.1,
    inc_y=0.1,

    show_op=True,
)


class MnistArithmetic(InternalEnv):
    """ Top left is (y=0, x=0). Corresponds to using origin='upper' in plt.imshow. """
    action_names = [
        'fovea_x += ', 'fovea_x -= ', 'fovea_x ++= ', 'fovea_x --= ',
        'fovea_y += ', 'fovea_y -= ', 'fovea_y ++= ', 'fovea_y --= ',
        'delta += ', 'delta -= ', 'delta ++= ', 'delta --= ',
        'store_op', 'add', 'inc', 'multiply', 'store', 'no-op/stop']

    image_width = Param()
    N = Param()
    inc_delta = Param()
    inc_x = Param()
    inc_y = Param()
    base = Param()

    @property
    def input_shape(self):
        return (self.N*self.N,)

    def __init__(self, env):
        self.build_attention = DRAW(self.N)

        digit_config = MNIST_CONFIG.copy(symbols=range(self.base))

        name = '{}_N={}_symbols={}.chk'.format(
            cfg.classifier_str, self.N, '_'.join(str(s) for s in range(self.base)))
        digit_pretrained = MnistPretrained(
            self.build_attention, cfg.build_classifier, name=name,
            var_scope_name='digit_classifier', mnist_config=digit_config)
        self.build_digit_classifier = ClassifierFunc(digit_pretrained, self.base + 1)

        op_config = MNIST_CONFIG.copy(symbols=[10, 12, 22])

        name = '{}_N={}_symbols={}.chk'.format(
            cfg.classifier_str, self.N, '_'.join(str(s) for s in op_config.symbols))
        op_pretrained = MnistPretrained(
            self.build_attention, cfg.build_classifier, name=name,
            var_scope_name='op_classifier', mnist_config=op_config)
        self.build_op_classifier = ClassifierFunc(op_pretrained, len(op_config.symbols) + 1)

        values = (
            [0., 0., 0., 0., 1., 0., 0.] +
            [np.zeros(self.N * self.N, dtype='f')])

        self.rb = RegisterBank(
            'MnistArithmeticRB',
            'op acc fovea_x fovea_y delta vision op_vision glimpse', None,
            values=values, output_names='acc', no_display='glimpse')
        super(MnistArithmetic, self).__init__()

    def build_init(self, r):
        self.build_placeholders(r)

        op, acc, fovea_x, fovea_y, delta, vision, op_vision, glimpse = self.rb.as_tuple(r)

        glimpse = self.build_attention(
            self.input_ph, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, sigma=1.0)
        glimpse = tf.reshape(glimpse, (-1, int(np.product(glimpse.shape[1:]))))

        digit_classification = tf.stop_gradient(self.build_digit_classifier(glimpse))
        vision = tf.cast(tf.expand_dims(tf.argmax(digit_classification, 1), 1), tf.float32)

        op_classification = tf.stop_gradient(self.build_op_classifier(glimpse))
        op_vision = tf.cast(tf.expand_dims(tf.argmax(op_classification, 1), 1), tf.float32)

        with tf.name_scope("MnistArithmetic"):
            new_registers = self.rb.wrap(
                glimpse=tf.reshape(glimpse, (-1, self.N*self.N), name="glimpse"),
                acc=tf.identity(acc, "acc"),
                op=tf.identity(op, "op"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                op_vision=tf.identity(op_vision, "op_vision"),
                delta=tf.identity(delta, "delta"))

        return new_registers

    def build_step(self, t, r, a):
        _op, _acc, _fovea_x, _fovea_y, _delta, _vision, _op_vision, _glimpse = self.rb.as_tuple(r)

        (inc_fovea_x, dec_fovea_x, inc_fovea_x_big, dec_fovea_x_big,
         inc_fovea_y, dec_fovea_y, inc_fovea_y_big, dec_fovea_y_big,
         inc_delta, dec_delta, inc_delta_big, dec_delta_big,
         store_op, add, inc, multiply, store, no_op) = self.unpack_actions(a)

        acc = (1 - add - inc - multiply - store) * _acc + \
            add * (_vision + _acc) + \
            multiply * (_vision * _acc) + \
            inc * (_acc + 1) + \
            store * _vision
        op = (1 - store_op) * _op + store_op * _op_vision

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

        glimpse = self.build_attention(
            self.input_ph, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, sigma=1.0)
        glimpse = tf.reshape(glimpse, (-1, int(np.product(glimpse.shape[1:]))))

        digit_classification = tf.stop_gradient(self.build_digit_classifier(glimpse))
        vision = tf.cast(tf.expand_dims(tf.argmax(digit_classification, 1), 1), tf.float32)

        op_classification = tf.stop_gradient(self.build_op_classifier(glimpse))
        op_vision = tf.cast(tf.expand_dims(tf.argmax(op_classification, 1), 1), tf.float32)

        with tf.name_scope("MnistArithmetic"):
            new_registers = self.rb.wrap(
                glimpse=tf.reshape(glimpse, (-1, self.N*self.N), name="glimpse"),
                acc=tf.identity(acc, "acc"),
                op=tf.identity(op, "op"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                op_vision=tf.identity(op_vision, "op_vision"),
                delta=tf.identity(delta, "delta"))

        rewards = self.build_rewards(new_registers)

        return (
            tf.fill((tf.shape(r)[0], 1), 0.0),
            rewards,
            new_registers)


def render_rollouts(env, rollouts):
    external_rollouts = rollouts._metadata['external_rollouts']
    external_obs = external_rollouts.o

    registers = np.concatenate(
        [rollouts.o, rollouts._metadata['final_registers'][np.newaxis, ...]],
        axis=0)
    actions = rollouts.a

    n_timesteps, batch_size, n_actions = actions.shape
    s = int(np.ceil(np.sqrt(batch_size)))

    fig, subplots = plt.subplots(2*s, s)

    env_subplots = subplots[::2, :].flatten()
    glimpse_subplots = subplots[1::2, :].flatten()

    image_width = env.internal.image_width
    N = env.internal.N

    raw_images = external_obs[0].reshape((-1, image_width, image_width))

    [ax.imshow(raw_img, cmap='gray', origin='upper') for raw_img, ax in zip(raw_images, env_subplots)]

    rectangles = [
        ax.add_patch(patches.Rectangle(
            (0.05, 0.05), 0.9, 0.9, alpha=0.6, transform=ax.transAxes))
        for ax in env_subplots]

    glimpses = [ax.imshow(np.random.randint(256, size=(N, N)), cmap='gray', origin='upper') for ax in glimpse_subplots]

    fovea_x = env.rb.get('fovea_x', registers)
    fovea_y = env.rb.get('fovea_y', registers)
    delta = env.rb.get('delta', registers)
    glimpse = env.rb.get('glimpse', registers)

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

    if cfg.save_display:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        _animation.save(str(Path(cfg.path) / 'animation.mp4'), writer=writer)

    if cfg.display:
        plt.show()
