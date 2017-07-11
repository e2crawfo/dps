import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from pathlib import Path

from dps import cfg
from dps.register import RegisterBank
from dps.environment import (
    RegressionDataset, RegressionEnv, CompositeEnv, TensorFlowEnv)
from dps.mnist import char_to_idx, load_emnist, MnistPretrained, ClassifierFunc


def digits_to_numbers(digits, base=10, axis=-1, keepdims=False):
    """ Assumes little-endian (least-significant stored first). """
    mult = base ** np.arange(digits.shape[axis])
    shape = [1] * digits.ndim
    shape[axis] = mult.shape[axis]
    mult = mult.reshape(shape)
    return (digits * mult).sum(axis=axis, keepdims=keepdims)


def numbers_to_digits(numbers, shape, base=10):
    numbers = numbers.copy()
    digits = []
    for i in range(shape):
        digits.append(numbers % base)
        numbers //= base
    return np.stack(digits, -1)


class Container(object):
    def __init__(self, X, Y):
        assert len(X) == len(Y)
        self.X, self.Y = X, Y

    def get_random(self):
        idx = np.random.randint(len(self.X))
        return self.X[idx], self.Y[idx]


class SimpleArithmeticDataset(RegressionDataset):
    def __init__(
            self, mnist, symbols, shape, n_digits, upper_bound, base, blank_char,
            n_examples, op_loc, for_eval=False, shuffle=True):

        assert 1 <= base <= 10

        # symbols is a list of pairs of the form (letter, reduction function)
        self.mnist = mnist
        self.symbols = symbols
        self.shape = shape
        self.n_digits = n_digits
        self.upper_bound = upper_bound
        self.base = base
        self.blank_char = blank_char
        self.op_loc = op_loc

        assert np.product(shape) >= n_digits + 1

        if self.mnist:
            functions = {char_to_idx(s): f for s, f in symbols}

            emnist_x, emnist_y, symbol_map = load_emnist(list(functions.keys()), balance=True)
            emnist_x = emnist_x.reshape(-1, 28, 28)
            emnist_x = np.uint8(255*np.minimum(emnist_x, 1))
            emnist_y = np.squeeze(emnist_y, 1)

            functions = {symbol_map[k]: v for k, v in functions.items()}

            symbol_reps = Container(emnist_x, emnist_y)

            mnist_x, mnist_y, symbol_map = load_emnist(list(range(base)), balance=True)
            mnist_x = mnist_x.reshape(-1, 28, 28)
            mnist_x = np.uint8(255*np.minimum(mnist_x, 1))
            mnist_y = np.squeeze(mnist_y, 1)

            digit_reps = Container(mnist_x, mnist_y)
            blank_element = np.zeros((28, 28))
        else:
            sorted_symbols = sorted(symbols, key=lambda x: x[0])
            functions = {i: f for i, (_, f) in enumerate(sorted_symbols)}
            symbol_values = sorted(functions.keys())

            symbol_reps = Container(symbol_values, symbol_values)
            digit_reps = Container(np.arange(base), np.arange(base))

            blank_element = np.array([-1])

        x, y = self.make_dataset(
            self.shape, self.n_digits, self.upper_bound, self.base,
            blank_element, symbol_reps, digit_reps,
            functions, n_examples, op_loc)

        super(SimpleArithmeticDataset, self).__init__(x, y, for_eval, shuffle)

    @staticmethod
    def make_dataset(
            shape, n_digits, upper_bound, base, blank_element,
            symbol_reps, digit_reps, functions, n_examples, op_loc):

        if n_examples == 0:
            return np.zeros((0,) + shape + blank_element.shape).astype('f'), np.zeros((0, 1)).astype('i')

        new_X, new_Y = [], []

        # Include a blank character, so it has to learn to skip over them.
        size = np.product(shape)

        element_shape = blank_element.shape
        _blank_element = blank_element.reshape((1,)*len(shape) + blank_element.shape)

        for j in range(n_examples):
            if upper_bound:
                n = np.random.randint(0, n_digits+1)
            else:
                n = n_digits

            if op_loc is None:
                indices = np.random.choice(size, n+1, replace=False)
            else:
                _op_loc = np.ravel_multi_index(op_loc, shape)
                indices = np.random.choice(size-1, n, replace=False)
                indices[indices == _op_loc] = size-1
                indices = np.append(_op_loc, indices)

            locs = list(zip(*np.unravel_index(indices, shape)))

            env = np.tile(_blank_element, shape+(1,)*len(element_shape))

            x, y = symbol_reps.get_random()
            func = functions[int(y)]
            env[locs[0]] = x

            ys = []

            for loc in locs[1:]:
                x, y = digit_reps.get_random()
                ys.append(y)
                env[loc] = x

            new_X.append(env)
            new_Y.append(func(ys))

        new_X = np.array(new_X).astype('f')
        new_Y = np.array(new_Y).astype('i').reshape(-1, 1)
        return new_X, new_Y


class SimpleArithmeticEnv(RegressionEnv):
    def __init__(self, mnist, shape, n_digits, upper_bound, base,
                 n_train, n_val, n_test, op_loc=None, start_loc=None):
        self.mnist = mnist
        self.shape = shape
        self.n_digits = n_digits
        self.upper_bound = upper_bound
        self.base = base
        self.blank_char = blank_char = 'b'
        self.symbols = symbols = [
            ('A', lambda x: sum(x)),
            ('M', lambda x: np.product(x)),
            ('C', lambda x: len(x))]

        self.op_loc = op_loc
        self.start_loc = op_loc

        super(SimpleArithmeticEnv, self).__init__(
            train=SimpleArithmeticDataset(
                mnist, symbols, shape, n_digits, upper_bound, base, blank_char, n_train, op_loc, for_eval=False),
            val=SimpleArithmeticDataset(
                mnist, symbols, shape, n_digits, upper_bound, base, blank_char, n_val, op_loc, for_eval=True),
            test=SimpleArithmeticDataset(
                mnist, symbols, shape, n_digits, upper_bound, base, blank_char, n_test, op_loc, for_eval=True))

    def __str__(self):
        return "<SimpleArithmeticEnv shape={} base={}>".format(self.height, self.shape, self.base)


class SimpleArithmetic(TensorFlowEnv):
    """ Top left is (x=0, y=0). Sign is in the bottom left of the input grid.

    For now, the location of the write head is the same as the x location of the read head.

    """
    action_names = ['>', '<', 'v', '^', 'store_op', '+', '+1', '*', 'store', 'noop']

    @property
    def element_shape(self):
        return (28, 28) if self.mnist else (1,)

    def static_inp_type_and_shape(self):
        return (tf.float32, self.shape + self.element_shape)

    make_input_available = True

    def __init__(self, env):
        self.mnist = env.mnist
        self.symbols = env.symbols
        self.shape = env.shape

        if not len(self.shape) == 2:
            raise Exception("Shape must have length 2.")

        self.n_digits = env.n_digits
        self.upper_bound = env.upper_bound
        self.base = env.base
        self.blank_char = env.blank_char

        self.start_loc = env.start_loc

        if self.mnist:
            build_classifier = cfg.build_classifier
            classifier_str = cfg.classifier_str

            digit_config = cfg.mnist_config.copy(symbol=list(range(self.base)))

            name = '{}_symbols={}.chk'.format(
                classifier_str, '_'.join(str(s) for s in digit_config.symbols))
            digit_pretrained = MnistPretrained(
                None, build_classifier, name=name, model_dir='/tmp/dps/mnist_pretrained/',
                var_scope_name='digit_classifier', mnist_config=digit_config)
            self.build_digit_classifier = ClassifierFunc(digit_pretrained, self.base + 1)

            op_config = cfg.mnist_config.copy(symbols=[10, 12, 22])

            name = '{}_symbols={}.chk'.format(
                classifier_str, '_'.join(str(s) for s in op_config.symbols))

            op_pretrained = MnistPretrained(
                None, build_classifier, name=name, model_dir='/tmp/dps/mnist_pretrained/',
                var_scope_name='op_classifier', mnist_config=op_config)
            self.build_op_classifier = ClassifierFunc(op_pretrained, len(op_config.symbols) + 1)

        else:
            self.build_digit_classifier = lambda x: tf.identity(x)
            self.build_op_classifier = lambda x: tf.identity(x)

        values = (
            [0., 0., 0., 0., 0., 0.] +
            [np.zeros(np.product(self.element_shape), dtype='f')])

        self.rb = RegisterBank(
            'SimpleArithmeticRB',
            'op acc fovea_x fovea_y vision op_vision', 'glimpse', values=values,
            output_names='acc', no_display='glimpse')

        super(SimpleArithmetic, self).__init__()

    def build_init(self, r, inp):
        op, acc, fovea_x, fovea_y, vision, op_vision, glimpse = self.rb.as_tuple(r)

        _fovea_x = tf.cast(fovea_x, tf.int32)
        _fovea_y = tf.cast(fovea_y, tf.int32)

        batch_size = tf.shape(inp)[0]
        indices = tf.concat([
            tf.reshape(tf.range(batch_size), (-1, 1)),
            _fovea_y,
            _fovea_x], axis=1)
        glimpse = tf.gather_nd(inp, indices)
        glimpse = tf.reshape(glimpse, (-1, np.product(self.element_shape)), name="glimpse")

        digit_classification = self.build_digit_classifier(glimpse)
        vision = tf.cast(digit_classification, tf.float32)

        op_classification = self.build_op_classifier(glimpse)
        op_vision = tf.cast(op_classification, tf.float32)

        if self.start_loc is not None:
            fovea_y = tf.fill((batch_size, 1), self.start_loc[0])
            fovea_x = tf.fill((batch_size, 1), self.start_loc[1])
        else:
            fovea_y = tf.random_uniform(tf.shape(fovea_y), 0, self.shape[0], dtype=tf.int32)
            fovea_x = tf.random_uniform(tf.shape(fovea_x), 0, self.shape[1], dtype=tf.int32)

        fovea_x = tf.cast(fovea_x, tf.float32)
        fovea_y = tf.cast(fovea_y, tf.float32)

        with tf.name_scope("SimpleArithmetic"):
            new_registers = self.rb.wrap(
                glimpse=glimpse,
                acc=tf.identity(acc, "acc"),
                op=tf.identity(op, "op"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                op_vision=tf.identity(op_vision, "op_vision"))

        return new_registers

    def build_step(self, t, r, a, inp):
        _op, _acc, _fovea_x, _fovea_y, _vision, _op_vision, _glimpse = self.rb.as_tuple(r)

        (inc_fovea_x, dec_fovea_x,
         inc_fovea_y, dec_fovea_y,
         store_op, add, inc, multiply, store, no_op) = (
            tf.split(a, self.n_actions, axis=1))

        acc = (1 - add - inc - multiply - store) * _acc + \
            add * (_vision + _acc) + \
            multiply * (_vision * _acc) + \
            inc * (_acc + 1) + \
            store * _vision
        op = (1 - store_op) * _op + store_op * _op_vision

        fovea_x = (1 - inc_fovea_x - dec_fovea_x) * _fovea_x + \
            inc_fovea_x * (_fovea_x + 1) + \
            dec_fovea_x * (_fovea_x - 1)

        fovea_y = (1 - inc_fovea_y - dec_fovea_y) * _fovea_y + \
            inc_fovea_y * (_fovea_y + 1) + \
            dec_fovea_y * (_fovea_y - 1)

        fovea_y = tf.clip_by_value(fovea_y, 0, self.shape[0]-1)
        fovea_x = tf.clip_by_value(fovea_x, 0, self.shape[1]-1)

        _fovea_x = tf.cast(fovea_x, tf.int32)
        _fovea_y = tf.cast(fovea_y, tf.int32)

        batch_size = tf.shape(inp)[0]
        indices = tf.concat([
            tf.reshape(tf.range(batch_size), (-1, 1)),
            _fovea_y,
            _fovea_x], axis=1)
        glimpse = tf.gather_nd(inp, indices)
        glimpse = tf.reshape(glimpse, (-1, np.product(self.element_shape)), name="glimpse")

        digit_classification = self.build_digit_classifier(glimpse)
        vision = tf.cast(digit_classification, tf.float32)

        op_classification = self.build_op_classifier(glimpse)
        op_vision = tf.cast(op_classification, tf.float32)

        with tf.name_scope("MnistArithmetic"):
            new_registers = self.rb.wrap(
                glimpse=glimpse,
                acc=tf.identity(acc, "acc"),
                op=tf.identity(op, "op"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                vision=tf.identity(vision, "vision"),
                op_vision=tf.identity(op_vision, "op_vision"))

        return tf.fill((tf.shape(r)[0], 1), 0.0), new_registers


def render_rollouts(env, actions, registers, reward, info):
    if not cfg.save_display and not cfg.display:
        print("Skipping rendering.")
        return

    external_obs = [i['external_obs'] for i in info]

    n_timesteps, batch_size, n_actions = actions.shape
    s = int(np.ceil(np.sqrt(batch_size)))

    fig, subplots = plt.subplots(s, 2*s)

    if batch_size == 1:
        env_subplots = [subplots[0]]
        glimpse_subplots = [subplots[1]]
    else:
        env_subplots = subplots[:, ::2].flatten()
        for i, ep in enumerate(env_subplots):
            ep.set_title(str(info[0]['y'][i]))
        glimpse_subplots = subplots[:, 1::2].flatten()

    plt.subplots_adjust(hspace=0.5)

    if env.env.mnist:
        images = []
        for ri in external_obs[0]:
            ri = [np.concatenate(r, axis=-1) for r in ri]
            ri = np.concatenate(ri, axis=0)
            images.append(ri)
        images = np.array(images)
    else:
        images = np.squeeze(external_obs[0], axis=-1)

    shape = env.internal.shape

    mx = images.max()
    mn = images.min()
    images = (images - mn) / (mx - mn)

    [ax.imshow(im, cmap='gray', origin='upper', extent=(0, shape[1], shape[0], 0), vmin=0.0, vmax=1.0) for im, ax in zip(images, env_subplots)]

    offset = 0.1
    s = 1 - 2*offset

    titles1 = [es.title for es in env_subplots]
    titles2 = [gs.title for gs in glimpse_subplots]
    actions_reduced = np.argmax(actions, axis=-1)

    # When specifying rectangles, you supply the bottom left corner, but not "bottom left" as your looking at it,
    # but bottom left in the co-ordinate system you're drawing in.
    rectangles = [
        ax.add_patch(patches.Rectangle((offset, offset), s, s, alpha=0.6, fill=True, transform=ax.transData))
        for ax in env_subplots]

    glimpse_shape = env.internal.element_shape
    if len(glimpse_shape) == 1:
        glimpse_shape = glimpse_shape + (1,)

    fovea_x = env.rb.get('fovea_x', registers)
    fovea_y = env.rb.get('fovea_y', registers)
    glimpse = env.rb.get('glimpse', registers)
    glimpse = (glimpse - mn) / (mx - mn)

    glimpses = [ax.imshow(im, cmap='gray', origin='upper') for im, ax in zip(images, glimpse_subplots)]

    [ax.imshow(im, cmap='gray', origin='upper', extent=(0, shape[1], shape[0], 0), vmin=0.0, vmax=1.0) for im, ax in zip(images, env_subplots)]

    def animate(i):
        for n, t in enumerate(titles1):
            t.set_text(env.internal.action_names[actions_reduced[i, n]])
        for n, t in enumerate(titles2):
            t.set_text("t={},y={},r={}".format(i, info[0]['y'][n][0], reward[i, n, 0]))

        # Find locations of bottom-left in fovea co-ordinate system, then transform to axis co-ordinate system.
        fx = fovea_x[i, :, :] + offset
        fy = fovea_y[i, :, :] + offset

        for x, y, rect in zip(fx, fy, rectangles):
            rect.set_x(x)
            rect.set_y(y)

        for g, gimg in zip(glimpse[i, :, :], glimpses):
            gimg.set_data(g.reshape(glimpse_shape))

        return rectangles + glimpses + titles1 + titles2

    _animation = animation.FuncAnimation(fig, animate, n_timesteps, blit=False, interval=1000, repeat=True)

    if cfg.save_display:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        _animation.save(str(Path(cfg.path) / 'animation.mp4'), writer=writer)

    if cfg.display:
        plt.show()


def render_rollouts_static(env, actions, registers, reward, info):
    if not cfg.save_display and not cfg.display:
        print("Skipping rendering.")
        return

    external_obs = [i['external_obs'] for i in info]

    if env.env.mnist:
        images = []
        for ri in external_obs[0]:
            ri = [np.concatenate(r, axis=-1) for r in ri]
            ri = np.concatenate(ri, axis=0)
            images.append(ri)
        images = np.array(images)
    else:
        images = np.squeeze(external_obs[0], axis=-1)
    mx = images.max()
    mn = images.min()
    images = (images - mn) / (mx - mn)

    n_timesteps, batch_size, n_actions = actions.shape
    fig, subplots = plt.subplots(batch_size, n_timesteps+1)

    shape = env.internal.shape

    actions_reduced = np.argmax(actions, axis=-1)
    offset = 0.1
    s = 1 - 2*offset

    fovea_x = env.rb.get('fovea_x', registers)
    fovea_y = env.rb.get('fovea_y', registers)
    acc = env.rb.get('acc', registers)

    for i in range(batch_size):
        for j in range(n_timesteps + 1):
            ax = subplots[i, j]
            ax.imshow(images[i], cmap='gray', origin='upper', extent=(0, shape[1], shape[0], 0), vmin=0.0, vmax=1.0)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            if j < n_timesteps:
                action_name = env.internal.action_names[actions_reduced[j, i]]
                if action_name == "store":
                    action_name = "+"
                elif action_name in ["noop", "store_op"]:
                    action_name = "#"

                text = "action: {}\nacc: {}".format(action_name, int(acc[j, i, 0]))

                ax.text(
                    0.5, 1.2, text, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')

            # Find locations of bottom-left in fovea co-ordinate system, then transform to axis co-ordinate system.
            fx = fovea_x[j, i, :] + offset
            fy = fovea_y[j, i, :] + offset

            rect = patches.Rectangle((fx, fy), s, s, alpha=0.6, fill=True, transform=ax.transData)
            ax.add_patch(rect)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.00, wspace=0.03, hspace=0.0)

    if cfg.save_display:
        pass  # TODO

    if cfg.display:
        plt.show()


def build_env():
    external = SimpleArithmeticEnv(
        cfg.mnist, cfg.shape, cfg.n_digits, cfg.upper_bound, cfg.base,
        cfg.n_train, cfg.n_val, cfg.n_test, cfg.op_loc, cfg.start_loc)
    internal = SimpleArithmetic(external)
    return CompositeEnv(external, internal)