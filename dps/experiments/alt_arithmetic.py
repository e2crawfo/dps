import tensorflow as tf
import numpy as np

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


class AltArithmeticDataset(RegressionDataset):
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

        super(AltArithmeticDataset, self).__init__(x, y, for_eval, shuffle)

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


class AltArithmeticEnv(RegressionEnv):
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

        super(AltArithmeticEnv, self).__init__(
            train=AltArithmeticDataset(
                mnist, symbols, shape, n_digits, upper_bound, base, blank_char, n_train, op_loc, for_eval=False),
            val=AltArithmeticDataset(
                mnist, symbols, shape, n_digits, upper_bound, base, blank_char, n_val, op_loc, for_eval=True),
            test=AltArithmeticDataset(
                mnist, symbols, shape, n_digits, upper_bound, base, blank_char, n_test, op_loc, for_eval=True))

    def __str__(self):
        return "<AltArithmeticEnv shape={} base={}>".format(self.height, self.shape, self.base)


class AltArithmetic(TensorFlowEnv):
    action_names = ['>', '<', 'v', '^', 'classify_digit', 'classify_op', '+', '+1', '*', '=', 'noop']

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

        self.init_classifiers()
        self.init_rb()

        super(AltArithmetic, self).__init__()

    def init_rb(self):
        values = (
            [0., 0., 0., 0., 0.] +
            [np.zeros(np.product(self.element_shape), dtype='f')])
        self.rb = RegisterBank(
            'AltArithmeticRB',
            'digit op acc fovea_x fovea_y glimpse', None, values=values,
            output_names='acc', no_display='glimpse')

    def init_classifiers(self):
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

    def build_init_glimpse(self, batch_size, inp, fovea_y, fovea_x):
        indices = tf.concat([
            tf.reshape(tf.range(batch_size), (-1, 1)),
            tf.cast(fovea_y, tf.int32),
            tf.cast(fovea_x, tf.int32)], axis=1)
        glimpse = tf.gather_nd(inp, indices)
        glimpse = tf.reshape(glimpse, (-1, np.product(self.element_shape)), name="glimpse")
        return glimpse

    def build_init_storage(self, batch_size):
        digit = tf.fill((batch_size, 1), -1.0)
        op = tf.fill((batch_size, 1), -1.0)
        return digit, op

    def build_init_fovea(self, batch_size, fovea_y, fovea_x):
        if self.start_loc is not None:
            fovea_y = tf.fill((batch_size, 1), self.start_loc[0])
            fovea_x = tf.fill((batch_size, 1), self.start_loc[1])
        else:
            fovea_y = tf.random_uniform(tf.shape(fovea_y), 0, self.shape[0], dtype=tf.int32)
            fovea_x = tf.random_uniform(tf.shape(fovea_x), 0, self.shape[1], dtype=tf.int32)
        fovea_y = tf.cast(fovea_y, tf.float32)
        fovea_x = tf.cast(fovea_x, tf.float32)
        return fovea_y, fovea_x

    def build_init(self, r, inp):
        digit, op, acc, fovea_x, fovea_y, glimpse = self.rb.as_tuple(r)

        batch_size = tf.shape(inp)[0]

        glimpse = self.build_init_glimpse(batch_size, inp, fovea_y, fovea_x)

        digit, op = self.build_init_storage(batch_size)

        fovea_y, fovea_x = self.build_init_fovea(batch_size, fovea_y, fovea_x)

        _, ret = self.build_return(digit, op, acc, fovea_x, fovea_y, glimpse)
        return ret

    def build_update_glimpse(self, inp, fovea_y, fovea_x):
        batch_size = tf.shape(inp)[0]
        indices = tf.concat([
            tf.reshape(tf.range(batch_size), (-1, 1)),
            tf.cast(fovea_y, tf.int32),
            tf.cast(fovea_x, tf.int32)],
            axis=1)
        glimpse = tf.gather_nd(inp, indices)
        glimpse = tf.reshape(glimpse, (-1, np.product(self.element_shape)), name="glimpse")
        return glimpse

    def build_update_storage(self, glimpse, digit, classify_digit, op, classify_op):
        digit_classification = self.build_digit_classifier(glimpse)
        digit_vision = tf.cast(digit_classification, tf.float32)
        digit = (1 - classify_digit) * digit + classify_digit * digit_vision

        op_classification = self.build_op_classifier(glimpse)
        op_vision = tf.cast(op_classification, tf.float32)
        op = (1 - classify_op) * op + classify_op * op_vision
        return digit, op

    def build_update_fovea(self, right, left, down, up, fovea_y, fovea_x):
        fovea_x = (1 - right - left) * fovea_x + \
            right * (fovea_x + 1) + \
            left * (fovea_x - 1)
        fovea_y = (1 - down - up) * fovea_y + \
            down * (fovea_y + 1) + \
            up * (fovea_y - 1)
        fovea_y = tf.clip_by_value(fovea_y, 0, self.shape[0]-1)
        fovea_x = tf.clip_by_value(fovea_x, 0, self.shape[1]-1)
        return fovea_y, fovea_x

    def build_return(self, digit, op, acc, fovea_x, fovea_y, glimpse):
        with tf.name_scope("AltArithmetic"):
            new_registers = self.rb.wrap(
                digit=tf.identity(digit, "digit"),
                op=tf.identity(op, "op"),
                acc=tf.identity(acc, "acc"),
                fovea_x=tf.identity(fovea_x, "fovea_x"),
                fovea_y=tf.identity(fovea_y, "fovea_y"),
                glimpse=glimpse)

        return tf.fill((tf.shape(digit)[0], 1), 0.0), new_registers

    def build_step(self, t, r, a, inp):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)

        (right, left, down, up, classify_digit, classify_op, add, inc, multiply, store, no_op) = (
            tf.split(a, self.n_actions, axis=1))

        acc = (1 - add - inc - multiply - store) * _acc + \
            add * (_digit + _acc) + \
            multiply * (_digit * _acc) + \
            inc * (_acc + 1) + \
            store * _digit

        glimpse = self.build_update_glimpse(inp, _fovea_y, _fovea_x)

        digit, op = self.build_update_storage(glimpse, _digit, classify_digit, _op, classify_op)

        fovea_y, fovea_x = self.build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(digit, op, acc, fovea_x, fovea_y, glimpse)


class AltArithmeticBadWiring(AltArithmetic):
    action_names = [
        '>', '<', 'v', '^', 'classify_digit', 'classify_op',
        '+', '+1', '*', '=', 'noop', '+ arg', '* arg', '= arg']

    def build_step(self, t, r, a, inp):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)

        (right, left, down, up, classify_digit, classify_op,
         add, inc, multiply, store, no_op, add_arg, mult_arg, store_arg) = (
            tf.split(a, self.n_actions, axis=1))

        acc = (1 - add - inc - multiply - store) * _acc + \
            add * (add_arg + _acc) + \
            multiply * (mult_arg * _acc) + \
            inc * (_acc + 1) + \
            store * store_arg

        glimpse = self.build_update_glimpse(inp, _fovea_y, _fovea_x)

        digit, op = self.build_update_storage(glimpse, _digit, classify_digit, _op, classify_op)

        fovea_y, fovea_x = self.build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(digit, op, acc, fovea_x, fovea_y, glimpse)


class AltArithmeticNoClassifiers(AltArithmetic):
    action_names = ['>', '<', 'v', '^', '+', '+1', '*', '=', 'noop', '+ arg', '* arg', '= arg']

    def init_classifiers(self):
        return

    def build_step(self, t, r, a, inp):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)

        (right, left, down, up, add, inc, multiply, store, no_op, add_arg, mult_arg, store_arg) = (
            tf.split(a, self.n_actions, axis=1))

        acc = (1 - add - inc - multiply - store) * _acc + \
            add * (add_arg + _acc) + \
            multiply * (mult_arg * _acc) + \
            inc * (_acc + 1) + \
            store * store_arg

        glimpse = self.build_update_glimpse(inp, _fovea_y, _fovea_x)

        fovea_y, fovea_x = self.build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(_digit, _op, acc, fovea_x, fovea_y, glimpse)


class AltArithmeticNoOps(AltArithmetic):
    action_names = ['>', '<', 'v', '^', 'classify_digit', 'classify_op', '=', 'noop', '= arg']

    def build_step(self, t, r, a, inp):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)

        (right, left, down, up, classify_digit, classify_op, store, no_op, store_arg) = (
            tf.split(a, self.n_actions, axis=1))

        acc = (1 - store) * _acc + store * store_arg

        glimpse = self.build_update_glimpse(inp, _fovea_y, _fovea_x)

        digit, op = self.build_update_storage(glimpse, _digit, classify_digit, _op, classify_op)

        fovea_y, fovea_x = self.build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(digit, op, acc, fovea_x, fovea_y, glimpse)


class AltArithmeticNoModules(AltArithmetic):
    action_names = ['>', '<', 'v', '^', '=', 'noop', '= arg']

    def init_classifiers(self):
        return

    def build_step(self, t, r, a, inp):
        _digit, _op, _acc, _fovea_x, _fovea_y, _glimpse = self.rb.as_tuple(r)
        right, left, down, up, store, no_op, store_arg = tf.split(a, self.n_actions, axis=1)

        acc = (1 - store) * _acc + store * store_arg

        glimpse = self.build_update_glimpse(inp, _fovea_y, _fovea_x)

        fovea_y, fovea_x = self.build_update_fovea(right, left, down, up, _fovea_y, _fovea_x)

        return self.build_return(_digit, _op, acc, fovea_x, fovea_y, glimpse)


def build_env():
    external = AltArithmeticEnv(
        cfg.mnist, cfg.shape, cfg.n_digits, cfg.upper_bound, cfg.base,
        cfg.n_train, cfg.n_val, cfg.n_test, cfg.op_loc, cfg.start_loc)

    if cfg.ablation == 'bad_wiring':
        internal = AltArithmeticBadWiring(external)
    elif cfg.ablation == 'no_classifiers':
        internal = AltArithmeticNoClassifiers(external)
    elif cfg.ablation == 'no_ops':
        internal = AltArithmeticNoOps(external)
    elif cfg.ablation == 'no_modules':
        internal = AltArithmeticNoModules(external)
    else:
        internal = AltArithmetic(external)

    return CompositeEnv(external, internal)
