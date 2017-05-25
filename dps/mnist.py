import time
from contextlib import ExitStack
from pathlib import Path
import dill
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from spectral_dagger.utils.experiment import ExperimentStore
from dps.environment import RegressionDataset
from dps.utils import (
    build_decaying_value, EarlyStopHook, gen_seed, DpsConfig,
    default_config, load_or_train)
from dps.attention import DRAW_attention_2D


class Rect(object):
    def __init__(self, x, y, w, h):
        self.left = x
        self.right = x+w
        self.top = y+h
        self.bottom = y

    def intersects(self, r2):
        r1 = self
        h_overlaps = (r1.left <= r2.right) and (r1.right >= r2.left)
        v_overlaps = (r1.bottom <= r2.top) and (r1.top >= r2.bottom)
        return h_overlaps and v_overlaps

    def __str__(self):
        return "<%d:%d %d:%d>" % (self.left, self.right, self.top, self.bottom)


class TranslatedMnistDataset(RegressionDataset):
    def __init__(self, W, n_digits, max_overlap, n_examples, symbols=None, function=None, for_eval=False, shuffle=True):

        self.W = W
        self.n_digits = n_digits
        self.max_overlap = max_overlap
        self.symbols = symbols or list(range(10))

        if function is None:
            function = lambda inputs: sum(inputs)
        self.function = function

        mnist_x, mnist_y, symbol_map = load_emnist(self.symbols)
        mnist_x = mnist_x.reshape(-1, 28, 28)

        x, y = self.make_dataset(
            self.n_digits, mnist_x, mnist_y, n_examples, self.W,
            self.max_overlap, self.function)

        super(TranslatedMnistDataset, self).__init__(x, y, for_eval, shuffle)

    @staticmethod
    def make_dataset(n_digits, X, Y, n_examples, W, max_overlap, function):
        if n_examples == 0:
            return np.zeros((0, W*W)).astype('f'), np.zeros((0, 1)).astype('i')

        new_X, new_Y = [], []

        for j in range(n_examples):
            i = 0
            while True:
                if W == 28:
                    rects = [Rect(0, 0, 28, 28)
                             for i in range(n_digits)]
                else:
                    rects = [Rect(np.random.randint(0, W-28),
                                  np.random.randint(0, W-28),
                                  28, 28)
                             for i in range(n_digits)]
                area = np.zeros((W, W), 'f')

                for rect in rects:
                    area[rect.left:rect.right, rect.bottom:rect.top] += 1

                if (area >= 2).sum() < max_overlap:
                    break

                i += 1

                if i > 1000:
                    raise Exception(
                        "Could not fit digits. "
                        "(n_digits: {}, W: {}, max_overlap: {})".format(n_digits, W, max_overlap))

            ids = np.random.randint(0, Y.shape[0], n_digits)
            o = np.zeros((W, W), 'f')
            for idx, rect in zip(ids, rects):
                o[rect.left:rect.right, rect.bottom:rect.top] += X[idx]

            new_X.append(np.uint8(255*np.minimum(o, 1)))
            new_Y.append(function([Y[idx] for idx in ids]))

        new_X = np.array(new_X).astype('f').reshape(len(new_X), -1)
        new_Y = np.array(new_Y).astype('i').reshape(-1, 1)
        return new_X, new_Y


chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "


def char_map(value):
    """ Assumes value between 0 and 1. """
    if value >= 1:
        value = 1 - 1e-6
    n_bins = len(chars)
    bin_id = int(value * n_bins)
    return chars[bin_id]


def image_to_string(array):
    if array.ndim == 1:
        array = array.reshape(-1, int(np.sqrt(array.shape[0])))
    image = [char_map(value) for value in array.flatten()]
    image = np.reshape(image, array.shape)
    return '\n'.join(''.join(c for c in row) for row in image)


def view_emnist(x, y, n):
    m = int(np.ceil(np.sqrt(n)))
    fig, subplots = plt.subplots(m, m)
    for i, s in enumerate(subplots.flatten()):
        s.imshow(x[i, :].reshape(28, 28).transpose())
        s.set_title(str(y[i, 0]))


def load_emnist(classes, balance=False):
    """ maps the symbols down to range(0, len(classes)) """
    import gzip
    data_dir = Path(default_config().data_dir).expanduser()
    emnist_dir = data_dir / 'emnist/emnist-byclass'
    y = []
    x = []
    symbol_map = {}
    for i, cls in enumerate(sorted(list(classes))):
        with gzip.open(str(emnist_dir / (str(cls) + '.pklz')), 'rb') as f:
            x.append(dill.load(f))
            y.extend([i] * x[-1].shape[0])
        symbol_map[cls] = i
    x = np.concatenate(x, axis=0)
    y = np.array(y).reshape(-1, 1)

    order = np.random.permutation(x.shape[0])

    x = x[order, :]
    y = y[order, :]

    if balance:
        class_count = min([(y == c).sum() for c in classes])
        keep_x, keep_y = [], []
        for i, cls in enumerate(classes):
            keep_indices, _ = np.nonzero(i == cls)
            keep_indices = keep_indices[:class_count]
            keep_x.append(x[keep_indices, :])
            keep_y.append(y[keep_indices, :])
        x = np.concatenate(keep_x, 0)
        y = np.concatenate(keep_y, 0)

    return x, y, symbol_map


def idx_to_char(i):
    assert isinstance(i, int)
    if i >= 72:
        raise Exception()
    elif i >= 36:
        char = chr(i-36+ord('a'))
    elif i >= 10:
        char = chr(i-10+ord('A'))
    elif i >= 0:
        char = str(i)
    else:
        raise Exception()
    return char


def char_to_idx(c):
    assert isinstance(c, str)
    assert len(c) == 1

    if c.isupper():
        idx = ord(c) - ord('A') + 10
    elif c.islower():
        idx = ord(c) - ord('A') + 36
    elif c.isnumeric():
        idx = int(c)
    else:
        raise Exception()
    return idx


class MnistArithmeticDataset(RegressionDataset):
    def __init__(self, symbols, W, n_digits, max_overlap, n_examples, base=10, for_eval=False, shuffle=True):
        assert 1 <= base <= 10
        # symbols is a list of pairs of the form (letter, reduction function)
        self.symbols = symbols
        self.W = W
        self.n_digits = n_digits
        self.max_overlap = max_overlap
        self.base = base

        mnist_x, mnist_y, symbol_map = load_emnist(list(range(base)))
        mnist_x = mnist_x.reshape(-1, 28, 28)

        # map each character to an index.
        functions = {char_to_idx(s): f for s, f in symbols}

        emnist_x, emnist_y, symbol_map = load_emnist(list(functions.keys()))
        emnist_x = emnist_x.reshape(-1, 28, 28)

        functions = {symbol_map[k]: v for k, v in functions.items()}

        x, y = self.make_dataset(
            self.n_digits, mnist_x, mnist_y, emnist_x, emnist_y, functions,
            n_examples, self.W, self.max_overlap)

        super(MnistArithmeticDataset, self).__init__(x, y, for_eval, shuffle)

    @staticmethod
    def make_dataset(
            n_digits, X, Y, eX, eY, functions, n_examples, W, max_overlap):
        # n_digits is really max_digits.
        assert n_digits > 0

        if n_examples == 0:
            return np.zeros((0, W*W)).astype('f'), np.zeros((0, 1)).astype('i')

        new_X, new_Y = [], []

        for j in range(n_examples):
            n = np.random.randint(1, n_digits+1)
            i = 0
            # Sample rectangles
            while True:
                rects = [Rect(np.random.randint(0, W-28),
                              np.random.randint(0, W-28),
                              28, 28)
                         for i in range(n + 1)]
                area = np.zeros((W, W), 'f')

                for rect in rects:
                    area[rect.left:rect.right, rect.bottom:rect.top] += 1

                if (area >= 2).sum() < max_overlap:
                    break

                i += 1

                if i > 1000:
                    raise Exception(
                        "Could not fit digits. "
                        "(n_digits: {}+1, W: {}, max_overlap: {})".format(n_digits, W, max_overlap))

            # Populate rectangles
            o = np.zeros((W, W), 'f')

            symbol_idx = np.random.randint(0, eY.shape[0])
            symbol_class = eY[symbol_idx, 0]
            func = functions[symbol_class]
            o[rect.left:rect.right, rect.bottom:rect.top] += eX[symbol_idx]

            ids = np.random.randint(0, Y.shape[0], n)

            for idx, rect in zip(ids, rects):
                o[rect.left:rect.right, rect.bottom:rect.top] += X[idx]

            new_X.append(np.uint8(255*np.minimum(o, 1)))
            new_Y.append(func([Y[idx] for idx in ids]))

        new_X = np.array(new_X).astype('f').reshape(len(new_X), -1)
        new_Y = np.array(new_Y).astype('i').reshape(-1, 1)
        return new_X, new_Y

    def visualize(self, n=9):
        m = int(np.ceil(np.sqrt(n)))
        fig, subplots = plt.subplots(m, m)
        size = int(np.sqrt(self.x.shape[1]))
        for i, s in enumerate(subplots.flatten()):
            s.imshow(self.x[i, :].reshape(size, size))
            s.set_title(str(self.y[i, 0]))


class MnistConfig(DpsConfig):
    batch_size = 64
    eval_step = 100
    max_steps = 100000
    patience = 10000
    lr_start, lr_denom, lr_decay = 0.001, 1000, 96
    optimizer_class = tf.train.AdamOptimizer
    threshold = 0.05

    n_train = 60000
    n_val = 1000
    symbols = list(range(10))
    log_name = 'mnist_pretrained'


def train_mnist(build_model, var_scope, path=None, config=None):

    config = config or MnistConfig()

    es = ExperimentStore(str(config.log_dir), max_experiments=5, delete_old=1)
    exp_dir = es.new_experiment('train_mnist', use_time=1, force_fresh=1)

    checkpoint_path = path or exp_dir.path_for('mnist.chk')

    print(config)
    with open(exp_dir.path_for('config'), 'w') as f:
        f.write(str(config))

    train_dataset = TranslatedMnistDataset(28, 1, np.inf, config.n_train, symbols=config.symbols)
    val_dataset = TranslatedMnistDataset(28, 1, np.inf, config.n_val, for_eval=True, symbols=config.symbols)
    obs_dim = 28 ** 2

    g = tf.Graph()
    with ExitStack() as stack:
        stack.enter_context(g.as_default())
        stack.enter_context(tf.variable_scope(var_scope))

        sess = tf.Session()

        tf.set_random_seed(gen_seed())

        train_writer = tf.summary.FileWriter(exp_dir.path_for('train'), g)
        val_writer = tf.summary.FileWriter(exp_dir.path_for('val'))
        print("Writing summaries to {}.".format(exp_dir.path))

        x_ph = tf.placeholder(tf.float32, (None, obs_dim))
        inference = build_model(x_ph)
        y_ph = tf.placeholder(tf.int64, (None))
        _y = tf.reshape(y_ph, (-1,))
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=_y, logits=tf.log(inference)))

        correct_prediction = tf.equal(tf.argmax(inference, 1), _y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

        lr = build_decaying_value(config.schedule('lr'), 'mnist_learning_rate')
        optimizer = config.optimizer_class(lr)

        train_op = optimizer.minimize(loss)

        summary_op = tf.summary.merge_all()
        tf.contrib.framework.get_or_create_global_step()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assert_variables_initialized())

        tvars = g.get_collection('trainable_variables')
        saver = tf.train.Saver(tvars)

        early_stop = EarlyStopHook(config.patience)

        step = 1

        while True:
            if step % config.eval_step == 0:
                then = time.time()
                x, y = train_dataset.next_batch(config.batch_size)
                train_summary, train_loss, train_acc, _ = sess.run(
                    [summary_op, loss, accuracy, train_op], {x_ph: x, y_ph: y})

                duration = time.time() - then
                train_writer.add_summary(train_summary, step)

                x, y = val_dataset.next_batch(config.batch_size)
                val_summary, val_loss, val_acc = sess.run(
                    [summary_op, loss, accuracy], {x_ph: x, y_ph: y})
                val_writer.add_summary(val_summary, step)

                print("Step={}, Train Loss/Acc={:06.4f}/{:06.4f}, Validation Loss/Acc={:06.4f}/{:06.4f}, "
                      "Duration={:06.4f} seconds, Epoch={:04.2f}.".format(
                          step, train_loss, train_acc, val_loss, val_acc,
                          duration, train_dataset.completion))

                new_best, stop = early_stop.check(val_loss, step)

                if new_best:
                    print("Storing new best on step {} "
                          "with validation loss of {}.".format(step, val_loss))
                    saver.save(sess, checkpoint_path)

                if stop:
                    print("Optimization complete, early stopping triggered.")
                    break

                if val_loss < config.threshold:
                    print("Optimization complete, validation loss threshold reached.")
                    break

                if step >= config.max_steps:
                    print("Optimization complete, maximum number of steps reached.")
                    break

            else:
                x, y = train_dataset.next_batch(config.batch_size)
                train_loss, _ = sess.run([loss, train_op], {x_ph: x, y_ph: y})

            step += 1


class DRAW(object):
    def __init__(self, N):
        self.N = N

    def __call__(self, inp, fovea_x=None, fovea_y=None, delta=None, sigma=None):
        reshape = len(inp.shape) == 2
        if reshape:
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

        if reshape:
            glimpse = tf.reshape(glimpse, (-1, self.N*self.N))

        return glimpse


class MnistPretrained(object):
    """ A wrapper around a classifier that initializes it with values stored on disk. """

    def __init__(
            self, build_preprocessor, build_classifier, var_scope_name='mnist',
            model_dir=None, name='model.chk', config=None, preprocess=False):
        """ If `preprocess` is False, preprocessor only applied during pre-training. """
        self._build_preprocessor = build_preprocessor
        self._build_classifier = build_classifier

        self.var_scope_name = var_scope_name
        self.var_scope = None
        self.model_dir = str(model_dir or Path(default_config().log_dir))
        self.name = name
        self.path = os.path.join(self.model_dir, name)
        self.n_builds = 0
        self.config = config or MnistConfig()
        self.preprocess = preprocess
        self.was_loaded = None

        self.n_symbols = len(self.config.symbols)

    def __call__(self, inp, preprocess=False):
        if preprocess and self._build_preprocessor is not None:
            prepped = self._build_preprocessor(inp)
        else:
            prepped = inp

        if self.n_builds == 0:
            # Create the network so there are variables to load into
            with tf.variable_scope(self.var_scope_name, reuse=False) as var_scope:
                outp = self._build_classifier(prepped, self.n_symbols)

            self.var_scope = var_scope

            builder = _MnistPretrainedBuilder(self._build_preprocessor, self._build_classifier, self.n_symbols)

            # Initializes created variables by loading from a file or training separately
            self.was_loaded = load_or_train(
                tf.get_default_session(), builder, train_mnist, self.var_scope, self.path, self.config)
            self.n_builds += 1
        else:
            with tf.variable_scope(self.var_scope, reuse=True) as var_scope:
                outp = self._build_classifier(prepped, self.n_symbols)

        return outp


class _MnistPretrainedBuilder(object):
    def __init__(self, bp, bc, n_symbols):
        self.bp, self.bc, self.n_symbols = bp, bc, n_symbols

    def __call__(self, inp):
        prepped = inp
        if self.bp is not None:
            prepped = self.bp(inp)
        inference = self.bc(prepped, self.n_symbols)
        return inference


if __name__ == "__main__":
    W = 100
    n_digits = 3
    max_overlap = 100
    n_examples = 400
    symbols = [('A', lambda x: sum(x)), ('M', lambda x: np.product(x)), ('C', lambda x: len(x))]
    dataset = MnistArithmeticDataset(symbols, W, n_digits, max_overlap, n_examples, base=2, for_eval=False, shuffle=True)
    dataset.visualize()
    plt.show()
