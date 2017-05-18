import time
from contextlib import ExitStack
from pathlib import Path
import dill

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from spectral_dagger.utils.experiment import ExperimentStore
from dps.environment import RegressionDataset
from dps.utils import build_decaying_value, EarlyStopHook, Config, gen_seed
from dps.utils import load_or_train as _load_or_train, parse_config, BaseConfig


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
    data_dir = Path(parse_config()['data_dir']).expanduser()
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

        # mnist = pickle.load(gzip.open(str(data_dir / 'mnist.pkl.gz'), 'r'), encoding='bytes')
        # mnist_x = np.concatenate((mnist[0][0], mnist[1][0], mnist[2][0]), axis=0)
        # mnist_x = mnist_x.reshape(-1, 28, 28)
        # mnist_y = np.concatenate((mnist[0][1], mnist[1][1], mnist[2][1]), axis=0)

        # keep = mnist_y < self.base
        # mnist_x = mnist_x[keep]
        # mnist_y = mnist_y[keep]

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


if __name__ == "__main__":
    W = 100
    n_digits = 3
    max_overlap = 100
    n_examples = 400
    symbols = [('A', lambda x: sum(x)), ('M', lambda x: np.product(x)), ('C', lambda x: len(x))]
    dataset = MnistArithmeticDataset(symbols, W, n_digits, max_overlap, n_examples, base=2, for_eval=False, shuffle=True)
    dataset.visualize()
    plt.show()


class MnistConfig(BaseConfig):
    batch_size = 64
    eval_step = 100
    max_steps = 100000
    patience = 10000
    lr_schedule = (0.001, 1000, 0.96, False)
    optimizer_class = tf.train.AdamOptimizer
    threshold = 0.05

    n_train = 60000
    n_val = 1000
    symbols = list(range(10))
    log_name = 'mnist_training'


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

        tf.summary.scalar('loss', loss)

        correct_prediction = tf.equal(tf.argmax(inference, 1), _y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        lr = build_decaying_value(config.lr_schedule, 'mnist_learning_rate')
        optimizer = config.optimizer_class(lr)

        # tvars = tf.trainable_variables()
        # gradients = tf.gradients(loss, tvars)
        # grads_and_vars = zip(gradients, tvars)
        # global_step = tf.contrib.framework.get_or_create_global_step()
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
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


def load_or_train(sess, build_model, var_scope, path=None, config=None):
    """ Load a pre-trained mnist model or train one and return it if none exists. """
    loaded = _load_or_train(sess, build_model, train_mnist, var_scope, path, config)
    return loaded
