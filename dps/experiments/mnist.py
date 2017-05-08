import pickle
import gzip
import time
from contextlib import ExitStack

import numpy as np
import tensorflow as tf

from spectral_dagger.utils.experiment import ExperimentStore
from dps.environment import RegressionDataset
from dps.utils import build_decaying_value, EarlyStopHook, Config, gen_seed


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
    def __init__(self, W, n_digits, max_overlap, n_examples, for_eval=False, shuffle=True):
        self.W = W
        self.n_digits = n_digits
        self.max_overlap = max_overlap

        mnist = pickle.load(gzip.open('/data/mnist.pkl.gz', 'r'), encoding='bytes')

        mnist_x = np.concatenate((mnist[0][0], mnist[1][0], mnist[2][0]), axis=0)
        mnist_x = mnist_x.reshape(-1, 28, 28)

        mnist_y = np.concatenate((mnist[0][1], mnist[1][1], mnist[2][1]), axis=0)

        x, y = self.make_dataset(
            self.n_digits, mnist_x, mnist_y, n_examples, self.W,
            self.max_overlap)

        super(TranslatedMnistDataset, self).__init__(x, y, for_eval, shuffle)

    @staticmethod
    def make_dataset(n_digits, X, Y, N, W, max_overlap):
        new_X, new_Y = [], []

        for j in range(N):
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

                if i > 1000: break

            ids = np.random.randint(0, Y.shape[0], n_digits)
            o = np.zeros((W, W), 'f')
            for idx, rect in zip(ids, rects):
                o[rect.left:rect.right, rect.bottom:rect.top] += X[idx]

            new_X.append(np.uint8(255*np.minimum(o, 1)))
            new_Y.append(sum(Y[idx] for idx in ids))

        new_X = np.array(new_X).astype('f').reshape(len(new_X), -1)
        new_Y = np.array(new_Y).astype('i')
        return new_X, new_Y


class MnistConfig(Config):
    batch_size = 64
    eval_step = 100
    patience = 1000
    lr_schedule = (0.0001, 1000, 0.96, False)
    optimizer_class = tf.train.AdamOptimizer
    threshold = 1e-2

    n_train = 60000
    n_val = 1000


def train_mnist(
        build_model, var_scope, config, filename=None, max_experiments=5, start_tensorboard=True):

    log_dir = getattr(config, 'logdir', '/tmp/mnist_training')
    es = ExperimentStore(log_dir, max_experiments=max_experiments, delete_old=1)
    exp_dir = es.new_experiment('mnist', use_time=1, force_fresh=1)
    config.path = exp_dir.path
    filename = filename or exp_dir.path_for('mnist.chk')
    print(config)
    with open(exp_dir.path_for('config'), 'w') as f:
        f.write(str(config))

    train_dataset = TranslatedMnistDataset(28, 1, np.inf, config.n_train)
    val_dataset = TranslatedMnistDataset(28, 1, np.inf, config.n_val, for_eval=True)
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
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_ph, logits=inference))

        tf.summary.scalar('loss', loss)

        correct_prediction = tf.equal(tf.argmax(inference, 1), y_ph)
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
                    saver.save(sess, filename)

                if stop:
                    print("Optimization complete, early stopping triggered.")
                    break

                if val_loss < config.threshold:
                    print("Optimization complete, validation loss threshold reached.")
                    break

            else:
                x, y = train_dataset.next_batch(config.batch_size)
                train_loss, _ = sess.run([loss, train_op], {x_ph: x, y_ph: y})

            step += 1
