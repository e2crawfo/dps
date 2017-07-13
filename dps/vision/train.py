import time
from contextlib import ExitStack
from pathlib import Path
import os

import numpy as np
import tensorflow as tf

from spectral_dagger.utils.experiment import ExperimentStore

from dps import cfg
from dps.utils import (
    build_scheduled_value, build_optimizer, EarlyStopHook,
    gen_seed, DpsConfig, load_or_train)
from dps.vision.dataset import TranslatedMnistDataset


MNIST_CONFIG = DpsConfig(
    batch_size=64,
    eval_step=1000,
    max_steps=100000,
    patience=10000,
    lr_schedule="exp 0.001 1000 0.96",
    optimizer_spec="adam",
    threshold=0.02,
    n_train=60000,
    n_val=5000,
    symbols=list(range(10)),
    include_blank=True,
    log_name='mnist_pretrained',
)


def train_mnist(build_model, var_scope, path=None):
    es = ExperimentStore(str(cfg.log_dir), max_experiments=5, delete_old=1)
    exp_dir = es.new_experiment('train_mnist', use_time=1, force_fresh=1)

    checkpoint_path = path or exp_dir.path_for('mnist.chk')

    print(cfg)
    with open(exp_dir.path_for('cfg'), 'w') as f:
        f.write(str(cfg))

    train_dataset = TranslatedMnistDataset(
        n_examples=cfg.n_train, symbols=cfg.symbols, include_blank=cfg.include_blank)
    val_dataset = TranslatedMnistDataset(
        n_examples=cfg.n_val, symbols=cfg.symbols, include_blank=cfg.include_blank)
    obs_shape = train_dataset.obs_shape

    output_size = len(cfg.symbols) + int(cfg.include_blank)

    with ExitStack() as stack:
        g = tf.Graph()

        if not cfg.use_gpu:
            stack.enter_context(g.device("/cpu:0"))

        stack.enter_context(g.as_default())
        stack.enter_context(tf.variable_scope(var_scope))

        sess = tf.Session()

        tf.set_random_seed(gen_seed())

        train_writer = tf.summary.FileWriter(exp_dir.path_for('train'), g)
        val_writer = tf.summary.FileWriter(exp_dir.path_for('val'))
        print("Writing summaries to {}.".format(exp_dir.path))

        is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

        x_ph = tf.placeholder(tf.float32, (None,) + obs_shape)
        inference = build_model(x_ph, output_size, is_training=is_training)
        y_ph = tf.placeholder(tf.int64, (None))
        _y = tf.reshape(y_ph, (-1,))
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=_y, logits=tf.log(inference)))

        correct_prediction = tf.equal(tf.argmax(inference, 1), _y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

        lr = build_scheduled_value(cfg.lr_schedule, 'mnist_learning_rate')
        optimizer = build_optimizer(cfg.optimizer_spec, lr)

        train_op = optimizer.minimize(loss)

        summary_op = tf.summary.merge_all()
        tf.contrib.framework.get_or_create_global_step()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assert_variables_initialized())

        tvars = g.get_collection('trainable_variables')
        saver = tf.train.Saver(tvars)

        early_stop = EarlyStopHook(cfg.patience)

        step = 1

        while True:
            if step % cfg.eval_step == 0:
                then = time.time()
                x, y = train_dataset.next_batch(cfg.batch_size)
                train_summary, train_loss, train_acc, _ = sess.run(
                    [summary_op, loss, accuracy, train_op], {x_ph: x, y_ph: y, is_training: True})

                duration = time.time() - then
                train_writer.add_summary(train_summary, step)

                x, y = val_dataset.next_batch(None, advance=False)
                val_summary, val_loss, val_acc = sess.run(
                    [summary_op, loss, accuracy], {x_ph: x, y_ph: y, is_training: False})
                val_writer.add_summary(val_summary, step)

                print("Step={}, Train Loss/Acc={:06.4f}/{:06.4f}, Validation Loss/Acc={:06.4f}/{:06.4f}, "
                      "Duration={:06.4f} seconds, Epoch={:04.2f}.".format(
                          step, train_loss, train_acc, val_loss, val_acc,
                          duration, train_dataset.completion))

                new_best, stop = early_stop.check(val_loss, step)

                if new_best:
                    print("Storing new best on step {} "
                          "with validation loss of {}.".format(step, val_loss))
                    best_path = saver.save(sess, checkpoint_path)
                    print("Saved to location: {}".format(best_path))

                if stop:
                    print("Optimization complete, early stopping triggered.")
                    break

                if val_loss < cfg.threshold:
                    print("Optimization complete, validation loss threshold reached.")
                    break

                if step >= cfg.max_steps:
                    print("Optimization complete, maximum number of steps reached.")
                    break

            else:
                x, y = train_dataset.next_batch(cfg.batch_size)
                train_loss, _ = sess.run([loss, train_op], {x_ph: x, y_ph: y, is_training: True})

            step += 1


class MnistPretrained(object):
    """ A wrapper around a classifier that initializes it with values stored on disk. """

    def __init__(
            self, build_preprocessor, build_classifier, var_scope_name='mnist',
            model_dir=None, mnist_config=None, name='model.chk'):
        """ If `preprocess` is False, preprocessor only applied during pre-training. """
        self._build_preprocessor = build_preprocessor
        self._build_classifier = build_classifier

        self.var_scope_name = var_scope_name
        self.var_scope = None
        self.model_dir = str(model_dir or Path(cfg.log_dir))
        self.name = name
        self.path = os.path.join(self.model_dir, name)
        self.n_builds = 0
        self.was_loaded = None
        self.mnist_config = mnist_config or MNIST_CONFIG

    def __call__(self, inp, output_size, is_training=False, preprocess=False):
        if preprocess and self._build_preprocessor is not None:
            prepped = self._build_preprocessor(inp)
        else:
            prepped = inp

        if self.n_builds == 0:
            # Create the network so there are variables to load into
            with tf.variable_scope(self.var_scope_name, reuse=False) as var_scope:
                outp = self._build_classifier(prepped, output_size, is_training)

            self.var_scope = var_scope

            builder = _MnistPretrainedBuilder(self._build_preprocessor, self._build_classifier)

            # Initializes created variables by loading from a file or training (if file isn't found)
            self.was_loaded = load_or_train(
                tf.get_default_session(), builder, train_mnist,
                self.var_scope, self.path, train_config=self.mnist_config)
            self.n_builds += 1
        else:
            with tf.variable_scope(self.var_scope, reuse=True) as var_scope:
                outp = self._build_classifier(prepped, output_size, is_training)

        return outp


class _MnistPretrainedBuilder(object):
    def __init__(self, bp, bc):
        self.bp, self.bc = bp, bc

    def __call__(self, inp, output_size, is_training):
        prepped = inp
        if self.bp is not None:
            prepped = self.bp(inp)
        inference = self.bc(prepped, output_size, is_training)
        return inference


class LeNet(object):
    def __init__(self, n_units=1024, dropout_keep_prob=0.5, scope='LeNet', output_size=None, **fc_kwargs):
        self.n_units = n_units
        self.dropout_keep_prob = dropout_keep_prob
        self.scope = scope
        self.output_size = output_size
        self.fc_kwargs = fc_kwargs

    def __call__(self, images, output_size=None, is_training=False):
        if (output_size is None) == (self.output_size is None):
            raise Exception("Conflicting or ambigous values received for attribute `output_size`.")
        output_size = self.output_size if output_size is None else output_size

        if len(images.shape) <= 1:
            raise Exception()

        if len(images.shape) == 2:
            s = int(np.sqrt(int(images.shape[1])))
            images = tf.reshape(images, (-1, s, s, 1))

        if len(images.shape) == 3:
            images = tf.expand_dims(images, -1)

        slim = tf.contrib.slim
        net = images
        with tf.variable_scope(self.scope, 'LeNet', [images, output_size]):
            net = slim.conv2d(net, 32, 5, scope='conv1')
            net = slim.max_pool2d(net, 2, 2, scope='pool1')
            net = slim.conv2d(net, 64, 5, scope='conv2')
            net = slim.max_pool2d(net, 2, 2, scope='pool2')
            net = slim.flatten(net)

            net = slim.fully_connected(net, self.n_units, scope='fc3', **self.fc_kwargs)
            net = slim.dropout(net, self.dropout_keep_prob, is_training=is_training, scope='dropout3')

            fc_kwargs = self.fc_kwargs.copy()
            fc_kwargs['activation_fn'] = None

            logits = slim.fully_connected(net, output_size, scope='fc4', **fc_kwargs)
            return logits


class ClassifierFunc(object):
    def __init__(self, f, output_size):
        self.f = f
        self.output_size = output_size

    def __call__(self, inp):
        x = inp
        x = self.f(x, self.output_size, is_training=False)
        x = tf.stop_gradient(x)
        x = tf.argmax(x, 1)
        x = tf.expand_dims(x, 1)
        return x
