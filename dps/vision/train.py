from pathlib import Path
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_a, vgg_16, vgg_19

from dps import cfg
from dps.train import training_loop
from dps.updater import DifferentiableUpdater
from dps.utils import DpsConfig, load_or_train, Config
from dps.environment import RegressionEnv, RegressionDataset

from mnist_arithmetic import load_emnist


class EmnistDataset(RegressionDataset):
    def __init__(self, n_examples, classes, **kwargs):
        x, y, class_map = load_emnist(cfg.data_dir, classes, max_examples=n_examples, **kwargs)
        super(EmnistDataset, self).__init__(x, y)


class GetMnistUpdater(object):
    """ Can be used as `get_updater` function.

    Parameters
    ----------
    f: function
        Accepts 3 arguments: input, output_size, is_training.

    """
    def __init__(self, f, var_scope='train_mnist'):
        self.f = f
        self.var_scope = var_scope

    def __call__(self, env):
        return DifferentiableUpdater(env, self.f, scope=self.var_scope)


def build_mnist_env():
    train_dataset = EmnistDataset(
        n_examples=cfg.n_train, classes=cfg.classes,
        downsample_factor=cfg.downsample_factor, one_hot=True,
        include_blank=cfg.include_blank
    )
    val_dataset = EmnistDataset(
        n_examples=cfg.n_val, classes=cfg.classes,
        downsample_factor=cfg.downsample_factor, one_hot=True,
        include_blank=cfg.include_blank
    )
    return RegressionEnv(train_dataset, val_dataset)


MNIST_CONFIG = DpsConfig(
    build_env=build_mnist_env,
    batch_size=128,
    eval_step=1000,
    max_steps=100000,
    patience=10000,
    lr_schedule="Exp(0.001, 0, 10000, 0.9)",
    optimizer_spec="adam",
    threshold=-np.inf,
    n_train=60000,
    n_val=100,
    include_blank=True,
    log_name='mnist_pretrained',
    downsample_factor=1,
    loss_type='xent',
)


def train_mnist(build_model, var_scope, path=None):
    with Config(save_path=path, get_updater=GetMnistUpdater(build_model, var_scope)):
        result = training_loop('train_mnist')
    return result


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
    def __init__(self, build_preprocessor, build_classifier):
        self.build_preprocessor, self.build_classifier = build_preprocessor, build_classifier

    def __call__(self, inp, output_size, is_training):
        prepped = inp
        if self.build_preprocessor is not None:
            prepped = self.build_preprocessor(inp)
        inference = self.build_classifier(prepped, output_size, is_training)
        return inference


class LeNet(object):
    def __init__(
            self, n_units=1024, dropout_keep_prob=0.5, scope='LeNet',
            conv_kwargs=None, fc_kwargs=None):

        self.n_units = n_units
        self.dropout_keep_prob = dropout_keep_prob
        self.scope = scope
        self.conv_kwargs = conv_kwargs or {}
        self.fc_kwargs = fc_kwargs or {}

    def __call__(self, images, output_size, is_training=False):
        output_size = int(output_size)
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
            net = slim.conv2d(net, 32, 5, scope='conv1', **self.conv_kwargs)
            net = slim.max_pool2d(net, 2, 2, scope='pool1')
            net = slim.conv2d(net, 64, 5, scope='conv2', **self.conv_kwargs)
            net = slim.max_pool2d(net, 2, 2, scope='pool2')
            net = slim.flatten(net)

            net = slim.fully_connected(net, self.n_units, scope='fc3', **self.fc_kwargs)
            net = slim.dropout(net, self.dropout_keep_prob, is_training=is_training, scope='dropout3')

            fc_kwargs = self.fc_kwargs.copy()
            fc_kwargs['activation_fn'] = None

            logits = slim.fully_connected(net, output_size, scope='fc4', **fc_kwargs)
            return logits


class VGGNet(object):

    def __init__(self, kind):
        assert kind in 'a 16 19'.split()
        self.kind = kind

    def __call__(self, images, output_size, is_training):
        output_size = int(output_size)
        if len(images.shape) <= 1:
            raise Exception()
        if len(images.shape) == 2:
            s = int(np.sqrt(int(images.shape[1])))
            images = tf.reshape(images, (-1, s, s, 1))
        if len(images.shape) == 3:
            images = tf.expand_dims(images, -1)

        if self.kind == 'a':
            return vgg_a(images, output_size, is_training)
        elif self.kind == '16':
            return vgg_16(images, output_size, is_training)
        elif self.kind == '19':
            return vgg_19(images, output_size, is_training)
        else:
            raise Exception()


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
