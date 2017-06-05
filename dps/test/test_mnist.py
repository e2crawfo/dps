import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from shutil import rmtree

from dps.attention import DRAW_attention_2D
from dps.utils import MLP, NumpySeed
from dps.mnist import (
    TranslatedMnistDataset, load_or_train, MnistConfig,
    MnistPretrained, DRAW, LeNet, train_mnist)


n_symbols = 10
N = 14


def _eval_model(sess, inference, x_ph, symbols):
    test_dataset = TranslatedMnistDataset(28, 1, np.inf, 10, for_eval=True, symbols=symbols)
    y_ph = tf.placeholder(tf.int64, (None))
    _y = tf.reshape(y_ph, (-1,))
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=_y, logits=tf.log(inference)))
    correct_prediction = tf.equal(tf.argmax(inference, 1), _y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    x, y = test_dataset.next_batch()
    _loss, _accuracy = sess.run([loss, accuracy], {x_ph: x, y_ph: y})
    print("Loss: {}, accuracy: {}".format(_loss, _accuracy))
    assert _accuracy > 0.7


def build_model(inp, output_size, is_training):
    batch_size = tf.shape(inp)[0]
    with tf.name_scope('DRAW'):
        glimpse = DRAW_attention_2D(
            tf.reshape(inp, (-1, 28, 28)),
            fovea_x=tf.zeros((batch_size, 1)),
            fovea_y=tf.zeros((batch_size, 1)),
            delta=tf.ones((batch_size, 1)),
            std=tf.ones((batch_size, 1)),
            N=N)
    glimpse = tf.reshape(glimpse, (-1, N**2))
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(glimpse, output_size)
    return tf.nn.softmax(logits)


def test_mnist_load_or_train():
    with NumpySeed(83849):
        symbols = list(np.random.choice(20, n_symbols, replace=False))
        config = MnistConfig(threshold=0.05, patience=np.inf, max_steps=10000000, symbols=symbols)
        config.eval_step = 100
        g = tf.Graph()
        with g.device("/cpu:0"):
            with g.as_default():
                with tf.variable_scope('mnist') as var_scope:
                    x_ph = tf.placeholder(tf.float32, (None, 28**2))
                    inference = build_model(x_ph, n_symbols, is_training=False)
                sess = tf.Session()

                checkpoint_dir = Path(config.log_root) / 'mnist_test/checkpoint'
                try:
                    rmtree(str(checkpoint_dir))
                except FileNotFoundError:
                    pass

                checkpoint_dir.mkdir(parents=True, exist_ok=False)

                loaded = load_or_train(
                    sess, build_model, train_mnist, var_scope,
                    str(checkpoint_dir / 'model.chk'), config=config)
                assert not loaded
                _eval_model(sess, inference, x_ph, symbols)

        g = tf.Graph()
        with g.device("/cpu:0"):
            with g.as_default():
                with tf.variable_scope('mnist') as var_scope:
                    x_ph = tf.placeholder(tf.float32, (None, 28**2))
                    inference = build_model(x_ph, n_symbols, is_training=False)
                sess = tf.Session()

                loaded = load_or_train(
                    sess, build_model, train_mnist, var_scope,
                    str(checkpoint_dir / 'model.chk'), config=config)
                assert loaded
                _eval_model(sess, inference, x_ph, symbols)


def mlp(inp, outp_size, is_training):
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(inp, outp_size)
    return tf.nn.softmax(logits)


def lenet(inp, output_size, is_training):
    logits = LeNet(1024, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)
    return tf.nn.softmax(logits)


@pytest.mark.parametrize('preprocess', [False])
@pytest.mark.parametrize('classifier', ['mlp', 'lenet'])
def test_mnist_pretrained(preprocess, classifier):
    if preprocess:
        prepper = DRAW(N)
    else:
        prepper = None

    if classifier == 'mlp':
        build_classifier = mlp
    else:
        build_classifier = lenet

    with NumpySeed(83849):
        symbols = list(np.random.choice(20, n_symbols, replace=False))
        config = MnistConfig(eval_step=100, max_steps=10000, symbols=list(symbols))
        g = tf.Graph()
        with g.as_default():

            sess = tf.Session()

            checkpoint_dir = Path(config.log_root) / 'mnist_test/checkpoint'
            try:
                rmtree(str(checkpoint_dir))
            except FileNotFoundError:
                pass

            checkpoint_dir.mkdir(parents=True, exist_ok=False)

            with sess.as_default():
                build_model = MnistPretrained(
                    prepper, build_classifier, model_dir=str(checkpoint_dir),
                    preprocess=True, config=config)
                x_ph = tf.placeholder(tf.float32, (None, 28**2))
                inference = build_model(x_ph, n_symbols, is_training=False, preprocess=True)
                assert not build_model.was_loaded
                _eval_model(sess, inference, x_ph, symbols)

        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()

            with sess.as_default():
                build_model = MnistPretrained(
                    prepper, build_classifier, model_dir=str(checkpoint_dir),
                    preprocess=True, config=config)
                x_ph = tf.placeholder(tf.float32, (None, 28**2))
                inference = build_model(x_ph, n_symbols, is_training=False, preprocess=True)
                assert build_model.was_loaded

                _eval_model(sess, inference, x_ph, symbols)
