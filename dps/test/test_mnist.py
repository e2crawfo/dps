import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from shutil import rmtree

from dps.utils import MLP, NumpySeed
from dps.vision import (
    TranslatedMnistDataset, load_or_train, MNIST_CONFIG,
    MnistPretrained, DRAW, LeNet, train_mnist)
from dps.vision.attention import DRAW_attention_2D


N = 14


def _eval_model(sess, inference, x_ph, symbols):
    test_dataset = TranslatedMnistDataset(n_examples=100, symbols=symbols)
    y_ph = tf.placeholder(tf.int64, (None))
    _y = tf.reshape(y_ph, (-1,))
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=_y, logits=tf.log(inference)))
    correct_prediction = tf.equal(tf.argmax(inference, 1), _y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    x, y = test_dataset.next_batch(advance=False)
    _loss, _accuracy = sess.run([loss, accuracy], {x_ph: x, y_ph: y})
    print("Loss: {}, accuracy: {}".format(_loss, _accuracy))
    assert _accuracy > 0.7


def build_model(inp, output_size, is_training):
    if len(inp.shape) == 2:
        s = int(np.sqrt(int(inp.shape[1])))
        inp = tf.reshape(inp, (-1, s, s))

    batch_size = tf.shape(inp)[0]
    with tf.name_scope('DRAW'):
        glimpse = DRAW_attention_2D(
            inp,
            fovea_x=tf.zeros((batch_size, 1)),
            fovea_y=tf.zeros((batch_size, 1)),
            delta=tf.ones((batch_size, 1)),
            std=tf.ones((batch_size, 1)),
            N=N)
    glimpse = tf.reshape(glimpse, (-1, N**2))
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(glimpse, output_size)
    return tf.nn.softmax(logits)


def make_checkpoint_dir(config):
    checkpoint_dir = Path(config.log_root) / 'mnist_test/checkpoint'
    try:
        rmtree(str(checkpoint_dir))
    except FileNotFoundError:
        pass
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return checkpoint_dir


def test_mnist_load_or_train():
    with NumpySeed(83849):
        n_symbols = 10
        symbols = list(np.random.choice(20, n_symbols, replace=False))

        config = MNIST_CONFIG.copy(
            threshold=0.20,
            patience=np.inf,
            max_steps=10000000,
            symbols=symbols,
            eval_step=100,
            include_blank=True)

        checkpoint_dir = make_checkpoint_dir(config)
        output_size = n_symbols + 1

        g = tf.Graph()
        with g.device("/cpu:0"):
            with g.as_default():

                with tf.variable_scope('mnist') as var_scope:
                    x_ph = tf.placeholder(tf.float32, (None, 28, 28))
                    inference = build_model(x_ph, output_size, is_training=False)

                sess = tf.Session()
                loaded = load_or_train(
                    sess, build_model, train_mnist, var_scope,
                    str(checkpoint_dir / 'model.chk'), train_config=config)
                assert not loaded

                _eval_model(sess, inference, x_ph, symbols)

        g = tf.Graph()
        with g.device("/cpu:0"):
            with g.as_default():

                with tf.variable_scope('mnist') as var_scope:
                    x_ph = tf.placeholder(tf.float32, (None, 28, 28))
                    inference = build_model(x_ph, output_size, is_training=False)

                sess = tf.Session()
                loaded = load_or_train(
                    sess, build_model, train_mnist, var_scope,
                    str(checkpoint_dir / 'model.chk'), train_config=config)
                assert loaded

                _eval_model(sess, inference, x_ph, symbols)


def mlp(inp, outp_size, is_training):
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(inp, outp_size)
    return tf.nn.softmax(logits)


def lenet(inp, output_size, is_training):
    logits = LeNet(1024, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)
    return tf.nn.softmax(logits)


@pytest.mark.parametrize('preprocess', [True, False])
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
        n_symbols = 10
        symbols = list(np.random.choice(20, n_symbols, replace=False))
        output_size = n_symbols + 1

        config = MNIST_CONFIG.copy(
            eval_step=100,
            max_steps=10000,
            symbols=list(symbols),
            include_blank=True,
            threshold=0.2)

        checkpoint_dir = make_checkpoint_dir(config)

        g = tf.Graph()
        with g.as_default():

            sess = tf.Session()

            with sess.as_default():
                build_model = MnistPretrained(
                    prepper, build_classifier, model_dir=str(checkpoint_dir), mnist_config=config)
                x_ph = tf.placeholder(tf.float32, (None, 28, 28))
                inference = build_model(x_ph, output_size, is_training=False, preprocess=True)
                assert not build_model.was_loaded

                _eval_model(sess, inference, x_ph, symbols)

        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()

            with sess.as_default():
                build_model = MnistPretrained(
                    prepper, build_classifier, model_dir=str(checkpoint_dir), mnist_config=config)
                x_ph = tf.placeholder(tf.float32, (None, 28, 28))
                inference = build_model(x_ph, output_size, is_training=False, preprocess=True)
                assert build_model.was_loaded

                _eval_model(sess, inference, x_ph, symbols)
