import numpy as np
import tensorflow as tf
from pathlib import Path
from shutil import rmtree

from dps.attention import DRAW_attention_2D
from dps.utils import MLP, NumpySeed
from dps.mnist import (
    TranslatedMnistDataset, load_or_train, MnistConfig, MnistPretrained, DRAW, train_mnist)


n_symbols = 20
N = 14


def _eval_model(sess, inference, x_ph):
    test_dataset = TranslatedMnistDataset(28, 1, np.inf, 10, for_eval=True, symbols=list(range(n_symbols)))
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


def build_model(inp):
    batch_size = tf.shape(inp)[0]
    glimpse = DRAW_attention_2D(
        tf.reshape(inp, (-1, 28, 28)),
        fovea_x=tf.zeros((batch_size, 1)),
        fovea_y=tf.zeros((batch_size, 1)),
        delta=tf.ones((batch_size, 1)),
        std=tf.ones((batch_size, 1)),
        N=N)
    glimpse = tf.reshape(glimpse, (-1, N**2))
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(glimpse, n_symbols)
    return tf.nn.softmax(logits)


def test_mnist_load_or_train():
    with NumpySeed(83849):
        config = MnistConfig(max_steps=10000, symbols=list(range(n_symbols)))
        g = tf.Graph()
        with g.as_default():
            with tf.variable_scope('mnist') as var_scope:
                x_ph = tf.placeholder(tf.float32, (None, 28**2))
                inference = build_model(x_ph)
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
            _eval_model(sess, inference, x_ph)

        g = tf.Graph()
        with g.as_default():
            with tf.variable_scope('mnist') as var_scope:
                x_ph = tf.placeholder(tf.float32, (None, 28**2))
                inference = build_model(x_ph)
            sess = tf.Session()

            loaded = load_or_train(
                sess, build_model, train_mnist, var_scope,
                str(checkpoint_dir / 'model.chk'), config=config)
            assert loaded
            _eval_model(sess, inference, x_ph)


def build_classifier(inp, outp_size):
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(inp, outp_size)
    return tf.nn.softmax(logits)


def test_mnist_pretrained():
    with NumpySeed(83849):
        config = MnistConfig(max_steps=10000, symbols=list(range(n_symbols)))
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
                    DRAW(N), build_classifier, model_dir=str(checkpoint_dir),
                    preprocess=True, config=config)
                x_ph = tf.placeholder(tf.float32, (None, 28**2))
                inference = build_model(x_ph, preprocess=True)
                assert not build_model.was_loaded

                _eval_model(sess, inference, x_ph)

        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()

            with sess.as_default():
                build_model = MnistPretrained(
                    DRAW(N), build_classifier, model_dir=str(checkpoint_dir),
                    preprocess=True, config=config)
                x_ph = tf.placeholder(tf.float32, (None, 28**2))
                inference = build_model(x_ph, preprocess=True)
                assert build_model.was_loaded

                _eval_model(sess, inference, x_ph)


def test_mnist_pretrained_no_preprocess():
    with NumpySeed(83849):
        config = MnistConfig(max_steps=10000, symbols=list(range(n_symbols)))
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
                    None, build_classifier, model_dir=str(checkpoint_dir), config=config)
                x_ph = tf.placeholder(tf.float32, (None, 28**2))
                inference = build_model(x_ph, preprocess=True)
                assert not build_model.was_loaded

                _eval_model(sess, inference, x_ph)

        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()

            with sess.as_default():
                build_model = MnistPretrained(
                    None, build_classifier, model_dir=str(checkpoint_dir), config=config)
                x_ph = tf.placeholder(tf.float32, (None, 28**2))
                inference = build_model(x_ph, preprocess=True)
                assert build_model.was_loaded

                _eval_model(sess, inference, x_ph)
