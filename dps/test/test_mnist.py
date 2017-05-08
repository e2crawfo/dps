import numpy as np
import tensorflow as tf
from pathlib import Path
from shutil import rmtree

from dps.attention import DRAW_attention_2D
from dps.utils import load_or_train, MLP, NumpySeed
from dps.experiments.mnist import TranslatedMnistDataset, train_mnist, MnistConfig


def _eval_model(sess, inference, x_ph):
    test_dataset = TranslatedMnistDataset(28, 1, np.inf, 10, for_eval=True)
    y_ph = tf.placeholder(tf.int64, (None))
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_ph, logits=inference))
    correct_prediction = tf.equal(tf.argmax(inference, 1), y_ph)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    x, y = test_dataset.next_batch()
    _loss, _accuracy = sess.run([loss, accuracy], {x_ph: x, y_ph: y})
    print("Loss: {}, accuracy: {}".format(_loss, _accuracy))
    assert _accuracy > 0.7


def build_model(inp):
    batch_size = tf.shape(inp)[0]
    N = 14
    glimpse = DRAW_attention_2D(
        tf.reshape(inp, (-1, 28, 28)),
        fovea=tf.zeros((batch_size, 2)),
        delta=tf.ones((batch_size, 1)),
        std=tf.ones((batch_size, 1)),
        N=N)
    glimpse = tf.reshape(glimpse, (-1, N**2))
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(glimpse, 10)
    return logits


def test_mnist_pretraining():
    with NumpySeed(83849):
        g = tf.Graph()
        with g.as_default():
            with tf.variable_scope('mnist') as var_scope:
                obs_dim = 28 ** 2
                x_ph = tf.placeholder(tf.float32, (None, obs_dim))
                inference = build_model(x_ph)
            sess = tf.Session()

            checkpoint_dir = Path('/tmp/mnist_training/checkpoint')
            try:
                rmtree(str(checkpoint_dir))
            except FileNotFoundError:
                pass

            checkpoint_dir.mkdir(parents=True, exist_ok=False)

            loaded = load_or_train(
                sess, build_model, train_mnist, var_scope,
                str(checkpoint_dir / 'model.chk'), MnistConfig())
            assert not loaded
            _eval_model(sess, inference, x_ph)

        g = tf.Graph()
        with g.as_default():
            with tf.variable_scope('mnist') as var_scope:
                obs_dim = 28 ** 2
                x_ph = tf.placeholder(tf.float32, (None, obs_dim))
                inference = build_model(x_ph)
            sess = tf.Session()

            loaded = load_or_train(
                sess, build_model, train_mnist, var_scope,
                str(checkpoint_dir / 'model.chk'), MnistConfig())
            assert loaded
            _eval_model(sess, inference, x_ph)
