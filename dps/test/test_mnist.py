import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from shutil import rmtree

from dps import cfg
from dps.utils import MLP, NumpySeed
from dps.vision import (
    load_or_train, MNIST_CONFIG, MnistPretrained, DRAW, LeNet, train_mnist, EmnistDataset)
from dps.vision.attention import DRAW_attention_2D


N = 7


class_pool = (
    [str(i) for i in range(10)] + [chr(i + ord('A')) for i in range(26)] +
    [chr(i + ord('a')) for i in range(26)]
)


def _eval_model(sess, inference, x_ph):
    test_dataset = EmnistDataset(
        n_examples=100, classes=cfg.classes, one_hot=True,
        include_blank=cfg.include_blank,
        downsample_factor=cfg.downsample_factor)

    x, y = test_dataset.next_batch(advance=False)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=inference))
    correct_prediction = tf.equal(tf.argmax(inference, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    _loss, _accuracy = sess.run([loss, accuracy], feed_dict={x_ph: x})
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
            N=N
        )
    glimpse = tf.reshape(glimpse, (-1, N**2))
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(inp, output_size)
    return logits


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
        n_classes = 10
        classes = np.random.choice(len(class_pool), n_classes, replace=False)
        classes = [class_pool[i] for i in classes]

        config = MNIST_CONFIG.copy(
            patience=np.inf,
            max_steps=10000000,
            classes=classes,
            include_blank=True,
            image_width=14,
            downsample_factor=2,
            threshold=0.2,
        )

        checkpoint_dir = make_checkpoint_dir(config)
        output_size = n_classes + 1

        g = tf.Graph()
        with g.device("/cpu:0"):
            with g.as_default():

                with tf.variable_scope('mnist') as var_scope:
                    x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
                    inference = build_model(x_ph, output_size, is_training=False)

                sess = tf.Session()
                loaded = load_or_train(
                    sess, build_model, train_mnist, var_scope,
                    str(checkpoint_dir / 'model.chk'), train_config=config)
                assert not loaded

                with config:
                    _eval_model(sess, inference, x_ph)

        g = tf.Graph()
        with g.device("/cpu:0"):
            with g.as_default():

                with tf.variable_scope('mnist') as var_scope:
                    x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
                    inference = build_model(x_ph, output_size, is_training=False)

                sess = tf.Session()
                loaded = load_or_train(
                    sess, build_model, train_mnist, var_scope,
                    str(checkpoint_dir / 'model.chk'), train_config=config)
                assert loaded

                with config:
                    _eval_model(sess, inference, x_ph)


def mlp(inp, outp_size, is_training):
    logits = MLP([100, 100], activation_fn=tf.nn.sigmoid)(inp, outp_size)
    return logits


def lenet(inp, output_size, is_training):
    logits = LeNet(1024, activation_fn=tf.nn.sigmoid)(inp, output_size, is_training)
    return logits


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
        n_classes = 10
        classes = np.random.choice(len(class_pool), n_classes, replace=False)
        classes = [class_pool[i] for i in classes]
        output_size = n_classes + 1

        config = MNIST_CONFIG.copy(
            patience=np.inf,
            max_steps=10000000,
            classes=classes,
            include_blank=True,
            image_width=14,
            downsample_factor=2,
            threshold=0.2,
        )

        checkpoint_dir = make_checkpoint_dir(config)

        g = tf.Graph()
        with g.as_default():

            sess = tf.Session()

            with sess.as_default():
                build_model = MnistPretrained(
                    prepper, build_classifier, model_dir=str(checkpoint_dir), mnist_config=config)
                x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
                inference = build_model(x_ph, output_size, is_training=False, preprocess=True)
                assert not build_model.was_loaded

                with config:
                    _eval_model(sess, inference, x_ph)

        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()

            with sess.as_default():
                build_model = MnistPretrained(
                    prepper, build_classifier, model_dir=str(checkpoint_dir), mnist_config=config)
                x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
                inference = build_model(x_ph, output_size, is_training=False, preprocess=True)
                assert build_model.was_loaded

                with config:
                    _eval_model(sess, inference, x_ph)
