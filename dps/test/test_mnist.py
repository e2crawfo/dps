import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from shutil import rmtree
from contextlib import ExitStack

from dps import cfg
from dps.utils import NumpySeed
from dps.utils.tf import MLP, LeNet, SalienceMap
from dps.vision import MNIST_CONFIG, OMNIGLOT_CONFIG, MNIST_SALIENCE_CONFIG, EmnistDataset, OmniglotDataset
from dps.train import load_or_train


def _eval_mnist_model(inference, x_ph):
    test_dataset = EmnistDataset(
        n_examples=100, classes=cfg.classes, one_hot=True,
        include_blank=cfg.include_blank,
        downsample_factor=cfg.downsample_factor)

    x, y = test_dataset.next_batch(advance=False)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=inference))
    correct_prediction = tf.equal(tf.argmax(inference, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.get_default_session()
    _loss, _accuracy = sess.run([loss, accuracy], feed_dict={x_ph: x})
    print("Loss: {}, accuracy: {}".format(_loss, _accuracy))
    assert _accuracy > 0.7


def _eval_omniglot_model(inference, x_ph):
    test_dataset = OmniglotDataset(
        classes=cfg.classes, one_hot=True, include_blank=cfg.include_blank,
        indices=cfg.test_indices, size=cfg.size
    )

    x, y = test_dataset.next_batch(advance=False)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=inference))
    correct_prediction = tf.equal(tf.argmax(inference, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.get_default_session()
    _loss, _accuracy = sess.run([loss, accuracy], feed_dict={x_ph: x})
    print("Loss: {}, accuracy: {}".format(_loss, _accuracy))
    assert _accuracy > 0.7


def make_checkpoint_dir(config, name):
    checkpoint_dir = Path(config.log_root) / name / 'checkpoint'
    try:
        rmtree(str(checkpoint_dir))
    except FileNotFoundError:
        pass
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return checkpoint_dir


def test_mnist_load_or_train():
    with NumpySeed(83849):
        n_classes = 10
        classes = EmnistDataset.sample_classes(n_classes)

        def build_function():
            return MLP([100, 100])

        config = MNIST_CONFIG.copy(
            build_function=build_function,
            patience=np.inf,
            max_steps=10000000,
            classes=classes,
            include_blank=True,
            image_width=14,
            downsample_factor=2,
            threshold=0.2,
            n_controller_units=100
        )

        checkpoint_dir = make_checkpoint_dir(config, 'test_mnist')
        output_size = n_classes + 1

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())

            f = build_function()
            x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
            inference = f(x_ph, output_size, False)

            loaded = load_or_train(config, f.scope, str(checkpoint_dir / 'model.chk'))
            assert not loaded

            with config:
                _eval_mnist_model(inference, x_ph)

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())

            f = build_function()
            x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
            inference = f(x_ph, output_size, False)

            loaded = load_or_train(config, f.scope, str(checkpoint_dir / 'model.chk'))
            assert loaded

            with config:
                _eval_mnist_model(inference, x_ph)


def build_mlp():
    return MLP([cfg.n_controller_units, cfg.n_controller_units])


def build_lenet():
    return LeNet(cfg.n_controller_units)


@pytest.mark.parametrize("build_function", [build_mlp, build_lenet])
def test_mnist_pretrained(build_function):
    with NumpySeed(83849):
        n_classes = 10
        classes = EmnistDataset.sample_classes(n_classes)

        config = MNIST_CONFIG.copy(
            build_function=build_function,
            patience=np.inf,
            max_steps=10000000,
            classes=classes,
            include_blank=True,
            image_width=14,
            downsample_factor=2,
            threshold=0.2,
            n_controller_units=100
        )

        checkpoint_dir = make_checkpoint_dir(config, 'test_mnist')
        output_size = n_classes + 1

        name_params = ['classes', 'include_blank', 'image_width']

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
            inference = f(x_ph, output_size, False)

            assert f.was_loaded is False

            with config:
                _eval_mnist_model(inference, x_ph)

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
            inference = f(x_ph, output_size, False)

            assert f.was_loaded is True

            with config:
                _eval_mnist_model(inference, x_ph)


@pytest.mark.parametrize("build_function", [build_lenet])
def test_omniglot(build_function):
    with NumpySeed(83849):
        n_classes = 10
        classes = OmniglotDataset.sample_classes(n_classes)

        config = OMNIGLOT_CONFIG.copy(
            build_function=build_function,
            patience=np.inf,
            classes=classes,
            include_blank=True,
            threshold=0.001,
        )

        checkpoint_dir = make_checkpoint_dir(config, 'test_omni')
        output_size = n_classes + 1

        name_params = ['classes', 'include_blank']

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())

            with config:
                f = build_function()
                f.set_pretraining_params(config, name_params, checkpoint_dir)
                x_ph = tf.placeholder(tf.float32, (None, np.product(config.size)))

                inference = f(x_ph, output_size, False)

            assert f.was_loaded is False

            with config:
                _eval_omniglot_model(inference, x_ph)

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())

            with config:
                f = build_function()
                f.set_pretraining_params(config, name_params, checkpoint_dir)
                x_ph = tf.placeholder(tf.float32, (None, np.product(config.size)))
                inference = f(x_ph, output_size, False)

            assert f.was_loaded is True

            with config:
                _eval_omniglot_model(inference, x_ph)


class salience_render_hook(object):
    def __init__(self):
        self.axes = None

    def __call__(self, updater):
        print("Rendering...")
        if cfg.show_plots or cfg.save_plots:
            import matplotlib.pyplot as plt
            n_train = n_val = 10
            n_plots = n_train + n_val
            train_x = updater.env.datasets['train'].x[:n_train, ...]
            train_y = updater.env.datasets['train'].y[:n_train, ...]
            val_x = updater.env.datasets['val'].x[:n_val, ...]
            val_y = updater.env.datasets['val'].y[:n_val, ...]

            x = np.concatenate([train_x, val_x], axis=0)
            y = np.concatenate([train_y, val_y], axis=0)

            sess = tf.get_default_session()
            _y = sess.run(updater.output, feed_dict={updater.x_ph: x})

            x = x.reshape(-1, cfg.image_width, cfg.image_width)
            y = y.reshape(-1, cfg.output_width, cfg.output_width)
            _y = _y.reshape(-2, cfg.output_width, cfg.output_width)

            fig, self.axes = plt.subplots(3, n_plots)

            for i in range(n_plots):
                self.axes[0, i].imshow(x[i])
                self.axes[1, i].imshow(y[i])
                self.axes[2, i].imshow(_y[i])

            if cfg.show_plots:
                plt.show(block=True)


def test_mnist_salience_pretrained():
    with NumpySeed(83849):
        # def build_function():
        #     return FullyConvolutional(
        #         [
        #             dict(num_outputs=16, kernel_size=10, activation_fn=tf.nn.relu, padding='valid'),
        #             dict(num_outputs=16, kernel_size=10, activation_fn=tf.nn.relu, padding='valid'),
        #             dict(num_outputs=1, kernel_size=11, activation_fn=None, padding='valid'),
        #         ],
        #         pool=False,
        #         flatten_output=True
        #     )
        # def build_function():
        #     return FullyConvolutional(
        #         [
        #             dict(num_outputs=16, kernel_size=5, activation_fn=tf.nn.relu, padding='valid'),
        #             dict(num_outputs=16, kernel_size=5, activation_fn=tf.nn.relu, padding='valid'),
        #             dict(num_outputs=16, kernel_size=5, activation_fn=tf.nn.relu, padding='valid'),
        #             dict(num_outputs=16, kernel_size=5, activation_fn=tf.nn.relu, padding='valid'),
        #             dict(num_outputs=1, kernel_size=11, activation_fn=None, padding='valid'),
        #         ],
        #         pool=False,
        #         flatten_output=True
        #     )

        config = MNIST_SALIENCE_CONFIG.copy(
            min_digits=1,
            max_digits=4,
            patience=np.inf,
            max_steps=10000000,
            image_width=3*14,
            output_width=14,
            downsample_factor=2,
            threshold=0.,
            render_hook=salience_render_hook(),
            render_step=100000,
            std=0.05
        )

        def build_function():
            return SalienceMap(
                5, MLP([100, 100]),  # LeNet(100),
                (config.output_width, config.output_width),
                std=config.std, flatten_output=True)
        config.build_function = build_function

        checkpoint_dir = make_checkpoint_dir(config, 'test_mnist')
        output_size = 1

        name_params = ['image_width']

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
            f(x_ph, output_size, False)

            assert f.was_loaded is False

            # with config:
            #     _eval_model(inference, x_ph)

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None, config.image_width**2))
            f(x_ph, output_size, False)

            assert f.was_loaded is True

            # with config:
            #     _eval_model(inference, x_ph)
