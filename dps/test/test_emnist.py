import pytest
import tensorflow as tf
from pathlib import Path
from shutil import rmtree
from contextlib import ExitStack

from dps import cfg
from dps.utils import NumpySeed
from dps.utils.tf import MLP, LeNet, SalienceMap
from dps.vision import SALIENCE_CONFIG, EMNIST_CONFIG, EmnistDataset, OMNIGLOT_CONFIG, OmniglotDataset
from dps.train import load_or_train


def _eval_model(test_dataset, inference, x_ph):
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


def build_mlp():
    return MLP([cfg.n_controller_units, cfg.n_controller_units])


def build_lenet():
    return LeNet(cfg.n_controller_units)


@pytest.mark.parametrize("build_function", [build_mlp, build_lenet])
def test_emnist_load_or_train(build_function):
    with NumpySeed(83849):
        n_classes = 10
        classes = EmnistDataset.sample_classes(n_classes)

        config = EMNIST_CONFIG.copy(
            build_function=build_function,
            classes=classes,
            threshold=0.2,
            n_controller_units=100
        )

        checkpoint_dir = make_checkpoint_dir(config, 'test_emnist')
        output_size = n_classes + 1

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)
            inference = f(x_ph, output_size, False)

            loaded = load_or_train(config, f.scope, str(checkpoint_dir / 'model.chk'))
            assert not loaded

            test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=cfg.loss_type == 'xent')
            _eval_model(test_dataset, inference, x_ph)

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)
            inference = f(x_ph, output_size, False)

            loaded = load_or_train(config, f.scope, str(checkpoint_dir / 'model.chk'))
            assert loaded

            test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=cfg.loss_type == 'xent')
            _eval_model(test_dataset, inference, x_ph)


@pytest.mark.parametrize("build_function", [build_mlp, build_lenet])
def test_emnist_pretrained(build_function):
    with NumpySeed(83849):
        n_classes = 10
        classes = EmnistDataset.sample_classes(n_classes)

        config = EMNIST_CONFIG.copy(
            build_function=build_function,
            classes=classes,
            threshold=0.2,
            n_controller_units=100
        )

        checkpoint_dir = make_checkpoint_dir(config, 'test_emnist')
        output_size = n_classes + 1

        name_params = 'classes include_blank shape n_controller_units'

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)
            inference = f(x_ph, output_size, False)

            assert f.was_loaded is False

            with config:
                test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=cfg.loss_type == 'xent')
                _eval_model(test_dataset, inference, x_ph)

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)
            inference = f(x_ph, output_size, False)

            assert f.was_loaded is True

            with config:
                test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=cfg.loss_type == 'xent')
                _eval_model(test_dataset, inference, x_ph)


@pytest.mark.parametrize("build_function", [build_lenet])
def test_omniglot(build_function):
    with NumpySeed(83849):
        n_classes = 10
        classes = OmniglotDataset.sample_classes(n_classes)

        config = OMNIGLOT_CONFIG.copy(
            build_function=build_function,
            classes=classes,
            threshold=0.2,
            n_controller_units=100
        )

        checkpoint_dir = make_checkpoint_dir(config, 'test_omni')
        output_size = n_classes + 1

        name_params = 'classes include_blank shape n_controller_units'

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)

            inference = f(x_ph, output_size, False)

            test_dataset = OmniglotDataset(indices=cfg.test_indices, one_hot=cfg.loss_type == 'xent')
            _eval_model(test_dataset, inference, x_ph)

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)
            inference = f(x_ph, output_size, False)

            assert f.was_loaded is True

            test_dataset = OmniglotDataset(indices=cfg.test_indices, one_hot=cfg.loss_type == 'xent')
            _eval_model(test_dataset, inference, x_ph)


def test_salience_pretrained(show_plots, save_plots):
    with NumpySeed(83849):
        config = SALIENCE_CONFIG.copy(
            show_plots=show_plots,
            save_plots=save_plots,
            classes=list(range(10)),
        )

        def build_function():
            return SalienceMap(
                5, MLP([100, 100, 100]), cfg.output_shape,
                std=cfg.std, flatten_output=cfg.flatten_output)
        config.build_function = build_function

        checkpoint_dir = make_checkpoint_dir(config, 'test_emnist')
        output_size = 1

        name_params = 'classes std min_digits max_digits sub_image_shape image_shape output_shape'

        g = tf.Graph()
        sess = tf.Session(graph=g)
        with ExitStack() as stack:
            stack.enter_context(g.device("/cpu:0"))
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None,) + config.image_shape)
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
            stack.enter_context(config)

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None,) + config.image_shape)
            f(x_ph, output_size, False)

            assert f.was_loaded is True

            # with config:
            #     _eval_model(inference, x_ph)
