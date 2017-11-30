import pytest
import tensorflow as tf
from contextlib import ExitStack
import subprocess
from collections import defaultdict
import shutil
import os

from dps import cfg
from dps.utils import NumpySeed
from dps.utils.tf import MLP, LeNet, SalienceMap
from dps.vision import SALIENCE_CONFIG, EMNIST_CONFIG, EmnistDataset, OMNIGLOT_CONFIG, OmniglotDataset
from dps.train import load_or_train


def get_graph_and_session():
    g = tf.Graph()
    session_config = tf.ConfigProto()
    session_config.intra_op_parallelism_threads = 1
    session_config.inter_op_parallelism_threads = 1
    sess = tf.Session(graph=g, config=session_config)
    return g, sess


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
    checkpoint_dir = os.path.join(config.log_root, name)
    try:
        shutil.rmtree(checkpoint_dir)
    except FileNotFoundError:
        pass
    os.makedirs(checkpoint_dir, exist_ok=False)
    return checkpoint_dir


def build_mlp():
    return MLP([cfg.n_controller_units, cfg.n_controller_units])


def build_lenet():
    return LeNet(cfg.n_controller_units)


@pytest.mark.parametrize("build_function", [build_mlp, build_lenet])
def test_emnist_load_or_train(build_function, test_config):
    with NumpySeed(83849):
        n_classes = 10
        classes = EmnistDataset.sample_classes(n_classes)

        config = EMNIST_CONFIG.copy(
            build_function=build_function,
            classes=classes,
            threshold=0.1,
            stopping_criteria_name="01_loss",
            n_controller_units=100,
        )
        config.update(test_config)

        checkpoint_dir = make_checkpoint_dir(config, 'test_emnist')
        output_size = n_classes + 1

        g, sess = get_graph_and_session()
        with ExitStack() as stack:
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)
            inference = f(x_ph, output_size, False)

            loaded = load_or_train(
                config, f.scope, os.path.join(checkpoint_dir, 'model'))
            assert not loaded

            test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=True)
            _eval_model(test_dataset, inference, x_ph)

        g, sess = get_graph_and_session()
        with ExitStack() as stack:
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)
            inference = f(x_ph, output_size, False)

            loaded = load_or_train(
                config, f.scope, os.path.join(checkpoint_dir, 'model'))
            assert loaded

            test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=True)
            _eval_model(test_dataset, inference, x_ph)


@pytest.mark.parametrize("build_function", [build_mlp, build_lenet])
def test_emnist_pretrained(build_function, test_config):
    with NumpySeed(83849):
        n_classes = 10
        classes = EmnistDataset.sample_classes(n_classes)

        config = EMNIST_CONFIG.copy(
            build_function=build_function,
            classes=classes,
            threshold=0.1,
            stopping_criteria_name="01_loss",
            n_controller_units=100,
        )
        config.update(test_config)

        checkpoint_dir = make_checkpoint_dir(config, 'test_emnist')
        output_size = n_classes + 1

        name_params = 'classes include_blank shape n_controller_units'

        g, sess = get_graph_and_session()
        with ExitStack() as stack:
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
                test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=True)
                _eval_model(test_dataset, inference, x_ph)

        g, sess = get_graph_and_session()
        with ExitStack() as stack:
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
                test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=True)
                _eval_model(test_dataset, inference, x_ph)


@pytest.mark.parametrize("build_function", [build_mlp, build_lenet])
def test_omniglot(build_function, test_config):
    with NumpySeed(83849):
        classes = ["Greek,18", "Greek,19"]
        n_classes = len(classes)

        config = OMNIGLOT_CONFIG.copy(
            patience=10000,
            build_function=build_function,
            classes=classes,
            n_controller_units=100,
            threshold=0.2,
            stopping_criteria_name="01_loss",
            train_indices=list(range(15)),
            val_indices=list(range(15, 20)),
            test_indices=list(range(15, 20)),
        )
        config.update(test_config)

        checkpoint_dir = make_checkpoint_dir(config, 'test_omni')
        output_size = n_classes + 1

        name_params = 'classes include_blank shape n_controller_units'

        g, sess = get_graph_and_session()

        with ExitStack() as stack:
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)

            inference = f(x_ph, output_size, False)

            test_dataset = OmniglotDataset(indices=cfg.test_indices, one_hot=True)
            _eval_model(test_dataset, inference, x_ph)

        g, sess = get_graph_and_session()

        with ExitStack() as stack:
            stack.enter_context(g.as_default())
            stack.enter_context(sess)
            stack.enter_context(sess.as_default())
            stack.enter_context(config)

            f = build_function()
            f.set_pretraining_params(config, name_params, checkpoint_dir)
            x_ph = tf.placeholder(tf.float32, (None,) + config.shape)
            inference = f(x_ph, output_size, False)

            assert f.was_loaded is True

            test_dataset = OmniglotDataset(indices=cfg.test_indices, one_hot=True)
            _eval_model(test_dataset, inference, x_ph)


@pytest.mark.slow
def test_salience_pretrained(test_config):
    with NumpySeed(83849):
        config = SALIENCE_CONFIG.copy(
            classes=list(range(10)),
            threshold=0.01,  # Can get down to 0.005, but takes too long for a test
            n_sub_image_examples=2000,
        )
        config.update(test_config)

        def build_function():
            return SalienceMap(
                5, MLP([100, 100, 100]), cfg.output_shape,
                std=cfg.std, flatten_output=cfg.flatten_output)
        config.build_function = build_function

        checkpoint_dir = make_checkpoint_dir(config, 'test_emnist')
        output_size = 1

        name_params = 'classes std min_digits max_digits sub_image_shape image_shape output_shape'

        g, sess = get_graph_and_session()

        with ExitStack() as stack:
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

        g, sess = get_graph_and_session()

        with ExitStack() as stack:
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


def _train_classifier(build_function, config, name_params, output_size, checkpoint_dir):
    checkpoint_dir = str(checkpoint_dir)
    g, sess = get_graph_and_session()

    with ExitStack() as stack:
        stack.enter_context(g.as_default())
        stack.enter_context(sess)
        stack.enter_context(sess.as_default())
        stack.enter_context(config)

        f = build_function()
        f.set_pretraining_params(config, name_params, checkpoint_dir)
        x_ph = tf.placeholder(tf.float32, (None,) + config.shape)

        # Trigger training
        f(x_ph, output_size, False)


def _get_deterministic_output(d):
    return subprocess.check_output(
        'grep "train_data\|val_data\|test_data" {}/*.stdout | cat -n'.format(d), shell=True).decode()


determinism_info = dict(
    emnist=dict(
        config=EMNIST_CONFIG,
        sample_classes=EmnistDataset.sample_classes,
    ),
    omni=dict(
        config=OMNIGLOT_CONFIG,
        sample_classes=OmniglotDataset.sample_classes
    ),
)


@pytest.mark.parametrize("dataset", "emnist omni".split())
def test_determinism(dataset, test_config):
    build_function = build_mlp  # Can't use build_lenet here as it is slightly non-deterministic for reasons unknown.
    with NumpySeed(83849):
        n_classes = 10

        info = determinism_info[dataset]

        classes = info['sample_classes'](n_classes)
        config = info['config'].copy(
            build_function=build_function,
            classes=classes,
            n_controller_units=100,
            threshold=0.2,
            stopping_criteria_name="01_loss",
            seed=334324923,
            display_step=100,
            eval_step=100,
            max_steps=1001,
            tee=False,
            n_train=500,
        )
        config.update(test_config)

        name_params = 'classes include_blank shape n_controller_units'
        output_size = n_classes + 1

        n_repeats = 10

        output = defaultdict(int)

        dir_names = []
        try:
            for i in range(n_repeats):
                checkpoint_dir = make_checkpoint_dir(config, 'test_{}_{}'.format(dataset, i))
                dir_names.append(checkpoint_dir)
                _train_classifier(build_function, config, name_params, output_size, checkpoint_dir)
                o = _get_deterministic_output(checkpoint_dir)
                output[o] += 1

            if len(output) != 1:
                for o in sorted(output):
                    print("\n" + "*" * 80)
                    print("The following occurred {} times:\n".format(output[o]))
                    print(o)
                raise Exception("Results were not deterministic.")
        finally:
            for dn in dir_names:
                try:
                    shutil.rmtree(dn)
                except FileNotFoundError:
                    pass
