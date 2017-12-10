import numpy as np
import tensorflow as tf

from dps import cfg
from dps.updater import DifferentiableUpdater
from dps.supervised import ClassificationEnv, RegressionEnv
from dps.config import DEFAULT_CONFIG
from dps.datasets import EmnistDataset, OmniglotDataset, SalienceDataset


def get_differentiable_updater(env):
    f = cfg.build_function()
    return DifferentiableUpdater(env, f)


# EMNIST ***************************************


def build_emnist_env():
    train_dataset = EmnistDataset(n_examples=cfg.n_train, one_hot=True)
    val_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=True)
    test_dataset = EmnistDataset(n_examples=cfg.n_val, one_hot=True)
    return ClassificationEnv(train_dataset, val_dataset, test_dataset, one_hot=True)


# For training networks on EMNIST datasets.
EMNIST_CONFIG = DEFAULT_CONFIG.copy(
    log_name='emnist_pretrained',
    get_updater=get_differentiable_updater,
    build_env=build_emnist_env,

    shape=(14, 14),
    batch_size=128,
    eval_step=100,
    max_steps=100000,
    patience=10000,
    lr_schedule="Exp(0.001, 0, 10000, 0.9)",
    optimizer_spec="adam",
    threshold=-np.inf,
    n_train=60000,
    n_val=100,
    include_blank=True,
    classes=list(range(10)),
    n_controller_units=100,
    use_gpu=True,
    gpu_allow_growth=True,
    seed=347405995
)


# OMNIGLOT ***************************************


def build_omniglot_env():
    train_dataset = OmniglotDataset(indices=cfg.train_indices, one_hot=True)
    val_dataset = OmniglotDataset(indices=cfg.val_indices, one_hot=True)
    test_dataset = OmniglotDataset(indices=cfg.test_indices, one_hot=True)
    return ClassificationEnv(train_dataset, val_dataset, test_dataset, one_hot=True)


# For training networks on OMNIGLOT datasets.
OMNIGLOT_CONFIG = DEFAULT_CONFIG.copy(
    log_name='omniglot_pretrained',
    get_updater=get_differentiable_updater,
    build_env=build_omniglot_env,

    shape=(14, 14),
    batch_size=8,
    eval_step=100,
    max_steps=100000,
    patience=10000,
    lr_schedule="Exp(0.001, 0, 10000, 0.9)",
    optimizer_spec="adam",
    threshold=-np.inf,
    train_indices=list(range(15)),
    val_indices=list(range(15, 17)),
    test_indices=list(range(17, 20)),
    include_blank=True,
    classes=["Latin,1", "Latin,2"],
    n_examples=None,
    n_controller_units=100,
    use_gpu=True,
    gpu_allow_growth=True,
    seed=936416219
)


# SALIENCE ***************************************


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

            x = x.reshape(-1, *cfg.image_shape)
            y = y.reshape(-1, *cfg.output_shape)
            _y = _y.reshape(-1, *cfg.output_shape)

            fig, self.axes = plt.subplots(3, n_plots)

            for i in range(n_plots):
                self.axes[0, i].imshow(x[i], vmin=0, vmax=255.0)
                self.axes[1, i].imshow(y[i], vmin=0, vmax=1.0)
                self.axes[2, i].imshow(_y[i], vmin=0, vmax=1.0)

            if cfg.save_plots:
                plt.savefig("salience")

            if cfg.show_plots:
                plt.show(block=True)


def build_salience_env():
    train_dataset = SalienceDataset(n_examples=cfg.n_train)
    val_dataset = SalienceDataset(n_examples=cfg.n_val)
    test_dataset = SalienceDataset(n_examples=cfg.n_val)
    return RegressionEnv(train_dataset, val_dataset, test_dataset)


SALIENCE_CONFIG = DEFAULT_CONFIG.copy(
    log_name='salience_pretrained',
    get_updater=get_differentiable_updater,
    build_env=build_salience_env,
    render_hook=salience_render_hook(),

    min_digits=0,
    max_digits=4,

    flatten_output=True,
    batch_size=32,
    eval_step=1000,
    display_step=1000,
    max_steps=100000,
    patience=10000,
    lr_schedule="Exp(0.0001, 0, 10000, 0.9)",
    optimizer_spec="adam",
    threshold=-np.inf,
    n_train=1000,
    n_val=100,
    classes=EmnistDataset.class_pool,
    max_overlap=20,
    std=0.05,
    sub_image_shape=(14, 14),
    n_sub_image_examples=None,
    image_shape=(3*14, 3*14),
    output_shape=(14, 14),
    gpu_allow_growth=True,
    seed=1723686433
)
