import numpy as np
import tensorflow as tf

from dps import cfg
from dps.updater import DifferentiableUpdater
from dps.env.supervised import ClassificationEnv, RegressionEnv
from dps.config import DEFAULT_CONFIG
from dps.datasets import EmnistDataset, OmniglotDataset


def get_differentiable_updater(env):
    f = cfg.build_function()
    return DifferentiableUpdater(env, f)


# EMNIST ***************************************


def build_emnist_env():
    train = EmnistDataset(n_examples=cfg.n_train, one_hot=True, example_range=(0., 0.9))
    val = EmnistDataset(n_examples=cfg.n_val, one_hot=True, example_range=(0.9, 0.95))
    return ClassificationEnv(train, val, one_hot=True)


# For training networks on EMNIST datasets.
EMNIST_CONFIG = DEFAULT_CONFIG.copy(
    name="emnist",
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
    seed=347405995,
    stopping_criteria="01_loss,min",
)


# OMNIGLOT ***************************************


def build_omniglot_env():
    train = OmniglotDataset(indices=cfg.train_indices, one_hot=True)
    val = OmniglotDataset(indices=cfg.val_indices, one_hot=True)
    return ClassificationEnv(train, val, one_hot=True)


# For training networks on OMNIGLOT datasets.
OMNIGLOT_CONFIG = DEFAULT_CONFIG.copy(
    name="omniglot",
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
    seed=936416219,
    stopping_criteria="01_loss,min",
)
