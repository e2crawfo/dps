import numpy as np

from dps import cfg
from dps.updater import DifferentiableUpdater
from dps.environment import RegressionEnv, RegressionDataset
from dps.utils import Param
from dps.config import DEFAULT_CONFIG

from mnist_arithmetic import load_emnist
from mnist_arithmetic import MnistArithmeticDataset as _MnistArithmeticDataset
from mnist_arithmetic import MnistSalienceDataset as _MnistSalienceDataset


class MnistArithmeticDataset(RegressionDataset):
    image_width = Param(28)
    max_overlap = Param(100)
    min_digits = Param(1)
    max_digits = Param(1)
    reductions = Param()
    base = Param(10)
    downsample_factor = Param(1)
    n_examples = Param()

    def __init__(self, *args, **kwargs):
        _dataset = _MnistArithmeticDataset(cfg.data_dir, **self.param_values())
        super(MnistArithmeticDataset, self).__init__(_dataset.x, _dataset.y)


class EmnistDataset(RegressionDataset):
    class_pool = ''.join(
        [str(i) for i in range(10)] +
        [chr(i + ord('A')) for i in range(26)] +
        [chr(i + ord('a')) for i in range(26)]
    )
    n_examples = Param()

    @staticmethod
    def sample_classes(n_classes):
        classes = np.random.choice(len(EmnistDataset.class_pool), n_classes, replace=False)
        return [EmnistDataset.class_pool[i] for i in classes]

    def __init__(self, n_examples, classes, **kwargs):
        x, y, class_map = load_emnist(cfg.data_dir, classes, max_examples=n_examples, **kwargs)
        if x.shape[0] < n_examples:
            raise Exception(
                "Too few datapoints. Requested {}, "
                "only {} are available.".format(n_examples, x.shape[0]))
        super(EmnistDataset, self).__init__(x, y)


class MnistSalienceDataset(RegressionDataset):
    classes = Param()
    min_digits = Param(1)
    max_digits = Param(1)
    downsample_factor = Param(1)
    image_width = Param(100)
    max_overlap = Param(200)
    output_width = Param(10)
    std = Param(0.1)
    flatten_output = Param(False)
    n_examples = Param()

    def __init__(self, *args, **kwargs):
        _dataset = _MnistSalienceDataset(cfg.data_dir, **self.param_values())
        super(MnistSalienceDataset, self).__init__(_dataset.x, _dataset.y)


def get_updater(env):
    f = cfg.build_function()
    return DifferentiableUpdater(env, f)


def build_mnist_env():
    train_dataset = EmnistDataset(
        n_examples=cfg.n_train, classes=cfg.classes,
        downsample_factor=cfg.downsample_factor, one_hot=True,
        include_blank=cfg.include_blank
    )
    val_dataset = EmnistDataset(
        n_examples=cfg.n_val, classes=cfg.classes,
        downsample_factor=cfg.downsample_factor, one_hot=True,
        include_blank=cfg.include_blank
    )
    test_dataset = EmnistDataset(
        n_examples=cfg.n_val, classes=cfg.classes,
        downsample_factor=cfg.downsample_factor, one_hot=True,
        include_blank=cfg.include_blank
    )
    return RegressionEnv(train_dataset, val_dataset, test_dataset)


MNIST_CONFIG = DEFAULT_CONFIG.copy(
    get_updater=get_updater,
    build_env=build_mnist_env,
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
    log_name='mnist_pretrained',
    downsample_factor=1,
    loss_type='xent',
    classes=list(range(10)),
)


def build_mnist_salience_env():
    train_dataset = MnistSalienceDataset(
        n_examples=cfg.n_train, flatten_output=True
    )
    val_dataset = MnistSalienceDataset(
        n_examples=cfg.n_val, flatten_output=True
    )
    test_dataset = MnistSalienceDataset(
        n_examples=cfg.n_val, flatten_output=True
    )
    return RegressionEnv(train_dataset, val_dataset, test_dataset)


MNIST_SALIENCE_CONFIG = DEFAULT_CONFIG.copy(
    get_updater=get_updater,
    build_env=build_mnist_salience_env,
    batch_size=32,
    eval_step=1000,
    display_step=10,
    max_steps=100000,
    patience=10000,
    lr_schedule="Exp(0.0001, 0, 10000, 0.9)",
    optimizer_spec="adam",
    threshold=-np.inf,
    n_train=10000,
    n_val=100,
    log_name='mnist_salience_pretrained',
    downsample_factor=2,
    loss_type='2-norm',
    classes=EmnistDataset.class_pool,
    max_overlap=20,
    std=0.01,
    output_width=14,
    image_width=3*14,
    flatten_output=True,
)
