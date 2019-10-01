import tensorflow_datasets as tfds
import copy

from dps.utils import Config
from dps.mnist_example import Updater as _Updater, mlp_config as _mlp_config
from dps.datasets.base import Environment


class Updater(_Updater):
    feature_name = 'features'


class IrisEnvironment(Environment):
    def __init__(self):
        dsets, info = tfds.load('iris:2.*.*', with_info=True)
        train = dsets['train']
        val = dsets['train']
        test = dsets['train']
        train.n_classes = 3
        self.datasets = dict(train=train, val=val, test=test)


def iris_config_func():
    env_name = 'iris'
    eval_step = 1000
    display_step = -1
    checkpoint_step = -1
    weight_step = -1
    backup_step = -1

    shuffle_buffer_size = 10000

    build_env = IrisEnvironment

    return locals()


iris_config = Config(iris_config_func())
mlp_config = copy.deepcopy(_mlp_config)
mlp_config['get_updater'] = Updater
