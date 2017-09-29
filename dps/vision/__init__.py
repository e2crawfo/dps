from dps import cfg
from dps.environment import RegressionDataset
from dps.utils import Parameterized, Param
from mnist_arithmetic import MnistArithmeticDataset as ExternalDataset


class MnistArithmeticDataset(ExternalDataset, RegressionDataset):
    image_width = Param(28)
    max_overlap = Param(100)
    min_digits = Param(1)
    max_digits = Param(1)
    reductions = Param()
    base = Param(10)
    downsample_factor = Param(1)

    def __init__(self, *args, **kwargs):
        super(MnistArithmeticDataset, self).__init__(cfg.data_dir, *args, **kwargs)

from .attention import DRAW, DiscreteAttn
from .train import (
    MnistPretrained, MNIST_CONFIG, ClassifierFunc, LeNet, VGGNet, load_or_train, train_mnist, EmnistDataset)
