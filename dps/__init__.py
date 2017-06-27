from .utils import ConfigStack, Config, DpsConfig

cfg = ConfigStack()

def reset_config():
    cfg.clear_stack(DpsConfig())

reset_config()

from .register import RegisterBank
from .policy import (
    Policy, ActionSelection, ReluSelect, SoftmaxSelect,
    GumbelSoftmaxSelect, EpsilonGreedySelect)


import matplotlib
matplotlib.use(cfg.mpl_backend)
