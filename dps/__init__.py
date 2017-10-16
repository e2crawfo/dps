from .utils import ConfigStack, Config

cfg = ConfigStack()

from .config import SystemConfig

def reset_config():
    cfg.clear_stack(SystemConfig())

reset_config()

import matplotlib
matplotlib.use(cfg.mpl_backend)
