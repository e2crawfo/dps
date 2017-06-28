from .utils import ConfigStack, Config, DpsConfig

cfg = ConfigStack()

def reset_config():
    cfg.clear_stack(DpsConfig())

reset_config()

import matplotlib
matplotlib.use(cfg.mpl_backend)
