from .utils import ConfigStack, Config, SystemConfig

cfg = ConfigStack()

def reset_config():
    cfg.clear_stack()

reset_config()
