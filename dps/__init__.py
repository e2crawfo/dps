try:
    import matplotlib
    # matplotlib.use("pdf")
except ImportError:
    pass

from .utils import ConfigStack, Config

cfg = ConfigStack()

def reset_config():
    cfg.clear_stack()

reset_config()
