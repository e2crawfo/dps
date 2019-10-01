try:
    import matplotlib
    # matplotlib.use("pdf")
except ImportError:
    pass

from .config import DEFAULT_CONFIG
from .utils import ConfigStack

cfg = ConfigStack()

def reset_config():
    cfg.clear_stack(DEFAULT_CONFIG.copy())

reset_config()
