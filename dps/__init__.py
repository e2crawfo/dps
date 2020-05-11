try:
    import matplotlib
except ImportError:
    pass
import os
os.environ['PYTHONSTARTUP'] = ''

from .config import DEFAULT_CONFIG
from .utils import ConfigStack

cfg = ConfigStack()

def reset_config():
    cfg.clear_stack(DEFAULT_CONFIG.copy())

reset_config()


def set_trace(context=11):
    """ We define our own trace function which first resets `stdout` and `stderr` to their default values.
        In the dps training loop we overwrite stdout and stderr so that they output to files on disk (in addition
        to the console); however, doing so interferes with the interactive debuggers (messes with completion, command
        history, navigation, etc). Temporarily resetting them to defaults fixes that.

    """
    import sys
    import inspect

    old = sys.stdout, sys.stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    up_one_level = inspect.currentframe().f_back

    try:
        import ipdb
        ipdb.set_trace(up_one_level, context=context)
    except ImportError:
        pdb.Pdb().set_trace(up_one_level)

    sys.stdout, sys.stderr = old
