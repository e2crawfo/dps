try:
    import matplotlib
except ImportError:
    pass
import os
os.environ['PYTHONSTARTUP'] = ''

from .utils import ConfigStack, get_default_config, process_path


def init(config=None):
    """ Create directories requires by dps. """
    if config is None:
        config = cfg

    make_dirs = config.get('make_dirs', True)

    def fixup_dir(name):
        attr_name = name + "_dir"
        dir_name = getattr(config, attr_name, None)
        if dir_name is None:
            dir_name = os.path.join(config.scratch_dir, name)
            dir_name = process_path(dir_name)
            setattr(config, attr_name, dir_name)

        if make_dirs:
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                print("Unable to create directory {}.".format(dir_name))
                traceback.print_exc()

    fixup_dir("data")
    fixup_dir("local_experiments")
    fixup_dir("parallel_experiments_build")
    fixup_dir("parallel_experiments_run")

    return config


cfg = ConfigStack()

def reset_config():
    cfg.clear_stack(get_default_config())

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
