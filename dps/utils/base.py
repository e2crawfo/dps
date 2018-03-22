from pprint import pformat
from contextlib import contextmanager
import numpy as np
import signal
import time
import re
import os
import traceback
import pdb
from collections.abc import MutableMapping
import subprocess
import copy
import datetime
import psutil
import resource
import sys
import shutil
import pandas as pd
import errno
from tempfile import NamedTemporaryFile
import dill
from functools import wraps
import inspect
import hashlib
import configparser
import socket
from zipfile import ZipFile
from scipy import stats

import clify
import dps


def square_subplots(N):
    sqrt_N = int(np.ceil(np.sqrt(N)))
    m = int(np.ceil(N / sqrt_N))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(m, sqrt_N)
    return fig, axes


def nvidia_smi():
    try:
        p = subprocess.run("nvidia-smi".split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.stdout.decode()
    except Exception as e:
        return "Exception while calling nvidia-smi: {}".format(e)


def view_readme_cl():
    return view_readme(".")


def view_readme(path):
    """ View readme files in a diretory of experiments, sorted by the time at
        which the experiment began execution.

    """
    import iso8601

    command = "find {} -name README.md".format(path).split()
    p = subprocess.run(command, stdout=subprocess.PIPE)
    readme_paths = [r for r in p.stdout.decode().split('\n') if r]
    datetimes = []
    for r in readme_paths:
        d = os.path.split(r)[0]
        with open(os.path.join(d, 'stdout'), 'r') as f:
            line = ''
            while not line.startswith("Starting training run"):
                line = f.readline()
        tokens = line.split()
        assert len(tokens) == 13
        dt = iso8601.parse_date(tokens[5] + " " + tokens[6][:-1])
        datetimes.append(dt)

    _sorted = sorted(zip(datetimes, readme_paths))

    for d, r in _sorted:
        print("\n" + "-" * 80 + "\n\n" + "====> {} <====".format(r))
        with open(r, 'r') as f:
            print(f.read())


def confidence_interval(data, coverage):
    return stats.t.interval(
        coverage, len(data)-1, loc=np.mean(data), scale=stats.sem(data))


def standard_error(data):
    return stats.sem(data)


def zip_root(zipfile):
    """ Get the name of the root directory inside a zip file, if it has one. """

    if not isinstance(zipfile, ZipFile):
        zipfile = ZipFile(zipfile, 'r')

    zip_root = min(
        (z.filename for z in zipfile.infolist()),
        key=lambda s: len(s))

    if zip_root.endswith('/'):
        zip_root = zip_root[:-1]

    return zip_root


def get_param_hash(d, name_params=None):
    if not name_params:
        name_params = d.keys()
    param_str = []
    for name in name_params:
        value = d[name]

        if callable(value):
            value = inspect.getsource(value)

        param_str.append("{}={}".format(name, value))
    param_str = "_".join(param_str)
    param_hash = hashlib.sha1(param_str.encode()).hexdigest()
    return param_hash


CLEAR_CACHE = False


def set_clear_cache(value):
    """ If called with True, then whenever `sha_cache` function is instantiated, it will ignore
        any cache saved to disk, and instead just call the function as normal, saving the results
        as the new cache value. """
    global CLEAR_CACHE
    CLEAR_CACHE = value


def sha_cache(directory, recurse=False):
    os.makedirs(directory, exist_ok=True)

    def decorator(func):
        sig = inspect.signature(func)

        def new_f(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            param_hash = get_param_hash(bound_args.arguments)
            filename = os.path.join(directory, "{}_{}.cache".format(func.__name__, param_hash))

            loaded = False
            try:
                if not CLEAR_CACHE:
                    with open(filename, 'rb') as f:
                        value = dill.load(f)
                    loaded = True
            except FileNotFoundError:
                pass
            finally:
                if not loaded:
                    value = func(**bound_args.arguments)
                    with open(filename, 'wb') as f:
                        dill.dump(value, f, protocol=dill.HIGHEST_PROTOCOL, recurse=recurse)
            return value
        return new_f
    return decorator


def _run_cmd(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()


class GitSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def summarize(self, n_logs=10, diff=False):
        s = []
        with cd(self.directory):
            s.append("*" * 40)
            s.append("GitSummary for directory {}\n".format(self.directory))

            s.append("log:\n")
            log = _run_cmd('git log -n {}'.format(n_logs))
            s.append(log)

            s.append("\nstatus:\n")
            status = _run_cmd('git status --porcelain')
            s.append(status)

            s.append("\ndiff:\n")
            if diff:
                diff = _run_cmd('git diff')
                s.append(diff)
            else:
                s.append("<ommitted>")

            s.append("\nEnd of GitSummary for directory {}".format(self.directory))
            s.append("*" * 40)
        return '\n'.join(s)

    def freeze(self):
        pass


def module_git_summary(module, **kwargs):
    module_dir = os.path.dirname(module.__file__)
    return GitSummary(module_dir)


def pip_freeze(**kwargs):
    return _run_cmd('pip freeze')


def one_hot(indices, depth):
    array = np.zeros(indices.shape + (depth,))
    batch_indices = np.unravel_index(range(indices.size), indices.shape)
    array[batch_indices + (indices.flatten(),)] = 1.0
    return array


@contextmanager
def remove(filenames):
    try:
        yield
    finally:
        if isinstance(filenames, str):
            filenames = filenames.split()
        for fn in filenames:
            try:
                shutil.rmtree(fn)
            except NotADirectoryError:
                os.remove(fn)
            except FileNotFoundError:
                pass


@contextmanager
def modify_env(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.

    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def make_symlink(target, name):
    """ NB: ``target`` is just used as a simple string when creating
    the link. That is, ``target`` is the location of the file we want
    to point to, relative to the location that the link resides.
    It is not the case that the target file is identified, and then
    some kind of smart process occurs to have the link point to that file.

    """
    try:
        os.remove(name)
    except OSError:
        pass

    os.symlink(target, name)


class ExperimentStore(object):
    """ Stores a collection of experiments. Each new experiment is assigned a fresh sub-path. """
    def __init__(self, path, prefix='exp', max_experiments=None, delete_old=False):
        self.path = os.path.abspath(str(path))
        assert prefix, "prefix cannot be empty"
        self.prefix = prefix
        self.max_experiments = max_experiments
        self.delete_old = delete_old
        os.makedirs(os.path.realpath(self.path), exist_ok=True)

    def new_experiment(self, name, seed, data=None, add_date=False, force_fresh=True, update_latest=True):
        """ Create a directory for a new experiment. """
        assert seed is not None and seed >= 0 and seed < np.iinfo(np.int32).max and isinstance(seed, int)

        if self.max_experiments is not None:
            experiments = os.listdir(self.path)
            n_experiments = len(experiments)

            if n_experiments >= self.max_experiments:
                if self.delete_old:
                    paths = [
                        os.path.join(self.path, p) for p in experiments
                        if p.startswith(self.prefix)]

                    sorted_by_modtime = sorted(
                        paths, key=lambda x: os.stat(x).st_mtime, reverse=True)

                    for p in sorted_by_modtime[self.max_experiments-1:]:
                        print("Deleting old experiment directory {}.".format(p))
                        try:
                            shutil.rmtree(p)
                        except NotADirectoryError:
                            os.remove(p)
                else:
                    raise Exception(
                        "Too many experiments (greater than {}) in "
                        "directory {}.".format(self.max_experiments, self.path))

        data = data or {}
        config_dict = data.copy()
        config_dict['seed'] = str(seed)

        filename = make_filename(
            self.prefix + '_' + name, add_date=add_date, config_dict=config_dict)

        if update_latest:
            make_symlink(filename, os.path.join(self.path, 'latest'))

        return ExperimentDirectory(
            os.path.join(self.path, filename), store=self, force_fresh=force_fresh)

    def __str__(self):
        return "ExperimentStore({})".format(self.path)

    def __repr__(self):
        return str(self)

    def get_latest_experiment(self, kind=None):
        path = self.path
        if kind is not None:
            path = os.path.join(self.path, kind)

        latest = os.readlink(os.path.join(path, 'latest'))
        return ExperimentDirectory(latest, store=self)

    def get_latest_results(self, filename='results'):
        exp_dir = self.get_latest_exp_dir()
        return pd.read_csv(exp_dir.path_for(filename))

    def experiment_finished(self, exp_dir, success):
        dest_name = 'complete' if success else 'incomplete'
        dest_path = os.path.join(self.path, dest_name)
        os.makedirs(dest_path, exist_ok=True)
        shutil.move(exp_dir.path, dest_path)
        exp_dir.path = os.path.join(dest_path, os.path.basename(exp_dir.path))


def _checked_makedirs(directory, force_fresh):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST or force_fresh:
            raise
    except FileExistsError:
        if force_fresh:
            raise


class ExperimentDirectory(object):
    """ Wraps a directory storing data related to an experiment. """
    def __init__(self, path, store=None, force_fresh=False):
        self.path = path
        _checked_makedirs(path, force_fresh)
        self.store = store

    def path_for(self, *path, is_dir=False):
        """ Get a path for a file, creating necessary subdirs. """
        path = os.path.join(*path)
        if is_dir:
            filename = ""
        else:
            path, filename = os.path.split(path)

        full_path = self.make_directory(path)
        return os.path.join(full_path, filename)

    def make_directory(self, path, exist_ok=True):
        full_path = os.path.join(self.path, path)
        os.makedirs(full_path, exist_ok=exist_ok)
        return full_path

    def record_environment(self, config=None, dill_recurse=False,
                           git_modules=None, git_diff=True):

        git_modules = [] if git_modules is None else git_modules
        if not isinstance(git_modules, list):
            git_modules = [git_modules]

        for module in git_modules:
            git_summary = module_git_summary(module)
            git_summary_path = self.path_for(module.__name__ + '_git_summary.txt')

            with open(git_summary_path, 'w') as f:
                f.write(git_summary.summarize(diff=git_diff))

        uname_path = self.path_for("uname.txt")
        subprocess.run("uname -a > {}".format(uname_path), shell=True)

        lscpu_path = self.path_for("lscpu.txt")
        subprocess.run("lscpu > {}".format(lscpu_path), shell=True)

        environ = {k.decode(): v.decode() for k, v in os.environ._data.items()}
        with open(self.path_for('os_environ.txt'), 'w') as f:
            f.write(pformat(environ))

        pip = pip_freeze()
        with open(self.path_for('pip_freeze.txt'), 'w') as f:
            f.write(pip)

        if config is not None:
            with open(self.path_for('config.pkl'), 'wb') as f:

                dill.dump(config, f, protocol=dill.HIGHEST_PROTOCOL,
                          recurse=dill_recurse)

            with open(self.path_for('config.txt'), 'w') as f:
                f.write(str(config))

    @property
    def host(self):
        with open(self.path_for('uname.txt'), 'r') as f:
            return f.read().split()[1]


def edit_text(prefix=None, editor="vim", initial_text=None):
    if editor != "vim":
        raise Exception("NotImplemented")

    with NamedTemporaryFile(mode='w',
                            prefix='',
                            suffix='.md',
                            delete=False) as temp_file:
        pass

    try:
        if initial_text:
            with open(temp_file.name, 'w') as f:
                f.write(initial_text)

        subprocess.call(['vim', '+', str(temp_file.name)])

        with open(temp_file.name, 'r') as f:
            text = f.read()
    finally:
        try:
            os.remove(temp_file.name)
        except FileNotFoundError:
            pass
    return text


class Tee(object):
    """ A stream that outputs to multiple streams.

    Does not close its streams; leaves responsibility for that with the caller.

    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def redirect_stream(stream_name, filename, mode='w', tee=False, **kwargs):
    assert stream_name in ['stdout', 'stderr']
    with open(str(filename), mode=mode, **kwargs) as f:
        old = getattr(sys, stream_name)

        new = f
        if tee:
            new = Tee(f, old)
        setattr(sys, stream_name, new)

        try:
            yield
        except BaseException:
            exc = traceback.format_exc()
            f.write(exc)
            raise
        finally:
            setattr(sys, stream_name, old)


def make_filename(main_title, directory='', config_dict=None, add_date=True,
                  sep='_', kvsep='=', extension='', omit=[]):
    """ Create a filename.

    Parameters
    ----------
    main_title: string
        The main title for the file.
    directory: string
        The directory to write the file to.
    config_dict: dict
        Keys and values that will be added to the filename. Key/value
        pairs are put into the filename by the alphabetical order of the keys.
    add_date: boolean
        Whether to append the current date/time to the filename.
    sep: string
        Separates items in the config dict in the returned filename.
    kvsep: string
        Separates keys from values in the returned filename.
    extension: string
        Appears at end of filename.

    """
    if config_dict is None:
        config_dict = {}
    if directory and directory[-1] != '/':
        directory += '/'

    labels = [directory + main_title]
    key_vals = list(config_dict.items())
    key_vals.sort(key=lambda x: x[0])

    for key, value in key_vals:
        if not isinstance(key, str):
            raise ValueError("keys in config_dict must be strings.")
        if not isinstance(value, str):
            raise ValueError("values in config_dict must be strings.")

        if not str(key) in omit:
            labels.append(kvsep.join([key, value]))

    if add_date:
        date_time_string = str(datetime.datetime.now()).split('.')[0]
        for c in ": -":
            date_time_string = date_time_string.replace(c, '_')
        labels.append(date_time_string)

    file_name = sep.join(labels)

    if extension:
        if extension[0] != '.':
            extension = '.' + extension

        file_name += extension

    return file_name


def parse_timedelta(d, fmt='%a %b  %d %H:%M:%S %Z %Y'):
    date = parse_date(d, fmt)
    return date - datetime.datetime.now()


def parse_date(d, fmt='%a %b  %d %H:%M:%S %Z %Y'):
    # default value for `fmt` is default format used by GNU `date`
    with open(os.devnull, 'w') as devnull:
        # A quick hack since just using the first option was causing weird things to happen, fix later.
        if " " in d:
            dstr = subprocess.check_output(["date", "-d", d], stderr=devnull)
        else:
            dstr = subprocess.check_output("date -d {}".format(d).split(), stderr=devnull)

    dstr = dstr.decode().strip()
    return datetime.datetime.strptime(dstr, fmt)


@contextmanager
def cd(path):
    """ A context manager that changes into given directory on __enter__,
        change back to original_file directory on exit. Exception safe.

    """
    path = str(path)
    old_dir = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(old_dir)


@contextmanager
def memory_limit(mb):
    """ Limit the physical memory available to the process. """
    rsrc = resource.RLIMIT_DATA
    prev_soft_limit, hard = resource.getrlimit(rsrc)
    resource.setrlimit(rsrc, (int(mb) * 1024**2, hard))
    yield
    resource.setrlimit(rsrc, (prev_soft_limit, hard))


def memory_usage(physical=False):
    """ return the memory usage in MB """
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    if physical:
        return info.rss / float(2 ** 20)
    else:
        return info.vms / float(2 ** 20)


# Character used for ascii art, sorted in order of increasing sparsity
ascii_art_chars = \
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "


def char_map(value):
    """ Maps a relative "sparsity" or "lightness" value in [0, 1) to a character. """
    if value >= 1:
        value = 1 - 1e-6
    n_bins = len(ascii_art_chars)
    bin_id = int(value * n_bins)
    return ascii_art_chars[bin_id]


def image_to_string(array):
    """ Convert an image stored as an array to an ascii art string """
    if array.ndim == 3:
        array = array.mean(-1)
    if array.ndim == 1:
        array = array.reshape(-1, int(np.sqrt(array.shape[0])))
    if not np.isclose(array.max(), 0.0):
        array = array / array.max()
    image = [char_map(value) for value in array.flatten()]
    image = np.reshape(image, array.shape)
    return '\n'.join(''.join(c for c in row) for row in image)


def shift_fill(a, n, axis=0, fill=0.0, reverse=False):
    """ shift n spaces backward along axis, filling rest in with 0's. if n is negative, shifts forward. """
    shifted = np.roll(a, n, axis=axis)
    shifted[:n, ...] = 0.0
    return shifted


def gen_seed():
    return np.random.randint(np.iinfo(np.int32).max)


class DataContainer(object):
    def __init__(self, X, Y):
        assert len(X) == len(Y)
        self.X, self.Y = X, Y

    def get_random(self):
        idx = np.random.randint(len(self.X))
        return self.X[idx], self.Y[idx]

    def get_random_with_label(self, label):
        valid = self.Y == label
        X = self.X[valid.flatten(), :]
        Y = self.Y[valid]
        idx = np.random.randint(len(X))
        return X[idx], Y[idx]

    def get_random_without_label(self, label):
        valid = self.Y != label
        X = self.X[valid.flatten(), :]
        Y = self.Y[valid]
        idx = np.random.randint(len(X))
        return X[idx], Y[idx]


def digits_to_numbers(digits, base=10, axis=-1, keepdims=False):
    """ Convert array of digits to number, assumes little-endian (least-significant first). """
    mult = base ** np.arange(digits.shape[axis])
    shape = [1] * digits.ndim
    shape[axis] = mult.shape[axis]
    mult = mult.reshape(shape)
    return (digits * mult).sum(axis=axis, keepdims=keepdims)


def numbers_to_digits(numbers, n_digits, base=10):
    """ Convert number to array of digits, assumed little-endian. """
    numbers = numbers.copy()
    digits = []
    for i in range(n_digits):
        digits.append(numbers % base)
        numbers //= base
    return np.stack(digits, -1)


NotSupplied = object()


class Param(object):
    def __init__(self, default=NotSupplied, aliases=None, help="", type=None):
        """ aliases are different ways to fill the value (i.e. from config or kwargs),
            but should not be used to access the value as a class attribute. """
        self.default = default
        if isinstance(aliases, str):
            aliases = aliases.split()
        self.aliases = aliases or []
        self.help = help
        self.type = type


class Parameterized(object):
    _resolved = False

    def __new__(cls, *args, **kwargs):
        obj = super(Parameterized, cls).__new__(cls)
        obj._resolve_params(**kwargs)
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def _resolve_params(self, **kwargs):
        if not self._resolved:
            param_names = self.param_names()
            for name in param_names:
                param = getattr(self.__class__, name)

                aliases = list([name] + param.aliases)

                value = NotSupplied

                # Check kwargs
                for alias in aliases:
                    if value is not NotSupplied:
                        break
                    value = kwargs.get(alias, NotSupplied)

                # Check cfg
                for alias in aliases:
                    if value is not NotSupplied:
                        break
                    value = getattr(dps.cfg, alias, NotSupplied)

                # Try the default value
                if value is NotSupplied:
                    if param.default is not NotSupplied:
                        value = param.default
                    else:
                        raise AttributeError(
                            "Could not find value for parameter {} for class {} "
                            "in either kwargs or config, and no default was provided.".format(
                                name, self.__class__))

                if param.type is not None:
                    value = param.type(value)

                setattr(self, name, value)
            self._resolved = True

    @classmethod
    def param_names(cls):
        params = []
        for p in dir(cls):
            try:
                if p != 'params' and isinstance(getattr(cls, p), Param):
                    params.append(p)
            except Exception:
                pass
        return params

    def param_values(self):
        if not self._resolved:
            raise Exception("Cannot supply `param_values` as parameters have not yet been resolved.")
        return {n: getattr(self, n) for n in self.param_names()}


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du', '-sh', str(path)]).split()[0].decode('utf-8')


class pdb_postmortem(object):
    def __enter__(self):
        pass

    def __exit__(self, type_, value, tb):
        if type_:
            traceback.print_exc()
            pdb.post_mortem(tb)
        return True


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def process_path(path, real_path=False):
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if real_path:
        path = os.path.realpath(path)
    return path


def path_stem(path):
    no_ext = os.path.splitext(path)[0]
    return os.path.basename(no_ext)


@contextmanager
def catch(exception_types, action=None):
    """ A try-except block as a context manager. """
    try:
        yield
    except exception_types as e:
        if isinstance(action, str):
            print(action)
        elif action:
            action(e)


class Alarm(BaseException):
    pass


def raise_alarm(*args, **kwargs):
    raise Alarm("Raised by `raise_alarm`.")


class time_limit(object):
    """ Example use:

        with time_limit(seconds=5) as tl:
            while True:
                pass

        if tl.ran_out:
            print("Time ran out.")

    """
    def __init__(self, seconds, verbose=False):
        self.seconds = seconds
        self.verbose = verbose
        self.ran_out = False

    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGALRM, raise_alarm)
        if self.seconds <= 0:
            raise_alarm("Didn't get started.")
        if not np.isinf(self.seconds):
            signal.alarm(int(np.floor(self.seconds)))
        self.then = time.time()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.elapsed_time = time.time() - self.then

        if exc_type is Alarm:
            self.ran_out = True
            if self.verbose:
                print("Block ran for {} seconds (limit was {}).".format(self.elapsed_time, self.seconds))
            return True
        else:
            signal.alarm(0)  # Cancel the alarm.
        return False


def timed_func(func):
    @wraps(func)
    def f(*args, **kwargs):
        with timed_block(func.__name__):
            return func(*args, **kwargs)
    return f


@contextmanager
def timed_block(name=None):
    if name is None:
        frame = inspect.stack()[1]
        name = "{}:{}".format(frame.filename, frame.lineno)
    start_time = time.time()
    yield
    print("Call to block <{}> took {} seconds.".format(name, time.time() - start_time))


# From py.test
class KeywordMapping(object):
    """ Provides a local mapping for keywords.

        Can be used to implement user-friendly name selection
        using boolean expressions.

        names=[orange], pattern = "ora and e" -> True
        names=[orange], pattern = "orang" -> True
        names=[orange], pattern = "orane" -> False
        names=[orange], pattern = "ora and z" -> False
        names=[orange], pattern = "ora or z" -> True

        Given a list of names, map any string that is a substring
        of one of those names to True.

        ``names`` are the things we are trying to select, ``pattern``
        is the thing we are using to select them. Note that supplying
        multiple names does not mean "apply the pattern to each one
        separately". Rather, we are selecting the list as a whole,
        which doesn't seem that useful. The different names should be
        thought of as different names for a single object.

    """
    def __init__(self, names):
        self._names = names

    def __getitem__(self, subname):
        if subname is "_":
            return True

        for name in self._names:
            if subname in name:
                return True
        return False

    def eval(self, pattern):
        return eval(pattern, {}, self)

    @staticmethod
    def batch(batch, pattern):
        """ Apply a single pattern to a batch of names. """
        return [KeywordMapping([b]).eval(pattern) for b in batch]


class SigTerm(Exception):
    pass


class NumpySeed(object):
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.state)


class _bool(object):
    def __new__(cls, val):
        if val in ("0", "False", "F", "false", "f"):
            return False
        return bool(val)


def popleft(l, default=None):
    if default is not None:
        try:
            return l.popleft()
        except IndexError:
            return default
    else:
        return l.popleft()


def nested_update(d, other):
    if not isinstance(d, dict) or not isinstance(other, dict):
        return

    for k, v in other.items():
        if k in d and isinstance(d[k], dict) and isinstance(v, dict):
            nested_update(d[k], v)
        else:
            d[k] = v


class Config(dict, MutableMapping):
    """ Note: multi-level setting will succeed more often with __setitem__ than __setattr__.

    This doesn't work:

    c = Config()
    c.a.b = 1

    But this does:

    c = Config()
    c["a:b"] = 1

    """
    _reserved_keys = None

    def __init__(self, _d=None, **kwargs):
        if _d:
            self.update(_d)
        self.update(kwargs)

    def flatten(self):
        return {k: self[k] for k in self._keys()}

    def _keys(self, sep=":"):
        stack = [iter(dict.items(self))]
        key_prefix = ''

        while stack:
            new = next(stack[-1], None)
            if new is None:
                stack.pop()
                key_prefix = key_prefix.rpartition(sep)[0]
                continue

            key, value = new
            nested_key = key_prefix + sep + key

            if isinstance(value, dict) and value:
                stack.append(iter(value.items()))
                key_prefix = nested_key
            else:
                yield nested_key[1:]

    def __iter__(self):
        return self._keys()

    def keys(self):
        return MutableMapping.keys(self)

    def values(self):
        return MutableMapping.values(self)

    def items(self):
        return MutableMapping.items(self)

    def __str__(self):
        items = {k: v for k, v in dict.items(self)}
        s = "{}(\n{}\n)".format(self.__class__.__name__, pformat(items))
        return s

    def __repr__(self):
        return str(self)

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, key):
        assert isinstance(key, str), "`Config` keys must be strings."
        if ':' in key:
            keys = key.split(':')
            value = self
            for k in keys:
                try:
                    value = value[k]
                except Exception:
                    try:
                        value = value[int(k)]
                    except Exception:
                        raise KeyError(
                            "Calling __getitem__ with key {} failed at component {}.".format(key, k))
            return value
        else:
            return super(Config, self).__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(key, str), "`Config` keys must be strings."
        if ':' in key:
            keys = key.split(':')
            to_set = self
            for k in keys[:-1]:
                try:
                    to_set = to_set[k]
                except Exception:
                    try:
                        to_set = to_set[int(k)]
                    except Exception:
                        to_set[k] = self.__class__()
                        to_set = to_set[k]
            try:
                to_set[keys[-1]] = value
            except Exception:
                to_set[int(keys[-1])] = value
        else:
            self._validate_key(key)
            return super(Config, self).__setitem__(key, value)

    def __delitem__(self, key):
        assert isinstance(key, str), "`Config` keys must be strings."
        if ':' in key:
            keys = key.split(':')
            to_del = self
            for k in keys[:-1]:
                try:
                    to_del = to_del[k]
                except Exception:
                    try:
                        to_del = to_del[int(k)]
                    except Exception:
                        raise KeyError("Calling __getitem__ with key {} failed at component {}.".format(key, k))
            try:
                del to_del[keys[-1]]
            except Exception:
                try:
                    to_del = to_del[int(k)]
                except Exception:
                    raise KeyError("Calling __getitem__ with key {} failed at component {}.".format(key, k))
        else:
            return super(Config, self).__delitem__(key)

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("Could not find attribute called `{}`.".format(key))

    def __setattr__(self, key, value):
        if key == '_reserved_keys':
            super(Config, self).__setattr__(key, value)
        else:
            self[key] = value

    def __enter__(self):
        ConfigStack._stack.append(self)

    def __exit__(self, exc_type, exc, exc_tb):
        popped = ConfigStack._stack.pop()
        assert popped == self, "Something is wrong with the config stack."
        return False

    def _validate_key(self, key):
        msg = "Bad key for config: `{}`.".format(key)
        assert isinstance(key, str), msg
        assert key.isidentifier(), msg
        assert not key.startswith('_'), msg
        assert key not in self._reserved_keys, msg

    def copy(self, _d=None, **kwargs):
        """ Copy and update at the same time. """
        new = copy.deepcopy(self)
        if _d:
            new.update(_d)
        new.update(**kwargs)
        return new

    def update(self, _d=None, **kwargs):
        nested_update(self, _d)
        nested_update(self, kwargs)

    def update_from_command_line(self):
        cl_args = clify.wrap_object(self).parse()
        self.update(cl_args)


class SystemConfig(Config):
    def __init__(self, _d=None, **kwargs):
        config = _load_system_config()
        if _d:
            config.update(_d)
        config.update(kwargs)
        super(SystemConfig, self).__init__(**config)


def _load_system_config(key=None):
    _config = configparser.ConfigParser()
    location = os.path.dirname(dps.__file__)
    _config.read(os.path.join(location, 'config.ini'))

    if not key:
        key = socket.gethostname()

    if 'travis' in key:
        key = 'travis'

    if key not in _config:
        key = 'DEFAULT'

    # Load default configuration from a file
    config = Config(
        hostname=socket.gethostname(),
        start_tensorboard=_config.getboolean(key, 'start_tensorboard'),
        reload_interval=_config.getint(key, 'reload_interval'),
        update_latest=_config.getboolean(key, 'update_latest'),
        save_summaries=_config.getboolean(key, 'save_summaries'),
        data_dir=process_path(_config.get(key, 'data_dir')),
        model_dir=process_path(_config.get(key, 'model_dir')),
        build_experiments_dir=process_path(_config.get(key, 'build_experiments_dir')),
        run_experiments_dir=process_path(_config.get(key, 'run_experiments_dir')),
        log_root=process_path(_config.get(key, 'log_root')),
        show_plots=_config.getboolean(key, 'show_plots'),
        save_plots=_config.getboolean(key, 'save_plots'),
        use_gpu=_config.getboolean(key, 'use_gpu'),
        tbport=_config.getint(key, 'tbport'),
        verbose=_config.getboolean(key, 'verbose'),
        per_process_gpu_memory_fraction=_config.getfloat(key, 'per_process_gpu_memory_fraction'),
        gpu_allow_growth=_config.getboolean(key, 'gpu_allow_growth'),
        parallel_exe=process_path(_config.get(key, 'parallel_exe')),
    )

    config.max_experiments = _config.getint(key, 'max_experiments')
    if config.max_experiments <= 0:
        config.max_experiments = np.inf
    return config


class ClearConfig(SystemConfig):
    pass


Config._reserved_keys = dir(Config)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigStack(dict, metaclass=Singleton):
    _stack = []

    @property
    def config_sequence(self):
        """ Get all configs up the the first occurence of an instance of ClearConfig """
        stack = ConfigStack._stack[::-1]
        for i, config in enumerate(stack):
            if isinstance(config, ClearConfig):
                return stack[:i+1]
        return stack

    def clear_stack(self, default=NotSupplied):
        self._stack.clear()
        if default is not None:
            if default is NotSupplied:
                self._stack.append(SystemConfig())
            else:
                self._stack.append(default)

    def __str__(self):
        return self.to_string(hidden=True)

    def __repr__(self):
        return str(self)

    def to_string(self, hidden=False):
        s = []

        seen_keys = set()
        reverse_stack = self._stack[::-1]
        visible_keys = [set() for config in reverse_stack]

        cleared = False
        for vk, config in zip(visible_keys, reverse_stack):
            if not cleared:
                for key in config.keys():
                    if key not in seen_keys:
                        vk.add(key)
                        seen_keys.add(key)

            if isinstance(config, ClearConfig):
                cleared = True

        for i, (vk, config) in enumerate(zip(visible_keys[::-1], reverse_stack[::-1])):
            visible_items = {k: v for k, v in config.items() if k in vk}

            if hidden:
                hidden_items = {k: v for k, v in config.items() if k not in vk}
                _s = "# {}: <{} -\nVISIBLE:\n{}\nHIDDEN:\n{}\n>".format(
                    i, config.__class__.__name__,
                    pformat(visible_items), pformat(hidden_items))
            else:
                _s = "# {}: <{} -\n{}\n>".format(i, config.__class__.__name__, pformat(visible_items))

            s.append(_s)

        s = '\n'.join(s)
        return "<{} -\n{}\n>".format(self.__class__.__name__, s)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        return iter(self._keys())

    def _keys(self):
        keys = set()
        for config in self.config_sequence:
            keys |= config.keys()
        return list(keys)

    def keys(self):
        return MutableMapping.keys(self)

    def values(self):
        return MutableMapping.values(self)

    def items(self):
        return MutableMapping.items(self)

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, key):
        for config in self.config_sequence:
            if key in config:
                return config[key]
        raise KeyError("Cannot find a value for key `{}`".format(key))

    def __setitem__(self, key, value):
        self._stack[-1][key] = value

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("No attribute named `{}`.".format(key))

    def __setattr__(self, key, value):
        setattr(self._stack[-1], key, value)

    def update(self, *args, **kwargs):
        self._stack[-1].update(*args, **kwargs)

    def freeze(self, remove_callable=False):
        _config = Config()
        for key in self.keys():
            value = self[key]
            if remove_callable and callable(value):
                value = str(value)
            _config[key] = value
        return _config

    @property
    def log_dir(self):
        return os.path.join(self.log_root, self.log_name)

    def update_from_command_line(self):
        cl_args = clify.wrap_object(self).parse()
        self.update(cl_args)
