from pprint import pformat
from contextlib import contextmanager
import numpy as np
from pathlib import Path
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
from itertools import cycle, islice
import resource
import sys
import shutil
import pandas as pd
import errno

import clify
import dps


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
        try:
            os.makedirs(os.path.realpath(self.path))
        except:
            pass

    def new_experiment(self, name, data=None, add_date=False, force_fresh=True, update_latest=True):
        """ Create a new experiment path. """
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

        filename = make_filename(self.prefix + '_' + name, add_date=add_date, config_dict=data)
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
        try:
            os.makedirs(dest_path)
        except:
            pass
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

    def path_for(self, path, is_dir=False):
        """ Get a path for a file, creating necessary subdirs. """
        if is_dir:
            filename = ""
        else:
            path, filename = os.path.split(path)

        full_path = self.make_directory(path)
        return os.path.join(full_path, filename)

    def make_directory(self, path):
        full_path = os.path.join(self.path, path)
        try:
            os.makedirs(full_path)
        except:
            pass
        return full_path


@contextmanager
def redirect_stream(stream, filename, mode='w', **kwargs):
    assert stream in ['stdout', 'stderr']
    with open(str(filename), mode=mode, **kwargs) as f:
        old = getattr(sys, stream)
        setattr(sys, stream, f)

        try:
            yield
        finally:
            setattr(sys, stream, old)


def make_filename(main_title, directory='', config_dict=None, add_date=True,
                  sep='_', kvsep=':', extension='', omit=[]):
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

        if not str(key) in omit and not hasattr(value, '__len__'):
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


def memory_usage(physical=False):
    """ return the memory usage in MB """
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    if physical:
        return info.rss / float(2 ** 20)
    else:
        return info.vms / float(2 ** 20)


class DataContainer(object):
    def __init__(self, X, Y):
        assert len(X) == len(Y)
        self.X, self.Y = X, Y

    def get_random(self):
        idx = np.random.randint(len(self.X))
        return self.X[idx], self.Y[idx]


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
    def __init__(self, default=NotSupplied):
        self.default = default


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
            for p in self.param_names():
                value = kwargs.get(p)
                if value is None:
                    try:
                        value = getattr(dps.cfg, p)
                    except AttributeError as e:
                        param = getattr(self, p)
                        if param.default is not NotSupplied:
                            value = param.default
                        else:
                            raise e

                setattr(self, p, value)
            self._resolved = True

    @classmethod
    def param_names(cls):
        params = []
        for p in dir(cls):
            try:
                if p != 'params' and isinstance(getattr(cls, p), Param):
                    params.append(p)
            except:
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


def process_path(path):
    return os.path.expandvars(os.path.expanduser(str(path)))


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


class Alarm(Exception):
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


class Schedule(object):
    pass


class MixtureSchedule(Schedule):
    def __init__(self, components, reset_n_steps, shared_clock=False, p=None, name=None):
        self.components = components
        self.n_components = len(components)
        self.reset_n_steps = reset_n_steps
        self.shared_clock = shared_clock
        self.p = p

    def build(self, t):
        t = t.copy()
        n_periods = int(np.ceil(len(t) / self.reset_n_steps))
        offsets = [0] * self.n_components
        signal = []
        for i in range(n_periods):
            if len(signal) >= len(t):
                break
            selected = np.random.choice(range(self.n_components), p=self.p)
            if self.shared_clock:
                start = offsets[0]
            else:
                start = offsets[selected]

            t_ = t[start:start+self.reset_n_steps]
            _signal = self.components[selected].build(t_)
            signal.extend(_signal)

            if self.shared_clock:
                offsets[0] += self.reset_n_steps
            else:
                offsets[selected] += self.reset_n_steps
        signal = np.array(signal).reshape(-1)[:len(t)]
        return signal


class ChainSchedule(Schedule):
    def __init__(self, components, component_n_steps, shared_clock=False):
        self.components = components
        self.n_components = len(components)
        self.component_n_steps = component_n_steps
        self.shared_clock = shared_clock

    def build(self, t):
        t = t.copy()
        try:
            int(self.component_n_steps)
            n_steps = [self.component_n_steps] * self.n_components
        except:
            n_steps = self.component_n_steps

        signal = []
        offsets = [0] * self.n_components
        for i in cycle(range(self.n_components)):
            if len(signal) >= len(t):
                break
            if self.shared_clock:
                start = offsets[0]
            else:
                start = offsets[i]

            t_ = t[start: start+n_steps[i]]
            _signal = self.components[i].build(t_).astype('f')
            signal.extend(list(_signal))

            if self.shared_clock:
                offsets[0] += n_steps[i]
            else:
                offsets[i] += n_steps[i]

        return np.array(signal).reshape(-1)[:len(t)]


class RepeatSchedule(Schedule):
    def __init__(self, schedule, period):
        self.schedule = schedule
        self.period = period

    def build(self, t):
        t = t.copy()
        t_ = t[:self.period]
        signal = cycle(self.schedule.build(t_))
        signal = islice(signal, len(t))
        signal = np.array(list(signal)).reshape(-1)
        return signal


class Exponential(Schedule):
    def __init__(self, start, end, decay_steps, decay_rate, staircase=False):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

        assert isinstance(self.decay_steps, int)
        assert self.decay_steps > 1
        assert 0 < self.decay_rate < 1

    def build(self, t):
        t = t.copy()
        if self.staircase:
            t //= self.decay_steps
        else:
            t /= self.decay_steps
        return (self.start - self.end) * (self.decay_rate ** t) + self.end


class Exp(Exponential):
    pass


class Polynomial(Schedule):
    def __init__(self, start, end, decay_steps, power=1.0):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.power = power

        assert isinstance(self.decay_steps, int)
        assert self.decay_steps > 1
        assert power > 0

    def build(self, t):
        t = t.copy()
        t = np.minimum(self.decay_steps, t)
        return (self.start - self.end) * ((1 - t / self.decay_steps) ** self.power) + self.end


class Poly(Polynomial):
    pass


class Reciprocal(Schedule):
    def __init__(self, start, end, decay_steps, gamma=1.0, staircase=False):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.gamma = gamma
        self.staircase = staircase

        assert isinstance(self.decay_steps, int)
        assert self.decay_steps > 1
        assert self.gamma > 0

    def build(self, t):
        t = t.copy()
        if self.staircase:
            t //= self.decay_steps
        else:
            t /= self.decay_steps
        return ((self.start - self.end) / (1 + t))**self.gamma + self.end


class Constant(Schedule):
    def __init__(self, value):
        self.value = value

    def build(self, t):
        t = t.copy()
        return np.array([self.value] * len(t))


def eval_schedule(schedule):
    if isinstance(schedule, str):
        schedule = eval(schedule)
    return schedule


def nested_update(d, other):
    if not isinstance(d, dict) or not isinstance(other, dict):
        return

    for k, v in other.items():
        if k in d and isinstance(d[k], dict) and isinstance(v, dict):
            nested_update(d[k], v)
        else:
            d[k] = v


class Config(dict, MutableMapping):
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

            if isinstance(value, dict):
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
                except KeyError:
                    raise KeyError("Calling __getitem__ with key {} failed at component {}.".format(key, k))
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
                except KeyError:
                    to_set[k] = self.__class__()
                    to_set = to_set[k]
            to_set[keys[-1]] = value
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
                except KeyError:
                    raise KeyError("Calling __getitem__ with key {} failed at component {}.".format(key, k))
            try:
                del to_del[keys[-1]]
            except KeyError:
                raise KeyError("Calling __getitem__ with key {} failed at component {}.".format(key, keys[-1]))
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
        new = copy.copy(self)
        if _d:
            new.update(_d)
        new.update(**kwargs)
        return new

    def update(self, _d=None, **kwargs):
        nested_update(self, _d)
        nested_update(self, kwargs)


class ClearConfig(Config):
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
                return stack[:i]
        return stack

    def clear_stack(self, default=None):
        self._stack.clear()
        if default is not None:
            self._stack.append(default)

    def __str__(self):
        items = {k: v for k, v in self.items()}
        s = "<{} -\n{}\n>".format(self.__class__.__name__, pformat(items))
        return s

    def __repr__(self):
        return str(self)

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
        cfg = Config()
        for key in self.keys():
            value = self[key]
            if remove_callable and callable(value):
                value = str(value)
            cfg[key] = value
        return cfg

    @property
    def log_dir(self):
        return str(Path(self.log_root) / self.log_name)

    def update_from_command_line(self):
        cl_args = clify.wrap_object(self).parse()
        self.update(cl_args)
