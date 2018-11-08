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
import importlib
import json

import clify
import dps


def create_maze(shape):
    # Random Maze Generator using Depth-first Search
    # http://en.wikipedia.org/wiki/Maze_generation_algorithm
    # FB - 20121214
    my, mx = shape
    maze = np.zeros(shape)
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # start the maze from a random cell
    stack = [(np.random.randint(0, mx), np.random.randint(0, my))]

    while len(stack) > 0:
        (cy, cx) = stack[-1]
        maze[cy, cx] = 1

        # find a new cell to add
        nlst = []  # list of available neighbors
        for i, (dy, dx) in enumerate(dirs):
            ny = cy + dy
            nx = cx + dx

            if ny >= 0 and ny < my and nx >= 0 and nx < mx:
                if maze[ny, nx] == 0:
                    # of occupied neighbors must be 1
                    ctr = 0
                    for _dy, _dx in dirs:
                        ex = nx + _dx
                        ey = ny + _dy

                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ey, ex] == 1:
                                ctr += 1

                    if ctr == 1:
                        nlst.append(i)

        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = np.random.choice(nlst)
            dy, dx = dirs[ir]
            cy += dy
            cx += dx
            stack.append((cy, cx))
        else:
            stack.pop()

    return maze


def header(message, n, char, nl=True):
    assert isinstance(char, str)
    banner = char * n
    newline = "\n" if nl else ""
    return "{}{} {} {}{}".format(newline, banner, message.strip(), banner, newline)


def print_header(message, n, char, nl=True):
    print(header(message, n, char, nl))


def generate_perlin_noise_2d(shape, res, normalize=False):
    """ each dim of shape must be divisible by corresponding dim of res

    from https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html

    """
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1]-1)) * g11, 2)

    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:, :, 0]) + t[:, :, 0]*n10
    n1 = n01*(1-t[:, :, 0]) + t[:, :, 0]*n11

    result = np.sqrt(2)*((1-t[:, :, 1])*n0 + t[:, :, 1]*n1)

    if normalize:
        result -= result.min()
        mx = result.max()
        if mx >= 1e-6:
            result /= mx

    return result


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def square_subplots(N, **kwargs):
    sqrt_N = int(np.ceil(np.sqrt(N)))
    m = int(np.ceil(N / sqrt_N))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(m, sqrt_N, **kwargs)
    return fig, axes


def nvidia_smi(robust=True):
    try:
        p = subprocess.run("nvidia-smi".split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.stdout.decode()
    except Exception as e:
        if robust:
            return "Exception while calling nvidia-smi: {}".format(e)
        else:
            raise


_nvidia_smi_processes_header = "|  GPU       PID   Type   Process name                             Usage      |"
_nvidia_smi_table_end = "+-----------------------------------------------------------------------------+"


def _nvidia_smi_parse_processes(s):
    lines = s.split('\n')
    header_idx = None
    table_end_idx = None
    for i, line in enumerate(lines):
        if line == _nvidia_smi_processes_header:
            header_idx = i
        elif header_idx is not None and line == _nvidia_smi_table_end:
            table_end_idx = i

    assert header_idx is not None, "Malformed nvidia-smi string:\n{}".format(s)
    assert table_end_idx is not None, "Malformed nvidia-smi string:\n{}".format(s)

    processes = []

    for line in lines[header_idx+2:table_end_idx]:
        tokens = line.split()
        gpu_idx = int(tokens[1])
        pid = int(tokens[2])
        type = tokens[3]
        process_name = tokens[4]
        memory_usage = tokens[5]
        memory_usage_mb = int(memory_usage[:-3])

        processes.append((gpu_idx, pid, type, process_name, memory_usage_mb))

    return processes


def gpu_memory_usage():
    """ return gpu memory usage for current process in MB """
    try:
        s = nvidia_smi(robust=False)
    except Exception:
        return 0

    gpu_processes = _nvidia_smi_parse_processes(s)

    my_pid = os.getpid()

    my_memory_usage_mb = 0

    for gpu_idx, pid, type, process_name, memory_usage_mb in gpu_processes:
        if pid == my_pid:
            my_memory_usage_mb += memory_usage_mb

    return my_memory_usage_mb


def view_readme_cl():
    return view_readme(".", 2)


def view_readme(path, max_depth):
    """ View readme files in a diretory of experiments, sorted by the time at
        which the experiment began execution.

    """
    import iso8601

    command = "find {} -maxdepth {} -name README.md".format(path, max_depth).split()
    p = subprocess.run(command, stdout=subprocess.PIPE)
    readme_paths = [r for r in p.stdout.decode().split('\n') if r]
    dates_paths = []
    for r in readme_paths:
        d = os.path.split(r)[0]

        try:
            with open(os.path.join(d, 'stdout'), 'r') as f:
                line = ''
                try:
                    while not line.startswith("Starting training run"):
                        line = next(f)
                except StopIteration:
                    line = None

            if line is not None:
                tokens = line.split()
                assert len(tokens) == 13
                dt = iso8601.parse_date(tokens[5] + " " + tokens[6][:-1])
                dates_paths.append((dt, r))
            else:
                raise Exception()
        except Exception:
            print("Omitting {} which has no valid `stdout` file.".format(r))

    _sorted = sorted(dates_paths)

    for d, r in _sorted:
        print("\n" + "-" * 80 + "\n\n" + "====> {} <====".format(r))
        print("Experiment started on {}\n".format(d))
        with open(r, 'r') as f:
            print(f.read())


def confidence_interval(data, coverage):
    from scipy import stats
    return stats.t.interval(
        coverage, len(data)-1, loc=np.mean(data), scale=stats.sem(data))


def standard_error(data):
    from scipy import stats
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
    for name in sorted(name_params):
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


def sha_cache(directory, recurse=False, verbose=False):
    os.makedirs(directory, exist_ok=True)

    def _print(s, verbose=verbose):
        if verbose:
            print("sha_cache: {}" .format(s))

    def decorator(func):
        sig = inspect.signature(func)

        def new_f(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            param_hash = get_param_hash(bound_args.arguments)
            filename = os.path.join(directory, "{}_{}.cache".format(func.__name__, param_hash))

            loaded = False
            try:
                if not CLEAR_CACHE:
                    _print("Attempting to load...")
                    with open(filename, 'rb') as f:
                        value = dill.load(f)
                    loaded = True
                    _print("Loaded successfully.")
            except FileNotFoundError:
                _print("File not found.")
                pass
            finally:
                if not loaded:
                    _print("Calling function...")
                    value = func(**bound_args.arguments)

                    _print("Saving results...")
                    with open(filename, 'wb') as f:
                        dill.dump(value, f, protocol=dill.HIGHEST_PROTOCOL, recurse=recurse)
            return value
        return new_f
    return decorator


def _run_cmd(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()


class GitSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def summarize(self, n_logs=10, diff=False):
        s = []
        with cd(self.directory):
            s.append("*" * 40)
            s.append("git summary for directory {}\n".format(self.directory))

            s.append("log:\n")
            log = _run_cmd('git log -n {}'.format(n_logs))
            s.append(log)

            s.append("\nstatus:\n")
            status = _run_cmd('git status --porcelain')
            s.append(status)

            s.append("\ndiff:\n")
            if diff:
                diff = _run_cmd('git diff HEAD')
                s.append(diff)
            else:
                s.append("<ommitted>")

            s.append("\nEnd of git summary for directory {}".format(self.directory))
            s.append("*" * 40 + "\n")
        return '\n'.join(s)

    def freeze(self):
        pass


def find_git_packages():
    all_packages = pip_freeze()
    all_packages = all_packages.split('\n')

    git_packages = [p.split('=')[-1] for p in all_packages if p.startswith('-e git+')]

    vc_packages = []

    for p in git_packages:
        package = importlib.import_module(p)
        directory = os.path.dirname(os.path.dirname(package.__file__))
        git_dir = os.path.join(directory, '.git')

        if os.path.isdir(git_dir):
            vc_packages.append(package)
    return vc_packages


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

    def record_environment(self, config=None, dill_recurse=False, git_diff=True):
        with open(self.path_for('context/git_summary.txt'), 'w') as f:
            git_packages = find_git_packages()
            for module in git_packages:
                git_summary = module_git_summary(module)
                f.write(git_summary.summarize(diff=git_diff))

        uname_path = self.path_for("context/uname.txt")
        subprocess.run("uname -a > {}".format(uname_path), shell=True)

        lscpu_path = self.path_for("context/lscpu.txt")
        subprocess.run("lscpu > {}".format(lscpu_path), shell=True)

        environ = {k.decode(): v.decode() for k, v in os.environ._data.items()}
        with open(self.path_for('context/os_environ.txt'), 'w') as f:
            f.write(pformat(environ))

        pip = pip_freeze()
        with open(self.path_for('context/pip_freeze.txt'), 'w') as f:
            f.write(pip)

        if config is not None:
            with open(self.path_for('config.pkl'), 'wb') as f:
                dill.dump(config, f, protocol=dill.HIGHEST_PROTOCOL, recurse=dill_recurse)

            with open(self.path_for('config.json'), 'w') as f:
                json.dump(config.freeze(), f, default=str, indent=4, sort_keys=True)

    @property
    def host(self):
        try:
            with open(self.path_for('context/uname.txt'), 'r') as f:
                return f.read().split()[1]
        except FileNotFoundError:
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
    """ return memory usage for current process in MB """
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
    shifted[:n, ...] = fill
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

        # Stored for copying purposes, to get parameter as they are before __init__ is called.
        obj._resolved_params = obj.param_values()

        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __str__(self):
        return "{}(\n{}\n)".format(self.__class__.__name__, pformat(self.param_values()))

    @classmethod
    def _get_param_value(cls, name, param, kwargs):
        aliases = list([name] + param.aliases)

        # Check kwargs
        for alias in aliases:
            value = kwargs.get(alias, NotSupplied)
            if value is not NotSupplied:
                return value

        # Check cfg with class name label
        for _cls in cls.__mro__:
            for alias in aliases:
                key = _cls.__name__ + ":" + alias
                value = getattr(dps.cfg, key, NotSupplied)
                if value is not NotSupplied:
                    return value

        # Check cfg
        for alias in aliases:
            value = getattr(dps.cfg, alias, NotSupplied)
            if value is not NotSupplied:
                return value

        # Try the default value
        if value is NotSupplied:
            if param.default is not NotSupplied:
                return param.default
            else:
                raise AttributeError(
                    "Could not find value for parameter `{}` for class `{}` "
                    "in either kwargs or config, and no default was provided.".format(
                        name, cls.__name__))

    def _resolve_params(self, **kwargs):
        if not self._resolved:
            for k, v in self._capture_param_values(**kwargs).items():
                setattr(self, k, v)
            self._resolved = True

    @classmethod
    def _capture_param_values(cls, **kwargs):
        """ Return the params that would be created if an object of the current class were constructed in the current context with the given kwargs. """
        param_values = dict()
        for name in cls.param_names():
            param = getattr(cls, name)
            value = cls._get_param_value(name, param, kwargs)
            if param.type is not None:
                value = param.type(value)
            param_values[name] = value
        return param_values

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

    def __deepcopy__(self, memo):
        cls = self.__class__
        kwargs = self._resolved_params
        result = cls.__new__(cls, **kwargs)
        result.__init__(**kwargs)
        memo[id(self)] = result
        return result


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
    _stack = []

    def __init__(self, seconds, verbose=False, timeout_callback=None):
        self.seconds = seconds
        self.verbose = verbose
        self.ran_out = False
        self.timeout_callback = timeout_callback

    def __str__(self):
        return (
            "time_limit(seconds={}, verbose={}, ran_out={}, "
            "timeout_callback={})".format(
                self.seconds, self.verbose, self.ran_out, self.timeout_callback))

    def __enter__(self):
        if time_limit._stack:
            raise Exception(
                "Only one instance of `time_limit` may be active at once. "
                "Another time_limit instance {} was already active.".format(
                    time_limit._stack[0]))

        self.old_handler = signal.signal(signal.SIGALRM, raise_alarm)

        if self.seconds <= 0:
            raise_alarm("Didn't get started.")

        if not np.isinf(self.seconds):
            signal.alarm(int(np.floor(self.seconds)))

        self.then = time.time()
        time_limit._stack.append(self)
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.elapsed_time = time.time() - self.then

        signal.signal(signal.SIGALRM, self.old_handler)
        time_limit._stack.pop()

        if exc_type is Alarm:
            self.ran_out = True

            if self.verbose:
                print("Block ran for {} seconds (limit was {}).".format(
                    self.elapsed_time, self.seconds))

            if self.timeout_callback:
                self.timeout_callback(self)

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
                nxt = None
                try:
                    nxt = to_set[k]
                except KeyError:
                    try:
                        nxt = to_set[int(k)]
                    except Exception:
                        pass

                if not isinstance(nxt, (list, dict)):
                    nxt = self.__class__()
                    to_set[k] = nxt

                to_set = nxt

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

    def freeze(self, remove_callable=False):
        _config = Config()
        for key in self.keys():
            value = self[key]
            if remove_callable and callable(value):
                value = str(value)
            _config[key] = value
        return _config


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
        return os.path.join(self.log_root, self.env_name)

    def update_from_command_line(self):
        cl_args = clify.wrap_object(self).parse()
        self.update(cl_args)
