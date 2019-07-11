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
import errno
import tempfile
import dill
from functools import wraps
import inspect
import hashlib
from zipfile import ZipFile
import importlib
import json
import gc
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio
from skimage.transform import resize

import clify
import dps


def copy_update(d, **kwargs):
    new = d.copy()
    new.update(kwargs)
    return new


def resize_image(img, shape, mode=None, preserve_range=True, anti_aliasing=None):
    if anti_aliasing is None:
        anti_aliasing = any(ns < s for ns, s in zip(shape, img.shape))

    if mode is None:
        mode = 'reflect' if anti_aliasing else 'edge'

    return resize(
        img, shape, mode=mode, preserve_range=preserve_range,
        anti_aliasing=anti_aliasing)


def video_stack(video):
    """ Take an ndarray with shape (*batch_shape, n_frames, H, W, D) representing a video
        and stack the frames together as different channel dimensions, resulting
        in an ndarray with shape (*batch_shape, H, W, D*n_frames) """
    video = np.array(video)
    *batch_shape, n_frames, H, W, D = video.shape
    perm = tuple(range(len(batch_shape))) + tuple(np.array([1, 2, 0, 3]) + len(batch_shape))
    return np.transpose(video, perm).reshape(*batch_shape, H, W, n_frames*D)


def video_unstack(stacked_video, n_frames):
    """ Inverse of the function `video_stack`. """
    stacked_video = np.array(stacked_video)
    *batch_shape, H, W, _D = stacked_video.shape
    D = int(_D / n_frames)
    assert D * n_frames == _D

    video = stacked_video.reshape(*batch_shape, H, W, n_frames, D)

    perm = tuple(range(len(batch_shape))) + tuple(np.array([2, 0, 1, 3]) + len(batch_shape))
    return np.transpose(video, perm)


def liang_barsky(bottom, top, left, right, y0, x0, y1, x1):
    """ Compute the intersection between a rectangle and a line segment.

        rect is sepcified by (bottom, top, left, right)
        line segment specified by (y0, x0), (y1, x1)

        If no intersection, returns None.

        Otherwise, returns (r, s) where (y0, x0) + r * (y1 - y0, x1 - x0) is the location of the "ingoing"
        intersection, and (y0, x0) + s * (y1 - y0, x1 - x0) is the location of the "outgoing" intersection.
        It will always hold that 0 <= r <= s <= 1. If the line segment starts inside the rectangle then r = 0;
        and if it stops inside the rectangle then s = 1.

    """
    assert bottom < top
    assert left < right

    dx = x1 - x0
    dy = y1 - y0

    checks = ((-dx, -(left - x0)),
              (dx, right - x0),
              (-dy, -(bottom - y0)),
              (dy, top - y0))

    out_in = [0]
    in_out = [1]

    for p, q in checks:
        if p == 0 and q < 0:
            return None

        if p != 0:
            target_list = out_in if p < 0 else in_out
            target_list.append(q / p)

    _out_in = max(out_in)
    _in_out = min(in_out)

    if _out_in < _in_out:
        return _out_in, _in_out
    else:
        return None


# NoAnswer = object()
# def _test_liang_barsky(*args, ref_answer=NoAnswer):
#     answer = liang_barsky(*args)
#     print("{}: {}".format(args, answer))
#
#     if ref_answer is not NoAnswer:
#         assert answer == ref_answer
# if __name__ == "__main__":
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, 1.5, 2.5, ref_answer=(1/4, 3/4))
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, 1.5, .99, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, 1.5, 1, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, 1.5, 1.01, ref_answer=(0.5 / 0.51, 1))
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, -1.5, -2.5, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 2.5, 0.5, 2.5, 2.5, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 0.5, 2.5, 2.5, 2.5, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 0, 0, 2, 2, ref_answer=(0.5, 1))
#     _test_liang_barsky(1, 2, 1, 2, 0, .99, 2, 2.99, ref_answer=(0.5, 0.505))
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 1.5, 3, 3, ref_answer=(0, 1/3))


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


def exactly_2d(x, return_leading_shape=False):
    leading_shape = x.shape[:-1]

    if return_leading_shape:
        return leading_shape, x.reshape(-1, x.shape[-1])
    else:
        return x.reshape(-1, x.shape[-1])


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


def animate(
        images, *other_images, labels=None, interval=500,
        path=None, square_grid=True, **kwargs):
    """ Assumes `images` has shape (batch_size, n_frames, H, W, D) """

    all_images = [images, *other_images]
    n_image_sets = len(all_images)
    B, T = images.shape[:2]

    if square_grid:
        fig, _axes = square_subplots(B, n_repeats=n_image_sets, repeat_horizontal=True)
    else:
        fig, _axes = plt.subplots(B, n_image_sets)

    axes = _axes.reshape(-1, n_image_sets)

    plots = np.zeros((B, n_image_sets), dtype=np.object)

    for i in range(B):
        if labels is not None:
            axes[i, 0].set_title(str(labels[i]))

        for j in range(n_image_sets):
            ax = axes[i, j]
            ax.set_axis_off()

            plots[i, j] = ax.imshow(np.squeeze(all_images[j][i, 0]))

    plt.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=.98, wspace=0.1, hspace=0.1)

    def func(t):
        for i in range(B):
            for j in range(n_image_sets):
                plots[i, j].set_array(np.squeeze(all_images[j][i, t]))

    anim = animation.FuncAnimation(fig, func, frames=T, interval=interval)

    if path is not None:
        path = path + '.mp4'
        anim.save(path, writer='ffmpeg', codec='hevc', extra_args=['-preset', 'ultrafast'])

    return fig, _axes, anim, path


def square_subplots(N, n_repeats=1, repeat_horizontal=True, **kwargs):
    sqrt_N = int(np.ceil(np.sqrt(N)))
    m = int(np.ceil(N / sqrt_N))
    import matplotlib.pyplot as plt
    if repeat_horizontal:
        fig, axes = plt.subplots(m, sqrt_N*n_repeats, **kwargs)
    else:
        fig, axes = plt.subplots(m*n_repeats, sqrt_N, **kwargs)
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

    if lines[header_idx+2].startswith('|  No running processes found'):
        return []

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
    """ View readme files in a directory of experiments, sorted by the time at
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


def find_git_directories():
    all_packages = pip_freeze()
    all_packages = all_packages.split('\n')
    git_packages = [p.split('=')[-1] for p in all_packages if p.startswith('-e git+')]

    version_controlled_dirs = set()
    for p in git_packages:
        package = importlib.import_module(p)
        directory = os.path.dirname(package.__file__)

        # Check whether any ancestor directory contains a .git directory
        while directory:
            git_dir = os.path.join(directory, '.git')
            if os.path.isdir(git_dir):
                version_controlled_dirs.add(directory)
                break
            directory = os.path.dirname(directory)

    return sorted(version_controlled_dirs)


def summarize_git_repo(directory, n_logs=10, diff=False):
    s = []
    with cd(directory):
        s.append("*" * 40)
        s.append("git summary for directory {}\n".format(directory))

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

        s.append("\nEnd of git summary for directory {}".format(directory))
        s.append("*" * 40 + "\n")
    return '\n'.join(s)


def summarize_git_repos(**summary_kwargs):
    s = []
    git_dirs = find_git_directories()
    for git_dir in git_dirs:
        git_summary = summarize_git_repo(git_dir, **summary_kwargs)
        s.append(git_summary)
    return '\n'.join(s)


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

        return ExperimentDirectory(os.path.join(self.path, filename), force_fresh=force_fresh)

    def __str__(self):
        return "ExperimentStore({})".format(self.path)

    def __repr__(self):
        return str(self)

    def experiment_finished(self, exp_dir, success):
        dest_name = 'complete' if success else 'incomplete'
        dest_path = os.path.join(self.path, dest_name)
        os.makedirs(dest_path, exist_ok=True)
        shutil.move(exp_dir.path, dest_path)
        exp_dir.path = os.path.join(dest_path, os.path.basename(exp_dir.path))

    def isolate_n_latest(self, n):
        files = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.startswith(self.prefix)]
        exp_dirs = [f for f in files if os.path.isdir(f)]
        exp_dirs_with_mtime = [(os.path.getmtime(d), d) for d in exp_dirs]
        latest_exp_dirs = [d for _, d in sorted(exp_dirs_with_mtime)[-n:]]

        latest_dir = os.path.join(self.path, 'tensorboard_{}_latest'.format(n))

        try:
            shutil.rmtree(latest_dir)
        except FileNotFoundError:
            pass

        os.makedirs(latest_dir, exist_ok=False)

        for exp_dir in latest_exp_dirs:
            make_symlink(os.path.join('..', exp_dir), os.path.join(latest_dir, os.path.basename(exp_dir)))

        return latest_dir


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

    def __init__(self, path, force_fresh=False):
        self.path = path
        _checked_makedirs(path, force_fresh)

    def path_for(self, *path, is_dir=False):
        """ Get path for a file, creating necessary subdirs. """
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
            f.write(summarize_git_repos(diff=git_diff))

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

    with tempfile.NamedTemporaryFile(mode='w',
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

    def fileno(self):
        for s in self.streams:
            if hasattr(s, "fileno"):
                return s.fileno()


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
        Whether to add the current date/time to the filename. If true, the date comes right after
        the main title.
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

    if add_date:
        date_time_string = str(datetime.datetime.now()).split('.')[0]
        for c in ": -":
            date_time_string = date_time_string.replace(c, '_')
        labels.append(date_time_string)

    key_vals = list(config_dict.items())
    key_vals.sort(key=lambda x: x[0])

    for key, value in key_vals:
        if not isinstance(key, str):
            raise ValueError("keys in config_dict must be strings.")
        if not isinstance(value, str):
            raise ValueError("values in config_dict must be strings.")

        if not str(key) in omit:
            labels.append(kvsep.join([key, value]))

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
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|()1{}[]?-_+~<>i!lI;:,\"^`'. "


def char_map(value):
    """ Maps a relative "sparsity" or "lightness" value in [0, 1) to a character. """
    if value >= 1:
        value = 1 - 1e-6
    n_bins = len(ascii_art_chars)
    bin_id = int(value * n_bins)
    return ascii_art_chars[bin_id]


def plt_to_img():
    with tempfile.TemporaryFile() as fp:
        plt.savefig(fp, format='png', bbox_inches='tight')
        fp.seek(0)
        img = imageio.imread(fp)
    plt.close('all')
    gc.collect()

    return img


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
    """ An object that can have `Param` class attributes. These class attributes will be
        turned into instance attributes at instance creation time. To set a value for the instance
        attributes, we perform the following checks (the first value that is found is used):

        1. check the kwargs passed into class constructor for a value with key "<param-name>"
        2. check dps.cfg for values with keys of the form "<class-name>:<param-name>". `class-name`
           can be the class of the object or any base class thereof. More derived/specific classes
           will be checked first (specifically, we iterate through classes in order of the MRO).
        3. check dps.cfg for values with name "<param-name>"
        4. fallback on the param's default value, if one was supplied.

        If no value is found by this point, an AttributeError is raised.

        A note on deep copies: at instance creation time, we store values for all the parameters.
        Deep copies are created by creating a new object using those creation-time parameter values.

    """
    _resolved = False

    def __new__(cls, *args, **kwargs):
        obj = super(Parameterized, cls).__new__(cls)
        obj._resolve_params(**kwargs)

        # Stored for copying purposes, to get parameter as they are before __init__ is called.
        obj._params_at_creation_time = obj.param_values()

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
        """ Return the params that would be created if an object of the
            current class were constructed in the current context with the given kwargs. """
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
            raise Exception("Parameters have not yet been resolved.")
        return {n: getattr(self, n) for n in self.param_names()}

    def __deepcopy__(self, memo):
        cls = self.__class__
        kwargs = self._params_at_creation_time
        result = cls.__new__(cls, **kwargs)
        result.__init__(**kwargs)
        memo[id(self)] = result
        return result


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du', '-sh', str(path)]).split()[0].decode('utf-8')


class pdb_postmortem:
    def __init__(self, do_it=True):
        self.do_it = do_it

    def __enter__(self):
        pass

    def __exit__(self, type_, value, tb):
        if self.do_it and type_:
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
        if subname == "_":
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
        if seed < 0:
            seed = None
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


class HierDict(dict):
    """ The main special property is that __setitem__ and __getitem__ create nested HierDicts
        when the keys are strings containing the character `self._sep`. This is similar to Config's
        behavior, but is intended to be more principled. Functions like `keys`, `values` and `items`
        only regard the top level of keys. Also, keys can be set and retrieved using attributes.

    """
    _sep = ":"
    _reserved_keys = None

    def __init__(self, sep=':'):
        self._sep = sep

    def flatten(self):
        return {k: self[k] for k in self._flat_keys()}

    def _flat_keys(self):
        stack = [iter(dict.items(self))]
        key_prefix = ''

        while stack:
            new = next(stack[-1], None)
            if new is None:
                stack.pop()
                key_prefix = key_prefix.rpartition(self._sep)[0]
                continue

            key, value = new
            nested_key = key_prefix + self._sep + key

            if isinstance(value, dict) and value:
                stack.append(iter(value.items()))
                key_prefix = nested_key
            else:
                yield nested_key[1:]

    def __str__(self):
        s = "{}{}\n".format(self.__class__.__name__, json.dumps(self, sort_keys=True, indent=4, default=str))
        return s

    def __repr__(self):
        s = "{}{}\n".format(self.__class__.__name__, json.dumps(self, sort_keys=True, indent=4, default=repr))
        return s

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, key):
        if isinstance(key, str) and self._sep in key:
            key, _, rest = key.partition(self._sep)
            return super().__getitem__(key).__getitem__(rest)
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, str) and self._sep in key:
            key, _, rest = key.partition(self._sep)
            try:
                d = self[key]
                if not isinstance(d, HierDict):
                    raise Exception("HierDict about to access object {} with key {}.".format(d, key))
            except KeyError:
                d = self[key] = HierDict(self._sep)

            d[rest] = value
        else:
            self._validate_key(key)
            return super().__setitem__(key, value)

    def __delitem__(self, key):
        if isinstance(key, str) and self._sep in key:
            key, _, rest = key.partition(self._sep)
            return super().__getitem__(key).__delitem__(rest)
        else:
            return super().__delitem__(key)

    def __getattr__(self, key):
        if key in HierDict._reserved_keys:
            return super().__getattr__(key)

        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("Could not find attribute called `{}`.".format(key))

    def __setattr__(self, key, value):
        if key in HierDict._reserved_keys:
            super().__setattr__(key, value)
            return

        self[key] = value

    def _validate_key(self, key):
        msg = "Bad key for config: `{}`.".format(key)
        assert not key.startswith('_'), msg
        assert key not in HierDict._reserved_keys, msg

    def copy(self, _d=None, **kwargs):
        """ Copy and update at the same time. """
        new = copy.deepcopy(self)
        if _d:
            new.update(_d)
        new.update(**kwargs)
        return new

    def update(self, _d=None, **kwargs):
        if _d is not None:
            for k, v in _d.items():
                self[k] = v

        for k, v in kwargs.items():
            self[k] = v


HierDict._reserved_keys = dir(HierDict)


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
        """
        TODO: there is currently a mismatch of behaviour between __setitem__ and update in the case
        of nested dictionaries. e.g.

        x = Config(y=dict(a=1))
        x.update(y=dict(b=1))
        print(x)

        >>> Config({'y': {'a': 1, 'b': 1}})

        x = Config(y=dict(a=1))
        x.y=dict(b=1)
        print(x)

        >>> Config({'y': {'b': 1}})

        update augments the current nested dictionary, __setitem__ replaces it.

        """
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

    def update_from_command_line(self, strict=True):
        cl_args = clify.wrap_object(self, strict=strict).parse()
        self.update(cl_args)

    def freeze(self, remove_callable=False):
        _config = Config()
        for key in self.keys():
            value = self[key]
            if remove_callable and callable(value):
                value = str(value)
            _config[key] = value
        return _config


Config._reserved_keys = dir(Config)


def update_scratch_dir(config, new_scratch_dir):
    def fixup_dir(name):
        attr_name = name + "_dir"
        dir_name = os.path.join(new_scratch_dir, name)
        dir_name = process_path(dir_name)
        setattr(config, attr_name, dir_name)
        os.makedirs(dir_name, exist_ok=True)

    fixup_dir("data")
    fixup_dir("model")
    fixup_dir("local_experiments")
    fixup_dir("parallel_experiments_build")
    fixup_dir("parallel_experiments_run")


config_template = """
config = dict(
    start_tensorboard=True,
    tbport=6006,
    reload_interval=10,
    show_plots=False,
    verbose=False,
    use_gpu=False,
    per_process_gpu_memory_fraction=0,
    gpu_allow_growth=True,
    parallel_exe="$HOME/.local/bin/parallel",
    scratch_dir="{scratch_dir}",
    slurm_preamble='''
export OMP_NUM_THREADS=1
module purge
module load python/3.6.3
module load scipy-stack
source "$VIRTUALENVWRAPPER_BIN"/virtualenvwrapper.sh
workon her_curriculum''',
    ssh_hosts=(
        ["ecrawf6@lab1-{{}}.cs.mcgill.ca".format(i+1) for i in range(16)]
        + ["ecrawf6@lab2-{{}}.cs.mcgill.ca".format(i+1) for i in range(51)]
        + ["ecrawf6@cs-{{}}.cs.mcgill.ca".format(i+1) for i in range(32)]
    ),
    ssh_options=(
        "-oPasswordAuthentication=no "
        "-oStrictHostKeyChecking=no "
        "-oConnectTimeout=5 "
        "-oServerAliveInterval=2"
    ),
)
"""


def _load_system_config(key=None):
    home = os.getenv("HOME")
    config_dir = os.path.join(home, ".config")
    config_loc = os.path.join(config_dir, "dps_config.py")

    if not os.path.exists(config_loc):
        print("Creating config at {}...".format(config_loc))
        default_scratch_dir = os.path.join(home, "dps_data")
        scratch_dir = input("Enter a location to create a scratch directory for dps "
                            "(for saving experiment results, cached datasets, etc.). "
                            "Leave blank to accept the default of '{}'.\n".format(default_scratch_dir))
        scratch_dir = process_path(scratch_dir) or default_scratch_dir

        config = config_template.format(scratch_dir=scratch_dir)

        with open(config_loc, "w") as f:
            f.write(config)

    config_module_spec = importlib.util.spec_from_file_location("dps_config", config_loc)
    config_module = config_module_spec.loader.load_module()

    config = Config(**config_module.config)

    def fixup_dir(name):
        attr_name = name + "_dir"
        dir_name = getattr(config, attr_name, None)
        if dir_name is None:
            dir_name = os.path.join(config.scratch_dir, name)
            dir_name = process_path(dir_name)
            setattr(config, attr_name, dir_name)
        os.makedirs(dir_name, exist_ok=True)

    fixup_dir("data")
    fixup_dir("model")
    fixup_dir("local_experiments")
    fixup_dir("parallel_experiments_build")
    fixup_dir("parallel_experiments_run")

    return config


SYSTEM_CONFIG = _load_system_config()


class ClearConfig(Config):
    def __init__(self, _d=None, **kwargs):
        config = _load_system_config()
        if _d:
            config.update(_d)
        config.update(kwargs)
        super().__init__(**config)


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
                self._stack.append(SYSTEM_CONFIG.copy())
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

    def update_from_command_line(self, strict=True):
        cl_args = clify.wrap_object(self, strict=strict).parse()
        self.update(cl_args)


def restart_tensorboard(logdir, port=6006, reload_interval=120):
    sp = subprocess
    print("Killing old tensorboard process...")
    try:
        command = "fuser {}/tcp -k".format(port)
        sp.run(command.split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        print("Killing tensorboard failed:")
        print(e.output)
    print("Restarting tensorboard process...")
    command = "tensorboard --logdir={} --port={} --reload_interval={}".format(logdir, port, reload_interval)
    print(command)
    sp.Popen(command.split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    print("Done restarting tensorboard.\n")


def map_structure(func, *args, is_leaf):
    if is_leaf(args[0]):
        return func(*args)
    else:
        if isinstance(args[0], dict):
            arg_keys = [a.keys() for a in args]
            assert all(keys == arg_keys[0] for keys in arg_keys), (
                "Arguments do not have same structure: {}.".format(arg_keys))

            new_dict = {
                k: map_structure(func, *[a[k] for a in args], is_leaf=is_leaf)
                for k in args[0]}
            return type(args[0])(new_dict)
        else:
            arg_lens = [len(a) for a in args]
            assert all(np.array(arg_lens) == arg_lens[0]), (
                "Arguments do not have same structure: {} ".format(arg_lens))

            new_list = [map_structure(func, *[a[i] for a in args], is_leaf=is_leaf)
                        for i in range(arg_lens[0])]
            return type(args[0])(new_list)


def test_map_structure():
    a = dict(a=[1, 2], b=3)
    b = dict(a=[3, 4], b=10)

    result = map_structure(lambda x, y: x + y, a, b, is_leaf=lambda x: isinstance(x, int))
    assert tuple(result["a"]) == (4, 6)
    assert result["b"] == 13

    result = map_structure(lambda *x: None, a, is_leaf=lambda x: isinstance(x, int))
    assert tuple(result["a"]) == (None, None)
    assert result["b"] is None

    result = map_structure(lambda *x: None, a, b, is_leaf=lambda x: isinstance(x, int))
    assert tuple(result["a"]) == (None, None)
    assert result["b"] is None
