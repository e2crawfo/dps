from contextlib import contextmanager
import numpy as np
import signal
import time
import re
import os
import traceback
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
from functools import wraps, partial
import inspect
import hashlib
from zipfile import ZipFile
import importlib
import json
import gc
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import matplotlib as mpl
import imageio
from skimage.transform import resize
from scipy.spatial.transform import Rotation
import pandas as pd
from tabulate import tabulate
from IPython.lib.pretty import pretty, for_type

import clify
import dps


# colors from the Dark2 colorscheme from colorbrewer
green = np.array([27, 158, 119]) / 255.
orange = np.array([217, 95, 2]) / 255.
blue = np.array([117, 112, 179]) / 255.
pink = np.array([231, 41, 138]) / 255.


def to_rgb(c):
    return np.array(mpl.colors.to_rgb(c))


def random_unit_vector(shape, axis=-1, dtype=np.float32):
    try:
        shape = (int(shape),)
    except (TypeError, ValueError):
        pass
    x = np.random.randn(*shape).astype(dtype)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.clip(norm, np.finfo(dtype).eps, None)
    return x / norm


def describe_tensor(tensor):
    df = pd.DataFrame(tensor.flatten())
    print(df.describe()[0])


def describe_structure(tensors, floatfmt=None):
    if floatfmt is None:
        floatfmt = ".3g"

    print()
    tensors = AttrDict(tensors).flatten()

    table = []
    for k, t in sorted(tensors.items()):
        if t is None:
            table.append(dict(key=k))
            continue

        flat_t = t.flatten()
        n_nan = np.isnan(flat_t).sum()
        n_inf = np.isinf(flat_t).sum()

        desc = pd.DataFrame(flat_t).describe()[0]

        record = dict(key=k, shape=tuple(t.shape), dtype=t.dtype, **desc, n_nan=n_nan, n_inf=n_inf)
        table.append(record)

    print(tabulate(table, headers="keys", floatfmt=floatfmt))
    print()


class PiecewiseConstantFn:
    def __init__(self, transitions, values):
        assert len(values) == len(transitions)+1
        assert tuple(sorted(transitions)) == tuple(transitions)

        self.transitions = tuple(transitions)
        self.values = tuple(values)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return "<{} transitions={}, values={}>".format(self.__class__.__name__, self.transitions, self.values)

    def __call__(self, t):
        for transition, value in zip(self.transitions, self.values):
            if t < transition:
                return value
        else:
            return self.values[-1]


def interpret_fov(fov):
    """ The focal lengths under give fov, assuming that the dimensions of the image plane are (2, 2).

        and

        0.5 times the dimensions of the image plane under given fov, assuming an image plane at distance 1 from camera.

    """
    try:
        fov_x, fov_y = tuple(fov)
    except Exception:
        fov_x, fov_y = fov + 0., fov + 0.

    fx = 1 / np.tan(fov_x/2)
    fy = 1 / np.tan(fov_y/2)

    sx = np.tan(fov_x/2)
    sy = np.tan(fov_y/2)

    return fx, fy, sx, sy


use_from_matrix = hasattr(Rotation, 'from_matrix')


def build_transformation_matrix(kind, r, t=None, **rotation_kwargs):
    """
    `kind` indicates how `r` is to be interpreted, and in particular specifies which `from_<blank>` function
    (exported by Rotation) to use. Must be one of: 'quat', 'rotvec', 'euler_<order>', 'matrix', 'dcm'.
    In the case of 'euler', the <order> portion should specify the axis order.
    The last two are synonymous (from_dcm was renamed to from_matrix).

    r specifies a (batch of) rotation matrices, t is a (batch of) translation vectors.

    """
    if kind == 'dcm' and use_from_matrix:
        kind = 'matrix'
    elif kind == 'matrix' and not use_from_matrix:
        kind = 'dcm'

    if kind in ['dcm', 'matrix']:
        n_trailing_dims = 2
    else:
        n_trailing_dims = 1
    batch_dims = r.shape[:-n_trailing_dims]
    trailing_dims = r.shape[-n_trailing_dims:]
    n_batch_dims = int(np.prod(batch_dims))
    r = r.reshape(n_batch_dims, *trailing_dims)

    if kind.startswith('euler'):
        order = kind.split('_')[1]
        function = partial(Rotation.from_euler, order)
    else:
        function = getattr(Rotation, 'from_{}'.format(kind))

    rotation = function(r, **rotation_kwargs)

    if use_from_matrix:
        R = rotation.as_matrix()
    else:
        R = rotation.as_dcm()

    output = np.zeros((*batch_dims, 4, 4))
    output[..., :3, :3] = R.reshape(*batch_dims, 3, 3)
    output[..., 3, 3] = 1.0

    if t is not None:
        output[..., :3, 3] = t

    return output


class RenderHook:
    N = 16
    is_training = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def start_stage(self, training_loop, updater, stage_idx):
        pass

    @staticmethod
    def normalize_images(images, dim=-1):
        mx = images.reshape(*images.shape[:-3], -1).max(axis=-1)
        return images / mx[..., None, None, None]

    @staticmethod
    def remove_patches(ax):
        for obj in ax.findobj(match=patches.Patch):
            try:
                obj.remove()
            except NotImplementedError:
                pass

    def imshow(self, ax, frame, remove_patches=True, vmin=0.0, vmax=1.0, **kwargs):
        """ If ax already has an image, uses set_array on that image instead of doing imshow.
            Allows this function to work well with animations. """

        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]

        frame = np.clip(frame, vmin, vmax)
        frame = np.where(np.isnan(frame), 0, frame)

        if ax.images:
            ax.images[0].set_array(frame)
            if remove_patches:
                self.remove_patches(ax)
        else:
            ax.imshow(frame, vmin=vmin, vmax=vmax, **kwargs)

    def path_for(self, name, updater, ext="pdf"):
        local_step = np.inf if dps.cfg.overwrite_plots else "{:0>10}".format(updater.step)

        if ext is None:
            basename = 'stage={:0>4}_local_step={}'.format(updater.stage_idx, local_step)
        else:
            basename = 'stage={:0>4}_local_step={}.{}'.format(updater.stage_idx, local_step, ext)
        return updater.exp_dir.path_for('plots', name, basename)

    def savefig(self, name, fig, updater, is_dir=True):
        if is_dir:
            path = self.path_for(name, updater)
            fig.savefig(path)
            plt.close(fig)

            shutil.copyfile(
                path,
                os.path.join(
                    os.path.dirname(path),
                    'latest_stage{:0>4}.pdf'.format(updater.stage_idx)))
        else:
            path = updater.exp_dir.path_for('plots', name + ".pdf")
            fig.savefig(path)
            plt.close(fig)

    def save_video(self, name, fig, anim, updater):
        path = self.path_for(name, updater, ext="mp4")
        anim.save(path, writer='ffmpeg', codec='hevc', extra_args=['-preset', 'ultrafast'])

        plt.close(fig)

        latest = os.path.join(os.path.dirname(path), 'latest_stage{:0>4}.mp4'.format(updater.stage_idx))
        shutil.copyfile(path, latest)


def flush_print(*args, flush=True, **kwargs):
    print(*args, **kwargs, flush=flush)


class HashableDist:
    """ Wraps a distribution from scipy.stats.distributions, making it hashable. """

    def __init__(self, dist_class, *args, **kwargs):
        dist = dist_class(*args, **kwargs)
        self._dist = dist
        self.dist_class = dist_class

    def __getattr__(self, attr):
        return getattr(self._dist, attr)

    def __str__(self):
        args_strs = [str(a) for a in self.args] + ["{}={}".format(k, v) for k, v in sorted(self.kwds.items())]
        args_str = ", ".join(args_strs)
        return "<{} - {}>".format(self.dist_class.name, args_str)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


def walk_images(path, extensions=None, concat=False):
    extensions = extensions or 'jpg jpeg png tif'
    if isinstance(extensions, str):
        extensions = extensions.split()
    extensions = [ext if ext.startswith('.') else '.' + ext for ext in extensions]

    for dir_name, _, filenames in os.walk(path):
        local_path_start = len(path) + (0 if path.endswith('/') else 1)
        local_path = dir_name[local_path_start:]

        for f in filenames:
            if any(f.endswith(ext) for ext in extensions):
                if concat:
                    yield os.path.join(path, local_path, f)
                else:
                    yield os.path.join(local_path, f)


def atleast_nd(array, n):
    diff = n - len(array.shape)
    if diff > 0:
        s = (Ellipsis,) + (None,) * diff
        array = array[s]
    return array


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


def liang_barsky(rect, line):
    """ Compute the intersection between an axis-aligned rectangle and a line segment, in n-dimensions.

        rect is an array-like with shape (2, n) where the first row gives the lower bounds of the rectangle
        in each dimension, and the second row gives the upper bounds.

        line is an array-like with shape (2, n) where the first row gives the start of the line segment,
        and the second row gives the end of the line segment.

        If no intersection, returns None.

        Otherwise, returns (r, s) where line[0] + r * (line[1] - line[0]) is the location of the "ingoing"
        intersection, and line[0] + s * (line[1] - line[0]) is the location of the "outgoing" intersection.
        It will always hold that 0 <= r <= s <= 1. If the line segment starts inside the rectangle then r = 0;
        and if it stops inside the rectangle then s = 1.

    """
    rect = np.array(rect)  # (2, n)
    line = np.array(line)  # (2, n)

    n = rect.shape[1]
    assert line.shape[1] == n
    assert rect.shape[0] == 2
    assert line.shape[0] == 2

    assert (rect[0] < rect[1]).all()

    delta = line[1] - line[0]

    checks_1 = np.stack([-delta, -(rect[0] - line[0])], axis=1)
    checks_2 = np.stack([delta, rect[1] - line[0]], axis=1)

    checks = np.concatenate([checks_1, checks_2], axis=0)

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


def create_maze(shape):
    # Random Maze Generator using Depth-first Search
    # http://en.wikipedia.org/wiki/Maze_generation_algorithm
    # FB - 20121214
    my, mx = shape
    maze = np.zeros(shape)
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # start the maze from a random cell
    stack = [(np.random.randint(0, my), np.random.randint(0, mx))]

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


@contextmanager
def numpy_print_options(**print_options):
    old_print_options = np.get_printoptions()
    try:
        np.set_printoptions(**print_options)
        yield
    finally:
        np.set_printoptions(**old_print_options)


def annotate_with_rectangles(ax, annotations, colors=None, lw=1):
    if colors is None:
        colors = list(plt.get_cmap('Dark2').colors)

    for valid, _, _id, t, b, l, r in annotations:
        if not valid:
            continue

        h = b - t
        w = r - l
        color = colors[int(_id) % len(colors)]
        rect = patches.Rectangle(
            (l, t), w, h, clip_on=False, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)


def animate(
        *images, labels=None, interval=500,
        path=None, block_shape=None, annotations=None, fig_unit_size=1,
        text=None, text_loc=None, fontsize='x-small', text_color='black', normalize=None,
        **kwargs):
    """ Assumes each element of `images` has shape (batch_size, n_frames, H, W, C). Each element is a set of images.

    `annotations` only implemented for the first set of images.

    `block_shape` controls the shape of the set of plots for an individual example.

    `text` should be an array of strings with shape (batch_size, n_frames)
    or a dictionary where the keys are indices and each value is an array of strings with shape (batch_size, n_frames).
    In the former case, text plotted only on the first image of each set. In the later case, the keys say
    which plot within the block to put the text on.

    `text_loc` specifies where to draw the text.

    `normalize` can be a list of length len(other_images) + 1. Each entry should be a bool saying
    whether the corresponding data should be normalized (over the whole video) or not.

    """
    n_image_sets = len(images)
    B, T = images[0].shape[:2]

    if block_shape is None:
        N = n_image_sets
        sqrt_N = int(np.ceil(np.sqrt(N)))
        m = int(np.ceil(N / sqrt_N))
        block_shape = (m, sqrt_N)

    images = [
        img[..., 0] if img.ndim == 5 and img.shape[-1] == 1 else img
        for img in images]

    assert np.prod(block_shape) >= n_image_sets

    fig, axes = square_subplots(B, block_shape=block_shape, fig_unit_size=fig_unit_size)
    time_text = fig.text(0.01, .99, 't=0', ha='left', va='top', transform=fig.transFigure, fontsize=12)

    plots = np.zeros_like(axes)
    text_elements = np.zeros_like(axes)

    if text is None:
        text = {}
    elif not isinstance(text, dict):
        text = {0: text}

    if text_loc is None:
        text_loc = (0.05, 0.95)

    if labels is not None:
        for j in range(n_image_sets):
            axes[0, j].set_title(str(labels[j]))

    for ax in axes.flatten():
        set_axis_off(ax)

    for i in range(B):
        for j in range(n_image_sets):
            ax = axes[i, j]

            _normalize = False
            if normalize is not None:
                _normalize = normalize[j]

            # A note on vmin/vmax: vmin and vmax are set permanently when imshow is called.
            # They are not modified when you call set_array.

            if _normalize:
                vmin = images[j][i].min()
                vmax = images[j][i].max()
                mean = images[j][i].mean()

                ax.set_ylabel('min={:.3f}, mean={:.3f}, max={:.3f}'.format(vmin, mean, vmax))
            else:
                vmin = 0.0
                vmax = 1.0

            plots[i, j] = ax.imshow(images[j][i, 0], vmin=vmin, vmax=vmax)

            text_elements[i, j] = ax.text(
                *text_loc, '', ha='left', va='top', transform=ax.transAxes, fontsize=fontsize, color=text_color)

    plt.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=.98, wspace=0.1, hspace=0.1)

    def func(t):
        time_text.set_text('t={}'.format(t))

        for i in range(B):
            for j in range(n_image_sets):
                plots[i, j].set_array(images[j][i, t])

                ax = axes[i, j]
                for obj in ax.findobj(match=plt.Rectangle):
                    try:
                        obj.remove()
                    except NotImplementedError:
                        pass

                if j in text:
                    text_elements[i, j].set_text(text[j][i, t])

            if annotations is not None:
                ax = axes[i, 0]
                annotate_with_rectangles(ax, annotations[i][t])

    anim = animation.FuncAnimation(fig, func, frames=T, interval=interval)

    if path is not None:
        if not path.endswith('.mp4'):
            path = path + '.mp4'

        anim.save(path, writer='ffmpeg', codec='hevc', extra_args=['-preset', 'ultrafast'])

    return fig, axes, anim, path


def add_rect(ax, top, left, height, width, color, lw=2, **kwargs):
    kwargs.update(linewidth=lw)
    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = 'none'
    rect = patches.Rectangle(
        (left, top), width, height, edgecolor=color, **kwargs)
    ax.add_patch(rect)


def add_dotted_rect(ax, top, left, height, width, c1, c2, **kwargs):
    if 'ls' in kwargs:
        del kwargs['ls']
    if 'linestyle' in kwargs:
        del kwargs['ls']

    add_rect(ax, top, left, height, width, c1, ls='-', **kwargs)
    add_rect(ax, top, left, height, width, c2, ls=':', **kwargs)


def square_subplots(N, block_shape=None, fig_unit_size=1, axes_off=False, **kwargs):
    w = int(np.ceil(np.sqrt(N)))
    h = int(np.ceil(N / w))

    if block_shape is None:
        block_shape = (1, 1)

    axes_shape = (h*block_shape[0], w*block_shape[1])

    if 'figsize' not in kwargs:
        # figsize is (width, height)
        kwargs['figsize'] = (
            axes_shape[1] * fig_unit_size,
            axes_shape[0] * fig_unit_size
        )

    fig, axes = plt.subplots(*axes_shape, **kwargs)
    axes = np.array(axes).reshape(*axes_shape)

    _axes = np.zeros((w*h, int(np.prod(block_shape))), dtype=np.object)
    for i in range(w*h):
        _h = i // w
        _w = i % w

        _axes[i, :] = axes[
            _h * block_shape[0]: (_h+1) * block_shape[0],
            _w * block_shape[1]: (_w+1) * block_shape[1]
        ].flatten()
    axes = np.array(_axes)

    if axes_off:
        for ax in axes.flatten():
            set_axis_off(ax)

    return fig, axes


def grid_subplots(h, w, fig_unit_size, axes_off=False):
    try:
        fig_unit_size = int(fig_unit_size)
        unit_size_h = unit_size_w = fig_unit_size
    except Exception:
        unit_size_h, unit_size_w = fig_unit_size
    fig, axes = plt.subplots(h, w, figsize=(w * unit_size_w, h * unit_size_h))
    axes = np.array(axes).reshape(h, w)  # to fix the inconsistent way axes is return if h==1 or w==1

    if axes_off:
        for ax in axes.flatten():
            set_axis_off(ax)

    return fig, axes


def set_axis_off(ax, remove_border=False):
    """ Differs from ax.set_axis_off() in that axis labels are still shown. """

    if remove_border:
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_xticks([])
    ax.set_yticks([])


def nvidia_smi(robust=True):
    try:
        p = subprocess.run("nvidia-smi".split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.stdout.decode()
    except Exception as e:
        if robust:
            return "Exception while calling nvidia-smi: {}".format(e)
        else:
            raise


# Different versions of nvidia-smi use different headers and require different functions for parsing processes.
_nvidia_smi_table_end = "+-----------------------------------------------------------------------------+"
_nvidia_smi_no_processes_found = "|  No running processes found                                                 |"


def _nvidia_smi_parse_processes(s):
    lines = s.split('\n')

    cuda_version = float(lines[2].split()[-2])
    if cuda_version >= 11:
        processes_header_line = "|        ID   ID                                                   Usage      |"

        def parse_process_line(tokens):
            """
            Example line:
            "|    0   N/A  N/A    240337      C   python                           1115MiB |"
            """
            gpu_idx = int(tokens[1])
            pid = int(tokens[4])
            type = tokens[5]
            process_name = ' '.join(tokens[6:-2])  # To handle process names containg spaces
            memory_usage = tokens[-2]
            memory_usage_mb = int(memory_usage[:-3])

            return (gpu_idx, pid, type, process_name, memory_usage_mb)
    else:
        processes_header_line = "|  GPU       PID   Type   Process name                             Usage      |"

        def parse_process_line(tokens):
            """
            Example line:
            "|    0      1234      C   python    72MiB |"
            """
            gpu_idx = int(tokens[1])
            pid = int(tokens[2])
            type = tokens[3]
            process_name = ' '.join(tokens[4:-2])  # To handle process names containg spaces
            memory_usage = tokens[-2]
            memory_usage_mb = int(memory_usage[:-3])

            return (gpu_idx, pid, type, process_name, memory_usage_mb)

    header_idx = None
    table_end_idx = None
    for i, line in enumerate(lines):
        if line == processes_header_line:
            header_idx = i
        elif header_idx is not None and line == _nvidia_smi_table_end:
            table_end_idx = i

    assert header_idx is not None, "Malformed nvidia-smi string:\n{}".format(s)
    assert table_end_idx is not None, "Malformed nvidia-smi string:\n{}".format(s)

    if lines[header_idx+2].startswith('|  No running processes found'):
        return []

    processes = []

    first_line = lines[header_idx+2]
    if first_line == _nvidia_smi_parse_processes:
        return []

    for line in lines[header_idx+2:table_end_idx]:
        tokens = line.split()
        process_info = parse_process_line(tokens)
        processes.append(process_info)

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
    # d = AttrDict(d)

    if not name_params:
        name_params = d.keys()
        # name_params = d.flatten().keys()

    param_str = []
    for name in sorted(name_params):
        param_str.append("{}={}".format(name, d[name]))

    param_str = "_".join(param_str)
    # param_str = ", ".join(param_str)
    param_hash = hashlib.sha1(param_str.encode()).hexdigest()
    return param_hash


CLEAR_CACHE = False


def set_clear_cache(value):
    """ If called with True, then whenever `sha_cache` function is instantiated, it will ignore
        any cache saved to disk, and instead just call the function as normal, saving the results
        as the new cache value. """
    global CLEAR_CACHE
    CLEAR_CACHE = value


def sha_cache(directory='.cache', recurse=False, verbose=False, exclude_kwargs=None):
    """

    Parameters
    ----------
    exclude_kwargs: str or list of str, optional
        List of kwarg names to exclude from the process of creating the caching key.

    """
    os.makedirs(directory, exist_ok=True)

    def _print(s, verbose=verbose):
        if verbose:
            print("sha_cache: {}" .format(s))

    if isinstance(exclude_kwargs, str):
        exclude_kwargs = exclude_kwargs.split()

    def decorator(func):
        sig = inspect.signature(func)

        def new_f(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            params = dict(bound_args.arguments).copy()
            params_for_hash = params.copy()

            if exclude_kwargs is not None:
                for kwarg in exclude_kwargs:
                    del params_for_hash[kwarg]

            param_hash = get_param_hash(params_for_hash)
            filename = os.path.join(directory, "{}_{}.cache".format(func.__name__, param_hash))
            params_filename = os.path.join(directory, "{}_{}.params".format(func.__name__, param_hash))

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
                    value = func(**params)

                    _print("Saving results...")

                    with open(filename, 'wb') as f:
                        dill.dump(value, f, protocol=dill.HIGHEST_PROTOCOL, recurse=recurse)

                    with open(params_filename, 'w') as f:
                        f.write(pformat(params_for_hash))
            return value
        return new_f
    return decorator


def _run_cmd(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()


def find_git_directories():
    import __main__ as main
    # Try current directory, and the directory that contains the script that is running.
    dirs = [os.getcwd(), os.path.realpath(main.__file__)]

    all_packages = pip_freeze()
    all_packages = all_packages.split('\n')
    git_packages = [p.split('=')[-1] for p in all_packages if p.startswith('-e git+')]

    for p in git_packages:
        try:
            package = importlib.import_module(p)
            directory = os.path.dirname(package.__file__)

            dirs.append(directory)
        except Exception:
            pass

    version_controlled_dirs = set()

    for directory in list(set(dirs)):
        # Check whether any ancestor directory contains a .git directory
        while directory:
            git_dir = os.path.join(directory, '.git')
            if os.path.isdir(git_dir):
                version_controlled_dirs.add(directory)
                break
            directory = os.path.dirname(directory)

    return sorted(version_controlled_dirs)


def summarize_git_repo(directory, n_logs=10, diff=False, terminal=False, porcelain_status=True):
    if terminal:
        try:
            import colorama as crama
        except ImportError:
            crama = None
    else:
        crama = None

    s = []
    with cd(directory):
        s.append("*" * 80)

        if crama is None:
            s.append("git summary for directory {}".format(directory))
        else:
            s.append("git summary for directory {}{}{}".format(crama.Fore.BLUE, directory, crama.Style.RESET_ALL))

        def cmd_string(_cmd):
            if crama is None:
                return "\n{}:\n".format(cmd)
            else:
                return "\n{}{}{}:\n".format(crama.Fore.YELLOW, cmd, crama.Style.RESET_ALL)

        cmd = 'git log -n {} --decorate=short'.format(n_logs)
        s.append(cmd_string(cmd))
        log = _run_cmd(cmd).strip('\n')
        s.append(log)

        cmd = 'git status' + (' --porcelain' if porcelain_status else '')

        s.append(cmd_string(cmd))
        status = _run_cmd(cmd).strip('\n')
        s.append(status)

        if diff:
            cmd = 'git diff HEAD'
            s.append(cmd_string(cmd))
            diff = _run_cmd(cmd).strip('\n')
            s.append(diff)

        s.append('')

    return '\n'.join(s)


def summarize_git_repos(**summary_kwargs):
    s = []
    git_dirs = find_git_directories()
    for git_dir in git_dirs:
        try:
            git_summary = summarize_git_repo(git_dir, **summary_kwargs)
            s.append(git_summary)
        except Exception:
            pass
    return '\n'.join(s)


def git_summary_cl():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-logs', type=int, default=1)
    parser.add_argument('--no-diff', action='store_true')
    args = parser.parse_args()

    print(summarize_git_repos(n_logs=args.n_logs, diff=not args.no_diff, terminal=True, porcelain_status=False))


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
    def __init__(self, path, prefix='exp', max_experiments=None, delete_old=False, include_store_name=True):
        self.path = os.path.abspath(str(path))
        assert prefix, "prefix cannot be empty"
        self.prefix = prefix

        if include_store_name:
            store_name = os.path.basename(path)
            store_name = store_name.replace('_', '-')
            self.prefix = f'{self.prefix}_{store_name}'

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
            f'{self.prefix}_{name}', add_date=add_date, config_dict=config_dict)

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
        exp_dirs = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.startswith(self.prefix)]
        exp_dirs = [f for f in exp_dirs if os.path.isdir(f)]
        exp_dirs_with_mtime = [(os.path.getmtime(d), d) for d in exp_dirs]
        latest_exp_dirs = [d for _, d in sorted(exp_dirs_with_mtime)[-n:]]

        latest_dir = os.path.join(self.path, 'tensorboard_latest')

        try:
            shutil.rmtree(latest_dir)
        except FileNotFoundError:
            pass

        os.makedirs(latest_dir, exist_ok=False)

        for exp_dir in latest_exp_dirs:
            make_symlink(os.path.join('..', exp_dir), os.path.join(latest_dir, os.path.basename(exp_dir)))

        # Look for a directory called `tensorboard_pinned`, and create links to each file therein from the n_latest dir.
        tensorboard_pinned = os.path.join(self.path, 'tensorboard_pinned')
        if os.path.isdir(tensorboard_pinned):
            for d in os.listdir(tensorboard_pinned):
                full = os.path.join('../tensorboard_pinned', d)
                make_symlink(full, os.path.join(latest_dir, d))

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


def pretty_func(f):
    f_string = str(f)
    if not f_string.startswith('<function '):
        return f_string

    if hasattr(f, '__name__'):
        name = f.__name__
    elif hasattr(f, '__class__'):
        name = f.__class__.__name__
    else:
        name = ''

    lmbda = lambda: 0

    info = dict(name=name)

    if name == lmbda.__name__:
        try:
            info['source'] = '"{}"'.format(inspect.getsource(f).strip())
        except OSError:
            pass

    try:
        source_lines = inspect.getsourcelines(f)
        start = source_lines[1]
        end = start + len(source_lines[0])
        info['linenos'] = (start, end)
    except (OSError, TypeError):
        info['linenos'] = None
    try:
        info['file'] = inspect.getsourcefile(f)
    except TypeError:
        info['file'] = None
    # try:
    #     info['module'] = inspect.getmodule(f)
    # except TypeError:
    #     info['module'] = None

    s = "<function " + ", ".join("{}={}".format(k, v) for k, v in info.items()) + ">"

    return s


class SourceJSONEncoder(json.JSONEncoder):
    """ Convert functions to more informative representation. """

    def default(self, obj):
        if callable(obj):
            return pretty_func(obj)

        return repr(obj)


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

    def record_environment(self, config=None, dill_recurse=False, git_mode='all'):

        assert git_mode in ['all', 'fast', 'none']
        if git_mode != 'none':
            with open(self.path_for('context/git_summary.txt'), 'w') as f:
                do_git_diff = git_mode == 'all'
                git_summary = summarize_git_repos(diff=do_git_diff)
                f.write(git_summary)

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
                try:
                    dill.dump(config, f, protocol=dill.HIGHEST_PROTOCOL, recurse=dill_recurse)
                except RecursionError:
                    print("Exception when writing config: ")
                    traceback.print_exc()

            with open(self.path_for('config.json'), 'w') as f:
                json.dump(config.freeze(), f, cls=SourceJSONEncoder, indent=4, sort_keys=True)

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
        for c in ": _":
            date_time_string = date_time_string.replace(c, '-')
        labels.append(date_time_string)

    key_vals = list(config_dict.items())
    key_vals.sort(key=lambda x: x[0])

    for key, value in key_vals:
        if not isinstance(key, str):
            raise ValueError("keys in config_dict must be strings.")
        if not isinstance(value, str):
            raise ValueError("values in config_dict must be strings.")

        if not str(key) in omit:
            key = key.replace('_', '-')
            value = value.replace('_', '-')
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
    shifted[:n] = fill
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


def pformat(v):
    return pretty(v)


def _list_repr_pretty(lst, p, cycle):
    if cycle:
        p.text('[...]')
    else:
        with p.group(4, '['):
            p.breakable('')
            for idx, item in enumerate(lst):
                if idx:
                    p.text(',')
                    p.breakable()
                p.pretty(item)
        p.breakable('')
        p.text(']')


for_type(list, _list_repr_pretty)


def _ndarray_repr_pretty(a, p, cycle):
    if cycle:
        p.text(f'ndarray(..., shape={a.shape}, dtype={a.dtype})')
    else:
        with p.group(4, 'ndarray('):
            p.breakable('')
            x = str(a)
            for idx, line in enumerate(x.split('\n')):
                if idx:
                    p.breakable()
                p.text(line)
            p.text(',')
            p.breakable()
            p.text(f'shape={a.shape}, dtype={a.dtype}')
        p.breakable('')
        p.text(')')


for_type(np.ndarray, _ndarray_repr_pretty)


def _named_dict_repr_pretty(name, d, p, cycle):
    if cycle:
        p.text(f'{name}(...)')
    else:
        with p.group(4, f'{name}({{'):
            p.breakable('')
            for idx, (k, v) in enumerate(sorted(d.items())):
                if idx:
                    p.text(',')
                    p.breakable()

                p.pretty(k)
                p.text(': ')
                p.pretty(v)
        p.breakable('')
        p.text('})')


NotSupplied = object()


class Param:
    def __init__(self, default=NotSupplied, aliases=None, help="", type=None):
        """ aliases are different ways to fill the value (i.e. from config or kwargs),
            but should not be used to access the value as a class attribute. """
        self.default = default
        if isinstance(aliases, str):
            aliases = aliases.split()
        self.aliases = aliases or []
        self.help = help
        self.type = type


class Parameterized:
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

    def __getnewargs_ex__(self):
        """ Required for pickling to work properly, ensures that when __new__ is called, the parameters are all present.

        """
        return ((), self.param_values())

    def __init__(self, *args, **kwargs):
        pass

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return pformat(self)

    def _repr_pretty_(self, p, cycle):
        d = self.param_values()
        _named_dict_repr_pretty(self.__class__.__name__, d, p, cycle)

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

    def param_values(self, original=False):
        if not self._resolved:
            raise Exception("Parameters have not yet been resolved.")
        if original:
            return {k: self._params_at_creation_time[k] for k in self._params_at_creation_time}
        else:
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


class launch_pdb_on_exception:
    def __init__(self, context=11):
        self.context = context

    def __enter__(self):
        pass

    def __exit__(self, type_, value, tb):
        from bdb import BdbQuit
        if type_ and type_ is not BdbQuit:
            traceback.print_exc()

            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            ipdb_postmortem(tb, self.context)
            return True


def ipdb_postmortem(tb=None, context=10):
    from ipdb.__main__ import _init_pdb, wrap_sys_excepthook

    wrap_sys_excepthook()
    p = _init_pdb(context)
    p.reset()
    if tb is None:
        # sys.exc_info() returns (type, value, traceback) if an exception is
        # being handled, otherwise it returns None
        tb = sys.exc_info()[2]
    if tb:
        p.interaction(None, tb)


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


_timer_stats = {}


@contextmanager
def timed_block(name=None, flag=True, record_stats=False, reset_stats=False):
    if name is None:
        frame = inspect.stack()[1]
        name = "{}:{}".format(frame.filename, frame.lineno)
    start_time = time.time()
    yield
    duration = time.time() - start_time

    if record_stats:
        if name not in _timer_stats or reset_stats:
            _timer_stats[name] = RunningStats()

        _timer_stats[name].add(duration)

    if flag:
        print(f"Call to block <{name}> took {duration} seconds.")

        if record_stats:
            mean, var, mn, mx = _timer_stats[name].get_stats()
            std = np.sqrt(var)
            print(f"Duration stats - mean={mean}, std={std}, mn={mn}, mx={mx}")


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


def _nested_update(d, other):
    if not isinstance(d, dict) or not isinstance(other, dict):
        return

    for k, v in other.items():
        if k in d and isinstance(d[k], dict) and isinstance(v, dict):
            _nested_update(d[k], v)
        else:
            d[k] = v


def _flatten_nested_dict(d, sep):
    """ returns a list of pairs of the form (flattened-key, value) """
    result = []
    for k, v in d.items():
        if not isinstance(k, str):
            result.append((k, v))

        elif isinstance(v, dict) and all([isinstance(_k, str) for _k in v.keys()]):
            v = _flatten_nested_dict(v, sep)
            for _k, _v in v:
                key = k + sep + _k
                result.append((key, _v))

        else:
            result.append((k, v))

    return result


def _check_for_prefixes(pairs, sep):
    """ Takes a list of pairs of the form (flattened-key, value).
        Check whether any of the flattened keys are prefixes of one another.

    """
    root = {}

    for k, v in pairs:
        if not isinstance(k, str):
            if k in root:
                raise Exception("Non-string key ({}) found multiple times.".format(k))
            root[k] = v
            continue

        subkeys = k.split(':')

        d = root
        for i, sk in enumerate(subkeys[:-1]):
            if sk in d:
                if d[sk] is None:
                    other = sep.join(subkeys[:i+1])
                    raise Exception("One key ({}) is a prefix of another ({}).".format(other, k))
                else:
                    d = d[sk]
            else:
                d[sk] = {}
                d = d[sk]

        sk = subkeys[-1]
        if sk in d:
            raise Exception("One key {} is a prefix of another.".format(k))

        d[sk] = None


class Config(dict):
    """ A hierarchical dict-like, tree-like data structure.

        Attribute access does the same as __getitem__ and __setitem__.

        Non-string keys are treated as normal, as are strings that do not contain the special character stored
        in `self._sep` (default ':'). If a string key does contain the sep character, then it is treated as a
        hierarchical key.  When *setting* with a hierarchical key, the key is split by the sep character, and
        the resulting keys are treated as a sequence of keys into nested Configs, creating necessary configs as we go.
        When *getting* with hierarchical keys, a similar procedure is followed but the intermediate dicts
        are not created.

        Whenever an added value is itself a hierarchical dicts with strings as keys, it is first flattened,
        and the results are stitched into the tree. For example (assuming ':' as the sep):

        x = Config()
        x['a'] = {'b:c': 0, {'d': {'e': 3}}}
        print(x)
        >>  Config{
                "a": {
                    "b": {
                        "c": 0
                    },
                    "d": {
                        "e": 3
                    }
                }
            }

        The `update` function works the same way.

        When updating or setting a value that is a hierarchical dict, it may be possible for the hierarchical dict
        to refer to the same tree location in different ways. For example {'a:b': 0, 'a': {'b': 1}}. The tree location
        'a:b' is assigned two different values, 0 and 1. Such situations will be detected and an error thrown.
        Of course, overwriting locations in separate calls is allowed, just not in the same call.

        So this is OK:
        x = Config()
        x.update({'a:b': 0})
        x.update({'a': {'b': 1}})

        but not this:
        x = Config()
        x.update({'a:b': 0, 'a': {'b': 1}})

        Recursion is stopped once we reached a non-dict value, or a dict that has at least one non-string key.
        TODO: example.

        One pitfall to be aware of: overwriting of nested dicts basically doesn't happen.

        x = Config(a=dict(b=1))

        >> Config{
                "a": {
                    "b": 1
                }
            }
        x['a'] = dict(c=2)
        print(x)
        >>  Config{
                "a": {
                    "b": 1,
                    "c": 2
                }
            }

        The two dicts assigned to x['a'] are combined, rather than one overwriting the other.

        However, there is one case where overwriting happens:

        x = Config({'a:b': 1})
        x["a:b:c"] = 2
        print(x)
        >>  Config{
                "a": {
                    "b": {
                        "c": 2
                    }
                }
            }

        The value 1, stored at "a:b" was deleted.

    """
    _sep = ":"
    _reserved_keys = None

    def __init__(self, _d=None, **kwargs):
        self.update(_d, **kwargs)

    def flatten(self):
        return dict(_flatten_nested_dict(self, self._sep))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return pformat(self)

    def _repr_pretty_(self, p, cycle):
        name = self.__class__.__name__
        _named_dict_repr_pretty(name, self, p, cycle)

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

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    def _set_item_dict(self, d):
        """ This will never be involved in recursive calls, because _flatten_nested_dict should remove
            all dictionary values from d.

            alg is: create a list of flat keys and values
            add each pairs to the dictionary after checking for prefixes.

        """
        flat = _flatten_nested_dict(d, self._sep)
        _check_for_prefixes(flat, self._sep)

        for k, v in flat:
            self[k] = v

    def __setitem__(self, key, value):
        """ Note that this is ``destructive''.

            For example:

            x = Config({'a:b': 1})
            x["a:b:c"] = 2
            print(x)
            {'a': {'b': {'c': 2}}}

            The value stored at 'a:b' was overwritten.

        """
        if not isinstance(key, str):
            return super().__setitem__(key, value)

        if isinstance(value, dict) and all([isinstance(s, str) for s in value.keys()]):
            self._set_item_dict({key: value})

        elif isinstance(key, str) and self._sep in key:
            key, _, rest = key.partition(self._sep)
            try:
                d = self[key]

                if not isinstance(d, self.__class__._dict_class):
                    # This is where the destruction happens
                    d = self.__class__._dict_class()
                    super().__setitem__(key, d)

            except KeyError:
                d = self.__class__._dict_class()
                super().__setitem__(key, d)

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
        if key in self.__class__._reserved_keys:
            return super().__getattr__(key)

        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("Could not find attribute called `{}`.".format(key))

    def __setattr__(self, key, value):
        if key in self.__class__._reserved_keys:
            super().__setattr__(key, value)
            return

        self[key] = value

    def _validate_key(self, key):
        msg = "Bad key for config: `{}`.".format(key)
        assert not key.startswith('_'), msg
        assert key not in self.__class__._reserved_keys, msg
        assert self._sep not in key, msg

    def deepcopy(self, _d=None, **kwargs):
        """ Deep copy and update at the same time. """
        new = copy.deepcopy(self)
        new.update(_d, **kwargs)
        return new

    def copy(self, _d=None, **kwargs):
        """ Copy and update at the same time.

        A pseudo-deep copy; copies everything except the leaves (the leaves being the stored values).

        """
        new = self.__class__._dict_class(self.flatten())
        new.update(_d, **kwargs)
        return new

    def update(self, _=None, **kwargs):
        if _ is not None:
            self._set_item_dict(dict(_))

        self._set_item_dict(kwargs)

    def update_from_command_line(self, strict=True):
        flattened = self.flatten()
        cl_object = clify.wrap_object(flattened, strict=strict)
        cl_args = cl_object.parse()
        self.update(cl_args)

    def __enter__(self):
        cfg._push_stack(self)

    def __exit__(self, exc_type, exc, exc_tb):
        cfg._pop_stack()
        return False

    def freeze(self, remove_callable=False):
        _config = Config()
        for key in self.flatten().keys():
            value = self[key]
            if remove_callable and callable(value):
                value = str(value)
            _config[key] = value
        return _config

    def map(self, func, *others, is_leaf=None):
        if is_leaf is None:
            is_leaf = lambda l: not isinstance(l, dict)
        return map_structure(func, self, *others, is_leaf=is_leaf)


Config._dict_class = Config
Config._reserved_keys = dir(Config)


class AttrDict(Config):
    pass


AttrDict._dict_class = AttrDict


DEFAULT_CONFIG = None


def get_default_config():
    """ Look for a file called `dps_config.py` in the following locations:
            1. dps.__path__[0]
            2. $HOME/.config
            3. Current working directory.

        All files that are found are loaded. Configs later in the list are loaded after, and on top of, earlier ones
        (so for config values that are specified by multiple files, the values from "later" files replace earlier ones).

        Each file that is found is loaded as a module, and should define a variable called `config` containing a
        dict storing the config values.

        This function returns a copy of that loaded config (a *copy*, so it cannot be modified in a useful way).

    """
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:

        config = Config()

        config_dirs = [
            dps.__path__[0],
            os.path.join(os.getenv('HOME'), '.config'),
            '.'
        ]
        for d in config_dirs:
            config_loc = os.path.join(d, 'dps_config.py')

            if not os.path.exists(config_loc):
                continue

            # print(f"Loading default config values from path {os.path.realpath(config_loc)}.")
            config_module_spec = importlib.util.spec_from_file_location("dps_config", config_loc)
            config_module = config_module_spec.loader.load_module()
            config.update(config_module.config)

        DEFAULT_CONFIG = config

    return DEFAULT_CONFIG.copy()


class ConfigStack(Config):
    """
    Changes made to the ConfigStack only last for the lifetime of the context in which those changes are made.
    Such changes are also propagated to deeper contexts.

    So we store all previous states of the Config, so that we can jump back to them when we pop out of the context.
    The raw_configs are like the deltas, and the configs themselves are the states.

    """
    def __init__(self):
        self._raw_config_stack = []
        self._config_stack = []
        self._push_stack(get_default_config())

        super().__init__()

    def _push_stack(self, raw_config):
        raw_config = raw_config.copy()
        self._raw_config_stack.append(raw_config)

        # Before updating, save config as it currently is.
        _copy = self.__class__._dict_class(self.flatten())
        self._config_stack.append(_copy)

        self.update(raw_config)

    def _pop_stack(self):
        assert self._config_stack
        prev_config = self._config_stack.pop()

        self.clear()
        self.update(prev_config)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, ConfigFn):
            value = value(self)
        return value

    def deepcopy(self, _d=None, **kwargs):
        raise Exception("NotImplemented")

    def copy(self, _d=None, **kwargs):
        raise Exception("NotImplemented")


ConfigStack._reserved_keys = sorted(set(
    Config._reserved_keys + dir(ConfigStack) + ['_raw_config_stack', '_config_stack']
))


cfg = ConfigStack()


class ConfigFn:
    """ If a ConfigStack is queried with a key, and the resulting value is an instance
        if this class, then the instance will be called and the result will be returned
        as the value.

    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __str__(self):
        fn_s = pretty_func(self.fn)
        fn_s = '\n'.join(['    ' + l for l in fn_s.split('\n') if l])

        return "ConfigFn(fn=\n    '''\n{}\n    '''\n)".format(fn_s)

    def __repr__(self):
        return str(self)


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
    """ Apply a function `func` to every element of a hierarchical data structure composed of
        dicts, lists and tuples. Multiple data structures can also be passed in, in which case they must all have the
        same structure (same keys for dicts, same lengths for lists and tuples). When multiple structures are passed in,
        then objects at the same "location" in the different data structures will be passed into `func` together; hence
        `func` must take number of arguments equal to len(args). Function `is_leaf` determines, whether a given input
        counts as a leaf element, in which case recursive decent halts and `func` is applied to that element.
        leaf-ness is checked only by looking at the values in the first data structure.

        Returned value has the same structure as the passed-in data structures.

    """
    if is_leaf(args[0]):
        return func(*args)
    else:
        if isinstance(args[0], dict):
            arg_keys = [a.keys() for a in args]

            left = set(arg_keys[0])
            for right in arg_keys[1:]:
                right = set(right)
                if left != right:
                    missing_from_left = right - left
                    missing_from_right = left - right

                    raise Exception(
                        f"Structures do not have same format.\nmissing_from_left: {missing_from_left}. "
                        f"\nmissing_from_right: {missing_from_right}."
                    )

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


def execute_command(command, shell=True, max_seconds=None, robust=False, output=None):
    """ Uses `subprocess` to execute `command`. Has a few added bells and whistles.

    if command returns non-zero exit status:
        if robust:
            returns as normal
        else:
            raise CalledProcessError

    Parameters
    ----------
    command: str
        The command to execute.


    Returns
    -------
    returncode, stdout, stderr

    """
    p = None
    try:
        assert isinstance(command, str)

        if output == "loud":
            print("\nExecuting command: " + (">" * 40) + "\n")
            print(command)

        if not shell:
            command = command.split()

        stdout = None if output == "loud" else subprocess.PIPE
        stderr = None if output == "loud" else subprocess.PIPE

        start = time.time()

        sys.stdout.flush()
        sys.stderr.flush()

        p = subprocess.Popen(command, shell=shell, universal_newlines=True,
                             stdout=stdout, stderr=stderr)

        interval_length = 1
        while True:
            try:
                p.wait(interval_length)
            except subprocess.TimeoutExpired:
                pass

            if p.returncode is not None:
                break

        if output == "loud":
            print("\nCommand took {} seconds.\n".format(time.time() - start))

        _stdout = "" if p.stdout is None else p.stdout.read()
        _stderr = "" if p.stderr is None else p.stderr.read()

        if p.returncode != 0:
            if isinstance(command, list):
                command = ' '.join(command)

            print("The following command returned with non-zero exit code "
                  "{}:\n    {}".format(p.returncode, command))

            if output is None or (output == "quiet" and not robust):
                print("\n" + "-" * 20 + " stdout " + "-" * 20 + "\n")
                print(_stdout)

                print("\n" + "-" * 20 + " stderr " + "-" * 20 + "\n")
                print(_stderr)

            if robust:
                return p.returncode, _stdout, _stderr
            else:
                raise subprocess.CalledProcessError(p.returncode, command, _stdout, _stderr)

        return p.returncode, _stdout, _stderr

    except BaseException as e:
        if p is not None:
            p.terminate()
            p.kill()
        raise e


def ssh_execute(command, host, ssh_options=None, **kwargs):
    ssh_options = ssh_options or {}
    if host == ":":
        cmd = command
    else:
        cmd = "ssh {ssh_options} -T {host} \"{command}\"".format(
            ssh_options=ssh_options, host=host, command=command)
    return execute_command(cmd, **kwargs)


class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

        self.min = None
        self.max = None

    def add(self, data):
        """ First dimension is batch dimension, all the rest are different dimensions of the input. """
        data = np.atleast_2d(data)

        for d in data:
            self.update(d)

    def update(self, data):
        if self.count == 0:
            self.mean = np.zeros_like(data)
            self.m2 = np.zeros_like(data)
            self.min = np.inf * np.ones_like(data)
            self.max = -np.inf * np.ones_like(data)

        self.count += 1
        delta = data - self.mean
        self.mean += delta / self.count
        delta2 = data - self.mean
        self.m2 += delta * delta2

        self.min = np.minimum(data, self.min)
        self.max = np.maximum(data, self.max)

    def get_stats(self):
        var = self.m2 / self.count
        return self.mean, var, self.min, self.max


if __name__ == "__main__":
    def _test_liang_barsky(*args, ref_answer):
        answer = liang_barsky(*args)
        print("{}: {}".format(args, answer))

        if ref_answer is not None:
            assert answer == ref_answer
        else:
            assert answer is None

    rect = [[1, 1], [2, 2]]

    _test_liang_barsky(rect, [[1.5, 0.5], [1.5, 2.5]], ref_answer=(1/4, 3/4))
    _test_liang_barsky(rect, [[1.5, 0.5], [1.5, .99]], ref_answer=None)
    _test_liang_barsky(rect, [[1.5, 0.5], [1.5, 1]], ref_answer=None)
    _test_liang_barsky(rect, [[1.5, 0.5], [1.5, 1.01]], ref_answer=(0.5 / 0.51, 1))
    _test_liang_barsky(rect, [[1.5, 0.5], [-1.5, -2.5]], ref_answer=None)
    _test_liang_barsky(rect, [[2.5, 0.5], [2.5, 2.5]], ref_answer=None)
    _test_liang_barsky(rect, [[0.5, 2.5], [2.5, 2.5]], ref_answer=None)
    _test_liang_barsky(rect, [[0, 0], [2, 2]], ref_answer=(0.5, 1))
    _test_liang_barsky(rect, [[0, .99], [2, 2.99]], ref_answer=(0.5, 0.505))
    _test_liang_barsky(rect, [[1.5, 1.5], [3, 3]], ref_answer=(0, 1/3))

    rect = [[1, 2, 3], [2, 3, 4]]
    line = [[1, 2, 2], [2, 3, 5]]
    _test_liang_barsky(rect, line, ref_answer=(1/3, 2/3))
