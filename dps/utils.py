import subprocess as sp
from pprint import pformat
from contextlib import contextmanager
import numpy as np
from pathlib import Path
import signal
import time
import configparser
import socket
import re
import os
import traceback
import pdb
from collections import deque
from collections.abc import MutableMapping
import subprocess
import copy
import datetime
import psutil
from itertools import cycle, islice
import resource
from datetime import timedelta

import clify

import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.slim import fully_connected

import dps


def make_directory_name(experiments_dir, network_name, add_date=True):
    if add_date:
        working_dir = os.path.join(experiments_dir, network_name + "_")
        dts = str(datetime.datetime.now()).split('.')[0]
        for c in [":", " ", "-"]:
            dts = dts.replace(c, "_")
        working_dir += dts
    else:
        working_dir = os.path.join(experiments_dir, network_name)

    return working_dir


def _parse_timedelta(s):
    """ ``s`` should be of the form HH:MM:SS """
    args = [int(i) for i in s.split(":")]
    return timedelta(hours=args[0], minutes=args[1], seconds=args[2])


def parse_timedelta(d, fmt='%a %b  %d %H:%M:%S %Z %Y'):
    """ ``s`` should be of the form HH:MM:SS """
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
    array = array / array.max()
    image = [char_map(value) for value in array.flatten()]
    image = np.reshape(image, array.shape)
    return '\n'.join(''.join(c for c in row) for row in image)


def shift_fill(a, n, axis=0, fill=0.0, reverse=False):
    """ shift n spaces backward along axis, filling rest in with 0's. if n is negative, shifts forward. """
    shifted = np.roll(a, n, axis=axis)
    shifted[:n, ...] = 0.0
    return shifted


def tf_roll(a, n, axis=0, fill=None, reverse=False):
    if reverse:
        a = tf.reverse(a, axis=[axis])
    assert n > 0

    pre_slices = [slice(None) for i in a.shape]
    pre_slices[axis] = slice(None, -n)

    pre = a[pre_slices]

    post_slices = [slice(None) for i in a.shape]
    post_slices[axis] = slice(-n, None)

    post = a[post_slices]

    if fill is not None:
        post = fill * tf.ones_like(post, dtype=a.dtype)

    r = tf.concat([post, pre], axis=axis)

    if reverse:
        r = tf.reverse(r, axis=[axis])
    return r


def tf_discount_matrix(base, T, n=None):
    x = tf.cast(tf.range(T), tf.float32)
    r = (x - x[:, None])
    if n is not None:
        r = tf.where(r >= n, np.inf * tf.ones_like(r), r)
    r = base ** r
    return tf.matrix_band_part(r, 0, -1)


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


def masked_mean(array, mask, axis=None, keep_dims=True):
    if axis is None:
        return tf.reduce_mean(tf.boolean_mask(array, tf.cast(mask, tf.bool)))
    else:
        denom = tf.reduce_sum(mask, axis=axis, keep_dims=keep_dims)
        denom = tf.where(tf.abs(denom) < 1e-6, np.inf * tf.ones_like(denom), denom)
        return tf.reduce_sum(array, axis=axis, keep_dims=keep_dims) / denom


def build_gradient_train_op(
        loss, tvars, optimizer_spec, lr_schedule,
        max_grad_norm=None, noise_schedule=None, global_step=None):
    """ By default, `global_step` is None, so the global step is not incremented. """

    pure_gradients = tf.gradients(loss, tvars)

    clipped_gradients = pure_gradients
    if max_grad_norm is not None and max_grad_norm > 0.0:
        clipped_gradients, _ = tf.clip_by_global_norm(pure_gradients, max_grad_norm)

    noisy_gradients = clipped_gradients
    if noise_schedule is not None:
        grads_and_vars = zip(clipped_gradients, tvars)
        noise = build_scheduled_value(noise_schedule, 'gradient_noise')
        noisy_gradients = add_scaled_noise_to_gradients(grads_and_vars, noise)

    grads_and_vars = list(zip(noisy_gradients, tvars))

    lr = build_scheduled_value(lr_schedule, 'learning_rate')
    optimizer = build_optimizer(optimizer_spec, lr)

    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    summaries = [
        tf.summary.scalar('grad_norm_pure', tf.global_norm(pure_gradients)),
        tf.summary.scalar('grad_norm_processed', tf.global_norm(noisy_gradients)),
    ]

    return train_op, summaries


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
            for p in self._params:
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

    @property
    def _params(self):
        params = []
        for p in dir(self):
            try:
                if p != 'params' and isinstance(getattr(self, p), Param):
                    params.append(p)
            except:
                pass
        return params


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
    return os.path.expandvars(os.path.expanduser(path))


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


def uninitialized_variables_initializer():
    """ init only uninitialized variables - from
        http://stackoverflow.com/questions/35164529/
        in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables """
    uninitialized_vars = []
    sess = tf.get_default_session()
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    uninit_init = tf.variables_initializer(uninitialized_vars)
    return uninit_init


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


def load_or_train(sess, build_model, train, var_scope, path=None, train_config=None):
    """ Attempts to load variables into ``var_scope`` from checkpoint stored at ``path``.

    If said variables are not found, trains a model using the function
    ``train`` and stores the resulting variables for future use.

    Returns True iff model was successfully loaded, False otherwise.

    """
    to_be_loaded = trainable_variables(var_scope.name)
    saver = tf.train.Saver(var_list=to_be_loaded)

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    success = False
    try:
        print("Trying to load variables for variable scope {} from checkpoint {}...".format(var_scope.name, path))
        saver.restore(sess, path)
        success = True
        print("Load successful.")
    except tf.errors.NotFoundError:
        print("Loading failed, training a model...")
        with train_config:
            train(build_model, var_scope, path)
        saver.restore(sess, path)
        print("Training successful.")
    return success


class MLP(object):
    def __init__(self, n_units=None, **fc_kwargs):
        self.n_units = n_units or []
        self.fc_kwargs = fc_kwargs

    def __call__(self, inp, output_size):
        if len(inp.shape) > 2:
            trailing_dim = np.product([int(s) for s in inp.shape[1:]])
            inp = tf.reshape(inp, (tf.shape(inp)[0], trailing_dim))
        hidden = inp
        for i, nu in enumerate(self.n_units):
            hidden = fully_connected(hidden, nu, **self.fc_kwargs)
        fc_kwargs = self.fc_kwargs.copy()
        fc_kwargs['activation_fn'] = None
        return fully_connected(hidden, output_size, **fc_kwargs)


class ScopedCell(RNNCell):
    """ An RNNCell that creates its own variable scope the first time `resolve_scope` is called.
        The scope is then carried around and used for any subsequent build operations, ensuring
        that all uses of an instance of this class use the same set of variables.

    """
    def __init__(self, name):
        self.scope = None
        self.name = name or self.__class__.__name__

    def resolve_scope(self):
        reuse = self.scope is not None
        if not reuse:
            with tf.variable_scope(self.name):
                self.scope = tf.get_variable_scope()
        return self.scope, reuse

    def _call(self, inp, state):
        raise Exception("NotImplemented")

    def __call__(self, inp, state):
        scope, reuse = self.resolve_scope()
        with tf.variable_scope(scope, reuse=reuse):
            return self._call(inp, state)


class ScopedCellWrapper(ScopedCell):
    """ Similar to ScopedCell, but used in cases where the cell we want to scope does not inherit from ScopedCell. """
    def __init__(self, cell, name):
        self.cell = cell
        super(ScopedCellWrapper, self).__init__(name)

    def __call__(self, inp, state):
        scope, reuse = self.resolve_scope()
        with tf.variable_scope(scope, reuse=reuse):
            return self.cell(inp, state)

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)


class FixedController(ScopedCell):
    """ A controller that outputs a fixed sequence of actions.

    Parameters
    ----------
    action_sequence: ndarray (n_timesteps, actions_dim)
        t-th row gives the action this controller will select at time t.

    """
    def __init__(self, action_sequence, name="fixed_controller"):
        self.action_sequence = np.array(action_sequence)
        super(FixedController, self).__init__(name)

    def _call(self, inp, state):
        action_seq = tf.constant(self.action_sequence, tf.float32)
        int_state = tf.squeeze(tf.cast(state, tf.int32), axis=1)
        actions = tf.gather(action_seq, int_state)

        return actions, state + 1

    def __len__(self):
        return len(self.action_sequence)

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.action_sequence.shape[1]

    def zero_state(self, batch_size, dtype):
        return tf.cast(tf.fill((batch_size, 1), 0), dtype)


class FixedDiscreteController(ScopedCell):
    """ A controller that outputs a fixed sequence of actions.

    Parameters
    ----------
    action_sequence: list of int
        t-th entry gives the idx of the action this controller will select at time t.
    actions_dim: int
        Number of actions.

    """
    def __init__(self, action_sequence, actions_dim, name="fixed_discrete_controller"):
        self.action_sequence = np.array(action_sequence)
        self.actions_dim = actions_dim
        super(FixedDiscreteController, self).__init__(name)

    def _call(self, inp, state):
        action_seq = tf.constant(self.action_sequence, tf.int32)
        int_state = tf.cast(state, tf.int32)
        action_idx = tf.gather(action_seq, int_state)
        actions = tf.one_hot(tf.reshape(action_idx, (-1,)), self.actions_dim)
        return actions, state + 1

    def __len__(self):
        return len(self.action_sequence)

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.actions_dim

    def zero_state(self, batch_size, dtype):
        return tf.cast(tf.fill((batch_size, 1), 0), dtype)


class NullCell(ScopedCell):
    """ A cell with no meaningful output. """
    def __init__(self, output_size=0, name="null_cell"):
        self._output_size = output_size
        super(NullCell, self).__init__(name)

    def _call(self, inp, state):
        batch_size = tf.shape(inp)[0]
        return tf.zeros((batch_size, self.output_size)), tf.zeros((batch_size, 1))

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, self.output_size), dtype=dtype)


class CompositeCell(ScopedCell):
    """ A wrapper around a cell that adds an additional transformation of the output.

    Parameters
    ----------
    cell: instance of RNNCell
        The cell to wrap.
    output: callable (Tensor, int) -> Tensor
        Maps from an input tensor and an output size to an output tensor.
    output_size: int
        The size of the output, passed as the second argument when calling ``output``.

    """
    def __init__(self, cell, output, output_size, name="composite_cell"):
        self.cell = cell
        self.output = output
        self._output_size = output_size
        super(CompositeCell, self).__init__(name)

    def _call(self, inp, state):
        output, new_state = self.cell(inp, state)
        return self.output(output, self._output_size), new_state

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)


class FeedforwardCell(ScopedCell):
    """ A wrapper around a feedforward network that turns it into an RNNCell with a dummy state.

    Parameters
    ----------
    ff: callable (Tensor, int) -> Tensor
        A function that generates the tensorflow ops implementing the
        feedforward network we want to wrap. Maps from an input tensor
        and an output size to an output tensor.
    output_size: int
        The size of the output, passed as the second argument when calling ``output``.

    """
    def __init__(self, ff, output_size, name="feedforward_cell"):
        self.ff = ff
        self._output_size = output_size

        super(FeedforwardCell, self).__init__(name)

    def _call(self, inp, state):
        output = self.ff(inp, self._output_size)
        return output, tf.zeros((tf.shape(inp)[0], 1))

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, 1))


def gen_seed():
    return np.random.randint(np.iinfo(np.int32).max)


def print_variables(collection, scope):
    g = tf.get_default_graph()
    variables = g.get_collection(collection, scope=scope)
    sess = tf.get_default_session()
    for v in variables:
        print("\n")
        print(v.name)
        print(sess.run(v))


def restart_tensorboard(logdir, port=6006, reload_interval=120):
    print("Killing old tensorboard process...")
    try:
        command = "fuser {}/tcp -k".format(port)
        sp.run(command.split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        print("Killing tensorboard failed:")
        print(e.output)
    print("Restarting tensorboard process...")
    command = "tensorboard --logdir={} --port={} --reload_interval={}".format(logdir, port, reload_interval)
    sp.Popen(command.split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    print("Done restarting tensorboard.")


def add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
    """Taken from tensorflow.

    Adds scaled noise from a 0-mean normal distribution to gradients.

    """
    gradients, variables = zip(*grads_and_vars)
    noisy_gradients = []
    for gradient in gradients:
        if gradient is None:
            noisy_gradients.append(None)
            continue
        if isinstance(gradient, ops.IndexedSlices):
            gradient_shape = gradient.dense_shape
        else:
            gradient_shape = gradient.get_shape()
        noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
        noisy_gradients.append(gradient + noise)
    return noisy_gradients


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


def scheduled_value_summaries():
    return tf.get_collection('scheduled_value_summaries')


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


def build_scheduled_value(schedule, name=None, global_step=None, dtype=None):
    """
    Parameters
    ----------
    schedule: str
        String of the form "kind arg1 arg2 ...". One exception is that
        constants can be specified by simply supplying the constant value,
        with no kind string.
    name: str
        Name to use for the output op. Also creates a summary that has this name.
    dtype: object convertible to tf.DType
        Will cast output value to this dtype.

    """
    op_name = name + "_schedule" if name else None
    try:
        schedule = "Constant({})".format(float(schedule))
    except (TypeError, ValueError):
        pass

    if isinstance(schedule, str):
        schedule = eval(schedule)
    assert isinstance(schedule, Schedule), "{} is not a schedule instance.".format(schedule)

    signal = schedule.build(np.arange(dps.cfg.max_steps+1).astype('f'))
    global_step = tf.contrib.framework.get_or_create_global_step() if global_step is None else global_step

    scheduled_value = tf.cast(tf.gather(signal, global_step), tf.float32)

    if dtype is not None:
        dtype = tf.as_dtype(np.dtype(dtype))
        scheduled_value = tf.cast(scheduled_value, dtype, name=op_name+"_cast")

    if name is not None:
        tf.summary.scalar(name, scheduled_value, collections=['scheduled_value_summaries'])

    return scheduled_value


def build_optimizer(spec, learning_rate):
    """

    Parameters
    ----------
    spec: str
        String of the form "kind arg1 arg2 ...".
    learning_rate: float
        First argument to the constructed optimizer.

    """
    assert isinstance(spec, str)
    kind, *args = spec.split()
    kind = kind.lower()
    args = deque(args)

    if kind == "adam":
        beta1 = float(popleft(args, 0.9))
        beta2 = float(popleft(args, 0.999))
        epsilon = float(popleft(args, 1e-08))
        use_locking = _bool(popleft(args, False))
        opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, use_locking=use_locking)
    elif kind == "rmsprop":
        decay = float(popleft(args, 0.95))
        momentum = float(popleft(args, 0.95))
        epsilon = float(popleft(args, 1e-8))
        use_locking = _bool(popleft(args, False))
        centered = _bool(popleft(args, False))
        opt = tf.train.RMSPropOptimizer(
            learning_rate, decay=decay, momentum=momentum,
            epsilon=epsilon, use_locking=use_locking, centered=centered)
    else:
        raise Exception(
            "No known optimizer with kind `{}` and args `{}`.".format(kind, args))

    return opt


def trainable_variables(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


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


Config._reserved_keys = dir(Config)


class DpsConfig(Config):
    def __init__(self, _d=None, **kwargs):
        config = _parse_dps_config_from_file()
        if _d:
            config.update(_d)
        config.update(kwargs)
        super(DpsConfig, self).__init__(**config)


def _parse_dps_config_from_file(key=None):
    _config = configparser.ConfigParser()
    location = Path(dps.__file__).parent
    _config.read(str(location / 'config.ini'))

    if not key:
        key = socket.gethostname().split('-')[0]

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
        log_root=process_path(_config.get(key, 'log_root')),
        display=_config.getboolean(key, 'display'),
        save_display=_config.getboolean(key, 'save_display'),
        mpl_backend=_config.get(key, 'mpl_backend'),
        use_gpu=_config.getboolean(key, 'use_gpu'),
        visualize=_config.getboolean(key, 'visualize'),
        tbport=_config.getint(key, 'tbport'),
        verbose=_config.getboolean(key, 'verbose'),
    )

    config.max_experiments = _config.getint(key, 'max_experiments')
    if config.max_experiments <= 0:
        config.max_experiments = np.inf
    return config


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigStack(dict, metaclass=Singleton):
    _stack = []

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
        for config in ConfigStack._stack[::-1]:
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
        for config in reversed(ConfigStack._stack):
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

    def freeze(self):
        cfg = Config()
        for key in self.keys():
            cfg[key] = self[key]
        return cfg

    @property
    def log_dir(self):
        return str(Path(self.log_root) / self.log_name)

    def update_from_command_line(self):
        cl_args = clify.wrap_object(self).parse()
        self.update(cl_args)


def lst_to_vec(lst):
    if isinstance(lst[0], np.ndarray):
        return np.concatenate([np.reshape(v, (-1,)) for v in lst], axis=0)
    elif isinstance(lst[0], tf.Tensor) or isinstance(lst[0], tf.Variable):
        return tf.concat([tf.reshape(v, (-1,)) for v in lst], axis=0)
    else:
        raise Exception()


def vec_to_lst(vec, reference):
    if isinstance(vec, np.ndarray):
        splits = np.split(vec, [r.size for r in reference])
        return [np.reshape(v, r.shape) for v, r in zip(splits, reference)]
    elif isinstance(vec, tf.Tensor) or isinstance(vec, tf.Variable):
        splits = tf.split(vec, [tf.size(r) for r in reference])
        return [tf.reshape(v, tf.shape(r)) for v, r in zip(splits, reference)]
    else:
        raise Exception()
