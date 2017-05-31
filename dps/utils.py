import subprocess as sp
from pprint import pformat
from contextlib import contextmanager
import numpy as np
import inspect
import types
from pathlib import Path
import signal
import time
import configparser
import socket
import re
import os
import traceback
import sys
import pdb
from collections import deque

import tensorflow as tf
from tensorflow.python.ops import random_ops, math_ops
from tensorflow.python.framework import ops, constant_op
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.slim import fully_connected

import dps


@contextmanager
def pdb_postmortem():
    try:
        yield
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


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
    raise Alarm()


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
            raise_alarm()
        if not np.isinf(self.seconds):
            signal.alarm(int(np.floor(self.seconds)))
        self.then = time.time()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if exc_type is Alarm:
            self.ran_out = True
            if self.verbose:
                print("Block ran for {} seconds (limit was {}).".format(time.time() - self.then, self.seconds))
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


def load_or_train(sess, build_model, train, var_scope, path=None, config=None):
    """ Attempts to load variables into ``var_scope`` from checkpoint stored at ``path``.

    If said variables are not found, trains a model using the function
    ``train`` and stores the resulting variables for future use.

    Returns True iff model was successfully loaded, False otherwise.

    """
    to_be_loaded = tf.get_collection('trainable_variables', scope=var_scope.name)
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
        train(build_model, var_scope, path, config)
        saver.restore(sess, path)
        print("Training successful.")
    return success


class MLP(object):
    def __init__(self, n_units=None, **fc_kwargs):
        self.n_units = n_units or []
        self.fc_kwargs = fc_kwargs

    def __call__(self, inp, output_size):
        hidden = inp
        for i, nu in enumerate(self.n_units):
            hidden = fully_connected(hidden, nu, **self.fc_kwargs)
        fc_kwargs = self.fc_kwargs.copy()
        fc_kwargs['activation_fn'] = None
        return fully_connected(hidden, output_size, **fc_kwargs)


class FixedController(RNNCell):
    """ A controller that outputs a fixed sequence of actions.

    Parameters
    ----------
    action_sequence: list of int
        t-th entry gives the idx of the action this controller will select at time t.
    n_actions: int
        Number of actions.

    """
    def __init__(self, action_sequence, n_actions):
        self.action_sequence = np.array(action_sequence)
        self.n_actions = n_actions

    def __call__(self, inp, state):
        action_seq = tf.constant(self.action_sequence, tf.int32)
        int_state = tf.cast(state, tf.int32)
        action_idx = tf.gather(action_seq, int_state)
        actions = tf.one_hot(tf.reshape(action_idx, (-1,)), self.n_actions)
        return actions, state + 1

    def __len__(self):
        return len(self.action_sequence)

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.n_actions

    def zero_state(self, batch_size, dtype):
        return tf.cast(tf.fill((batch_size, 1), 0), dtype)


class CompositeCell(RNNCell):
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
    def __init__(self, cell, output, output_size):
        self.cell = cell
        self.output = output
        self._output_size = output_size

    def __call__(self, inp, state):
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


class FeedforwardCell(RNNCell):
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
    def __init__(self, ff, output_size):
        self.ff = ff
        self._output_size = output_size

    def __call__(self, inp, state):
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


class EarlyStopHook(object):
    def __init__(self, patience, name=None):
        self.patience = patience
        self.name = name

        self._early_stopped = 0
        self._best_value_gstep = None
        self._best_value_lstep = None
        self._best_value = None

        self._stage = 1
        self._history = []  # each element: (best_global_step, best_local_step, best_loss_value)

    @property
    def early_stopped(self):
        """Returns True if this monitor caused an early stop."""
        return self._early_stopped

    @property
    def best_step(self):
        """Returns the step at which the best early stopping metric was found."""
        return self._best_value_step

    @property
    def best_value(self):
        """Returns the best early stopping metric value found so far."""
        return self._best_value

    def check(self, validation_loss, global_step, local_step=None):
        local_step = global_step if local_step is None else local_step

        new_best = self._best_value is None or validation_loss < self._best_value
        if new_best:
            self._best_value = validation_loss
            self._best_value_gstep = global_step
            self._best_value_lstep = local_step

        stop = local_step - self._best_value_lstep > self.patience
        if stop:
            print("Stopping. Best step: global {}, local {} with loss = {}." .format(
                  self._best_value_lstep, self._best_value_gstep, self._best_value))
            self._early_stopped = True
        return new_best, stop

    def end_stage(self):
        self._history.append((self._best_value_gstep, self._best_value_lstep, self._best_value))
        self._stage += 1
        self._best_value = None
        self._best_value_gstep = None
        self._best_value_lstep = None
        self._early_stopped = 0

    def summarize(self):
        s = ""
        for stage, (bvgs, bvls, bv) in enumerate(self._history):
            s += "Stage {} ".format(stage) + "*" * 30 + '\n'
            s += "* best value: {}\n".format(bv)
            s += "* global step: {}\n".format(bvgs)
            s += "* local step: {}\n\n".format(bvls)
        return s


def restart_tensorboard(logdir, port=6006):
    print("Killing old tensorboard process...")
    try:
        command = "fuser {}/tcp -k".format(port)
        sp.run(command.split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        print("Killing tensorboard failed:")
        print(e.output)
    print("Restarting tensorboard process...")
    sp.Popen("tensorboard --logdir={} --port={}".format(logdir, port).split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
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


def adj_inverse_time_decay(initial, global_step, decay_steps, decay_rate, gamma,
                           staircase=False, name=None):
    """Applies inverse time decay to the initial learning rate.

    Adapted from tf.train.inverse_time_decay (added `gamma` arg.)

    The function returns the decayed learning rate.  It is computed as:

    ```python
    decayed_value = initial / (1 + decay_rate * t/decay_steps)
    ```

    Args:
      initial: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      global_step: A Python number.
        Global step to use for the decay computation.  Must not be negative.
      decay_steps: How often to apply decay.
      decay_rate: A Python number.  The decay rate.
      staircase: Whether to apply decay in a discrete staircase, as opposed to
        continuous, fashion.
      gamma: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The power to raise output to.
      name: String.  Optional name of the operation.  Defaults to
        'InverseTimeDecay'.

    Returns:
      A scalar `Tensor` of the same type as `initial`.  The decayed
      learning rate.

    Raises:
      ValueError: if `global_step` is not supplied.

    """
    if global_step is None:
        raise ValueError("global_step is required for adj_inverse_time_decay.")

    with ops.name_scope(name, "AdjInverseTimeDecay",
                        [initial, global_step, decay_rate]) as name:
        initial = ops.convert_to_tensor(initial, name="initial")
        dtype = initial.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        decay_rate = math_ops.cast(decay_rate, dtype)
        p = global_step / decay_steps
        if staircase:
            p = math_ops.floor(p)
        const = math_ops.cast(constant_op.constant(1), initial.dtype)
        denom = math_ops.add(const, math_ops.multiply(decay_rate, p))
        quotient = math_ops.div(initial, denom)
        gamma = math_ops.cast(gamma, dtype)
        return math_ops.pow(quotient, gamma, name=name)


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


def build_scheduled_value(schedule, name, dtype=None):
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

    Valid values for `kind`
    -----------------------
    constant value
        schedule_value = value

    exponential initial decay_steps decay_rate staircase(optional)
        t = floor(global_step / decay_rate) if staircase else (global_step / decay_rate)
        scheduled_value = learning_rate *
                                decay_rate ^ (global_step / decay_steps)

    polynomial initial decay_steps end power cycle(optional)
        if cycle:
            decay_steps = decay_steps * ceil(global_step / decay_steps)
        else:
            global_step = min(global_step, decay_steps)

        scheduled_value = (learning_rate - end_learning_rate) *
                                (1 - global_step / decay_steps) ^ (power) +
                                end_learning_rate
    inverse_time initial decay_steps decay_rate staircase(optional)
        t = floor(global_step / decay_rate) if staircase else (global_step / decay_rate)
        scheduled_value = learning_rate / (1 + decay_rate * t)

    """
    try:
        schedule = "constant {}".format(float(schedule))
    except (TypeError, ValueError):
        pass

    assert isinstance(schedule, str)
    kind, *args = schedule.split()
    kind = kind.lower()
    args = deque(args)

    if kind == "constant":
        scheduled_value = tf.constant(float(args[0]), dtype=dtype)
    elif kind == "exponential" or kind == "exp":
        initial = float(popleft(args))
        decay_steps = int(popleft(args))
        decay_rate = float(popleft(args))
        staircase = _bool(popleft(args, False))
        global_step = tf.contrib.framework.get_or_create_global_step()

        scheduled_value = tf.train.exponential_decay(
            initial, global_step, decay_steps, decay_rate, staircase, name=name)

    elif kind == "polynomial" or kind == "poly":
        initial = float(popleft(args))
        decay_steps = int(popleft(args))
        end = float(popleft(args))
        power = float(popleft(args))
        cycle = _bool(popleft(args, False))
        global_step = tf.contrib.framework.get_or_create_global_step()

        scheduled_value = tf.train.polynomial_decay(
            initial, global_step, decay_steps, end, power, cycle, name=name)
    elif kind == "inverse_time":
        initial = float(popleft(args))
        decay_steps = int(popleft(args))
        decay_rate = float(popleft(args))
        staircase = _bool(popleft(args, False))
        global_step = tf.contrib.framework.get_or_create_global_step()

        scheduled_value = tf.train.inverse_time_decay(
            initial, global_step, decay_steps, decay_rate, staircase, name=name)

    elif kind == "adj_inverse_time":
        initial = float(popleft(args))
        decay_steps = int(popleft(args))
        decay_rate = float(popleft(args))
        gamma = float(popleft(args))
        staircase = _bool(popleft(args, False))
        global_step = tf.contrib.framework.get_or_create_global_step()

        scheduled_value = adj_inverse_time_decay(
            initial, global_step, decay_steps, decay_rate, gamma, staircase, name=name)
    else:
        raise NotImplementedError(
            "No known schedule with kind `{}` and args `{}`.".format(kind, args))

    if dtype is not None:
        dtype = tf.as_dtype(np.dtype(dtype))
        scheduled_value = tf.cast(scheduled_value, dtype, name=name+"_cast")

    if name is not None:
        tf.summary.scalar(name, scheduled_value)

    return scheduled_value


def build_optimizer(spec, learning_rate):
    """
    `learning_rate` is always supplied as the first argument.

    Parameters
    ----------
    spec: str
        String of the form "kind arg1 arg2 ...".

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
        raise NotImplementedError(
            "No known optimizer with kind `{}` and args `{}`.".format(kind, args))

    return opt


class Config(object):
    _stack = []

    def __init__(self, **kwargs):
        self.update(kwargs)

    def __str__(self):
        attrs = {attr: getattr(self, attr) for attr in self.list_attrs()}
        s = "<{} -\n{}\n>".format(self.__class__.__name__, pformat(attrs))
        return s

    def __repr__(self):
        return str(self)

    def as_default(self):
        return context(self.__class__, self)

    def update(self, other, clobber=True):
        if isinstance(other, Config):
            for attr in other.list_attrs():
                if not hasattr(self, attr) or clobber:
                    setattr(self, attr, getattr(other, attr))
        elif isinstance(other, dict):
            for attr, value in other.items():
                if not hasattr(self, attr) or clobber:
                    setattr(self, attr, value)
        else:
            raise NotImplementedError()

    def list_attrs(self):
        return (
            attr for attr in dir(self)
            if (not attr.startswith('_') and
                not isinstance(getattr(self, attr), types.MethodType)))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return self.list_attrs()


def _parse_config_from_file(cls, key=None):
    config = configparser.ConfigParser()
    location = Path(dps.__file__).parent
    config.read(str(location / 'config.ini'))

    if not key:
        key = socket.gethostname().split('-')[0]

    if key not in config:
        key = 'DEFAULT'

    # Load default configuration from a file
    cls.hostname = socket.gethostname()
    cls.start_tensorboard = config.getboolean(key, 'start_tensorboard')
    cls.update_latest = config.getboolean(key, 'update_latest')
    cls.save_summaries = config.getboolean(key, 'save_summaries')
    cls.data_dir = process_path(config.get(key, 'data_dir'))
    cls.log_root = process_path(config.get(key, 'log_root'))
    cls.display = config.getboolean(key, 'display')
    cls.save_display = config.getboolean(key, 'save_display')
    cls.mpl_backend = config.get(key, 'mpl_backend')
    cls.use_gpu = config.getboolean(key, 'use_gpu')
    cls.visualize = config.getboolean(key, 'visualize')
    cls.tbport = config.getint(key, 'tbport')

    cls.max_experiments = config.getint(key, 'max_experiments')
    if cls.max_experiments <= 0:
        cls.max_experiments = np.inf
    return cls


@_parse_config_from_file
class DpsConfig(Config):
    log_dir = None
    log_name = "default"

    def __init__(self, **kwargs):
        super(DpsConfig, self).__init__(**kwargs)
        if self.log_dir is None:
            self.log_dir = str(Path(self.log_root) / self.log_name)


Config._stack.append(DpsConfig())


@contextmanager
def context(cls, obj):
    cls._stack.append(obj)
    yield
    cls._stack.pop()


def default_config():
    if not Config._stack:
        raise ValueError("Trying to get default config, but config stack is empty.")
    return Config._stack[-1]


def get_config(f, name):
    """ ``f`` is the name of the file where the relevant configs are defined, usually stored in __file__.
        name is the lowercase prefix of the config class we want to retrieve and instantiate.

    """
    config_classes = get_all_subclasses(Config)
    config_classes = list(set([c for c in config_classes if inspect.getfile(c) == f]))
    names = [c.__name__ for c in config_classes]
    assert len(names) == len(set(names)), "Duplicate config names: {}.".format(names)
    d = {n.split("Config")[0].lower(): c for n, c in zip(names, config_classes)}
    name = name.lower()
    try:
        return d[name]()
    except KeyError:
        raise KeyError("Unknown config name {}.".format(name))


def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses
