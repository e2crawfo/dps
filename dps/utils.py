import subprocess as sp
from pprint import pformat
from contextlib import contextmanager
import numpy as np
import inspect
import types

import tensorflow as tf
from tensorflow.python.ops import random_ops, math_ops
from tensorflow.python.framework import ops, constant_op
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.slim import fully_connected


def visual_filter(mu, diag_std, locations, values=None):
    """
    Parameters
    ----------
    mu: Tensor (batch_size, dim)
        Row i gives the centre of attention for batch-elment i.
    diag_std: Tensor (batch_size, dim)
        Row i gives the diagonal entries of the covariance matrix for batch element i.
    locations: Tensor (n_points, dim)
        Row j gives the j-th point to evaluate the filter at.
    values: Tensor (batch_size, n_points)
        Entry (i, j) gives the value of the function that we are filtering for batch-element
        i at the location given by ``locations[j, :]``. If not supplied, the filter
        itself is returned.

    Returns
    -------
    if values is not None:
        vision: Tensor (batch_size, 1)
            Row i gives the function for batch-element i (given by ``values[:, i]``) filtered by
            the filter for batch-element i (specified by ``mu[i, :]`` and ``diag_std[i, :]``).
    else:
        filt: Tensor (batch_size, n_points)
            Row i gives the filter for batch-element i.

    """
    assert len(locations.shape) == 2
    locations = tf.expand_dims(locations, 1)
    dist = tf.contrib.distributions.MultivariateNormalDiag(mu, diag_std)
    probs = tf.transpose(dist.pdf(locations))

    # Don't need to sqrt because we have stds rather than variances.
    Z = tf.reduce_prod(diag_std, axis=-1, keep_dims=True) * ((2 * np.pi) ** (int(mu.shape[1]) / 2))

    filt = Z * probs

    if values is not None:
        vision = tf.reduce_sum(values * filt, axis=1, keep_dims=True)
        return vision
    else:
        return filt


class MLP(object):
    def __init__(self, n_units=None, activation_fn=tf.nn.relu):
        self.n_units = n_units or []
        self.activation_fn = activation_fn

    def __call__(self, inp, output_size):
        hidden = inp
        for nu in self.n_units:
            hidden = fully_connected(hidden, nu, activation_fn=self.activation_fn)
        return fully_connected(hidden, output_size, activation_fn=None)


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

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.n_actions

    def zero_state(self, batch_size, dtype):
        return tf.cast(tf.fill((batch_size, 1), 0), dtype)


class CompositeCell(RNNCell):
    """ A wrapper around a cell that adds an arithmetical transformation of the output.

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
    def __init__(self, patience, n=1, name=None):
        self.patience = patience
        self.name = name

        self._early_stopped = 0
        self._best_value_gstep = None
        self._best_value_lstep = None
        self._best_value = None

        self._stage = 1
        self._history = {}  # stage -> (best_value_step, best_value)

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

    def check(self, global_step, local_step, validation_loss):
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
        self._history[self._stage] = (self._best_value_gstep, self._best_value_lstep, self._best_value)
        self._stage += 1
        self._best_value = None
        self._best_value_gstep = None
        self._best_value_lstep = None
        self._early_stopped = 0

    def summarize(self):
        s = ""
        for stage in sorted(self._history):
            bvgs, bvls, bv = self._history[stage]
            s += "Stage {} ".format(stage) + "*" * 30 + '\n'
            s += "* best value: {}\n".format(bv)
            s += "* global step: {}\n".format(bvgs)
            s += "* local step: {}\n\n".format(bvls)
        return s


def restart_tensorboard(logdir):
    print("Killing old tensorboard process...")
    try:
        sp.run("fuser 6006/tcp -k".split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as e:
        print("Killing tensorboard failed:")
        print(e.output)
    print("Restarting tensorboard process...")
    sp.Popen("tensorboard --logdir={}".format(logdir).split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    print("Done restarting tensorboard.")


def MSE(outputs, targets):
    return tf.reduce_mean(tf.square(tf.subtract(targets, outputs)))


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
    return list(zip(noisy_gradients, variables))


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

    with ops.name_scope(name, "InverseTimeDecay",
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


def build_decaying_value(schedule, name=None):
    global_step = tf.contrib.framework.get_or_create_global_step()
    start, decay_steps, decay_rate, staircase = schedule
    decaying_value = tf.train.exponential_decay(
        start, global_step, decay_steps, decay_rate, staircase=staircase)
    if name is not None:
        tf.summary.scalar(name, decaying_value)
    return decaying_value


class Config(object):
    _stack = []

    def __str__(self):
        attrs = {attr: getattr(self, attr) for attr in self.list_attrs()}
        s = "<{} -\n{}\n>".format(self.__class__.__name__, pformat(attrs))
        return s

    def as_default(self):
        return context(self.__class__, self)

    def update(self, other):
        for attr in other.list_attrs():
            setattr(self, attr, getattr(other, attr))

    def list_attrs(self):
        return (
            attr for attr in dir(self)
            if (not attr.startswith('_') and
                not isinstance(getattr(self, attr), types.MethodType)))


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
