import subprocess as sp
import numpy as np
from pathlib import Path
from collections import deque

import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.slim import fully_connected

import dps
from dps.utils.base import Schedule, _bool, popleft, eval_schedule


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


def scheduled_value_summaries():
    return tf.get_collection('scheduled_value_summaries')


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


def trainable_variables(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


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

    schedule = eval_schedule(schedule)
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
