import subprocess as sp
import numpy as np
from collections import deque
import os
from pathlib import Path
import hashlib

import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_a, vgg_16, vgg_19
from tensorflow.contrib.slim import fully_connected

import dps
from dps.utils.base import Schedule, _bool, popleft, eval_schedule


if tf.__version__ >= "1.2":
    RNNCell = tf.nn.rnn_cell.RNNCell
else:
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell


def resize_image_with_crop_or_pad(img, target_height, target_width):
    if tf.__version__ >= "1.1":
        return tf.image.resize_image_with_crop_or_pad(img, target_height, target_width)
    else:
        batch_size = tf.shape(img)[0]
        img_height = int(img.shape[1])
        img_width = int(img.shape[2])
        depth = int(img.shape[3])

        upper_height = int(np.ceil((target_height - img_height) / 2))
        upper = tf.zeros((batch_size, upper_height, img_width, depth))

        lower_height = int(np.floor((target_height - img_height) / 2))
        lower = tf.zeros((batch_size, lower_height, img_width, depth))

        img = tf.concat([upper, img, lower], axis=1)

        left_width = int(np.ceil((target_width - img_width) / 2))
        left = tf.zeros((batch_size, target_height, left_width, depth))

        right_width = int(np.floor((target_width - img_width) / 2))
        right = tf.zeros((batch_size, target_height, right_width, depth))

        img = tf.concat([left, img, right], axis=2)

        return img


def extract_glimpse_numpy_like(inp, glimpse_shape, glimpse_offsets, name=None, uniform_noise=None):
    """ Taken from: https://github.com/tensorflow/tensorflow/issues/2134#issuecomment-262525617

    Works like numpy with pixel coordinates starting at (0, 0), returns:
       inp[:, glimpse_offset[0] : glimpse_offset[0] + glimpse_size[0],
                glimpse_offset[1] : glimpse_offset[1] + glimpse_size[1], :]

    """
    assert(len(glimpse_shape) == 2)
    inp_shape = tuple(inp.get_shape().as_list())  # includes batch and number of channels
    corrected_offsets = 2 * glimpse_offsets - np.array(inp_shape[1:3]) + np.array(glimpse_shape)
    return tf.image.extract_glimpse(
        inp, glimpse_shape, corrected_offsets, centered=True, normalized=False,
        uniform_noise=uniform_noise, name=name)


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


class ScopedFunction(object):
    """
    Parameters
    ----------
    scope: string or VariableScope instance
        The scope where we will build the variables, or a string giving the name of a variable
        scope to be created.

    """
    def __init__(self, scope=None):
        if scope is None:
            scope = self.__class__.__name__
        if isinstance(scope, tf.VariableScope):
            self.scope_name = scope.name
            self.scope = scope
        else:
            self.scope_name = scope
            self.scope = None

        self.n_builds = 0

        self.initialized = False
        self.path = None
        self.directory = None
        self.train_config = None
        self.was_loaded = None

        self.do_pretraining = False

    def resolve_scope(self):
        if self.scope is None:
            with tf.variable_scope(self.scope_name):
                self.scope = tf.get_variable_scope()

    def _call(self, inp, output_size, is_training):
        raise Exception("NotImplemented")

    def __call__(self, inp, output_size, is_training):
        self.resolve_scope()

        with tf.variable_scope(self.scope, reuse=self.n_builds > 0):
            outp = self._call(inp, output_size, is_training)

        self.n_builds += 1

        self._maybe_initialize()

        return outp

    def _maybe_initialize(self):
        if not self.initialized and (self.n_builds > 0 and self.do_pretraining):
            from dps.train import load_or_train
            self.was_loaded = load_or_train(self.train_config, self.scope, self.path)
            self.initialized = True

    def set_pretraining_params(self, train_config, name_params=None, directory=None):
        assert train_config is not None
        self.directory = str(directory or Path(dps.cfg.log_dir))
        if isinstance(name_params, str):
            name_params = name_params.split()
        name_params = sorted(name_params or [])
        param_hash = get_param_hash(train_config, name_params)
        filename = "{}_{}.chk".format(self.scope_name, param_hash)
        self.path = os.path.join(self.directory, filename)
        self.train_config = train_config
        self.do_pretraining = True

        self._maybe_initialize()


def get_param_hash(train_config, name_params):
    param_str = []
    for name in name_params:
        value = train_config[name]
        try:
            value = sorted(value)
        except:
            pass
        param_str.append("{}={}".format(name, value))
    param_str = "_".join(param_str)
    param_hash = hashlib.sha1(param_str.encode()).hexdigest()
    return param_hash


class ScopedFunctionWrapper(ScopedFunction):
    """ Similar to ScopedFunction, but used in cases where the function we want
        to scope does not inherit from ScopedFunction. """

    def __init__(self, function, scope=None):
        self.function = function
        super(ScopedFunctionWrapper, self).__init__(scope)

    def _call(self, inp, output_size, is_training):
        return self.function(inp, output_size, is_training)


class MLP(ScopedFunction):
    def __init__(self, n_units=None, scope=None, **fc_kwargs):
        self.n_units = n_units or []
        self.fc_kwargs = fc_kwargs
        super(MLP, self).__init__(scope)

    def _call(self, inp, output_size, is_training):
        if len(inp.shape) > 2:
            trailing_dim = np.product([int(s) for s in inp.shape[1:]])
            inp = tf.reshape(inp, (tf.shape(inp)[0], trailing_dim))
        hidden = inp
        for i, nu in enumerate(self.n_units):
            hidden = fully_connected(hidden, nu, **self.fc_kwargs)
        fc_kwargs = self.fc_kwargs.copy()
        fc_kwargs['activation_fn'] = None
        return fully_connected(hidden, output_size, **fc_kwargs)


class LeNet(ScopedFunction):
    def __init__(
            self, n_units=1024, dropout_keep_prob=0.5,
            conv_kwargs=None, fc_kwargs=None, scope=None):

        self.n_units = n_units
        self.dropout_keep_prob = dropout_keep_prob
        self.conv_kwargs = conv_kwargs or {}
        self.fc_kwargs = fc_kwargs or {}
        super(LeNet, self).__init__(scope)

    def _call(self, images, output_size, is_training):
        output_size = int(output_size)
        if len(images.shape) <= 1:
            raise Exception()

        if len(images.shape) == 2:
            s = int(np.sqrt(int(images.shape[1])))
            images = tf.reshape(images, (-1, s, s, 1))

        if len(images.shape) == 3:
            images = tf.expand_dims(images, -1)

        slim = tf.contrib.slim
        net = images
        net = slim.conv2d(net, 32, 5, scope='conv1', **self.conv_kwargs)
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        net = slim.conv2d(net, 64, 5, scope='conv2', **self.conv_kwargs)
        net = slim.max_pool2d(net, 2, 2, scope='pool2')
        net = slim.flatten(net)

        net = slim.fully_connected(net, self.n_units, scope='fc3', **self.fc_kwargs)
        net = slim.dropout(net, self.dropout_keep_prob, is_training=is_training, scope='dropout3')

        fc_kwargs = self.fc_kwargs.copy()
        fc_kwargs['activation_fn'] = None

        net = slim.fully_connected(net, output_size, scope='fc4', **fc_kwargs)
        return net


class VGGNet(ScopedFunction):

    def __init__(self, kind, scope=None):
        assert kind in 'a 16 19'.split()
        self.kind = kind
        super(VGGNet, self).__init__(scope)

    def _call(self, images, output_size, is_training):
        output_size = int(output_size)
        if len(images.shape) <= 1:
            raise Exception()
        if len(images.shape) == 2:
            s = int(np.sqrt(int(images.shape[1])))
            images = tf.reshape(images, (-1, s, s, 1))
        if len(images.shape) == 3:
            images = tf.expand_dims(images, -1)

        if self.kind == 'a':
            return vgg_a(images, output_size, is_training)
        elif self.kind == '16':
            return vgg_16(images, output_size, is_training)
        elif self.kind == '19':
            return vgg_19(images, output_size, is_training)
        else:
            raise Exception()


class FullyConvolutional(ScopedFunction):
    def __init__(self, layer_kwargs, pool=True, flatten_output=False, scope=None):
        self.layer_kwargs = layer_kwargs
        self.pool = pool
        self.flatten_output = flatten_output
        self.scope = scope
        super(FullyConvolutional, self).__init__(scope)

    def _call(self, images, output_size, is_training):
        output_size = int(output_size)
        if len(images.shape) <= 1:
            raise Exception()

        if len(images.shape) == 2:
            s = int(np.sqrt(int(images.shape[1])))
            images = tf.reshape(images, (-1, s, s, 1))

        if len(images.shape) == 3:
            images = tf.expand_dims(images, -1)
        slim = tf.contrib.slim
        net = images

        for i, kw in enumerate(self.layer_kwargs):
            net = slim.conv2d(net, scope='conv'+str(i), **kw)
            if self.pool:
                net = slim.max_pool2d(net, 2, 2, scope='pool'+str(i))

        if self.flatten_output:
            net = tf.reshape(net, (tf.shape(net)[0], int(np.prod(net.shape[1:]))))

        return net


class SalienceMap(ScopedFunction):
    def __init__(
            self, n_locs, func, output_dims, std=None,
            flatten_output=False, scope=None):
        self.n_locs = n_locs
        self.func = func
        self.output_dims = output_dims
        self.std = std
        self.flatten_output = flatten_output
        super(SalienceMap, self).__init__(scope)

    def _call(self, inp, output_size, is_training):
        if self.std is None:
            func_output = self.func(inp, self.n_locs*5, is_training)
        else:
            func_output = self.func(inp, self.n_locs*3, is_training)

        y = (np.arange(self.output_dims[0]).astype('f') + 0.5) / self.output_dims[0]
        x = (np.arange(self.output_dims[1]).astype('f') + 0.5) / self.output_dims[1]
        yy, xx = tf.meshgrid(y, x, indexing='ij')
        yy = yy[None, ...]
        xx = xx[None, ...]
        output = None

        params = tf.nn.sigmoid(func_output/100.)

        per_loc_params = tf.split(params, self.n_locs, axis=1)
        for p in per_loc_params:
            if self.std is None:
                weight, mu_y, mu_x, std_y, std_x = tf.unstack(p, axis=1)
                std_y = std_y[:, None, None]
                std_x = std_x[:, None, None]
            else:
                weight, mu_y, mu_x = tf.unstack(p, axis=1)
                try:
                    std_y = float(self.std)
                    std_x = float(self.std)
                except (TypeError, ValueError):
                    std_y, std_x = self.std

            weight = weight[:, None, None]
            mu_y = mu_y[:, None, None]
            mu_x = mu_x[:, None, None]

            new = weight * tf.exp(
                0.5 * (
                    0. -
                    ((yy - mu_y)/std_y)**2 -
                    ((xx - mu_x)/std_x)**2
                )
            )

            if output is None:
                output = new
            else:
                output += new

        if self.flatten_output:
            output = tf.reshape(
                output,
                (tf.shape(output)[0], int(np.prod(output.shape[1:])))
            )

        return output


class ScopedCell(RNNCell):
    """ An RNNCell that creates its own variable scope the first time `resolve_scope` is called.
        The scope is then carried around and used for any subsequent build operations, ensuring
        that all uses of an instance of this class use the same set of variables.

    """
    def __init__(self, name):
        self.scope = None
        super(ScopedCell, self).__init__(name=name or self.__class__.__name__)

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
    outp: callable (Tensor, int) -> Tensor
        Maps from an input tensor and an output size to an output tensor.
    output_size: int
        The size of the output, passed as the second argument when calling ``output``.
    inp: callable (Tensor, int) -> Tensor
        Maps from an input tensor and an output size to a new input tensor for cell.
    inp_size: int
        Size of the vector that `input` maps to
        Maps from an input tensor and an output size to a new input tensor for cell.

    """
    def __init__(self, cell, outp, output_size, inp=None, name="composite_cell"):
        self.cell = cell
        self.outp = outp
        self._output_size = output_size
        self.inp = inp

        super(CompositeCell, self).__init__(name)

    def _call(self, inp, state):
        if self.inp is not None:
            inp = self.inp(inp)
        output, new_state = self.cell(inp, state)
        return self.outp(output, self._output_size, False), new_state

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
    global_step = tf.train.get_or_create_global_step() if global_step is None else global_step

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
