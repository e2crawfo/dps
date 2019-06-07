import numpy as np
from collections import OrderedDict, defaultdict
import os
import hashlib
import pprint
import argparse
from tabulate import tabulate
import shutil
import matplotlib.pyplot as plt

import tensorflow as tf
try:
    from tensorflow.nn import dynamic_rnn, bidirectional_dynamic_rnn
except Exception:
    pass
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

import dps
from dps import cfg
from dps.utils.base import Parameterized, Param, Config, HierDict
from dps.utils.inspect_checkpoint import get_tensors_from_checkpoint_file  # noqa: F401


def apply_object_wise(func, *signals, restore_shape=True, n_trailing_dims=1, **func_kwargs):
    signal = signals[0]

    shape = tf_shape(signal)
    n_leading_dims = len(shape) - n_trailing_dims
    leading_shape = shape[:n_leading_dims]
    leading_dim = tf.reduce_prod(leading_shape)

    signals = [
        tf.reshape(s, (leading_dim, *tf_shape(s)[n_leading_dims:]))
        for s in signals]

    outputs = func(*signals, **func_kwargs)

    try:
        iter(outputs)
        outputs = list(outputs)
        is_iter = True
    except Exception:
        outputs = [outputs]
        is_iter = False

    if restore_shape:
        outputs = [tf.reshape(o, (*leading_shape, *tf_shape(o)[1:])) for o in outputs]

    if is_iter:
        return outputs
    else:
        return outputs[0]


def tf_cosine_similarity(a, b, keepdims=False):
    """ Supports broadcasting. """
    normalize_a = tf.nn.l2_normalize(a, axis=-1)
    normalize_b = tf.nn.l2_normalize(b, axis=-1)
    return tf.reduce_sum(normalize_a * normalize_b, axis=-1, keepdims=keepdims)


def tf_log_factorial(n):
    return tf.lgamma(tf.cast(n+1, tf.float32))


def tf_binomial_coefficient(n, k):
    log_bc = tf_log_factorial(n) - tf_log_factorial(k) - tf_log_factorial(n-k)
    return tf.exp(log_bc)


def tf_tensor_shape(shape):
    _tuple = []
    for i in shape:
        try:
            i = int(i)
        except (ValueError, TypeError):
            i = None
        _tuple.append(i)
    return tf.TensorShape(_tuple)


def tf_shape(tensor):
    """ Returns a tuple whose length is equal to the length of `tensor.shape`. Static shape is
        used where possible, and dynamic shape is used everywhere else.

    """
    assert isinstance(tensor, tf.Tensor)
    static_shape = tensor.shape
    dynamic_shape = tf.unstack(tf.shape(tensor))

    shape = []

    for d, s in zip(dynamic_shape, static_shape):
        if s is None or s.value is None:
            shape.append(d)
        else:
            shape.append(int(s))

    return tuple(shape)


def apply_mask_and_group_at_front(data, mask):
    """ For masking data and converting it into a format suitable for input into an RNN.
        Finds all the elements of data that correspond to "on" elements of the mask,
        and collects them all into a sequence for each batch element. Elements are
        collected in row-major order.


        >> data = np.arange(24).reshape(2, 2, 2, 3)
            array([[[[ 0,  1,  2],
                     [ 3,  4,  5]],
                    [[ 6,  7,  8],
                     [ 9, 10, 11]]],
                   [[[12, 13, 14],
                     [15, 16, 17]],
                    [[18, 19, 20],
                     [21, 22, 23]]]])

        >> mask = np.random.randint(2, size=(2, 2, 2))
            array([[[1, 1],
                    [0, 1]],
                   [[1, 0],
                    [0, 0]]])

        >> result, _, _ = apply_mask_and_group_at_front(data, mask)
        >> tf.Session.run(result)
            array([[[ 0,  1,  2],
                    [ 3,  4,  5],
                    [ 9, 10, 11]],

                   [[12, 13, 14],
                    [ 0,  0,  0],
                    [ 0,  0,  0]]])

    """
    mask = tf.cast(mask, tf.bool)

    batch_size = tf.shape(data)[0]

    if len(mask.shape) == len(data.shape):
        assert mask.shape[-1] == 1
        mask = mask[..., 0]

    assert len(mask.shape) == len(data.shape)-1
    # assert data.shape[1:-1] == mask.shape[1:]  Doesn't work if shapes partially unknown

    A = data.shape[-1]
    data = tf.reshape(data, (batch_size, -1, A))
    mask = tf.reshape(mask, (batch_size, -1))

    # data where the mask is "on". dimension should be (total_n_on, A)
    on_data = tf.boolean_mask(data, mask)

    # number of "on" elements in each batch element
    n_on = tf.reduce_sum(tf.layers.flatten(tf.to_int32(mask)), axis=1)

    # create an index array that can be used to index into on_data
    seq_mask = tf.sequence_mask(n_on)
    int_seq_mask = tf.to_int32(seq_mask)
    max_n_on = tf.shape(seq_mask)[1]
    indices = tf.cumsum(tf.reshape(int_seq_mask, (-1,)), exclusive=True, reverse=False)

    # Make sure dummy indices at the end are within bounds
    indices = tf.minimum(indices, tf.shape(on_data)[0]-1)

    result = tf.gather(on_data, indices)
    result = tf.reshape(result, (batch_size, max_n_on, A))

    # zero out the extra elements we've gathered
    result *= tf.cast(seq_mask, result.dtype)[:, :, None]
    return result, n_on, seq_mask


def tf_inspect_cl():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--names-only", action="store_true")
    args, _ = parser.parse_known_args()

    path = os.path.realpath(args.path)
    variables = get_tensors_from_checkpoint_file(path)

    if args.names_only:
        pprint.pprint(list(variables.keys()))
    else:
        pprint.pprint(variables)


RNNCell = tf.nn.rnn_cell.RNNCell


def count_trainable_variables(variables=None, var_scope=None):
    assert (variables is None) != (var_scope is None)

    if var_scope is not None:
        variables = trainable_variables(var_scope, for_opt=True)

    return np.sum([np.prod(v.get_shape().as_list()) for v in variables])


def walk_variable_scopes(max_depth=None):
    def _fmt(i):
        return "{:,}".format(i)

    all_fixed = set(tf.get_collection(FIXED_COLLECTION, scope=""))

    fixed = defaultdict(int)
    trainable = defaultdict(int)
    shapes = {}

    for v in trainable_variables("", for_opt=False):
        n_variables = int(np.prod(v.get_shape().as_list()))

        if v in all_fixed:
            fixed[""] += n_variables
            trainable[""] += 0
        else:
            fixed[""] += 0
            trainable[""] += n_variables
        shapes[v.name] = tuple(v.get_shape().as_list())

        name_so_far = ""

        for token in v.name.split("/"):
            name_so_far += token
            if v in all_fixed:
                fixed[name_so_far] += n_variables
                trainable[name_so_far] += 0
            else:
                fixed[name_so_far] += 0
                trainable[name_so_far] += n_variables
            name_so_far += "/"

    table = ["scope shape n_trainable n_fixed total".split()]

    any_shapes = False
    for scope in sorted(fixed, reverse=True):
        depth = sum(c == "/" for c in scope) + 1

        if max_depth is not None and depth > max_depth:
            continue

        if scope in shapes:
            shape_str = "{}".format(shapes[scope])
            any_shapes = True
        else:
            shape_str = ""

        table.append([
            scope,
            shape_str,
            _fmt(trainable[scope]),
            _fmt(fixed[scope]),
            _fmt(trainable[scope] + fixed[scope])])

    if not any_shapes:
        table = [row[:1] + row[2:] for row in table]

    print("TensorFlow variable scopes (down to maximum depth of {}):".format(max_depth))
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


def tf_normal_kl(q_mean, q_std, p_mean, p_std):
    return tf.log(p_std / q_std) + (q_std**2 + (q_mean - p_mean)**2) / (2 * p_std**2) - 0.5


def tf_mean_sum(t):
    """ Average over batch dimension, sum over all other dimensions """
    return tf.reduce_mean(tf.reduce_sum(tf.layers.flatten(t), axis=1))


def tf_atleast_nd(array, n):
    diff = n - len(array.shape)
    if diff > 0:
        s = (Ellipsis,) + (None,) * diff
        array = array[s]
    return array


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


def extract_glimpse_numpy_like(inp, glimpse_shape, glimpse_offsets, name=None, uniform_noise=None, fill_value=None):
    """ Based on: https://github.com/tensorflow/tensorflow/issues/2134#issuecomment-262525617

    Works like numpy with pixel coordinates starting at (0, 0), returns:
       inp[:, glimpse_offset[0] : glimpse_offset[0] + glimpse_size[0],
                glimpse_offset[1] : glimpse_offset[1] + glimpse_size[1], :]

    """
    assert(len(glimpse_shape) == 2)
    inp_shape = tuple(inp.get_shape().as_list())  # includes batch and number of channels
    corrected_offsets = 2 * glimpse_offsets - np.array(inp_shape[1:3]) + np.array(glimpse_shape)
    glimpses = tf.image.extract_glimpse(
        inp, glimpse_shape, corrected_offsets, centered=True, normalized=False,
        uniform_noise=uniform_noise, name=name)

    if fill_value is not None:
        glimpse_offsets = tf.cast(glimpse_offsets, tf.int32)
        y_indices = tf.range(glimpse_shape[0])
        y_indices = tf.reshape(y_indices, (1, -1))
        y_indices += glimpse_offsets[:, 0:1]
        valid_y = tf.cast(tf.logical_and(0 <= y_indices, y_indices < tf.shape(inp)[1]), tf.float32)
        valid_y = tf.expand_dims(valid_y, axis=-1)
        valid_y = tf.expand_dims(valid_y, axis=-1)

        glimpses = valid_y * glimpses + (1 - valid_y) * fill_value

        x_indices = tf.range(glimpse_shape[1])
        x_indices = tf.reshape(x_indices, (1, -1))
        x_indices += glimpse_offsets[:, 1:2]
        valid_x = tf.cast(tf.logical_and(0 <= x_indices, x_indices < tf.shape(inp)[2]), tf.float32)
        valid_x = tf.expand_dims(valid_x, axis=1)
        valid_x = tf.expand_dims(valid_x, axis=-1)

        glimpses = valid_x * glimpses + (1 - valid_x) * fill_value

    return glimpses


def uninitialized_variables_initializer():
    print("\nStarting variable init.")
    sess = tf.get_default_session()

    print("\nFinding uninitialized vars...")
    import time
    start = time.time()
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    uninitialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print("Took {} seconds".format(time.time() - start))
    print("\nInitializing {} var arrays...".format(len(uninitialized_vars)))
    start = time.time()
    uninit_init_op = tf.variables_initializer(uninitialized_vars)
    print("Took {} seconds.".format(time.time() - start))
    return uninit_init_op


FIXED_COLLECTION = "FIXED_COLLECTION"


def trainable_variables(scope, for_opt):
    if isinstance(scope, tf.VariableScope):
        scope = scope.name

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    if for_opt:
        fixed = set(tf.get_collection(FIXED_COLLECTION, scope=scope))
        variables = [v for v in variables if v not in fixed]

    return variables


class ScopedFunction(Parameterized):
    """
    Parameters
    ----------
    scope: string or VariableScope instance
        The scope where we will build the variables, or a string giving the name of a variable
        scope to be created within the variable scope where this function is first called.

    Attributes
    ----------
    scope: VariableScope instance or None
        If a VariableScope is passed to __init__, it is stored here. Otherwise, the first time
        that this instance of ScopedFunction is called, a new variable scope is created inside the
        scope in which the function is called. The name of the new variable scope is given by self.name.
    initialized: bool
        False up until the end of the first time that this instance of ScopedFunction is called.

    """
    fixed_values = Param(None)
    fixed_weights = Param("")
    no_gradient = Param("")

    def __init__(self, scope=None, **kwargs):
        if scope is None:
            scope = self.__class__.__name__

        if isinstance(scope, tf.VariableScope):
            self.name = scope.name.split("/")[-1]
            self.scope = scope
        else:
            self.name = scope
            self.scope = None

        self.initialized = False
        self.path = None
        self.directory = None
        self.train_config = None
        self.was_loaded = None
        self.do_pretraining = False
        self.fixed_variables = False

        self.fixed_values = self.fixed_values or {}

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        if isinstance(self.no_gradient, str):
            self.no_gradient = self.no_gradient.split()

        print(
            "\nBuilding {}(name={}) with args:\n{}".format(
                self.__class__.__name__, self.name, pprint.pformat(self._params_at_creation_time)))

    def trainable_variables(self, for_opt):
        return trainable_variables(self.scope, for_opt)

    def resolve_scope(self):
        if self.scope is None:
            with tf.variable_scope(self.name):
                self.scope = tf.get_variable_scope()

    def _call(self, *args, **kwargs):
        raise Exception("NotImplemented")

    def __call__(self, *args, **kwargs):
        self.resolve_scope()

        first_call = not self.initialized

        with tf.variable_scope(self.scope, reuse=self.initialized):
            if first_call:
                print("\nEntering var scope '{}' for first time.".format(self.scope.name))

            outp = self._call(*args, **kwargs)

            if first_call:
                s = "Leaving var scope '{}' for first time.".format(self.scope.name)
                if isinstance(outp, tf.Tensor):
                    s += " Actual output shape: {}.".format(outp.shape)
                print(s)

        self._maybe_initialize()

        return outp

    def _maybe_initialize(self):
        """ Initialize the network once it has been built. """

        if not self.initialized:
            if self.do_pretraining:
                from dps.train import load_or_train

                filename = "{}_{}.chk".format(self.name, self.param_hash)
                self.path = os.path.join(self.directory, filename)

                self.was_loaded = load_or_train(self.train_config, self.scope, self.path, target_var_scope=self.name)

                param_path = os.path.join(self.directory, "{}_{}.params".format(self.name, self.param_hash))

                if not os.path.exists(param_path):
                    with open(param_path, 'w') as f:
                        f.write(str(self.param_dict))

            if self.fixed_variables:
                for v in self.trainable_variables(False):
                    tf.add_to_collection(FIXED_COLLECTION, v)

            self.initialized = True

    def maybe_build_subnet(self, network_name, key=None, builder=None, builder_name=None):
        existing = getattr(self, network_name, None)

        if existing is None:
            if builder is None:
                if builder_name is None:
                    builder_name = "build_" + network_name

                builder = getattr(self, builder_name, None)

                if builder is None:
                    builder = getattr(cfg, builder_name, None)

                if builder is None:
                    raise AttributeError(
                        "No builder with name `{}` found for building subnet `{}`".format(builder_name, network_name))

            network = builder(scope=network_name)
            setattr(self, network_name, network)

            if key is None:
                key = network_name

            if key in self.fixed_weights:
                network.fix_variables()

            return network
        else:
            return existing

    def set_pretraining_params(self, train_config, name_params=None, directory=None):
        if self.initialized:
            raise Exception("ScopedFunction with scope {} has already been initialized, "
                            "it is an error to call `set_pretraining_params` at this point")

        assert train_config is not None
        self.train_config = train_config

        self.do_pretraining = True

        if isinstance(name_params, str):
            name_params = name_params.split()
        name_params = sorted(name_params or [])
        self.param_hash = get_param_hash(train_config, name_params)

        self.directory = directory or os.path.join(cfg.local_experiments_dir, cfg.env_name)

        self.param_dict = OrderedDict((key, train_config[key]) for key in name_params)

    def fix_variables(self):
        if self.initialized:
            raise Exception("ScopedFunction with scope {} has already been initialized, "
                            "it is an error to call `fix_variables` at this point")
        self.fixed_variables = True

    def save(self, session, filename):
        updater_variables = {v.name: v for v in self.trainable_variables(for_opt=False)}
        saver = tf.train.Saver(updater_variables)
        path = saver.save(tf.get_default_session(), filename)
        return path

    def restore(self, session, path):
        updater_variables = {v.name: v for v in self.trainable_variables(for_opt=False)}
        saver = tf.train.Saver(updater_variables)
        saver.restore(tf.get_default_session(), path)


def get_param_hash(train_config, name_params):
    param_str = []
    for name in name_params:
        value = train_config[name]
        try:
            value = sorted(value)
        except (TypeError, ValueError):
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


class IdentityFunction(ScopedFunction):
    def _call(self, inp, output_size, is_training):
        return inp


class MLP(ScopedFunction):
    n_units = Param(None)
    fc_kwargs = Param(None)

    def _call(self, inp, output_size, is_training):
        from tensorflow.contrib.slim import fully_connected
        inp = tf.layers.flatten(inp)

        n_units = self.n_units or []
        fc_kwargs = self.fc_kwargs or {}
        fc_kwargs = fc_kwargs.copy()

        hidden = inp
        for i, nu in enumerate(n_units):
            hidden = fully_connected(hidden, nu, **fc_kwargs)

        _fc_kwargs = fc_kwargs.copy()
        _fc_kwargs['activation_fn'] = None

        try:
            output_dim = int(np.product([int(i) for i in output_size]))
            output_shape = output_size
        except Exception:
            output_dim = int(output_size)
            output_shape = (output_dim,)

        hidden = fully_connected(hidden, output_dim, **_fc_kwargs)
        hidden = tf.reshape(hidden, (tf.shape(inp)[0], *output_shape), name="mlp_out")
        return hidden


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
        if len(images.shape) <= 1:
            raise Exception()

        if len(images.shape) == 2:
            s = int(np.sqrt(int(images.shape[1])))
            images = tf.reshape(images, (-1, s, s, 1))

        if len(images.shape) == 3:
            images = images[..., None]

        slim = tf.contrib.slim
        net = images
        net = slim.conv2d(net, 32, 5, scope='conv1', **self.conv_kwargs)
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        net = slim.conv2d(net, 64, 5, scope='conv2', **self.conv_kwargs)
        net = slim.max_pool2d(net, 2, 2, scope='pool2')

        trailing_dim = np.product([int(s) for s in net.shape[1:]])
        net = tf.reshape(net, (tf.shape(net)[0], trailing_dim))

        net = slim.fully_connected(net, self.n_units, scope='fc3', **self.fc_kwargs)
        net = slim.dropout(net, self.dropout_keep_prob, is_training=is_training, scope='dropout3')

        fc_kwargs = self.fc_kwargs.copy()
        fc_kwargs['activation_fn'] = None

        try:
            _output_size = output_size[0]
            assert len(output_size) == 1
        except Exception:
            _output_size = output_size

        net = slim.fully_connected(net, _output_size, scope='fc4', **fc_kwargs)
        return net


class VGGNet(ScopedFunction):

    def __init__(self, kind, scope=None):
        assert kind in 'a 16 19'.split()
        self.kind = kind
        super(VGGNet, self).__init__(scope)

    def _call(self, images, output_size, is_training):
        if len(images.shape) <= 1:
            raise Exception()
        if len(images.shape) == 2:
            s = int(np.sqrt(int(images.shape[1])))
            images = tf.reshape(images, (-1, s, s, 1))
        if len(images.shape) == 3:
            images = images[..., None]
        from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_a, vgg_16, vgg_19

        try:
            _output_size = output_size[0]
            assert len(output_size) == 1
        except IndexError:
            _output_size = output_size

        if self.kind == 'a':
            return vgg_a(images, _output_size, is_training)
        elif self.kind == '16':
            return vgg_16(images, _output_size, is_training)
        elif self.kind == '19':
            return vgg_19(images, _output_size, is_training)
        else:
            raise Exception()


class ConvNet(ScopedFunction):
    """
    Parameters
    ----------
    layers: list of dict
        Each entry supplies parameters for a layer of the network. Valid argument names are:
            kind
            filters (required, int)
            kernel_size (required, int or pair of ints)
            strides (defaults to 1, int or pair of ints)
            pool (defaults to False, bool, whether to apply 2x2 pooling with stride 2,
                  pooling is never done on final layer)

        Uses 'padding' == valid.

    """
    nonlinearities = dict(
        relu=tf.nn.relu,
        sigmoid=tf.nn.sigmoid,
        tanh=tf.nn.tanh,
        elu=tf.nn.elu,
        linear=lambda x: x,
        softmax=tf.nn.softmax
    )

    def __init__(self, layers, scope=None, **kwargs):
        self.layers = layers
        self.volumes = []
        super(ConvNet, self).__init__(scope)

    @staticmethod
    def _output_shape_1d(inp_dim, f, s, padding, pool):
        if padding == "SAME" or padding == "RIGHT_ONLY":
            if inp_dim % s == 0:
                p = f - s
            else:
                p = f - (inp_dim % s)

            out_dim = int((inp_dim + p - f) / s) + 1
        else:
            out_dim = int((inp_dim - f) / s) + 1
        return out_dim

    @staticmethod
    def predict_output_shape(input_shape, layers):
        """ Get spatial shape of the output given a spatial shape of the input. """
        shape = [int(i) for i in input_shape]
        for layer in layers:
            kernel_size = layer['kernel_size']
            if isinstance(kernel_size, tuple):
                f0, f1 = kernel_size
            else:
                f0, f1 = kernel_size, kernel_size

            strides = layer['strides']
            if isinstance(strides, tuple):
                strides0, strides1 = strides
            else:
                strides0, strides1 = strides, strides

            padding = layer.get('padding', 'VALID')
            pool = layer.get('pool', False)

            shape[0] = ConvNet._output_shape_1d(shape[0], f0, strides0, padding, pool)
            shape[1] = ConvNet._output_shape_1d(shape[1], f1, strides1, padding, pool)

        return shape

    @staticmethod
    def predict_padding(input_shape, layer):
        """ Predict padding that would be used by the "SAME" tensorflow padding settings. """
        shape = [int(i) for i in input_shape]
        kernel_size = layer['kernel_size']
        if isinstance(kernel_size, tuple):
            f0, f1 = kernel_size
        else:
            f0, f1 = kernel_size, kernel_size

        strides = layer['strides']
        if isinstance(strides, tuple):
            strides0, strides1 = strides
        else:
            strides0, strides1 = strides, strides

        if shape[0] % strides0 == 0:
            pad0 = max(f0 - strides0, 0)
        else:
            pad0 = max(f0 - (shape[0] % strides0), 0)

        if shape[1] % strides1 == 0:
            pad1 = max(f1 - strides1, 0)
        else:
            pad1 = max(f1 - (shape[1] % strides1), 0)

        return pad0, pad1

    @staticmethod
    def _apply_layer(volume, layer_spec, idx, is_final, is_training):
        from tensorflow.contrib.slim import fully_connected
        kind = layer_spec.get('kind', 'conv')

        if kind == 'conv':
            filters = layer_spec['filters']
            if filters is None:
                filters = tf_shape(volume)[-1]
            strides = layer_spec['strides']
            transpose = layer_spec.get('transpose', False)
            kernel_size = layer_spec['kernel_size']
            padding = layer_spec.get('padding', 'VALID')
            dropout = layer_spec.get('dropout', False)
            pool = layer_spec.get('pool', False)
            nl_string = layer_spec.get('nl', 'relu')
            nl = ConvNet.nonlinearities[nl_string or 'relu']

            if transpose:
                volume = tf.layers.conv2d_transpose(
                    volume, filters=filters, kernel_size=kernel_size,
                    strides=strides, padding=padding, name="fcn-conv_transpose{}".format(idx))
            else:
                if padding == "RIGHT_ONLY":
                    pad0, pad1 = ConvNet.predict_padding(volume.shape[1:3], layer_spec)
                    paddings = [[0, 0], [0, pad0], [0, pad1], [0, 0]]
                    volume = tf.pad(volume, paddings, mode="CONSTANT")
                    padding = "VALID"

                volume = tf.layers.conv2d(
                    volume, filters=filters, kernel_size=kernel_size,
                    strides=strides, padding=padding, name="fcn-conv{}".format(idx))

            if not is_final:
                volume = nl(volume, name="fcn-{}{}".format(nl_string, idx))

                if pool:
                    volume = tf.layers.max_pooling2d(
                        volume, pool_size=2, strides=2, name='fcn-pool{}'.format(idx))

            if dropout:
                volume = tf.contrib.slim.dropout(volume, 0.5, is_training=is_training)

        elif kind == 'fc':
            n_units = layer_spec['n_units']
            output_shape = layer_spec.get('output_shape', None)
            nl_string = layer_spec.get('nl', 'relu')
            nl = ConvNet.nonlinearities[nl_string or 'relu']

            volume = tf.layers.flatten(volume)
            volume = fully_connected(volume, n_units, activation_fn=nl)

            if output_shape is not None:
                batch_size = tf.shape(volume)[0]
                volume = tf.reshape(volume, (batch_size, *output_shape))
        elif kind == 'global_pool':  # a global spatial pooling layer
            pool_kind = layer_spec.get('pool_kind', 'mean')
            keepdims = layer_spec.get('keepdims', False)

            if pool_kind == "max":
                volume = tf.reduce_max(volume, axis=(1, 2), keepdims=keepdims)
            elif pool_kind == "mean":
                volume = tf.reduce_mean(volume, axis=(1, 2), keepdims=keepdims)
            elif pool_kind == "sum":
                volume = tf.reduce_sum(volume, axis=(1, 2), keepdims=keepdims)

        layer_string = ', '.join("{}={}".format(k, v) for k, v in sorted(layer_spec.items()))
        output_shape = tuple(int(i) for i in volume.shape[1:])
        print("CNN >>> Applying layer {} of kind {}: {}. Output shape: {}".format(idx, kind, layer_string, output_shape))

        return volume

    def _call(self, inp, output_size, is_training):
        final_n_channels = output_size

        print("--- Entering CNN(name={}) ---".format(self.name))
        volume = inp
        self.volumes = [volume]

        for i, layer in enumerate(self.layers):
            final = i == len(self.layers) - 1

            if final and final_n_channels is not None:
                layer['filters'] = final_n_channels

            volume = self._apply_layer(volume, layer, i, final, is_training)
            self.volumes.append(volume)

        print("--- Leaving CNN(name={}) ---".format(self.name))

        return volume


class GridConvNet(ConvNet):
    def __init__(self, layers, n_grid_dims=2, scope=None, **kwargs):
        self.layers = layers
        self.n_grid_dims = n_grid_dims
        self.volumes = []
        super(ConvNet, self).__init__(scope)

    @staticmethod
    def compute_receptive_field(ndim, layers):
        j = np.array((1,)*ndim)
        r = np.array((1,)*ndim)
        receptive_fields = []

        for layer in layers:
            kernel_size = np.array(layer['kernel_size'])
            stride = np.array(layer['strides'])
            r = r + (kernel_size-1) * j
            j = j * stride
            receptive_fields.append(dict(size=r, translation=j))
        return receptive_fields

    def _call(self, inp, output_size, is_training):
        final_n_channels = output_size
        volume = inp
        self.volumes = [volume]

        receptive_fields = self.compute_receptive_field(len(inp.shape)-2, self.layers)
        print("Receptive fields for {} (GridConvNet)".format(self.name))
        pprint.pprint(receptive_fields)

        grid_cell_size = receptive_fields[-1]["translation"][:self.n_grid_dims]
        rf_size = receptive_fields[-1]["size"][:self.n_grid_dims]
        pre_padding = np.floor(rf_size / 2 - grid_cell_size / 2).astype('i')
        image_shape = np.array([int(i) for i in inp.shape[1:self.n_grid_dims+1]])
        n_grid_cells = np.ceil(image_shape / grid_cell_size).astype('i')
        required_image_size = rf_size + (n_grid_cells-1) * grid_cell_size
        post_padding = required_image_size - image_shape - pre_padding

        print("{} (GridConvNet):".format(self.name))
        print("rf_size: {}".format(rf_size))
        print("grid_cell_size: {}".format(grid_cell_size))
        print("n_grid_cells: {}".format(n_grid_cells))
        print("pre_padding: {}".format(pre_padding))
        print("post_padding: {}".format(post_padding))
        print("required_image_size: {}".format(required_image_size))

        padding = (
            [[0, 0]]
            + list(zip(pre_padding, post_padding))
            + [[0, 0]] * (len(inp.shape) - 1 - self.n_grid_dims)
        )

        volume = tf.pad(inp, padding, mode="CONSTANT")

        for i, layer in enumerate(self.layers):
            padding_type = layer.get('padding', 'VALID')
            if padding_type != 'VALID':
                raise Exception("Layer {} trying to use padding type {} in GridConvNet.".format(i, padding_type))

            final = i == len(self.layers) - 1

            if final and final_n_channels is not None:
                layer['filters'] = final_n_channels

            volume = self._apply_layer(volume, layer, i, final, is_training)
            self.volumes.append(volume)

        return volume, n_grid_cells, grid_cell_size


class GridTransposeConvNet(GridConvNet):
    """ Incomplete, particularly figuring out the correct amount of padding..."""

    def _call(self, inp, output_size, is_training):
        volume = inp
        self.volumes = [volume]

        reverse_layers = self.layers[::-1]

        *image_shape, final_n_channels = output_size

        receptive_fields = self.compute_receptive_field(len(inp.shape)-2, reverse_layers)
        print("Inverse receptive fields for {} (GridTransposeConvNet)".format(self.name))
        pprint.pprint(receptive_fields)

        grid_cell_size = receptive_fields[-1]["translation"][:self.n_grid_dims]
        rf_size = receptive_fields[-1]["size"][:self.n_grid_dims]
        pre_padding = np.floor(rf_size / 2 - grid_cell_size / 2).astype('i')
        image_shape = np.array([int(i) for i in inp.shape[1:self.n_grid_dims+1]])
        n_grid_cells = np.ceil(image_shape / grid_cell_size).astype('i')
        required_image_size = rf_size + (n_grid_cells-1) * grid_cell_size
        post_padding = required_image_size - image_shape - pre_padding

        print("{} (GridTransposeConvNet):".format(self.name))
        print("rf_size: {}".format(rf_size))
        print("grid_cell_size: {}".format(grid_cell_size))
        print("n_grid_cells: {}".format(n_grid_cells))
        print("pre_padding: {}".format(pre_padding))
        print("post_padding: {}".format(post_padding))
        print("required_image_size: {}".format(required_image_size))

        for i, layer in enumerate(self.layers):
            padding_type = layer.get('padding', 'VALID')
            if padding_type != 'VALID':
                raise Exception("Layer {} trying to use padding type {} in GridTransposeConvNet.".format(i, padding_type))

            final = i == len(self.layers) - 1

            if final and final_n_channels is not None:
                layer['filters'] = final_n_channels

            volume = self._apply_layer(volume, layer, i, final, is_training)
            self.volumes.append(volume)

        slices = (
            [slice(None)]
            + [slice(pre, post) for pre, post in zip(pre_padding, post_padding)]
            + [slice(None)] * (len(inp.shape) - 1 - self.n_grid_dims)
        )

        volume = volume[slices]

        return volume, n_grid_cells, grid_cell_size


class RecurrentGridConvNet(GridConvNet):
    """ Operates on video rather than images. Apply a GridConvNet to each frame independently,
        and integrate information over time by using a recurrent network, where each spatial location
        has its own hidden state. The same recurrent network is used to update all spatial locations.

    """
    build_cell = Param()
    bidirectional = Param()

    forward_cell = None
    backward_cell = None

    def _call(self, inp, output_size, is_training):
        final_n_channels = output_size
        B, T, *rest = tf_shape(inp)
        inp = tf.reshape(inp, (B*T, *rest))

        processed, n_grid_cells, grid_cell_size = super()._call(inp, final_n_channels, is_training)

        _, H, W, C = tf_shape(processed)
        processed = tf.reshape(processed, (B, T, H, W, C))

        if self.build_cell is None:
            return processed, n_grid_cells, grid_cell_size

        processed = tf.transpose(processed, (1, 0, 2, 3, 4))
        processed = tf.reshape(processed, (T, B*H*W, C))

        if self.forward_cell is None:
            self.forward_cell = self.build_cell(n_hidden=final_n_channels, scope="forward_cell")

        if self.bidirectional:
            if self.backward_cell is None:
                self.backward_cell = self.build_cell(n_hidden=final_n_channels, scope="backward_cell")

            (fw_output, bw_output), final_state = bidirectional_dynamic_rnn(
                self.forward_cell, self.backward_cell, processed,
                initial_state_fw=self.forward_cell.zero_state(B*H*W, tf.float32),
                initial_state_bw=self.backward_cell.zero_state(B*H*W, tf.float32),
                parallel_iterations=1, swap_memory=False, time_major=True)
            output = (fw_output + bw_output) / 2

        else:
            output, final_state = dynamic_rnn(
                self.forward_cell, processed, initial_state=self.forward_cell.zero_state(B*H*W, tf.float32),
                parallel_iterations=1, swap_memory=False, time_major=True)

        output = tf.reshape(output, (T, B, H, W, C))
        output = tf.transpose(output, (1, 0, 2, 3, 4))
        return output, n_grid_cells, grid_cell_size


def pool_objects(op, objects, mask):
    batch_size = tf.shape(objects)[0]
    n_objects = tf.reduce_prod(tf.shape(objects)[1:-1])
    obj_dim = int(objects.shape[-1])

    mask = tf.reshape(mask, (batch_size, n_objects, 1))

    if op == "concat" or op is None:
        objects *= tf.to_float(mask)
        pooled_objects = tf.reshape(objects, (batch_size, n_objects*obj_dim))
    elif op == "sum":
        objects *= tf.to_float(mask)
        pooled_objects = tf.reduce_sum(objects, axis=1, keepdims=False)
    elif op == "max":
        mask = tf.tile(tf.cast(mask, tf.bool), (1, 1, obj_dim))
        objects = tf.where(mask, objects, -np.inf * tf.ones_like(objects))
        pooled_objects = tf.reduce_max(objects, axis=1, keepdims=False)
    else:
        raise Exception("Unknown symmetric op: {}. "
                        "Valid values are: None, concat, mean, max.".format(op))
    return pooled_objects


class ObjectNetwork(ScopedFunction):
    n_repeats = Param()
    d = Param()
    symmetric_op = Param()
    layer_norm = Param()
    use_mask = Param(help="If True, extract mask from objects by taking first element.")

    input_network = None
    object_network = None
    output_network = None

    def process_objects(self, batch_size, n_objects, objects, is_training):
        if self.object_network is None:
            self.object_network = dps.cfg.build_on_object_network(scope="object_network")

        for i in range(self.n_repeats):
            prev_objects = objects
            objects = self.object_network(prev_objects, self.d, is_training)
            objects += prev_objects

            if self.layer_norm:
                objects = tf.contrib.layers.layer_norm(objects, self.d)

        return objects

    def _call(self, inp, output_size, is_training):
        if self.input_network is None:
            self.input_network = dps.cfg.build_on_input_network(scope="input_network")
        if self.output_network is None:
            self.output_network = dps.cfg.build_on_output_network(scope="output_network")

        if self.use_mask:
            final_dim = int(inp.shape[-1])
            mask, inp = tf.split(inp, (1, final_dim-1), axis=-1)
            inp, _, mask = apply_mask_and_group_at_front(inp, mask)
        else:
            mask = tf.ones_like(inp[..., 0])

        batch_size = tf.shape(inp)[0]
        n_objects = tf.reduce_prod(tf.shape(inp)[1:-1])
        obj_dim = int(inp.shape[-1])

        inp = tf.reshape(inp, (batch_size*n_objects, obj_dim))
        objects = self.input_network(inp, self.d, is_training)

        objects = self.process_objects(batch_size, n_objects, objects, is_training)

        objects = tf.reshape(objects, (batch_size, n_objects, self.d))
        mask = tf.reshape(mask, (batch_size, n_objects))
        pooled_objects = pool_objects(self.symmetric_op, objects, mask)
        return self.output_network(pooled_objects, output_size, is_training)


class AttentionalRelationNetwork(ObjectNetwork):
    """ Implements one of the "attention blocks" from "Relational Deep Reinforcement Learning". """
    n_heads = Param()

    query_network = None
    key_network = None
    value_network = None

    def process_objects(self, batch_size, n_objects, objects, is_training):
        if self.query_network is None:
            self.query_network = dps.cfg.build_arn_network(scope="query_network")
        if self.key_network is None:
            self.key_network = dps.cfg.build_arn_network(scope="key_network")
        if self.value_network is None:
            self.value_network = dps.cfg.build_arn_network(scope="value_network")
        if self.object_network is None:
            self.object_network = dps.cfg.build_arn_object_network(scope="object_network")

        for i in range(self.n_repeats):
            a = []
            for h in range(self.n_heads):
                query = self.query_network(objects, self.d, is_training)
                query = tf.reshape(query, (batch_size, n_objects, self.d))
                if self.layer_norm:
                    query = tf.contrib.layers.layer_norm(query, self.d)

                key = self.key_network(objects, self.d, is_training)
                key = tf.reshape(key, (batch_size, n_objects, self.d))
                if self.layer_norm:
                    key = tf.contrib.layers.layer_norm(key, self.d)

                value = self.value_network(objects, self.d, is_training)
                value = tf.reshape(value, (batch_size, n_objects, self.d))
                if self.layer_norm:
                    value = tf.contrib.layers.layer_norm(value, self.d)

                s = tf.matmul(query, key, transpose_b=True)
                w = tf.nn.softmax(s/np.sqrt(self.d), axis=2)
                _a = tf.matmul(w, value)

                a.append(_a)

            a = tf.concat(a, axis=2)
            a = tf.reshape(a, (batch_size * n_objects, self.n_heads * self.d))

            prev_objects = objects
            objects = self.object_network(a, self.d, is_training)
            objects += prev_objects

            if self.layer_norm:
                objects = tf.contrib.layers.layer_norm(objects, self.d)

        return objects


class RelationNetwork(ScopedFunction):
    """ TODO: make this inherit from ObjectNetwork. """
    f = None
    g = None

    f_dim = Param()
    symmetric_op = Param()

    def _call(self, inp, output_size, is_training):
        # Assumes objects range of all but the first and last dimensions
        batch_size = tf.shape(inp)[0]
        spatial_shape = inp.shape[1:-1]
        n_objects = int(np.prod(spatial_shape))
        obj_dim = int(inp.shape[-1])
        inp = tf.reshape(inp, (batch_size, n_objects, obj_dim))

        if self.f is None:
            self.f = dps.cfg.build_relation_network_f(scope="relation_network_f")

        if self.g is None:
            self.g = dps.cfg.build_relation_network_g(scope="relation_network_g")

        f_inputs = []
        for i in range(n_objects):
            for j in range(n_objects):
                f_inputs.append(tf.concat([inp[:, i, :], inp[:, j, :]], axis=1))
        f_inputs = tf.concat(f_inputs, axis=0)

        f_output = self.f(f_inputs, self.f_dim, is_training)
        f_output = tf.split(f_output, n_objects**2, axis=0)

        if self.symmetric_op == "concat" or self.symmetric_op is None:
            g_input = tf.concat(f_output, axis=1)
        elif self.symmetric_op == "mean":
            g_input = tf.stack(f_output, axis=0)
            g_input = tf.reduce_mean(g_input, axis=0, keepdims=False)
        elif self.symmetric_op == "max":
            g_input = tf.stack(f_output, axis=0)
            g_input = tf.reduce_max(g_input, axis=0, keepdims=False)
        else:
            raise Exception("Unknown symmetric op for RelationNetwork: {}. "
                            "Valid values are: None, concat, mean, max.".format(self.symmetric_op))

        return self.g(g_input, output_size, is_training)


class VectorQuantization(ScopedFunction):
    H = Param()
    W = Param()
    K = Param()
    D = Param()
    common_embedding = Param(help="If True, all latent variables share a common set of embedding vectors.")

    _embedding = None

    def __call__(self, inp, output_size, is_training):
        if self._embedding is None:
            initializer = tf.truncated_normal_initializer(stddev=0.1)
            shape = (self.K, self.D)
            if not self.common_embedding:
                shape = (self.H, self.W,) + shape
            self._embedding = tf.get_variable("embedding", shape, initializer=initializer)

        self.z_e = inp

        if self.common_embedding:
            # self._embedding has shape (K, D), i.e. same dictionary used for all latents
            embedding = self._embedding[None, None, None, ...]
        elif len(self._embedding.shape) == 4:
            # self._embedding has shape (H, W, K, D), i.e. different dictionary for each latent
            embedding = self._embedding[None, ...]
        # shape of embedding should now be (1, H, W, K, D) either way

        z_e = self.z_e[..., None, :]  # (batch, H, W, 1, D)
        sum_squared_error = tf.reduce_sum((z_e - embedding) ** 2, axis=-1)
        self.k = k = tf.argmin(sum_squared_error, axis=-1)  # (batch, H, W)
        one_hot_k = tf.stop_gradient(tf.one_hot(k, self.K)[..., None])
        self.z_q = tf.reduce_sum(self._embedding[None, ...] * one_hot_k, axis=3)  # (batch, H, W, D)

        # On the forward pass z_q gets sent through, but the gradient gets sent back to z_e
        return tf.stop_gradient(self.z_q - self.z_e) + self.z_e


class VQ_ConvNet(ConvNet):
    H = Param()
    W = Param()
    K = Param()
    D = Param()

    common_embedding = Param(help="If True, all latent variables share a common set of embedding vectors.")

    _vq = None

    def _call(self, inp, output_size, is_training):
        if self._vq is None:
            self._vq = VectorQuantization(
                H=self.H, W=self.W, K=self.K, D=self.D,
                common_embedding=self.common_embedding)

        inp = self._vq(inp, (self.H, self.W, self.D), is_training)
        return super(VQ_ConvNet, self)._call(inp, output_size, is_training)


class SalienceMap(ScopedFunction):
    def __init__(
            self, n_locs, func, output_shape, std=None,
            flatten_output=False, scope=None):
        self.n_locs = n_locs
        self.func = func
        self.output_shape = output_shape
        self.std = std
        self.flatten_output = flatten_output
        super(SalienceMap, self).__init__(scope)

    def _call(self, inp, output_size, is_training):
        if self.std is None:
            func_output = self.func(inp, self.n_locs*5, is_training)
        else:
            func_output = self.func(inp, self.n_locs*3, is_training)

        y = (np.arange(self.output_shape[0]).astype('f') + 0.5) / self.output_shape[0]
        x = (np.arange(self.output_shape[1]).astype('f') + 0.5) / self.output_shape[1]
        yy, xx = tf.meshgrid(y, x, indexing='ij')
        yy = yy[None, ...]
        xx = xx[None, ...]
        output = None

        params = tf.nn.sigmoid(func_output/10.)

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
                    0.
                    - ((yy - mu_y)/std_y)**2
                    - ((xx - mu_x)/std_x)**2
                )
            )

            if output is None:
                output = new
            else:
                output = tf.maximum(new, output)

        if self.flatten_output:
            output = tf.reshape(
                output,
                (tf.shape(output)[0], int(np.prod(output.shape[1:])))
            )

        return output


class ScopedCell(ScopedFunction):
    @property
    def state_size(self):
        raise Exception("NotImplemented")

    @property
    def output_size(self):
        raise Exception("NotImplemented")

    def zero_state(self, batch_size, dtype):
        raise Exception("NotImplemented")


class ScopedCellWrapper(ScopedCell):
    """ Similar to ScopedCell, but used in cases where the cell we want to scope does not inherit from ScopedCell. """
    def __init__(self, cell, scope=None, **kwargs):
        self.cell = cell
        super(ScopedCellWrapper, self).__init__(scope=scope, **kwargs)

    def _call(self, inp, state):
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
    action_sequence: ndarray (n_timesteps,) + action_shape
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
    n_actions: int
        Number of actions.

    """
    def __init__(self, action_sequence, n_actions, name="fixed_discrete_controller"):
        self.action_sequence = np.array(action_sequence)
        self.n_actions = n_actions
        super(FixedDiscreteController, self).__init__(name)

    def _call(self, inp, state):
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
    """ A wrapper around a cell that adds additional transformations to the input and output.

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
    ignore_state = True

    def __init__(self, ff, output_size, name="feedforward_cell"):
        self.ff = ff
        self._output_size = output_size

        super(FeedforwardCell, self).__init__(name)

    def _call(self, inp, state):
        output = self.ff(inp, self._output_size, False)
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


def get_scheduled_values():
    sess = tf.get_default_session()
    return getattr(sess, "scheduled_values", {})


def build_scheduled_value(schedule, name=None, global_step=None, dtype=None):
    """
    Parameters
    ----------
    schedule: str
        Either a schedule object, or a string which returns a schedule object when eval-ed.
        One exception is that constants can be specified by simply supplying the constant value.
    name: str
        Name to use for the output op. Also creates a record in
        `tf.get_default_session().scheduled_values` with this name
    dtype: object convertible to tf.DType
        Will cast output value to this dtype.

    """
    op_name = name + "_schedule" if name else None

    schedule = eval_schedule(schedule)
    assert isinstance(schedule, Schedule), "{} is not a schedule instance.".format(schedule)

    global_step = tf.train.get_or_create_global_step() if global_step is None else global_step
    scheduled_value = schedule.build(global_step)

    if dtype is not None:
        dtype = tf.as_dtype(np.dtype(dtype))
        scheduled_value = tf.cast(scheduled_value, dtype, name=op_name)
    else:
        scheduled_value = tf.cast(scheduled_value, tf.float32, name=op_name)

    if name is not None:
        sess = tf.get_default_session()
        if not hasattr(sess, "scheduled_values"):
            sess.scheduled_values = {}
        sess.scheduled_values[name] = scheduled_value

    return scheduled_value


def eval_schedule(schedule):
    if isinstance(schedule, Schedule):
        return schedule

    try:
        schedule = "Constant({})".format(float(schedule))
    except (TypeError, ValueError):
        pass

    if isinstance(schedule, str):
        schedule = eval(schedule)

    return schedule


class Schedule(object):
    pass


class RepeatSchedule(Schedule):
    def __init__(self, schedule, period):
        self.schedule = schedule
        self.period = period

    def build(self, t):
        return self.schedule.build(t % self.period)


class LookupSchedule(Schedule):
    def __init__(self, sequence, delay=1):
        self.sequence = np.array(sequence)
        self.delay = delay

    def build(self, t):
        self.tf_sequence = tf.constant(self.sequence)
        idx = (t // self.delay) % len(self.sequence)
        return self.tf_sequence[idx]


class Exponential(Schedule):
    def __init__(self, start, end, decay_steps, decay_rate, staircase=False, log=False):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.log = log

        assert isinstance(self.decay_steps, int)
        assert self.decay_steps >= 1
        assert 0 <= self.decay_rate <= 1

    def build(self, t):
        if self.staircase:
            t = tf.to_float(t // self.decay_steps)
        else:
            t = t / self.decay_steps
        value = (self.start - self.end) * (self.decay_rate ** t) + self.end

        if self.log:
            value = tf.log(value + 1e-6)

        return value


class Exp(Exponential):
    pass


class Polynomial(Schedule):
    def __init__(self, start, end, decay_steps, power=1.0):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.power = power

        assert isinstance(self.decay_steps, int)
        assert self.decay_steps >= 1
        assert power > 0

    def build(self, t):
        t = tf.minimum(tf.cast(self.decay_steps, tf.int64), t)
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
        assert self.decay_steps >= 1
        assert self.gamma > 0

    def build(self, t):
        if self.staircase:
            t = tf.to_float(t // self.decay_steps)
        else:
            t = t / self.decay_steps
        return ((self.start - self.end) / (1 + t))**self.gamma + self.end


class Constant(Schedule):
    def __init__(self, value):
        self.value = value

    def build(self, t):
        return tf.constant(self.value)


# class MixtureSchedule(Schedule):
#     def __init__(self, components, reset_n_steps, shared_clock=False, p=None, name=None):
#         self.components = components
#         self.n_components = len(components)
#         self.reset_n_steps = reset_n_steps
#         self.shared_clock = shared_clock
#         self.p = p
#
#     def build(self, t):
#         t = t.copy()
#         n_periods = int(np.ceil(len(t) / self.reset_n_steps))
#         offsets = [0] * self.n_components
#         signal = []
#         for i in range(n_periods):
#             if len(signal) >= len(t):
#                 break
#             selected = np.random.choice(range(self.n_components), p=self.p)
#             if self.shared_clock:
#                 start = offsets[0]
#             else:
#                 start = offsets[selected]
#
#             t_ = t[start:start+self.reset_n_steps]
#             _signal = self.components[selected].build(t_)
#             signal.extend(_signal)
#
#             if self.shared_clock:
#                 offsets[0] += self.reset_n_steps
#             else:
#                 offsets[selected] += self.reset_n_steps
#         signal = np.array(signal).reshape(-1)[:len(t)]
#         return signal
#
#
# class ChainSchedule(Schedule):
#     def __init__(self, components, component_n_steps, shared_clock=False):
#         self.components = components
#         self.n_components = len(components)
#         self.component_n_steps = component_n_steps
#         self.shared_clock = shared_clock
#
#     def build(self, t):
#         try:
#             int(self.component_n_steps)
#             n_steps = [self.component_n_steps] * self.n_components
#         except Exception:
#             n_steps = self.component_n_steps
#
#         signal = []
#         offsets = [0] * self.n_components
#         for i in cycle(range(self.n_components)):
#             if len(signal) >= len(t):
#                 break
#             if self.shared_clock:
#                 start = offsets[0]
#             else:
#                 start = offsets[i]
#
#             t_ = t[start: start+n_steps[i]]
#             _signal = self.components[i].build(t_).astype('f')
#             signal.extend(list(_signal))
#
#             if self.shared_clock:
#                 offsets[0] += n_steps[i]
#             else:
#                 offsets[i] += n_steps[i]
#
#         return np.array(signal).reshape(-1)[:len(t)]


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
    kind, *kwargs = spec.split(',')
    kind = kind.lower()
    kwargs = [kw.split('=') for kw in kwargs]
    kwargs = {k.lower(): v for k, v in kwargs}

    if kind == "adam":
        beta1 = float(kwargs.get('beta1', 0.9))
        beta2 = float(kwargs.get('beta2', 0.999))
        epsilon = float(kwargs.get('epsilon', 1e-08))
        use_locking = bool(kwargs.get('use_locking', False))
        opt = tf.train.AdamOptimizer(
            learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, use_locking=use_locking)
    elif kind == "rmsprop":
        decay = float(kwargs.get('decay', 0.9))
        momentum = float(kwargs.get('momentum', 0.0))
        epsilon = float(kwargs.get('epsilon', 1e-10))
        use_locking = bool(kwargs.get('use_locking', False))
        centered = bool(kwargs.get('centered', False))

        opt = tf.train.RMSPropOptimizer(
            learning_rate, decay=decay, momentum=momentum,
            epsilon=epsilon, use_locking=use_locking, centered=centered)
    else:
        raise Exception(
            "No known optimizer with kind `{}` and kwargs `{}`.".format(kind, kwargs))

    return opt


def masked_mean(array, mask, axis=None, keepdims=False):
    denom = tf.count_nonzero(mask, axis=axis, keepdims=keepdims)
    denom = tf.maximum(denom, 1)
    denom = tf.to_float(denom)
    return tf.reduce_sum(array * mask, axis=axis) / denom


def build_gradient_train_op(
        loss, tvars, optimizer_spec, lr_schedule, max_grad_norm=None,
        noise_schedule=None, global_step=None, record_prefix=None,
        grad_n_record_groups=None):
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

    valid_lr = tf.Assert(
        tf.logical_and(tf.less(lr, 1.0), tf.less(0.0, lr)),
        [lr], name="valid_learning_rate")

    optimizer = build_optimizer(optimizer_spec, lr)

    with tf.control_dependencies([valid_lr]):
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    pre = record_prefix + "_" if record_prefix else ""

    records = {
        pre + 'grad_norm_pure': tf.global_norm(pure_gradients),
        pre + 'grad_norm_processed': tf.global_norm(noisy_gradients),
    }

    if grad_n_record_groups:
        # Break variables into groups until we have at least `grad_n_record_groups`-many groups,
        # and then record the RMSE for the gradient with respect to each group of variables.
        groups = HierDict()
        for v, g in zip(tvars, pure_gradients):
            name = v.name.replace('/', ':').replace('-', '_')
            groups[name] = g

        sep = '/'

        prefix_groups = {'': groups}
        while len(prefix_groups) < grad_n_record_groups:
            new_prefix_groups = {}

            expanded = False

            for p, g in prefix_groups.items():
                if isinstance(g, HierDict):
                    if len(g) == 1:
                        new_prefix_groups[p] = next(iter(g.values()))
                    else:
                        for k, v in g.items():
                            if p:
                                new_prefix_groups[p + sep + k] = v
                            else:
                                new_prefix_groups[k] = v
                    expanded = True
                else:
                    assert isinstance(g, tf.Tensor)
                    new_prefix_groups[p] = g

            prefix_groups = new_prefix_groups

            if not expanded:
                break

        variable_lists = {
            p: [g] if isinstance(g, tf.Tensor) else list(g.flatten().values())
            for p, g in prefix_groups.items()}

        normalized_gradient_norms = {
            pre + 'normed_subnet_grad/' + p: tf.global_norm(g) / np.sqrt(int(sum(np.prod(v.shape) for v in g)))
            for p, g in variable_lists.items()}

        records.update(normalized_gradient_norms)

    return train_op, records


def tf_roll(a, n, axis=0, fill=None, reverse=False):
    """ n > 0 corresponds to taking the final n elements, and putting them at the start.
        n < 0 corresponds to taking the first -n elements, and putting them at the end.

        If fill is not None then it should be a value with the same type as `a`.
        The space that is "vacated" (starting at the beginning of
        the array if n > 0, starting at the end if n < 0) is filled with the given value
        instead of a part of the original array.

        fill can also be an array that this broadcastable to the required shape (we just
        element-wise multiply fill with an array of ones of the appropriate shape).


    """
    if reverse:
        a = tf.reverse(a, axis=[axis])

    pre_slices = [slice(None) for i in a.shape]
    pre_slices[axis] = slice(None, -n)

    pre = a[pre_slices]

    post_slices = [slice(None) for i in a.shape]
    post_slices[axis] = slice(-n, None)

    post = a[post_slices]

    if fill is not None:
        if n > 0:
            post = fill * tf.ones_like(post, dtype=a.dtype)
        else:
            pre = fill * tf.ones_like(pre, dtype=a.dtype)

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


class RenderHook(object):
    N = 16
    is_training = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def remove_rects(ax):
        for obj in ax.findobj(match=plt.Rectangle):
            try:
                obj.remove()
            except NotImplementedError:
                pass

    def imshow(self, ax, frame, remove_rects=True, vmin=0.0, vmax=1.0, **kwargs):
        """ If ax already has an image, uses set_array on that image instead of doing imshow.
            Allows this function to work well with animations. """

        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]

        frame = np.clip(frame, vmin, vmax)
        frame = np.where(np.isnan(frame), 0, frame)

        if ax.images:
            ax.images[0].set_array(frame)
            if remove_rects:
                self.remove_rects(ax)
        else:
            ax.imshow(frame, vmin=vmin, vmax=vmax, **kwargs)

    def get_feed_dict(self, updater):
        return updater.data_manager.do_val(self.is_training)

    def build_fetches(self, updater):
        return self.fetches

    def start_stage(self, training_loop, updater, stage_idx):
        fetches = self.build_fetches(updater)

        if isinstance(fetches, str):
            fetches = fetches.split()

        try:
            tensors = updater.tensors
        except AttributeError:
            tensors = updater._tensors

        tensors_config = Config(tensors)
        to_fetch = {k: tensors_config[k] for k in fetches}
        self.to_fetch = nest.map_structure(lambda s: s[:self.N], to_fetch)

    def _fetch(self, updater, fetches=None):
        feed_dict = self.get_feed_dict(updater)
        sess = tf.get_default_session()
        return sess.run(self.to_fetch, feed_dict=feed_dict)

    def path_for(self, name, updater, ext="pdf"):
        local_step = (
            np.inf if dps.cfg.overwrite_plots else "{:0>10}".format(updater.n_updates))

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
