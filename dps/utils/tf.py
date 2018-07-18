import subprocess as sp
import numpy as np
from collections import deque, OrderedDict, defaultdict
import os
import hashlib
import pprint
import argparse
from tabulate import tabulate
import shutil
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.slim import fully_connected

import dps
from dps.utils.base import _bool, popleft, Parameterized, Param
from dps.utils.inspect_checkpoint import get_tensors_from_checkpoint_file  # noqa: F401


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


if tf.__version__ >= "1.2":
    RNNCell = tf.nn.rnn_cell.RNNCell
else:
    from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell


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

    for v in trainable_variables("", for_opt=False):
        n_variables = int(np.prod(v.get_shape().as_list()))

        if v in all_fixed:
            fixed[""] += n_variables
            trainable[""] += 0
        else:
            fixed[""] += 0
            trainable[""] += n_variables

        name_so_far = ""

        for token in v.name.split("/")[:-1]:
            name_so_far += token
            if v in all_fixed:
                fixed[name_so_far] += n_variables
                trainable[name_so_far] += 0
            else:
                fixed[name_so_far] += 0
                trainable[name_so_far] += n_variables
            name_so_far += "/"

    table = ["scope trainable fixed total".split()]
    for scope in sorted(fixed, reverse=True):
        depth = sum(c == "/" for c in scope) + 1

        if max_depth is not None and depth > max_depth:
            continue

        table.append([
            scope,
            _fmt(trainable[scope]),
            _fmt(fixed[scope]),
            _fmt(trainable[scope] + fixed[scope])])

    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


def tf_normal_kl(q_mean, q_std, p_mean, p_std):
    return tf.log(p_std / q_std) + (q_std**2 + (q_mean - p_mean)**2) / (2 * p_std**2) - 0.5


def tf_mean_sum(t):
    """ Average over batch dimension, sum over all other dimensions """
    return tf.reduce_mean(tf.reduce_sum(tf.layers.flatten(t), axis=1))


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

    def trainable_variables(self, for_opt):
        return trainable_variables(self.scope, for_opt)

    def resolve_scope(self):
        if self.scope is None:
            with tf.variable_scope(self.name):
                self.scope = tf.get_variable_scope()

    def _call(self, inp, output_size, is_training):
        raise Exception("NotImplemented")

    def __call__(self, inp, output_size, is_training):
        self.resolve_scope()

        first_call = not self.initialized

        with tf.variable_scope(self.scope, reuse=self.initialized):
            if first_call:
                print("Entering var scope '{}' for first time.".format(self.scope.name))

            outp = self._call(inp, output_size, is_training)

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

        self.directory = directory or dps.cfg.log_dir

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
    def __init__(self, n_units=None, scope=None, **fc_kwargs):
        self.n_units = n_units or []
        self.fc_kwargs = fc_kwargs
        super(MLP, self).__init__(scope)

    def _call(self, inp, output_size, is_training):
        inp = tf.layers.flatten(inp)

        hidden = inp
        for i, nu in enumerate(self.n_units):
            hidden = fully_connected(hidden, nu, **self.fc_kwargs)

        fc_kwargs = self.fc_kwargs.copy()
        fc_kwargs['activation_fn'] = None

        try:
            output_dim = int(np.product([int(i) for i in output_size]))
            output_shape = output_size
        except Exception:
            output_dim = int(output_size)
            output_shape = (output_dim,)

        hidden = fully_connected(hidden, output_dim, **fc_kwargs)
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
    layout: list of dict
        Each entry supplies parameters for a layer of the network. Valid argument names are:
            kind
            filters (required, int)
            kernel_size (required, int or pair of ints)
            strides (defaults to 1, int or pair of ints)
            pool (defaults to False, bool, whether to apply 2x2 pooling with stride 2, pooling is never done on final layer)

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

    def __init__(self, layout, check_output_shape=False, scope=None, **kwargs):
        self.layout = layout
        self.check_output_shape = check_output_shape
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
    def predict_output_shape(input_shape, layout):
        """ Get spatial shape of the output given a spatial shape of the input. """
        shape = [int(i) for i in input_shape]
        for layer in layout:
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
        kind = layer_spec.get('kind', 'conv')

        if kind == 'conv':
            filters = layer_spec['filters']
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

        layer_string = ', '.join("{}={}".format(k, v) for k, v in sorted(layer_spec.items()))
        output_shape = tuple(int(i) for i in volume.shape[1:])
        print("CNN >>> Applying layer {} of kind {}: {}. Output shape: {}".format(idx, kind, layer_string, output_shape))

        return volume

    def _call(self, inp, output_size, is_training):
        volume = inp
        self.volumes = [volume]

        print("Predicted output shape is: {}".format(self.predict_output_shape(inp.shape[1:3], self.layout)))

        for i, layer_spec in enumerate(self.layout):
            volume = self._apply_layer(volume, layer_spec, i, i == len(self.layout) - 1, is_training)
            self.volumes.append(volume)

        if self.check_output_shape and output_size is not None:
            actual_shape = tuple(int(i) for i in volume.shape[1:])

            if actual_shape == output_size:
                print("CCN >>> Shape check passed.")
            else:
                raise Exception(
                    "Shape-checking turned on, and actual shape {} does not "
                    "match desired shape {}.".format(actual_shape, output_size))

        return volume


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
                    0. -
                    ((yy - mu_y)/std_y)**2 -
                    ((xx - mu_x)/std_x)**2
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
    print(command)
    sp.Popen(command.split(), stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    print("Done restarting tensorboard.\n")


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


def get_scheduled_value_summaries():
    return tf.get_collection('scheduled_value_summaries')


def build_scheduled_value(schedule, name=None, global_step=None, dtype=None):
    """
    Parameters
    ----------
    schedule: str
        String which returns a schedule object when eval-ed. One exception is that
        constants can be specified by simply supplying the constant value,
        with no kind string.
    name: str
        Name to use for the output op. Also creates a summary that has this name.
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
        tf.summary.scalar(name, scheduled_value, collections=['scheduled_value_summaries'])

    return scheduled_value


def eval_schedule(schedule):
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


class Exponential(Schedule):
    def __init__(self, start, end, decay_steps, decay_rate, staircase=False, log=False):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.log = log

        assert isinstance(self.decay_steps, int)
        assert self.decay_steps > 1
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
        assert self.decay_steps > 1
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
        assert self.decay_steps > 1
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


def masked_mean(array, mask, axis=None, keepdims=False):
    denom = tf.count_nonzero(mask, axis=axis, keepdims=keepdims)
    denom = tf.maximum(denom, 1)
    denom = tf.to_float(denom)
    return tf.reduce_sum(array * mask, axis=axis) / denom


def build_gradient_train_op(
        loss, tvars, optimizer_spec, lr_schedule, max_grad_norm=None,
        noise_schedule=None, global_step=None, summary_prefix=None, return_summaries=True):
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

    pre = summary_prefix + "_" if summary_prefix else ""
    records = {
        pre + 'grad_norm_pure': tf.global_norm(pure_gradients),
        pre + 'grad_norm_processed': tf.global_norm(noisy_gradients),
    }

    if return_summaries:
        summaries = [tf.scalar.summary(k, v) for k, v in records.items()]
        return train_op, summaries
    else:
        return train_op, records


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


class RenderHook(object):
    N = 16

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def imshow(self, ax, frame, **kwargs):
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]
        frame = np.clip(frame, 0.0, 1.0)

        ax.imshow(frame, vmin=0.0, vmax=1.0, **kwargs)

    def _fetch(self, updater):
        fetches = self.fetches

        if isinstance(fetches, str):
            fetches = fetches.split()

        feed_dict = updater.data_manager.do_val()

        to_fetch = {
            k: updater.network._tensors[k][:self.N]
            for k in fetches}

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)

        return fetched

    def savefig(self, name, fig, updater, is_dir=True):
        if is_dir:
            local_step = np.inf if dps.cfg.overwrite_plots else "{:0>10}".format(updater.n_updates)
            path = updater.exp_dir.path_for(
                'plots', name,
                'stage={:0>4}_local_step={}.pdf'.format(updater.stage_idx, local_step))
            fig.savefig(path)
            plt.close(fig)

            shutil.copyfile(
                path,
                os.path.join(os.path.dirname(path), 'latest_stage{:0>4}.pdf'.format(updater.stage_idx)))
        else:
            path = updater.exp_dir.path_for('plots', name + ".pdf")
            fig.savefig(path)
            plt.close(fig)
