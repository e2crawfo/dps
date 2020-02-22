from collections import defaultdict
import numpy as np
from tabulate import tabulate
import torch
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
import pprint

from dps.utils.base import RenderHook as _RenderHook, Config, map_structure, Param, Parameterized


def to_np(cuda_tensor):
    return cuda_tensor.detach().cpu().numpy()


def walk_variable_scopes(model, max_depth=None):
    def _fmt(i):
        return "{:,}".format(i)

    fixed_vars = set()

    n_fixed = defaultdict(int)
    n_trainable = defaultdict(int)
    shapes = {}

    for name, v in model.named_parameters():
        n_variables = int(np.prod(tuple(v.size())))

        if name in fixed_vars:
            n_fixed[""] += n_variables
            n_trainable[""] += 0
        else:
            n_fixed[""] += 0
            n_trainable[""] += n_variables

        shapes[name] = tuple(v.shape)

        name_so_far = ""

        for token in name.split("."):
            name_so_far += token
            if v in fixed_vars:
                n_fixed[name_so_far] += n_variables
                n_trainable[name_so_far] += 0
            else:
                n_fixed[name_so_far] += 0
                n_trainable[name_so_far] += n_variables
            name_so_far += "."

    table = ["scope shape n_trainable n_fixed total".split()]

    any_shapes = False
    for scope in sorted(n_fixed, reverse=True):
        depth = sum(c == "." for c in scope) + 1

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
            _fmt(n_trainable[scope]),
            _fmt(n_fixed[scope]),
            _fmt(n_trainable[scope] + n_fixed[scope])])

    if not any_shapes:
        table = [row[:1] + row[2:] for row in table]

    print("PyTorch variable scopes (down to maximum depth of {}):".format(max_depth))
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


class RenderHook(_RenderHook):
    def process_data(self, data, N=None):
        if N is None:
            N = self.N
        return map_structure(
            lambda t: t[:N] if isinstance(t, torch.Tensor) else t,
            data, is_leaf=lambda t: not isinstance(t, (dict, list, tuple, set)))

    def get_tensors(self, data, updater):
        tensors, recorded_tensors, losses = updater.model(data, plot=True, is_training=False)
        tensors = Config(tensors)
        tensors = map_structure(
            lambda t: to_np(t) if isinstance(t, torch.Tensor) else t,
            tensors, is_leaf=lambda rec: not isinstance(rec, dict))
        return tensors, recorded_tensors, losses


class ParameterizedModule(torch.nn.Module, Parameterized):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        Parameterized.__init__(self, **kwargs)


class ConvNet(ParameterizedModule):
    layer_specs = Param()

    nonlinearities = dict(
        relu=F.relu,
        sigmoid=F.sigmoid,
        tanh=F.tanh,
        elu=F.elu,
        softmax=F.softmax,
        linear=lambda x: x,
    )

    def __init__(self, input_n_channels, output_size=None, input_image_shape=None, **kwargs):
        super().__init__(**kwargs)

        self.module_list = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleDict()

        spatial_shape = input_image_shape
        print("Spatial shape: {}".format(spatial_shape))
        prev_n_channels = input_n_channels
        prev_n_units = None

        for i, layer_spec in enumerate(self.layer_specs):
            kind = layer_spec['kind']

            is_last = i == len(self.layer_specs)-1

            if kind == 'conv':
                n_filters = layer_spec['n_filters']
                kernel_size = layer_spec['kernel_size']
                stride = layer_spec.get('stride', 1)
                padding = layer_spec.get('padding', 0)

                if is_last and output_size is not None:
                    n_filters = output_size

                layer = torch.nn.Conv2d(prev_n_channels, n_filters, kernel_size, stride=stride, padding=padding)
                self.module_list.append(layer)

                if not is_last and layer_spec.get('batch_norm', False):
                    bn = torch.nn.BatchNorm2d(n_filters)
                    self.batch_norms[str(i)] = bn

                prev_n_channels = n_filters

                if spatial_shape is not None:
                    spatial_shape = self.conv_output_shape(spatial_shape, kernel_size, stride, padding)

            elif kind == 'fc':
                if spatial_shape is not None:
                    prev_n_units = prev_n_channels * spatial_shape[0] * spatial_shape[1]
                else:
                    assert prev_n_units is not None
                spatial_shape = None

                n_units = layer_spec['n_units']

                if is_last and output_size is not None:
                    n_units = output_size

                layer = torch.nn.Linear(prev_n_units, n_units)
                self.module_list.append(layer)

                if not is_last and layer_spec.get('batch_norm', False):
                    bn = torch.nn.BatchNorm1d(n_units)
                    self.batch_norms[str(i)] = bn

                prev_n_units = n_units
            else:
                raise Exception("Unknown layer kind: {}".format(kind))

            print(self.module_list[-1])
            if spatial_shape is not None:
                print("Spatial shape after applying layer: {}".format(spatial_shape))

    def forward(self, x):
        for i, (layer, layer_spec) in enumerate(zip(self.module_list, self.layer_specs)):
            is_last = i == len(self.module_list) - 1

            if layer_spec['kind'] == 'fc':
                b = x.shape[0]
                x = x.view(b, -1)

            x = layer(x)

            if layer_spec.get('layer_norm', False):
                x = torch.nn.LayerNorm(x)

            bn_key = str(i)
            if bn_key in self.batch_norms:
                x = self.batch_norms[bn_key](x)

            if not is_last:
                nl_key = layer_spec.get('nl', 'relu')
                nl = ConvNet.nonlinearities[nl_key]
                x = nl(x)

        return x

    @staticmethod
    def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        """ Utility function for computing output shapes of convolutions. """

        if type(h_w) is not tuple:
            h_w = (h_w, h_w)

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        if type(stride) is not tuple:
            stride = (stride, stride)

        if type(pad) is not tuple:
            pad = (pad, pad)

        h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
        w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

        return (h, w)


class SimpleConvNet(ConvNet):
    """ A standard ConvNet that ends with a series of fully connected layers. """

    n_layers_in_between = Param()
    n_conv_blocks = Param()
    stride = Param()
    base_n_filters = Param()
    max_n_filters = Param()
    kernel_size = Param()

    n_fc_layers = Param()
    n_fc_units = Param()

    batch_norm = Param()

    layer_specs = None

    def __init__(self, input_n_channels, output_size=None, input_image_shape=None, **kwargs):
        layer_specs = []

        def add_conv_layer(n_filters, stride=1):
            nonlocal layer_specs
            n_filters = min(n_filters, self.max_n_filters)
            layer_specs.append(
                dict(
                    kind='conv', n_filters=n_filters, stride=stride,
                    kernel_size=self.kernel_size, batch_norm=self.batch_norm
                )
            )

        for j in range(self.n_layers_in_between):
            add_conv_layer(self.base_n_filters)

        for i in range(1, self.n_conv_blocks+1):
            n_filters = (self.stride**i) * self.base_n_filters

            add_conv_layer(n_filters, stride=self.stride)

            for j in range(self.n_layers_in_between):
                add_conv_layer(n_filters)

        for i in range(self.n_fc_layers):
            layer_specs.append(
                dict(kind='fc', batch_norm=self.batch_norm, n_units=self.n_fc_units)
            )

        self.layer_specs = layer_specs
        pprint.pprint(self.layer_specs)

        super().__init__(input_n_channels, output_size=output_size, input_image_shape=input_image_shape, **kwargs)


activations = dict(
    relu=F.relu,
    sigmoid=F.sigmoid,
    tanh=F.tanh,
    elu=F.elu,
    linear=lambda x: x,
)
activations[None] = lambda x: x
nonlinearities = activations


activation_module_classes = dict(
    relu=torch.nn.ReLU,
    sigmoid=torch.nn.Sigmoid,
    tanh=torch.nn.Tanh,
    elu=torch.nn.ELU,
    linear=torch.nn.Identity,
)
activation_module_classes[None] = torch.nn.Identity


def activation_module(s):
    return activation_module_classes[s]()


class MLP(ParameterizedModule):
    n_hidden_units = Param()
    nl = Param()
    batch_norm = Param(False)
    layer_norm = Param(False)

    def __init__(self, n_inputs, n_outputs, **kwargs):
        super().__init__(**kwargs)

        assert not (self.batch_norm and self.layer_norm)

        self.module_list = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        prev_n_units = n_inputs

        for n_units in self.n_hidden_units:
            self.module_list.append(torch.nn.Linear(prev_n_units, n_units))
            prev_n_units = n_units

            if self.batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(n_units))

            if self.layer_norm:
                self.layer_norms.append(LayerNorm(n_units))

        self.module_list.append(torch.nn.Linear(prev_n_units, n_outputs))

    def forward(self, x):
        for i, layer in enumerate(self.module_list):
            x = layer(x)

            is_last = i == len(self.module_list)-1

            if not is_last:
                x = nonlinearities[self.nl](x)

            # Apparently it's better to apply norm after the non-linearity.
            if not is_last:
                if self.batch_norm:
                    x = self.batch_norms[i](x)

                if self.layer_norm:
                    x = self.layer_norms[i](x)

        return x


class UNET(ParameterizedModule):
    encoder_layer_specs = Param()
    batch_norm = Param()
    preserve_shape = Param()

    def __init__(self, input_n_channels, output_size=None, **kwargs):
        super().__init__(**kwargs)

        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        self.encoder_batch_norm_layers = torch.nn.ModuleList()
        self.decoder_batch_norm_layers = torch.nn.ModuleList()

        prev_n_channels = input_n_channels

        # TODO: initialize weights with truncated normal.
        #  kernel_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.0001)

        self.skip_volume_indices = []

        for i, layer_spec in enumerate(self.encoder_layer_specs):
            n_filters = layer_spec['n_filters']
            kernel_size = layer_spec['kernel_size']
            stride = layer_spec.get('stride', 1)

            layer = torch.nn.Conv2d(
                prev_n_channels, n_filters, kernel_size=kernel_size, stride=stride,
            )
            print("UNET Encoder layer {}: n_in={}, n_out={}".format(i, prev_n_channels, n_filters))
            self.encoder_layers.append(layer)

            if self.batch_norm:
                self.encoder_batch_norm_layers.append(torch.nn.BatchNorm2d(n_filters))

            if stride > 1:
                self.skip_volume_indices.append(i)

            prev_n_channels = n_filters

        self.skip_volume_indices = list(reversed(self.skip_volume_indices))

        for j, skip_volume_idx in enumerate(self.skip_volume_indices):
            layer_spec = self.encoder_layer_specs[skip_volume_idx]
            layer = self.encoder_layers[skip_volume_idx]
            skip_volume_n_channels = layer.in_channels

            kernel_size = layer_spec['kernel_size']
            stride = layer_spec['stride']

            is_last = j == len(self.skip_volume_indices)-1

            if is_last and output_size is not None:
                n_output_channels = output_size
            else:
                n_output_channels = skip_volume_n_channels

            assert prev_n_channels % (stride**2) == 0
            n_input_channels = skip_volume_n_channels + prev_n_channels // (stride**2)

            layer = torch.nn.Conv2d(
                n_input_channels, n_output_channels, kernel_size=kernel_size,
            )
            self.decoder_layers.append(layer)

            print("UNET Decoder layer {}: n_in={}, n_out={}".format(j, n_input_channels, n_output_channels))

            if self.batch_norm:
                self.decoder_batch_norm_layers.append(torch.nn.BatchNorm2d(n_output_channels))

            prev_n_channels = n_output_channels

    def forward(self, inp):
        encoder_volumes = [inp]
        volume = inp
        for i, (layer_spec, layer) in enumerate(zip(self.encoder_layer_specs, self.encoder_layers)):
            # print("Applying encoder layer: {}".format(layer))

            if self.preserve_shape:
                volume = self.pad_to_preserve_shape(volume, layer_spec)
                # print("Padding to preserve shape, shape of input volume after padding: {}".format(volume.shape))

            volume = layer(volume)
            volume = F.relu(volume)

            if self.batch_norm:
                volume = self.encoder_batch_norm_layers[i](volume)

            encoder_volumes.append(volume)

            # print("Output shape: {}".format(volume.shape))
            # print()

        output_volumes = []

        for i, skip_volume_idx in enumerate(self.skip_volume_indices):
            matching_encoder_layer_spec = self.encoder_layer_specs[skip_volume_idx]
            stride = matching_encoder_layer_spec['stride']

            upsampled_volume = F.pixel_shuffle(volume, stride)
            # upsampled_volume = tf.depth_to_space(volume, block_size=strides)

            skip_volume = encoder_volumes[skip_volume_idx]

            # print("Input shapes are: {}, {}".format(upsampled_volume.shape, skip_volume.shape))

            upsampled_volume, skip_volume = self.min_pad(upsampled_volume, skip_volume)

            # print("Output shapes are: {}, {}".format(upsampled_volume.shape, skip_volume.shape))

            volume = torch.cat([upsampled_volume, skip_volume], axis=1)

            if self.preserve_shape:
                layer_spec = dict(kernel_size=matching_encoder_layer_spec['kernel_size'], stride=1)
                volume = self.pad_to_preserve_shape(volume, layer_spec)
                # print("Padding to preserve shape, shape of input volume after padding: {}".format(volume.shape))

            decoder_layer = self.decoder_layers[i]
            volume = decoder_layer(volume)

            is_last = i == len(self.skip_volume_indices)-1

            if not is_last:
                volume = F.relu(volume)

                if self.batch_norm:
                    volume = self.decoder_batch_norm_layers[i](volume)

            # print("Deconv output shapes is: {}".format(volume.shape))

            output_volumes.append(volume)

        out = output_volumes[-1][:, :, :inp.shape[1], :inp.shape[2]]
        embedding = encoder_volumes[-1]

        # Note that output_volumes is returned in order of increasing resolution

        return out, embedding, output_volumes

    @staticmethod
    def pad(volume, layer):
        """
        The effect of a conv: floor((n - k) / s) + 1
        Want n - k divisible by s. This corresponds to the "complete" case, where the filters
        exactly fill the input, and there are no input slots that get left out.
        Desired shape: ceil((n-k) / s) * s + k

        effect of Transpose Conv: (n-1) * s + k

        """
        s = layer.get('stride', 1)
        k = layer['kernel_size']
        spatial_shape = np.array(volume.shape[2:4]).astype('i')
        desired_shape = np.ceil((spatial_shape - k) / s).astype('i') * s + k
        padding = desired_shape - spatial_shape
        padded = F.pad(volume, (0, padding[1], 0, padding[0]))
        return padded

    @staticmethod
    def min_pad(v1, v2):
        s1 = min(v1.shape[2], v2.shape[2])
        s2 = min(v1.shape[3], v2.shape[3])

        v1 = v1[:, :, :s1, :s2]
        v2 = v2[:, :, :s1, :s2]

        return v1, v2

    @staticmethod
    def pad_to_preserve_shape(volume, layer_spec, balanced=True):
        """ Pad volume such that the shape after applying `layer_spec` is ceil(spatial_shape / stride) """
        s = layer_spec.get('stride', 1)
        k = layer_spec['kernel_size']
        spatial_shape = np.array(volume.shape[2:4]).astype('i')
        required_shape = (np.ceil((spatial_shape / s).astype('i')) - 1) * s + k
        padding = required_shape - spatial_shape

        if balanced:
            pre = np.floor(padding / 2).astype('i')
            post = np.ceil(padding / 2).astype('i')
            padded = F.pad(volume, (pre[1], post[1], pre[0], post[0],))
        else:
            padded = F.pad(volume, (0, padding[1], 0, padding[0]))

        return padded


def normal_kl(mean, std, prior_mean, prior_std):
    var = std**2
    prior_var = prior_std**2

    return 0.5 * (
        torch.log(prior_var) - torch.log(var)
        - 1.0 + var / prior_var
        + (mean - prior_mean)**2 / prior_var
    )


def normal_vae(mean, std, prior_mean, prior_std):
    sample = mean + torch.randn(mean.shape, device=std.device) * std
    kl = normal_kl(mean, std, prior_mean, prior_std)
    return sample, kl


def build_cam2world(angle, t, do_correction=False):
    """ Angle specified as Tait-Bryan Angles (basically Euler angles) with extrinsic order 'xyz'.

    roll: counterclockwise rotation of gamma about x axis.
    pitch: counterclockwise rotation of beta about y axis.
    yaw: counterclockwise rotation of alpha about z axis.

    world_point = (R_yaw * R_pitch * R_roll) cam_point

    If `do_correction` is true, we assume that in the camera coordinate frame,
    z increases into the frame, y increases downward, x increases rightward (looking out from the camera),
    and therefore we do an initial correction rotation to get a coordinate system where
    x increases into the frame, y increases leftward, z increases upward.

    """
    leading_dims = angle.shape[:-1]
    angle = angle.view(-1, angle.shape[-1])
    t = t.view(-1, t.shape[-1])

    device = angle.device
    so3_a = np.array([
        [0, -1, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]).astype('f')
    so3_a = torch.from_numpy(so3_a).to(device)

    so3_b = np.array([
        [0, 0, 1, 0, 0, 0, -1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0]
    ]).astype('f')
    so3_b = torch.from_numpy(so3_b).to(device)

    so3_y = np.array([
        [0, 0, 0, 0, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ]).astype('f')
    so3_y = torch.from_numpy(so3_y).to(device)

    sin = torch.sin(angle)
    cos = torch.cos(angle)
    v = torch.stack([sin, cos, torch.ones_like(sin)], axis=2)

    soa = torch.matmul(v[:, 0], so3_a)
    soa = torch.reshape(soa, (-1, 3, 3))

    sob = torch.matmul(v[:, 1], so3_b)
    sob = torch.reshape(sob, (-1, 3, 3))

    soy = torch.matmul(v[:, 2], so3_y)
    soy = torch.reshape(soy, (-1, 3, 3))

    so3 = torch.matmul(soa, torch.matmul(sob, soy))

    if do_correction:
        # rotate pi/2 CW around positive x axis, then pi/2 CW around positive z axis (intrinsically).
        correction = np.array([
            [0., 0., 1.],
            [-1., 0., 0.],
            [0., -1., 0.],
        ]).astype('f')
        correction = torch.from_numpy(correction).to(device)
        correction = correction[None, :, :].repeat(so3.shape[0], 1, 1)

        so3 = torch.matmul(so3, correction)

    mat = torch.cat([so3, t[:, :, None]], dim=2)

    b = sin.shape[0]
    row = torch.FloatTensor([0., 0., 0., 1.]).to(device)[None, None, :].repeat(b, 1, 1)
    mat = torch.cat([mat, row], dim=1)

    mat = mat.view(*leading_dims, 4, 4)

    return mat


def torch_mean_sum(x, n_mean=1):
    """ Average over batch dim, sum over all other dims. """
    _sum = x.sum(dim=tuple(range(n_mean, x.ndim)))
    return _sum.mean()


def transformation_matrix_to_pose(T):
    """ T: torch.Tensor(b, 4, 4) or (b, 3, 4) """
    R = T[:, :3]
    t = T[:, :3, 3]
    axis_angle = rotation_matrix_to_angle_axis(R)
    return axis_angle, t


# Taken from torchgeometry.


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1

    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k

    return angle_axis


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4
