from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
import pprint
import copy
from tabulate import tabulate

from dps.utils.base import RenderHook as _RenderHook, AttrDict, Config, map_structure, Param, Parameterized, pformat
from dps.utils.base import describe_tensor as _describe_tensor, describe_structure as _describe_structure


def compute_ssim(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    x = F.pad(x, (1,)*4)
    y = F.pad(y, (1,)*4)

    avg_pool2d = lambda t: F.avg_pool2d(t, 3, 1)

    mu_x = avg_pool2d(x)
    mu_y = avg_pool2d(y)

    sigma_x = avg_pool2d(x ** 2) - mu_x ** 2
    sigma_y = avg_pool2d(y ** 2) - mu_y ** 2
    sigma_xy = avg_pool2d(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def describe_tensor(tensor):
    tensor = to_np(tensor)
    _describe_tensor(tensor)


def describe_structure(structure):
    structure = map_structure(
        lambda t: to_np(t) if isinstance(t, (torch.Tensor, np.ndarray)) else t,
        structure, is_leaf=lambda t: not isinstance(t, dict)
    )
    _describe_structure(structure)


def repeat_along_axis(a, dim, n_repeats):
    """ Repeats in a different order than `repeat`.

        repeat_along_axis(arange(3), dim=0, n_repeats=2) -> (0, 0, 1, 1, 2, 2)

        whereas

        arange(3).repeat(2) -> (0, 1, 2, 0, 1, 2)

    """
    repeats = [1] * (a.ndim+1)
    repeats[dim+1] = n_repeats
    new_shape = list(a.shape)
    new_shape[dim] *= n_repeats
    return a.unsqueeze(dim+1).repeat(repeats).reshape(new_shape)


def reshape_and_apply(func, *signals, n_batch_dims, restore_shape=True, **func_kwargs):
    """
    permute so that batch dims are at the front, reshape to have a single batch dim, apply function,
    then restore shape to outputs.

    Assumes all elements of `signals` have the same batch dims.

    n_batch_dims: int or length-2 tuple
        If int, then gives the number of batch dims at the front of the shape. If tuple, then first element
        gives number of batch dims at front, second gives number of batch dims at the back.

    """
    try:
        n_leading_batch_dims, n_trailing_batch_dims = tuple(n_batch_dims)
    except Exception:
        n_leading_batch_dims = n_batch_dims
        n_trailing_batch_dims = 0

    n_leading_batch_dims = int(n_leading_batch_dims)
    n_trailing_batch_dims = int(n_trailing_batch_dims)
    n_batch_dims = n_leading_batch_dims + n_trailing_batch_dims

    _signals = []

    for signal in signals:
        assert 0 < n_batch_dims < signal.ndim

        dims = list(range(signal.ndim))

        leading_batch_dims = dims[:n_leading_batch_dims]

        trailing_idx = len(dims)-n_trailing_batch_dims
        trailing_batch_dims = dims[trailing_idx:]
        other_dims = dims[n_leading_batch_dims:trailing_idx]

        perm = leading_batch_dims + trailing_batch_dims + other_dims

        leading_batch_shape = signal.shape[:n_leading_batch_dims]
        trailing_batch_shape = signal.shape[trailing_idx:]
        other_shape = signal.shape[n_leading_batch_dims:trailing_idx]

        batch_dim = int(np.prod(leading_batch_shape) * np.prod(trailing_batch_shape))
        batch_shape = leading_batch_shape + trailing_batch_shape

        signal = signal.permute(perm).reshape(batch_dim, *other_shape)

        _signals.append(signal)

    outputs = func(*_signals, **func_kwargs)

    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
        is_iter = False
    else:
        outputs = list(outputs)
        is_iter = True

    if restore_shape:
        _outputs = []
        for o in outputs:
            o_shape = o.shape[1:]
            o = o.reshape(*batch_shape, *o_shape)

            if n_trailing_batch_dims:
                perm = list(range(o.ndim))
                perm = (
                    perm[:n_leading_batch_dims]
                    + perm[n_batch_dims:]
                    + perm[n_leading_batch_dims:n_batch_dims]
                )
                o = o.permute(*perm)

            _outputs.append(o)

        outputs = _outputs

    if is_iter:
        return outputs
    else:
        return outputs[0]


def to_np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return map_structure(to_np, tensor, is_leaf=lambda t: isinstance(t, (np.ndarray, torch.Tensor)))


def walk_variable_scopes(model, max_depth=None):
    def _fmt(i):
        return "{:,}".format(i)

    n_fixed = defaultdict(int)
    n_trainable = defaultdict(int)
    shapes = {}

    for name, v in model.named_parameters():
        n_variables = int(np.prod(tuple(v.size())))

        if v.requires_grad:
            n_fixed[""] += 0
            n_trainable[""] += n_variables
        else:
            n_fixed[""] += n_variables
            n_trainable[""] += 0

        shapes[name] = tuple(v.shape)

        name_so_far = ""

        for token in name.split("."):
            name_so_far += token

            if v.requires_grad:
                n_fixed[name_so_far] += 0
                n_trainable[name_so_far] += n_variables
            else:
                n_fixed[name_so_far] += n_variables
                n_trainable[name_so_far] += 0

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
    N = None

    def __init__(self, n_render=None, **kwargs):
        super().__init__(n_render=n_render, **kwargs)

    @staticmethod
    def process_data(data, n_render):
        return map_structure(
            lambda t: t[:n_render] if isinstance(t, torch.Tensor) else t,
            data, is_leaf=lambda t: not isinstance(t, (dict, list, tuple, set)))

    def get_tensors(self, updater, train_mode=False, train_data=False, data_iterator=None):
        if train_mode:
            updater.model.train()
        else:
            updater.model.eval()

        if data_iterator is None:
            if train_data:
                data_iterator = updater.train_iterator
            else:
                data_iterator = updater.data_manager.do_val()

        step = updater._n_experiences

        n_collected = 0
        _tensors = []
        _data = []

        with torch.no_grad():
            while True:
                data = AttrDict(next(data_iterator))
                tensors, recorded_tensors, losses = updater.model(data, step)

                tensors = Config(tensors)
                tensors = map_structure(
                    lambda t: to_np(t) if isinstance(t, torch.Tensor) else t,
                    tensors, is_leaf=lambda rec: not isinstance(rec, dict))

                data = Config(data)
                data = map_structure(
                    lambda t: to_np(t) if isinstance(t, torch.Tensor) else t,
                    data, is_leaf=lambda rec: not isinstance(rec, dict))

                _tensors.append(tensors)
                _data.append(data)

                n_collected += recorded_tensors['batch_size']
                if self.n_render is None or n_collected >= self.n_render:
                    break

        _tensors = map_structure(
            lambda *t: np.concatenate(t, axis=0)[:self.n_render],
            *_tensors, is_leaf=lambda rec: not isinstance(rec, dict))
        _data = map_structure(
            lambda *t: np.concatenate(t, axis=0)[:self.n_render],
            *_data, is_leaf=lambda rec: not isinstance(rec, dict))

        return _tensors, _data


def scheduled_value(obj, step):
    if callable(obj):
        return obj(step)
    else:
        return obj


class ParameterizedModule(torch.nn.Module, Parameterized):
    _path = "root"
    step = 0

    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        Parameterized.__init__(self, **kwargs)

        self.scheduled_values = dict()
        self.current_scheduled_values = dict()

        print(
            "\nBuilding pytorch module {} with args:\n{}".format(
                self.__class__.__name__, pformat(self._params_at_creation_time)))

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def update_global_step(self, step):
        ParameterizedModule.step = step

    def build_scheduled_value(self, name, value=None):
        value = value or getattr(self, name)
        if callable(value):
            self.scheduled_values[name] = value
        else:
            setattr(self, name, value)

    def get_scheduled_values(self):
        scheduled_values = {}
        for name, module in self.named_modules():
            sv = getattr(module, 'current_scheduled_values', {})
            for k, v in sv.items():
                if not name:
                    name = "root"
                scheduled_values[name + '.' + k] = v
        return scheduled_values

    def forward(self, *args, **kwargs):
        for name, value in self.scheduled_values.items():
            _value = value(ParameterizedModule.step)
            setattr(self, name, _value)
            self.current_scheduled_values[name] = _value

        return self._forward(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        raise Exception("NotImplementedError")

    def extra_repr(self):
        return "{}\n".format(pformat(self.param_values()))


def _init_recurrent_weights(module):
    for m in module.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


class CellWrapper(ParameterizedModule):
    """ Combine a recurrent cell with a learnable initial state, and packs hidden state such that all state is stored
        in a single tensor. """
    cell_class = Param()
    cell_kwargs = Param(None)

    def __init__(self, input_size, hidden_size, *args, **kwargs):
        super().__init__(**kwargs)
        cell_kwargs = self.cell_kwargs or {}

        self.cell = self.cell_class(input_size, hidden_size, *args, **cell_kwargs)

        dummy_input = torch.zeros((1, input_size))
        dummy_state = self.cell(dummy_input)

        if isinstance(dummy_state, torch.Tensor):
            self.state_is_tuple = False
            dummy_state = (dummy_state,)
        else:
            self.state_is_tuple = True

        self.state_component_sizes = [d.shape[-1] for d in dummy_state]

        dummy_state = torch.cat(dummy_state, dim=-1)

        state_shape = dummy_state.shape[1:]

        self._initial_state = torch.nn.Parameter(torch.zeros(state_shape), requires_grad=False)
        # self._initial_state = torch.nn.Parameter(torch.zeros(state_shape), requires_grad=True)

        self.init_weights()

    def initial_state(self, b):
        return repeat_along_axis(self._initial_state.clone()[None, ...], dim=0, n_repeats=b)

    def forward(self, inp, hidden=None):
        if hidden is None:
            hidden = self.initial_hidden

        if self.state_is_tuple:
            hidden = torch.split(hidden, self.state_component_sizes, dim=-1)

        return self.cell(inp, hidden)

    def init_weights(self):
        # nn.init.xavier_uniform_(self._initial_state)

        self.apply(_init_recurrent_weights)

        # --- forget_gate_init ---

        for name, parameter in self.named_parameters():
            if "bias" not in name:
                continue
            n = parameter.size(0)
            start, end = n // 4, n // 2
            parameter.data[start:end].fill_(1.)


class ConvNet(ParameterizedModule):
    layer_specs = Param()
    preserve_shape = Param(False)
    batch_norm = Param(False)

    def __init__(self, input_shape, output_size=None, **kwargs):
        super().__init__(**kwargs)

        try:
            input_shape = tuple(input_shape)
            input_n_filters, *spatial_shape = input_shape
            assert len(spatial_shape) == 2
        except Exception:
            input_n_filters = input_shape
            spatial_shape = None

        self.module_list = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleDict()

        print("Spatial shape: {}".format(spatial_shape))
        prev_n_filters = input_n_filters
        prev_n_units = None

        for i, layer_spec in enumerate(self.layer_specs):
            kind = layer_spec.get('kind', 'conv')

            is_last = i == len(self.layer_specs)-1

            if kind == 'conv':
                n_filters = layer_spec['n_filters']
                kernel_size = layer_spec['kernel_size']
                stride = layer_spec.get('stride', 1)

                if is_last and output_size is not None:
                    n_filters = output_size
                elif n_filters is None:
                    n_filters = prev_n_filters

                layer = torch.nn.Conv2d(prev_n_filters, n_filters, kernel_size, stride=stride)
                self.module_list.append(layer)

                if not is_last and layer_spec.get('batch_norm', self.batch_norm):
                    bn = torch.nn.BatchNorm2d(n_filters)
                    self.batch_norms[str(i)] = bn

                prev_n_filters = n_filters

                if spatial_shape is not None:
                    if self.preserve_shape and stride == 1:
                        pass
                    else:
                        spatial_shape = self.conv_output_shape(spatial_shape, kernel_size, stride)

            elif kind == 'fc':
                if spatial_shape is not None:
                    prev_n_units = prev_n_filters * spatial_shape[0] * spatial_shape[1]
                else:
                    assert prev_n_units is not None
                spatial_shape = None

                n_units = layer_spec['n_units']

                if is_last and output_size is not None:
                    n_units = output_size

                layer = torch.nn.Linear(prev_n_units, n_units)
                self.module_list.append(layer)

                if not is_last and layer_spec.get('batch_norm', self.batch_norm):
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

            kind = layer_spec.get('kind', 'conv')

            if kind == 'fc':
                b = x.shape[0]
                x = x.view(b, -1)

            if kind == 'conv':
                stride = layer_spec.get('stride', 1)

                if self.preserve_shape and stride == 1:
                    kernel_size = layer_spec['kernel_size']
                    left_pad = int(np.floor((kernel_size-1) / 2))
                    right_pad = int(np.ceil((kernel_size-1) / 2))
                    x = torch.nn.functional.pad(x, (left_pad, right_pad, left_pad, right_pad))

            x = layer(x)

            bn_key = str(i)
            if bn_key in self.batch_norms:
                x = self.batch_norms[bn_key](x)

            if not is_last:
                nl_key = layer_spec.get('nl', 'relu')
                x = nonlinearities[nl_key](x)

        return x

    @staticmethod
    def conv_output_shape(h_w, kernel_size=1, stride=1):
        """ Utility function for computing output shapes of convolutions. """

        try:
            h_w = tuple(h_w)
            assert len(h_w) == 2
        except Exception:
            h_w = (h_w, h_w)

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        if type(stride) is not tuple:
            stride = (stride, stride)

        print(h_w, kernel_size, stride)

        h = (h_w[0] - (kernel_size[0] - 1) - 1) // stride[0] + 1
        w = (h_w[1] - (kernel_size[1] - 1) - 1) // stride[1] + 1

        return (h, w)


class GridConvNet(ConvNet):
    preserve_shape = False

    def __init__(self, input_shape, output_size=None, n_grid_dims=2, **kwargs):
        input_n_channels, *image_shape = input_shape

        super().__init__(input_shape, output_size=output_size, **kwargs)

        receptive_field_info, required_image_size, pre_padding, post_padding = (
            self.compute_receptive_field_info(image_shape, self.layer_specs))

        print("Receptive field info for GridConvNet")
        pprint.pprint(receptive_field_info)
        print("required_image_size: {}".format(required_image_size))
        print("pre_padding: {}".format(pre_padding))
        print("post_padding: {}".format(post_padding))

        self.pre_padding = pre_padding
        self.post_padding = post_padding

        self.padding = [d for s in reversed(list(zip(self.pre_padding, self.post_padding))) for d in s]

        self.layer_info = copy.deepcopy(receptive_field_info)

        # for li, v in zip(self.layer_info, self.volumes):
        #     li['volume_size'] = v.shape[1:-1]
        #     li['volume'] = v

    def forward(self, inp):
        padded_inp = F.pad(inp, self.padding)
        return super().forward(padded_inp)

    @staticmethod
    def compute_receptive_field_info(image_shape, layers):
        ndim = len(image_shape)
        image_shape = tuple(int(i) for i in image_shape)
        grid_cell_size = np.array((1,)*ndim)
        rf_size = np.array((1,)*ndim)
        info = []

        for layer in layers:
            kind = layer.get('kind', 'conv')
            assert kind == 'conv'

            kernel_size = np.array(layer['kernel_size']) * ([1] * ndim)
            stride = np.array(layer['stride']) * ([1] * ndim)

            rf_size = rf_size + (kernel_size-1) * grid_cell_size
            grid_cell_size = grid_cell_size * stride

            # scale wrt largest dimension
            normed_j = np.array(grid_cell_size) / np.array(image_shape).max()
            normed_r = np.array(rf_size) / np.array(image_shape).max()

            info.append(
                dict(
                    kernel_size=kernel_size,
                    stride=stride,
                    rf_size=rf_size,
                    grid_cell_size=grid_cell_size,
                    normed_rf_size=normed_r,
                    normed_grid_cell_size=normed_j
                )
            )

        n_grid_cells = np.ceil(image_shape / grid_cell_size).astype('i')
        required_image_size = rf_size + (n_grid_cells-1) * grid_cell_size
        pre_padding = np.floor(rf_size / 2 - grid_cell_size / 2).astype('i')
        post_padding = required_image_size - image_shape - pre_padding

        volume_dimensions = required_image_size

        for i, _info in enumerate(info):
            grid_offset = -pre_padding + _info['rf_size'] / 2 - _info['grid_cell_size'] / 2

            new_volume_dimensions = (volume_dimensions - _info['kernel_size']) / _info['stride'] + 1
            new_volume_dimensions = new_volume_dimensions.astype('i')

            _info.update(
                grid_offset=grid_offset,
                n_grid_cells=new_volume_dimensions,
                virtual_image_size=new_volume_dimensions*_info['grid_cell_size']
            )

            volume_dimensions = new_volume_dimensions

        assert (info[-1]['n_grid_cells'] == n_grid_cells).all()
        assert (np.abs(info[-1]['grid_offset']) <= 0.5).all()

        return info, required_image_size, pre_padding, post_padding


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

    def __init__(self, input_shape, output_size=None, **kwargs):
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

        super().__init__(input_shape, output_size=output_size, **kwargs)


class ConvTransposeNet(ParameterizedModule):
    n_fc_layers = Param()
    n_fc_units = Param()
    batch_norm = Param()
    conv_layer_specs = Param()
    nl = Param()

    def __init__(self, output_n_channels, input_n_features, output_image_shape, **kwargs):
        super().__init__(**kwargs)

        spatial_shape = output_image_shape

        print("Spatial shape: {}".format(spatial_shape))
        prev_n_channels = output_n_channels

        shapes_after_slice = [spatial_shape]
        target_shapes = []
        conv_layers = []
        batch_norm_layers = []

        # We have to start from the output image and work backwards to get the right shapes.

        for i, layer_spec in enumerate(reversed(self.conv_layer_specs)):
            is_last = i == 0  # i == 0 because of the reversal.

            n_filters = layer_spec['n_filters']
            kernel_size = layer_spec['kernel_size']
            stride = layer_spec.get('stride', 1)

            shape_after_slice = shapes_after_slice[-1]
            target_shape = self.get_target_shape(shape_after_slice, kernel_size, stride)
            target_shapes.append(target_shape)

            layer = torch.nn.ConvTranspose2d(n_filters, prev_n_channels, kernel_size, stride=stride)
            conv_layers.append(layer)

            if not is_last and self.batch_norm:
                bn = torch.nn.BatchNorm2d(prev_n_channels)
                batch_norm_layers.append(bn)

            prev_n_channels = n_filters

            new_spatial_shape = self.conv_transpose_input_shape(target_shape, kernel_size, stride)
            shapes_after_slice.append(new_spatial_shape)

            print("ConvTranspose layer {}: shapes_after_slice: {}, output_shape: {}, input_shape: {}".format(
                i, shape_after_slice, target_shape, new_spatial_shape))

        self.conv_layers = torch.nn.ModuleList(list(reversed(conv_layers)))
        self.batch_norm_layers = torch.nn.ModuleList(list(reversed(batch_norm_layers)))
        self.shapes_after_slice = list(reversed(shapes_after_slice[:-1]))
        self.conv_input_shape = (n_filters, *new_spatial_shape)

        conv_input_size = int(np.prod(self.conv_input_shape))

        self.fully_connected = MLP(
            input_n_features, conv_input_size,
            n_hidden_units=[self.n_fc_units] * self.n_fc_layers,
            batch_norm=self.batch_norm,
            nl=self.nl,
        )

    def forward(self, x):
        b = x.shape[0]

        fc_output = self.fully_connected(x)
        volume = fc_output.reshape(b, *self.conv_input_shape)

        for i, (layer, layer_spec) in enumerate(zip(self.conv_layers, self.conv_layer_specs)):
            volume = layer(volume)
            shape_after_slice = self.shapes_after_slice[i]

            volume = volume[..., :shape_after_slice[0], :shape_after_slice[1]]

            is_last = i == len(self.conv_layers) - 1

            if not is_last and self.batch_norm:
                volume = self.batch_norm_layers[i](volume)

            if not is_last:
                volume = nonlinearities[self.nl](volume)

        return volume

    @staticmethod
    def conv_transpose_input_shape(output_shape, kernel_size=1, stride=1):
        """ Utility function for computing input shapes given desired output shape. """

        h, w = output_shape

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        if type(stride) is not tuple:
            stride = (stride, stride)

        H = (h - kernel_size[0]) // stride[0] + 1
        W = (w - kernel_size[1]) // stride[1] + 1

        return (H, W)

    @staticmethod
    def get_target_shape(shape, kernel_size, stride):
        """ Get a shape larger than the given shape that can be mapped to via a ConvTranspose with the given params. """
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        if type(stride) is not tuple:
            stride = (stride, stride)

        h, w = shape
        H = int(np.ceil((h - kernel_size[0]) / stride[0])) * stride[0] + kernel_size[0]
        W = int(np.ceil((h - kernel_size[1]) / stride[1])) * stride[1] + kernel_size[1]

        return (H, W)


class SimpleConvTransposeNet(ConvTransposeNet):
    """ A standard ConvNet that ends with a series of fully connected layers. """

    n_layers_in_between = Param()
    n_conv_blocks = Param()
    stride = Param()
    base_n_filters = Param()
    max_n_filters = Param()
    kernel_size = Param()

    conv_layer_specs = None

    def __init__(self, output_n_channels, input_n_features, output_image_shape, **kwargs):
        conv_layer_specs = []

        def add_conv_layer(n_filters, stride=1):
            nonlocal conv_layer_specs
            n_filters = min(n_filters, self.max_n_filters)
            conv_layer_specs.append(
                dict(n_filters=n_filters, stride=stride, kernel_size=self.kernel_size)
            )

        for j in range(self.n_layers_in_between):
            add_conv_layer(self.base_n_filters)

        for i in range(1, self.n_conv_blocks+1):
            n_filters = (self.stride**i) * self.base_n_filters

            add_conv_layer(n_filters, stride=self.stride)

            for j in range(self.n_layers_in_between):
                add_conv_layer(n_filters)

        self.conv_layer_specs = list(reversed(conv_layer_specs))
        pprint.pprint(self.conv_layer_specs)

        super().__init__(output_n_channels, input_n_features, output_image_shape, **kwargs)


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
    nl = Param('relu')
    batch_norm = Param(False)
    layer_norm = Param(False)

    def __init__(self, n_inputs, n_outputs, **kwargs):
        super().__init__(**kwargs)

        try:
            n_inputs = tuple(n_inputs)
            n_inputs = int(np.prod(n_inputs))
        except Exception:
            pass

        try:
            n_outputs = tuple(n_outputs)
            n_outputs = int(np.prod(n_outputs))
        except Exception:
            pass

        assert not (self.batch_norm and self.layer_norm)

        self.module_list = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        self.n_inputs = n_inputs
        prev_n_units = n_inputs

        for n_units in self.n_hidden_units:
            self.module_list.append(torch.nn.Linear(prev_n_units, n_units))
            prev_n_units = n_units

            if self.batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(n_units))

            if self.layer_norm:
                self.layer_norms.append(LayerNorm(n_units))

        self.module_list.append(torch.nn.Linear(prev_n_units, n_outputs))

    def forward(self, x, flatten=False):
        b = x.shape[0]
        x = x.reshape(b, -1)

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


class SpatialAttentionLayer(ParameterizedModule):
    """ For the input we are given data and an array of locations. For the output we are just given an array of locations.

    Kind of interesting: this can be viewed as a differentiable way of converting a sparse matrix representation
    of an image to a dense representation of an image, assuming the output locations are the locations of image pixels.
    Input data is a list of locations paired with data, conceptually similar to sparse matrix representations.

    """
    kernel_std = Param()
    build_mlp = Param()

    query_dim = Param()
    ref_dim = Param()
    loc_dim = Param()
    n_hidden = Param()

    process_queries = Param()
    layer_norm = Param()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.process_queries:
            self.query_func = self.build_mlp(self.query_dim, self.n_hidden)

        n_in = self.ref_dim + self.loc_dim
        if self.process_queries:
            n_in += self.n_hidden
        self.relation_func = self.build_mlp(n_in, self.n_hidden)

        self.final_func = self.build_mlp(self.n_hidden, self.n_hidden)

        if self.layer_norm:
            self.layer_norm_1 = torch.nn.LayerNorm(self.n_hidden)
            self.layer_norm_2 = torch.nn.LayerNorm(self.n_hidden)

    def _forward(self, reference_locs, reference_features, query_locs, query_features, reference_mask=None):
        """
        reference_locs: (B, loc_dim, n_ref)
        reference_features: (B, n_hidden, n_ref)
        query_locs: (B, loc_dim, n_query)
        query_features: (B, n_hidden, n_query) or None

        reference_mask: (B, n_ref) (optional)

        Returns
        -------
        output: (B, n_hidden, n_query)
        attention_weights: (B, n_ref, n_query)

        """
        reference_locs = reference_locs.permute(0, 2, 1)
        reference_features = reference_features.permute(0, 2, 1)
        query_locs = query_locs.permute(0, 2, 1)

        if query_features is not None:
            query_features = query_features.permute(0, 2, 1)

        assert (query_features is not None) == self.process_queries

        n_query = query_locs.shape[-2]
        b, n_ref, _ = reference_features.shape

        # --- process queries, if we have them ---

        if self.process_queries:
            processed_queries = reshape_and_apply(
                self.query_func, query_features, n_batch_dims=2)  # (B, n_query, n_hidden)
        else:
            processed_queries = None

        # --- process all query-reference pairs, taking relative position into account ---

        adjusted_locs = reference_locs[:, None, :, :] - query_locs[:, :, None, :]  # (B, n_query, n_ref, loc_dim)
        adjusted_features = reference_features[:, None].repeat(1, n_query, 1, 1)  # (B, n_query, n_ref, ref_dim)
        relation_input = torch.cat([adjusted_features, adjusted_locs], axis=3)

        if self.process_queries:
            _processed_queries = processed_queries[:, :, None].repeat(1, 1, n_ref, 1)  # (b, n_query, n_ref, n_hidden)
            relation_input = torch.cat([relation_input, _processed_queries], dim=3)

        V = reshape_and_apply(self.relation_func, relation_input, n_batch_dims=3)  # (B, n_query, n_ref, n_hidden)

        # --- for each query, get a set of attention weights over the references, based on spatial proximity ---

        attention_weights = torch.exp(-0.5 * ((adjusted_locs / self.kernel_std)**2).sum(dim=3))

        # TODO: Uncomment this to have the attention weights be normalized over space.
        # attention_weights = (
        #     attention_weights / (2 * np.pi) ** (self.loc_dim / 2) / self.kernel_std**self.loc_dim
        # )  # (B, n_query, n_ref)

        if reference_mask is not None:
            # Assign 0 attention weight to references that have a 0 in reference_mask
            attention_weights = attention_weights * reference_mask[:, None, :].float()

        # --- apply attention weights ---

        result = (V * attention_weights[..., None]).sum(dim=2)  # (B, n_query, n_hidden)

        # --- finish up ---

        if self.process_queries:
            result += processed_queries

        if self.layer_norm:
            result = self.layer_norm_1(result)

        final_result = reshape_and_apply(self.final_func, result, n_batch_dims=2)

        if self.layer_norm:
            result = self.layer_norm_2(result + final_result)
        else:
            result = result + final_result

        return result.permute(0, 2, 1), attention_weights.permute(0, 2, 1)


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


def build_cam2world(yaw_pitch_roll, t, do_correction=False):
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
    leading_dims = yaw_pitch_roll.shape[:-1]
    yaw_pitch_roll = yaw_pitch_roll.view(-1, yaw_pitch_roll.shape[-1])
    t = t.view(-1, t.shape[-1])

    device = yaw_pitch_roll.device
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

    sin = torch.sin(yaw_pitch_roll)
    cos = torch.cos(yaw_pitch_roll)
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
    sum_dims = tuple(range(n_mean, x.ndim))

    if sum_dims:
        x = x.sum(sum_dims)

    return x.mean()


def transformation_matrix_to_pose(T, angle_format):
    """ T: torch.Tensor(b, 4, 4) or (b, 3, 4) """
    *leading_dims, h, w = T.shape
    T = T.reshape(-1, h, w)

    R = T[:, :3]
    if angle_format == 'axis_angle':
        rotation = rotation_matrix_to_angle_axis(R)
    else:
        raise Exception("Angle format {} not implemented.".format(angle_format))

    position = T[:, :3, 3]

    rotation = rotation.reshape(*leading_dims, 3)
    position = position.reshape(*leading_dims, 3)

    return rotation, position


def interpolate_2d(v, resize_to, chw=True):
    if not chw:
        ndim = len(v.shape)
        v = v.permute(*list(range(ndim-3)), ndim-1, ndim-3, ndim-2)

    *leading_shape, c, h, w = v.shape
    v = v.reshape(-1, c, h, w)
    v = F.interpolate(v, resize_to, mode='bilinear', align_corners=False)
    v = v.reshape(*leading_shape, c, *resize_to)

    if not chw:
        ndim = len(v.shape)
        v = v.permute(*list(range(ndim-3)), ndim-2, ndim-1, ndim-3)

    return v


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
