import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from sklearn.cluster import k_means
import collections

from dps import cfg
from dps.datasets import EMNIST_ObjectDetection
from dps.updater import Updater
from dps.utils import Config, Param, Parameterized, square_subplots, prime_factors
from dps.utils.tf import (
    FullyConvolutional, build_gradient_train_op, trainable_variables,
    tf_mean_sum, build_scheduled_value)
from dps.env.advanced.yolo import mAP
from dps.tf_ops import render_sprites
from dps.utils.tf import MLP

tf_flatten = tf.layers.flatten


class Env(object):
    def __init__(self):
        train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


def get_updater(env):
    network = YoloRL_Network(env)
    return YoloRL_Updater(env, network)


class Backbone(FullyConvolutional):
    pixels_per_cell = Param()
    kernel_size = Param()
    n_channels = Param()
    n_final_layers = Param(2)

    def __init__(self, **kwargs):
        sh = sorted(prime_factors(self.pixels_per_cell[0]))
        sw = sorted(prime_factors(self.pixels_per_cell[1]))
        assert max(sh) <= 4
        assert max(sw) <= 4

        if len(sh) < len(sw):
            sh = sh + [1] * (len(sw) - len(sh))
        elif len(sw) < len(sh):
            sw = sw + [1] * (len(sh) - len(sw))

        layout = [dict(filters=self.n_channels, kernel_size=4, strides=(_sh, _sw), padding="RIGHT_ONLY")
                  for _sh, _sw in zip(sh, sw)]

        # These layers don't change the shape
        layout += [
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME")
            for i in range(self.n_final_layers)]

        super(Backbone, self).__init__(layout, check_output_shape=True, **kwargs)


class NextStep(FullyConvolutional):
    kernel_size = Param()
    n_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME"),
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME"),
        ]
        super(NextStep, self).__init__(layout, check_output_shape=True, **kwargs)


class ObjectDecoder(FullyConvolutional):
    n_decoder_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=4, kernel_size=4, strides=1, padding="SAME", transpose=True),  # For 14 x 14 output
        ]
        super(ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)


def reconstruction_cost(_tensors, updater):
    pp_reconstruction_loss = _tensors['per_pixel_reconstruction_loss']
    batch_size = pp_reconstruction_loss.shape[0]
    pp_reconstruction_loss = pp_reconstruction_loss.reshape(batch_size, -1)
    reconstruction_loss = np.sum(pp_reconstruction_loss, axis=1)
    return reconstruction_loss.reshape(batch_size, 1, 1, 1, 1)


reconstruction_cost.keys_accessed = "per_pixel_reconstruction_loss"


def local_reconstruction_cost(_tensors, network):
    centers_h = (np.arange(network.H) + 0.5) * network.pixels_per_cell[0]
    centers_w = (np.arange(network.W) + 0.5) * network.pixels_per_cell[1]

    loc_h = (np.arange(network.image_height) + 0.5)[..., None]
    loc_w = (np.arange(network.image_width) + 0.5)[..., None]

    dist_h = np.abs(loc_h - centers_h)
    dist_w = np.abs(loc_w - centers_w)

    all_filtered = []

    loss = _tensors['per_pixel_reconstruction_loss']

    for b in range(network.B):
        max_distance_h = network.pixels_per_cell[0] / 2 + network.max_hw * network.anchor_boxes[b, 0] / 2
        max_distance_w = network.pixels_per_cell[1] / 2 + network.max_hw * network.anchor_boxes[b, 1] / 2

        # Rectangle filtering
        filt_h = (dist_h < max_distance_h).astype('f')
        filt_w = (dist_w < max_distance_w).astype('f')

        # Sum over channel dimension
        signal = loss.sum(axis=-1)

        signal = np.dot(signal, filt_w)
        signal = np.tensordot(signal, filt_h, [1, 0])
        signal = np.transpose(signal, (0, 2, 1))

        all_filtered.append(signal)

    return np.stack(all_filtered, axis=-1)[..., None]


local_reconstruction_cost.keys_accessed = "per_pixel_reconstruction_loss"


def _local_filter(signal, neighbourhood_size):
    if neighbourhood_size == 0:
        return signal

    _, H, W, B, _ = signal.shape

    if neighbourhood_size is None:
        neighbourhood_size = max(H, W)

    dist_h = np.abs(np.arange(H) - np.arange(H)[..., None])
    dist_w = np.abs(np.arange(W) - np.arange(W)[..., None])

    # Rectangle filtering
    filt_h = (dist_h <= neighbourhood_size).astype('f')
    filt_w = (dist_w <= neighbourhood_size).astype('f')

    # Sum over box and channel dimensions
    signal = signal.sum(axis=(3, 4))

    signal = np.dot(signal, filt_w)
    signal = np.tensordot(signal, filt_h, [1, 0])
    signal = np.transpose(signal, (0, 2, 1))

    return signal[..., None, None]


class AreaCost(object):
    keys_accessed = "program:obj area"

    def __init__(self, target_area, neighbourhood_size):
        self.target_area = target_area
        self.neighbourhood_size = neighbourhood_size

    def __call__(self, _tensors, updater):
        selected_area_cost = _tensors['program']['obj'] * np.abs(_tensors['area'] - self.target_area)
        return _local_filter(selected_area_cost, self.neighbourhood_size)


class NonzeroCost(object):
    keys_accessed = "program:obj"

    def __init__(self, neighbourhood_size=0):
        self.neighbourhood_size = neighbourhood_size

    def __call__(self, _tensors, network):
        return _local_filter(_tensors['program']['obj'], self.neighbourhood_size)


def negative_mAP_cost(_tensors, network):
    annotations = network._other["annotations"]
    if annotations is None:
        return np.zeros((1, 1, 1, 1, 1))
    else:
        gt = []
        pred_boxes = []

        obj = _tensors['program']['obj']
        top, left, height, width = np.split(_tensors['normalized_box'], 4, axis=-1)

        top = network.image_height * top
        height = network.image_height * height
        bottom = top + height

        left = network.image_width * left
        width = network.image_width * width
        right = left + width

        for idx, a in enumerate(annotations):
            _a = [[0, *rest] for cls, *rest in a]
            gt.append(_a)

            pb = []

            for i in range(network.H):
                for j in range(network.W):
                    for b in range(network.B):
                        o = obj[idx, i, j, b, 0]
                        if o > 0.0:
                            pb.append(
                                [0, o,
                                 top[idx, i, j, b, 0],
                                 bottom[idx, i, j, b, 0],
                                 left[idx, i, j, b, 0],
                                 right[idx, i, j, b, 0]])

            pred_boxes.append(pb)

        _map = mAP(pred_boxes, gt, 1)
        _map = _map.reshape(-1, 1, 1, 1, 1)
        return -_map


negative_mAP_cost.keys_accessed = "normalized_box program:obj"


def count_error_cost(_tensors, updater):
    annotations = updater._other["annotations"]
    if annotations is None:
        return np.zeros((1, 1, 1, 1, 1))
    else:
        obj = _tensors["program"]["obj"]
        n_objects_pred = obj.reshape(obj.shape[0], -1).sum(axis=1)
        n_objects_true = np.array([len(a) for a in annotations])
        error = (n_objects_pred != n_objects_true).astype('f')
        return error.reshape(-1, 1, 1, 1, 1)


count_error_cost.keys_accessed = "program:obj"


def count_1norm_cost(_tensors, updater):
    annotations = updater._other["annotations"]
    if annotations is None:
        return np.zeros((1, 1, 1, 1, 1))
    else:
        obj = _tensors["program"]["obj"]
        n_objects_pred = obj.reshape(obj.shape[0], -1).sum(axis=1)
        n_objects_true = np.array([len(a) for a in annotations])
        dist = np.abs(n_objects_pred - n_objects_true).astype('f')
        return dist.reshape(-1, 1, 1, 1, 1)


count_1norm_cost.keys_accessed = "program:obj"


def build_xent_loss(logits, targets):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)


def build_squared_loss(logits, targets):
    actions = tf.sigmoid(logits)
    return (actions - targets)**2


def build_1norm_loss(logits, targets):
    actions = tf.sigmoid(logits)
    return tf.abs(actions - targets)


loss_builders = {
    'xent': build_xent_loss,
    'squared': build_squared_loss,
    '1norm': build_1norm_loss,
}


class YoloRL_Network(Parameterized):
    pixels_per_cell = Param()
    A = Param(help="Dimension of attribute vector.")
    anchor_boxes = Param(help="List of (h, w) pairs.")
    object_shape = Param()

    use_input_attention = Param()
    decoder_logit_scale = Param()

    max_hw = Param()
    min_hw = Param()

    box_std = Param()
    attr_std = Param()
    z_std = Param()
    obj_exploration = Param()
    obj_default = Param()

    n_passthrough_features = Param()

    use_baseline = Param()
    nonzero_weight = Param()
    area_weight = Param()

    local_reconstruction_cost = Param()
    area_neighbourhood_size = Param()
    nonzero_neighbourhood_size = Param()

    target_area = Param(0.)

    fixed_values = Param()
    fixed_weights = Param()
    order = Param()

    sequential_cfg = Param(dict(
        on=False,
        lookback_shape=(2, 2, 2),
    ))

    def __init__(self, env, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.H = int(np.ceil(self.image_height / self.pixels_per_cell[0]))
        self.W = int(np.ceil(self.image_width / self.pixels_per_cell[1]))
        self.B = len(self.anchor_boxes)

        self.anchor_boxes = np.array(self.anchor_boxes)

        self.COST_funcs = {}

        _reconstruction_cost_func = local_reconstruction_cost if self.local_reconstruction_cost else reconstruction_cost
        self.COST_funcs['reconstruction'] = (1, _reconstruction_cost_func, "both")

        _nonzero_cost_func = NonzeroCost(self.nonzero_neighbourhood_size)
        self.COST_funcs['nonzero'] = (self.nonzero_weight, _nonzero_cost_func, "obj")

        _area_cost_func = AreaCost(self.target_area, self.area_neighbourhood_size)
        self.COST_funcs['area'] = (self.area_weight, _area_cost_func, "obj")

        # For evaluation purposes only
        self.COST_funcs['negative_mAP'] = (0, negative_mAP_cost, "obj")
        self.COST_funcs['count_error'] = (0, count_error_cost, "obj")
        self.COST_funcs['count_1norm'] = (0, count_1norm_cost, "obj")

        for name, (_, func, _) in self.COST_funcs.items():
            assert isinstance(getattr(func, "keys_accessed"), str), name

        self.object_decoder = None

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        if isinstance(self.order, str):
            self.order = self.order.split()
        assert set(self.order) == set("box obj z attr".split())

        self.backbone = None
        self.layer_params = dict(
            box=dict(
                rep_builder=self._build_box,
                fixed="box" in self.fixed_weights,
                output_size=4,
                sample_size=4,
                network=None,
                sigmoid=True,
            ),
            obj=dict(
                rep_builder=self._build_obj,
                fixed="obj" in self.fixed_weights,
                output_size=1,
                sample_size=1,
                network=None,
                sigmoid=True,
            ),
            z=dict(
                rep_builder=self._build_z,
                fixed="z" in self.fixed_weights,
                output_size=1,
                sample_size=1,
                network=None,
                sigmoid=True,
            ),
            attr=dict(
                rep_builder=self._build_attr,
                fixed="attr" in self.fixed_weights,
                output_size=self.A,
                sample_size=self.A,
                network=None,
                sigmoid=False,
            ),
        )

    @property
    def inp(self):
        return self._tensors["inputs"]["inp"]

    @property
    def inp_ph(self):
        return self._tensors["inputs"]["inp_ph"]

    @property
    def batch_size(self):
        return self._tensors["batch_size"]

    @property
    def is_training(self):
        return self._tensors["inputs"]["is_training"]

    @property
    def float_is_training(self):
        return self._tensors["inputs"]["float_is_training"]

    @property
    def fixed_samples(self):
        return self._tensors["fixed_samples"]

    @property
    def float_fixed_samples(self):
        return self._tensors["float_fixed_samples"]

    def trainable_variables(self, for_opt):
        scoped_functions = (
            [self.object_decoder] +
            [self.layer_params[kind]["network"] for kind in self.order]
        )

        if self.sequential_cfg['on']:
            scoped_functions.append(self.backbone)

        tvars = []
        for sf in scoped_functions:
            tvars.extend(trainable_variables(sf.scope, for_opt=for_opt))

        if self.sequential_cfg['on']:
            tvars.append(self.edge_weights)

        return tvars

    def _process_cost(self, cost):
        assert cost.ndim == 5
        return cost * np.ones((1, self.H, self.W, self.B, 1))

    def _compute_routing(self, program):
        """ Compute a routing matrix based on sampled program, for use in program interpretation step.

        Returns
        -------
        max_objects: int
            Maximum number of objects in a single batch element.
        n_objects: (batch_size,) ndarray
            Number of objects in each batch element.
        routing: (batch_size, max_objects, 4) ndarray
            Routing array used as input to tf.gather_nd when interpreting program to form an image.

        """
        batch_size = program['obj'].shape[0]
        flat_obj = program['obj'].reshape(batch_size, -1)
        n_objects = np.sum(flat_obj, axis=1).astype('i')
        max_objects = n_objects.max()
        flat_z = program['z'].reshape(batch_size, -1)
        target_shape = program['attr'].shape[1:4]
        assert len(target_shape) == 3  # (batch_size, H, W, B)

        routing = np.zeros((batch_size, max_objects, 4))

        for b, fz in enumerate(flat_z):
            indexed = [(_fz, i) for (i, _fz) in enumerate(fz) if flat_obj[b, i] > 0]

            # Sort turned-on objects by increasing z value
            _sorted = sorted(indexed)

            for j, (_, i) in enumerate(_sorted):
                routing[b, j, :] = (b,) + np.unravel_index(i, target_shape)

        return max_objects, n_objects, routing

    def _update_feed_dict_with_routing(self, feed_dict):
        """ Generate program and compute routing matrix """

        feed_dict[self.fixed_samples] = False

        batch_size = feed_dict[self._tensors["inputs"]["inp_ph"]].shape[0]

        # Add dummy values for the input samples
        for input_sample in self.input_samples.values():
            feed_dict[input_sample] = np.zeros([batch_size] + [int(i) for i in input_sample.shape[1:]])

        sess = tf.get_default_session()
        program, samples = sess.run([self.program, self.samples], feed_dict=feed_dict)

        feed_dict.update({self.input_samples[k]: v for k, v in samples.items()})
        feed_dict[self.fixed_samples] = True

        max_objects, n_objects, routing = self._compute_routing(program)
        feed_dict[self._tensors["max_objects"]] = max_objects
        feed_dict[self._tensors["n_objects"]] = n_objects
        feed_dict[self._tensors["routing"]] = routing

    def _update_feed_dict_with_costs(self, feed_dict):
        """ Compute costs (negative rewards) once program has been generated and routing matrix computed. """

        sess = tf.get_default_session()

        fetch_keys = set()
        for _, f, _ in self.COST_funcs.values():
            for key in f.keys_accessed.split():
                fetch_keys.add(key)

        fetches = {}
        for key in list(fetch_keys):
            dst = fetches
            src = self._tensors
            subkeys = key.split(":")

            for i, _key in enumerate(subkeys):
                if i == len(subkeys)-1:
                    dst[_key] = src[_key]
                else:
                    if _key not in dst:
                        dst[_key] = dict()
                    dst = dst[_key]
                    src = src[_key]

        _tensors = sess.run(self._tensors, feed_dict=feed_dict)

        costs = {
            self.COST_tensors[name]: self._process_cost(f(_tensors, self))
            for name, (_, f, _) in self.COST_funcs.items()
        }
        feed_dict.update(costs)

    def update_feed_dict(self, feed_dict, other):
        self._other = other
        self._update_feed_dict_with_routing(feed_dict)
        self._update_feed_dict_with_costs(feed_dict)

    def _get_scheduled_value(self, name):
        scalar = self._tensors.get(name, None)
        if scalar is None:
            schedule = getattr(self, name)
            scalar = self._tensors[name] = build_scheduled_value(schedule, name)
        return scalar

    def _build_placeholders(self):
        self._tensors.update(
            max_objects=tf.placeholder(tf.int32, (), name="max_objects"),
            n_objects=tf.placeholder(tf.int32, (None,), name="n_objects"),
            routing=tf.placeholder(tf.int32, (None, None, 4), name="routing"),
            fixed_samples=tf.placeholder(tf.bool, ()),
            batch_size=tf.shape(self.inp)[0]
        )
        self._tensors["float_fixed_samples"] = tf.to_float(self._tensors["fixed_samples"])

        H, W, B = self.H, self.W, self.B

        self.COST_tensors = {}

        for name, _ in self.COST_funcs.items():
            cost = tf.placeholder(
                tf.float32, (None, H, W, B, 1), name="COST_{}_ph".format(name))
            self.COST_tensors[name] = cost

        self.input_samples = {}

        for kind, params in self.layer_params.items():
            self.input_samples[kind] = tf.placeholder(
                tf.float32, (None, H, W, B, params['sample_size']),
                name="{}_input_samples".format(kind))

    def _build_box(self, box_logits, input_samples, is_training):
        box = tf.nn.sigmoid(tf.clip_by_value(box_logits, -10., 10.))
        cell_y, cell_x, h, w = tf.split(box, 4, axis=-1)

        h = float(self.max_hw - self.min_hw) * h + self.min_hw
        w = float(self.max_hw - self.min_hw) * w + self.min_hw

        if "h" in self.fixed_values:
            h = float(self.fixed_values["h"]) * tf.ones_like(h, dtype=tf.float32)
        if "w" in self.fixed_values:
            w = float(self.fixed_values["w"]) * tf.ones_like(w, dtype=tf.float32)
        if "cell_y" in self.fixed_values:
            cell_y = float(self.fixed_values["cell_y"]) * tf.ones_like(cell_y, dtype=tf.float32)
        if "cell_x" in self.fixed_values:
            cell_x = float(self.fixed_values["cell_x"]) * tf.ones_like(cell_x, dtype=tf.float32)

        box = tf.concat([cell_y, cell_x, h, w], axis=-1)

        box_std = self._get_scheduled_value("box_std")

        box_noise = (
            self.float_fixed_samples * input_samples +
            (1 - self.float_fixed_samples) * tf.random_normal(tf.shape(box), name="box_noise")
        )

        noisy_box = box + box_noise * box_std * self.float_is_training

        return dict(
            _tensors=dict(
                cell_y=cell_y,
                cell_x=cell_x,
                h=h,
                w=w
            ),
            samples=box_noise,
            program=noisy_box,
        )

    def _build_obj(self, obj_logits, input_samples, is_training, **kwargs):
        obj_logits = tf.clip_by_value(obj_logits, -10., 10.)

        obj_params = tf.nn.sigmoid(obj_logits)

        obj_exploration = self.float_is_training * self._get_scheduled_value("obj_exploration")
        obj_default = self._get_scheduled_value("obj_default")

        obj_params = (1 - obj_exploration) * obj_params + obj_default * obj_exploration

        obj_dist = tf.distributions.Bernoulli(probs=obj_params)

        obj_samples = tf.stop_gradient(obj_dist.sample())
        obj_samples = tf.to_float(obj_samples)

        obj_samples = (
            self.float_fixed_samples * input_samples +
            (1 - self.float_fixed_samples) * obj_samples
        )

        obj_log_probs = obj_dist.log_prob(obj_samples)
        obj_log_probs = tf.where(tf.is_nan(obj_log_probs), -100.0 * tf.ones_like(obj_log_probs), obj_log_probs)

        obj_entropy = obj_dist.entropy()

        if "obj" in self.fixed_values:
            obj_samples = float(self.fixed_values["obj"]) * tf.ones_like(obj_samples, dtype=tf.float32)

        return dict(
            samples=obj_samples,
            entropy=obj_entropy,
            log_probs=obj_log_probs,
            logits=obj_logits,
            program=obj_samples
        )

    def _build_z(self, z_logits, input_samples, is_training):
        z_logits = tf.clip_by_value(z_logits, -10., 10.)

        z_params = tf.nn.sigmoid(z_logits)
        z_std = self.float_is_training * self._get_scheduled_value("z_std")

        z_dist = tf.distributions.Normal(loc=z_params, scale=z_std)

        z_samples = tf.stop_gradient(z_dist.sample())

        z_samples = (
            self.float_fixed_samples * input_samples +
            (1 - self.float_fixed_samples) * z_samples
        )

        z_log_probs = z_dist.log_prob(z_samples)
        z_log_probs = tf.where(tf.is_nan(z_log_probs), -100.0 * tf.ones_like(z_log_probs), z_log_probs)

        z_entropy = z_dist.entropy()

        if "z" in self.fixed_values:
            z_samples = float(self.fixed_values["z"]) * tf.ones_like(z_samples, dtype=tf.float32)

        return dict(
            samples=z_samples,
            entropy=z_entropy,
            log_probs=z_log_probs,
            logits=z_logits,
            program=z_samples
        )

    def _build_attr(self, attr, input_samples, is_training):
        attr_std = self._get_scheduled_value("attr_std")

        attr_noise = (
            self.float_fixed_samples * input_samples +
            (1 - self.float_fixed_samples) * tf.random_normal(tf.shape(attr), name="attr_noise")
        )

        noisy_attr = attr + attr_noise * attr_std * self.float_is_training

        return dict(
            samples=attr_noise,
            program=noisy_attr,
        )

    def _build_program_generator(self):
        H, W, B = self.H, self.W, self.B
        program, features = None, None

        for i, kind in enumerate(self.order):
            kind = self.order[i]
            params = self.layer_params[kind]
            rep_builder = params["rep_builder"]
            output_size = params["output_size"]
            network = params["network"]
            fixed = params["fixed"]

            final = kind == self.order[-1]

            out_channel_dim = B * output_size
            if not final:
                out_channel_dim += self.n_passthrough_features

            if network is None:
                builder = cfg.build_next_step if i else cfg.build_backbone

                network = builder(scope="{}_network".format(kind))
                network.layout[-1]['filters'] = out_channel_dim

                if fixed:
                    network.fix_variables()
                self.layer_params[kind]["network"] = network

            if i:
                layer_inp = features
                _program = tf.reshape(program, (-1, H, W, B * int(program.shape[-1])))
                layer_inp = tf.concat([features, _program], axis=-1)
            else:
                layer_inp = self.inp

            network_output = network(layer_inp, (H, W, out_channel_dim), self.is_training)

            features = network_output[..., B*output_size:]

            output = network_output[..., :B*output_size]
            output = tf.reshape(output, (-1, H, W, B, output_size))

            input_samples = self.input_samples[kind]
            built = rep_builder(output, input_samples, self.is_training)

            assert 'program' in built
            for info_type in self.info_types:
                if info_type in built:
                    self._tensors[info_type][kind] = built[info_type]

            if '_tensors' in built:
                self._tensors.update(built['_tensors'])

            if i:
                program = tf.concat([program, self.program[kind]], axis=-1)
            else:
                program = self.program[kind]

    def _get_sequential_input(self, backbone_output, program, h, w, b, edge_element):
        inp = [backbone_output[:, h, w, b, :]]
        for i in range(self.sequential_cfg.lookback_shape[0]):
            for j in range(self.sequential_cfg.lookback_shape[1]):
                for k in range(self.sequential_cfg.lookback_shape[2]):

                    if i == j == k == 0:
                        continue

                    _i = h - i
                    _j = w - j
                    _k = b - k

                    if _i < 0 or _j < 0 or _k < 0:
                        inp.append(edge_element)
                    else:
                        program_element = program[_i, _j, _k]
                        inp.append(program_element)

        return tf.concat(inp, axis=1)

    def _make_empty(self):
        return np.array([{} for i in range(self.H * self.W * self.B)]).reshape(self.H, self.W, self.B)

    def _build_program_generator_sequential(self):
        H, W, B = self.H, self.W, self.B

        if self.backbone is None:
            self.backbone = cfg.build_backbone(scope="backbone")
            self.backbone.layout[-1]['filters'] = B * self.n_passthrough_features

        inp = self._tensors["inputs"]["inp"]
        backbone_output = self.backbone(inp, (H, W, B*self.n_passthrough_features), self.is_training)
        backbone_output = tf.reshape(backbone_output, (-1, H, W, B, self.n_passthrough_features))

        total_output_size = sum(v["output_size"] for v in self.layer_params.values())

        self.edge_weights = tf.get_variable("edge_weights", shape=(1, total_output_size), dtype=tf.float32)
        sizes = [self.layer_params[kind]['output_size'] for kind in self.order]
        edge_weights = tf.split(self.edge_weights, sizes, axis=1)
        _edge_weights = []
        for ew, kind in zip(edge_weights, self.order):
            if self.layer_params[kind]['sigmoid']:
                ew = tf.nn.sigmoid(ew)
            _edge_weights.append(ew)
        edge_element = tf.concat(_edge_weights, axis=1)
        edge_element = tf.tile(edge_element, (self.batch_size, 1))

        program_info = {
            info_type: collections.defaultdict(self._make_empty)
            for info_type in self.info_types}

        _tensors = collections.defaultdict(self._make_empty)

        program = np.empty((H, W, B), dtype=np.object)

        for h in range(self.H):
            for w in range(self.W):
                for b in range(self.B):
                    layer_inp = self._get_sequential_input(backbone_output, program, h, w, b, edge_element)

                    hwb_program = []

                    for i, kind in enumerate(self.order):
                        kind = self.order[i]
                        params = self.layer_params[kind]
                        rep_builder = params["rep_builder"]
                        output_size = params["output_size"]
                        network = params["network"]
                        fixed = params["fixed"]

                        if network is None:
                            builder = self.sequential_cfg.build_next_step
                            network = builder(scope="{}_sequential_network".format(kind))
                            if fixed:
                                network.fix_variables()
                            params["network"] = network

                        output = network(layer_inp, output_size, self.is_training)
                        input_samples = self.input_samples[kind][:, h, w, b, :]
                        built = rep_builder(output, input_samples, self.is_training)

                        assert 'program' in built
                        for info_type, v in program_info.items():
                            if info_type in built:
                                v[kind][h, w, b] = built[info_type]

                        if '_tensors' in built:
                            for k, v in built['_tensors'].items():
                                _tensors[k][h, w, b] = v

                        program_for_next_step = built['program']
                        layer_inp = tf.concat([layer_inp, program_for_next_step], axis=1)

                        hwb_program.append(built['program'])

                    program[h, w, b] = tf.concat(hwb_program, axis=1)
                    assert program[h, w, b].shape[1] == total_output_size

        # Now stitch everything together into respectable tensors.
        for info_type, value in program_info.items():
            for kind, v in value.items():
                t1 = []
                for h in range(H):
                    t2 = []
                    for w in range(W):
                        t2.append(tf.stack([v[h, w, b] for b in range(B)], axis=1))
                    t1.append(tf.stack(t2, axis=1))

                self._tensors[info_type][kind] = tf.stack(t1, axis=1)

        for k, v in _tensors.items():
            t1 = []
            for h in range(H):
                t2 = []
                for w in range(W):
                    t2.append(tf.stack([v[h, w, b] for b in range(B)], axis=1))
                t1.append(tf.stack(t2, axis=1))

            self._tensors[k] = tf.stack(t1, axis=1)

    def _build_program_interpreter(self):
        # --- Compute sprites from attrs using object decoder ---

        attrs = self.program['attr']

        routed_attrs = tf.gather_nd(attrs, self._tensors["routing"])
        object_decoder_in = tf.reshape(routed_attrs, (-1, 1, 1, self.A))

        object_logits = self.object_decoder(
            object_decoder_in, self.object_shape + (self.image_depth+1,), self.is_training)

        objects = tf.nn.sigmoid(
            self.decoder_logit_scale * tf.clip_by_value(object_logits, -10., 10.))

        objects = tf.reshape(
            objects, (self.batch_size, self._tensors["max_objects"]) + self.object_shape + (self.image_depth+1,))

        if "alpha" in self.fixed_values:
            obj_img, obj_alpha = tf.split(objects, [3, 1], axis=-1)
            fixed_obj_alpha = float(self.fixed_values["alpha"]) * tf.ones_like(obj_alpha, dtype=tf.float32)
            objects = tf.concat([obj_img, fixed_obj_alpha], axis=-1)

        self._tensors["objects"] = objects

        # --- Compute sprite locations from box parameters ---

        # All in cell-local co-ordinates, should be invariant to image size.
        boxes = self.program['box']
        cell_y, cell_x, h, w = tf.split(boxes, 4, axis=-1)

        anchor_box_h = self.anchor_boxes[:, 0].reshape(1, 1, 1, self.B, 1)
        anchor_box_w = self.anchor_boxes[:, 1].reshape(1, 1, 1, self.B, 1)

        # box height and width normalized to image height and width
        ys = h * anchor_box_h / self.image_height
        xs = w * anchor_box_w / self.image_width

        # box centre normalized to image height and width
        yt = (
            (self.pixels_per_cell[0] / self.image_height) *
            (cell_y + tf.range(self.H, dtype=tf.float32)[None, :, None, None, None])
        )
        xt = (
            (self.pixels_per_cell[1] / self.image_width) *
            (cell_x + tf.range(self.W, dtype=tf.float32)[None, None, :, None, None])
        )

        # `render_sprites` requires box top-left, whereas y and x give box center
        yt -= ys / 2
        xt -= xs / 2

        self._tensors["normalized_box"] = tf.concat([yt, xt, ys, xs], axis=-1)

        scales = tf.gather_nd(tf.concat([ys, xs], axis=-1), self._tensors["routing"])
        offsets = tf.gather_nd(tf.concat([yt, xt], axis=-1), self._tensors["routing"])

        # --- Compose images ---

        output = render_sprites.render_sprites(
            objects,
            self._tensors["n_objects"],
            scales,
            offsets,
            self._tensors["inputs"]["background"]
        )

        output = tf.clip_by_value(output, 1e-6, 1-1e-6)
        output_logits = tf.log(output / (1 - output))

        # --- Store values ---

        self._tensors['area'] = (ys * float(self.image_height)) * (xs * float(self.image_width))
        self._tensors['output'] = output
        self._tensors['output_logits'] = output_logits

    def build_graph(self, loss_key, **network_inputs):

        # --- initialize containers for storing outputs ---

        self._tensors = dict(
            samples=dict(),
            entropy=dict(),
            log_probs=dict(),
            logits=dict(),
            program=dict(),
        )
        self.info_types = list(self._tensors.keys())

        self._tensors["inputs"] = network_inputs

        self.program = self._tensors["program"]
        self.samples = self._tensors["samples"]
        self.log_probs = self._tensors["log_probs"]

        # --- build graph ---

        self._build_placeholders()

        if self.sequential_cfg['on']:
            self._build_program_generator_sequential()
        else:
            self._build_program_generator()

        if self.object_decoder is None:
            self.object_decoder = cfg.build_object_decoder(scope="object_decoder")

            if "decoder" in self.fixed_weights:
                self.object_decoder.fix_variables()

        self._build_program_interpreter()

        # --- specify values to record/summarize ---

        recorded_tensors = {}

        recorded_tensors['cell_y'] = tf.reduce_mean(self._tensors["cell_y"])
        recorded_tensors['cell_x'] = tf.reduce_mean(self._tensors["cell_x"])
        recorded_tensors['h'] = tf.reduce_mean(self._tensors["h"])
        recorded_tensors['w'] = tf.reduce_mean(self._tensors["w"])

        recorded_tensors['obj'] = tf.reduce_mean(self._tensors["program"]["obj"])
        recorded_tensors['obj_entropy'] = tf.reduce_mean(self._tensors["entropy"]["obj"])
        recorded_tensors['obj_log_probs'] = tf.reduce_mean(self._tensors["log_probs"]["obj"])
        recorded_tensors['obj_logits'] = tf.reduce_mean(self._tensors["logits"]["obj"])

        recorded_tensors['z'] = tf.reduce_mean(self._tensors["program"]["z"])
        recorded_tensors['z_entropy'] = tf.reduce_mean(self._tensors["entropy"]["z"])
        recorded_tensors['z_log_probs'] = tf.reduce_mean(self._tensors["log_probs"]["z"])
        recorded_tensors['z_logits'] = tf.reduce_mean(self._tensors["logits"]["z"])

        recorded_tensors['attr'] = tf.reduce_mean(self._tensors["program"]["attr"])

        # --- compute rl loss ---

        COST = tf.zeros((self.batch_size, self.H, self.W, self.B, 1))
        COST_obj = tf.zeros((self.batch_size, self.H, self.W, self.B, 1))
        COST_z = tf.zeros((self.batch_size, self.H, self.W, self.B, 1))

        for name, (weight, _, kind) in self.COST_funcs.items():
            cost = self.COST_tensors[name]
            weight = build_scheduled_value(weight, "COST_{}_weight".format(name))
            weighted_cost = weight * cost

            if kind == "both":
                COST += weighted_cost
            elif kind == "obj":
                COST_obj += weighted_cost
            elif kind == "z":
                COST_z += weighted_cost
            else:
                raise Exception("NotImplemented")

            recorded_tensors["COST_{}".format(name)] = tf.reduce_mean(cost)
            recorded_tensors["WEIGHTED_COST_{}".format(name)] = tf.reduce_mean(weighted_cost)

        recorded_tensors["TOTAL_COST"] = (
            tf.reduce_mean(COST) +
            tf.reduce_mean(COST_obj) +
            tf.reduce_mean(COST_z)
        )

        if self.use_baseline:
            COST -= tf.reduce_mean(COST, axis=0, keepdims=True)
            COST_obj -= tf.reduce_mean(COST_obj, axis=0, keepdims=True)
            COST_z -= tf.reduce_mean(COST_z, axis=0, keepdims=True)

        rl_loss_map = (
            (COST_obj + COST) * self.log_probs['obj'] +
            (COST_z + COST) * self.log_probs['z']
        )

        # --- required for computing the reconstruction COST ---

        output_logits = self._tensors['output_logits']
        inp = self._tensors['inputs']['inp']
        self._tensors['per_pixel_reconstruction_loss'] = loss_builders[loss_key](output_logits, inp)

        # --- store losses ---

        losses = {}

        losses['rl'] = tf.reduce_mean(rl_loss_map)

        recorded_tensors['area_loss'] = tf_mean_sum(tf.abs(self._tensors['area'] - self.target_area) * self.program['obj'])
        losses['weighted_area'] = self.area_weight * recorded_tensors['area_loss']

        losses['reconstruction'] = tf_mean_sum(loss_builders[loss_key](output_logits, inp))

        return {
            "tensors": self._tensors,
            "recorded_tensors": recorded_tensors,
            "losses": losses
        }


class YoloRL_Updater(Updater):
    xent_loss = Param()
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    eval_modes = "val".split()

    def __init__(self, env, network, scope=None, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.network = network

        self.datasets = env.datasets

        for dset in self.datasets.values():
            dset.reset()

        self.scope = scope
        self._n_experiences = 0
        self._n_updates = 0

    @property
    def completion(self):
        return self.datasets['train'].completion

    def trainable_variables(self, for_opt):
        return self.network.trainable_variables(for_opt)

    def _update(self, batch_size, collect_summaries):
        feed_dict, _other = self.make_feed_dict(batch_size, 'train', False)

        self.network.update_feed_dict(feed_dict, _other)

        sess = tf.get_default_session()
        summary = b''
        if collect_summaries:
            _, record, summary = sess.run(
                [self.train_op, self.recorded_tensors, self.summary_op], feed_dict=feed_dict)
        else:
            _, record = sess.run(
                [self.train_op, self.recorded_tensors], feed_dict=feed_dict)

        return dict(train=(record, summary))

    def _evaluate(self, batch_size, mode):
        assert mode in self.eval_modes

        sess = tf.get_default_session()
        feed_dicts = self.make_feed_dict(batch_size, 'val', True, whole_epoch=True)

        record = collections.defaultdict(float)
        summary = b''
        n_points = 0

        for _batch_size, feed_dict, other in feed_dicts:
            self.network.update_feed_dict(feed_dict, other)

            _record, summary = sess.run(
                [self.recorded_tensors, self.summary_op], feed_dict=feed_dict)

            for k, v in _record.items():
                record[k] += _batch_size * v

            n_points += _batch_size

        for k, v in record.items():
            record[k] /= n_points

        return record, summary

    def _make_feed_dict(self, batch, evaluate):
        if len(batch) == 1:
            inp, annotations = batch[0], None
        elif len(batch) == 2:
            inp, annotations = batch
        else:
            raise Exception()

        other = dict(annotations=annotations)

        # Compute the mode colour of each image.
        discrete = np.uint8(np.floor(inp * 255.))
        discrete = discrete.reshape(discrete.shape[0], -1, discrete.shape[-1])
        modes = []
        for row in discrete:
            counts = collections.Counter(tuple(t) for t in row)
            mode, mode_count = counts.most_common(1)[0]
            modes.append(mode)
        modes = np.array(modes) / 255

        feed_dict = {
            self.inp_ph: inp,
            self.inp_mode: modes,
            self.is_training: not evaluate
        }

        return inp.shape[0], feed_dict, other

    def make_feed_dict(self, batch_size, mode, evaluate, whole_epoch=False):
        """
        If `whole_epoch` is True, create multiple feed dicts, each containing at most
        `batch_size` data points, until the epoch is completed. In this case return a list
        whose elements are of the form `(batch_size, feed_dict, other)`, where `other`
        is a dictionary containing metadata about each sub-batch.

        """
        dicts = []
        dset = self.datasets[mode]

        if whole_epoch:
            dset.reset_epoch()

        epochs_before = dset.epochs_completed

        while True:
            batch = dset.next_batch(batch_size=batch_size, advance=not evaluate or whole_epoch, rollover=not whole_epoch)
            actual_batch_size, feed_dict, other = self._make_feed_dict(batch, evaluate)

            if whole_epoch:
                dicts.append((actual_batch_size, feed_dict, other))

                if dset.epochs_completed != epochs_before:
                    break
            else:
                return feed_dict, other

        return dicts

    def _build_placeholders(self):
        self.inp_ph = tf.placeholder(tf.float32, (None,) + self.obs_shape, name="inp_ph")
        self.inp = tf.clip_by_value(self.inp_ph, 1e-6, 1-1e-6, name="inp")
        self.inp_mode = tf.placeholder(tf.float32, (None, self.image_depth), name="inp_mode_ph")

        self.is_training = tf.placeholder(tf.bool, ())
        self.float_is_training = tf.to_float(self.is_training)

    def _build_graph(self):
        self._build_placeholders()

        if cfg.background_cfg.mode == "static":
            with cfg.background_cfg.static_cfg:
                from kmodes.kmodes import KModes
                print("Clustering...")
                print(cfg.background_cfg.static_cfg)

                cluster_data = self.datasets["train"].X
                image_shape = cluster_data.shape[1:]
                indices = np.random.choice(
                    cluster_data.shape[0], replace=False, size=cfg.n_clustering_examples)
                cluster_data = cluster_data[indices, ...]
                cluster_data = cluster_data.reshape(cluster_data.shape[0], -1)

                if cfg.use_k_modes:
                    km = KModes(n_clusters=cfg.n_clusters, init='Huang', n_init=1, verbose=1)
                    km.fit(cluster_data)
                    centroids = km.cluster_centroids_ / 255.
                else:
                    cluster_data = cluster_data / 255.
                    result = k_means(cluster_data, cfg.n_clusters)
                    centroids = result[0]

                centroids = np.maximum(centroids, 1e-6)
                centroids = np.minimum(centroids, 1-1e-6)
                centroids = centroids.reshape(cfg.n_clusters, *image_shape)
        elif cfg.background_cfg.mode == "mode":
            self.background = self.inp_mode[:, None, None, :] * tf.ones_like(self.inp)
        else:
            self.background = tf.zeros_like(self.inp)

        loss_key = 'xent' if self.xent_loss else 'squared'
        network_outputs = self.network.build_graph(
            inp=self.inp,
            inp_ph=self.inp_ph,
            is_training=self.is_training,
            float_is_training=self.float_is_training,
            background=self.background,
            loss_key=loss_key
        )

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]

        recorded_tensors = {}

        output_logits = network_tensors["output_logits"]
        recorded_tensors.update({
            name + "_loss": tf_mean_sum(builder(output_logits, self.inp))
            for name, builder in loss_builders.items()
        })

        recorded_tensors['loss'] = 0
        for name, tensor in network_losses.items():
            recorded_tensors['loss'] += tensor
            recorded_tensors[name + '_loss'] = tensor
        self.loss = recorded_tensors['loss']

        intersection = recorded_tensors.keys() & network_recorded_tensors.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)
        recorded_tensors.update(network_recorded_tensors)

        self.recorded_tensors = recorded_tensors

        _summary = [tf.summary.scalar(name, t) for name, t in recorded_tensors.items()]

        # --- train op ---

        tvars = self.trainable_variables(for_opt=True)

        self.train_op, train_summary = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.summary_op = tf.summary.merge(_summary + train_summary)


class YoloRL_RenderHook(object):
    def __init__(self, N=16):
        self.N = N

    def __call__(self, updater):
        if updater.stage_idx == 0:
            path = updater.exp_dir.path_for('plots', 'frames.pdf')
            if not os.path.exists(path):
                fig, axes = square_subplots(16)
                for ax, frame in zip(axes.flatten(), updater.datasets['train'].x):
                    ax.imshow(frame)

                fig.savefig(path)
                plt.close(fig)

        fetched = self._fetch(self.N, updater)

        self._plot_reconstruction(updater, fetched)
        self._plot_patches(updater, fetched, 4)

    def _fetch(self, N, updater):
        network = updater.network

        feed_dict, network._other = updater.make_feed_dict(N, 'val', True)
        images = feed_dict[network.inp_ph]

        network._update_feed_dict_with_routing(feed_dict)

        to_fetch = network.program.copy()
        to_fetch["output"] = network._tensors["output"]
        to_fetch["objects"] = network._tensors["objects"]
        to_fetch["routing"] = network._tensors["routing"]
        to_fetch["n_objects"] = network._tensors["n_objects"]
        to_fetch["normalized_box"] = network._tensors["normalized_box"]
        to_fetch["background"] = network._tensors["inputs"]["background"]

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        fetched.update(images=images)
        fetched["annotations"] = network._other["annotations"]

        return fetched

    def _plot_reconstruction(self, updater, fetched):
        images = fetched['images']
        background = fetched['background']
        N = images.shape[0]

        output = fetched['output']

        _, image_height, image_width, _ = images.shape

        obj = fetched['obj'].reshape(N, -1)

        annotations = fetched["annotations"]
        if annotations is None:
            annotations = [[]] * N

        box = (
            fetched['normalized_box'] *
            [image_height, image_width, image_height, image_width]
        )

        box = box.reshape(box.shape[0], -1, 4)

        sqrt_N = int(np.ceil(np.sqrt(N)))

        fig, axes = plt.subplots(3*sqrt_N, sqrt_N, figsize=(20, 20))
        axes = np.array(axes).reshape(3*sqrt_N, sqrt_N)
        for n, (pred, gt, bg) in enumerate(zip(output, images, background)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            ax1 = axes[3*i, j]
            ax1.imshow(pred)
            ax1.set_title('reconstruction')

            ax2 = axes[3*i+1, j]
            ax2.imshow(gt)
            ax2.set_title('actual')

            # Plot proposed bounding boxes
            for o, b in zip(obj[n], box[n]):
                t, l, h, w = b

                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=1, edgecolor="xkcd:azure", facecolor='none', alpha=o)
                ax1.add_patch(rect)
                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=1, edgecolor="xkcd:azure", facecolor='none', alpha=o)
                ax2.add_patch(rect)

            # Plot true bounding boxes
            for a in annotations[n]:
                _, t, b, l, r = a
                h = b - t
                w = r - l

                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=1, edgecolor="xkcd:yellow", facecolor='none')
                ax1.add_patch(rect)

                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=1, edgecolor="xkcd:yellow", facecolor='none')
                ax2.add_patch(rect)

            ax3 = axes[3*i+2, j]
            ax3.imshow(bg)
            ax3.set_title('background')

        fig.suptitle('Stage={}. After {} experiences ({} updates, {} experiences per batch).'.format(
            updater.stage_idx, updater.n_experiences, updater.n_updates, cfg.batch_size))

        path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), 'sampled_reconstruction.pdf')
        fig.savefig(path)

        plt.close(fig)

    def _plot_patches(self, updater, fetched, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        objects = fetched['objects']

        H, W, B = updater.network.H, updater.network.W, updater.network.B

        obj = fetched['obj']
        n_objects = fetched['n_objects']
        routing = fetched['routing']
        z = fetched['z']

        for idx in range(N):
            fig, axes = plt.subplots(2*H, W * B, figsize=(20, 20))
            axes = np.array(axes).reshape(2*H, W * B)

            for h in range(H):
                for w in range(W):
                    for b in range(B):
                        _obj = obj[idx, h, w, b, 0]
                        _z = z[idx, h, w, b, 0]

                        ax = axes[2*h, w * B + b]

                        if h == 0 and b == 0:
                            ax.set_title("w={}".format(w))
                        if w == 0 and b == 0:
                            ax.set_ylabel("h={}".format(h))

                        ax = axes[2*h+1, w * B + b]

                        ax.set_title("obj={}, z={}, b={}".format(_obj, _z, b))

            for i in range(n_objects[idx]):
                _, h, w, b = routing[idx, i]

                ax = axes[2*h, w * B + b]

                ax.imshow(objects[idx, i, :, :, :3])

                ax = axes[2*h+1, w * B + b]
                ax.imshow(objects[idx, i, :, :, 3])

            path = updater.exp_dir.path_for(
                'plots', 'stage{}'.format(updater.stage_idx),
                'sampled_patches', '{}.pdf'.format(idx))
            fig.savefig(path)
            plt.close(fig)


xkcd_colors = 'viridian,cerulean,vermillion,lavender,celadon,fuchsia,saffron,cinnamon,greyish,vivid blue'.split(',')


# env config

config = Config(
    log_name="yolo_rl",
    build_env=Env,
    seed=347405995,

    use_dataset_cache=True,
    min_chars=12,
    max_chars=12,
    n_sub_image_examples=0,

    image_shape=(84, 84),
    max_overlap=400,
    sub_image_shape=(14, 14),

    characters=list(range(10)),
    sub_image_size_std=0.0,
    colours="white",

    n_distractors_per_image=0,

    backgrounds="",
    backgrounds_sample_every=False,
    background_colours="",

    background_cfg=dict(mode="none"),

    object_shape=(14, 14),

    xent_loss=True,

    postprocessing="random",
    n_samples_per_image=4,
    tile_shape=(42, 42),
    max_attempts=1000000,

    preserve_env=True,

    n_train=25000,
    n_val=1e2,
    n_test=1e2,
)

# model config


config.update(
    get_updater=get_updater,

    lr_schedule=1e-4,
    batch_size=32,

    optimizer_spec="adam",
    use_gpu=True,
    gpu_allow_growth=True,
    preserve_env=True,
    stopping_criteria="TOTAL_COST,min",
    eval_mode="val",
    threshold=-np.inf,
    max_grad_norm=1.0,
    max_experiments=None,

    eval_step=100,
    render_step=100,
    display_step=1000,
    max_steps=1e7,
    patience=10000,

    render_hook=YoloRL_RenderHook(),

    # network params

    build_backbone=Backbone,
    build_next_step=NextStep,
    build_object_decoder=ObjectDecoder,

    use_input_attention=False,
    decoder_logit_scale=10.0,

    pixels_per_cell=(12, 12),

    anchor_boxes=[
        [7, 7],
        [7, 7]
    ],

    kernel_size=(1, 1),

    n_channels=128,
    n_decoder_channels=128,
    A=100,

    n_passthrough_features=100,

    max_hw=0.3,  # Maximum for the bounding box multiplier.
    min_hw=3.0,  # Minimum for the bounding box multiplier.

    box_std=0.0,
    attr_std=0.0,
    z_std=0.1,
    obj_exploration=0.05,
    obj_default=0.5,

    # Costs
    use_baseline=True,
    area_weight=2.,
    nonzero_weight=150.,

    local_reconstruction_cost=True,
    area_neighbourhood_size=1,
    nonzero_neighbourhood_size=1,

    fixed_values=dict(),
    fixed_weights="",
    order="box obj z attr",

    sequential_cfg=dict(
        on=True,
        lookback_shape=(2, 2, 2),
        build_next_step=lambda scope: MLP([100, 100], scope=scope),
    ),

    curriculum=[
        dict(fixed_weights="obj", fixed_values=dict(obj=1), max_steps=10000, patience=10000),
        dict(obj_exploration=0.2),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
        dict(obj_exploration=0.03),
        dict(obj_exploration=0.02),
        dict(obj_exploration=0.01),
        dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),  # Test on full size images.
    ],
)
