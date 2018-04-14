import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import k_means
import collections

from dps import cfg
from dps.datasets import EMNIST_ObjectDetection
from dps.updater import Updater
from dps.utils import Config, Param, square_subplots, prime_factors
from dps.utils.tf import (
    VectorQuantization, FullyConvolutional, build_gradient_train_op,
    trainable_variables, build_scheduled_value, tf_mean_sum)
from dps.env.advanced.yolo import mAP
from dps.tf_ops import render_sprites
from dps.train import PolynomialScheduleHook

tf_flatten = tf.layers.flatten


class Env(object):
    def __init__(self):
        train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


def get_updater(env):
    return YoloRL_Updater(env)


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

        layout = [dict(filters=self.n_channels, kernel_size=4, strides=(_sh, _sw), padding="SAME")
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


class VQ_ObjectDecoder(FullyConvolutional):
    vq_input_shape = Param()
    K = Param()
    common_embedding = Param()
    n_decoder_channels = Param()
    beta = Param()

    _vq = None

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=4, kernel_size=4, strides=1, padding="SAME", transpose=True),  # For 14 x 14 output
        ]
        super(VQ_ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)

    def _call(self, inp, output_size, is_training):
        if self._vq is None:
            H, W, D = self.vq_input_shape
            self._vq = VectorQuantization(
                H=H, W=W, D=D, K=self.K, common_embedding=self.common_embedding)

        batch_size = tf.shape(inp)[0]
        inp = tf.reshape(inp, (batch_size,) + self.vq_input_shape)

        quantized_inp = self._vq(inp, self.vq_input_shape, is_training)
        return super(VQ_ObjectDecoder, self)._call(quantized_inp, output_size, is_training)


def reconstruction_cost(network_outputs, updater):
    return network_outputs['reconstruction_loss'][..., None, None, None]


def specific_reconstruction_cost(network_outputs, updater):
    centers_h = (np.arange(updater.H) + 0.5) * updater.pixels_per_cell[0]
    centers_w = (np.arange(updater.W) + 0.5) * updater.pixels_per_cell[1]

    loc_h = (np.arange(updater.image_height) + 0.5)[..., None]
    loc_w = (np.arange(updater.image_width) + 0.5)[..., None]

    dist_h = np.abs(loc_h - centers_h)
    dist_w = np.abs(loc_w - centers_w)

    all_filtered = []

    loss = network_outputs['per_pixel_reconstruction_loss']

    for b in range(updater.B):
        max_distance_h = updater.pixels_per_cell[0] / 2 + updater.max_hw * updater.anchor_boxes[b, 0] / 2
        max_distance_w = updater.pixels_per_cell[1] / 2 + updater.max_hw * updater.anchor_boxes[b, 1] / 2

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


def area_cost(network_outputs, updater):
    return network_outputs['area_loss'][..., None, None, None]


def specific_area_cost(network_outputs, updater):
    return network_outputs['program']['obj'] * network_outputs['area']


def nonzero_cost(network_outputs, updater):
    obj_samples = network_outputs['program']['obj']
    obj_samples = obj_samples.reshape(obj_samples.shape[0], -1)
    return np.count_nonzero(obj_samples, axis=1)[..., None, None, None, None].astype('f')


def specific_nonzero_cost(network_outputs, updater):
    return network_outputs['program']['obj']


def negative_mAP_cost(network_outputs, updater):
    annotations = updater._other["annotations"]
    if annotations is None:
        return np.zeros((1, 1, 1, 1, 1))
    else:
        gt = []
        pred_boxes = []

        obj = network_outputs['program']['obj']
        top, left, height, width = np.split(network_outputs['normalized_box'], 4, axis=-1)

        top = updater.image_height * top
        height = updater.image_height * height
        bottom = top + height

        left = updater.image_width * left
        width = updater.image_width * width
        right = left + width

        for idx, a in enumerate(annotations):
            _a = [[0, *rest] for cls, *rest in a]
            gt.append(_a)

            pb = []

            for i in range(updater.H):
                for j in range(updater.W):
                    for b in range(updater.B):
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


def count_error_cost(network_outputs, updater):
    annotations = updater._other["annotations"]
    if annotations is None:
        return np.zeros((1, 1, 1, 1, 1))
    else:
        obj = network_outputs["program"]["obj"]
        n_objects_pred = obj.reshape(obj.shape[0], -1).sum(axis=1)
        n_objects_true = np.array([len(a) for a in annotations])
        error = (n_objects_pred != n_objects_true).astype('f')
        return error.reshape(-1, 1, 1, 1, 1)


def count_1norm_cost(network_outputs, updater):
    annotations = updater._other["annotations"]
    if annotations is None:
        return np.zeros((1, 1, 1, 1, 1))
    else:
        obj = network_outputs["program"]["obj"]
        n_objects_pred = obj.reshape(obj.shape[0], -1).sum(axis=1)
        n_objects_true = np.array([len(a) for a in annotations])
        dist = np.abs(n_objects_pred - n_objects_true).astype('f')
        return dist.reshape(-1, 1, 1, 1, 1)


class YoloRL_Updater(Updater):
    pixels_per_cell = Param()
    image_shape = Param()
    A = Param(help="Dimension of attribute vector.")
    anchor_boxes = Param(help="List of (h, w) pairs.")
    object_shape = Param()

    use_input_attention = Param()
    decoder_logit_scale = Param()

    max_hw = Param()
    min_hw = Param()

    box_std = Param()
    attr_std = Param()

    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    n_passthrough_features = Param()

    xent_loss = Param()

    use_baseline = Param()
    nonzero_weight = Param()
    area_weight = Param()
    use_specific_costs = Param()

    obj_exploration = Param()
    obj_default = Param()

    z_std = Param()

    fixed_box = Param()
    fixed_obj = Param()
    fixed_z = Param()
    fixed_attr = Param()

    fixed_object_decoder = Param()

    fix_values = Param()
    order = Param()

    eval_modes = "val".split()

    def __init__(self, env, scope=None, **kwargs):
        self.anchor_boxes = np.array(self.anchor_boxes)
        self.H = int(np.ceil(self.image_shape[0] / self.pixels_per_cell[0]))
        self.W = int(np.ceil(self.image_shape[1] / self.pixels_per_cell[1]))
        self.B = len(self.anchor_boxes)

        self.datasets = env.datasets

        for dset in self.datasets.values():
            dset.reset()

        self.obs_shape = self.datasets['train'].x.shape[1:]
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.COST_funcs = {}

        if self.nonzero_weight > 0.0:
            cost_func = specific_nonzero_cost if self.use_specific_costs else nonzero_cost
            self.COST_funcs['nonzero'] = (self.nonzero_weight, cost_func, "obj")

        if self.area_weight > 0.0:
            cost_func = specific_area_cost if self.use_specific_costs else area_cost
            self.COST_funcs['area'] = (self.area_weight, cost_func, "obj")

        cost_func = specific_reconstruction_cost if self.use_specific_costs else reconstruction_cost
        self.COST_funcs['reconstruction'] = (1, cost_func, "both")
        self.COST_funcs['negative_mAP'] = (0, negative_mAP_cost, "obj")
        self.COST_funcs['count_error'] = (0, count_error_cost, "obj")
        self.COST_funcs['count_1norm'] = (0, count_1norm_cost, "obj")

        self.scope = scope
        self._n_experiences = 0
        self._n_updates = 0

        self.object_decoder = None

        if isinstance(self.order, str):
            self.order = self.order.split()
        assert set(self.order) == set("box obj z attr".split())

        self.layer_params = dict(
            box=dict(
                rep_builder=self._build_box,
                fixed=self.fixed_box,
                output_size=4,
                network=None
            ),
            obj=dict(
                rep_builder=self._build_obj,
                fixed=self.fixed_obj,
                output_size=1,
                network=None
            ),
            z=dict(
                rep_builder=self._build_z,
                fixed=self.fixed_z,
                output_size=1,
                network=None
            ),
            attr=dict(
                rep_builder=self._build_attr,
                fixed=self.fixed_attr,
                output_size=self.A,
                network=None
            ),
        )

    @property
    def completion(self):
        return self.datasets['train'].completion

    def trainable_variables(self, for_opt):
        scoped_functions = (
            [self.object_decoder] +
            [self.layer_params[kind]["network"] for kind in self.order]
        )

        tvars = []
        for sf in scoped_functions:
            tvars.extend(trainable_variables(sf.scope, for_opt=for_opt))

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
        # --- sample program only

        sess = tf.get_default_session()
        program, samples = sess.run([self.program, self.samples], feed_dict=feed_dict)
        feed_dict.update({self.samples[k]: v for k, v in samples.items()})

        # --- compute routing based on sampled program

        max_objects, n_objects, routing = self._compute_routing(program)
        feed_dict[self.max_objects] = max_objects
        feed_dict[self.n_objects] = n_objects
        feed_dict[self.routing] = routing

    def _update_feed_dict_with_costs(self, feed_dict):
        sess = tf.get_default_session()
        network_outputs = sess.run(self.network_outputs, feed_dict=feed_dict)

        costs = {
            self.COST_tensors[name]: self._process_cost(f(network_outputs, self))
            for name, (_, f, _) in self.COST_funcs.items()
        }
        feed_dict.update(costs)

    def _update(self, batch_size, collect_summaries):
        feed_dict, self._other = self.make_feed_dict(batch_size, 'train', False)

        self._update_feed_dict_with_routing(feed_dict)
        self._update_feed_dict_with_costs(feed_dict)

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
            self._other = other
            self._update_feed_dict_with_routing(feed_dict)
            self._update_feed_dict_with_costs(feed_dict)

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
        whose elements are of the form `(batch_size, feed_dict)`.

        """
        dicts = []
        dset = self.datasets[mode]

        if whole_epoch:
            dset.reset_epoch()

        epochs_before = dset.epochs_completed

        while True:
            batch = dset.next_batch(batch_size=batch_size, advance=True, rollover=not whole_epoch)
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

        self.max_objects = tf.placeholder(tf.int32, (), name="max_objects")
        self.n_objects = tf.placeholder(tf.int32, (None,), name="n_objects")
        self.routing = tf.placeholder(tf.int32, (None, None, 4), name="routing")

        self.is_training = tf.placeholder(tf.bool, ())
        self.float_is_training = tf.to_float(self.is_training)

        self.batch_size = tf.shape(self.inp)[0]
        H, W, B = self.H, self.W, self.B

        self.COST_tensors = {}

        for name, _ in self.COST_funcs.items():
            cost = tf.placeholder(
                tf.float32, (None, H, W, B, 1), name="COST_{}_ph".format(name))
            self.COST_tensors[name] = cost

    def _build_box(self, box_logits, is_training):
        box = tf.nn.sigmoid(tf.clip_by_value(box_logits, -10., 10.))
        cell_y, cell_x, h, w = tf.split(box, 4, axis=-1)

        h = float(self.max_hw - self.min_hw) * h + self.min_hw
        w = float(self.max_hw - self.min_hw) * w + self.min_hw

        if "h" in self.fix_values:
            h = float(self.fix_values["h"]) * tf.ones_like(h, dtype=tf.float32)
        if "w" in self.fix_values:
            w = float(self.fix_values["w"]) * tf.ones_like(w, dtype=tf.float32)
        if "cell_y" in self.fix_values:
            cell_y = float(self.fix_values["cell_y"]) * tf.ones_like(cell_y, dtype=tf.float32)
        if "cell_x" in self.fix_values:
            cell_x = float(self.fix_values["cell_x"]) * tf.ones_like(cell_x, dtype=tf.float32)

        box = tf.concat([cell_y, cell_x, h, w], axis=-1)

        box_std = build_scheduled_value(self.box_std, name="box_std")
        box_noise = tf.random_normal(tf.shape(box), name="box_noise")
        noisy_box = box + box_noise * box_std * self.float_is_training

        self.network_outputs["cell_y"] = cell_y
        self.network_outputs["cell_x"] = cell_x
        self.network_outputs["h"] = h
        self.network_outputs["w"] = w
        self.samples["box"] = box_noise

        return noisy_box

    def _build_obj(self, obj_logits, is_training):
        obj_logits = tf.clip_by_value(obj_logits, -10., 10.)

        obj_params = tf.nn.sigmoid(obj_logits)
        obj_exploration = build_scheduled_value(self.obj_exploration, "obj_exploration") * self.float_is_training
        obj_params = (1 - obj_exploration) * obj_params + self.obj_default * obj_exploration

        obj_dist = tf.distributions.Bernoulli(probs=obj_params)

        obj_samples = tf.stop_gradient(obj_dist.sample())
        obj_samples = tf.to_float(obj_samples)

        obj_log_probs = obj_dist.log_prob(obj_samples)
        obj_log_probs = tf.where(tf.is_nan(obj_log_probs), -100.0 * tf.ones_like(obj_log_probs), obj_log_probs)

        obj_entropy = obj_dist.entropy()

        if "obj" in self.fix_values:
            obj_samples = float(self.fix_values["obj"]) * tf.ones_like(obj_samples, dtype=tf.float32)

        self.samples["obj"] = obj_samples
        self.entropy["obj"] = obj_entropy
        self.log_probs["obj"] = obj_log_probs
        self.logits["obj"] = obj_logits

        return obj_samples

    def _build_z(self, z_logits, is_training):
        z_logits = tf.clip_by_value(z_logits, -10., 10.)

        z_params = tf.nn.sigmoid(z_logits)
        z_std = build_scheduled_value(self.z_std, "z_std") * self.float_is_training

        z_dist = tf.distributions.Normal(loc=z_params, scale=z_std)

        z_samples = tf.stop_gradient(z_dist.sample())

        z_log_probs = z_dist.log_prob(z_samples)
        z_log_probs = tf.where(tf.is_nan(z_log_probs), -100.0 * tf.ones_like(z_log_probs), z_log_probs)

        z_entropy = z_dist.entropy()

        if "z" in self.fix_values:
            z_samples = float(self.fix_values["z"]) * tf.ones_like(z_samples, dtype=tf.float32)

        self.samples["z"] = z_samples
        self.entropy["z"] = z_entropy
        self.log_probs["z"] = z_log_probs
        self.logits["z"] = z_logits

        return z_samples

    def _build_attr(self, attr_mean, is_training):
        attr_std = build_scheduled_value(self.attr_std, name="attr_std")
        attr_noise = tf.random_normal(tf.shape(attr_mean), name="attr_noise")

        noisy_attr = attr_mean + attr_noise * attr_std * self.float_is_training

        self.samples["attr"] = attr_noise

        return noisy_attr

    def _build_program_generator(self):
        inp, is_training = self.inp, self.is_training
        H, W, B = self.H, self.W, self.B
        program, features = None, None

        self.program = {}
        self.samples = {}
        self.entropy = {}
        self.log_probs = {}
        self.logits = {}

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
                layer_inp = inp

            network_output = network(layer_inp, (H, W, out_channel_dim), is_training)

            features = network_output[..., B*output_size:]

            output = network_output[..., :B*output_size]
            output = tf.reshape(output, (-1, H, W, B, output_size))

            representation = rep_builder(output, is_training)

            self.program[kind] = representation

            if i:
                program = tf.concat([program, representation], axis=-1)
            else:
                program = representation

        # --- finalize ---

        self.network_outputs.update(
            program=self.program, samples=self.samples, entropy=self.entropy,
            log_probs=self.log_probs, logits=self.logits)

    def _build_program_interpreter(self):
        # --- Compute sprites from attrs using object decoder ---

        attrs = self.program['attr']

        routed_attrs = tf.gather_nd(attrs, self.routing)
        object_decoder_in = tf.reshape(routed_attrs, (-1, 1, 1, self.A))

        object_logits = self.object_decoder(
            object_decoder_in, self.object_shape + (self.image_depth+1,), self.is_training)

        objects = tf.nn.sigmoid(
            self.decoder_logit_scale * tf.clip_by_value(object_logits, -10., 10.))

        objects = tf.reshape(
            objects, (self.batch_size, self.max_objects) + self.object_shape + (self.image_depth+1,))

        if "alpha" in self.fix_values:
            obj_img, obj_alpha = tf.split(objects, [3, 1], axis=-1)
            fixed_obj_alpha = float(self.fix_values["alpha"]) * tf.ones_like(obj_alpha, dtype=tf.float32)
            objects = tf.concat([obj_img, fixed_obj_alpha], axis=-1)

        self.network_outputs["objects"] = objects

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
            (self.pixels_per_cell[0] / self.image_shape[0]) *
            (cell_y + tf.range(self.H, dtype=tf.float32)[None, :, None, None, None])
        )
        xt = (
            (self.pixels_per_cell[1] / self.image_shape[1]) *
            (cell_x + tf.range(self.W, dtype=tf.float32)[None, None, :, None, None])
        )

        # `render_sprites` requires box top-left, whereas y and x give box centre
        yt -= ys / 2
        xt -= xs / 2

        self.network_outputs["normalized_box"] = tf.concat([yt, xt, ys, xs], axis=-1)

        scales = tf.gather_nd(tf.concat([ys, xs], axis=-1), self.routing)
        offsets = tf.gather_nd(tf.concat([yt, xt], axis=-1), self.routing)

        # --- Compose images ---

        output = render_sprites.render_sprites(objects, self.n_objects, scales, offsets, self.background)
        output = tf.clip_by_value(output, 1e-6, 1-1e-6)
        output_logits = tf.log(output / (1 - output))

        # --- Store values ---

        self.network_outputs['area'] = (ys * float(self.image_height)) * (xs * float(self.image_width))
        self.network_outputs['output'] = output
        self.network_outputs['output_logits'] = output_logits

    def build_area_loss(self, batch=True):
        selected_area = self.network_outputs['area'] * self.program['obj']

        if batch:
            selected_area = tf_flatten(selected_area)
            return tf.reduce_sum(selected_area, keepdims=True, axis=1)
        else:
            return selected_area

    def build_xent_loss(self, logits, targets, batch=True):
        per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

        if batch:
            per_pixel_loss = tf_flatten(per_pixel_loss)
            return tf.reduce_sum(per_pixel_loss, keepdims=True, axis=1)
        else:
            return per_pixel_loss

    def build_squared_loss(self, logits, targets, batch=True):
        actions = tf.sigmoid(logits)
        per_pixel_loss = (actions - targets)**2

        if batch:
            per_pixel_loss = tf_flatten(per_pixel_loss)
            return tf.reduce_sum(per_pixel_loss, keepdims=True, axis=1)
        else:
            return per_pixel_loss

    def build_1norm_loss(self, logits, targets, batch=True):
        actions = tf.sigmoid(logits)
        per_pixel_loss = tf.abs(actions - targets)

        if batch:
            per_pixel_loss = tf_flatten(per_pixel_loss)
            return tf.reduce_sum(per_pixel_loss, keepdims=True, axis=1)
        else:
            return per_pixel_loss

    def _build_graph(self):
        self.network_outputs = {}

        self._build_placeholders()

        self._build_program_generator()

        if self.object_decoder is None:
            self.object_decoder = cfg.build_object_decoder(scope="object_decoder")

            if self.fixed_object_decoder:
                self.object_decoder.fix_variables()

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

        self._build_program_interpreter()

        loss_key = 'xent_loss' if self.xent_loss else 'squared_loss'

        # --- update network outputs ---

        output_logits = self.network_outputs["output_logits"]

        reconstruction_loss = getattr(self, 'build_' + loss_key)(output_logits, self.inp)
        mean_reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        per_pixel_reconstruction_loss = getattr(self, 'build_' + loss_key)(output_logits, self.inp, batch=False)

        area_loss = self.build_area_loss()
        mean_area_loss = tf.reduce_mean(area_loss)

        self.network_outputs.update(
            per_pixel_reconstruction_loss=per_pixel_reconstruction_loss,
            reconstruction_loss=reconstruction_loss,
            area_loss=area_loss
        )

        # --- losses and recorded value ---

        recorded_tensors = {}

        recorded_tensors['obj_entropy'] = tf_mean_sum(self.entropy['obj'])

        recorded_tensors.update({
            name: tf.reduce_mean(getattr(self, 'build_' + name)(output_logits, self.inp))
            for name in ['xent_loss', 'squared_loss', '1norm_loss']
        })

        recorded_tensors['attr'] = tf.reduce_mean(self.program['attr'])

        recorded_tensors['cell_y'] = tf.reduce_mean(self.network_outputs["cell_y"])
        recorded_tensors['cell_x'] = tf.reduce_mean(self.network_outputs["cell_x"])

        recorded_tensors['h'] = tf.reduce_mean(self.network_outputs["h"])
        recorded_tensors['w'] = tf.reduce_mean(self.network_outputs["w"])

        recorded_tensors['obj_logits'] = tf.reduce_mean(self.logits["obj"])
        recorded_tensors['obj'] = tf.reduce_mean(self.program['obj'])

        recorded_tensors['reconstruction_loss'] = mean_reconstruction_loss
        recorded_tensors['area_loss'] = mean_area_loss

        # --- compute rl surrogate loss ---

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

        recorded_tensors["obj_log_probs"] = tf.reduce_mean(self.log_probs['obj'])
        recorded_tensors["z_log_probs"] = tf.reduce_mean(self.log_probs['z'])

        surrogate_loss_map = (
            (COST_obj + COST) * self.log_probs['obj'] +
            (COST_z + COST) * self.log_probs['z']
        )

        recorded_tensors['surrogate_loss'] = tf.reduce_mean(surrogate_loss_map)
        recorded_tensors['loss'] = recorded_tensors['surrogate_loss'] + mean_reconstruction_loss

        # --- compute other losses ---

        if self.area_weight > 0.0:
            recorded_tensors['loss'] += self.area_weight * mean_area_loss

        if isinstance(self.object_decoder, VQ_ObjectDecoder):
            recorded_tensors['loss'] += recorded_tensors['decoder_embedding_error']
            recorded_tensors['loss'] += self.object_decoder.beta * recorded_tensors['decoder_commitment_error']

        self.loss = recorded_tensors['loss']
        self.recorded_tensors = recorded_tensors

        _summary = [tf.summary.scalar(name, t) for name, t in self.recorded_tensors.items()]

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
        feed_dict, updater._other = updater.make_feed_dict(N, 'val', True)
        images = feed_dict[updater.inp_ph]

        updater._update_feed_dict_with_routing(feed_dict)

        to_fetch = updater.program.copy()
        to_fetch["output"] = updater.network_outputs["output"]
        to_fetch["objects"] = updater.network_outputs["objects"]
        to_fetch["routing"] = updater.routing
        to_fetch["n_objects"] = updater.n_objects
        to_fetch["normalized_box"] = updater.network_outputs["normalized_box"]
        to_fetch["background"] = updater.background

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        fetched.update(images=images)
        fetched["annotations"] = updater._other["annotations"]

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

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

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

        H, W, B = updater.H, updater.W, updater.B

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

            path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), 'sampled_patches', '{}.pdf'.format(idx))
            fig.savefig(path)
            plt.close(fig)


xkcd_colors = 'viridian,cerulean,vermillion,lavender,celadon,fuchsia,saffron,cinnamon,greyish,vivid blue'.split(',')


config = Config(
    log_name="yolo_rl",
    get_updater=YoloRL_Updater,
    build_env=Env,

    # dataset params

    min_chars=1,
    max_chars=1,
    characters=[0, 1, 2],
    n_sub_image_examples=0,
    xent_loss=True,
    sub_image_shape=(14, 14),
    use_dataset_cache=True,

    n_train=1e5,
    n_val=1e2,
    n_test=1e2,

    # training loop params

    lr_schedule=1e-4,
    preserve_env=True,
    batch_size=16,
    eval_step=100,
    display_step=1000,
    max_steps=1e7,
    patience=10000,
    optimizer_spec="adam",
    use_gpu=True,
    gpu_allow_growth=True,
    seed=347405995,
    stopping_criteria="TOTAL_COST,min",
    eval_mode="val",
    threshold=-np.inf,
    max_grad_norm=1.0,
    max_experiments=None,
    render_hook=YoloRL_RenderHook(),
    render_step=5000,

    # model params

    build_backbone=Backbone,
    build_next_step=NextStep,
    build_object_decoder=ObjectDecoder,

    background_cfg=dict(
        mode="none",

        static_cfg=dict(
            n_clusters=4,
            use_k_modes=True,  # If False, use k_means.
            n_clustering_examples=True,
            n_clustering_bins=None,
            min_cluster_distance=None,
        ),

        mode_threshold=0.99,
    ),

    use_input_attention=False,
    decoder_logit_scale=10.0,

    image_shape=(28, 28),
    object_shape=(14, 14),
    anchor_boxes=[[28, 28]],
    pixels_per_cell=(28, 28),
    kernel_size=(1, 1),
    n_channels=128,
    n_decoder_channels=128,
    A=100,

    backgrounds="",
    backgrounds_sample_every=False,
    background_colours="",

    n_passthrough_features=100,

    max_hw=1.0,  # Maximum for the bounding box multiplier.
    min_hw=0.0,  # Minimum for the bounding box multiplier.

    box_std=0.1,
    attr_std=0.0,
    z_std=0.1,

    obj_exploration=0.05,
    obj_default=0.5,

    # Costs
    use_baseline=True,
    nonzero_weight=0.0,
    area_weight=0.0,
    use_specific_costs=False,

    curriculum=[
        dict()
    ],

    fixed_box=False,
    fixed_obj=False,
    fixed_z=False,
    fixed_attr=False,
    fixed_object_decoder=False,

    fix_values=dict(),
    order="box obj z attr",

    # VQ object decoder params
    beta=4.0,
    vq_input_shape=(3, 3, 25),
    K=5,
    common_embedding=False,
)


good_config = config.copy(
    image_shape=(50, 50),
    object_shape=(14, 14),
    anchor_boxes=[[14, 14]],
    pixels_per_cell=(12, 12),
    kernel_size=(3, 3),

    max_hw=3.0,
    min_hw=0.25,

    colours="white",
    max_overlap=100,

    # backgrounds="red_x blue_x green_x red_circle blue_circle green_circle",
    # backgrounds_resize=True,

    sub_image_size_std=0.4,
    max_chars=3,
    min_chars=1,
    characters=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

    fix_values=dict(),
    obj_exploration=0.0,
    lr_schedule=1e-4,
    use_specific_costs=True,

    curriculum=[
        dict(fixed_obj=True, fix_values=dict(obj=1), max_steps=10000, area_weight=0.02),
    ],

    nonzero_weight=90.,
    area_weight=0.2,

    box_std=0.1,
    attr_std=0.0,

    render_step=5000,

    n_distractors_per_image=0,
)


uniform_size_config = good_config.copy(
    max_hw=1.5,
    min_hw=0.5,
    sub_image_size_std=0.0,
)

small_test_config = uniform_size_config.copy(
    kernel_size=(3, 3),
    min_chars=1,
    max_chars=3,
    object_shape=(14, 14),
    anchor_boxes=[[7, 7]],
    image_shape=(28, 28),
    max_overlap=100,
    curriculum=[dict(fix_values=dict(obj=1), max_steps=10000, area_weight=0.2)],
)


fragment = [
    dict(obj_exploration=0.2),
    dict(obj_exploration=0.1),
    dict(obj_exploration=0.05),
]


small_test_reset_config = small_test_config.copy(
    curriculum=[
        dict(
            do_train=False,
            load_path="/media/data/dps_data/logs/yolo_rl/exp_yolo_rl_seed=347405995_2018_04_13_21_10_40/weights/best_of_stage_0",
            fix_values=dict(obj=1), max_steps=10000, area_weight=0.2
        )
    ],

    patience=5000,
    area_weight=0.2,

    hooks=[
        PolynomialScheduleHook(
            attr_name="nonzero_weight",
            query_name="best_COST_reconstruction",
            base_configs=fragment, tolerance=2,
            initial_value=90, scale=5, power=1.0)
    ]
)



# fragment = [
#     dict(obj_exploration=0.2),
#     dict(obj_exploration=0.1),
#     dict(obj_exploration=0.05),
# ]
# 
# 
# uniform_size_config = good_config.copy(
#     image_shape=(28, 28),
#     max_hw=1.5,
#     min_hw=0.5,
#     sub_image_size_std=0.0,
#     max_overlap=150,
#     patience=2500,
# 
#     hooks=[
#         PolynomialScheduleHook(
#             attr_name="nonzero_weight",
#             query_name="best_COST_reconstruction",
#             base_configs=fragment, tolerance=2,
#             initial_value=90, scale=5, power=1.0)
#     ]
# )
# 
# first_stage_config = uniform_size_config.copy(
#     hooks=[],
#     patience=10000,
#     curriculum=[
#         dict(fixed_obj=True, fix_values=dict(obj=1), max_steps=100000, area_weight=3.),
#     ],
# )
# 
# 
# small_test_config = uniform_size_config.copy(
#     kernel_size=(3, 3),
#     min_chars=1,
#     max_chars=3,
#     object_shape=(14, 14),
#     anchor_boxes=[[7, 7]],
#     image_shape=(28, 28),
#     max_overlap=100,
# )


# simple_config = first_stage_config.copy(
#     kernel_size=(1, 1),
#     min_chars=1,
#     max_chars=1,
#     pixels_per_cell=(16, 16),
#     object_shape=(14, 14),
#     sub_image_shape=(14, 14),
#     anchor_boxes=[[14, 14]],
#     image_shape=(16, 16),
#     max_overlap=100,
# )


vq_config = good_config.copy(
    # VQ object decoder params
    build_object_decoder=VQ_ObjectDecoder,
    beta=4.0,
    vq_input_shape=(2, 2, 25),
    K=5,
    common_embedding=False
)
