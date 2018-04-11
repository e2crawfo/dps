import tensorflow as tf
import numpy as np
import sonnet as snt
import matplotlib.pyplot as plt
import os
from sklearn.cluster import k_means
from kmodes.kmodes import KModes
import collections

from dps import cfg
from dps.datasets import EMNIST_ObjectDetection
from dps.updater import Updater
from dps.utils import Config, Param, square_subplots, prime_factors
from dps.utils.tf import (
    VectorQuantization, FullyConvolutional, build_gradient_train_op,
    trainable_variables, build_scheduled_value, tf_mean_sum)
from dps.env.advanced.yolo import mAP

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
    return network_outputs['program_fields']['obj'] * network_outputs['area']


def nonzero_cost(network_outputs, updater):
    obj_samples = network_outputs['program_fields']['obj']
    obj_samples = obj_samples.reshape(obj_samples.shape[0], -1)
    return np.count_nonzero(obj_samples, axis=1)[..., None, None, None, None].astype('f')


def specific_nonzero_cost(network_outputs, updater):
    return network_outputs['program_fields']['obj']


def negative_mAP_cost(network_outputs, updater):
    if updater.annotations is None:
        return np.zeros((1, 1, 1, 1, 1))
    else:
        gt = []
        pred_boxes = []

        obj = network_outputs['program_fields']['obj']
        xs, xt, ys, yt = np.split(network_outputs['program_fields']['box'], 4, axis=-1)

        top = updater.image_height * 0.5 * (yt - ys + 1)
        height = updater.image_height * ys
        bottom = top + height

        left = updater.image_width * 0.5 * (xt - xs + 1)
        width = updater.image_width * xs
        right = left + width

        for idx, a in enumerate(updater.annotations):
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
        _map = _map.reshape((-1, 1, 1, 1, 1))
        return -_map


class YoloRL_Updater(Updater):
    pixels_per_cell = Param()
    image_shape = Param()
    A = Param(help="Dimension of attribute vector.")
    anchor_boxes = Param(help="List of (h, w) pairs.")
    object_shape = Param()

    use_input_attention = Param()
    decoder_logit_scale = Param()

    diff_weight = Param()
    rl_weight = Param()

    obj_sparsity = Param()

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

    fixed_box = Param()
    fixed_obj = Param()
    fixed_attr = Param()
    fixed_object_decoder = Param()

    fix_values = Param()
    order = Param()

    eval_modes = "rl_val diff_val".split()

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

        assert self.diff_weight > 0 or self.rl_weight > 0

        self.COST_funcs = {}

        if self.nonzero_weight is not None:
            cost_func = specific_nonzero_cost if self.use_specific_costs else nonzero_cost
            self.COST_funcs['nonzero'] = (self.nonzero_weight, cost_func)

        if self.area_weight > 0.0:
            cost_func = specific_area_cost if self.use_specific_costs else area_cost
            self.COST_funcs['area'] = (self.area_weight, cost_func)

        cost_func = specific_reconstruction_cost if self.use_specific_costs else reconstruction_cost
        self.COST_funcs['reconstruction'] = (1, cost_func)
        self.COST_funcs['negative_mAP'] = (0, negative_mAP_cost)

        self.scope = scope
        self._n_experiences = 0
        self._n_updates = 0

        self.object_decoder = None

        if isinstance(self.order, str):
            self.order = self.order.split()
        assert set(self.order) == set("box obj attr".split())

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
            attr=dict(
                rep_builder=self._build_attr,
                fixed=self.fixed_box,
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

    def _update(self, batch_size, collect_summaries):
        feed_dict = self.make_feed_dict(batch_size, 'train', False)
        sess = tf.get_default_session()

        result = {}

        if self.diff_weight > 0:
            feed_dict[self.diff] = True

            diff_summary = b''
            if collect_summaries:
                _, diff_record, diff_summary = sess.run(
                    [self.diff_train_op, self.recorded_tensors, self.diff_summary_op], feed_dict=feed_dict)
            else:
                _, diff_record = sess.run(
                    [self.diff_train_op, self.recorded_tensors], feed_dict=feed_dict)

            result.update(diff=(diff_record, diff_summary))

        if self.rl_weight > 0:
            feed_dict[self.diff] = False

            sample_feed_dict = self._sample(feed_dict)
            feed_dict.update(sample_feed_dict)

            rl_summary = b''
            if collect_summaries:
                _, rl_record, rl_summary = sess.run(
                    [self.rl_train_op, self.rl_recorded_tensors, self.rl_summary_op], feed_dict=feed_dict)
            else:
                _, rl_record = sess.run(
                    [self.rl_train_op, self.rl_recorded_tensors], feed_dict=feed_dict)

            result.update(rl=(rl_record, rl_summary))

        return result

    def _process_cost(self, cost):
        assert cost.ndim == 5
        return cost * np.ones((1, self.H, self.W, self.B, 1))

    def _sample(self, feed_dict):
        sess = tf.get_default_session()

        network_outputs = sess.run(self.network_outputs, feed_dict=feed_dict)

        sample_feed_dict = {self.samples[k]: v for k, v in network_outputs['samples'].items()}

        for name, (_, f) in self.COST_funcs.items():
            sample_feed_dict[self.COST_components[name]] = self._process_cost(f(network_outputs, self))

        return sample_feed_dict

    def _evaluate(self, batch_size, mode):
        assert mode in self.eval_modes

        sess = tf.get_default_session()
        feed_dicts = self.make_feed_dict(batch_size, 'val', True, whole_epoch=True)

        record = collections.defaultdict(float)
        summary = b''
        n_points = 0

        for _batch_size, feed_dict in feed_dicts:
            if mode == "rl_val":
                feed_dict[self.diff] = False

                sample_feed_dict = self._sample(feed_dict)
                feed_dict.update(sample_feed_dict)

                _record, summary = sess.run(
                    [self.rl_recorded_tensors, self.rl_summary_op], feed_dict=feed_dict)

            elif mode == "diff_val":
                feed_dict[self.diff] = True

                _record, summary = sess.run(
                    [self.recorded_tensors, self.diff_summary_op], feed_dict=feed_dict)

            for k, v in _record.items():
                record[k] += _batch_size * v

            n_points += _batch_size

        for k, v in record.items():
            record[k] /= n_points

        return record, summary

    def _make_feed_dict(self, batch, evaluate):
        if len(batch) == 1:
            inp, self.annotations = batch[0], None
        elif len(batch) == 2:
            inp, self.annotations = batch
        else:
            raise Exception()

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
        return inp.shape[0], feed_dict

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
            actual_batch_size, feed_dict = self._make_feed_dict(batch, evaluate)

            if whole_epoch:
                dicts.append((actual_batch_size, feed_dict))

                if dset.epochs_completed != epochs_before:
                    break
            else:
                return feed_dict

        return dicts

    def _build_placeholders(self):
        self.inp_ph = tf.placeholder(tf.float32, (None,) + self.obs_shape, name="inp_ph")
        self.inp = tf.clip_by_value(self.inp_ph, 1e-6, 1-1e-6, name="inp")
        self.inp_mode = tf.placeholder(tf.float32, (None, self.image_depth), name="inp_mode_ph")

        self.diff = tf.placeholder(tf.bool, ())
        self.float_diff = tf.to_float(self.diff)

        self.is_training = tf.placeholder(tf.bool, ())
        self.float_is_training = tf.to_float(self.is_training)

        self.batch_size = tf.shape(self.inp)[0]
        H, W, B = self.H, self.W, self.B

        self.COST_components = {}
        self.COST = tf.zeros((self.batch_size, H, W, B, 1))

        for name, (weight, _) in self.COST_funcs.items():
            cost = self.COST_components[name] = tf.placeholder(
                tf.float32, (None, H, W, B, 1), name="COST_{}_ph".format(name))
            weight = build_scheduled_value(weight, "COST_{}_weight".format(name))
            self.COST += weight * cost

    def _build_box(self, box_mean_logits, is_training):
        H, W, B = self.H, self.W, self.B
        image_height, image_width = self.image_height, self.image_width

        box_std = build_scheduled_value(self.box_std, name="box_std")
        cell_yx_logits, hw_logits = tf.split(box_mean_logits, 2, axis=-1)

        # ------

        cell_yx = tf.nn.sigmoid(tf.clip_by_value(cell_yx_logits, -10., 10.))

        cell_y, cell_x = tf.split(cell_yx, 2, axis=-1)
        if "cell_y" in self.fix_values:
            cell_y = float(self.fix_values["cell_y"]) * tf.ones_like(cell_y, dtype=tf.float32)
        if "cell_x" in self.fix_values:
            cell_x = float(self.fix_values["cell_x"]) * tf.ones_like(cell_x, dtype=tf.float32)
        cell_yx = tf.concat([cell_y, cell_x], axis=-1)

        cell_yx_noise = tf.random_normal(tf.shape(cell_yx), name="cell_yx_noise")
        noisy_cell_yx = cell_yx + cell_yx_noise * box_std * self.float_is_training

        noisy_cell_y, noisy_cell_x = tf.split(noisy_cell_yx, 2, axis=-1)

        noisy_y = (
            (noisy_cell_y + tf.range(H, dtype=tf.float32)[None, :, None, None, None]) *
            (self.pixels_per_cell[0] / self.image_shape[0])
        )
        noisy_x = (
            (noisy_cell_x + tf.range(W, dtype=tf.float32)[None, None, :, None, None]) *
            (self.pixels_per_cell[1] / self.image_shape[1])
        )

        # Transform to the co-ordinate system expected by AffineGridWarper
        yt = 2 * noisy_y - 1
        xt = 2 * noisy_x - 1

        # ------

        hw = float(self.max_hw - self.min_hw) * tf.nn.sigmoid(tf.clip_by_value(hw_logits, -10., 10.)) + self.min_hw

        h, w = tf.split(hw, 2, axis=-1)
        if "h" in self.fix_values:
            h = float(self.fix_values["h"]) * tf.ones_like(h, dtype=tf.float32)
        if "w" in self.fix_values:
            w = float(self.fix_values["w"]) * tf.ones_like(w, dtype=tf.float32)
        hw = tf.concat([h, w], axis=-1)

        hw_noise = tf.random_normal(tf.shape(hw), name="hw_noise")

        noisy_hw = hw + hw_noise * box_std * self.float_is_training

        normalized_anchor_boxes = self.anchor_boxes / [image_height, image_width]
        normalized_anchor_boxes = normalized_anchor_boxes.reshape(1, 1, 1, B, 2)

        noisy_hw = noisy_hw * normalized_anchor_boxes
        noisy_h, noisy_w = tf.split(noisy_hw, 2, axis=-1)

        # Transform to the co-ordinate system expected by AffineGridWarper
        ys = noisy_h
        xs = noisy_w

        # ------

        self.area = (ys * float(self.image_height)) * (xs * float(self.image_width))

        self.cell_y, self.cell_x = tf.split(cell_yx, 2, axis=-1)
        self.h, self.w = tf.split(hw, 2, axis=-1)

        self.samples.update(cell_yx=cell_yx_noise, hw=hw_noise)

        box_representation = tf.concat([xs, xt, ys, yt], axis=-1)
        return box_representation

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

        obj_representation = self.float_diff * obj_params + (1 - self.float_diff) * obj_samples

        if "obj" in self.fix_values:
            obj_representation = float(self.fix_values["obj"]) * tf.ones_like(obj_representation, dtype=tf.float32)

        self.obj_logits = obj_logits
        self.samples["obj"] = obj_samples
        self.entropy["obj"] = obj_entropy
        self.log_probs["obj"] = obj_log_probs

        return obj_representation

    def _build_attr(self, attr_mean, is_training):
        attr_std = build_scheduled_value(self.attr_std, name="attr_std")
        attr_noise = tf.random_normal(tf.shape(attr_mean), name="attr_noise")

        noisy_attr = attr_mean + attr_noise * attr_std * self.float_is_training

        self.samples["attr"] = attr_noise
        self.attr = noisy_attr

        return noisy_attr

    def _build_program_generator(self):
        inp, is_training = self.inp, self.is_training
        H, W, B = self.H, self.W, self.B
        program, features = None, None

        self.program_fields = {}
        self.samples = {}
        self.entropy = {}
        self.log_probs = {}

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

            self.program_fields[kind] = representation

            if i:
                program = tf.concat([program, representation], axis=-1)
            else:
                program = representation

        # --- finalize ---
        self.program = program

        self.network_outputs['samples'] = self.samples
        self.network_outputs['area'] = self.area
        self.network_outputs['program_fields'] = self.program_fields

    def _build_program_interpreter(self):
        H, W, B, A = self.H, self.W, self.B, self.A
        object_shape, image_height, image_width, image_depth = (
            self.object_shape, self.image_height, self.image_width, self.image_depth)

        boxes = self.program_fields['box']
        _boxes = tf.reshape(boxes, (-1, 4))

        obj = self.program_fields['obj']
        attr = self.program_fields['attr']

        object_decoder_in = tf.reshape(attr, (-1, 1, 1, A))

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()

        warper = snt.AffineGridWarper(
            (image_height, image_width), object_shape, transform_constraints)
        inverse_warper = warper.inverse()

        if self.use_input_attention:
            grid_coords = warper(_boxes)
            grid_coords = tf.reshape(grid_coords, (self.batch_size, H, W, B,) + object_shape + (2,))
            input_glimpses = tf.contrib.resampler.resampler(self.inp, grid_coords)
            input_glimpses = tf.reshape(input_glimpses, (-1,) + object_shape + (image_depth,))
            object_decoder_in = [object_decoder_in, input_glimpses]

        object_decoder_output = self.object_decoder(object_decoder_in, object_shape + (image_depth+1,), self.is_training)

        if isinstance(self.object_decoder, VQ_ObjectDecoder):
            vq = self.object_decoder._vq

            z_e = tf.reshape(vq.z_e, (self.batch_size, H, W, B) + self.object_decoder.vq_input_shape)
            z_q = tf.reshape(vq.z_q, (self.batch_size, H, W, B) + self.object_decoder.vq_input_shape)

            nonzero = tf.maximum(tf.to_float(tf.count_nonzero(obj)), 1.0)
            p = self.object_decoder.vq_input_shape[0] * self.object_decoder.vq_input_shape[1]
            N = tf.to_float(nonzero) * float(p)

            commitment_error = obj[..., None, None] * (z_e - tf.stop_gradient(z_q))**2
            self.decoder_commitment_error = tf.reduce_sum(commitment_error) / N

            embedding_error = obj[..., None, None] * (tf.stop_gradient(z_e) - z_q)**2
            self.decoder_embedding_error = tf.reduce_sum(embedding_error) / N

        object_decoder_output = tf.nn.sigmoid(
            self.decoder_logit_scale * tf.clip_by_value(object_decoder_output, -10., 10.))

        _object_decoder_output = tf.reshape(
            object_decoder_output, (-1, H, W, B,) + object_shape + (image_depth+1,))
        self.object_decoder_images, self.object_decoder_alpha = tf.split(_object_decoder_output, [image_depth, 1], axis=-1)

        if "alpha" in self.fix_values:
            images, alpha = tf.split(object_decoder_output, [image_depth, 1], axis=-1)
            alpha = float(self.fix_values["alpha"]) * tf.ones_like(alpha, dtype=tf.float32)
            object_decoder_output = tf.concat([images, alpha], axis=-1)

        grid_coords = inverse_warper(_boxes)

        # --- build predicted images ---

        object_decoder_transformed = tf.contrib.resampler.resampler(object_decoder_output, grid_coords)
        object_decoder_transformed = tf.reshape(
            object_decoder_transformed,
            [-1, H, W, B, image_height, image_width, image_depth+1]
        )

        transformed_images, transformed_alphas = tf.split(object_decoder_transformed, [image_depth, 1], axis=-1)

        weighted_images = obj[..., None, None] * transformed_images
        self.foreground = tf.reduce_max(
            tf.reshape(weighted_images, [-1, H*W*B, image_height, image_width, image_depth]),
            axis=1,
        )

        weighted_alphas = obj[..., None, None] * transformed_alphas
        self.alpha = tf.reduce_max(
            tf.reshape(weighted_alphas, [-1, H*W*B, image_height, image_width, 1]),
            axis=1,
        )

        self.output = self.alpha * self.foreground + (1 - self.alpha) * self.background

        self.output = tf.clip_by_value(self.output, 1e-6, 1-1e-6)
        self.output_logits = tf.log(self.output / (1 - self.output))

        self.network_outputs['output'] = self.output
        self.network_outputs['output_logits'] = self.output_logits

    def build_area_loss(self, batch=True):
        selected_area = self.area * self.program_fields['obj']

        if batch:
            selected_area = tf_flatten(selected_area)
            return tf.reduce_sum(selected_area, axis=1, keep_dims=True)
        else:
            return selected_area

    def build_xent_loss(self, logits, targets, batch=True):
        per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

        if batch:
            per_pixel_loss = tf_flatten(per_pixel_loss)
            return tf.reduce_sum(per_pixel_loss, keep_dims=True, axis=1)
        else:
            return per_pixel_loss

    def build_squared_loss(self, logits, targets, batch=True):
        actions = tf.sigmoid(logits)
        per_pixel_loss = (actions - targets)**2

        if batch:
            per_pixel_loss = tf_flatten(per_pixel_loss)
            return tf.reduce_sum(per_pixel_loss, keep_dims=True, axis=1)
        else:
            return per_pixel_loss

    def build_1norm_loss(self, logits, targets, batch=True):
        actions = tf.sigmoid(logits)
        per_pixel_loss = tf.abs(actions - targets)

        if batch:
            per_pixel_loss = tf_flatten(per_pixel_loss)
            return tf.reduce_sum(per_pixel_loss, keep_dims=True, axis=1)
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

        recorded_tensors = {}

        reconstruction_loss = getattr(self, 'build_' + loss_key)(self.output_logits, self.inp)
        mean_reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        per_pixel_reconstruction_loss = getattr(self, 'build_' + loss_key)(self.output_logits, self.inp, batch=False)

        area_loss = self.build_area_loss()
        mean_area_loss = tf.reduce_mean(area_loss)

        # --- network output ---

        self.network_outputs.update(
            per_pixel_reconstruction_loss=per_pixel_reconstruction_loss,
            reconstruction_loss=reconstruction_loss,
            area_loss=area_loss
        )

        # --- entropy ---

        recorded_tensors['obj_entropy'] = tf_mean_sum(self.entropy['obj'])

        # --- recorded values ---

        recorded_tensors.update({
            name: tf.reduce_mean(getattr(self, 'build_' + name)(self.output_logits, self.inp))
            for name in ['xent_loss', 'squared_loss', '1norm_loss']
        })

        recorded_tensors['attr'] = tf.reduce_mean(self.attr)

        recorded_tensors['cell_y'] = tf.reduce_mean(self.cell_y)
        recorded_tensors['cell_x'] = tf.reduce_mean(self.cell_x)

        recorded_tensors['h'] = tf.reduce_mean(self.h)
        recorded_tensors['w'] = tf.reduce_mean(self.w)

        recorded_tensors['obj_logits'] = tf.reduce_mean(self.obj_logits)
        recorded_tensors['obj'] = tf.reduce_mean(self.program_fields['obj'])

        recorded_tensors['reconstruction_loss'] = mean_reconstruction_loss
        recorded_tensors['area_loss'] = mean_area_loss

        recorded_tensors['diff_loss'] = mean_reconstruction_loss

        if self.area_weight > 0.0:
            recorded_tensors['diff_loss'] += self.area_weight * mean_area_loss

        if self.obj_sparsity:
            recorded_tensors['obj_sparsity_loss'] = tf_mean_sum(self.program_fields['obj'])
            recorded_tensors['diff_loss'] += self.obj_sparsity * recorded_tensors['obj_sparsity_loss']

        if isinstance(self.object_decoder, VQ_ObjectDecoder):
            recorded_tensors['decoder_commitment_error'] = self.decoder_commitment_error
            recorded_tensors['decoder_embedding_error'] = self.decoder_embedding_error

            recorded_tensors['diff_loss'] += recorded_tensors['decoder_embedding_error']
            recorded_tensors['diff_loss'] += self.object_decoder.beta * recorded_tensors['decoder_commitment_error']

        self.diff_loss = recorded_tensors['diff_loss']
        self.recorded_tensors = recorded_tensors

        # --- rl recorded values ---

        _rl_recorded_tensors = {}

        for name, _ in self.COST_funcs.items():
            _rl_recorded_tensors["COST_{}".format(name)] = tf.reduce_mean(self.COST_components[name])

        _rl_recorded_tensors["COST"] = tf.reduce_mean(self.COST)
        _rl_recorded_tensors["TOTAL_COST"] = _rl_recorded_tensors["COST"]

        _rl_recorded_tensors["obj_log_probs"] = tf.reduce_mean(self.log_probs['obj'])

        if self.use_baseline:
            adv = self.COST - tf.reduce_mean(self.COST, axis=0, keep_dims=True)
        else:
            adv = self.COST

        self.rl_surrogate_loss_map = adv * self.log_probs['obj']
        _rl_recorded_tensors['rl_loss'] = tf.reduce_mean(self.rl_surrogate_loss_map)

        _rl_recorded_tensors['surrogate_loss'] = _rl_recorded_tensors['rl_loss'] + mean_reconstruction_loss

        if self.area_weight > 0.0:
            _rl_recorded_tensors['surrogate_loss'] += self.area_weight * mean_area_loss

        if isinstance(self.object_decoder, VQ_ObjectDecoder):
            _rl_recorded_tensors['surrogate_loss'] += recorded_tensors['decoder_embedding_error']
            _rl_recorded_tensors['surrogate_loss'] += self.object_decoder.beta * recorded_tensors['decoder_commitment_error']

        self.surrogate_loss = _rl_recorded_tensors['surrogate_loss']

        self.rl_recorded_tensors = _rl_recorded_tensors.copy()
        self.rl_recorded_tensors.update(self.recorded_tensors)

        _summary = [tf.summary.scalar(name, t) for name, t in self.recorded_tensors.items()]
        _rl_summary = [tf.summary.scalar(name, t) for name, t in _rl_recorded_tensors.items()]

        # --- rl train op ---

        tvars = self.trainable_variables(for_opt=True)

        self.rl_train_op, rl_train_summary = build_gradient_train_op(
            self.surrogate_loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule, summary_prefix="surrogate")

        self.rl_summary_op = tf.summary.merge(_summary + _rl_summary + rl_train_summary)

        # --- diff train op ---

        self.diff_train_op, diff_train_summary = build_gradient_train_op(
            self.diff_loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.diff_summary_op = tf.summary.merge(_summary + diff_train_summary)


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

        rl_fetched = self._fetch(self.N, updater, True)

        self._plot_reconstruction(updater, rl_fetched, True)
        self._plot_patches(updater, rl_fetched, True, 4)

        diff_fetched = self._fetch(self.N, updater, False)

        self._plot_reconstruction(updater, diff_fetched, False)
        self._plot_patches(updater, diff_fetched, False, 4)

    def _fetch(self, N, updater, sampled):
        feed_dict = updater.make_feed_dict(N, 'val', True)
        images = feed_dict[updater.inp]
        feed_dict[updater.diff] = not sampled

        to_fetch = updater.program_fields.copy()
        to_fetch["output"] = updater.output
        to_fetch["object_decoder_images"] = updater.object_decoder_images
        to_fetch["object_decoder_alpha"] = updater.object_decoder_alpha
        to_fetch["background"] = updater.background

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        fetched.update(images=images)

        fetched["annotations"] = updater.annotations

        return fetched

    def _plot_reconstruction(self, updater, fetched, sampled):
        images = fetched['images']
        background = fetched['background']
        N = images.shape[0]

        output = fetched['output']

        _, image_height, image_width, _ = images.shape

        obj = fetched['obj'].reshape(N, -1)

        annotations = fetched["annotations"]
        if annotations is None:
            annotations = [[]] * N

        xs, xt, ys, yt = np.split(fetched['box'].reshape(-1, 4), 4, axis=-1)

        top = image_height * 0.5 * (yt - ys + 1)
        left = image_width * 0.5 * (xt - xs + 1)
        height = image_height * ys
        width = image_width * xs

        box = np.concatenate([top, left, height, width], axis=-1)
        box = box.reshape(N, -1, 4)

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
                _, top, bottom, left, right = a
                h = bottom - top
                w = right - left

                rect = patches.Rectangle(
                    (left, top), w, h, linewidth=1, edgecolor="xkcd:yellow", facecolor='none')
                ax1.add_patch(rect)

                rect = patches.Rectangle(
                    (left, top), w, h, linewidth=1, edgecolor="xkcd:yellow", facecolor='none')
                ax2.add_patch(rect)

            ax3 = axes[3*i+2, j]
            ax3.imshow(bg)
            ax3.set_title('background')

        fig.suptitle('Sampled={}. Stage={}. After {} experiences ({} updates, {} experiences per batch).'.format(
            sampled, updater.stage_idx, updater.n_experiences, updater.n_updates, cfg.batch_size))

        plot_name = ('sampled_' if sampled else '') + 'reconstruction.pdf'
        path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), plot_name)
        fig.savefig(path)

        plt.close(fig)

    def _plot_patches(self, updater, fetched, sampled, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        object_decoder_images = fetched['object_decoder_images']
        object_decoder_alpha = fetched['object_decoder_alpha']

        H, W, B = updater.H, updater.W, updater.B

        obj = fetched['obj']

        for idx in range(N):
            fig, axes = plt.subplots(2*H, W * B, figsize=(20, 20))
            axes = np.array(axes).reshape(2*H, W * B)

            for i in range(H):
                for j in range(W):
                    for b in range(B):
                        ax = axes[2*i, j * B + b]

                        _obj = obj[idx, i, j, b, 0]

                        ax.set_title("obj = {}".format(_obj))
                        ax.set_xlabel("b = {}".format(b))

                        if b == 0:
                            ax.set_ylabel("grid_cell: ({}, {})".format(i, j))

                        ax.imshow(object_decoder_images[idx, i, j, b])

                        ax = axes[2*i+1, j * B + b]
                        ax.imshow(object_decoder_alpha[idx, i, j, b, :, :, 0])

            dir_name = ('sampled_' if sampled else '') + 'patches'
            path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), dir_name, '{}.pdf'.format(idx))
            fig.savefig(path)
            plt.close(fig)


xkcd_colors = 'viridian,cerulean,vermillion,lavender,celadon,fuchsia,saffron,cinnamon,greyish,vivid blue'.split(',')


diff_mode = dict(rl_weight=0.0, diff_weight=1.0)
rl_mode = dict(rl_weight=1.0, diff_weight=0.0)
combined_mode = dict(rl_weight=1.0, diff_weight=1.0)


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
    eval_mode="rl_val",
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

    diff_weight=0.0,
    rl_weight=0.0,

    obj_sparsity=0.0,  # Within a single image, we want as few bounding boxes to be active as possible

    max_hw=1.0,  # Maximum for the bounding box multiplier.
    min_hw=0.0,  # Minimum for the bounding box multiplier.

    # VAE
    box_std=0.1,
    attr_std=0.0,

    obj_exploration=0.05,
    obj_default=0.5,

    # Costs
    use_baseline=True,
    nonzero_weight=0.0,
    area_weight=0.0,
    use_specific_costs=False,

    curriculum=[
        rl_mode,
    ],

    fixed_box=False,
    fixed_obj=False,
    fixed_attr=False,
    fixed_object_decoder=False,

    fix_values=dict(),
    order="box obj attr",

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

    sub_image_size_std=0.4,
    max_chars=3,
    min_chars=1,
    characters=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

    fix_values=dict(),
    obj_exploration=0.0,
    lr_schedule=1e-4,
    use_specific_costs=True,

    curriculum=[
        dict(fix_values=dict(obj=1), max_steps=10000, area_weight=0.02),
        dict(obj_exploration=0.2),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
    ],

    nonzero_weight=90.,
    area_weight=0.1,

    box_std=0.1,
    attr_std=0.0,

    render_step=5000,

    n_distractors_per_image=0,

    **rl_mode,
)


uniform_size_config = good_config.copy(
    max_hw=1.5,
    min_hw=0.5,
    sub_image_size_std=0.0,
)


uniform_size_reset_config = uniform_size_config.copy(
    box_std=0.0,
    curriculum=[
        dict(obj_exploration=0.2,
             load_path="/media/data/dps_data/logs/yolo_rl/exp_yolo_rl_seed=347405995_2018_04_05_09_50_56/weights/best_of_stage_1",),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
    ]
)

small_test_config = uniform_size_config.copy(
    kernel_size=(3, 3),
    min_chars=1,
    max_chars=3,
    object_shape=(14, 14),
    anchor_boxes=[[7, 7]],
    image_shape=(28, 28),
    max_overlap=100,
)


with_background_config = good_config.copy(
    backgrounds="red_x blue_x green_x red_circle blue_circle green_circle",
    backgrounds_resize=True,
    image_shape=(48, 48),

    curriculum=[
        dict(
            fix_values=dict(obj=0), patience=5000,
            area_weight=0.00,
            nonzero_weight=0.00,
            fixed_object_decoder=True,
            fixed_box=True,
            fixed_attr=True,
            fixed_obj=True,),
        dict(
            fix_values=dict(obj=1, alpha=1-1e-6),
            area_weight=0.0,
            patience=5000,),
        dict(
            fix_values=dict(obj=1, alpha=1-1e-6),
            area_weight=0.02,
            patience=5000,)
    ],

    max_hw=1.5,
    min_hw=0.5,
    sub_image_size_std=0.0,
)

vq_config = good_config.copy(
    # VQ object decoder params
    build_object_decoder=VQ_ObjectDecoder,
    beta=4.0,
    vq_input_shape=(2, 2, 25),
    K=5,
    common_embedding=False,
)
