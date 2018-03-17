import tensorflow as tf
import numpy as np
import sonnet as snt

from dps import cfg
from dps.train import PolynomialScheduleHook
from dps.datasets import EMNIST_ObjectDetection
from dps.updater import Updater
from dps.utils import Config, Param
from dps.utils.tf import (
    FullyConvolutional, build_gradient_train_op,
    trainable_variables, build_scheduled_value,
    tf_normal_kl, tf_mean_sum, ScopedFunction
)

tf_flatten = tf.layers.flatten


class Env(object):
    pass


def build_env():
    return Env()


def get_updater(env):
    return YoloRL_Updater()


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


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
    n_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_channels, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_channels, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_channels, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=3, kernel_size=4, strides=1, padding="SAME", transpose=True),  # For 14 x 14 output
        ]
        super(ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)


class ObjectEncoderDecoder(FullyConvolutional):
    n_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_channels, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_channels, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_channels, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=3, kernel_size=4, strides=1, padding="SAME", transpose=True),  # For 14 x 14 output
        ]
        super(ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)


class PassthroughDecoder(ScopedFunction):
    def _call(self, inp, output_shape, is_training):
        _, input_glimpses = inp
        return input_glimpses


class ObjectDecoder28x28(FullyConvolutional):
    n_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_channels, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_channels, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_channels, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=3, kernel_size=3, strides=2, padding="SAME", transpose=True),  # For 28 x 28 output
        ]
        super(ObjectDecoder28x28, self).__init__(layout, check_output_shape=True, **kwargs)


class StaticObjectDecoder(ScopedFunction):
    """ An object decoder that outputs a learnable image, ignoring input. """
    def __init__(self, initializer=None, **kwargs):
        super(StaticObjectDecoder, self).__init__(**kwargs)
        if initializer is None:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.initializer = initializer

    def _call(self, inp, output_shape, is_training):
        output = tf.get_variable(
            "learnable_bias",
            shape=output_shape,
            dtype=tf.float32,
            initializer=self.initializer,
            trainable=True,
        )
        return tf.tile(output[None, ...], (tf.shape(inp)[0],) + tuple(1 for s in output_shape))


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

    for b in range(updater.B):
        max_distance_h = updater.pixels_per_cell[0] / 2 + updater.max_hw * updater.anchor_boxes[b, 0] / 2
        max_distance_w = updater.pixels_per_cell[1] / 2 + updater.max_hw * updater.anchor_boxes[b, 1] / 2

        # Rectangle filtering
        filt_h = (dist_h < max_distance_h).astype('f')
        filt_w = (dist_w < max_distance_w).astype('f')

        # Gaussian filtering
        # h_means = (np.arange(H) + 0.5) * network_outputs["pixels_per_cell"][0]
        # w_means = (np.arange(W) + 0.5) * network_outputs["pixels_per_cell"][1]

        # std = 1.0

        # filt_h = np.exp(-0.5 * (loc_h - h_means)**2 / std)
        # filt_w = np.exp(-0.5 * (loc_w - w_means)**2 / std)

        # if False:  # Normalize
        #     filt_h /= np.sqrt(2 * np.pi) * std
        #     filt_w /= np.sqrt(2 * np.pi) * std

        signal = network_outputs['per_pixel_reconstruction_loss']

        # Sum over channel dimension
        signal = signal.sum(axis=-1)

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


class YoloRL_Updater(Updater):
    pixels_per_cell = Param()
    image_shape = Param()
    C = Param()
    A = Param(help="Dimension of attribute vector.")
    anchor_boxes = Param(help="List of (h, w) pairs.")
    object_shape = Param()

    use_input_attention = Param()
    decoders_output_logits = Param()
    decoder_logit_scale = Param()

    diff_weight = Param()
    rl_weight = Param()

    obj_sparsity = Param()
    cls_sparsity = Param()

    max_hw = Param()
    min_hw = Param()

    box_std = Param()
    attr_std = Param()
    minimize_kl = Param()

    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    pass_samples = Param()
    n_passthrough_features = Param()
    pass_boxes_to_decoder = Param()

    xent_loss = Param()

    use_baseline = Param()
    nonzero_weight = Param()
    area_weight = Param()
    use_specific_costs = Param()
    use_specific_reconstruction = Param()

    obj_exploration = Param()
    obj_default = Param()
    cls_exploration = Param()

    fixed_box = Param()
    fixed_obj = Param()
    fixed_cls = Param()
    fixed_attr = Param()

    fixed_object_decoder = Param()

    fix_values = Param()
    dynamic_partition = Param()
    order = Param()

    eval_modes = "rl_val diff_val".split()

    def __init__(self, scope=None, **kwargs):
        self.anchor_boxes = np.array(self.anchor_boxes)
        self.H = int(np.ceil(self.image_shape[0] / self.pixels_per_cell[0]))
        self.W = int(np.ceil(self.image_shape[1] / self.pixels_per_cell[1]))
        self.B = len(self.anchor_boxes)

        self._make_datasets()

        self.obs_shape = self.datasets['train'].x.shape[1:]
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        assert self.diff_weight > 0 or self.rl_weight > 0

        self.COST_funcs = {}

        if self.nonzero_weight is not None:
            cost_func = specific_nonzero_cost if self.use_specific_costs else nonzero_cost
            self.COST_funcs['nonzero'] = (self.nonzero_weight, nonzero_cost, "obj")

        if self.area_weight > 0.0:
            cost_func = specific_area_cost if self.use_specific_costs else area_cost
            self.COST_funcs['area'] = (self.area_weight, cost_func, "obj")

        cost_func = specific_reconstruction_cost if self.use_specific_reconstruction else reconstruction_cost
        self.COST_funcs['reconstruction'] = (1, cost_func, "both")

        self.scope = scope
        self._n_experiences = 0
        self._n_updates = 0

        self.object_decoders = None

        self.predict_box_std = False
        try:
            box_std = float(self.box_std)
            self.predict_box_std = box_std < 0
        except (TypeError, ValueError):
            pass

        self.predict_attr_std = False
        try:
            attr_std = float(self.attr_std)
            self.predict_attr_std = attr_std < 0
        except (TypeError, ValueError):
            pass

        if isinstance(self.order, str):
            self.order = self.order.split()
        assert set(self.order) == set("box obj cls attr".split())
        assert len(self.order) == 4

        self.layer_params = dict(
            box=dict(
                rep_builder=self._build_box,
                fixed=self.fixed_box,
                output_size=4*(2 if self.predict_box_std else 1),
                network=None
            ),
            obj=dict(
                rep_builder=self._build_obj,
                fixed=self.fixed_obj,
                output_size=1,
                network=None
            ),
            cls=dict(
                rep_builder=self._build_cls,
                fixed=self.fixed_cls,
                output_size=self.C,
                network=None
            ),
            attr=dict(
                rep_builder=self._build_attr,
                fixed=self.fixed_box,
                output_size=self.A*(2 if self.predict_attr_std else 1),
                network=None
            ),
        )

    def _make_datasets(self):
        train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train), shuffle=True)
        val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val), shuffle=True)

        self.datasets = dict(train=train, val=val)

    @property
    def completion(self):
        return self.datasets['train'].completion

    def trainable_variables(self, for_opt, rl_only=False):
        scoped_functions = (
            self.object_decoders +
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
        network_outputs['shape'] = self.H, self.W, self.B

        sample_feed_dict = {self.samples[k]: v for k, v in network_outputs['samples'].items()}

        for name, (_, f, _) in self.COST_funcs.items():
            sample_feed_dict[self.COST_components[name]] = self._process_cost(f(network_outputs, self))

        return sample_feed_dict

    def _evaluate(self, batch_size, mode):
        assert mode in self.eval_modes

        feed_dict = self.make_feed_dict(None, 'val', True)

        sess = tf.get_default_session()

        record, summary = {}, b''

        if mode == "rl_val":
            feed_dict[self.diff] = False

            sample_feed_dict = self._sample(feed_dict)
            feed_dict.update(sample_feed_dict)

            record, summary = sess.run(
                [self.rl_recorded_tensors, self.rl_summary_op], feed_dict=feed_dict)

        elif mode == "diff_val":
            feed_dict[self.diff] = True

            record, summary = sess.run(
                [self.recorded_tensors, self.diff_summary_op], feed_dict=feed_dict)

        return record, summary

    def make_feed_dict(self, batch_size, mode, evaluate):
        inp, *_ = self.datasets[mode].next_batch(batch_size=batch_size, advance=not evaluate)
        return {self.inp: inp, self.is_training: not evaluate}

    def _build_placeholders(self):
        inp = tf.placeholder(tf.float32, (None,) + self.obs_shape, name="inp_ph")
        self.inp = tf.clip_by_value(inp, 1e-6, 1-1e-6, name="inp")

        self.diff = tf.placeholder(tf.bool, ())
        self.float_diff = tf.to_float(self.diff)

        self.is_training = tf.placeholder(tf.bool, ())
        self.float_is_training = tf.to_float(self.is_training)

        self.batch_size = tf.shape(self.inp)[0]
        H, W, B = self.H, self.W, self.B

        self.COST = tf.zeros((self.batch_size, H, W, B, 1))
        self.COST_obj = tf.zeros((self.batch_size, H, W, B, 1))
        self.COST_cls = tf.zeros((self.batch_size, H, W, B, 1))
        self.COST_components = {}

        for name, (weight, _, kind) in self.COST_funcs.items():
            cost = self.COST_components[name] = tf.placeholder(
                tf.float32, (None, H, W, B, 1), name="COST_{}_ph".format(name))

            weight = build_scheduled_value(weight, "COST_{}_weight".format(name))

            if kind == "both":
                self.COST += weight * cost
            elif kind == "obj":
                self.COST_obj += weight * cost
            elif kind == "cls":
                self.COST_cls += weight * cost
            else:
                raise Exception("Unknown kind {}".format(kind))

    def _build_box(self, box_params, is_training):
        H, W, B = self.H, self.W, self.B
        image_height, image_width = self.image_height, self.image_width

        if self.predict_box_std:
            box_mean_logits, box_std_logits = tf.split(box_params, 2, axis=-1)
            box_std = tf.nn.sigmoid(tf.clip_by_value(box_std_logits, -10., 10.))
        else:
            box_mean_logits = box_params
            _box_std = build_scheduled_value(self.box_std)
            box_std = _box_std * tf.ones_like(box_mean_logits)

        cell_yx_logits, hw_logits = tf.split(box_mean_logits, 2, axis=-1)
        cell_yx_std, hw_std = tf.split(box_std, 2, axis=-1)

        # ------

        cell_yx = tf.nn.sigmoid(tf.clip_by_value(cell_yx_logits, -10., 10.))

        cell_y, cell_x = tf.split(cell_yx, 2, axis=-1)

        if "cell_y" in self.fix_values:
            cell_y = float(self.fix_values["cell_y"]) * tf.ones_like(cell_y, dtype=tf.float32)
        if "cell_x" in self.fix_values:
            cell_x = float(self.fix_values["cell_x"]) * tf.ones_like(cell_x, dtype=tf.float32)

        cell_yx = tf.concat([cell_y, cell_x], axis=-1)

        cell_yx_noise = tf.random_normal(tf.shape(cell_yx), name="cell_yx_noise")

        noisy_cell_yx = cell_yx + cell_yx_noise * cell_yx_std * self.float_is_training

        # ------

        hw = float(self.max_hw - self.min_hw) * tf.nn.sigmoid(tf.clip_by_value(hw_logits, -10., 10.)) + self.min_hw

        h, w = tf.split(hw, 2, axis=-1)
        if "h" in self.fix_values:
            h = float(self.fix_values["h"]) * tf.ones_like(h, dtype=tf.float32)
        if "w" in self.fix_values:
            w = float(self.fix_values["w"]) * tf.ones_like(w, dtype=tf.float32)
        hw = tf.concat([h, w], axis=-1)

        hw_noise = tf.random_normal(tf.shape(hw), name="hw_noise")

        noisy_hw = hw + hw_noise * hw_std * self.float_is_training

        normalized_anchor_boxes = self.anchor_boxes / [image_height, image_width]
        normalized_anchor_boxes = normalized_anchor_boxes.reshape(1, 1, 1, B, 2)

        noisy_hw = noisy_hw * normalized_anchor_boxes

        # ------

        noisy_cell_y, noisy_cell_x = tf.split(noisy_cell_yx, 2, axis=-1)
        noisy_h, noisy_w = tf.split(noisy_hw, 2, axis=-1)

        ys = noisy_h
        xs = noisy_w

        noisy_y = (
            (noisy_cell_y + tf.range(H, dtype=tf.float32)[None, :, None, None, None]) *
            (self.pixels_per_cell[0] / self.image_shape[0])
        )
        noisy_x = (
            (noisy_cell_x + tf.range(W, dtype=tf.float32)[None, None, :, None, None]) *
            (self.pixels_per_cell[1] / self.image_shape[1])
        )

        yt = 2 * noisy_y - 1
        xt = 2 * noisy_x - 1

        self.area = (ys * float(self.image_height)) * (xs * float(self.image_width))

        self.cell_y, self.cell_x = tf.split(cell_yx, 2, axis=-1)
        self.h, self.w = tf.split(hw, 2, axis=-1)
        self.cell_yx_dist = dict(mean=cell_yx, std=cell_yx_std)
        self.hw_dist = dict(mean=hw, std=hw_std)

        self.samples.update(cell_yx=cell_yx_noise, hw=hw_noise)

        box_representation = tf.concat([xs, xt, ys, yt], axis=-1)
        return box_representation

    def _build_obj(self, obj_logits, is_training):
        obj_logits = tf.clip_by_value(obj_logits, -10., 10.)
        self.obj_logits = obj_logits

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

        self.samples["obj"] = obj_samples
        self.entropy["obj"] = obj_entropy
        self.log_probs["obj"] = obj_log_probs

        return obj_representation

    def _build_cls(self, cls_logits, is_training):
        cls_logits = tf.minimum(10.0, cls_logits)

        self.cls_logits = cls_logits

        cls_params = tf.nn.softmax(cls_logits)

        cls_exploration = build_scheduled_value(self.cls_exploration, "cls_exploration") * self.float_is_training
        cls_params = (1 - cls_exploration) * cls_params + cls_exploration / self.C

        cls_dist = tf.distributions.Categorical(probs=cls_params)

        cls_samples = tf.stop_gradient(cls_dist.sample())
        cls_samples = tf.one_hot(cls_samples, self.C)
        cls_samples = tf.to_float(cls_samples)

        cls_log_probs = cls_dist.log_prob(tf.argmax(cls_samples, axis=-1))[..., None]
        cls_log_probs = tf.where(tf.is_nan(cls_log_probs), -100.0 * tf.ones_like(cls_log_probs), cls_log_probs)

        cls_entropy = cls_dist.entropy()

        cls_representation = self.float_diff * cls_params + (1 - self.float_diff) * cls_samples

        if "cls" in self.fix_values:
            cls_representation = float(self.fix_values["cls"]) * tf.ones_like(cls_representation, dtype=tf.float32)

        self.samples["cls"] = cls_samples
        self.entropy["cls"] = cls_entropy
        self.log_probs["cls"] = cls_log_probs

        return cls_representation

    def _build_attr(self, attr_params, is_training):
        if self.predict_attr_std:
            attr, attr_std_logits = tf.split(attr_params, 2, axis=-1)
            attr_std = tf.exp(attr_std_logits)
        else:
            attr = attr_params
            _attr_std = build_scheduled_value(self.attr_std)
            attr_std = _attr_std * tf.ones_like(attr)

        attr_noise = tf.random_normal(tf.shape(attr_std), name="attr_noise")
        noisy_attr = attr + attr_noise * attr_std * self.float_is_training

        self.attr_dist = dict(mean=attr, std=attr_std)
        self.samples["attr"] = attr_noise

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
                if self.pass_samples:
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
        cls = self.program_fields['cls']
        attr = self.program_fields['attr']

        if self.pass_boxes_to_decoder:
            object_decoder_in = tf.concat([boxes, attr], axis=-1)
            object_decoder_in = tf.reshape(attr, (-1, 1, 1, 4 + A))
        else:
            object_decoder_in = tf.reshape(attr, (-1, 1, 1, A))

        self.object_decoder_output = {}
        per_class_images = {}

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()

        warper = snt.AffineGridWarper(
            (image_height, image_width), object_shape, transform_constraints)
        inverse_warper = warper.inverse()

        for c, od in enumerate(self.object_decoders):
            if self.use_input_attention:
                grid_coords = warper(_boxes)
                grid_coords = tf.reshape(grid_coords, (self.batch_size, H, W, B,) + object_shape + (2,))
                input_glimpses = tf.contrib.resampler.resampler(self.inp, grid_coords)
                input_glimpses = tf.reshape(input_glimpses, (-1,) + object_shape + (image_depth,))
                object_decoder_in = [object_decoder_in, input_glimpses]

            object_decoder_output = od(object_decoder_in, object_shape + (image_depth,), self.is_training)

            if self.decoders_output_logits:
                object_decoder_output = tf.nn.sigmoid(
                    self.decoder_logit_scale * tf.clip_by_value(object_decoder_output, -10., 10.))

            self.object_decoder_output[c] = tf.reshape(
                object_decoder_output, (-1, H, W, B,) + object_shape + (image_depth,))

            grid_coords = inverse_warper(_boxes)

            # --- build predicted images ---

            object_decoder_transformed = tf.contrib.resampler.resampler(object_decoder_output, grid_coords)
            object_decoder_transformed = tf.reshape(
                object_decoder_transformed,
                [-1, H, W, B, image_height, image_width, image_depth]
            )

            weighted_images = (
                cls[..., c:c+1, None, None] *
                obj[..., None, None] *
                object_decoder_transformed
            )
            per_class_images[c] = tf.reshape(weighted_images, [-1, H*W*B, image_height, image_width, image_depth])

        self.output = tf.concat([per_class_images[c] for c in range(self.C)], axis=1)
        self.output = tf.reduce_max(self.output, axis=1)

        _output = tf.clip_by_value(self.output, 1e-6, 1-1e-6)
        self.output_logits = tf.log(_output / (1 - _output))

        self.network_outputs['output'] = self.output
        self.network_outputs['output_logits'] = self.output_logits

    def _build_program_interpreter_with_dynamic_partition(self):
        H, W, B, A = self.H, self.W, self.B, self.A
        object_shape, image_height, image_width, image_depth = (
            self.object_shape, self.image_height, self.image_width, self.image_depth)

        boxes = self.program_fields['box']
        obj = self.program_fields['obj']
        cls = self.program_fields['cls']
        attr = self.program_fields['attr']

        self.object_decoder_output = {}
        self.batch_indices = {}
        self.box_indices = {}

        per_class_images = {}

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()

        warper = snt.AffineGridWarper(
            (image_height, image_width), object_shape, transform_constraints)
        inverse_warper = warper.inverse()

        for c, od in enumerate(self.object_decoders):
            mask = tf.to_int32((obj[..., 0] * cls[..., c]) > 0)

            # Make sure at least one box is active per batch-element.
            mask = tf.maximum(
                mask,
                tf.to_int32(tf.tile(tf.reshape(tf.one_hot(0, H * W * B), (1, H, W, B)), (self.batch_size, 1, 1, 1)))
            )

            boxes_off, boxes_on = tf.dynamic_partition(boxes, mask, 2)
            obj_off, obj_on = tf.dynamic_partition(obj, mask, 2)
            cls_off, cls_on = tf.dynamic_partition(cls[..., c:c+1], mask, 2)
            attr_off, attr_on = tf.dynamic_partition(attr, mask, 2)

            if self.pass_boxes_to_decoder:
                object_decoder_in = tf.concat([boxes_on, attr_on], axis=-1)
                object_decoder_in = tf.reshape(object_decoder_in, (-1, 1, 1, 4 + A))
            else:
                object_decoder_in = tf.reshape(attr_on, (-1, 1, 1, A))

            batch_indices = tf.tile(tf.range(self.batch_size)[:, None, None, None], (1, H, W, B))
            _, batch_indices = tf.dynamic_partition(batch_indices, mask, 2)

            self.batch_indices[c] = batch_indices

            box_indices = tf.reshape(tf.tile(tf.range(H * W * B)[None, :], (self.batch_size, 1)), (-1, H, W, B))
            _, box_indices = tf.dynamic_partition(box_indices, mask, 2)

            self.box_indices[c] = box_indices

            if self.use_input_attention:
                raise Exception("NotImplemented")

                grid_coords = warper(boxes_on)
                grid_coords = tf.reshape(grid_coords, (self.batch_size, H, W, B,) + object_shape + (2,))
                input_glimpses = tf.contrib.resampler.resampler(self.inp, grid_coords)
                input_glimpses = tf.reshape(input_glimpses, (-1,) + object_shape + (image_depth,))
                object_decoder_in = [object_decoder_in, input_glimpses]

            object_decoder_output = od(object_decoder_in, object_shape + (image_depth,), self.is_training)

            if self.decoders_output_logits:
                object_decoder_output = tf.nn.sigmoid(
                    self.decoder_logit_scale * tf.clip_by_value(object_decoder_output, -10., 10.))

            self.object_decoder_output[c] = object_decoder_output

            # --- build predicted images by warping object_decoder_output and taking the max ---

            grid_coords = inverse_warper(boxes_on)
            object_decoder_transformed = tf.contrib.resampler.resampler(object_decoder_output, grid_coords)

            weighted_images = cls_on[..., None, None] * obj_on[..., None, None] * object_decoder_transformed

            per_class_images[c] = tf.segment_max(weighted_images, batch_indices)

        self.output = tf.stack([per_class_images[c] for c in range(self.C)], axis=1)
        self.output = tf.reduce_max(self.output, axis=1)

        original_batch_size = tf.shape(boxes)[0]
        new_batch_size = tf.shape(self.output)[0]

        correct_shape = tf.Assert(
            tf.equal(original_batch_size, new_batch_size),
            [original_batch_size, new_batch_size, batch_indices],
            name="batch_size_check")

        with tf.control_dependencies([correct_shape]):
            _output = tf.clip_by_value(self.output, 1e-6, 1-1e-6)
            self.output_logits = tf.log(_output / (1 - _output))

            self.network_outputs['correct_shape'] = correct_shape
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

        if self.object_decoders is None:
            self.object_decoders = [cfg.build_object_decoder(scope="object_decoder_{}".format(i)) for i in range(self.C)]

            if self.fixed_object_decoder:
                for od in self.object_decoders:
                    od.fix_variables()

        if self.dynamic_partition:
            self._build_program_interpreter_with_dynamic_partition()
        else:
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

        # --- kl ---
        cell_yx_kl = tf_normal_kl(
            self.cell_yx_dist['mean'], self.cell_yx_dist['std'],
            cfg.cell_yx_target_mean, cfg.cell_yx_target_std)
        recorded_tensors['cell_yx_kl'] = tf_mean_sum(cell_yx_kl)

        hw_kl = tf_normal_kl(
            self.hw_dist['mean'], self.hw_dist['std'],
            cfg.hw_target_mean, cfg.hw_target_std)
        recorded_tensors['hw_kl'] = tf_mean_sum(hw_kl)

        attr_kl = tf_normal_kl(
            self.attr_dist['mean'], self.attr_dist['std'],
            cfg.attr_target_mean, cfg.attr_target_std)
        recorded_tensors['attr_kl'] = tf_mean_sum(attr_kl)

        # --- entropy ---

        recorded_tensors['obj_entropy'] = tf_mean_sum(self.entropy['obj'])
        recorded_tensors['cls_entropy'] = tf_mean_sum(self.entropy['cls'])

        # --- recorded values ---

        recorded_tensors.update({
            name: tf.reduce_mean(getattr(self, 'build_' + name)(self.output_logits, self.inp))
            for name in ['xent_loss', 'squared_loss', '1norm_loss']
        })

        recorded_tensors['attr'] = tf.reduce_mean(self.attr_dist['mean'])
        recorded_tensors['attr_std'] = tf.reduce_mean(self.attr_dist['std'])

        recorded_tensors['cell_y'] = tf.reduce_mean(self.cell_y)
        recorded_tensors['cell_x'] = tf.reduce_mean(self.cell_x)

        recorded_tensors['cell_yx_std'] = tf.reduce_mean(self.cell_yx_dist['std'])

        recorded_tensors['h'] = tf.reduce_mean(self.h)
        recorded_tensors['w'] = tf.reduce_mean(self.w)

        recorded_tensors['hw_std'] = tf.reduce_mean(self.hw_dist['std'])

        recorded_tensors['obj_logits'] = tf.reduce_mean(self.obj_logits)
        recorded_tensors['cls_logits'] = tf.reduce_mean(self.cls_logits)

        recorded_tensors['obj'] = tf.reduce_mean(self.program_fields['obj'])
        recorded_tensors['cls_max'] = tf.reduce_mean(tf.reduce_max(self.program_fields['cls'], axis=-1))
        recorded_tensors['cls'] = tf.reduce_mean(self.program_fields['cls'])

        recorded_tensors['reconstruction_loss'] = mean_reconstruction_loss
        recorded_tensors['area_loss'] = mean_area_loss

        recorded_tensors['diff_loss'] = mean_reconstruction_loss

        if self.area_weight > 0.0:
            recorded_tensors['diff_loss'] += self.area_weight * mean_area_loss

        if self.obj_sparsity:
            recorded_tensors['obj_sparsity_loss'] = tf_mean_sum(self.program_fields['obj'])
            recorded_tensors['diff_loss'] += self.obj_sparsity * recorded_tensors['obj_sparsity_loss']

        if self.cls_sparsity:
            recorded_tensors['cls_sparsity_loss'] = tf_mean_sum(self.program_fields['cls'])
            recorded_tensors['diff_loss'] += self.cls_sparsity * recorded_tensors['_cls_sparsity_loss']

        if self.minimize_kl:
            recorded_tensors['diff_loss'] -= recorded_tensors['cls_entropy']
            recorded_tensors['diff_loss'] -= recorded_tensors['obj_entropy']

            recorded_tensors['diff_loss'] += recorded_tensors['cell_yx_kl']
            recorded_tensors['diff_loss'] += recorded_tensors['hw_kl']
            recorded_tensors['diff_loss'] += recorded_tensors['attr_kl']

        self.diff_loss = recorded_tensors['diff_loss']
        self.recorded_tensors = recorded_tensors

        # --- rl recorded values ---

        _rl_recorded_tensors = {}

        for name, _ in self.COST_funcs.items():
            _rl_recorded_tensors["COST_{}".format(name)] = tf.reduce_mean(self.COST_components[name])

        _rl_recorded_tensors["COST"] = tf.reduce_mean(self.COST)
        _rl_recorded_tensors["COST_obj"] = tf.reduce_mean(self.COST_obj)
        _rl_recorded_tensors["COST_cls"] = tf.reduce_mean(self.COST_cls)
        _rl_recorded_tensors["TOTAL_COST"] = (
            _rl_recorded_tensors["COST"] +
            _rl_recorded_tensors["COST_obj"] +
            _rl_recorded_tensors["COST_cls"]
        )

        _rl_recorded_tensors["obj_log_probs"] = tf.reduce_mean(self.log_probs['obj'])
        _rl_recorded_tensors["cls_log_probs"] = tf.reduce_mean(self.log_probs['cls'])

        if self.use_baseline:
            adv = self.COST - tf.reduce_mean(self.COST, axis=0, keep_dims=True)
            adv_obj = self.COST_obj - tf.reduce_mean(self.COST_obj, axis=0, keep_dims=True)
            adv_cls = self.COST_cls - tf.reduce_mean(self.COST_cls, axis=0, keep_dims=True)
        else:
            adv = self.COST
            adv_obj = self.COST_obj
            adv_cls = self.COST_cls

        self.rl_surrogate_loss_map = (
            (adv + adv_obj) * self.log_probs['obj'] +
            (adv + adv_cls) * self.log_probs['cls']
        )
        _rl_recorded_tensors['rl_loss'] = tf.reduce_mean(self.rl_surrogate_loss_map)

        _rl_recorded_tensors['surrogate_loss'] = _rl_recorded_tensors['rl_loss'] + mean_reconstruction_loss

        # if self.area_weight > 0.0:
        #     _rl_recorded_tensors['surrogate_loss'] += self.area_weight * mean_area_loss

        if self.minimize_kl:
            _rl_recorded_tensors['surrogate_loss'] -= recorded_tensors['cls_entropy']
            _rl_recorded_tensors['surrogate_loss'] -= recorded_tensors['obj_entropy']

            _rl_recorded_tensors['surrogate_loss'] += recorded_tensors['cell_yx_kl']
            _rl_recorded_tensors['surrogate_loss'] += recorded_tensors['hw_kl']
            _rl_recorded_tensors['surrogate_loss'] += recorded_tensors['attr_kl']

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
    def __call__(self, updater, N=16):
        rl_fetched = self._fetch(N, updater, True)

        self._plot_reconstruction(updater, rl_fetched, True)
        self._plot_patches(updater, rl_fetched, True)

        diff_fetched = self._fetch(N, updater, False)

        self._plot_reconstruction(updater, diff_fetched, False)
        self._plot_patches(updater, diff_fetched, False)

    def _fetch(self, N, updater, sampled):
        feed_dict = updater.make_feed_dict(N, 'val', True)
        images = feed_dict[updater.inp]
        feed_dict[updater.diff] = not sampled

        to_fetch = updater.program_fields.copy()
        to_fetch["output"] = updater.output
        to_fetch["object_decoder_output"] = updater.object_decoder_output

        if updater.dynamic_partition:
            to_fetch["batch_indices"] = updater.batch_indices
            to_fetch["box_indices"] = updater.box_indices

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        fetched.update(images=images)
        return fetched

    def _plot_reconstruction(self, updater, fetched, sampled):
        images = fetched['images']
        N = images.shape[0]

        output = fetched['output']

        _, image_height, image_width, _ = images.shape

        obj = fetched['obj'].reshape(N, -1)
        max_cls = np.argmax(fetched['cls'], axis=-1).reshape(N, -1)

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

        fig, axes = plt.subplots(2*sqrt_N, sqrt_N, figsize=(20, 20))
        axes = np.array(axes).reshape(2*sqrt_N, sqrt_N)
        for n, (pred, gt) in enumerate(zip(output, images)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            ax1 = axes[2*i, j]
            ax1.imshow(pred)
            ax1.set_title('reconstruction')

            ax2 = axes[2*i+1, j]
            ax2.imshow(gt)
            ax2.set_title('actual')

            for o, c, b in zip(obj[n], max_cls[n], box[n]):
                t, l, h, w = b

                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=3,
                    edgecolor=cfg.class_colours[int(c)], facecolor='none',
                    alpha=o)
                ax1.add_patch(rect)
                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=3,
                    edgecolor=cfg.class_colours[int(c)], facecolor='none',
                    alpha=o)
                ax2.add_patch(rect)

        fig.suptitle('Sampled={}. Stage={}. After {} experiences ({} updates, {} experiences per batch).'.format(
            sampled, updater.stage_idx, updater.n_experiences, updater.n_updates, cfg.batch_size))

        plot_name = ('sampled_' if sampled else '') + 'reconstruction.pdf'
        path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), plot_name)
        fig.savefig(path)

        plt.close(fig)

    def _plot_patches(self, updater, fetched, sampled):
        if updater.dynamic_partition:
            return self._plot_patches_dynamic(updater, fetched, sampled)

        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        object_decoder_output = fetched['object_decoder_output']

        H, W, C, B = [getattr(updater, a) for a in "H W C B".split()]

        N = fetched['images'].shape[0]

        obj = fetched['obj']
        cls = fetched['cls']
        activation = obj * cls

        for idx in range(N):
            fig, axes = plt.subplots(H * C, W * B, figsize=(20, 20))
            axes = np.array(axes).reshape(H * C, W * B)

            for i in range(H):
                for j in range(W):
                    for c in range(C):
                        for b in range(B):
                            ax = axes[i * C + c, j * B + b]

                            _cls = cls[idx, i, j, b, c]
                            _obj = obj[idx, i, j, b, 0]
                            _act = activation[idx, i, j, b, c]

                            ax.set_title("obj, cls, obj*cls = {}, {}, {}".format(_cls, _obj, _act))
                            ax.set_xlabel("(c, b) = ({}, {})".format(c, b))

                            if c == 0 and b == 0:
                                ax.set_ylabel("grid_cell: ({}, {})".format(i, j))

                            ax.imshow(object_decoder_output[c][idx, i, j, b])

            dir_name = ('sampled_' if sampled else '') + 'patches'
            path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), dir_name, '{}.pdf'.format(idx))
            fig.savefig(path)
            plt.close(fig)

    def _plot_patches_dynamic(self, updater, fetched, sampled):

        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        object_decoder_output = fetched['object_decoder_output']
        batch_indices = fetched['batch_indices']
        box_indices = fetched['box_indices']

        batch_size = fetched['images'].shape[0]
        H, W, C, B = [getattr(updater, a) for a in "H W C B".split()]

        obj = fetched['obj']
        cls = fetched['cls']
        activation = obj * cls

        for batch_idx in range(batch_size):
            fig, axes = plt.subplots(H * C, W * B, figsize=(20, 20))
            axes = np.array(axes).reshape(H * C, W * B)

            for c in range(C):
                mask = batch_indices[c] == batch_idx
                images = object_decoder_output[c][mask, ...]
                indices = list(box_indices[c][mask])
                raw_idx = 0

                for i in range(H):
                    for j in range(W):
                        for b in range(B):
                            ax = axes[i * C + c, j * B + b]

                            _cls = cls[batch_idx, i, j, b, c]
                            _obj = obj[batch_idx, i, j, b, 0]
                            _act = activation[batch_idx, i, j, b, c]

                            ax.set_title("obj, cls, obj*cls = {}, {}, {}".format(_obj, _cls, _act))
                            ax.set_xlabel("(c, b) = ({}, {})".format(c, b))

                            if c == 0 and b == 0:
                                ax.set_ylabel("grid_cell: ({}, {})".format(i, j))

                            if raw_idx in indices:
                                ax.imshow(images[indices.index(raw_idx)])

                            raw_idx += 1

            dir_name = ('sampled_' if sampled else '') + 'patches'
            path = updater.exp_dir.path_for(
                'plots',
                'stage{}'.format(updater.stage_idx),
                dir_name,
                '{}.pdf'.format(batch_idx))
            fig.savefig(path)
            plt.close(fig)


xkcd_colors = 'viridian,cerulean,vermillion,lavender,celadon,fuchsia,saffron,cinnamon,greyish,vivid blue'.split(',')


diff_mode = dict(rl_weight=0.0, diff_weight=1.0)
rl_mode = dict(rl_weight=1.0, diff_weight=0.0)
combined_mode = dict(rl_weight=1.0, diff_weight=1.0)


config = Config(
    log_name="yolo_rl",
    build_env=build_env,
    get_updater=get_updater,
    min_chars=1,
    max_chars=1,
    characters=[0, 1, 2],
    n_sub_image_examples=0,
    build_backbone=Backbone,
    build_next_step=NextStep,
    build_object_decoder=ObjectDecoder,
    xent_loss=True,
    sub_image_shape=(14, 14),

    render_hook=YoloRL_RenderHook(),
    render_step=5000,

    use_input_attention=False,
    decoders_output_logits=True,
    decoder_logit_scale=10.0,

    # model params
    image_shape=(28, 28),
    object_shape=(14, 14),
    anchor_boxes=[[28, 28]],
    pixels_per_cell=(28, 28),
    kernel_size=(1, 1),
    n_channels=128,
    D=3,
    C=1,
    A=100,

    pass_samples=True,  # Note that AIR basically uses pass_samples=True
    n_passthrough_features=100,
    pass_boxes_to_decoder=False,

    # display params
    class_colours=['xkcd:' + c for c in xkcd_colors],

    n_train=1e5,
    n_val=1e2,
    n_test=1e2,

    diff_weight=0.0,
    rl_weight=0.0,

    obj_sparsity=0.0,  # Within a single image, we want as few bounding boxes to be active as possible
    cls_sparsity=0.0,  # We want each of the class distributions to be as sparse as possible

    max_hw=1.0,  # Maximum for the bounding box multiplier.
    min_hw=0.0,  # Minimum for the bounding box multiplier.

    # VAE
    box_std=0.1,
    attr_std=0.0,
    minimize_kl=False,

    cell_yx_target_mean=0.5,
    cell_yx_target_std=1.0,
    hw_target_mean=0.5,
    hw_target_std=1.0,
    attr_target_mean=0.0,
    attr_target_std=1.0,

    obj_exploration=0.05,
    obj_default=0.5,
    cls_exploration=0.05,

    # Costs
    use_baseline=True,
    nonzero_weight=0.0,
    area_weight=0.0,
    use_specific_costs=False,
    use_specific_reconstruction=False,

    curriculum=[
        rl_mode,
    ],

    # training params
    beta=1.0,
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

    fixed_box=False,
    fixed_obj=False,
    fixed_cls=False,
    fixed_attr=False,

    fixed_object_decoder=False,

    fix_values=dict(),
    dynamic_partition=False,
    order="box obj cls attr",
)


nonzero_weight = 37.5


good_config = config.copy(
    image_shape=(40, 40),
    object_shape=(14, 14),
    anchor_boxes=[[14, 14]],
    pixels_per_cell=(12, 12),
    kernel_size=(3, 3),

    use_specific_costs=True,
    use_specific_reconstruction=False,
    max_hw=1.0,
    min_hw=0.5,
    max_chars=3,
    min_chars=1,
    characters=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

    dynamic_partition=True,
    fix_values=dict(),
    obj_exploration=0.0,
    nonzero_weight=0.0,
    lr_schedule=1e-4,

    curriculum=[
        dict(fix_values=dict(obj=1), lr_schedule=1e-4, dynamic_partition=False),
        dict(fix_values=dict(obj=1), lr_schedule=1e-5, dynamic_partition=False),
        dict(fix_values=dict(obj=1), lr_schedule=1e-6, dynamic_partition=False),
        dict(obj_exploration=0.2, nonzero_weight=nonzero_weight, lr_schedule=1e-6),
        dict(obj_exploration=0.1, nonzero_weight=nonzero_weight, lr_schedule=1e-6),
        dict(obj_exploration=0.05, nonzero_weight=nonzero_weight, lr_schedule=1e-6),
    ],

    box_std=-1.,
    attr_std=0.0,
    minimize_kl=True,

    cell_yx_target_mean=0.5,
    cell_yx_target_std=100.0,
    hw_target_mean=0.0,
    hw_target_std=1.0,
    attr_target_mean=0.0,
    attr_target_std=100.0,
    **rl_mode,
)


good_experimental_config = good_config.copy(
    lr_schedule=1e-4,
    curriculum=[
        dict(fix_values=dict(obj=1), dynamic_partition=False),
    ],
    hooks=[
        PolynomialScheduleHook(
            "nonzero_weight", "best_COST_reconstruction",
            base_configs=[
                dict(obj_exploration=0.2,),
                dict(obj_exploration=0.1,),
                dict(obj_exploration=0.05,),
            ],
            tolerance=0.1, scale=1., power=2., initial_value=1.0),
    ],
    colours="red",
    max_overlap=100,
    use_specific_reconstruction=True,
)

good_denser_bigger_config = good_experimental_config.copy(
    do_train=False,
    image_shape=(80, 80),
    min_chars=1,
    max_chars=11,
    n_train=100,
    hooks=[],
    curriculum=[dict()],
    load_path="/data/dps_data/logs/yolo_rl/exp_yolo_rl_seed=347405995_2018_03_16_09_48_57/weights/best_of_stage_20"
)

static_decoder_config = good_experimental_config.copy(
    build_object_decoder=StaticObjectDecoder,
    C=3,
    A=2,
    characters=[0, 1],
    decoder_logit_scale=100.,
    patience=20000,
    curriculum=[
        dict(fix_values=dict(obj=1), dynamic_partition=False, lr_schedule=1e-4),
        dict(fix_values=dict(obj=1), dynamic_partition=False, lr_schedule=1e-5),
        dict(fix_values=dict(obj=1), dynamic_partition=False, lr_schedule=1e-6),
    ],
)

# classification_config = config.copy(
#     log_name="yolo_rl_classify",
#     C=2,
#     image_shape=(16, 16),
#     pixels_per_cell=(16, 16),
#     sub_image_shape=(14, 14),
#     object_shape=(14, 14),
#     anchor_boxes=[[14, 14]],
#     kernel_size=(1, 1),
#     colours="red",
# 
#     use_specific_costs=True,
#     max_hw=1.0,
#     min_hw=0.5,
#     max_chars=1,
#     min_chars=1,
#     characters=list(range(10)),
# 
#     dynamic_partition=False,
#     fix_values=dict(obj=1, cell_x=0.5, cell_y=0.5, h=1.0, w=1.0),
#     obj_exploration=0.0,
#     nonzero_weight=0.0,
#     lr_schedule=1e-4,
# 
#     curriculum=[
#         dict(cls_exploration=0.5),
#         dict(cls_exploration=0.4),
#         dict(cls_exploration=0.3),
#         dict(cls_exploration=0.2),
#         dict(cls_exploration=0.1),
#     ],
#     decoder_logit_scale=10.0,
# 
#     box_std=-1.,
#     attr_std=0.0,
#     minimize_kl=True,
# 
#     cell_yx_target_mean=0.5,
#     cell_yx_target_std=100.0,
#     hw_target_mean=0.0,
#     hw_target_std=1.0,
#     attr_target_mean=0.0,
#     attr_target_std=100.0,
#     **rl_mode,
# )
# 
# 
# test_stage_hooks = good_config.copy(
#     curriculum=[
#         dict(fix_values=dict(obj=1), lr_schedule=1e-4, dynamic_partition=False),
#         dict(fix_values=dict(obj=1), lr_schedule=1e-5, dynamic_partition=False),
#         dict(fix_values=dict(obj=1), lr_schedule=1e-6, dynamic_partition=False),
#     ],
#     hooks=[
#         PolynomialScheduleHook(
#             "nonzero_weight", "best_COST_reconstruction",
#             base_configs=[
#                 dict(obj_exploration=0.2, lr_schedule=1e-6),
#                 dict(obj_exploration=0.1, lr_schedule=1e-6),
#             ],
#             tolerance=0.1, scale=5.),
#     ],
#     max_steps=11,
#     n_train=100,
#     render_step=0,
# )
# 
# 
# test_dynamic_partition_config = good_config.copy(
#     dynamic_partition=True,
#     curriculum=[
#         dict(obj_exploration=1.0, obj_default=0.5, dynamic_partition=False),
#         dict(obj_exploration=1.0, obj_default=0.9, dynamic_partition=False),
#         dict(obj_exploration=1.0, obj_default=0.1, dynamic_partition=False),
#         dict(obj_exploration=1.0, obj_default=0.5),
#         dict(obj_exploration=1.0, obj_default=0.9),
#         dict(obj_exploration=1.0, obj_default=0.1),
#     ],
#     max_steps=201,
#     display_step=10,
# )
# 
# test_larger_config = good_config.copy(
#     curriculum=[
#         dict(load_path="/data/dps_data/logs/yolo_rl/exp_yolo_rl_seed=347405995_2018_03_12_21_50_31/weights/best_of_stage_0", do_train=False),
#     ],
#     image_shape=(80, 80),
#     n_train=100,
# )
# 
# 
# passthrough_config = config.copy(
#     log_name="yolo_passthrough",
#     build_object_decoder=PassthroughDecoder,
#     C=1,
#     A=2,
#     characters=[0],
#     # characters=[0, 1, 2],
#     object_shape=(28, 28),
#     anchor_boxes=[[28, 28]],
#     sub_image_shape=(28, 28),
# 
#     use_input_attention=True,
#     decoders_output_logits=False,
# 
#     fix_values=dict(cls=1),
# 
#     lr_schedule=1e-6,
#     pixels_per_cell=(10, 10),
#     nonzero_weight=40.0,
#     obj_exploration=0.30,
#     cls_exploration=0.30,
# 
#     box_std=-1.,
#     attr_std=0.0,
#     minimize_kl=True,
# 
#     cell_yx_target_mean=0.5,
#     cell_yx_target_std=1.0,
#     hw_target_mean=0.0,
#     hw_target_std=1.0,
#     attr_target_mean=0.0,
#     attr_target_std=1.0,
# )
