import tensorflow as tf
import numpy as np
import sonnet as snt

from dps import cfg
from dps.datasets import AutoencodeDataset, EMNIST_ObjectDetection
from dps.updater import Updater
from dps.utils import Config, Param
from dps.utils.tf import (
    FullyConvolutional, build_gradient_train_op,
    trainable_variables, build_scheduled_value,
    tf_normal_kl, tf_mean_sum
)

tf_flatten = tf.layers.flatten


class Env(object):
    pass


def build_env():
    return Env()


def get_updater(env):
    return YoloRL_Updater()


class Backbone(FullyConvolutional):
    H = Param()
    W = Param()

    def __init__(self, **kwargs):
        H, W = self.H, self.W
        assert H == W
        assert H in (1, 2, 3, 7)
        if H == 1:
            ksize = 4
        elif H == 2:
            ksize = 3
        else:
            ksize = 2

        layout = [
            dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=ksize, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=(H, W), strides=1, padding="SAME"),
            dict(filters=256, kernel_size=(H, W), strides=1, padding="SAME"),
        ]
        if H == 7:
            layout = layout[:2] + layout[4:] + layout[-1:]
        super(Backbone, self).__init__(layout, check_output_shape=True, **kwargs)


class NextStep(FullyConvolutional):
    H = Param()
    W = Param()

    def __init__(self, **kwargs):
        kernel_size = (self.H, self.W)
        layout = [
            dict(filters=128, kernel_size=kernel_size, strides=1, padding="SAME"),
            dict(filters=256, kernel_size=kernel_size, strides=1, padding="SAME"),
            dict(filters=128, kernel_size=kernel_size, strides=1, padding="SAME"),
            dict(filters=256, kernel_size=kernel_size, strides=1, padding="SAME"),
        ]
        super(NextStep, self).__init__(layout, check_output_shape=True, **kwargs)


class ObjectDecoder(FullyConvolutional):
    def __init__(self, **kwargs):
        layout = [
            dict(filters=128, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=256, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=3, kernel_size=4, strides=1, padding="SAME", transpose=True),  # For 14 x 14 output
        ]
        super(ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)


def reconstruction_cost(cost_input):
    return cost_input['reconstruction_loss'][..., None, None, None]


def prediction_map_cost(cost_input):
    return cost_input['prediction_map_loss'][..., None, None, None]


def obj_nonzero_cost(cost_input):
    obj_samples = cost_input['obj']
    obj_samples = obj_samples.reshape(obj_samples.shape[0], -1)
    return np.count_nonzero(obj_samples, axis=1)[..., None, None, None, None].astype('f')


class YoloRL_Updater(Updater):
    H = Param()
    W = Param()
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

    box_std = Param()
    attr_std = Param()
    minimize_kl = Param()

    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    pass_samples=Param(),
    n_passthrough_features = Param()

    xent_loss = Param()

    use_baseline = Param()
    obj_nonzero_weight = Param()
    prediction_map_weight = Param()

    obj_exploration = Param()
    cls_exploration = Param()

    fixed_backbone = Param()
    fixed_next_step = Param()
    fixed_object_decoder = Param()

    fix_values = Param()

    stopping_criteria = "loss,min"
    eval_modes = "rl_val diff_val".split()

    def __init__(self, scope=None, **kwargs):
        self.anchor_boxes = np.array(self.anchor_boxes)
        self.B = len(self.anchor_boxes)

        self._make_datasets()

        self.obs_shape = self.datasets['train'].x.shape[1:]
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        assert self.diff_weight > 0 or self.rl_weight > 0

        self.COST_funcs = {}

        if self.obj_nonzero_weight is not None:
            self.COST_funcs['obj_nonzero'] = (self.obj_nonzero_weight, obj_nonzero_cost, "obj")

        if self.prediction_map_weight > 0.0:
            self.COST_funcs['prediction_map'] = (self.prediction_map_weight, prediction_map_cost, "obj")

        self.COST_funcs['reconstruction'] = (1, reconstruction_cost, "both")

        self.scope = scope
        self._n_experiences = 0
        self._n_updates = 0

        self.backbone = None
        self.obj_network = None
        self.cls_network = None
        self.attr_network = None
        self.object_decoders = None

        self.predict_box_std = False
        try:
            box_std = float(self.box_std)
            self.predict_box_std = box_std < 0
        except (TypeError, ValueError):
            pass
        self.n_box_params = 8 if self.predict_box_std else 4

        self.predict_attr_std = False
        try:
            attr_std = float(self.attr_std)
            self.predict_attr_std = attr_std < 0
        except (TypeError, ValueError):
            pass
        self.n_attr_params = 2 if self.predict_attr_std else 1

    def _make_datasets(self):
        _train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train)).x
        train = AutoencodeDataset(_train, image=True, shuffle=True)

        _val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val)).x
        val = AutoencodeDataset(_val, image=True, shuffle=False)

        self.datasets = dict(train=train, val=val)

    @property
    def completion(self):
        return self.datasets['train'].completion

    def trainable_variables(self, for_opt, rl_only=False):
        scoped_functions = (
            self.object_decoders +
            [self.backbone, self.obj_network, self.cls_network, self.attr_network]
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
        cost_input = sess.run(self.cost_input, feed_dict=feed_dict)
        cost_input['shape'] = self.H, self.W, self.B

        sample_feed_dict = {self.samples[k]: s for k, s in cost_input.items() if k in self.samples}

        for name, (_, f, _) in self.COST_funcs.items():
            sample_feed_dict[self.COST_components[name]] = self._process_cost(f(cost_input))

        # sample_feed_dict contains entries for only self.samples and costs
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
        inp = self.datasets[mode].next_batch(batch_size=batch_size, advance=not evaluate)
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

    def _build_program_generator(self):
        inp, is_training = self.inp, self.is_training
        H, W, B, C, A = self.H, self.W, self.B, self.C, self.A
        image_height, image_width = self.image_height, self.image_width

        # --- bounding boxes ---

        n_box_params = self.n_box_params

        box_output = self.backbone(inp, (H, W, B * n_box_params + self.n_passthrough_features), is_training)

        features = box_output[..., B * n_box_params:]

        box_params = box_output[..., :B * n_box_params]
        box_params = tf.reshape(box_params, (-1, H, W, B, n_box_params))

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

        hw = float(self.max_hw) * tf.nn.sigmoid(tf.clip_by_value(hw_logits, -10., 10.))
        # hw = 2 * tf.nn.sigmoid(tf.clip_by_value(hw_logits, -10., 10.))

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

        noisy_y = (noisy_cell_y + tf.range(H, dtype=tf.float32)[None, :, None, None, None]) / H
        noisy_x = (noisy_cell_x + tf.range(W, dtype=tf.float32)[None, None, :, None, None]) / W

        yt = 2 * noisy_y - 1
        xt = 2 * noisy_x - 1

        box_representation = tf.concat([xs, xt, ys, yt], axis=-1)

        program = box_representation

        # --- objectness ---

        obj_input = features
        if self.pass_samples:
            _program = tf.reshape(program, (-1, H, W, B*4))
            obj_input = tf.concat([features, _program], axis=-1)

        obj_param_depth = B * 1

        obj_output = self.obj_network(obj_input, (H, W, obj_param_depth + self.n_passthrough_features), is_training)

        features = obj_output[..., obj_param_depth:]

        obj_logits = obj_output[..., :obj_param_depth]
        obj_logits = tf.reshape(obj_logits, (-1, H, W, B, 1))
        obj_logits = tf.clip_by_value(obj_logits, -10., 10.)
        self.obj_logits = obj_logits

        obj_params = tf.nn.sigmoid(obj_logits)
        obj_exploration = build_scheduled_value(self.obj_exploration, "obj_exploration") * self.float_is_training
        obj_params = (1 - obj_exploration) * obj_params + 0.5 * obj_exploration

        obj_dist = tf.distributions.Bernoulli(probs=obj_params)

        obj_samples = tf.stop_gradient(obj_dist.sample())
        obj_samples = tf.to_float(obj_samples)

        obj_log_probs = obj_dist.log_prob(obj_samples)
        obj_log_probs = tf.where(tf.is_nan(obj_log_probs), -100.0 * tf.ones_like(obj_log_probs), obj_log_probs)

        obj_entropy = obj_dist.entropy()

        obj_representation = self.float_diff * obj_params + (1 - self.float_diff) * obj_samples

        if "obj" in self.fix_values:
            obj_representation = float(self.fix_values["obj"]) * tf.ones_like(obj_representation, dtype=tf.float32)

        program = tf.concat([program, obj_representation], axis=-1)

        # --- classes ---

        cls_input = features
        if self.pass_samples:
            _program = tf.reshape(program, (-1, H, W, B*(4 + 1)))
            cls_input = tf.concat([features, _program], axis=-1)

        cls_param_depth = B * C
        cls_output = self.cls_network(cls_input, (H, W, cls_param_depth + self.n_passthrough_features), is_training)

        features = cls_output[..., cls_param_depth:]

        cls_logits = cls_output[..., :cls_param_depth]
        cls_logits = tf.reshape(cls_logits, (-1, H, W, B, C))
        cls_logits = tf.minimum(10.0, cls_logits)

        self.cls_logits = cls_logits

        cls_params = tf.nn.softmax(cls_logits)

        cls_exploration = build_scheduled_value(self.cls_exploration, "cls_exploration") * self.float_is_training
        cls_params = (1 - cls_exploration) * cls_params + cls_exploration / self.C

        cls_dist = tf.distributions.Categorical(probs=cls_params)

        cls_samples = tf.stop_gradient(cls_dist.sample())
        cls_samples = tf.one_hot(cls_samples, C)
        cls_samples = tf.to_float(cls_samples)

        cls_log_probs = cls_dist.log_prob(tf.argmax(cls_samples, axis=-1))[..., None]
        cls_log_probs = tf.where(tf.is_nan(cls_log_probs), -100.0 * tf.ones_like(cls_log_probs), cls_log_probs)

        cls_entropy = cls_dist.entropy()

        cls_representation = self.float_diff * cls_params + (1 - self.float_diff) * cls_samples

        if "cls" in self.fix_values:
            cls_representation = float(self.fix_values["cls"]) * tf.ones_like(cls_representation, dtype=tf.float32)

        program = tf.concat([program, cls_representation], axis=-1)

        # --- attributes ---

        attr_input = features
        if self.pass_samples:
            _program = tf.reshape(program, (-1, H, W, B*(4 + 1 + C)))
            attr_input = tf.concat([features, _program], axis=-1)

        n_attr_params = self.n_attr_params
        attr_output = self.attr_network(attr_input, (H, W, B * n_attr_params * A), is_training)

        attr_params = tf.reshape(attr_output, (-1, H, W, B, n_attr_params * A))

        if self.predict_attr_std:
            attr, attr_std_logits = tf.split(attr_params, 2, axis=-1)
            attr_std = tf.exp(attr_std_logits)
        else:
            attr = attr_params
            _attr_std = build_scheduled_value(self.attr_std)
            attr_std = _attr_std * tf.ones_like(attr)

        attr_noise = tf.random_normal(tf.shape(attr_std), name="attr_noise")
        noisy_attr = attr + attr_noise * attr_std * self.float_is_training

        program = tf.concat([program, noisy_attr], axis=-1)

        # --- finalize ---

        self.cell_y, self.cell_x = tf.split(cell_yx, 2, axis=-1)
        self.h, self.w = tf.split(hw, 2, axis=-1)

        self.program_fields = dict(
            box=box_representation, obj=obj_representation,
            cls=cls_representation, attr=noisy_attr)

        self.samples = dict(cell_yx=cell_yx_noise, hw=hw_noise, obj=obj_samples, cls=cls_samples, attr=attr_noise)
        self.log_probs = dict(obj=obj_log_probs, cls=cls_log_probs)
        self.entropy = dict(obj=obj_entropy, cls=cls_entropy)
        self.program = program

        self.cell_yx_dist = dict(mean=cell_yx, std=cell_yx_std)
        self.hw_dist = dict(mean=hw, std=hw_std)
        self.attr_dist = dict(mean=attr, std=attr_std)

    def _build_program_interpreter(self):
        H, W, B, A = self.H, self.W, self.B, self.A
        object_shape, image_height, image_width, image_depth = (
            self.object_shape, self.image_height, self.image_width, self.image_depth)

        boxes = self.program_fields['box']
        _boxes = tf.reshape(boxes, (-1, 4))

        obj = self.program_fields['obj']
        cls = self.program_fields['cls']
        attr = self.program_fields['attr']

        object_decoder_in = tf.concat([boxes, attr], axis=-1)
        object_decoder_in = tf.reshape(object_decoder_in, (-1, 1, 1, 4 + A))

        self.object_decoder_output = {}
        per_class_images = {}
        per_class_map = {}  # Map of where predictions are being made.

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

            # --- build prediction map ---

            _map = tf.contrib.resampler.resampler(tf.ones_like(object_decoder_output[..., 0:1]), grid_coords)
            _map = tf.reshape(
                _map, [-1, H, W, B, image_height, image_width, 1]
            )

            weighted_map = (
                cls[..., c:c+1, None, None] *
                obj[..., None, None] *
                _map
            )
            per_class_map[c] = tf.reshape(weighted_map, [-1, H*W*B, image_height, image_width, 1])

        self.output = tf.concat([per_class_images[c] for c in range(self.C)], axis=1)
        self.output = tf.reduce_max(self.output, axis=1)

        _output = tf.clip_by_value(self.output, 1e-6, 1-1e-6)
        self.output_logits = tf.log(_output / (1 - _output))

        self.prediction_map = tf.concat([per_class_map[c] for c in range(self.C)], axis=1)
        self.prediction_map = tf.reduce_sum(self.prediction_map, axis=1)

        # Samples contains all values upon which costs may be based
        self.cost_input = self.samples.copy()
        self.cost_input['inp'] = self.inp
        self.cost_input['output'] = self.output
        self.cost_input['output_logits'] = self.output_logits
        self.cost_input['object_decoder_output'] = self.object_decoder_output

        self.cost_input.update(self.program_fields)

    def build_prediction_map_loss(self):
        _map = tf_flatten(self.prediction_map)
        return tf.reduce_sum(_map, axis=1, keep_dims=True)

    def build_xent_loss(self, logits, targets):
        targets = tf_flatten(targets)
        logits = tf_flatten(logits)

        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits),
            keep_dims=True, axis=1
        )

    def build_2norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)

        targets = tf_flatten(targets)
        actions = tf_flatten(actions)

        return tf.sqrt(tf.reduce_sum((actions - targets)**2, keep_dims=True, axis=1))

    def build_1norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)

        targets = tf_flatten(targets)
        actions = tf_flatten(actions)

        return tf.reduce_sum(tf.abs(actions - targets), keep_dims=True, axis=1)

    def _build_graph(self):
        self._build_placeholders()

        if self.backbone is None:
            self.backbone = cfg.build_backbone(scope="backbone")
            self.backbone.layout[-1]['filters'] = self.B * self.n_box_params + self.n_passthrough_features

            if self.fixed_backbone:
                self.backbone.fix_variables()

        if self.obj_network is None:
            self.obj_network = cfg.build_obj_network(scope="obj_network")
            self.obj_network.layout[-1]['filters'] = self.B + self.n_passthrough_features

            if self.fixed_next_step:
                self.obj_network.fix_variables()

        if self.cls_network is None:
            self.cls_network = cfg.build_cls_network(scope="cls_network")
            self.cls_network.layout[-1]['filters'] = self.B * self.C + self.n_passthrough_features

            if self.fixed_next_step:
                self.cls_network.fix_variables()

        if self.attr_network is None:
            self.attr_network = cfg.build_attr_network(scope="attr_network")
            self.attr_network.layout[-1]['filters'] = self.B * self.n_attr_params * self.A

            if self.fixed_next_step:
                self.attr_network.fix_variables()

        self._build_program_generator()

        if self.object_decoders is None:
            self.object_decoders = [cfg.build_object_decoder(scope="object_decoder_{}".format(i)) for i in range(self.C)]

            if self.fixed_object_decoder:
                for od in self.object_decoders:
                    od.fix_variables()

        self._build_program_interpreter()

        loss_key = 'xent_loss' if self.xent_loss else '2norm_loss'

        recorded_tensors = {}

        reconstruction_loss = getattr(self, 'build_' + loss_key)(self.output_logits, self.inp)
        mean_reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        prediction_map_loss = self.build_prediction_map_loss()
        mean_prediction_map_loss = tf.reduce_mean(prediction_map_loss)

        # --- cost input ---

        self.cost_input.update(
            reconstruction_loss=reconstruction_loss,
            prediction_map_loss=prediction_map_loss
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
            for name in ['xent_loss', '2norm_loss', '1norm_loss']
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
        recorded_tensors['prediction_map_loss'] = mean_prediction_map_loss

        recorded_tensors['diff_loss'] = mean_reconstruction_loss

        if self.prediction_map_weight > 0.0:
            recorded_tensors['diff_loss'] += self.prediction_map_weight * mean_prediction_map_loss

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

        if self.prediction_map_weight > 0.0:
            _rl_recorded_tensors['surrogate_loss'] += self.prediction_map_weight * mean_prediction_map_loss

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

            for o, c, b in zip(obj[n], max_cls[n], box[n]):
                t, l, h, w = b

                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=3,
                    edgecolor=cfg.class_colours[int(c)], facecolor='none',
                    alpha=o)
                ax1.add_patch(rect)

            ax2 = axes[2*i+1, j]
            ax2.imshow(gt)
            ax2.set_title('actual')

        fig.suptitle('Sampled={}. Stage={}. After {} experiences ({} updates, {} experiences per batch).'.format(
            sampled, updater.stage_idx, updater.n_experiences, updater.n_updates, cfg.batch_size))

        plot_name = ('sampled_' if sampled else '') + 'reconstruction.pdf'
        path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), plot_name)
        fig.savefig(path)

        plt.close(fig)

    def _plot_patches(self, updater, fetched, sampled):
        # Create a plot showing what each object is generating (for the 1st image only)
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
    build_obj_network=NextStep,
    build_cls_network=NextStep,
    build_attr_network=NextStep,
    build_object_decoder=ObjectDecoder,
    xent_loss=True,
    image_shape=(28, 28),
    sub_image_shape=(14, 14),

    render_hook=YoloRL_RenderHook(),
    render_step=500,

    use_input_attention=False,
    decoders_output_logits=True,
    decoder_logit_scale=10.0,

    # model params
    object_shape=(14, 14),
    anchor_boxes=[[28, 28]],
    H=1,
    W=1,
    D=3,
    C=1,
    A=100,

    pass_samples=True,  # Note that AIR basically uses pass_samples=True
    n_passthrough_features=100,

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
    cls_exploration=0.05,

    # Costs
    use_baseline=True,
    obj_nonzero_weight=0.0,
    prediction_map_weight=0.0,

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
    patience=0,
    optimizer_spec="adam",
    use_gpu=True,
    gpu_allow_growth=True,
    seed=347405995,
    stopping_criteria="reconstruction_loss,min",
    eval_mode="rl_val",
    threshold=-np.inf,
    max_grad_norm=1.0,

    max_experiments=None,

    fixed_backbone=False,
    fixed_next_step=False,
    fixed_object_decoder=False,

    fix_values=dict(),
)

experimental_config = config.copy(
    obj_exploration=1.0,
)

multi_config = config.copy(
    H=2,
    W=2,
    anchor_boxes=[[14, 14]],
    obj_nonzero_weight=20.0,
    obj_exploration=0.10,
    cls_exploration=0.10,
)


multi_small_config = multi_config.copy(
    box_std=0.1,
    attr_std=0.01,
    minimize_kl=True,
    cell_yx_target_mean=0.5,
    cell_yx_target_std=100.0,
    hw_target_mean=0.1,
    hw_target_std=1.0,
    attr_target_mean=0.0,
    attr_target_std=100.0,
)
