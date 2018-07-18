import numpy as np
import tensorflow as tf
import os

from dps import cfg
from dps.updater import DifferentiableUpdater
from dps.env.supervised import SupervisedEnv
from dps.datasets import EmnistObjectDetectionDataset
from dps.utils import Config, Param
from dps.utils.tf import build_gradient_train_op, ConvNet


class MODE(object):
    """ Namespace for modes that control the channel dimensionality of the output. """
    DEFAULT = "DEFAULT"
    BBOX_PER_CLASS = "BBOX_PER_CLASS"
    ATTR_PER_CLASS = "ATTR_PER_CLASS"
    BBOX_ATTR_PER_CLASS = "BBOX_ATTR_PER_CLASS"

    modes = [DEFAULT, BBOX_PER_CLASS, ATTR_PER_CLASS, BBOX_ATTR_PER_CLASS]

    @classmethod
    def validate_mode(cls, mode):
        assert mode in MODE.modes, "Unknown mode: {}. Valid modes are: {}".format(mode, cls.modes)


class YOLOv2_SupervisedEnv(SupervisedEnv):
    """ You Only Look Once, version 2 (aka YOLO9000).

    Assumed image format: matrix-style.
        y comes before x, y index increases moving downward, x index increases moving rightward.

    """
    H = Param()
    W = Param()
    C = Param()
    anchor_boxes = Param(help="List of (h, w) pairs.")
    scale_class = Param()
    scale_obj = Param()
    scale_no_obj = Param()
    scale_coord = Param()
    use_squared_loss = Param()
    conf_target = Param()

    def __init__(self, train, val, test, **kwargs):
        _anchor_boxes = np.array(self.anchor_boxes)
        _anchor_boxes_area = _anchor_boxes[:, :1] * _anchor_boxes[:, 1:]
        self.anchor_boxes = np.concatenate([_anchor_boxes, _anchor_boxes_area], axis=1)

        self.B = len(self.anchor_boxes)
        self.D = 4 + 1 + self.C

        self.action_shape = (self.H, self.W, self.channel_dim)

        super(YOLOv2_SupervisedEnv, self).__init__(train, val, test)

        self.image_height, self.image_width, self.image_d = self.obs_shape
        self.cell_height = self.image_height / self.H
        self.cell_width = self.image_width / self.W

        self._normalized_anchor_boxes = self.anchor_boxes[:, :2] / [self.image_height, self.image_width]
        self._normalized_anchor_boxes = self._normalized_anchor_boxes.reshape(1, 1, self.B, 2)

        assert self.conf_target in "conf conf_times_iou iou".split()

    @property
    def channel_dim(self):
        return self.B*self.D

    def __str__(self):
        s = "YOLO_SupervisedLoss:\n"
        s += '\tH       = {}\n'.format(self.H)
        s += '\tW       = {}\n'.format(self.W)
        s += '\tn_anchor_boxes (B) = {}\n'.format(self.B)
        s += '\tanchor_boxes = {}\n'.format(self.anchor_boxes)
        s += '\tn_classes (C) = {}\n'.format(self.C)
        s += '\tscales  = {}\n'.format([self.scale_class, self.scale_obj, self.scale_no_obj, self.scale_coord])

    def next_batch(self, batch_size, mode, process=True, evaluate=True):
        images, annotations = self.datasets[mode].next_batch(batch_size=batch_size, advance=not evaluate)
        if process:
            annotations = self.process_annotations(annotations)
        return images, annotations

    def make_feed_dict(self, batch_size, mode, evaluate):
        x, target = self.next_batch(batch_size, mode, process=True, evaluate=evaluate)
        return {self.x: x, self.target: target, self.is_training: not evaluate}

    def process_annotations(self, annotations):
        """ Turn a batch of bounding box annotations into a target volume.

        Parameters
        ----------
        annotations:
            List of annotations to process. Each element is
            a list of bounding boxes for a single image (the image itself
            is not required at this point).

            Each annotation has the form:  cls (int), y_min, y_max, x_min, x_max (in pixels)
            The last 4 entries in the annotation have units of pixels.

        Returns
        -------
        target_volume

        """
        H, W, B, C = self.H, self.W, self.B, self.C
        D = 4 + 1 + C

        # All images must have same size
        image_height, image_width = self.image_height, self.image_width

        cell_height = 1. * image_height / H
        cell_width = 1. * image_width / W

        target_volume = np.zeros((len(annotations), H, W, B * (4 + 1 + C)), dtype=np.float32)

        for batch_idx, objects in enumerate(annotations):
            # Populate these.
            coords = np.zeros([H, W, B, 4])
            confs = np.zeros([H, W, B, 1])
            probs = np.zeros([H, W, B, C])

            # assume anchor_boxes has shape (n_anchor_boxes, 3), columns are (height, width, area), all in pixels.
            for obj in objects:
                # in image coordinates
                cls, y_min, y_max, x_min, x_max = obj

                assert y_max > y_min
                assert x_max > x_min

                # bbox center in image coordinates
                center_y = .5 * (y_min + y_max)
                center_x = .5 * (x_min + x_max)

                # bbox center in grid cell coordinates
                cy = center_y / cell_height
                cx = center_x / cell_width

                if cy >= H or cx >= W:
                    raise Exception()

                bbox_height = float(y_max - y_min) / image_height
                bbox_width = float(x_max - x_min) / image_width

                # index of cell that the object is assigned to
                cell_idx_y = int(np.floor(cy))
                cell_idx_x = int(np.floor(cx))

                # bbox center in cell coordinates
                cell_center_y = cy - cell_idx_y
                cell_center_x = cx - cell_idx_x

                # find anchor box whose shape is most like ground-truth shape
                pixel_height = y_max - y_min
                pixel_width = x_max - x_min
                pixel_area = 1. * pixel_height * pixel_width

                overlap_height = np.minimum(pixel_height, self.anchor_boxes[:, 0])
                overlap_width = np.minimum(pixel_width, self.anchor_boxes[:, 1])
                overlap_area = overlap_height * overlap_width
                IOU = overlap_area / (self.anchor_boxes[:, 2] + pixel_area - overlap_area)
                anchor_box_idx = np.argmax(IOU)

                _coords = cell_center_y, cell_center_x, bbox_height, bbox_width

                coords[cell_idx_y, cell_idx_x, anchor_box_idx, :] = _coords
                confs[cell_idx_y, cell_idx_x, anchor_box_idx, :] = 1
                probs[cell_idx_y, cell_idx_x, anchor_box_idx, cls] = 1

            target = np.concatenate([coords, confs, probs], axis=-1)
            target_volume[batch_idx, ...] = target.reshape(H, W, B*D)

        return target_volume

    def _build(self):
        """ self.action and self.target are both tensors with shape (batch_dim, H, W, B * (4 + 1 + C)) """
        H, W, B, C = self.H, self.W, self.B, self.C
        D = 4 + 1 + C

        self.predictions = {}

        target = tf.reshape(self.target, [-1, H, W, B, D])
        prediction = tf.reshape(self.prediction, [-1, H, W, B, D])

        _yx, _hw, _confs, _probs = tf.split(target, [2, 2, 1, C], -1)
        yx_logits, hw_logits, conf_logits, class_logits = tf.split(prediction, [2, 2, 1, C], -1)

        # predict bbox center in cell coordinates
        yx = tf.nn.sigmoid(yx_logits)

        # use anchor boxes to predict box height and width (normalized to image size)
        hw = tf.exp(hw_logits) * self._normalized_anchor_boxes

        yx_loss = self.scale_coord * _confs * (yx - _yx)**2
        hw_loss = self.scale_coord * _confs * (tf.sqrt(hw) - tf.sqrt(_hw))**2

        # compute iou for conf regression target
        y, x = tf.split(yx, 2, axis=-1)
        h, w = tf.split(hw, 2, axis=-1)

        self.predictions.update(cell_y=y, cell_x=x, normalized_h=h, normalized_w=w)

        if self.conf_target in ["conf_times_iou", "iou"]:
            y_min, y_max = y - 0.5 * h, y + 0.5 * h
            x_min, x_max = x - 0.5 * w, x + 0.5 * w
            area = h * w

            _y, _x = tf.split(_yx, 2, axis=-1)
            _h, _w = tf.split(_hw, 2, axis=-1)

            _y_min, _y_max = _y - 0.5 * _h, _y + 0.5 * _h
            _x_min, _x_max = _x - 0.5 * _w, _x + 0.5 * _w
            _area = _h * _w

            top = tf.maximum(y_min, _y_min)
            bottom = tf.minimum(y_max, _y_max)
            left = tf.maximum(x_min, _x_min)
            right = tf.minimum(x_max, _x_max)

            overlap_height = tf.maximum(0., bottom - top)
            overlap_width = tf.maximum(0., right - left)
            overlap_area = overlap_height * overlap_width

            iou = overlap_area / (area + _area - overlap_area)

            conf_target = iou
            if self.conf_target == "conf_times_iou":
                conf_target *= _confs

            self.predictions.update(iou=iou)
        else:
            conf_target = _confs

        if self.use_squared_loss:
            sigmoid = tf.nn.sigmoid(conf_logits)
            conf_loss = self.scale_obj * _confs * (sigmoid - conf_target)**2
            conf_loss += self.scale_no_obj * (1 - _confs) * (sigmoid - conf_target)**2

            softmax = tf.nn.softmax(class_logits)
            prob_loss = self.scale_class * _confs * (softmax - _probs)**2

            self.predictions.update(confs=sigmoid, probs=softmax)
        else:
            conf_loss = self.scale_obj * _confs * (
                tf.nn.sigmoid_cross_entropy_with_logits(labels=conf_target, logits=conf_logits))
            conf_loss += self.scale_no_obj * (1 - _confs) * (
                tf.nn.sigmoid_cross_entropy_with_logits(labels=conf_target, logits=conf_logits))

            _prob_loss = tf.nn.softmax_cross_entropy_with_logits(labels=_probs, logits=class_logits)
            prob_loss = self.scale_class * _confs * _prob_loss[..., None]

            self.predictions.update(confs=tf.nn.sigmoid(conf_logits), probs=tf.nn.softmax(class_logits))

        loss_volume = tf.concat([yx_loss, hw_loss, conf_loss, prob_loss], axis=-1)  # (batch_dim, H, W, B, D)
        batch_size = tf.shape(target)[0]
        per_examples_loss = tf.reduce_sum(tf.reshape(loss_volume, [batch_size, -1]), axis=1)
        return {"loss": tf.reduce_mean(per_examples_loss, name="yolo_loss")}


# def output_channel_dim(self):
#     A, B, C = self.n_attrs, self.n_anchor_boxes, self.n_classes
#     if self.mode == MODE.DEFAULT:
#         return (1 + 4 + A + C) * B
#     elif self.mode == MODE.BBOX_PER_CLASS:
#         return (1 + C*4 + A + C) * B
#     elif self.mode == MODE.ATTR_PER_CLASS:
#         return (1 + 4 + C*A + C) * B
#     elif self.mode == MODE.BBOX_ATTR_PER_CLASS:
#         return (1 + C*4 + C*A + C) * B
#     else:
#         MODE.validate_mode(self.mode)


class TinyYoloBackbone(ConvNet):
    def __init__(self):
        # Notice the striding.
        layout = [
            dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
        ]
        super(TinyYoloBackbone, self).__init__(layout, check_output_shape=True)


class TinyYoloBackboneWithSharpening(ConvNet):
    def __init__(self):
        # Best configuration so far. Notice the striding, and the final two layers.
        layout = [
            dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
            dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
        ]
        super(TinyYoloBackboneWithSharpening, self).__init__(layout, check_output_shape=True)


class YoloBackbone(ConvNet):
    def __init__(self):
        layout = [
            dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=128, kernel_size=1, strides=1, padding="VALID"),
            dict(filters=128, kernel_size=1, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=1, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=1, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=1, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=1, strides=1, padding="VALID"),
        ]
        super(YoloBackbone, self).__init__(layout, check_output_shape=True)


def build_fcn():
    return TinyYoloBackboneWithSharpening()
    # return YoloBackbone()
    # return TinyYoloBackbone()


def get_differentiable_updater(env):
    fcn = cfg.build_fully_conv_net()
    fcn.layout[-1]['filters'] = env.channel_dim
    return DifferentiableUpdater(env, fcn)


def build_env():
    train = EmnistObjectDetectionDataset(n_examples=int(cfg.n_train), example_range=(0., 0.9))
    val = EmnistObjectDetectionDataset(n_examples=int(cfg.n_val), example_range=(0.9, 0.95))
    test = EmnistObjectDetectionDataset(n_examples=int(cfg.n_val), example_range=(0.95, 1.))

    return YOLOv2_SupervisedEnv(train, val, test, C=len(cfg.characters))


def compute_iou(box, others):
    # box: y_min, y_max, x_min, x_max, area
    # others: (n_boxes, 5)
    top = np.maximum(box[0], others[:, 0])
    bottom = np.minimum(box[1], others[:, 1])
    left = np.maximum(box[2], others[:, 2])
    right = np.minimum(box[3], others[:, 3])

    overlap_height = np.maximum(0., bottom - top)
    overlap_width = np.maximum(0., right - left)
    overlap_area = overlap_height * overlap_width

    return overlap_area / (box[4] + others[:, 4] - overlap_area)


def nms(bbox_bounds, class_probs, prob_threshold, iou_threshold):
    """ Perform non-maximum suppresion on predictions for a single image.

    Parameters
    ----------
        bbox_bounds: (H * W * B, 4)
        class_probs: (H * W * B, n_classes)
    Returns
    --------
        [class, conf, y_min, y_max, x_min, x_max] * n_boxes

    """
    n_classes = class_probs.shape[1]
    selected = []

    bbox_area = (bbox_bounds[:, 1] - bbox_bounds[:, 0]) * (bbox_bounds[:, 3] - bbox_bounds[:, 2])
    bbox_bounds = np.concatenate([bbox_bounds, bbox_area[:, None]], axis=1)

    for c in range(n_classes):
        _valid = class_probs[:, c] > prob_threshold
        _class_probs = class_probs[_valid, c]
        _bbox_bounds = bbox_bounds[_valid, :]

        while _bbox_bounds.shape[0]:
            best_idx = np.argmax(_class_probs)
            best_bbox = _bbox_bounds[best_idx]

            selected.append([c, _class_probs[best_idx], *best_bbox[:4]])

            iou = compute_iou(best_bbox, _bbox_bounds)
            _valid = iou < iou_threshold
            _class_probs = _class_probs[_valid]
            _bbox_bounds = _bbox_bounds[_valid, :]

    return selected


def top_n(n, bbox_bounds, class_probs):
    """ Get top n boxes from each class """
    n_classes = class_probs.shape[1]
    selected = []

    bbox_area = (bbox_bounds[:, 1] - bbox_bounds[:, 0]) * (bbox_bounds[:, 3] - bbox_bounds[:, 2])
    bbox_bounds = np.concatenate([bbox_bounds, bbox_area[:, None]], axis=1)

    for c in range(n_classes):
        order = sorted(range(class_probs.shape[0]), key=lambda i: -class_probs[i, c])

        for idx in order[:n]:
            selected.append([c, class_probs[idx, c], *bbox_bounds[idx, :4]])

    return selected


def yolo_render_hook(updater):
    # Run the network on a subset of the evaluation data, fetch the output
    N = 16

    env = updater.env
    images, gt_boxes = env.next_batch(N, 'val', process=False, evaluate=True)
    sess = tf.get_default_session()

    fetch = [
        env.predictions['cell_y'], env.predictions['cell_x'],
        env.predictions['normalized_h'], env.predictions['normalized_w'],
        env.predictions['confs'], env.predictions['probs'],
    ]

    feed_dict = {updater.env.x: images, updater.env.is_training: False}

    cell_y, cell_x, normalized_h, normalized_w, confs, probs = sess.run(fetch, feed_dict=feed_dict)

    center_y = (cell_y + np.arange(env.H)[None, :, None, None, None]) * env.cell_height
    center_x = (cell_x + np.arange(env.W)[None, None, :, None, None]) * env.cell_width
    height = normalized_h * env.image_height
    width = normalized_w * env.image_width

    # Convert to format that is convenient for computing IoU between bboxes.
    y_min = center_y - 0.5 * height
    y_max = center_y + 0.5 * height

    x_min = center_x - 0.5 * width
    x_max = center_x + 0.5 * width

    bbox_bounds = np.stack([y_min, y_max, x_min, x_max], axis=-1)
    bbox_bounds = bbox_bounds.reshape(N, -1, 4)

    class_probs = (confs * probs).reshape(N, -1, probs.shape[-1])

    prob_threshold = cfg.prob_threshold
    if isinstance(cfg.prob_threshold, float):
        prob_threshold = [prob_threshold]

    sqrt_N = int(np.ceil(np.sqrt(N)))

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    for pt in prob_threshold:
        fig, axes = plt.subplots(2*sqrt_N, sqrt_N, figsize=(20, 20))
        axes = np.array(axes).reshape(2*sqrt_N, sqrt_N)
        all_boxes = []
        for n, (image, gt_bxs) in enumerate(zip(images, gt_boxes)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            boxes = nms(bbox_bounds[n], class_probs[n], pt, cfg.iou_threshold)

            all_boxes.append(boxes)

            ax1 = axes[2*i, j]
            ax1.imshow(image)

            _p = [b[1] for b in boxes]
            max_p = max(_p) if _p else 0.0

            for c, p, top, bottom, left, right, _ in boxes:
                width = bottom - top
                height = right - left
                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=3,
                    edgecolor=cfg.class_colours[c], facecolor='none', alpha=p/max_p)
                ax1.add_patch(rect)

            # print("Image {}".format(n))
            # for c in range(class_probs.shape[-1]):
            #     print("Max probability for class {}: {}".format(c, class_probs[n, ..., c].max()))

            ax2 = axes[2*i+1, j]
            ax2.imshow(image)

        mean_ap = mAP(all_boxes, gt_boxes, n_classes=env.C)
        print("pt: {}".format(pt))
        print("mAP: {}".format(mean_ap))

        fig.suptitle('After {} experiences ({} updates, {} experiences per batch). prob_threshold={}. mAP={}'.format(
            updater.n_experiences, updater.n_updates, cfg.batch_size, pt, mean_ap))

        fig.savefig(os.path.join(cfg.path, 'plots', 'yolo_output_pt={}.pdf'.format(pt)))
        plt.close(fig)


def mAP(pred_boxes, gt_boxes, n_classes, recall_values=None, iou_threshold=None):
    """ Calculate mean average precision on a dataset.

    Averages over:
        classes, recall_values, iou_threshold

    pred_boxes: [[class, conf, y_min, y_max, x_min, x_max] * n_boxes] * n_images
    gt_boxes: [[class, y_min, y_max, x_min, x_max] * n_boxes] * n_images

    """
    if recall_values is None:
        recall_values = np.linspace(0.0, 1.0, 11)

    if iou_threshold is None:
        iou_threshold = np.linspace(0.5, 0.95, 10)

    ap = []

    for c in range(n_classes):
        _ap = []
        for iou_thresh in iou_threshold:
            predicted_list = []  # Each element is (confidence, ground-truth (0 or 1))
            n_positives = 0

            for pred, gt in zip(pred_boxes, gt_boxes):
                # Within a single image

                # Sort by decreasing confidence within current class.
                pred_c = sorted([b for cls, *b in pred if cls == c], key=lambda k: -k[0])
                area = [(ymax - ymin) * (xmax - xmin) for _, ymin, ymax, xmin, xmax in pred_c]
                pred_c = [(*b, a) for b, a in zip(pred_c, area)]

                gt_c = [b for cls, *b in gt if cls == c]
                n_positives += len(gt_c)

                if not gt_c:
                    predicted_list.extend((conf, 0) for conf, *b in pred_c)
                    continue

                gt_c = np.array(gt_c)
                gt_c_area = (gt_c[:, 1] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 2])
                gt_c = np.concatenate([gt_c, gt_c_area[..., None]], axis=1)

                for conf, *box in pred_c:
                    iou = compute_iou(box, gt_c)
                    best_idx = np.argmax(iou)
                    best_iou = iou[best_idx]
                    if best_iou > iou_thresh:
                        predicted_list.append((conf, 1.))
                        gt_c = np.delete(gt_c, best_idx, axis=0)
                    else:
                        predicted_list.append((conf, 0.))

                    if not gt_c.shape[0]:
                        break

            if not predicted_list:
                ap.append(0.0)
                continue

            # Sort predictions by decreasing confidence.
            predicted_list = np.array(sorted(predicted_list, key=lambda k: -k[0]), dtype=np.float32)

            # Compute AP
            cs = np.cumsum(predicted_list[:, 1])
            precision = cs / (np.arange(predicted_list.shape[0]) + 1)
            recall = cs / n_positives

            for r in recall_values:
                p = precision[recall >= r]
                _ap.append(0. if p.size == 0 else p.max())
        ap.append(np.mean(_ap))
    return np.mean(ap)


class Yolo_RLUpdater(DifferentiableUpdater):
    def _build_distributions(self, coords_yx, coords_hw, confs, probs, exploration):
        b_exp, s_exp, yx_exp, hw_exp = exploration

        effective_example_size = 1. / yx_exp
        c0 = (1 - coords_yx) * effective_example_size
        c1 = coords_yx * effective_example_size
        yx = tf.distributions.Beta(concentration0=c0, concentration1=c1)

        effective_example_size = 1. / hw_exp
        c0 = (1 - coords_yx) * effective_example_size
        c1 = coords_yx * effective_example_size
        hw = tf.distributions.Beta(concentration0=c0, concentration1=c1)

        existence = tf.distributions.Bernoulli(probs=confs/b_exp)
        cls = tf.distributions.Categorical(probs=probs/s_exp)

        return yx, hw, existence, cls

    def _build_log_prob(self, coords, confs, probs):
        pass

    def _build_graph(self):
        # Run network forward, sample from induced distribution, get reward of sample, construct REINFORCE loss function.
        self.x_ph = tf.placeholder(tf.float32, (None,) + self.obs_shape, name="x_ph")
        self.output = self.f(self.x_ph, self.action_shape, self.is_training)

        dists = self._build_distributions(
            self.env.adjusted_coords_yx,
            self.env.adjusted_coords_hw,
            self.env.adjusted_confs,
            self.env.adjusted_probs
        )

        self._samples = [d.sample() for d in dists]
        self._input_samples = [
            tf.placeholder(tf.float32, shape=(None, *s.shape[1:]), name="input_sample")
            for s in self._samples]
        self._reward = tf.placeholder(tf.float32, (None, 1), name="reward")
        self._log_prob = [d.log_probs(i) for i, d in zip(self._input_samples, dists)]

        self.loss = self._log_prob * self._reward

        self.recorded_tensors = [
            tf.reduce_mean(getattr(self.env, 'build_' + name)(self.output, self.target_ph))
            for name in self.env.recorded_names
        ]

        tvars = self.trainable_variables(for_opt=True)
        self.train_op, train_summaries = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.summary_op = tf.summary.merge(
            [tf.summary.scalar("loss_per_ep", self.loss)] +
            [tf.summary.scalar("reward_per_ep", self._reward)] +
            [tf.summary.scalar(name, t) for name, t in zip(self.env.recorded_names, self.recorded_tensors)] +
            train_summaries)

    def _update(self, batch_size, collect_summaries):
        self.set_is_training(True)
        x, y = self.env.next_batch(batch_size, mode='train')

        feed_dict = {self.x_ph: x}
        sess = tf.get_default_session()

        samples = sess.run(self._samples, feed_dict=feed_dict)
        reward = self.get_reward(samples)

        feed_dict.update({i: s for i, s in zip(self._input_samples, samples)})
        feed_dict.update({self._reward: reward})

        if collect_summaries:
            train_summaries, _, *recorded_values = sess.run(
                [self.summary_op, self.train_op] + self.recorded_tensors, feed_dict=feed_dict)
            return train_summaries, b'', dict(zip(self.env.recorded_names, recorded_values)), {}
        else:
            _, *recorded_values = sess.run([self.train_op] + self.recorded_tensors, feed_dict=feed_dict)
            return b'', b'', dict(zip(self.env.recorded_names, recorded_values)), {}


xkcd_colors = 'viridian,cerulean,vermillion,lavender,celadon,fuchsia,saffron,cinnamon,greyish,vivid blue'.split(',')


config = Config(
    env_name="yolo",

    get_updater=get_differentiable_updater,
    render_hook=yolo_render_hook,
    render_step=500,

    curriculum=[dict(lr_schedule=lr) for lr in [1e-4, 1e-5, 1e-6]],
    preserve_env=True,

    # backbone
    build_fully_conv_net=build_fcn,

    # env parameters
    build_env=build_env,
    max_overlap=20,
    image_shape=(40, 40),
    characters=[0, 1, 2, 3],
    min_chars=1,
    max_chars=3,
    patch_shape=(14, 14),
    n_patch_examples=0,
    colours='red green blue',

    anchor_boxes=[[14, 14]],

    # display params
    prob_threshold=1e-4,
    iou_threshold=0.5,
    class_colours=['xkcd:' + c for c in xkcd_colors],

    # number of grid cells - depends on both the FCN backbone and the image size
    H=7,
    W=7,

    n_train=1e5,
    n_val=1e2,

    # loss params
    scale_class=1.0,
    scale_obj=1.0,
    scale_no_obj=0.5,
    scale_coord=5.0,

    use_squared_loss=True,
    conf_target="conf",

    # training params
    batch_size=16,
    # batch_size=64,
    eval_step=100,
    max_steps=1e7,
    patience=10000,
    optimizer_spec="adam",
    use_gpu=True,
    gpu_allow_growth=True,
    seed=347405995,
    stopping_criteria="loss,min",
    threshold=-np.inf,
    max_grad_norm=1.0,
)
