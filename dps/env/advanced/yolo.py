import numpy as np
import tensorflow as tf
import os

from dps import cfg
from dps.updater import DifferentiableUpdater
from dps.env.supervised import SupervisedEnv
from dps.datasets.base import EMNIST_ObjectDetection
from dps.utils import Config, Param
from dps.utils.tf import ScopedFunction


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

    def next_batch(self, batch_size, mode, process=True):
        advance = mode == 'train'
        images, annotations = self.datasets[mode].next_batch(batch_size=batch_size, advance=advance)
        if process:
            annotations = self.process_annotations(annotations)
        return images, annotations

    def process_annotations(self, annotations):
        """ Turn a batch of bounding box annotations into a target volume.

        Parameters
        ----------
        annotations:
            Batch (list) of annotations to process. Each annotation is
            a list of bounding boxes for a single image (the image itself
            is not required at this point).

            Each annotation has the form:  cls (int), y_min, y_max, x_min, x_max
            The last 4 entries in the annotation have units of pixels.

        Returns
        -------
        target_volume

        """
        H, W, B, C = self.H, self.W, self.B, self.C
        HW = H * W
        D = 4 + 1 + C

        # All images must have same size
        image_height, image_width = self.image_height, self.image_width

        cell_height = 1. * image_height / H
        cell_width = 1. * image_width / W

        target_volume = np.zeros((len(annotations), H, W, B * (4 + 1 + C)), dtype=np.float32)

        for batch_idx, objects in enumerate(annotations):
            # Populate these.
            coords = np.zeros([HW, B, 4])
            confs = np.zeros([HW, B, 1])
            probs = np.zeros([HW, B, C])

            # assume anchor_boxes has shape (n_anchor_boxes, 3), columns are (h, w, a), all in pixels.
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

                bbox_height_target = np.sqrt(float(y_max - y_min) / image_height)
                bbox_width_target = np.sqrt(float(x_max - x_min) / image_width)

                # bbox center in cell coordinates
                cell_center_y = cy - np.floor(cy)
                cell_center_x = cx - np.floor(cx)

                # index of cell that the object is assigned to
                cell_idx = int(np.floor(cy) * W + np.floor(cx))

                # find anchor box whose shape is most like ground-truth shape
                pixel_height = y_max - y_min
                pixel_width = x_max - x_min
                pixel_area = 1. * pixel_height * pixel_width

                overlap_height = np.minimum(pixel_height, self.anchor_boxes[:, 0])
                overlap_width = np.minimum(pixel_width, self.anchor_boxes[:, 1])
                overlap_area = overlap_height * overlap_width
                IOU = overlap_area / (self.anchor_boxes[:, 2] + pixel_area - overlap_area)
                anchor_box_idx = np.argmax(IOU)

                _coords = cell_center_y, cell_center_x, bbox_height_target, bbox_width_target

                coords[cell_idx, anchor_box_idx, :] = np.array(_coords, ndmin=2)
                confs[cell_idx, anchor_box_idx, :] = 1
                probs[cell_idx, anchor_box_idx, cls] = 1

            target = np.concatenate([coords, confs, probs], axis=2)
            target_volume[batch_idx, ...] = target.reshape(H, W, B*D)

        return target_volume

    def build_loss(self, action, target):
        """
        action: tensor, (batch_dim, H, W, B * (4 + 1 + C))
            Output of the network.
        target: tensor, (batch_dim, H, W, B * (4 + 1 + C))
            Target volume.

        """
        H, W, B, C = self.H, self.W, self.B, self.C
        HW = H * W
        D = 4 + 1 + C

        action = tf.reshape(action, [-1, HW, B, D])
        target = tf.reshape(target, [-1, HW, B, D])

        _coords, _confs, _probs = tf.split(target, [4, 1, C], 3)
        coords, confs, probs = tf.split(action, [4, 1, C], 3)

        # predict bbox center in cell coordinates
        adjusted_coords_yx = tf.nn.sigmoid(coords[..., :2])

        # use anchor boxes to predict sqrt of box height and width (normalized to image size)
        adjusted_coords_hw = tf.sqrt(tf.exp(coords[..., 2:]) * self._normalized_anchor_boxes)

        adjusted_coords = tf.concat([adjusted_coords_yx, adjusted_coords_hw], 3)
        self.adjusted_coords = tf.reshape(adjusted_coords, [-1, H, W, B, 4])
        coord_loss = self.scale_coord * _confs * (adjusted_coords - _coords)**2

        if self.use_squared_loss:
            adjusted_confs = tf.nn.sigmoid(confs)
            self.adjusted_confs = tf.reshape(adjusted_confs, [-1, H, W, B, 1])

            conf_loss = self.scale_obj * _confs * (adjusted_confs - _confs)**2
            conf_loss += self.scale_no_obj * (1 - _confs) * (adjusted_confs - _confs)**2

            adjusted_probs = tf.nn.softmax(probs)
            self.adjusted_probs = tf.reshape(adjusted_probs, [-1, H, W, B, C])

            prob_loss = self.scale_class * _confs * (adjusted_probs - _probs)**2
        else:
            conf_loss = self.scale_obj * _confs * (
                tf.nn.sigmoid_cross_entropy_with_logits(labels=_confs, logits=confs))
            conf_loss += self.scale_no_obj * (1 - _confs) * (
                tf.nn.sigmoid_cross_entropy_with_logits(labels=_confs, logits=confs))

            self.adjusted_confs = tf.reshape(tf.nn.sigmoid(confs), [-1, H, W, B, 1])

            _prob_loss = tf.nn.softmax_cross_entropy_with_logits(labels=_probs, logits=probs)
            prob_loss = self.scale_class * _confs * _prob_loss[..., None]

            self.adjusted_probs = tf.reshape(tf.nn.softmax(probs), [-1, H, W, B, C])

        loss_volume = tf.concat([coord_loss, conf_loss, prob_loss], axis=3)  # (batch_dim, HW, B, D)

        return tf.reduce_sum(tf.reshape(loss_volume, [-1, HW*B*D]), axis=1)


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


class FullyConvolutional(ScopedFunction):
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
    def __init__(self, layout, check_output_shape=False, scope=None):
        self.layout = layout
        self.check_output_shape = check_output_shape
        super(FullyConvolutional, self).__init__(scope)

    def output_shape(self, input_shape):
        """ Get spatial shape of the output given a spatial shape of the input. """
        shape = [int(i) for i in input_shape]
        for layer in self.layout:
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

            shape[0] = int((shape[0] - f0) / strides0) + 1
            shape[1] = int((shape[1] - f1) / strides1) + 1

            if layer.get('pool', False):
                shape[0] = int(shape[0] / 2)
                shape[1] = int(shape[0] / 2)
        return shape

    def _call(self, inp, output_size, is_training):
        volume = inp
        print("Predicted output shape is: {}".format(self.output_shape(inp.shape[1:3])))

        for i, layer in enumerate(self.layout):
            filters = layer['filters']
            strides = layer['strides']
            kernel_size = layer['kernel_size']

            volume = tf.layers.conv2d(
                volume, filters=filters, kernel_size=kernel_size,
                strides=strides, padding="valid", name="fcn-conv{}".format(i))

            if i < len(self.layout) - 1:
                volume = tf.nn.relu(volume, name="fcn-relu{}".format(i))

                if layer.get('pool', False):
                    volume = tf.layers.max_pooling2d(
                        volume, pool_size=2, strides=2, name='fcn-pool{}'.format(i))

            print("FCN >>> Spatial shape after layer {}: {}".format(i, tuple(int(i) for i in volume.shape[1:3])))

        if self.check_output_shape and output_size is not None:
            actual_shape = tuple(int(i) for i in volume.shape[1:])

            if actual_shape == output_size:
                print("FCN >>> Shape check passed.")
            else:
                raise Exception(
                    "Shape-checking turned on, and actual shape {} does not "
                    "match desired shape {}.".format(actual_shape, output_size))

        return volume


# class TinyYoloBackbone(FullyConvolutional):
#     def __init__(self):
#         layout = [
#             dict(filters=64, kernel_size=6, strides=1),
#             dict(filters=128, kernel_size=6, strides=1),
#             dict(filters=128, kernel_size=8, strides=1),
#             dict(filters=128, kernel_size=6, strides=1),
#             dict(filters=512, kernel_size=6, strides=1),
#         ]
#         super(TinyYoloBackbone, self).__init__(layout)

class TinyYoloBackbone(FullyConvolutional):
    def __init__(self):
        layout = [
            dict(filters=64, kernel_size=6, strides=1),
            dict(filters=128, kernel_size=6, strides=1),
            dict(filters=256, kernel_size=8, strides=1),
            dict(filters=256, kernel_size=6, strides=1),
            dict(filters=512, kernel_size=6, strides=1),
        ]
        super(TinyYoloBackbone, self).__init__(layout)


class YoloBackbone(FullyConvolutional):
    def __init__(self):
        layout = [
            dict(filters=64, kernel_size=5, strides=1),
            dict(filters=64, kernel_size=5, strides=1),
            dict(filters=128, kernel_size=5, strides=1),
            dict(filters=128, kernel_size=5, strides=1),
            dict(filters=256, kernel_size=4, strides=1),
            dict(filters=256, kernel_size=5, strides=1),
            dict(filters=256, kernel_size=6, strides=1),
            dict(filters=512, kernel_size=6, strides=1),
        ]
        super(YoloBackbone, self).__init__(layout, check_output_shape=True)


def build_fcn():
    return YoloBackbone()
    # return TinyYoloBackbone()


def get_differentiable_updater(env):
    fcn = cfg.build_fully_conv_net()
    fcn.layout[-1]['filters'] = env.channel_dim
    return DifferentiableUpdater(env, fcn)


def build_env():
    train = EMNIST_ObjectDetection(n_examples=cfg.n_train)
    val = EMNIST_ObjectDetection(n_examples=cfg.n_val)
    test = EMNIST_ObjectDetection(n_examples=cfg.n_val)

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

            selected.append([c, _class_probs[best_idx], *best_bbox])

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
            selected.append([c, class_probs[idx, c], *bbox_bounds[idx, :]])

    return selected


def yolo_render_hook(updater):
    # Run the network on a subset of the evaluation data, fetch the output
    N = 16

    updater.set_is_training(False)
    env = updater.env
    images, gt_boxes = env.next_batch(N, 'val', process=False)
    sess = tf.get_default_session()

    fetch = [env.adjusted_coords, env.adjusted_confs, env.adjusted_probs]
    coords, confs, probs = sess.run(fetch, feed_dict={updater.x_ph: images})

    center_y = (coords[..., 0] + np.arange(env.H)[None, :, None, None]) * env.cell_height
    center_x = (coords[..., 1] + np.arange(env.W)[None, None, :, None]) * env.cell_width
    height = coords[..., 2]**2 * env.image_height
    width = coords[..., 3]**2 * env.image_width

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

            # boxes = top_n(5, bbox_bounds[n], class_probs[n])
            boxes = nms(bbox_bounds[n], class_probs[n], pt, cfg.iou_threshold)
            all_boxes.append(boxes)

            ax1 = axes[2*i, j]
            ax1.imshow(image)

            for c, p, top, bottom, left, right, _ in boxes:
                width = bottom - top
                height = right - left
                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1,
                    edgecolor=cfg.class_colours[c], facecolor='none')
                ax1.add_patch(rect)

            print("Image {}".format(n))
            for c in range(class_probs.shape[-1]):
                print("Max probability for class {}: {}".format(c, class_probs[n, ..., c].max()))

            ax2 = axes[2*i+1, j]
            ax2.imshow(image)

        mean_ap = mAP(all_boxes, gt_boxes, n_classes=env.C)
        print("mAP: {}".format(mean_ap))

        fig.suptitle('After {} experiences ({} updates, {} experiences per batch). prob_threshold={}. mAP={}'.format(
            updater.n_experiences, updater.n_updates, cfg.batch_size, pt, mean_ap))

        fig.savefig(os.path.join(str(cfg.path), 'plots', 'yolo_output_pt={}.pdf'.format(pt)))
        plt.close(fig)


def mAP(pred_boxes, gt_boxes, n_classes, recall_values=None, iou_threshold=0.5):
    """ Calculate mean average precision on a dataset.

    pred_boxes: [[class, conf, y_min, y_max, x_min, x_max] * n_boxes] * n_images
    gt_boxes: [[class, y_min, y_max, x_min, x_max] * n_boxes] * n_images

    """
    if recall_values is None:
        recall_values = np.linspace(0.0, 1.0, 11)

    ap = []

    for c in range(n_classes):
        predicted_list = []  # Each element is (confidence, ground-truth (0 or 1))

        n_positives = 0

        for pred, gt in zip(pred_boxes, gt_boxes):
            # Within a single image
            pred_c = sorted([b for cls, *b in pred if cls == c], key=lambda k: -k[0])

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
                if best_iou > iou_threshold:
                    predicted_list.append((conf, 1.))
                    gt_c = np.delete(gt_c, best_idx, axis=0)
                else:
                    predicted_list.append((conf, 0.))

                if not gt_c.shape[0]:
                    break

        if not predicted_list:
            ap.append(0.0)
            continue

        # Sort predictions by confidence.
        predicted_list = np.array(sorted(predicted_list, key=lambda k: -k[0]), dtype=np.float32)

        # Compute AP
        cs = np.cumsum(predicted_list[:, 1])
        precision = cs / (np.arange(predicted_list.shape[0]) + 1)
        recall = cs / n_positives

        _ap = []

        for r in recall_values:
            p = precision[recall >= r]
            _ap.append(0. if p.size == 0 else p.max())

        ap.append(np.mean(_ap))

    return np.mean(ap)


config = Config(
    log_name="yolo",

    get_updater=get_differentiable_updater,
    render_hook=yolo_render_hook,
    render_step=500,

    # backbone
    build_fully_conv_net=build_fcn,

    predict_counts=False,

    # env parameters
    build_env=build_env,
    max_overlap=20,
    image_shape=(40, 40),
    characters=[0, 1, 2],
    min_chars=2,
    max_chars=3,
    # characters=list(range(10)) + [chr(i + ord('a')) for i in range(10)],
    sub_image_shape=(14, 14),
    n_sub_image_examples=1000,
    colours='red green blue',

    anchor_boxes=[[14, 14]],

    # display params
    prob_threshold=1. / (2**np.arange(1, 11)),
    # prob_threshold=0.6,  # Andrew Ng
    # iou_threshold=0.0,
    iou_threshold=0.5,  # Andrew Ng
    class_colours='white yellow purple'.split(),

    # number of grid cells - the depends on both the FCN backbone and the image size
    H=7,
    W=7,

    n_train=100000,
    n_val=100,
    n_test=100,

    # loss params
    scale_class=1.0,
    scale_obj=1.0,
    scale_no_obj=0.5,
    scale_coord=5.0,

    use_squared_loss=True,

    # training params
    batch_size=64,
    eval_step=100,
    max_steps=1e7,
    patience=10000,
    lr_schedule="1e-5",
    optimizer_spec="adam",
    use_gpu=True,
    gpu_allow_growth=True,
    seed=347405995,
    stopping_criteria="loss,min",
    threshold=-np.inf,
    max_grad_norm=5.0,
)
