import tensorflow as tf
import numpy as np
import os

from dps import cfg
from dps.env.advanced.mnist_vae import VAE_Env
from dps.datasets import AutoencodeDataset, EMNIST_ObjectDetection
from dps.updater import DifferentiableUpdater
from dps.utils import Config, Param
from dps.utils.tf import FullyConvolutional, ScopedFunction


def build_env():
    _train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train)).x
    train = AutoencodeDataset(_train, image=True)

    _val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val)).x
    val = AutoencodeDataset(_val, image=True)

    _test = EMNIST_ObjectDetection(n_examples=int(cfg.n_val)).x
    test = AutoencodeDataset(_test, image=True)

    return YOLOv2_UnsupervisedEnv(train, val, test)


def get_updater(env):
    network = YOLOv2_UnsupervisedNetwork()
    return DifferentiableUpdater(env, network)


class TinyYoloBackbone1D(FullyConvolutional):
    def __init__(self, **kwargs):
        layout = [
            # dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            # dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            # dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
            # dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
            # dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
            # dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            # dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
            # dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
            dict(filters=128, kernel_size=3, strides=2, padding="SAME"),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME"),
            # dict(filters=256, kernel_size=3, strides=1, padding="VALID"),  # To make output have size 3
            # dict(filters=256, kernel_size=3, strides=1, padding="VALID"),
            dict(filters=256, kernel_size=4, strides=1, padding="VALID"),  # To make output have size 1
            dict(filters=256, kernel_size=4, strides=1, padding="VALID"),
            # dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
            # dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
            # dict(filters=256, kernel_size=7, strides=1, padding="SAME"),
        ]
        super(TinyYoloBackbone1D, self).__init__(layout, check_output_shape=True, **kwargs)


class ObjectDecoder(FullyConvolutional):
    def __init__(self, **kwargs):
        layout = [
            dict(filters=128, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=256, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=256, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=3, kernel_size=4, strides=1, padding="SAME", transpose=True),  # For 14 x 14 output

            # dict(filters=256, kernel_size=7, strides=1, padding="SAME", transpose=True),
            # dict(filters=256, kernel_size=4, strides=1, padding="VALID", transpose=True),
            # dict(filters=256, kernel_size=4, strides=1, padding="VALID", transpose=True),
            # dict(filters=256, kernel_size=3, strides=2, padding="SAME", transpose=True),
            # dict(filters=3, kernel_size=3, strides=2, padding="SAME", transpose=True),  # For 28 x 28 output
        ]
        super(ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)


class YOLOv2_UnsupervisedNetwork(ScopedFunction):
    H = Param()
    W = Param()
    C = Param()
    A = Param(help="Dimension of attribute vector.")
    anchor_boxes = Param(help="List of (h, w) pairs.")
    object_shape = Param()
    position_noise_std = Param()
    size_noise_std = Param()
    sigmoid_prob = Param()

    def __init__(self, scope=None):
        self._encoder = None
        self._object_decoders = None
        self.anchor_boxes = np.array(self.anchor_boxes)
        self.B = len(self.anchor_boxes)
        self.D = 4 + 1 + self.C + self.A

        super(YOLOv2_UnsupervisedNetwork, self).__init__(scope=scope)

    @property
    def channel_dim(self):
        return self.B*self.D

    def __call__(self, inp, output_size, is_training):
        H, W, B, C, A = self.H, self.W, self.B, self.C, self.A

        if self._encoder is None:
            self._encoder = cfg.build_encoder()
            self._encoder.layout[-1]['filters'] = self.channel_dim

        if self._object_decoders is None:
            self._object_decoders = [cfg.build_object_decoder(scope="object_decoder_{}".format(i)) for i in range(C)]

        _code = self._encoder(inp, (H, W, self.channel_dim), is_training)
        self._code = tf.reshape(_code, (-1, H, W, B, self.D))

        yx_logits, hw_logits, conf_logits, class_logits, attributes = tf.split(self._code, [2, 2, 1, C, A], -1)

        self.predictions = dict(attributes=attributes, code=self._code)

        # predict bbox center in cell coordinates
        cell_yx = tf.nn.sigmoid(yx_logits)
        if self.position_noise_std:
            cell_yx += tf.random_normal(tf.shape(cell_yx), 0.0, self.position_noise_std)

        image_height, image_width, image_depth = [int(i) for i in inp.shape[1:]]

        normalized_anchor_boxes = self.anchor_boxes / [image_height, image_width]
        normalized_anchor_boxes = normalized_anchor_boxes.reshape(1, 1, 1, B, 2)

        # use anchor boxes to predict box height and width (normalized to image size)
        hw = tf.nn.sigmoid(hw_logits) * normalized_anchor_boxes
        if self.size_noise_std:
            hw += tf.random_normal(tf.shape(hw), 0.0, self.size_noise_std)

        cell_y, cell_x = tf.split(cell_yx, 2, axis=-1)
        h, w = tf.split(hw, 2, axis=-1)

        self.predictions.update(cell_y=cell_y, cell_x=cell_x, normalized_h=h, normalized_w=w)

        y = (cell_y + tf.range(H, dtype=tf.float32)[None, :, None, None, None]) / H  # In normalized (0, 1) image coordinates
        x = (cell_x + tf.range(W, dtype=tf.float32)[None, None, :, None, None]) / W

        y_min, y_max = y - 0.5 * h, y + 0.5 * h
        x_min, x_max = x - 0.5 * w, x + 0.5 * w

        self.predictions['boxes_normalized'] = tf.concat([y_min, y_max, x_min, x_max], axis=-1)

        _y_min = tf.reshape(y_min, (-1, 1))
        _y_max = tf.reshape(y_max, (-1, 1))
        _h = _y_max - _y_min

        _x_min = tf.reshape(x_min, (-1, 1))
        _x_max = tf.reshape(x_max, (-1, 1))
        _w = _x_max - _x_min

        boxes = [-_y_min/_h, -_x_min/_w, 1 + (1 - _y_max)/_h, 1 + (1 - _x_max)/_w]
        boxes = tf.concat(boxes, axis=1)

        sigmoid = tf.nn.sigmoid(conf_logits)

        if self.sigmoid_prob:
            sigmoid_samples = tf.distributions.Bernoulli(probs=self.sigmoid_prob).sample(tf.shape(sigmoid))
        else:
            sigmoid_samples = tf.distributions.Bernoulli(probs=sigmoid).sample()

        _sigmoid_samples = tf.to_float(sigmoid_samples)
        _sigmoid_samples = tf.stop_gradient(tf.maximum(_sigmoid_samples, 1e-6))

        softmax = tf.nn.softmax(class_logits)

        softmax_samples = tf.distributions.Categorical(probs=softmax).sample()
        softmax_samples = tf.one_hot(softmax_samples, depth=C)

        _softmax_samples = tf.to_float(softmax_samples)
        _softmax_samples = tf.stop_gradient(tf.maximum(_softmax_samples, 1e-6))

        self.predictions.update(
            confs=sigmoid, probs=softmax,
            sampled_confs=sigmoid_samples,
            sampled_probs=softmax_samples)

        object_decoder_in = tf.concat([y, x, hw, attributes], axis=-1)
        object_decoder_in = tf.reshape(object_decoder_in, (-1, 1, 1, 4 + A))

        output_logits = None
        sampled_output_logits = None

        self.predictions['object_decoder_sigmoid'] = {}

        for c, od in enumerate(self._object_decoders):
            object_decoder_out = od(object_decoder_in, self.object_shape + (image_depth,), is_training)

            self.predictions['object_decoder_sigmoid'][c] = tf.reshape(
                tf.nn.sigmoid(object_decoder_out), (-1, H, W, B,) + self.object_shape + (image_depth,))

            object_decoder_transformed = tf.image.crop_and_resize(
                image=object_decoder_out,
                boxes=boxes,
                box_ind=tf.range(tf.shape(object_decoder_out)[0]),
                crop_size=(image_height, image_width),
                extrapolation_value=-100,
            )

            object_decoder_transformed = tf.reshape(
                object_decoder_transformed,
                [-1, H, W, B, image_height, image_width, image_depth]
            )

            # --- Unsampled ---
            weighted_images = (
                object_decoder_transformed +
                tf.log(softmax[..., c:c+1, None, None]) +
                tf.log(sigmoid[..., None, None])
            )
            weighted_images = tf.reshape(weighted_images, [-1, H*W*B, image_height, image_width, image_depth])

            # Because we are going to be taking the sigmoid of the logits to get the final image,
            # we want to make sure that we are *adding* in probability space, rather than multiplying.
            per_class_images = tf.reduce_logsumexp(weighted_images, axis=1)

            if output_logits is None:
                output_logits = per_class_images
            else:
                output_logits = tf.reduce_logsumexp(
                    tf.stack([output_logits, per_class_images], axis=0),
                    axis=0)

            # --- Sampled ---
            sampled_weighted_images = (
                object_decoder_transformed +
                tf.log(_softmax_samples[..., c:c+1, None, None]) +
                tf.log(_sigmoid_samples[..., None, None])
            )
            sampled_weighted_images = tf.reshape(
                sampled_weighted_images, [-1, H*W*B, image_height, image_width, image_depth])

            sampled_per_class_images = tf.reduce_logsumexp(sampled_weighted_images, axis=1)

            if sampled_output_logits is None:
                sampled_output_logits = sampled_per_class_images
            else:
                sampled_output_logits = tf.reduce_logsumexp(
                    tf.stack([sampled_output_logits, sampled_per_class_images], axis=0),
                    axis=0)

        self.predictions['output_logits'] = output_logits
        self.predictions['output_images'] = tf.nn.sigmoid(output_logits)

        self.predictions['sampled_output_logits'] = sampled_output_logits
        self.predictions['sampled_output_images'] = tf.nn.sigmoid(sampled_output_logits)

        return output_logits


class YOLOv2_UnsupervisedEnv(VAE_Env):
    object_sparsity = Param()
    class_sparsity = Param()
    sample_weight = Param()

    def _build(self):
        output_logits = self.f.predictions['output_logits']
        recorded_tensors = {
            name: tf.reduce_mean(getattr(self, 'build_' + name)(output_logits, self.x))
            for name in ['xent_loss', '2norm_loss', '1norm_loss']
        }

        sampled_output_logits = self.f.predictions['sampled_output_logits']
        recorded_tensors.update({
            "sampled_" + name: tf.reduce_mean(getattr(self, 'build_' + name)(sampled_output_logits, self.x))
            for name in ['xent_loss', '2norm_loss', '1norm_loss']
        })

        recorded_tensors['confs'] = tf.reduce_mean(self.f.predictions['confs'])
        recorded_tensors['probs_max'] = tf.reduce_mean(tf.reduce_max(self.f.predictions['probs'], axis=-1))
        recorded_tensors['probs'] = tf.reduce_mean(self.f.predictions['probs'])

        loss_key = 'xent_loss' if self.xent_loss else '2norm_loss'
        recorded_tensors['loss'] = recorded_tensors[loss_key]

        batch_size = tf.to_float(tf.shape(self.f.predictions['confs']))[0]

        if self.object_sparsity:
            recorded_tensors['_object_sparsity_loss'] = tf.reduce_sum(self.f.predictions['confs']) / batch_size
            recorded_tensors['object_sparsity_loss'] = self.object_sparsity * recorded_tensors['_object_sparsity_loss']

            recorded_tensors['loss'] += recorded_tensors['object_sparsity_loss']

        if self.class_sparsity:
            recorded_tensors['_class_sparsity_loss'] = tf.reduce_sum(self.f.predictions['probs']) / batch_size
            recorded_tensors['class_sparsity_loss'] = self.class_sparsity * recorded_tensors['_class_sparsity_loss']

            recorded_tensors['loss'] += recorded_tensors['class_sparsity_loss']

        if self.sample_weight:
            recorded_tensors['_sample_loss'] = recorded_tensors["sampled_" + loss_key]
            recorded_tensors['sample_loss'] = self.sample_weight * recorded_tensors['_sample_loss']

            recorded_tensors['loss'] += recorded_tensors['sample_loss']

        return recorded_tensors


class YoloUnsupRenderHook(object):

    def __call__(self, updater, N=16):
        fetched = self._fetch(N, updater)

        self._plot_reconstruction(updater, fetched, True)
        self._plot_patches(updater, fetched, True)

        self._plot_reconstruction(updater, fetched, False)
        self._plot_patches(updater, fetched, False)

    def _fetch(self, N, updater):
        feed_dict = updater.env.make_feed_dict(N, 'val', True)
        images = feed_dict[updater.env.x]

        to_fetch = {
            name: updater.f.predictions[name]
            for name in "boxes_normalized output_images "
                        "confs probs object_decoder_sigmoid".split()}

        to_fetch.update({
            name: updater.f.predictions[name]
            for name in "sampled_output_images sampled_confs sampled_probs".split()})

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        fetched.update(images=images)
        return fetched

    def _plot_reconstruction(self, updater, fetched, sampled):
        images = fetched['images']
        boxes_normalized = fetched['boxes_normalized']
        N = images.shape[0]

        if sampled:
            output_images = fetched['sampled_output_images']
            confs = fetched['sampled_confs']
            probs = fetched['sampled_probs']
        else:
            output_images = fetched['output_images']
            confs = fetched['confs']
            probs = fetched['probs']

        _, image_height, image_width, _ = images.shape

        boxes = boxes_normalized * [image_height, image_height, image_width, image_width]
        y_min, y_max, x_min, x_max = np.split(boxes, 4, axis=-1)

        height = y_max - y_min
        width = x_max - x_min

        cls = np.argmax(probs, axis=-1)[..., None]
        area = height * width
        bbox_bounds = np.stack([cls, confs, y_min, height, x_min, width, area], axis=-1)
        bbox_bounds = bbox_bounds.reshape(N, -1, 7)

        sqrt_N = int(np.ceil(np.sqrt(N)))

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, axes = plt.subplots(2*sqrt_N, sqrt_N, figsize=(20, 20))
        axes = np.array(axes).reshape(2*sqrt_N, sqrt_N)
        for n, (pred, gt) in enumerate(zip(output_images, images)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            ax1 = axes[2*i, j]
            ax1.imshow(pred)
            ax1.set_title('reconstruction')

            boxes = bbox_bounds[n]

            for c, conf, top, height, left, width, _ in boxes:
                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=3,
                    edgecolor=cfg.class_colours[int(c)], facecolor='none',
                    alpha=conf)
                ax1.add_patch(rect)

            ax2 = axes[2*i+1, j]
            ax2.imshow(gt)
            ax2.set_title('actual')

        fig.suptitle('Sampled={}. After {} experiences ({} updates, {} experiences per batch).'.format(
            sampled, updater.n_experiences, updater.n_updates, cfg.batch_size))

        if sampled:
            fig.savefig(os.path.join(cfg.path, 'plots', 'sampled_reconstruction.pdf'))
        else:
            fig.savefig(os.path.join(cfg.path, 'plots', 'reconstruction.pdf'))

        plt.close(fig)

    def _plot_patches(self, updater, fetched, sampled):
        # Create a plot showing what each object is generating (for the 1st image only)
        import matplotlib.pyplot as plt

        object_decoder_sigmoid = fetched['object_decoder_sigmoid']

        H, W, C, B = [getattr(updater.f, a) for a in "H W C B".split()]
        fig, axes = plt.subplots(H * C, W * B, figsize=(20, 20))
        axes = np.array(axes).reshape(H * C, W * B)

        if sampled:
            confs = fetched['sampled_confs']
            probs = fetched['sampled_probs']
        else:
            confs = fetched['confs']
            probs = fetched['probs']

        p = (confs * probs).max(-1, keepdims=True)
        c = np.argmax(p, axis=-1)[..., None]

        for i in range(H):
            for j in range(W):
                for c in range(C):
                    for b in range(B):
                        ax = axes[i * C + c, j * B + b]

                        prob = probs[0, i, j, b, c]
                        conf = confs[i, i, j, b, 0]

                        ax.set_ylabel("class: {}, prob: {}".format(c, prob))
                        ax.set_xlabel("anchor box {}, conf: {}".format(b, conf))

                        if c == 0 and b == 0:
                            ax.set_title("({}, {})".format(i, j))

                        ax.imshow(object_decoder_sigmoid[c][0, i, j, b])

        if sampled:
            fig.savefig(os.path.join(cfg.path, 'plots', 'sampled_patches.pdf'))
        else:
            fig.savefig(os.path.join(cfg.path, 'plots', 'patches.pdf'))

        plt.close(fig)


xkcd_colors = 'viridian,cerulean,vermillion,lavender,celadon,fuchsia,saffron,cinnamon,greyish,vivid blue'.split(',')


config = Config(
    log_name="yolo_unsup_single",
    build_env=build_env,
    get_updater=get_updater,
    min_chars=1,
    max_chars=1,
    characters=[0, 1, 2],
    n_sub_image_examples=0,
    build_encoder=TinyYoloBackbone1D,
    build_object_decoder=ObjectDecoder,
    xent_loss=True,
    image_shape=(28, 28),
    sub_image_shape=(14, 14),

    render_hook=YoloUnsupRenderHook(),
    render_step=500,

    # model params
    object_shape=(14, 14),
    anchor_boxes=[[14, 14]],
    H=1,
    W=1,
    C=1,
    A=100,

    # display params
    class_colours=['xkcd:' + c for c in xkcd_colors],

    n_train=1e5,
    n_val=1e2,
    n_test=1e2,

    object_sparsity=0.0,  # Within a single image, we want as few bounding boxes to be active as possible
    class_sparsity=0.0,  # We want each of the class distributions to be as spare as possible
    sample_weight=0.0,
    sigmoid_prob=0.0,
    position_noise_std=0.1,
    size_noise_std=0.1,

    # training params
    beta=1.0,
    curriculum=[dict(lr_schedule=lr) for lr in [1e-4, 1e-5, 1e-6]],
    # curriculum=[dict(lr_schedule=lr) for lr in [1e-4]],
    preserve_env=True,
    batch_size=16,
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

    max_experiments=None,
)
