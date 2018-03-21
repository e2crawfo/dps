import tensorflow as tf

from dps import cfg
from dps.config import Config
from dps.env.supervised import ClassificationEnv
from dps.vision import EMNIST_CONFIG
from dps.datasets import VisualArithmeticDataset
from dps.utils.tf import LeNet, ScopedFunction
from dps.updater import DifferentiableUpdater


def sl_build_env():
    largest_digit = max(int(i) for i in cfg.digits)
    kwargs = dict(one_hot=True, min_digits=1, max_digits=1, largest_digit=largest_digit)
    train = VisualArithmeticDataset(n_examples=cfg.n_train, example_range=(0.0, 0.9), **kwargs)
    val = VisualArithmeticDataset(n_examples=cfg.n_val, example_range=(0.9, 0.95), **kwargs)
    test = VisualArithmeticDataset(n_examples=cfg.n_val, example_range=(0.95, 1.0), **kwargs)

    return ClassificationEnv(train, val, test, one_hot=True)


def sl_get_updater(env):
    f = AttentionClassifier(pretrain_classifier=cfg.pretrain_classifier, fix_classifier=cfg.fix_classifier)
    return DifferentiableUpdater(env, f)


def per_minibatch_conv2d(inp, F, padding="VALID"):
    """ inp has shape (MB, H, W, in_channels)
        F has shape (MB, FH, FW, in_channels, out_channels)

    """
    MB = tf.shape(inp)[0]
    _, H, W, in_channels = inp.shape
    _, FH, FW, _, out_channels = F.shape

    inp_r = tf.transpose(inp, [1, 2, 0, 3])
    inp_r = tf.reshape(inp, [1, H, W, MB*in_channels])

    F = tf.transpose(F, [1, 2, 0, 3, 4])
    F = tf.reshape(F, [FH, FW, MB*in_channels, out_channels])

    out = tf.nn.depthwise_conv2d(
        inp_r,
        filter=F,
        strides=[1, 1, 1, 1],
        padding=padding)

    _, out_H, out_W, _ = out.shape

    out = tf.reshape(out, [out_H, out_W, MB, in_channels, out_channels])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_sum(out, axis=3)
    return out


class Attention(object):
    def __call__(self, images, output_shape, is_training):
        if len(images.shape) == 3:
            images = images[..., None]

        image_shape = tuple(int(i) for i in images.shape[1:-1])
        vf = image_shape[0] - output_shape[0] + 1
        hf = image_shape[1] - output_shape[1] + 1

        activation_network = LeNet(n_units=128)
        activations = activation_network(images, vf*hf, is_training)
        weights = tf.nn.softmax(activations)
        filt = tf.reshape(weights, (-1, vf, hf, 1, 1))

        return per_minibatch_conv2d(images, filt, padding="VALID")


class AttentionClassifier(ScopedFunction):
    def __init__(self, pretrain_classifier=True, fix_classifier=True, scope=None):
        self.pretrain_classifier = pretrain_classifier
        self.fix_classifier = fix_classifier
        super(AttentionClassifier, self).__init__(scope)

    def _call(self, images, output_size, is_training):
        attention = Attention()
        attended = attention(images, cfg.sub_image_shape, is_training)

        config = cfg.emnist_config.copy(shape=cfg.sub_image_shape)
        classifier = cfg.emnist_config.build_function()

        if self.pretrain_classifier:
            classifier.set_pretraining_params(
                config, name_params='classes include_blank shape n_controller_units',
                directory=cfg.model_dir + '/emnist_pretrained'
            )

        if self.fix_classifier:
            classifier.fix_variables()

        out = classifier(attended, output_size, is_training)
        return out


config = Config(
    name="ATTENTION",
    log_name='translated_mnist',
    build_env=sl_build_env,
    get_updater=sl_get_updater,

    fix_classifier=True,
    pretrain_classifier=True,

    curriculum=[dict()],
    reductions="sum",
    threshold=0.04,
    T=30,

    image_shape=(42, 42),
    draw_offset=(0, 0),
    draw_shape=None,
    sub_image_shape=(14, 14),

    n_train=10000,
    n_val=100,
    stopping_criteria="01_loss,min",

    emnist_config=EMNIST_CONFIG.copy(
        build_function=lambda: LeNet(128, scope="digit_classifier")
    ),

    cpu_ram_limit_mb=12*1024,
    use_gpu=True,
    gpu_allow_growth=True,
    per_process_gpu_memory_fraction=0.22,

    optimizer_spec="adam",
    lr_schedule=1e-4,
    power_through=True,
    noise_schedule=0.0,
    max_grad_norm=None,
    l2_weight=0.0,

    batch_size=64,
    n_controller_units=128,
    patience=5000,
    preserve_policy=True,
)
