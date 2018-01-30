import tensorflow as tf

from dps import cfg
from dps.utils import Config
from dps.utils.tf import ScopedFunction
from dps.datasets import AtariAutoencodeDataset
from dps.env.supervised import BernoulliSigmoid
from dps.updater import DifferentiableUpdater


def build_env():
    train = AtariAutoencodeDataset(n_examples=cfg.n_train, policy=cfg.train_policy)
    val = AtariAutoencodeDataset(n_examples=cfg.n_val, policy=cfg.val_policy)
    test = AtariAutoencodeDataset(n_examples=cfg.n_val, policy=cfg.test_policy)
    return BernoulliSigmoid(train, val, test)


class AtariNet(ScopedFunction):
    """ The no-action feedforward network from the "Action-Conditional" paper.

    Predicts frames based on a stack of other frames.

    """
    def __init__(self, **kwargs):
        super(AtariNet, self).__init__(**kwargs)

    def _call(self, inp, output_size, is_training):
        # output_size, is_training are ignored

        volume = inp
        conv2d = tf.layers.conv2d
        conv2d_transpose = tf.layers.conv2d_transpose

        # volume.shape = (*, 210, 160, 3 * n_input_frames)

        volume = conv2d(volume, filters=64, kernel_size=8, strides=2, padding="valid")
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 102, 77, 64)

        volume = conv2d(volume, filters=128, kernel_size=6, strides=2, padding="valid")
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 49, 36, 128)

        volume = conv2d(volume, filters=128, kernel_size=6, strides=2, padding="valid")
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 22, 16, 128)

        volume = conv2d(volume, filters=128, kernel_size=4, strides=2, padding="valid")
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 10, 7, 128)

        volume = conv2d(volume, filters=2048, kernel_size=(10, 7), padding="valid")  # fully connected layer
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 1, 1, 2048)

        volume = conv2d(volume, filters=2048, kernel_size=(1, 1), padding="valid")  # fully connected layer
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 1, 1, 2048)

        volume = conv2d_transpose(volume, filters=128, kernel_size=(10, 7), padding="valid")  # fully connected layer
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 10, 7, 128)

        volume = conv2d_transpose(volume, filters=128, kernel_size=4, strides=2, padding="valid")
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 22, 16, 128)

        volume = conv2d_transpose(volume, filters=128, kernel_size=6, strides=2, padding="valid")
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 49, 36, 128)

        volume = conv2d_transpose(volume, filters=128, kernel_size=6, strides=2, padding="valid")
        volume = tf.nn.relu(volume)

        # volume.shape = (*, 102, 77, 128)

        volume = conv2d_transpose(volume, filters=3, kernel_size=8, strides=2, padding="valid")

        # volume.shape = (*, 210, 160, 3)

        return volume


def build_model():
    return AtariNet()


def get_updater(env):
    model = cfg.build_model()
    return DifferentiableUpdater(env, model)


config = Config(
    log_name='atari_autoencode',
    get_updater=get_updater,
    build_env=build_env,
    build_model=build_model,

    # Traing specific
    threshold=0.04,
    curriculum=[dict()],
    n_train=10000,
    n_val=100,
    use_gpu=True,
    batch_size=16,
    optimizer_spec="adam",
    opt_steps_per_update=1,
    lr_schedule="1e-4",

    atari_game="AsteroidsNoFrameskip-v4",
    train_policy=None,
    val_policy=None,
    test_policy=None,
)

feedforward_config = config
