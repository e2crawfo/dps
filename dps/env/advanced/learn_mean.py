import tensorflow as tf
import numpy as np
import os

from dps import cfg
from dps.updater import DifferentiableUpdater
from dps.env.supervised import BernoulliSigmoid
from dps.datasets import EMNIST_ObjectDetection, AutoencodeDataset
from dps.utils import Config, Param
from dps.utils.tf import ScopedFunction


class SimpleDecoder(ScopedFunction):
    def __init__(self, initializer=None, **kwargs):
        super(SimpleDecoder, self).__init__(**kwargs)
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


def build_env():
    _train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train)).x
    train = AutoencodeDataset(_train, image=True)

    _val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val)).x
    val = AutoencodeDataset(_val, image=True)

    _test = EMNIST_ObjectDetection(n_examples=int(cfg.n_val)).x
    test = AutoencodeDataset(_test, image=True)

    return LearnMean(train, val, test)


def get_updater(env):
    model = SimpleDecoder()
    return DifferentiableUpdater(env, model)


class LearnMean(BernoulliSigmoid):
    xent_loss = Param()

    def __init__(self, train, val, test=None, **kwargs):
        assert isinstance(train, AutoencodeDataset)
        assert isinstance(val, AutoencodeDataset)
        if test:
            assert isinstance(test, AutoencodeDataset)
        super(LearnMean, self).__init__(train, val, test, **kwargs)

    def make_feed_dict(self, batch_size, mode, evaluate):
        x = self.datasets[mode].next_batch(batch_size=batch_size, advance=not evaluate)
        return {self.x: x, self.is_training: not evaluate}

    def _build_placeholders(self):
        self.x = tf.placeholder(tf.float32, (None,) + self.obs_shape, name="x")
        self.is_training = tf.placeholder(tf.bool, (), name="is_training")

    def _build(self):
        self.logits = 0.01 * self.prediction
        self.sigmoids = tf.nn.sigmoid(self.logits)

        recorded_tensors = {
            name: tf.reduce_mean(getattr(self, 'build_' + name)(self.logits, self.x))
            for name in ['xent_loss', '2norm_loss', '1norm_loss']
        }

        loss_key = 'xent_loss' if self.xent_loss else '2norm_loss'
        recorded_tensors['loss'] = recorded_tensors[loss_key]
        return recorded_tensors

    def build_xent_loss(self, logits, targets):
        batch_size = tf.shape(self.x)[0]
        targets = tf.reshape(targets, (batch_size, -1))
        logits = tf.reshape(logits, (batch_size, -1))
        return tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits),
            keep_dims=True, axis=1
        )

    def build_2norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)

        batch_size = tf.shape(self.x)[0]
        targets = tf.reshape(targets, (batch_size, -1))
        actions = tf.reshape(actions, (batch_size, -1))

        return tf.reduce_mean((actions - targets)**2, keep_dims=True, axis=1)

    def build_1norm_loss(self, logits, targets):
        actions = tf.sigmoid(logits)

        batch_size = tf.shape(self.x)[0]
        targets = tf.reshape(targets, (batch_size, -1))
        actions = tf.reshape(actions, (batch_size, -1))

        return tf.reduce_mean(tf.abs(actions - targets), keep_dims=True, axis=1)


def mnist_vae_render_hook(updater):
    # Run the network on a subset of the evaluation data, fetch the output
    N = 16

    env = updater.env
    feed_dict = env.make_feed_dict(N, 'val', True)
    images = feed_dict[env.x]

    sess = tf.get_default_session()
    sigmoids = sess.run(env.sigmoids, feed_dict=feed_dict)

    sqrt_N = int(np.ceil(np.sqrt(N)))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2*sqrt_N, sqrt_N, figsize=(20, 20))
    axes = np.array(axes).reshape(2*sqrt_N, sqrt_N)
    for n, (pred, gt) in enumerate(zip(sigmoids, images)):
        i = int(n / sqrt_N)
        j = int(n % sqrt_N)

        ax1 = axes[2*i, j]
        ax1.imshow(pred)
        ax1.set_title('prediction')

        ax2 = axes[2*i+1, j]
        ax2.imshow(gt)
        ax2.set_title('ground_truth')

    fig.suptitle('After {} experiences ({} updates, {} experiences per batch).'.format(
        updater.n_experiences, updater.n_updates, cfg.batch_size))

    # import matplotlib.pyplot as plt
    # import numpy as np
    # _train = _train.astype('f') / 255.
    # n_plots = 5
    # plt.subplot(n_plots, 1, 1)
    # plt.imshow(np.mean(_train[0:2, ...], axis=0))
    # plt.subplot(n_plots, 1, 2)
    # plt.imshow(_train[0])
    # plt.subplot(n_plots, 1, 3)
    # plt.imshow(_train[1])
    # plt.subplot(n_plots, 1, 4)
    # plt.imshow(np.sum(_train[0:2, ...], axis=0))
    # plt.subplot(n_plots, 1, 5)
    # plt.imshow(np.mean(_train, axis=0))
    # plt.show()

    fig.savefig(os.path.join(cfg.path, 'plots', 'reconstruction.pdf'))
    plt.close(fig)


config = Config(
    log_name="learn_mean",
    build_env=build_env,
    get_updater=get_updater,
    min_chars=1,
    max_chars=1,
    characters=[0],
    sub_image_shape=(28, 28),
    xent_loss=True,

    render_hook=mnist_vae_render_hook,
    render_step=500,

    image_shape=(28, 28),
    # image_shape=(40, 40),

    n_train=1e5,
    n_val=1e2,
    n_test=1e2,

    # curriculum=[dict(lr_schedule=lr) for lr in [1e-1]],
    curriculum=[dict(lr_schedule=lr) for lr in [1e-4, 1e-5, 1e-6]],
    preserve_env=True,

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