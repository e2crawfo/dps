import tensorflow as tf
import numpy as np
import os

from dps import cfg
from dps.updater import DifferentiableUpdater
from dps.env.supervised import BernoulliSigmoid
from dps.datasets import EMNIST_ObjectDetection
from dps.utils import Config, Param
from dps.utils.tf import FullyConvolutional, ScopedFunction, tf_mean_sum


tf_flatten = tf.layers.flatten


def build_env():
    train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train), example_range=(0., 0.9))
    val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val), example_range=(0.9, 0.95))
    test = EMNIST_ObjectDetection(n_examples=int(cfg.n_val), example_range=(0.95, 1.))

    return VQVAE_Env(train, val, test)


def get_updater(env):
    model = VQVAE()
    return DifferentiableUpdater(env, model)


class Encoder(FullyConvolutional):
    def __init__(self):
        layout = [
            dict(filters=128, kernel_size=4, strides=2, padding="SAME"),
            dict(filters=128, kernel_size=4, strides=2, padding="SAME"),
            dict(filters=128, kernel_size=7, strides=7, padding="SAME"),
        ]
        super(Encoder, self).__init__(layout, check_output_shape=True)


class Decoder(FullyConvolutional):
    def __init__(self):
        layout = [
            dict(filters=128, kernel_size=7, strides=7, padding="SAME", transpose=True),
            dict(filters=128, kernel_size=4, strides=2, padding="SAME", transpose=True),
            dict(filters=3, kernel_size=4, strides=2, padding="SAME", transpose=True),
        ]
        super(Decoder, self).__init__(layout, check_output_shape=True)


class VQVAE_Env(BernoulliSigmoid):
    xent_loss = Param()
    beta = Param()

    def __init__(self, train, val, test=None, **kwargs):
        self.obs_shape = train.x[0].shape
        self.action_shape = self.obs_shape

        super(VQVAE_Env, self).__init__(train, val, test, **kwargs)

    def make_feed_dict(self, batch_size, mode, evaluate):
        x, *_ = self.datasets[mode].next_batch(batch_size=batch_size, advance=not evaluate)
        return {self.x: x, self.is_training: not evaluate}

    def _build_placeholders(self):
        self.x = tf.placeholder(tf.float32, (None,) + self.obs_shape, name="x")
        self.is_training = tf.placeholder(tf.bool, (), name="is_training")

    def _build(self):
        self.logits = self.prediction
        self.sigmoids = tf.nn.sigmoid(self.logits)

        recorded_tensors = {
            name: tf.reduce_mean(getattr(self, 'build_' + name)(self.prediction, self.x))
            for name in ['xent_loss', 'squared_loss', '1norm_loss']
        }

        # In principle, these two loss terms could be combined into a single term.
        # The main point of separating them out is so that the two terms
        # can be weighted differently. In their formulation, the embedding loss
        # is always given a weight of 1, while the weight on the commitment loss is
        # given by a hyper-parameter beta.

        f = self.f
        commitment_error = (f.z_e - tf.stop_gradient(f.z_q))**2
        recorded_tensors['commitment_error'] = tf_mean_sum(commitment_error)

        embedding_error = (tf.stop_gradient(f.z_e) - f.z_q)**2
        recorded_tensors['embedding_error'] = tf_mean_sum(embedding_error)

        loss_key = 'xent_loss' if self.xent_loss else 'squared_loss'

        recorded_tensors['loss'] = recorded_tensors[loss_key]
        recorded_tensors['loss'] += recorded_tensors['embedding_error']
        recorded_tensors['loss'] += self.beta * recorded_tensors['commitment_error']

        return recorded_tensors

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


class VQVAE(ScopedFunction):
    H = Param()
    W = Param()
    K = Param()
    D = Param()

    common_embedding = Param(help="If True, all latent variables share a common set of embedding vectors.")

    _encoder = None
    _decoder = None
    _embedding = None

    def __call__(self, inp, output_size, is_training):
        if self._encoder is None:
            self._encoder = cfg.build_encoder()
            self._encoder.layout[-1]["filters"] = self.D

        if self._decoder is None:
            self._decoder = cfg.build_decoder()

        if self._embedding is None:
            initializer = tf.truncated_normal_initializer(stddev=0.1)
            shape = (self.K, self.D)
            if not self.common_embedding:
                shape = (self.H, self.W,) + shape
            self._embedding = tf.get_variable("embedding", shape, initializer=initializer)

        self.z_e = self._encoder(inp, (self.H, self.W, self.D), is_training)

        if self.common_embedding:
            # self._embedding has shape (K, D), i.e. same dictionary used for all latents
            embedding = self._embedding[None, None, None, ...]
        elif len(self._embedding.shape) == 4:
            # self._embedding has shape (H, W, K, D), i.e. different dictionary for each latent
            embedding = self._embedding[None, ...]
        # shape of embedding should now be (1, H, W, K, D) either way

        z_e = self.z_e[..., None, :]  # (batch, H, W, 1, D)
        sum_squared_error = tf.reduce_sum((z_e - embedding) ** 2, axis=-1)
        self.k = k = tf.argmin(sum_squared_error, axis=-1)  # (batch, H, W)
        one_hot_k = tf.stop_gradient(tf.one_hot(k, self.K)[..., None])
        self.z_q = tf.reduce_sum(self._embedding[None, ...] * one_hot_k, axis=3)  # (batch, H, W, D)

        # On the forward pass z_q gets sent through, but the gradient gets sent back to z_e
        decoder_input = tf.stop_gradient(self.z_q - self.z_e) + self.z_e

        return self._decoder(decoder_input, output_size, is_training)


def vq_vae_render_hook(updater):
    # Run the network on a subset of the evaluation data, fetch the output
    N = 16

    env = updater.env
    feed_dict = env.make_feed_dict(N, 'val', True)
    images = feed_dict[env.x]

    sess = tf.get_default_session()
    k, sigmoids = sess.run([env.f.k, env.sigmoids], feed_dict=feed_dict)

    sqrt_N = int(np.ceil(np.sqrt(N)))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2*sqrt_N, sqrt_N, figsize=(20, 20))
    axes = np.array(axes).reshape(2*sqrt_N, sqrt_N)
    for n, (_k, pred, gt) in enumerate(zip(k, sigmoids, images)):
        i = int(n / sqrt_N)
        j = int(n % sqrt_N)

        ax1 = axes[2*i, j]
        ax1.imshow(pred)
        if _k.shape == (1, 1):
            ax1.set_title('prediction. class = {}'.format(_k[0, 0]))
        else:
            ax1.set_title('prediction.')

        ax2 = axes[2*i+1, j]
        ax2.imshow(gt)
        ax2.set_title('ground_truth')

    fig.suptitle('After {} experiences ({} updates, {} experiences per batch).'.format(
        updater.n_experiences, updater.n_updates, cfg.batch_size))

    fig.savefig(os.path.join(cfg.path, 'plots', 'reconstruction.pdf'))
    plt.close(fig)


config = Config(
    log_name="mnist_vqvae",
    build_env=build_env,
    get_updater=get_updater,
    min_chars=1,
    max_chars=1,
    characters=list(range(10)),
    sub_image_shape=(28, 28),
    build_encoder=Encoder,
    build_decoder=Decoder,
    xent_loss=True,

    render_hook=vq_vae_render_hook,
    render_step=5000,

    beta=1.0,
    common_embedding=False,

    image_shape=(28, 28),
    H=1,
    W=1,
    K=10,
    D=100,

    n_train=1e5,
    n_val=1e2,
    n_test=1e2,

    curriculum=[dict()],
    lr_schedule=1e-4,
    preserve_env=True,

    # training params
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
    colours="red",
)
