import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dps.experiments.mnist import TranslatedMnistDataset


def DRAW_attention_2D(inp, fovea, delta, std, N, normalize=False):
    """
    Parameters
    ----------
    inp: Tensor (batch_size, B, A)
    fovea: Tensor (batch_size, 2)
    delta: Tensor (batch_size, 1)
    std: Tensor (batch_size, 1)
    N: int
    normalize: bool
        Whether to normalize the filter before applying it.

    """
    A = int(inp.shape[1])
    B = int(inp.shape[2])

    fovea_x = (fovea[:, :1] + 1) * A/2
    fovea_y = (fovea[:, 1:] + 1) * B/2

    mu_x = fovea_x + np.linspace(-A/2, A/2, N).reshape(1, -1) * delta
    mu_y = fovea_y + np.linspace(-B/2, B/2, N).reshape(1, -1) * delta
    std = std + mu_x - mu_x

    loc_x = tf.constant(np.arange(A) + 0.5, dtype=tf.float32)
    loc_y = tf.constant(np.arange(B) + 0.5, dtype=tf.float32)

    X_filt = gaussian_filter(mu_x, std, loc_x, normalize=normalize)
    Y_filt = gaussian_filter(mu_y, std, loc_y, normalize=normalize)
    return tf.matmul(Y_filt, tf.matmul(inp, X_filt, adjoint_b=True))


def gaussian_filter(mu, std, locations, normalize=False):
    """ Make a 1-D Gaussian filter.

    Parameters
    ----------
    mu: Tensor (batch_size, n_filters)
    std: Tensor (batch_size, n_filters)
    locations: Tensor (n_points)

    Returns
    -------
    filt: Tensor (batch_size, n_filters, n_points)

    """
    mu = tf.expand_dims(mu, 2)
    std = tf.expand_dims(std, 2)
    locations = tf.reshape(locations, (1, 1, -1))
    filt = tf.exp(-0.5 * ((mu - locations)**2) / std)
    if normalize:
        filt /= tf.sqrt(2 * np.pi) * std
    return filt


def apply_gaussian_filter(mu, std, locations, values, normalize=False):
    """ Create and apply a single 1-D gaussian filter. """
    filt = gaussian_filter(mu, std, locations, normalize)
    filt = tf.squeeze(filt, axis=[1])
    vision = tf.reduce_sum(values * filt, axis=-1, keep_dims=True)
    return vision


def test_mnist():
    import pickle
    import gzip
    params = tf.constant([
        [0.0, 0.0, 1.0, 1, 1],
        [-1.0, 0.0, 1.0, 1, 1],
        [0.0, 0.0, 1.0, 1, 1],
        [1.0, 0.0, 1.0, 1, 1],

        [0, 0, 0.1, 0.1, 1],
        [0, 0, 0.1, 0.2, 1],
        [0, 0, 0.1, 0.3, 1],
        [0, 0, 0.1, 0.4, 1],

        [0, 0, 0.1, 0.5, 1],
        [0, 0, 0.5, 0.5, 1],
        [0, 0, 1.0, 0.5, 1],
        [0, 0, 2.0, 0.5, 1],

        [0, 0, 0.1, 0.1, 1],
        [0, 0, 0.1, 0.2, 1],
        [0, 0, 0.1, 0.3, 1],
        [0, 0, 0.1, 0.4, 1],
    ], dtype=tf.float32)
    batch_size = int(params.shape[0])

    z = gzip.open('/data/mnist.pkl.gz', 'rb')
    (train, _), (dev, _), (test, _) = pickle.load(z, encoding='bytes')
    train = train[:batch_size, :]
    W = int(np.sqrt(train[0].shape[0]))
    train = tf.constant(train.reshape(-1, W, W))

    N = 20
    tf_attended_images = DRAW_attention_2D(train, params[:, 0:2], params[:, 2:3], params[:, 3:4], N, normalize=0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    attended_images = sess.run(tf_attended_images)

    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.hstack(np.hstack(attended_images[:batch_size].reshape((4,4,N,N))/255.)), cmap='gray')
    plt.show()


def show_parameter_effect():
    from itertools import product
    n_digits = 1
    max_overlap = 200
    W = 100
    n_images = 2
    mnist = TranslatedMnistDataset(W, n_digits, max_overlap, n_images, for_eval=True)

    images, _ = mnist.next_batch(2)
    images = np.reshape(images, (-1, W, W))

    N = 28

    gx = 0.0
    gy = 0.0

    log_sigma = np.linspace(-3, 3, 11)
    log_delta = np.linspace(-3, 3, 11)
    cols = ["log D = {0:.2f}".format(float(d)) for d in log_delta]
    rows = ["log S = {0:.2f}".format(float(s)) for s in log_sigma]
    S, D = log_sigma.size, log_delta.size
    n_settings = log_sigma.size * log_delta.size
    params = []
    for s, d in product(log_sigma, log_delta):
        params.append([gx, gy, np.exp(d), np.exp(s)])
    params = np.array(params).astype('f')
    params = np.tile(params, (n_images, 1))
    params = tf.constant(params, dtype=tf.float32)

    images = np.repeat(images, n_settings, axis=0)
    images = tf.constant(images, dtype=tf.float32)

    tf_attended_images = DRAW_attention_2D(images, params[:, 0:2], params[:, 2:3], params[:, 3:4], N, normalize=0)
    sess = tf.Session()
    glimpses = sess.run(tf_attended_images)

    for i in range(n_images):
        fig, axes = plt.subplots(nrows=S, ncols=D, figsize=(10, 10))

        for j in range(S):
            for k in range(D):
                plt.subplot(S, D, j*D + k + 1)
                if j == 0:
                    plt.title(cols[k])
                if k == 0:
                    plt.ylabel(rows[j])
                plt.imshow(glimpses[i*S*D + j*D + k, :, :], cmap='gray')

    plt.show()


if __name__ == "__main__":
    show_parameter_effect()
    # test_mnist()
