import numpy as np
import tensorflow as tf


def DRAW_attention_2D(inp, fovea_x, fovea_y, delta, std, N, normalize=False):
    """
    Parameters
    ----------
    inp: Tensor (batch_size, B, A)
    fovea_x: Tensor (batch_size, 1)
    fovea_y: Tensor (batch_size, 1)
    delta: Tensor (batch_size, 1)
    std: Tensor (batch_size, 1)
    N: int
    normalize: bool
        Whether to normalize the filter before applying it.

    """
    B = int(inp.shape[1])
    A = int(inp.shape[2])

    fovea_x = (fovea_x + 1) * A/2
    fovea_y = (fovea_y + 1) * B/2

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
