import numpy as np
import tensorflow as tf


def discrete_attention(inp, fovea_x, fovea_y, delta, N):
    top_left_x = (fovea_x - delta + 1) / 2.0
    top_left_y = (fovea_y - delta + 1) / 2.0

    bottom_right_x = (fovea_x + delta + 1) / 2.0
    bottom_right_y = (fovea_y + delta + 1) / 2.0

    boxes = tf.concat([top_left_y, top_left_x, bottom_right_y, bottom_right_x], axis=1)

    inp = tf.expand_dims(inp, 3)

    result = tf.image.crop_and_resize(
        image=inp,
        boxes=boxes,
        box_ind=tf.range(tf.shape(inp)[0]),
        crop_size=(N, N))
    result = tf.squeeze(result, 3)
    return result


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


class DRAW(object):
    def __init__(self, N):
        self.N = N

    def __call__(self, inp, fovea_x=None, fovea_y=None, delta=None, sigma=None):
        reshape = len(inp.shape) == 2
        if reshape:
            s = int(np.sqrt(int(inp.shape[1])))
            inp = tf.reshape(inp, (-1, s, s))

        batch_size = tf.shape(inp)[0]
        if fovea_x is None:
            fovea_x = tf.zeros((batch_size, 1))
        if fovea_y is None:
            fovea_y = tf.zeros((batch_size, 1))
        if delta is None:
            delta = tf.ones((batch_size, 1))
        if sigma is None:
            sigma = tf.ones((batch_size, 1))

        glimpse = DRAW_attention_2D(
            inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, std=sigma, N=self.N)

        if reshape:
            glimpse = tf.reshape(glimpse, (-1, self.N*self.N))

        return glimpse


class DiscreteAttn(object):
    def __init__(self, N):
        self.N = N

    def __call__(self, inp, fovea_x=None, fovea_y=None, delta=None):
        reshape = len(inp.shape) == 2
        if reshape:
            s = int(np.sqrt(int(inp.shape[1])))
            inp = tf.reshape(inp, (-1, s, s))

        batch_size = tf.shape(inp)[0]
        if fovea_x is None:
            fovea_x = tf.zeros((batch_size, 1))
        if fovea_y is None:
            fovea_y = tf.zeros((batch_size, 1))
        if delta is None:
            delta = tf.ones((batch_size, 1))

        glimpse = discrete_attention(
            inp, fovea_x=fovea_x, fovea_y=fovea_y, delta=delta, N=self.N)

        if reshape:
            glimpse = tf.reshape(glimpse, (-1, self.N*self.N))

        return glimpse
