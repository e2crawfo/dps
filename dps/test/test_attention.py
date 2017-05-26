import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dps.mnist import TranslatedMnistDataset
from dps.attention import DRAW_attention_2D, discrete_attention


def test_draw_mnist(display):
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
    tf_attended_images = DRAW_attention_2D(
        train, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4], N, normalize=0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    attended_images = sess.run(tf_attended_images)

    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.hstack(np.hstack(attended_images[:batch_size].reshape((4, 4, N, N))/255.)), cmap='gray')
    if display:
        plt.show()


def test_draw_parameter_effect(display):
    from itertools import product
    n_digits = 1
    max_overlap = 200
    W = 28
    n_images = 2
    mnist = TranslatedMnistDataset(W, n_digits, max_overlap, n_images, for_eval=True)

    images, _ = mnist.next_batch(2)
    images = np.reshape(images, (-1, W, W))

    N = 14

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

    tf_attended_images = DRAW_attention_2D(
        images, params[:, :1], params[:, 1:2], params[:, 2:3], params[:, 3:4], N, normalize=0)
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

    if display:
        plt.show()


def test_discrete_mnist(display):
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

    N = 10
    tf_attended_images = discrete_attention(train, params[:, 0:1], params[:, 1:2], params[:, 2:3], N)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    attended_images = sess.run(tf_attended_images)

    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.hstack(np.hstack(attended_images[:batch_size].reshape((4, 4, N, N))/255.)), cmap='gray')
    if display:
        plt.show()


if __name__ == "__main__":
    test_discrete_mnist(1)
