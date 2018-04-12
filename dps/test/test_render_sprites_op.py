import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker
import numpy as np
import pytest
import os

from dps.datasets.load import load_backgrounds
from dps.datasets.base import EmnistDataset
from dps.utils import NumpySeed
from dps.tf_ops import render_sprites


def get_session():
    return tf.Session(config=tf.ConfigProto(log_device_placement=True))


def squash_01(val, squash_factor):
    assert ((0 <= val) * (val <= 1)).all()
    val = np.array(val, dtype=np.float32)

    if squash_factor:
        assert squash_factor > 0
        return (val - 0.5) * squash_factor + 0.5
    else:
        return val


def get_data(random_alpha=False, squash=None):
    draw_shape = (56, 56)
    batch_size = 2
    n_sprites = [4, 3]
    max_sprites = max(n_sprites)

    dset = EmnistDataset(classes=[0, 1, 2, 3], include_blank=False, n_examples=100, shape=(28, 28), one_hot=False)

    white = np.array([1., 1., 1.])[None, None, :]
    black = np.array([0., 0., 0.])[None, None, :]
    green = np.array([0., 1., 0.])[None, None, :]
    cyan = np.array([0., 1., 1.])[None, None, :]
    colours = [white, black, green, cyan]
    sprite_pool = [dset.x[list(dset.y).index(idx)][..., None] / 255. for idx in range(4)]
    _sprite_pool = []
    for i, sp in enumerate(sprite_pool):
        colour = colours[i]

        if random_alpha:
            alpha = np.random.rand(*sp[..., :1].shape)
        else:
            alpha = (sp.sum(-1) > 0)[..., None].astype('f')

        alpha = squash_01(alpha, squash)

        sp = colour * sp
        sp = np.concatenate([sp, alpha], axis=-1)
        _sprite_pool.append(sp)

    sprite_pool = _sprite_pool

    first0, first1, first2, first3 = sprite_pool
    sprites0 = np.stack([first0, first1, first2, first3], axis=0)
    sprites1 = np.stack([first3, first2, first1, np.zeros_like(first1)], axis=0)
    sprites = np.stack([sprites0, sprites1], axis=0).astype('f')

    scales = np.ones((batch_size, max_sprites, 2)).astype('f')
    offsets = np.zeros_like(scales)

    backgrounds = np.array(load_backgrounds("red_x blue_circle", draw_shape)) / 255.
    backgrounds = backgrounds.astype('f')

    sprites = squash_01(sprites, squash)
    n_sprites = np.array(n_sprites, dtype=np.int32)
    scales = squash_01(scales, squash)
    offsets = squash_01(offsets, squash)
    backgrounds = squash_01(backgrounds, squash)

    return sprites, n_sprites, scales, offsets, backgrounds


def run(device, show_plots, process_data=None, **get_data_kwargs):
    with NumpySeed(100):
        data = get_data(**get_data_kwargs)

        if process_data is None:
            process_data = lambda *x: x

        sprites, n_sprites, scales, offsets, backgrounds = process_data(*data)

        with tf.device('/{}:0'.format(device)):
            images = render_sprites.render_sprites(sprites, n_sprites, scales, offsets, backgrounds)
            sess = get_session()
            result = sess.run(images)

        result = np.clip(result, 1e-6, 1-1e-6)

    if show_plots:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(result[0])
        ax2.imshow(result[1])
        plt.show()


def visible_gpu():
    d = os.getenv("CUDA_VISIBLE_DEVICES").split(",")[0]
    try:
        d = int(d)
    except Exception:
        return False
    return d >= 0


@pytest.mark.skipif(not render_sprites.lib_avail(), reason="_render_sprites.so not available")
@pytest.mark.parametrize("device", "cpu gpu".split())
def test_render_sprites_mostly_opaque(device, show_plots):
    if device == "gpu" and visible_gpu():
        pytest.xfail("no gpu is visible")

    def process_data(sprites, n_sprites, scales, offsets, backgrounds):
        batch_size, max_sprites, *_ = sprites.shape
        sprites[..., 3] = 1.0  # Make the image opaque
        scales = 0.5 * np.ones((batch_size, max_sprites, 2)).astype('f')
        offsets = np.array([[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]])
        offsets = np.tile(offsets[None, ...], (batch_size, 1, 1)).astype('f')
        return sprites, n_sprites, scales, offsets, backgrounds

    run(device, show_plots, process_data)


@pytest.mark.skipif(not render_sprites.lib_avail(), reason="_render_sprites.so not available")
@pytest.mark.parametrize("device", "cpu gpu".split())
def test_render_sprites_background_alpha(device, show_plots):
    if device == "gpu" and visible_gpu():
        pytest.xfail("no gpu is visible")

    def process_data(sprites, n_sprites, scales, offsets, backgrounds):
        batch_size, max_sprites, *_ = sprites.shape
        scales = 0.5 * np.ones((batch_size, max_sprites, 2)).astype('f')
        offsets = np.array([[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]])
        offsets = np.tile(offsets[None, ...], (batch_size, 1, 1)).astype('f')
        return sprites, n_sprites, scales, offsets, backgrounds

    run(device, show_plots, process_data)


@pytest.mark.skipif(not render_sprites.lib_avail(), reason="_render_sprites.so not available")
@pytest.mark.parametrize("device", "cpu gpu".split())
def test_render_sprites_overlap(device, show_plots):
    if device == "gpu" and visible_gpu():
        pytest.xfail("no gpu is visible")
    run(device, show_plots)


@pytest.mark.skipif(not render_sprites.lib_avail(), reason="_render_sprites.so not available")
@pytest.mark.parametrize("device", "cpu gpu".split())
@pytest.mark.slow
def test_gradient(device):

    if device == "gpu" and visible_gpu():
        pytest.xfail("no gpu is visible")

    with NumpySeed(100):
        with tf.device('/{}:0'.format(device)):
            sprites, n_sprites, scales, offsets, backgrounds = get_data(random_alpha=True, squash=0.99)

            sprites_tf = constant_op.constant(sprites)
            n_sprites_tf = constant_op.constant(n_sprites)
            scales_tf = constant_op.constant(scales)
            offsets_tf = constant_op.constant(offsets)
            backgrounds_tf = constant_op.constant(backgrounds)

            images = render_sprites.render_sprites(sprites_tf, n_sprites_tf, scales_tf, offsets_tf, backgrounds_tf)

            sess = get_session()
            with sess.as_default():
                with tf.device(device):
                    err = gradient_checker.compute_gradient_error(
                        [sprites_tf, scales_tf, offsets_tf, backgrounds_tf],
                        [sprites.shape, scales.shape, offsets.shape, backgrounds.shape],
                        images,
                        backgrounds.shape,
                        [sprites, scales, offsets, backgrounds],
                        delta=0.002)

            print("Jacobian error: {}".format(err))
            threshold = 2e-4
            assert err < threshold, "Jacobian error ({}) exceeded threshold ({})".format(err, threshold)
