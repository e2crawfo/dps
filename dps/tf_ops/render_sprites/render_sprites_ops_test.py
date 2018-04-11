import tensorflow as tf
import numpy as np

from dps.tf_ops.render_sprites import render_sprites

from dps.datasets.load import load_backgrounds
from dps.datasets.base import EmnistDataset
from dps.utils import image_to_string


def get_session():
    return tf.Session(config=tf.ConfigProto(log_device_placement=True))


def test_render_sprites_mostly_opaque():
    draw_shape = (56, 56)

    backgrounds = np.array(load_backgrounds("red_x blue_circle", draw_shape)) / 255.
    backgrounds = backgrounds.astype('f')
    dset = EmnistDataset(classes=[0, 1, 2, 3], include_blank=False, n_examples=100, shape=(28, 28), one_hot=False)

    colour = np.array([1., 1., 1.])[None, None, :]
    sprite_pool = [colour * dset.x[list(dset.y).index(idx)][..., None] / 255. for idx in range(4)]
    sprite_pool = [np.concatenate([sp, np.ones_like(sp)[..., :1]], axis=2) for sp in sprite_pool]

    first0, first1, first2, first3 = sprite_pool

    sprites0 = np.stack([first0, first1, first2, first3], axis=0)
    sprites1 = np.stack([first3, first2, first1, np.zeros_like(first1)], axis=0)
    sprites = np.stack([sprites0, sprites1], axis=0).astype('f')

    batch_size = 2
    n_sprites = [4, 3]
    max_sprites = max(n_sprites)

    scales = 0.5 * np.ones((batch_size, max_sprites, 2)).astype('f')
    offsets = np.array([[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]])
    offsets = np.tile(offsets[None, ...], (batch_size, 1, 1)).astype('f')

    images = render_sprites(sprites, n_sprites, scales, offsets, backgrounds)

    sess = get_session()
    result = sess.run(images)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(result[0])
    ax2.imshow(result[1])
    plt.show()


def test_render_sprites_background_alpha():
    draw_shape = (56, 56)

    backgrounds = np.array(load_backgrounds("red_x blue_circle", draw_shape)) / 255.
    backgrounds = backgrounds.astype('f')
    dset = EmnistDataset(classes=[0, 1, 2, 3], include_blank=False, n_examples=100, shape=(28, 28), one_hot=False)

    colour = np.array([1., 1., 1.])[None, None, :]
    sprite_pool = [colour * dset.x[list(dset.y).index(idx)][..., None] / 255. for idx in range(4)]
    _sprite_pool = []
    for sp in sprite_pool:
        alpha = (sp.sum(-1) > 0)[..., None].astype('f')
        sp = np.concatenate([sp, alpha], axis=-1)
        _sprite_pool.append(sp)
    sprite_pool = _sprite_pool

    first0, first1, first2, first3 = sprite_pool

    sprites0 = np.stack([first0, first1, first2, first3], axis=0)
    sprites1 = np.stack([first3, first2, first1, np.zeros_like(first1)], axis=0)
    sprites = np.stack([sprites0, sprites1], axis=0).astype('f')

    batch_size = 2
    n_sprites = [4, 3]
    max_sprites = max(n_sprites)

    scales = 0.5 * np.ones((batch_size, max_sprites, 2)).astype('f')
    offsets = np.array([[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]])
    offsets = np.tile(offsets[None, ...], (batch_size, 1, 1)).astype('f')

    images = render_sprites(sprites, n_sprites, scales, offsets, backgrounds)

    sess = get_session()
    result = sess.run(images)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(result[0])
    ax2.imshow(result[1])
    plt.show()


def test_render_sprites_overlap():
    draw_shape = (56, 56)

    backgrounds = np.array(load_backgrounds("red_x blue_circle", draw_shape)) / 255.
    backgrounds = backgrounds.astype('f')
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

        alpha = 0.98 * (sp.sum(-1) > 0)[..., None].astype('f')
        print(image_to_string(sp))
        # alpha = (sp.sum(-1) > 0)[..., None].astype('f')

        sp = colour * sp
        sp = np.concatenate([sp, alpha], axis=-1)
        _sprite_pool.append(sp)

    sprite_pool = _sprite_pool

    first0, first1, first2, first3 = sprite_pool

    sprites0 = np.stack([first0, first1, first2, first3], axis=0)
    sprites1 = np.stack([first3, first2, first1, np.zeros_like(first1)], axis=0)
    sprites = np.stack([sprites0, sprites1], axis=0).astype('f')

    batch_size = 2
    n_sprites = [4, 3]
    max_sprites = max(n_sprites)

    scales = np.ones((batch_size, max_sprites, 2)).astype('f')
    offsets = np.zeros_like(scales)

    images = render_sprites(sprites, n_sprites, scales, offsets, backgrounds)

    sess = get_session()
    result = sess.run(images)

    result = np.clip(result, 1e-6, 1-1e-6)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(result[0])
    ax2.imshow(result[1])
    plt.show()


def test_gradient():
    from tensorflow.python.framework import constant_op

    draw_shape = (56, 56)

    backgrounds = np.array(load_backgrounds("red_x blue_circle", draw_shape)) / 255.
    backgrounds = backgrounds.astype('f')
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

        alpha = 0.98 * (sp.sum(-1) > 0)[..., None].astype('f')

        sp = colour * sp
        sp = np.concatenate([sp, alpha], axis=-1)
        _sprite_pool.append(sp)

    sprite_pool = _sprite_pool

    first0, first1, first2, first3 = sprite_pool

    sprites0 = np.stack([first0, first1, first2, first3], axis=0)
    sprites1 = np.stack([first3, first2, first1, np.zeros_like(first1)], axis=0)
    sprites = np.stack([sprites0, sprites1], axis=0).astype('f')

    batch_size = 2
    n_sprites = [4, 3]
    max_sprites = max(n_sprites)

    scales = np.ones((batch_size, max_sprites, 2)).astype('f')
    offsets = np.zeros_like(scales)

    sprites = np.array(sprites)
    n_sprites = np.array(n_sprites, dtype=np.int32)
    scales = np.array(scales)
    offsets = np.array(offsets)
    backgrounds = np.array(backgrounds)

    sprites_tf = constant_op.constant(sprites)
    n_sprites_tf = constant_op.constant(n_sprites)
    scales_tf = constant_op.constant(scales)
    offsets_tf = constant_op.constant(offsets)
    backgrounds_tf = constant_op.constant(backgrounds)

    images = render_sprites(sprites_tf, n_sprites_tf, scales_tf, offsets_tf, backgrounds_tf)

    from tensorflow.python.ops import gradient_checker

    sess = get_session()
    with sess.as_default():
        tf_err = gradient_checker.compute_gradient_error([sprites_tf, n_sprites_tf, scales_tf, offsets_tf, backgrounds_tf],
                                                         [sprites.shape, n_sprites.shape, scales.shape, offsets.shape, backgrounds.shape],
                                                         images,
                                                         backgrounds.shape,
                                                         [sprites, n_sprites, scales, offsets, backgrounds],
                                                         delta=0.002)

    import pdb; pdb.set_trace()


    print(err)
    eps = 2e-4



if __name__ == "__main__":
    from dps.utils import NumpySeed
    with NumpySeed(100):
        with tf.device('/cpu:0'):
        # with tf.device('/gpu:0'):
            # test_render_sprites()
            # test_render_sprites_background_alpha()
            # test_render_sprites_overlap()
            test_gradient()
            # ss = get_session()
            # x = [1, 2, 3]
            # y = tf.maximum(x, 1)
            # print(ss.run(y))
