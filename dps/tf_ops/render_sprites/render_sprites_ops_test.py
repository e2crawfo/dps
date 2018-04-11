import tensorflow as tf
import numpy as np

from dps.tf_ops.render_sprites import render_sprites

from dps.datasets.load import load_backgrounds
from dps.datasets.base import EmnistDataset



def test_render_sprites(self):
    draw_shape = (56, 56)

    backgrounds = load_backgrounds("red_x blue_circle", draw_shape)
    dset = EMNIST_ObjectDetection(classes=[0, 1, 2, 3], include_blank=False, n_examples=100)

    first0 = dset.x[dset.y.index(0)]
    first1 = dset.x[dset.y.index(1)]
    first2 = dset.x[dset.y.index(2)]
    first3 = dset.x[dset.y.index(3)]

    sprites0 = np.stack([first0, first1, first2, first3], axis=0)
    sprites1 = np.stack([first3, first2, first1], axis=0)
    sprites = np.stack([sprites0, sprites1], axis=0)

    batch_size = 2
    n_sprites = [4, 3]
    max_sprites = max(n_sprites)

    scales = 0.5 * np.ones((batch_size, max_sprites, 2))
    offsets = np.array([[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]], dtype=np.float32)
    offsets = np.tile(offset[None, ...], axis=(0, 1, 1))

    images = render_sprites(sprites, n_sprites, scales, offsets, backgrounds)






