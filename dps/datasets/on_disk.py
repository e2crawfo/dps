import numpy as np
import skimage
import os

from dps.datasets.base import (
    ImageDataset, ImageFeature, VariableShapeArrayFeature, ArrayFeature, StringFeature
)
from dps.utils import Param, atleast_nd, walk_images


class ImagesOnDiskDataset(ImageDataset):
    image_files = Param()
    image_shape = None

    """
    Params inherited from ImageDataset, listed here for clarity.

    postprocessing = Param("")
    tile_shape = Param(None)
    n_samples_per_image = Param(1)
    n_frames = Param(0)
    image_dataset_version = Param(1)

    """
    _artifact_names = ['depth']
    depth = None

    @property
    def features(self):
        if self._features is None:
            annotation_shape = (self.n_frames, -1, 7) if self.n_frames > 0 else (-1, 7)
            self._features = [
                ImageFeature("image", self.obs_shape, dtype=np.uint8, strict=True),
                VariableShapeArrayFeature("annotations", annotation_shape),
                ArrayFeature("offset", (2,), dtype=np.int32),
                StringFeature("filename"),
            ]

        return self._features

    def _read_image(self, path):
        return skimage.io.imread(path)

    def _process_image(self, image):
        return image

    def _make(self):
        image_files = self.image_files
        if isinstance(image_files, str):
            image_files = image_files.split()

        paths = []
        for f in image_files:
            if os.path.isdir(f):
                paths.extend(walk_images(f, concat=True))
            else:
                paths.append(f)

        for p in paths:
            image = self._read_image(p)
            image = self._process_image(image)

            image = atleast_nd(image, 3)
            if self.depth is None:
                self.depth = image.shape[2]

            self._write_example(
                image=image,
                annotations=[],
                filename=p,
            )

        return dict(depth=self.depth)


if __name__ == "__main__":
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from skimage.morphology import opening, disk

    # image_file = os.path.join(skimage.data.data_dir, 'astronaut.png')
    image_file = '/media/data/dps_data/data/hubble_extreme_deep_field_example.tif'
    image = skimage.io.imread(image_file)

    threshold = 0.25

    mask = (image / 255).sum(axis=2) > threshold

    structure_elem = disk(3)
    _mask = opening(mask, structure_elem)

    processed_image = ((image / 255 * _mask[..., None]) * 255).astype(image.dtype)

    to_plot = [image, mask, _mask, processed_image]
    n_plots = len(to_plot)

    unit_size = 5

    fig, axes = plt.subplots(1, n_plots, figsize=(unit_size*n_plots, unit_size))

    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()

    for img, ax in zip(to_plot, axes):
        ax.imshow(img)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01)
    plt.show()

    n = 32

    dset = ImagesOnDiskDataset(
        image_files=image_file, postprocessing='tile_pad',
        n_samples_per_image=n, tile_shape=(200, 200),
        _no_cache=True,
    )
    print(dset.depth)

    sess = tf.Session()
    with sess.as_default():
        dset.visualize(n)
