import numpy as np
import imageio
import os
from skimage.transform import resize

from dps.datasets.base import PatchesDataset
from dps.utils import Param


class RandomShapesDataset(PatchesDataset):
    """ Display a random number of random shapes in random positions. """
    shapes = Param()
    min_shapes = Param()
    max_shapes = Param()
    patch_shape = Param()

    n_classes = 1

    def _make(self):
        if isinstance(self.shapes, str):
            self.shapes = self.shapes.split()
        self.shapes = list(self.shapes)

        self.images = {}
        for shape in self.shapes:
            f = os.path.join(os.path.dirname(__file__), "shape_images", "{}.png".format(shape))
            image = imageio.imread(f)
            image = resize(image, self.patch_shape, mode='edge', preserve_range=True)

            # Use only the alpha channel.
            self.images[shape] = image[..., 3]
        super(ShapesDataset, self)._make()

    def _sample_patches(self):
        n_shapes = np.random.randint(self.min_shapes, self.max_shapes+1)
        shapes = np.random.choice(self.shapes, size=n_shapes)
        shape_images = [self._colourize(self.images[shape]) for shape in shapes]
        shapes = [self.shapes.index(shape) for shape in shapes]
        return shape_images, shapes, 0


class ShapesDataset(PatchesDataset):
    """ Display a specific set of shapes in ranom positions. """
    shapes = Param(help="space-separated string of shape specs. Each shape spec is of form color,shape")
    patch_shape = Param()

    n_classes = 1

    def _make(self):
        if isinstance(self.shapes, str):
            self.shapes = self.shapes.split()

        self.patches = []
        for spec in self.shapes:
            colour, shape = spec.split(",")

            f = os.path.join(os.path.dirname(__file__), "shape_images", "{}.png".format(shape))
            image = imageio.imread(f)
            image = image[..., 3]
            image = resize(image, self.patch_shape, mode='edge', preserve_range=True)
            image = self._colourize(image, colour)

            self.patches.append(image)

        super(ShapesDataset, self)._make()

    def _sample_patches(self):
        return list(np.random.permutation(self.patches)), [0 for i in range(len(self.patches))], 0


if __name__ == "__main__":
    import tensorflow as tf
    # dset = RandomShapesDataset(
    #     n_examples=20, shapes="circle", colours="black blue", background_colours="white",
    #     min_shapes=2, max_shapes=4, image_shape=(48, 48), patch_shape=(14, 14), max_overlap=98)
    shapes = "green,circle blue,circle orange,circle teal,circle red,circle black,circle"
    dset = ShapesDataset(
        n_examples=20, background_colours="white", shapes=shapes,
        image_shape=(48, 48), patch_shape=(14, 14), max_overlap=98)

    sess = tf.Session()
    with sess.as_default():
        dset.visualize()