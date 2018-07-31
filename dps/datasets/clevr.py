import os
import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize

from dps import cfg
from dps.datasets.base import ImageDataset, ImageFeature
from dps.utils import Param


class ClevrDataset(ImageDataset):
    clevr_kind = Param()
    image_shape = Param()

    _features = None

    @property
    def features(self):
        if self._features is None:
            self._features = [ImageFeature("image", self.obs_shape)]
        return self._features

    @property
    def depth(self):
        return 3

    def _make(self):
        assert self.clevr_kind in "train val test".split()
        directory = os.path.join(cfg.data_dir, "CLEVR_v1.0/images", self.clevr_kind)
        files = os.listdir(directory)
        files = np.random.choice(files, size=self.n_examples, replace=False)

        for f in files:
            image = imageio.imread(os.path.join(directory, f))
            image = image[..., :3]  # Get rid of alpa channel.

            if image.shape[:2] != self.image_shape:
                image = resize(image, self.image_shape, mode='edge', preserve_range=True)

            self._write_example(image=image)

    def visualize(self, n=4):
        batch_size = n
        images, *_ = self.sample(n)

        for i in range(batch_size):
            fig, ax = plt.subplots(1, 1)
            ax.imshow(images[i])
            plt.show()

    def sample(self, n=4):
        batch_size = n
        dset = tf.data.TFRecordDataset(self.filename)
        dset = dset.batch(batch_size).map(self.parse_example_batch)

        iterator = dset.make_one_shot_iterator()

        sess = tf.get_default_session()

        return sess.run(iterator.get_next())


if __name__ == "__main__":
    dset = ClevrDataset(
        n_examples=11, clevr_kind="train", seed=0, image_shape=(90, 120),
        postprocessing="random", tile_shape=(60, 60), n_samples_per_image=10)

    with tf.Session().as_default():
        dset.visualize(16)
