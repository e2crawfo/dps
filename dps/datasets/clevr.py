import os
import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from itertools import product
import collections

from dps import cfg
from dps.datasets.base import ImageDataset, ImageFeature, NestedListFeature, IntegerFeature
from dps.utils import Param, resize_image


class ClevrDataset(ImageDataset):
    clevr_kind = Param()
    image_shape = Param()
    clevr_background_mode = Param()
    example_range = Param()

    _features = None

    @property
    def has_annotations(self):
        return self.clevr_kind in "train val".split()

    @property
    def features(self):
        if self._features is None:
            self._features = [
                ImageFeature("image", self.obs_shape),
                NestedListFeature("annotations", 5),
                IntegerFeature("label", 1),
            ]

            if self.clevr_background_mode is not None:
                self._features.append(
                    ImageFeature("background", self.obs_shape)
                )

        return self._features

    @property
    def depth(self):
        return 3

    def get_idx_from_filename(self, filename):
        filename = os.path.split(filename)[1]
        idx_start = 10 if self.clevr_kind == "val" else 12
        return int(filename[idx_start:idx_start+6])

    @staticmethod
    def compute_pixelwise_mean(files, shape):
        mean = None
        n_points = 0
        for k, f in enumerate(files):
            if k % 100 == 0:
                print("Processing files {}".format(k))
            image = imageio.imread(os.path.join(f))
            image = resize_image(image[:, :, :3], shape)

            if mean is None:
                mean = image
            else:
                mean = (mean * n_points + image) / (n_points + 1)
            n_points += 1

        return mean.astype(image.dtype)

    @staticmethod
    def compute_pixelwise_mode(files, shape):
        counters = None

        for k, f in enumerate(files):
            if k % 100 == 0:
                print("Processing files {}".format(k))
            image = imageio.imread(os.path.join(f))
            image = resize_image(image[:, :, :3], shape)

            if counters is None:
                counters = np.array([collections.Counter() for i in range(image.shape[0] * image.shape[1])])
                counters = counters.reshape(image.shape[:2])

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    counters[i, j].update([tuple(image[i, j])])

        mode_image = [[c.most_common(1)[0][0] for c in row] for row in counters]
        return np.array(mode_image).astype(image.dtype)

    # @staticmethod
    # def compute_pixelwise_mode(files):
    #     import scipy
    #     images = []
    #     for f in files:
    #         image = imageio.imread(os.path.join(f))
    #         image = image[..., :3]  # Get rid of alpha channel.
    #         images.append(image)
    #     images = np.array(images).astype('i')
    #     images = images[..., 0] + 256 * images[..., 1] * 256 * 256 * images[..., 2]
    #     mode_image = scipy.stats.mode(images, axis=0)[0]
    #     r = mode_image % 256
    #     mode_image = (mode_image - r) / 256
    #     g = mode_image % 256
    #     mode_image = (mode_image - g) / 256
    #     b = mode_image
    #     mode_image = np.stack([r, g, b], axis=2)
    #     return np.array(mode_image).astype(image.dtype)

    @staticmethod
    def compute_pixelwise_mode2(files, shape):
        data = []

        for f in files[:10]:
            image = imageio.imread(os.path.join(f))
            image = resize_image(image[:, :, :3], shape)
            data.extend(list(image.reshape(-1, 3)))

        data = np.random.permutation(data)[:10000]

        from sklearn import cluster
        n_clusters = 64
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        print("Clustering...")
        kmeans.fit(data)

        values = kmeans.cluster_centers_.squeeze()[:, None, None, :]

        counters = None

        print("Counting...")
        for f in files:
            image = imageio.imread(os.path.join(f))
            image = resize_image(image[:, :, :3], shape)

            if counters is None:
                counters = np.array([collections.Counter() for i in range(image.shape[0] * image.shape[1])])
                counters = counters.reshape(image.shape[:2])

            distances = np.linalg.norm(values - image[None, :, :, :], axis=3)
            indices = np.argmin(distances, axis=0)

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    counters[i, j].update([indices[i, j]])

        print("Computing mode...")
        mode_image = [[values[c.most_common(1)[0][0], 0, 0, :] for c in row] for row in counters]

        print("Done...")
        return np.array(mode_image).astype(image.dtype)

    def _make(self):
        assert self.clevr_kind in "train val test".split()

        if self.has_annotations:
            sizes = ['large', 'small']
            colors = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
            materials = ['rubber', 'metal']
            shapes = ['cube', 'sphere', 'cylinder']
            self.class_names = [' '.join(lst) for lst in product(sizes, colors, materials, shapes)]

            scene_file = os.path.join(
                cfg.data_dir, "CLEVR_v1.0/scenes/CLEVR_{}_scenes.json".format(self.clevr_kind))
            with open(scene_file, 'r') as f:
                scenes = json.load(f)['scenes']

        directory = os.path.join(cfg.data_dir, "CLEVR_v1.0/images", self.clevr_kind)
        files = os.listdir(directory)

        if self.example_range is not None:
            assert self.example_range[0] < self.example_range[1]
            files = [(f, self.get_idx_from_filename(f)) for f in files]
            files = [f for f, idx in files if self.example_range[0] <= idx < self.example_range[1]]

        files = np.random.choice(files, size=int(self.n_examples), replace=False)
        files = [os.path.join(directory, f) for f in files]

        background = None
        if self.clevr_background_mode == "mean":
            background = self.compute_pixelwise_mean(files[:5000], self.image_shape)
        elif self.clevr_background_mode == "median":
            background = self.compute_pixelwise_median(files[:5000], self.image_shape)
        elif self.clevr_background_mode == "mode":
            background = self.compute_pixelwise_mode(files[:5000], self.image_shape)
            # background = self.compute_pixelwise_mode2(files, self.image_shape)
        if background is not None:
            background = background.astype('uint8')

        for k, f in enumerate(files):
            if k % 100 == 0:
                print("Processing image {}".format(k))
            image = imageio.imread(f)
            image = image[..., :3]  # Get rid of alpha channel.

            if image.shape[:2] != self.image_shape:
                image = resize_image(image, self.image_shape)

            if self.has_annotations:
                idx = self.get_idx_from_filename(f)
                scene = scenes[idx]
                assert scene['image_filename'] == os.path.split(f)[1]
                annotations = self.extract_bounding_boxes(scene)
            else:
                annotations = []

            self._write_example(image=image, annotations=annotations, label=0, background=background)

    def extract_bounding_boxes(self, scene):
        objs = scene['objects']
        rotation = scene['directions']['right']

        orig_image_height = 320.0
        orig_image_width = 480.0
        image_height, image_width = self.image_shape

        annotations = []

        for i, obj in enumerate(objs):
            [x, y, z] = obj['pixel_coords']

            [x1, y1, z1] = obj['3d_coords']

            cos_theta, sin_theta, _ = rotation

            x1 = x1 * cos_theta + y1 * sin_theta
            y1 = x1 * -sin_theta + y1 * cos_theta

            height_d = 6.9 * z1 * (15 - y1) / 2.0
            height_u = height_d
            width_l = height_d
            width_r = height_d

            if obj['shape'] == 'cylinder':
                d = 9.4 + y1
                h = 6.4
                s = z1

                height_u *= (s*(h/d + 1)) / ((s*(h/d + 1)) - (s*(h-s)/d))
                height_d = height_u * (h-s+d) / (h + s + d)

                width_l *= 11/(10 + y1)
                width_r = width_l

            if obj['shape'] == 'cube':
                height_u *= 1.3 * 10 / (10 + y1)
                height_d = height_u
                width_l = height_u
                width_r = height_u

            obj_name = obj['size'] + ' ' + obj['color'] + ' ' + obj['material'] + ' ' + obj['shape']

            cls = self.class_names.index(obj_name)
            ymin = image_height * (y - height_d) / orig_image_height
            ymax = image_height * (y + height_u) / orig_image_height
            xmin = image_width * (x - width_l) / orig_image_width
            xmax = image_width * (x + width_r) / orig_image_width

            annotations.append([cls, ymin, ymax, xmin, xmax])

        return annotations

    def visualize(self, n=4):
        batch_size = n

        if self.has_annotations:
            images, annotations, n_annotations, _, backgrounds = self.sample(n)

            for i in range(batch_size):
                fig, axes = plt.subplots(1, 2)
                ax = axes[0]
                ax.imshow(images[i])
                for cls, ymin, ymax, xmin, xmax in annotations[i]:
                    height = ymax - ymin
                    width = xmax - xmin
                    rect = patches.Rectangle(
                        (xmin, ymin), width, height, linewidth=2,
                        edgecolor="xkcd:azure", facecolor='none')
                    ax.add_patch(rect)
                axes[1].imshow(backgrounds[i])
                plt.show()
        else:
            images, _, _, _, backgrounds = self.sample(n)

            for i in range(batch_size):
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(images[i])
                ax[1].imshow(backgrounds[i])
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
        n_examples=10000, clevr_kind="train", seed=0, image_shape=(80, 120),
        n_samples_per_image=10, clevr_background_mode="mean")
    # dset = ClevrDataset(
    #     n_examples=20, clevr_kind="train", seed=0, image_shape=(80, 120),
    #     postprocessing="random", tile_shape=(60, 60), n_samples_per_image=10)

    with tf.Session().as_default():
        dset.visualize(4)
