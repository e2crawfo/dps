import os
import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
import json
from itertools import product

from dps import cfg
from dps.datasets.base import ImageDataset, ImageFeature, NestedListFeature
from dps.utils import Param


class ClevrDataset(ImageDataset):
    clevr_kind = Param()
    image_shape = Param()

    _features = None

    @property
    def has_annotations(self):
        return self.clevr_kind in "train val".split()

    @property
    def features(self):
        if self._features is None:
            if self.has_annotations:
                self._features = [ImageFeature("image", self.obs_shape), NestedListFeature("annotations", 5)]
            else:
                self._features = [ImageFeature("image", self.obs_shape)]
        return self._features

    @property
    def depth(self):
        return 3

    def get_idx_from_filename(self, filename):
        idx_start = 10 if self.clevr_kind == "val" else 12
        return int(filename[idx_start:idx_start+6])

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
        files = np.random.choice(files, size=self.n_examples, replace=False)

        for f in files:
            image = imageio.imread(os.path.join(directory, f))
            image = image[..., :3]  # Get rid of alpa channel.

            if image.shape[:2] != self.image_shape:
                image = resize(image, self.image_shape, mode='edge', preserve_range=True)

            if self.has_annotations:
                idx = self.get_idx_from_filename(f)
                scene = scenes[idx]
                assert scene['image_filename'] == f
                annotations = self.extract_bounding_boxes(scene)
                self._write_example(image=image, annotations=annotations)
            else:
                self._write_example(image=image)

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
            images, annotations, n_annotations = self.sample(n)

            for i in range(batch_size):
                fig, ax = plt.subplots(1, 1)
                ax.imshow(images[i])
                for cls, ymin, ymax, xmin, xmax in annotations[i]:
                    height = ymax - ymin
                    width = xmax - xmin
                    rect = patches.Rectangle(
                        (xmin, ymin), width, height, linewidth=2, edgecolor="xkcd:azure", facecolor='none')
                    ax.add_patch(rect)
                plt.show()
        else:
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
        n_examples=20, clevr_kind="train", seed=0, image_shape=(80, 120),
        postprocessing="random", tile_shape=(60, 60), n_samples_per_image=10)

    with tf.Session().as_default():
        dset.visualize(16)
