import numpy as np
import tensorflow as tf

from dps import cfg
from dps.utils import Param
from dps.datasets import load_emnist
from dps.dataset import DatasetBuilder, PatchesBuilder, GridPatchesBuilder, tf_image_representation


def tf_annotation_representation(annotation, prefix=""):
    """ Get a representation of an annotation list suitable for passing to tf.train.Features """
    features = dict(
        length=tf.train.Feature(float_list=tf.train.FloatList(value=[len(annotation)])),
        annotation=tf.train.Feature(float_list=tf.train.FloatList(value=annotation))
    )

    if prefix:
        features = {"{}_{}".format(prefix, name): rep for name, rep in features.items()}

    return features


class ObjectDetectionBuilder(DatasetBuilder):
    postprocessing = Param("")
    tile_shape = Param(None)
    n_samples_per_image = Param(1)

    def _write_example(self, image, image_label, annotation):
        if self.postprocessing == "tile":
            images, annotations = self._tile_postprocess(image, annotation)
        elif self.postprocessing == "random":
            images, annotations = self._random_postprocess(image, annotation)
        else:
            images, annotations = [image], [annotation]

        for image, annotation in zip(images, annotations):
            features = tf_image_representation(image)
            features.update(tf_annotation_representation(annotation))
            example = tf.train.Example(features=tf.train.Features(feature=features))

            self._writer.write(example.SerializeToString())

    def _tile_postprocess(self, image, annotations):
        height, width, n_channels = image.shape

        hangover = width % self.tile_shape[1]
        if hangover != 0:
            pad_amount = self.tile_shape[1] - hangover
            pad_shape = (height, pad_amount)
            padding = np.zeros(pad_shape)
            image = np.concat([image, padding], axis=2)

        hangover = height % self.tile_shape[0]
        if hangover != 0:
            pad_amount = self.tile_shape[0] - hangover
            pad_shape = list(image.shape)
            pad_shape[1] = pad_amount
            padding = np.zeros(pad_shape)
            image = np.concat([image, padding], axis=1)

        pad_height = self.tile_shape[0] - height % self.tile_shape[0]
        pad_width = self.tile_shape[1] - width % self.tile_shape[1]
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

        H = int(height / self.tile_shape[0])
        W = int(width / self.tile_shape[1])

        slices = np.split(image, W, axis=1)
        new_shape = (H, *self.tile_shape, n_channels)
        slices = [np.reshape(s, new_shape) for s in slices]
        new_images = np.concatenate(slices, axis=1)
        new_images = new_images.reshape(H * W, *self.tile_shape, n_channels)

        new_annotations = []

        for h in range(H):
            for w in range(W):
                offset = (h * self.tile_shape[0], w * self.tile_shape[1])
                _new_annotations = []
                for l, top, bottom, left, right in annotations:
                    # Transform to tile co-ordinates
                    top = top - offset[0]
                    bottom = bottom - offset[0]
                    left = left - offset[1]
                    right = right - offset[1]

                    # Restrict to chosen crop
                    top = np.clip(top, 0, self.tile_shape[0])
                    bottom = np.clip(bottom, 0, self.tile_shape[0])
                    left = np.clip(left, 0, self.tile_shape[1])
                    right = np.clip(right, 0, self.tile_shape[1])

                    invalid = (bottom - top < 1e-6) or (right - left < 1e-6)

                    if not invalid:
                        _new_annotations.append((l, top, bottom, left, right))

                new_annotations.append(_new_annotations)

        return new_images, new_annotations

    def _random_postprocess(self, image, annotations):
        height, width, _ = image.shape
        new_images = []
        new_annotations = []

        for j in range(self.n_samples_per_image):
            _top = np.random.randint(0, height-self.tile_shape[0]+1)
            _left = np.random.randint(0, width-self.tile_shape[1]+1)

            crop = image[_top:_top+self.tile_shape[0], _left:_left+self.tile_shape[1], ...]
            new_images.append(crop)

            offset = (_top, _left)
            _new_annotations = []
            for l, top, bottom, left, right in annotations:
                top = top - offset[0]
                bottom = bottom - offset[0]
                left = left - offset[1]
                right = right - offset[1]

                top = np.clip(top, 0, self.tile_shape[0])
                bottom = np.clip(bottom, 0, self.tile_shape[0])
                left = np.clip(left, 0, self.tile_shape[1])
                right = np.clip(right, 0, self.tile_shape[1])

                invalid = (bottom - top < 1e-6) or (right - left < 1e-6)

                if not invalid:
                    _new_annotations.append((l, top, bottom, left, right))

            new_annotations.append(_new_annotations)

        return new_images, new_annotations


class EmnistObjectDetection(ObjectDetectionBuilder, PatchesBuilder):
    min_chars = Param(2)
    max_chars = Param(3)
    characters = Param(
        [str(i) for i in range(10)] +
        [chr(i + ord('A')) for i in range(26)] +
        [chr(i + ord('a')) for i in range(26)]
    )
    patch_shape = Param((14, 14))
    n_patch_examples = Param(None)
    example_range = Param(None)
    colours = Param('red green blue')

    def _make(self):
        assert self.min_chars <= self.max_chars

        emnist_x, emnist_y, self.classmap = load_emnist(cfg.data_dir, self.characters, balance=True,
                                                        shape=self.patch_shape, one_hot=False,
                                                        n_examples=self.n_patch_examples,
                                                        example_range=self.example_range)

        self.char_reps = zip(emnist_x, emnist_y)
        result = super(EmnistObjectDetection, self)._make()
        del self.char_reps

        return result

    def _sample_patches(self):
        n_chars = np.random.randint(self.min_chars, self.max_chars+1)

        if not n_chars:
            return [], []

        indices = [np.random.randint(len(self.char_reps)) for i in range(n_chars)]
        chars = [self.char_reps[i] for i in indices]
        char_x, char_y = zip(*chars)
        char_x = [self._colourize(cx) for cx in char_x]

        return char_x, char_y, None

    def visualize(self, n=9):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        m = int(np.ceil(np.sqrt(n)))
        fig, subplots = plt.subplots(m, m)

        height = self.x.shape[1]

        for i, ax in enumerate(subplots.flatten()):
            ax.imshow(self.x[i, ...])
            for cls, top, bottom, left, right in self.y[i]:
                width = right - left
                height = bottom - top
                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
        plt.show()


class GridEmnistObjectDetection(EmnistObjectDetection, GridPatchesBuilder):
    pass
