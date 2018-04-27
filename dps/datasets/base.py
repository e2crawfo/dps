import numpy as np
from skimage.transform import resize
import os
import tensorflow as tf

from dps import cfg
from dps.utils import image_to_string, Param, Parameterized, get_param_hash
from dps.datasets import (
    load_emnist, load_omniglot, omniglot_classes,
    load_backgrounds, background_names
)


class Dataset(Parameterized):
    n_examples = Param(None)

    def __init__(self, shuffle=True, **kwargs):
        print("Trying to find dataset in cache...")

        directory = os.path.join(cfg.data_dir, "cached_datasets", self.__class__.__name__)
        os.makedirs(directory, exist_ok=True)

        params = self.param_values()
        param_hash = get_param_hash(params)
        print("Params: {}".format(params))
        print("Param hash: {}".format(param_hash))

        self.filename = os.path.join(directory, str(param_hash))

        if not os.path.exists(self.filename):
            print("File for dataset not found, creating...")

            self._writer = tf.python_io.TFRecordWriter(self.filename)
            try:
                self._make()
                self._writer.close()
            except Exception:
                self._writer.close()
                try:
                    os.remove(self.filename)
                except FileNotFoundError:
                    pass
                raise

            with open(self.filename + ".cfg", 'w') as f:
                f.write(str(params))
        else:
            print("Found.")

    def _make(self):
        raise Exception("AbstractMethod. When insantiating `Dataset` directly, "
                        "`tracks` must be provided as an argument to `__init__`.")

    def _write_example(self):
        raise Exception("AbstractMethod.")


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_image_representation(image, prefix=""):
    """ Get a representation of an image suitable for passing to tf.train.Features """
    height, width, n_channels = image.shape
    image_raw = image.tostring()
    features = dict(
        height=_int64_feature(height),
        width=_int64_feature(width),
        n_channels=_int64_feature(n_channels),
        image_raw=_bytes_feature(image_raw))

    if prefix:
        features = {"{}_{}".format(prefix, name): rep for name, rep in features.items()}

    return features


def tf_annotation_representation(annotation, prefix=""):
    """ Get a representation of an annotation list suitable for passing to tf.train.Features """
    flat_annotation = [v for a in annotation for v in a]  # each has 5 values

    features = dict(
        n_annotations=tf.train.Feature(int64_list=tf.train.Int64List(value=[len(annotation)])),
        annotations=tf.train.Feature(float_list=tf.train.FloatList(value=flat_annotation))
    )

    if prefix:
        features = {"{}_{}".format(prefix, name): rep for name, rep in features.items()}

    return features


class ImageClassification(Dataset):
    one_hot = Param()
    classes = Param()
    image_shape = Param()
    include_blank = Param()

    features = {
        "image_raw": tf.FixedLenFeature((), dtype=tf.string),
        "label": tf.FixedLenFeature((), dtype=tf.int64),
    }

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def obs_shape(self):
        return self.image_shape + (self.depth,)

    @property
    def depth(self):
        return 1

    def _write_example(self, image, label):
        features = tf_image_representation(image)
        features.update(label=_int64_feature(label))
        example = tf.train.Example(features=tf.train.Features(feature=features))

        self._writer.write(example.SerializeToString())

    def parse_example_batch(self, example_proto):
        features = tf.parse_example(example_proto, features=ImageClassification.features)

        images = tf.decode_raw(features['image_raw'], tf.uint8)
        images = tf.reshape(images, (-1,) + self.obs_shape)
        images = tf.image.convert_image_dtype(images, tf.float32)
        images = tf.clip_by_value(images, 1e-6, 1-1e-6)

        label = tf.cast(features['label'], tf.int32)

        if self.one_hot:
            label = tf.one_hot(label, self.n_classes)

        return images, label


class Emnist(ImageClassification):
    """
    Download and pre-process EMNIST dataset:
    python scripts/download.py emnist <desired location>

    """
    balance = Param(False)
    example_range = Param(None)

    class_pool = ''.join(
        [str(i) for i in range(10)] +
        [chr(i + ord('A')) for i in range(26)] +
        [chr(i + ord('a')) for i in range(26)]
    )

    def _make(self):
        param_values = self.param_values()
        param_values['one_hot'] = False
        param_values['shape'] = param_values['image_shape']
        del param_values['image_shape']

        x, y, class_map = load_emnist(cfg.data_dir, **param_values)

        if x.shape[0] < self.n_examples:
            raise Exception(
                "Too few datapoints. Requested {}, "
                "only {} are available.".format(self.n_examples, x.shape[0]))

        for _x, _y in x, y:
            self._write_example(_x, class_map[_y])


class Omniglot(ImageClassification):
    indices = Param()

    @staticmethod
    def sample_classes(n_classes):
        class_pool = omniglot_classes()
        classes = np.random.choice(len(class_pool), n_classes, replace=False)
        return [class_pool[i] for i in classes]

    def _make(self, **kwargs):
        param_values = self.param_values()
        param_values['one_hot'] = False
        param_values['shape'] = param_values['image_shape']
        del param_values['image_shape']
        del param_values['n_examples']

        x, y, class_map = load_omniglot(cfg.data_dir, **param_values)

        if x.shape[0] < self.n_examples:
            raise Exception(
                "Too few datapoints. Requested {}, "
                "only {} are available.".format(self.n_examples, x.shape[0]))

        for _x, _y in x, y:
            self._write_example(_x, class_map[_y])


class Rectangle(object):
    def __init__(self, y, x, h, w):
        self.top = y
        self.bottom = y+h
        self.left = x
        self.right = x+w

        self.h = h
        self.w = w

    def intersects(self, r2):
        r1 = self
        h_overlaps = (r1.left <= r2.right) and (r1.right >= r2.left)
        v_overlaps = (r1.top <= r2.bottom) and (r1.bottom >= r2.top)
        return h_overlaps and v_overlaps

    def centre(self):
        return (
            self.top + (self.bottom - self.top) / 2.,
            self.left + (self.right - self.left) / 2.
        )

    def __str__(self):
        return "Rectangleangle({}:{}, {}:{})".format(self.top, self.bottom, self.left, self.right)


class Patches(Dataset):
    max_overlap = Param(10)
    image_shape = Param((100, 100))
    draw_shape = Param(None)
    draw_offset = Param((0, 0))
    patch_size_std = Param(None)
    distractor_shape = Param((3, 3))
    n_distractors_per_image = Param(0)
    backgrounds = Param(
        "", help="Can be either be 'all', in which a random background will be selected for "
                 "each constructed image, or a list of strings, giving the names of backgrounds "
                 "to use.")
    backgrounds_sample_every = Param(
        False, help="If True, sample a new sub-region of background for each image. Otherwise, "
                    "sample a small set of regions initially, and use those for all images.")
    backgrounds_resize = Param(False)
    background_colours = Param("")
    max_attempts = Param(10000)
    colours = Param('red green blue')

    postprocessing = Param("")
    tile_shape = Param(None)
    n_samples_per_image = Param(1)
    one_hot = Param(True)

    features = {
        "image_raw": tf.FixedLenFeature((), dtype=tf.string),
        "annotations": tf.VarLenFeature(dtype=tf.float32),
        "n_annotations": tf.FixedLenFeature((), dtype=tf.int64),
        "label": tf.FixedLenFeature((), dtype=tf.int64),
    }

    @property
    def n_classes(self):
        raise Exception("AbstractMethod")

    @property
    def depth(self):
        return 3 if self.colours else 1

    @property
    def obs_shape(self):
        if self.postprocessing:
            return self.tile_shape + (self.depth,)
        else:
            return self.image_shape + (self.depth,)

    def _write_example(self, image, annotation, label):
        if self.postprocessing == "tile":
            images, annotations = self._tile_postprocess(image, annotation)
        elif self.postprocessing == "random":
            images, annotations = self._random_postprocess(image, annotation)
        else:
            images, annotations = [image], [annotation]

        for image, annotation in zip(images, annotations):
            features = tf_image_representation(image)
            features.update(tf_annotation_representation(annotation))
            features.update(label=_int64_feature(label))

            example = tf.train.Example(features=tf.train.Features(feature=features))

            self._writer.write(example.SerializeToString())

    def parse_example_batch(self, example_proto):
        features = tf.parse_example(example_proto, features=Patches.features)

        images = tf.decode_raw(features['image_raw'], tf.uint8)
        images = tf.reshape(images, (-1,) + self.obs_shape)
        images = tf.image.convert_image_dtype(images, tf.float32)
        images = tf.clip_by_value(images, 1e-6, 1-1e-6)

        n_annotations = tf.cast(features['n_annotations'], tf.int32)
        max_n_annotations = tf.reduce_max(n_annotations)

        annotations = tf.sparse_tensor_to_dense(features['annotations'], default_value=0)
        annotations = tf.reshape(annotations, (-1, max_n_annotations, 5))

        label = tf.cast(features['label'], tf.int32)

        if self.one_hot:
            label = tf.one_hot(label, self.n_classes)

        return images, annotations, n_annotations, label

    def _make(self):
        if self.n_examples == 0:
            return np.zeros((0,) + self.image_shape).astype('uint8'), np.zeros((0, 1)).astype('i')

        # --- prepare colours ---

        colours = self.colours
        if isinstance(colours, str):
            colours = colours.split()

        import matplotlib as mpl
        colour_map = mpl.colors.get_named_colors_mapping()
        self._colours = [np.array(mpl.colors.to_rgb(colour_map[cn]))[None, None, :] for cn in colours]

        # --- prepare shapes ---

        self.draw_shape = self.draw_shape or self.image_shape
        self.draw_offset = self.draw_offset or (0, 0)

        draw_shape = self.draw_shape
        if self.depth is not None:
            draw_shape = draw_shape + (self.depth,)

        # --- prepare backgrounds ---

        if self.backgrounds == "all":
            backgrounds = background_names()
        elif isinstance(self.backgrounds, str):
            backgrounds = self.backgrounds.split()
        else:
            backgrounds = self.backgrounds

        if backgrounds:
            if self.backgrounds_resize:
                backgrounds = load_backgrounds(backgrounds, draw_shape)
            else:
                backgrounds = load_backgrounds(backgrounds)

                if not self.backgrounds_sample_every:
                    _backgrounds = []
                    for b in backgrounds:
                        top = np.random.randint(b.shape[0] - draw_shape[0] + 1)
                        left = np.random.randint(b.shape[1] - draw_shape[1] + 1)
                        _backgrounds.append(
                            b[top:top+draw_shape[0], left:left+draw_shape[1], ...] + 0
                        )
                    backgrounds = _backgrounds

        background_colours = self.background_colours
        if isinstance(self.background_colours, str):
            background_colours = background_colours.split()
        _background_colours = []
        from matplotlib.colors import to_rgb
        for bc in background_colours:
            color = to_rgb(bc)
            color = np.array(color)[None, None, :]
            color = np.uint8(255. * color)
            _background_colours.append(color)
        background_colours = _background_colours

        # --- start dataset creation ---

        for j in range(self.n_examples):

            # --- populate background ---

            if backgrounds:
                b_idx = np.random.randint(len(backgrounds))
                background = backgrounds[b_idx]
                if self.backgrounds_sample_every:
                    top = np.random.randint(background.shape[0] - draw_shape[0] + 1)
                    left = np.random.randint(background.shape[1] - draw_shape[1] + 1)
                    image = background[top:top+draw_shape[0], left:left+draw_shape[1], ...] + 0
                else:
                    image = background + 0
            elif background_colours:
                color = background_colours[np.random.randint(len(background_colours))]
                image = color * np.ones(draw_shape, 'uint8')
            else:
                image = np.zeros(draw_shape, 'uint8')

            # --- sample and populate patches ---

            patches, patch_labels, image_label = self._sample_patches()
            patch_shapes = np.array([img.shape for img in patches])

            locs = self._sample_patch_locations(
                patch_shapes,
                max_overlap=self.max_overlap,
                size_std=self.patch_size_std)

            draw_offset = self.draw_offset

            for patch, loc in zip(patches, locs):
                if patch.shape[:2] != (loc.h, loc.w):
                    patch = resize(patch, (loc.h, loc.w), mode='edge', preserve_range=True)

                if patch.shape[-1] == 4:
                    alpha, patch = np.split(patch, [3, 1], axis=-1)

                    current = image[loc.top:loc.bottom, loc.left:loc.right, ...]
                    image[loc.top:loc.bottom, loc.left:loc.right, ...] = alpha * patch + (1 - alpha) * current
                else:
                    current = image[loc.top:loc.bottom, loc.left:loc.right, ...]
                    image[loc.top:loc.bottom, loc.left:loc.right, ...] = np.maximum(current, patch)

            # --- add distractors ---

            if self.n_distractors_per_image > 0:
                distractor_patches = self._sample_distractors()
                distractor_shapes = [img.shape for img in distractor_patches]
                distractor_locs = self._sample_patch_locations(distractor_shapes)

                for patch, loc in zip(distractor_patches, distractor_locs):
                    if patch.shape[:2] != (loc.h, loc.w):
                        patch = resize(patch, (loc.h, loc.w), mode='edge', preserve_range=True)

                    if patch.shape[-1] == 4:
                        alpha, patch = np.split(patch, [3, 1], axis=-1)

                        current = image[loc.top:loc.bottom, loc.left:loc.right, ...]
                        image[loc.top:loc.bottom, loc.left:loc.right, ...] = alpha * patch + (1 - alpha) * current
                    else:
                        current = image[loc.top:loc.bottom, loc.left:loc.right, ...]
                        image[loc.top:loc.bottom, loc.left:loc.right, ...] = np.maximum(patch, current)

            # --- possibly crop entire image ---

            if self.draw_shape != self.image_shape or draw_offset != (0, 0):
                image_shape = self.image_shape
                if self.depth is not None:
                    image_shape = image_shape + (self.depth,)

                draw_top = np.maximum(-draw_offset[0], 0)
                draw_left = np.maximum(-draw_offset[1], 0)

                draw_bottom = np.minimum(-draw_offset[0] + self.image_shape[0], self.draw_shape[0])
                draw_right = np.minimum(-draw_offset[1] + self.image_shape[1], self.draw_shape[1])

                image_top = np.maximum(draw_offset[0], 0)
                image_left = np.maximum(draw_offset[1], 0)

                image_bottom = np.minimum(draw_offset[0] + self.draw_shape[0], self.image_shape[0])
                image_right = np.minimum(draw_offset[1] + self.draw_shape[1], self.image_shape[1])

                _image = np.zeros(image_shape, 'uint8')
                _image[image_top:image_bottom, image_left:image_right, ...] = \
                    image[draw_top:draw_bottom, draw_left:draw_right, ...]

                image = _image

            annotations = self._get_annotations(draw_offset, patches, locs, patch_labels)

            self._write_example(image, annotations, image_label)

            if j % 1000 == 0:
                print(image_label)
                print(image_to_string(image))
                print("\n")

    def _get_annotations(self, draw_offset, patches, locs, labels):
        new_labels = []
        for patch, loc, label in zip(patches, locs, labels):
            nz_y, nz_x = np.nonzero(patch.sum(axis=2))

            # In draw co-ordinates
            top = (nz_y.min() / patch.shape[0]) * loc.h + loc.top
            bottom = (nz_y.max() / patch.shape[0]) * loc.h + loc.top
            left = (nz_x.min() / patch.shape[1]) * loc.w + loc.left
            right = (nz_x.max() / patch.shape[1]) * loc.w + loc.left

            # Transform to image co-ordinates
            top = top + draw_offset[0]
            bottom = bottom + draw_offset[0]
            left = left + draw_offset[1]
            right = right + draw_offset[1]

            top = np.clip(top, 0, self.image_shape[0])
            bottom = np.clip(bottom, 0, self.image_shape[0])
            left = np.clip(left, 0, self.image_shape[1])
            right = np.clip(right, 0, self.image_shape[1])

            invalid = (bottom - top < 1e-6) or (right - left < 1e-6)

            if not invalid:
                new_labels.append((label, top, bottom, left, right))

        return new_labels

    def _sample_patches(self):
        raise Exception("AbstractMethod")

    def _sample_patch_locations(self, patch_shapes, max_overlap=None, size_std=None):
        """ Sample random locations within draw_shape. """
        if len(patch_shapes) == 0:
            return []

        patch_shapes = np.array(patch_shapes)
        n_rects = patch_shapes.shape[0]
        i = 0
        while True:
            if size_std is None:
                shape_multipliers = 1.
            else:
                shape_multipliers = np.maximum(np.random.randn(n_rects, 2) * size_std + 1.0, 0.5)

            _patch_shapes = np.ceil(shape_multipliers * patch_shapes[:, :2]).astype('i')

            rects = [
                Rectangle(
                    np.random.randint(0, self.draw_shape[0]-m+1),
                    np.random.randint(0, self.draw_shape[1]-n+1), m, n)
                for m, n in _patch_shapes]
            area = np.zeros(self.draw_shape, 'uint8')

            for rect in rects:
                area[rect.top:rect.bottom, rect.left:rect.right] += 1

            if max_overlap is None or (area[area >= 2]-1).sum() < max_overlap:
                break

            i += 1

            if i > self.max_attempts:
                raise Exception(
                    "Could not fit rectangles. "
                    "(n_rects: {}, draw_shape: {}, max_overlap: {})".format(
                        n_rects, self.draw_shape, max_overlap))
        return rects

    def _sample_distractors(self):
        distractor_images = []

        patches = []
        while not patches:
            patches, y, _ = self._sample_patches()

        for i in range(self.n_distractors_per_image):
            idx = np.random.randint(len(patches))
            patch = patches[idx]
            m, n, *_ = patch.shape
            source_y = np.random.randint(0, m-self.distractor_shape[0]+1)
            source_x = np.random.randint(0, n-self.distractor_shape[1]+1)

            img = patch[
                source_y:source_y+self.distractor_shape[0],
                source_x:source_x+self.distractor_shape[1]]

            distractor_images.append(img)

        return distractor_images

    def _colourize(self, img, colour_idx=None):
        """ Apply a colour to a gray-scale image. """

        if not self._colours:
            return img[..., None]

        if colour_idx is None:
            colour_idx = np.random.randint(len(self._colours))

        colour = self._colours[colour_idx]
        colourized = np.array(img[..., None] * colour, np.uint8)
        return colourized

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

    def visualize(self, n=9):
        import matplotlib.pyplot as plt
        m = int(np.ceil(np.sqrt(n)))
        fig, subplots = plt.subplots(m, m)
        size = int(np.sqrt(self.x.shape[1]))
        for i, s in enumerate(subplots.flatten()):
            s.imshow(self.x[i, :].reshape(size, size))
            s.set_title(str(self.y[i, 0]))


class GridPatches(Patches):
    grid_shape = Param((2, 2))
    spacing = Param((0, 0))
    random_offset_range = Param(None)

    def _make(self):
        self.grid_size = np.product(self.grid_shape)
        self.cell_shape = (
            self.patch_shape[0] + self.spacing[0],
            self.patch_shape[1] + self.spacing[1])
        return super(GridPatches, self)._make()

    def _sample_patch_locations(self, patch_shapes, **kwargs):
        n_patches = len(patch_shapes)
        if not n_patches:
            return []
        indices = np.random.choice(self.grid_size, n_patches, replace=False)

        grid_locs = list(zip(*np.unravel_index(indices, self.grid_shape)))
        top_left = np.array(grid_locs) * self.cell_shape

        if self.random_offset_range is not None:
            grid_offset = (
                np.random.randint(self.random_offset_range[0]),
                np.random.randint(self.random_offset_range[1]),
            )
            top_left += grid_offset

        return [Rectangle(t, l, m, n) for (t, l), (m, n, _) in zip(top_left, patch_shapes)]


class VisualArithmetic(Patches):
    """ A dataset for the VisualArithmetic task.

    An image dataset that requires performing different arithmetical
    operations on digits.

    Each image contains a letter specifying an operation to be performed, as
    well as some number of digits. The corresponding label is whatever one gets
    when applying the given operation to the given collection of digits.

    The operation to be performed in each image, and the digits to perform them on,
    are represented using images from the EMNIST dataset.

    Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
    EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373.

    """
    reductions = Param("A:sum,M:prod")
    min_digits = Param(2)
    max_digits = Param(3)
    digits = Param(list(range(10)))

    largest_digit = Param(1000, help="All digits larger than this are lumped in with the largest "
                                     "so there are largest_digit + 1 (for zero) classes.")

    patch_shape = Param((14, 14))
    n_patch_examples = Param(None)
    example_range = Param(None)

    reductions_dict = {
        "sum": sum,
        "prod": np.product,
        "max": max,
        "min": min,
        "len": len,
    }

    @property
    def n_classes(self):
        return self.largest_digit + 1

    def _make(self):
        self.digits = [int(d) for d in self.digits]
        assert self.min_digits <= self.max_digits

        reductions = self.reductions
        if isinstance(reductions, str):
            if ":" not in reductions:
                reductions = self.reductions_dict[reductions.strip()]
            else:
                _reductions = {}
                delim = ',' if ',' in reductions else ' '
                for pair in reductions.split(delim):
                    char, key = pair.split(':')
                    _reductions[char] = self.reductions_dict[key]
                reductions = _reductions

        if isinstance(reductions, dict):
            op_characters = sorted(reductions)
            emnist_x, emnist_y, character_map = load_emnist(cfg.data_dir, op_characters, balance=True,
                                                            shape=self.patch_shape, one_hot=False,
                                                            n_examples=self.n_patch_examples,
                                                            example_range=self.example_range)
            emnist_y = emnist_y.flatten()

            self._remapped_reductions = {character_map[k]: v for k, v in reductions.items()}

            self.op_reps = list(zip(emnist_x, emnist_y))
        else:
            assert callable(reductions)
            self.op_reps = None
            self.func = reductions

        mnist_x, mnist_y, classmap = load_emnist(cfg.data_dir, self.digits, balance=True,
                                                 shape=self.patch_shape, one_hot=False,
                                                 n_examples=self.n_patch_examples,
                                                 example_range=self.example_range)
        mnist_y = mnist_y.flatten()

        inverted_classmap = {v: k for k, v in classmap.items()}
        mnist_y = np.array([inverted_classmap[y] for y in mnist_y])

        self.digit_reps = list(zip(mnist_x, mnist_y))

        result = super(VisualArithmetic, self)._make()

        del self.digit_reps
        del self.op_reps

        return result

    def _sample_patches(self):
        n_digits = np.random.randint(self.min_digits, self.max_digits+1)

        indices = [np.random.randint(len(self.digit_reps)) for i in range(n_digits)]
        digits = [self.digit_reps[i] for i in indices]

        digit_x, digit_y = list(zip(*digits))

        digit_x = [self._colourize(dx) for dx in digit_x]

        if self.op_reps is not None:
            op_idx = np.random.randint(len(self.op_reps))
            op_x, op_y = self.op_reps[op_idx]
            op_x = self._colourize(op_x)
            func = self._remapped_reductions[int(op_y)]
            patches = [op_x] + list(digit_x)
        else:
            func = self.func
            patches = list(digit_x)

        y = func(digit_y)
        y = min(y, self.largest_digit)

        return patches, digit_y, y


class GridArithmetic(VisualArithmetic, GridPatches):
    pass


class EmnistObjectDetection(Patches):
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

    @property
    def n_classes(self):
        return 1

    def _make(self):
        assert self.min_chars <= self.max_chars

        emnist_x, emnist_y, self.classmap = load_emnist(
            cfg.data_dir, self.characters, balance=True, shape=self.patch_shape,
            one_hot=False, n_examples=self.n_patch_examples,
            example_range=self.example_range)

        self.char_reps = list(zip(emnist_x, emnist_y))
        result = super(EmnistObjectDetection, self)._make()
        del self.char_reps

        return result

    def _sample_patches(self):
        n_chars = np.random.randint(self.min_chars, self.max_chars+1)

        if not n_chars:
            return [], [], 0

        indices = [np.random.randint(len(self.char_reps)) for i in range(n_chars)]
        chars = [self.char_reps[i] for i in indices]
        char_x, char_y = list(zip(*chars))
        char_x = [self._colourize(cx) for cx in char_x]

        return char_x, char_y, 0

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
                    (left, top), width, height, linewidth=1,
                    edgecolor='white', facecolor='none')

                ax.add_patch(rect)
        plt.show()


class GridEmnistObjectDetection(EmnistObjectDetection, GridPatches):
    pass
