import numpy as np
import os
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pprint
import shutil
import time
import abc
from itertools import zip_longest

from dps import cfg
from dps.utils import Param, Parameterized, get_param_hash, NumpySeed, animate, resize_image
from dps.datasets import (
    load_emnist, load_omniglot, omniglot_classes,
    load_backgrounds, background_names
)
from dps.datasets.parallel import make_dataset_in_parallel


class RawDataset(Parameterized):
    """ A non-tensorflow dataset, wrapper for data that we might want to cache. """
    seed = Param(None)

    def __init__(self, **kwargs):
        start = time.time()
        print("Trying to find dataset in cache...")

        directory = kwargs.get(
            "data_dir",
            os.path.join(cfg.data_dir, "cached_datasets", self.__class__.__name__))
        os.makedirs(directory, exist_ok=True)

        params = self.param_values()
        param_hash = get_param_hash(params)
        print(self.__class__.__name__)
        print("Params:")
        pprint.pprint(params)
        print("Param hash: {}".format(param_hash))

        self.directory = os.path.join(directory, str(param_hash))
        cfg_filename = os.path.join(self.directory, "config.txt")

        if not os.path.exists(cfg_filename):

            # Start fresh
            try:
                shutil.rmtree(self.directory)
            except FileNotFoundError:
                pass

            print("Directory for dataset not found, creating...")
            os.makedirs(self.directory, exist_ok=False)

            try:
                with NumpySeed(self.seed):
                    self._make()

                print("Done creating dataset.")
            except BaseException:
                try:
                    shutil.rmtree(self.directory)
                except FileNotFoundError:
                    pass
                raise

            with open(cfg_filename, 'w') as f:
                f.write(pprint.pformat(params))
        else:
            print("Found.")

        print("Took {} seconds.".format(time.time() - start))
        print("Features for dataset: ")
        pprint.pprint(self.features)
        print()

    def _make(self):
        """ Write data to `self.directory`. """
        raise Exception("AbstractMethod.")


class Feature(metaclass=abc.ABCMeta):
    """ Each Dataset class defines a set of features. Each feature defines 3 things:
        1. How it gets turned into a dictionary of tf.train.Features (get_write_features), used for
           storing data in a TFRecord format.
        2. How it gets turned into a dictionary of objects similar to tf.FixedLenFeature (get_read_features)
           used for unpacking the from the TFRecord format.
        3. How it gets turned into a dictionary of Tensors representing a batch (process_batch)

    """
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<{} - {}>".format(
            self.__class__.__name__,
            ", ".join("{}={}".format(k, v) for k, v in self.__dict__.items())
        )

    def __str__(self):
        return repr(self)

    @abc.abstractmethod
    def get_write_features(self, array):
        pass

    @abc.abstractmethod
    def get_read_features(self):
        pass

    @abc.abstractmethod
    def process_batch(self, data):
        pass


def _bytes_feature(value):
    if isinstance(value, np.ndarray):
        value = value.tostring()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value, is_list=False):
    if not is_list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value, is_list=False):
    if not is_list:
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class ArrayFeature(Feature):
    def __init__(self, name, shape, dtype=np.float32):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def get_write_features(self, array):
        array = np.array(array).astype(self.dtype)
        assert array.shape == self.shape, "{} vs {}".format(array.shape, self.shape)

        return {self.name: _bytes_feature(array)}

    def get_read_features(self):
        return {self.name: tf.FixedLenFeature((), dtype=tf.string)}

    def process_batch(self, records):
        data = tf.decode_raw(records[self.name], tf.as_dtype(self.dtype))
        return tf.reshape(data, (-1,) + self.shape)


class VariableShapeArrayFeature(Feature):
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.ndim = len(shape)

    def get_write_features(self, data):
        data = np.array(data)
        return {
            self.name + "/shape": _int64_feature(list(data.shape), is_list=True),
            self.name + "/data": _float_feature(list(data.flatten()), is_list=True),
        }

    def get_read_features(self):
        return {
            self.name + "/shape": tf.FixedLenFeature((self.ndim,), dtype=tf.int64),
            self.name + "/data": tf.VarLenFeature(dtype=tf.float32),
        }

    def process_batch(self, records):
        data = records[self.name + '/data']
        data = tf.sparse_tensor_to_dense(data, default_value=0)
        shapes = tf.cast(records[self.name + '/shape'], tf.int32)
        max_shape = tf.cast(tf.reduce_max(shapes, axis=0), tf.int32)

        max_shape_static = tuple(s if s >= 0 else ms for s, ms in zip(self.shape, tf.unstack(max_shape)))

        def map_fn(inp):
            data, shape = inp
            size = tf.reduce_prod(shape)
            data = data[:size]
            data = tf.reshape(data, shape)
            mask = tf.ones_like(data, dtype=tf.bool)

            pad_amount = tf.stack([tf.zeros_like(max_shape), max_shape - shape], axis=0)
            pad_amount = tf.transpose(pad_amount)

            data = tf.pad(data, pad_amount)
            data = tf.reshape(data, max_shape_static)

            mask = tf.pad(mask, pad_amount)
            mask = tf.reshape(mask, max_shape_static)

            return data, mask

        data, mask = tf.map_fn(map_fn, (data, shapes), dtype=(tf.float32, tf.bool))
        return dict(data=data, shapes=shapes, mask=mask)


class ImageFeature(ArrayFeature):
    """ Stores images on disk as uint8, converts them to float32 at runtime.

    Can also be used for video, use a shape with 4 entries, first entry being the number of frames.

    """
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.dtype = np.uint8

    def process_batch(self, records):
        images = super(ImageFeature, self).process_batch(records)
        images = tf.image.convert_image_dtype(images, tf.float32)
        return images


class IntegerFeature(Feature):
    """ If `maximum` is supplied, the integer is returned as a one-hot vector. """
    def __init__(self, name, maximum=None):
        self.name = name
        self.maximum = maximum

    def get_write_features(self, integer):
        return {self.name: _int64_feature(integer)}

    def get_read_features(self):
        return {self.name: tf.FixedLenFeature((), dtype=tf.int64)}

    def process_batch(self, records):
        integer = tf.cast(records[self.name], tf.int32)
        if self.maximum is not None:
            integer = tf.one_hot(integer, self.maximum)
        return integer


class FloatFeature(Feature):
    def __init__(self, name):
        self.name = name

    def get_write_features(self, f):
        return {self.name: _float_feature(f)}

    def get_read_features(self):
        return {self.name: tf.FixedLenFeature((), dtype=tf.float32)}

    def process_batch(self, records):
        f = tf.cast(records[self.name], tf.float32)
        return f


class NestedListFeature(Feature):
    def __init__(self, name, sublist_length):
        self.name = name
        self.sublist_length = sublist_length

    def get_write_features(self, nested_list):
        for sublist in nested_list:
            assert len(sublist) == self.sublist_length

        flat_list = [v for sublist in nested_list for v in sublist]

        return {
            self.name + "/n_sublists": _int64_feature(len(nested_list)),
            self.name + "/data": _float_feature(flat_list, is_list=True),
        }

    def get_read_features(self):
        return {
            self.name + "/n_sublists": tf.FixedLenFeature((), dtype=tf.int64),
            self.name + "/data": tf.VarLenFeature(dtype=tf.float32),
        }

    def process_batch(self, records):
        n_sublists = tf.cast(records[self.name + "/n_sublists"], tf.int32)
        max_n_sublists = tf.reduce_max(n_sublists)

        list_data = records[self.name + '/data']
        batch_size = tf.shape(list_data)[0]
        data = tf.sparse_tensor_to_dense(list_data, default_value=0)
        data = tf.reshape(data, (batch_size, max_n_sublists, self.sublist_length))
        return data, n_sublists


class Dataset(Parameterized):
    """ A parameterized dataset.

    Constructs a filename for caching by hashing a dictionary containing the parameter values (sorted by key).

    If `data_dir` is in kwargs, then look for (and save) the cache file inside `data_dir`.
    Otherwise, looks inside cfg.data_dir/cached_datasets/self.__class__.__name__.

    If `no_make` is in kwargs and is True, than raise an exception if dataset not found in cache.

    If `run_kwargs` is in kwargs, the corresponding value should be a dictionary of arguments which
    will be used to run the dataset creation in parallel.

    """
    n_examples = Param(None)
    seed = Param(None)

    _features = None
    _iterator = None
    _get_next = None

    def __init__(self, shuffle=True, **kwargs):
        start = time.time()
        print("Trying to find dataset in cache...")

        directory = kwargs.get(
            "data_dir",
            os.path.join(cfg.data_dir, "cached_datasets", self.__class__.__name__))
        os.makedirs(directory, exist_ok=True)

        params = self.param_values()
        param_hash = get_param_hash(params)
        print(self.__class__.__name__)
        print("Params:")
        pprint.pprint(params)
        print("Param hash: {}".format(param_hash))

        self.filename = os.path.join(directory, str(param_hash))
        cfg_filename = self.filename + ".cfg"

        no_cache = os.getenv("DPS_NO_CACHE")
        if no_cache:
            print("Skipping dataset cache as DPS_NO_CACHE is set (value is {}).".format(no_cache))

        # We require cfg_filename to exist as it marks that dataset creation completed successfully.
        if no_cache or not os.path.exists(self.filename) or not os.path.exists(cfg_filename):

            if kwargs.get("no_make", False):
                raise Exception("`no_make` is True, but dataset was not found in cache.")

            # Start fresh
            try:
                os.remove(self.filename)
            except FileNotFoundError:
                pass
            try:
                os.remove(cfg_filename)
            except FileNotFoundError:
                pass

            print("File for dataset not found, creating...")

            run_kwargs = kwargs.get('run_kwargs', None)
            if run_kwargs is not None:
                # Create the dataset in parallel and write it to the cache.
                make_dataset_in_parallel(run_kwargs, self.__class__, params)
            else:
                self._writer = tf.python_io.TFRecordWriter(self.filename)
                try:
                    with NumpySeed(self.seed):
                        self._make()
                    self._writer.close()
                    print("Done creating dataset.")
                except BaseException:
                    self._writer.close()

                    try:
                        os.remove(self.filename)
                    except FileNotFoundError:
                        pass
                    try:
                        os.remove(cfg_filename)
                    except FileNotFoundError:
                        pass

                    raise

            with open(cfg_filename, 'w') as f:
                f.write(pprint.pformat(params))
        else:
            print("Found.")

        print("Took {} seconds.".format(time.time() - start))
        print("Features for dataset: ")
        pprint.pprint(self.features)
        print()

    def _make(self):
        raise Exception("AbstractMethod.")

    @property
    def features(self):
        raise Exception("AbstractProperty")

    def _write_example(self, **kwargs):
        write_features = {}
        for f in self.features:
            write_features.update(f.get_write_features(kwargs[f.name]))
        example = tf.train.Example(features=tf.train.Features(feature=write_features))
        self._writer.write(example.SerializeToString())

    def parse_example_batch(self, example_proto):
        features = {}
        for f in self.features:
            features.update(f.get_read_features())
        data = tf.parse_example(example_proto, features=features)

        result = {}
        for f in self.features:
            result[f.name] = f.process_batch(data)

        result = self.parse_example_batch_postprocess(result)

        return result

    def parse_example_batch_postprocess(self, data):
        return data

    @property
    def iterator(self):
        if self._iterator is not None:
            return self._iterator

        dset = tf.data.TFRecordDataset(self.filename)
        dset = dset.repeat().batch(cfg.batch_size).map(self.parse_example_batch)

        self._iterator = dset.make_one_shot_iterator()
        return self._iterator

    @property
    def get_next(self):
        if self._get_next is not None:
            return self._get_next

        self._get_next = self.iterator.get_next()
        return self._get_next

    def next_batch(self):
        sess = tf.get_default_session()
        result = sess.run(self.get_next)
        return result

    def sample(self, n=4):
        batch_size = n
        dset = tf.data.TFRecordDataset(self.filename)
        dset = dset.batch(batch_size).map(self.parse_example_batch)

        iterator = dset.make_one_shot_iterator()

        sess = tf.get_default_session()

        _sample = sess.run(iterator.get_next())

        return _sample


class ImageClassificationDataset(Dataset):
    one_hot = Param()
    classes = Param()
    image_shape = Param()
    include_blank = Param()

    _features = None

    @property
    def features(self):
        if self._features is None:
            self._features = [
                ImageFeature("image", self.obs_shape),
                IntegerFeature("label", self.n_classes if self.one_hot else None),
            ]
        return self._features

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def obs_shape(self):
        return self.image_shape + (self.depth,)

    @property
    def action_shape(self):
        return self.n_classes if self.one_hot else 1

    @property
    def depth(self):
        return 1


class EmnistDataset(ImageClassificationDataset):
    """
    Download and pre-process EMNIST dataset:
    python scripts/download.py emnist <desired location>

    """
    balance = Param(False)
    example_range = Param(None)

    class_pool = ''.join(
        [str(i) for i in range(10)]
        + [chr(i + ord('A')) for i in range(26)]
        + [chr(i + ord('a')) for i in range(26)]
    )

    @staticmethod
    def sample_classes(n_classes):
        classes = np.random.choice(len(EmnistDataset.class_pool), n_classes, replace=False)
        return [EmnistDataset.class_pool[i] for i in classes]

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
            self._write_example(image=_x, label=class_map[_y])


class OmniglotDataset(ImageClassificationDataset):
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
            self._write_example(image=_x, label=class_map[_y])


class ImageDataset(Dataset):
    image_shape = Param((100, 100))
    postprocessing = Param("")
    tile_shape = Param(None)
    n_samples_per_image = Param(1)
    n_frames = Param(0)

    @property
    def obs_shape(self):
        leading_shape = (self.n_frames,) if self.n_frames > 0 else ()
        if self.postprocessing:
            return leading_shape + self.tile_shape + (self.depth,)
        else:
            return leading_shape + self.image_shape + (self.depth,)

    def _write_single_example(self, **kwargs):
        return Dataset._write_example(self, **kwargs)

    def _write_example(self, **kwargs):
        image = kwargs['image']
        annotation = kwargs.get("annotations", [])
        label = kwargs.get("label", None)
        background = kwargs.get("background", None)

        if self.postprocessing and self.n_frames > 0:
            raise Exception("NotImplemented")

        if self.postprocessing == "tile":
            images, annotations, backgrounds = self._tile_postprocess(image, annotation, background=background)
        elif self.postprocessing == "random":
            images, annotations, backgrounds = self._random_postprocess(image, annotation, background=background)
        else:
            images, annotations, backgrounds = [image], [annotation], [background]

        for img, a, bg in zip_longest(images, annotations, backgrounds):
            self._write_single_example(image=img, annotations=a, label=label, background=bg)

    @staticmethod
    def tile_sample(image, tile_shape):
        height, width, n_channels = image.shape

        hangover = width % tile_shape[1]
        if hangover != 0:
            pad_amount = tile_shape[1] - hangover
            pad_shape = (height, pad_amount)
            padding = np.zeros(pad_shape)
            image = np.concat([image, padding], axis=2)

        hangover = height % tile_shape[0]
        if hangover != 0:
            pad_amount = tile_shape[0] - hangover
            pad_shape = list(image.shape)
            pad_shape[1] = pad_amount
            padding = np.zeros(pad_shape)
            image = np.concat([image, padding], axis=1)

        pad_height = tile_shape[0] - height % tile_shape[0]
        pad_width = tile_shape[1] - width % tile_shape[1]
        image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

        H = int(height / tile_shape[0])
        W = int(width / tile_shape[1])

        slices = np.split(image, W, axis=1)
        new_shape = (H, *tile_shape, n_channels)
        slices = [np.reshape(s, new_shape) for s in slices]
        new_images = np.concatenate(slices, axis=1)
        new_images = new_images.reshape(H * W, *tile_shape, n_channels)
        return new_images

    def _tile_postprocess(self, image, annotations, background=None):
        new_images = self.tile_sample(image, self.tile_shape)
        new_annotations = []

        H = int(image.shape[0] / self.tile_shape[0])
        W = int(image.shape[1] / self.tile_shape[1])

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

        new_backgrounds = []
        if background is not None:
            new_backgrounds = self.tile_sample(background, self.tile_shape)

        return new_images, new_annotations, new_backgrounds

    def _random_postprocess(self, image, annotations, background=None):
        height, width, _ = image.shape
        new_images = []
        new_annotations = []
        new_backgrounds = []

        for j in range(self.n_samples_per_image):
            _top = np.random.randint(0, height-self.tile_shape[0]+1)
            _left = np.random.randint(0, width-self.tile_shape[1]+1)

            crop = image[_top:_top+self.tile_shape[0], _left:_left+self.tile_shape[1], ...]
            new_images.append(crop)

            if background is not None:
                bg_crop = background[_top:_top+self.tile_shape[0], _left:_left+self.tile_shape[1], ...]
                new_backgrounds.append(bg_crop)

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

        return new_images, new_annotations, new_backgrounds

    def visualize(self, n=4):
        sample = self.sample(n)
        images = sample["image"]
        # annotations, annotations_shape, annotations_mask = sample["annotations"]
        labels = sample["label"]

        if self.n_frames == 0:
            images = images[:, None]

        fig, *_ = animate(images, labels=labels)

        plt.show()
        plt.close(fig)


class Rectangle(object):
    def __init__(self, top, left, h, w, v=None):
        self.top = top
        self.left = left

        self.h = h
        self.w = w

        self.v = v or np.zeros(2)

    @property
    def bottom(self):
        return self.top + self.h

    @property
    def right(self):
        return self.left + self.w

    def move(self, movement):
        self.top += movement[0]
        self.left += movement[1]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Rectangle({}:{}, {}:{})".format(self.top, self.bottom, self.left, self.right)

    def intersects(self, r2):
        return self.overlap_area(r2) > 0

    def overlap_area(self, r2):
        overlap_bottom = np.minimum(self.bottom, r2.bottom)
        overlap_top = np.maximum(self.top, r2.top)

        overlap_right = np.minimum(self.right, r2.right)
        overlap_left = np.maximum(self.left, r2.left)

        area = np.maximum(overlap_bottom - overlap_top, 0) * np.maximum(overlap_right - overlap_left, 0)
        return area

    def centre(self):
        return (
            self.top + (self.bottom - self.top) / 2.,
            self.left + (self.right - self.left) / 2.
        )

    def update(self, shape):
        """
        For each of the 4 corners, create a line segment imposed by the movement.
        Find the earliest "time" in the movement that one of the corners intersects one of the walls.
        (If there is no such intersection, we're done)
        move all corners based on that time point (so the object is against one of the walls),
        and flip the velocity vector appropriately.
        Then repeat this for the "remainder" of the movement.

        Return the new position and new velocity.

        Could use 2 opposing corners instead of 4.
        Actually we only have to use the corner that is in the direction of the movement.

        And the direction of the movement determines which two walls can be intersected.

        """
        velocity = self.v.copy()
        while True:
            if velocity[0] > 0:
                v_distance_to_wall = shape[0] - self.bottom
            else:
                v_distance_to_wall = self.top
            v_t = v_distance_to_wall / np.abs(velocity[0])

            if velocity[1] > 0:
                h_distance_to_wall = shape[1] - self.right
            else:
                h_distance_to_wall = self.left
            h_t = h_distance_to_wall / np.abs(velocity[1])

            if v_t > 1 and h_t > 1:
                self.move(velocity)
                break
            elif v_t < h_t:
                self.move(v_t * velocity)
                velocity = (1 - v_t) * np.array([-velocity[0], velocity[1]])
            else:
                self.move(h_t * velocity)
                velocity = (1 - h_t) * np.array([velocity[0], -velocity[1]])

        self.v = velocity * np.linalg.norm(self.v) / np.linalg.norm(velocity)


class PatchesDataset(ImageDataset):
    max_overlap = Param()
    draw_shape = Param(None)
    draw_offset = Param((0, 0))
    patch_size_std = Param(None)
    distractor_shape = Param((3, 3))
    n_distractors_per_image = Param(0)
    backgrounds = Param(
        "", help="Can be either be 'all', in which a random background will be selected for "
                 "each constructed image, or a list of strings, giving the names of backgrounds "
                 "to use.")
    backgrounds_resize = Param(False)
    background_colours = Param("")
    max_attempts = Param(10000)
    colours = Param('red green blue')
    one_hot = Param(True)
    patch_speed = Param(10, help="In pixels per frame.")

    _features = None

    @property
    def features(self):
        if self._features is None:
            annotation_shape = (self.n_frames, -1, 6) if self.n_frames > 0 else (-1, 6)
            self._features = [
                ImageFeature("image", self.obs_shape),
                VariableShapeArrayFeature("annotations", annotation_shape),
                IntegerFeature("label", self.n_classes if self.one_hot else None)]

        return self._features

    @property
    def n_classes(self):
        raise Exception("AbstractMethod")

    @property
    def depth(self):
        return 3 if self.colours else 1

    def _make(self):
        """
        To handle both images and videos:

        * for each example, sample patch locations as well as velocities
        *    want to make it so they they don't have to return velocities. can use *rest
        *    in case velocities not return, use a velocity of 0.
        * go through all frames for the example, using an identical process to render each frame
        * if n_frames == 0, remove the frame dimension, so they really are just images.
        * assume a fixed background for the entire video, for now.
        """
        if self.n_examples == 0:
            return np.zeros((0,) + self.image_shape).astype('uint8'), np.zeros((0, 1)).astype('i')

        # --- prepare colours ---

        colours = self.colours
        if colours is None:
            colours = []
        if isinstance(colours, str):
            colours = colours.split()

        colour_map = mpl.colors.get_named_colors_mapping()

        self._colours = []
        for c in colours:
            c = mpl.colors.to_rgb(colour_map[c])
            c = np.array(c)[None, None, :]
            c = np.uint8(255. * c)
            self._colours.append(c)

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

        background_colours = self.background_colours
        if isinstance(self.background_colours, str):
            background_colours = background_colours.split()
        _background_colours = []
        for bc in background_colours:
            color = mpl.colors.to_rgb(bc)
            color = np.array(color)[None, None, :]
            color = np.uint8(255. * color)
            _background_colours.append(color)
        background_colours = _background_colours

        # --- start dataset creation ---

        for j in range(int(self.n_examples)):
            if j % 1000 == 0:
                print("Working on datapoint {}...".format(j))

            # --- populate background ---

            if backgrounds:
                b_idx = np.random.randint(len(backgrounds))
                background = backgrounds[b_idx]
                top = np.random.randint(background.shape[0] - draw_shape[0] + 1)
                left = np.random.randint(background.shape[1] - draw_shape[1] + 1)
                base_image = background[top:top+draw_shape[0], left:left+draw_shape[1], ...] + 0

            elif background_colours:
                color = background_colours[np.random.randint(len(background_colours))]
                base_image = color * np.ones(draw_shape, 'uint8')

            else:
                base_image = np.zeros(draw_shape, 'uint8')

            # --- sample and populate patches ---

            locs, patches, patch_labels, image_label = self._sample_image()

            draw_offset = self.draw_offset

            images = []
            annotations = []
            for frame in range(max(self.n_frames, 1)):
                image = base_image.copy()

                for patch, loc in zip(patches, locs):
                    if patch.shape[:2] != (loc.h, loc.w):
                        patch = resize_image(patch, (loc.h, loc.w))

                    top = int(np.clip(loc.top, 0, image.shape[0]))
                    bottom = int(np.clip(loc.bottom, 0, image.shape[0]))
                    left = int(np.clip(loc.left, 0, image.shape[1]))
                    right = int(np.clip(loc.right, 0, image.shape[1]))

                    patch_top = top - int(loc.top)
                    patch_bottom = bottom - int(loc.top)
                    patch_left = left - int(loc.left)
                    patch_right = right - int(loc.left)

                    intensity = patch[patch_top:patch_bottom, patch_left:patch_right, :-1]
                    alpha = patch[patch_top:patch_bottom, patch_left:patch_right, -1:].astype('f') / 255.

                    current = image[top:bottom, left:right, ...]
                    image[top:bottom, left:right, ...] = np.uint8(alpha * intensity + (1 - alpha) * current)

                # --- add distractors ---

                if self.n_distractors_per_image > 0:
                    distractor_patches = self._sample_distractors()
                    distractor_shapes = [img.shape for img in distractor_patches]
                    distractor_locs = self._sample_patch_locations(distractor_shapes)

                    for patch, loc in zip(distractor_patches, distractor_locs):
                        if patch.shape[:2] != (loc.h, loc.w):
                            anti_aliasing = patch.shape[0] < loc.h or patch.shape[1] < loc.w
                            patch = resize_image(patch, (loc.h, loc.w))

                        intensity = patch[:, :, :-1]
                        alpha = patch[:, :, -1:].astype('f') / 255.

                        current = image[loc.top:loc.bottom, loc.left:loc.right, ...]
                        image[loc.top:loc.bottom, loc.left:loc.right, ...] = (
                            np.uint8(alpha * intensity + (1 - alpha) * current))

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

                _annotations = self._get_annotations(draw_offset, patches, locs, patch_labels)

                images.append(image)
                annotations.append(_annotations)

                for loc in locs:
                    loc.update(image.shape)

            if self.n_frames == 0:
                images = images[0]
                annotations = annotations[0]

            self._write_example(image=images, annotations=annotations, label=image_label)

    def _get_annotations(self, draw_offset, patches, locs, labels):
        new_labels = []
        for patch, loc, label in zip(patches, locs, labels):
            nz_y, nz_x = np.nonzero(patch[:, :, -1])

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

            valid = not ((bottom - top < 1e-6) or (right - left < 1e-6))

            new_labels.append((valid, label, top, bottom, left, right))

        return new_labels

    def _sample_image(self):
        patches, patch_labels, image_label = self._sample_patches()
        patch_shapes = np.array([img.shape for img in patches])

        locs = self._sample_patch_locations(
            patch_shapes,
            max_overlap=self.max_overlap,
            size_std=self.patch_size_std)

        velocity = np.random.randn(len(locs), 2)
        velocity /= np.maximum(np.linalg.norm(velocity, keepdims=True), 1e-6)
        velocity *= self.patch_speed

        for loc, v in zip(locs, velocity):
            loc.v = v

        return locs, patches, patch_labels, image_label

    def _sample_patches(self):
        raise Exception("AbstractMethod")

    def _sample_patch_locations(self, patch_shapes, max_overlap=None, size_std=None):
        """ Sample random locations within draw_shape. """
        if len(patch_shapes) == 0:
            return []

        patch_shapes = np.array(patch_shapes)
        n_rects = patch_shapes.shape[0]

        n_tries_outer = 0
        while True:
            rects = []
            for i in range(n_rects):
                n_tries_inner = 0
                while True:
                    if size_std is None:
                        shape_multipliers = 1.
                    else:
                        shape_multipliers = np.maximum(np.random.randn(2) * size_std + 1.0, 0.5)

                    m, n = np.ceil(shape_multipliers * patch_shapes[i, :2]).astype('i')

                    rect = Rectangle(
                        np.random.randint(0, self.draw_shape[0]-m+1),
                        np.random.randint(0, self.draw_shape[1]-n+1), m, n)

                    if max_overlap is None:
                        rects.append(rect)
                        break
                    else:
                        overlap_area = 0
                        violation = False
                        for r in rects:
                            overlap_area += rect.overlap_area(r)
                            if overlap_area > max_overlap:
                                violation = True
                                break

                        if not violation:
                            rects.append(rect)
                            break

                    n_tries_inner += 1

                    if n_tries_inner > self.max_attempts/10:
                        break

                if len(rects) < i + 1:  # No rectangle found
                    break

            if len(rects) == n_rects:
                break

            n_tries_outer += 1

            if n_tries_outer > self.max_attempts:
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

    def _colourize(self, img, colour=None):
        """ Apply a colour to a gray-scale image. """

        if isinstance(colour, str):
            colour = mpl.colors.to_rgb(colour)
            colour = np.array(colour)[None, None, :]
            colour = np.uint8(255. * colour)
        else:
            if colour is None:
                colour = np.random.randint(len(self._colours))
            colour = self._colours[int(colour)]

        rgb = np.tile(colour, img.shape + (1,))
        alpha = img[:, :, None]

        return np.concatenate([rgb, alpha], axis=2).astype(np.uint8)


class GridPatchesDataset(PatchesDataset):
    grid_shape = Param((2, 2))
    spacing = Param((0, 0))
    random_offset_range = Param(None)

    def _make(self):
        self.grid_size = np.product(self.grid_shape)
        self.cell_shape = (
            self.patch_shape[0] + self.spacing[0],
            self.patch_shape[1] + self.spacing[1])
        return super(GridPatchesDataset, self)._make()

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


class VisualArithmeticDataset(PatchesDataset):
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

        result = super(VisualArithmeticDataset, self)._make()

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


class GridArithmeticDataset(VisualArithmeticDataset, GridPatchesDataset):
    pass


class EmnistObjectDetectionDataset(PatchesDataset):
    min_chars = Param(2)
    max_chars = Param(3)
    characters = Param(
        [str(i) for i in range(10)]
        + [chr(i + ord('A')) for i in range(26)]
        + [chr(i + ord('a')) for i in range(26)]
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
        result = super(EmnistObjectDetectionDataset, self)._make()
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
        m = int(np.ceil(np.sqrt(n)))
        fig, subplots = plt.subplots(m, m)

        height = self.x.shape[1]

        for i, ax in enumerate(subplots.flatten()):
            ax.imshow(self.x[i, ...])
            for cls, top, bottom, left, right in self.y[i]:
                width = right - left
                height = bottom - top

                rect = mpl.patches.Rectangle(
                    (left, top), width, height, linewidth=1,
                    edgecolor='white', facecolor='none')

                ax.add_patch(rect)
        plt.show()


class GridEmnistObjectDetectionDataset(EmnistObjectDetectionDataset, GridPatchesDataset):
    pass


if __name__ == "__main__":
    dset = VisualArithmeticDataset(
        n_examples=18, reductions="sum", largest_digit=28,
        min_digits=9, max_digits=9, image_shape=(48, 48),
        max_overlap=98, colours="white blue", n_frames=10)

    sess = tf.Session()
    with sess.as_default():
        dset.visualize()
