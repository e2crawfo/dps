import dill
import gzip
import imageio
import numpy as np
import os
import shutil
import warnings

from dps import cfg
from dps.utils import image_to_string, resize_image


def background_names():
    backgrounds_dir = os.path.join(cfg.data_dir, 'backgrounds')
    return sorted(
        f.split('.')[0]
        for f in os.listdir(backgrounds_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    )


def hard_background_names():
    backgrounds_dir = os.path.join(cfg.data_dir, 'backgrounds')
    return sorted(
        f.split('.')[0]
        for f in os.listdir(backgrounds_dir)
        if f.endswith('.jpg')
    )


def load_backgrounds(background_names, shape=None):
    if isinstance(background_names, str):
        background_names = background_names.split()

    backgrounds_dir = os.path.join(cfg.data_dir, 'backgrounds')
    backgrounds = []
    for name in background_names:
        f = os.path.join(backgrounds_dir, '{}.jpg'.format(name))
        try:
            b = imageio.imread(f)
        except FileNotFoundError:
            f = os.path.join(backgrounds_dir, '{}.png'.format(name))
            b = imageio.imread(f)

        if shape is not None and b.shape != shape:
            b = resize_image(b, shape)
            b = np.uint8(b)

        backgrounds.append(b)
    return backgrounds


def emnist_classes():
    return (
        [str(i) for i in range(10)]
        + [chr(i + ord('A')) for i in range(26)]
        + [chr(i + ord('a')) for i in range(26)]
    )


emnist_filenames = [c + ".pklz" for c in emnist_classes()]


def _validate_emnist(path):
    path = str(path)
    if not os.path.isdir(path):
        return False
    return set(os.listdir(path)) == set(emnist_filenames)


def convert_emnist_and_store(path, new_image_shape):
    """ Images are stored on disk in float format. """
    if new_image_shape == (28, 28):
        raise Exception("Original shape of EMNIST is (28, 28).")

    print("Converting (28, 28) EMNIST dataset to {}...".format(new_image_shape))

    emnist_dir = os.path.join(path, 'emnist')
    new_dir = os.path.join(path, 'emnist_{}_by_{}'.format(*new_image_shape))
    try:
        shutil.rmtree(str(new_dir))
    except FileNotFoundError:
        pass

    os.mkdir(new_dir)

    classes = ''.join(
        [str(i) for i in range(10)]
        + [chr(i + ord('A')) for i in range(26)]
        + [chr(i + ord('a')) for i in range(26)]
    )

    for i, cls in enumerate(sorted(classes)):
        with gzip.open(os.path.join(emnist_dir, str(cls) + '.pklz'), 'rb') as f:
            _x = dill.load(f)

            new_x = []
            for img in _x:
                img = resize_image(img, new_image_shape, preserve_range=False)
                new_x.append(img)

            print(cls)
            print(image_to_string(_x[0]))
            _x = np.array(new_x, dtype=_x.dtype)
            print(image_to_string(_x[0]))

            path_i = os.path.join(new_dir, cls + '.pklz')
            with gzip.open(path_i, 'wb') as f:
                dill.dump(_x, f, protocol=dill.HIGHEST_PROTOCOL)


def load_emnist(
        path, classes, balance=False, include_blank=False,
        shape=None, n_examples=None, example_range=None, show=False):
    """ Load emnist data from disk by class.

    Elements of `classes` pick out which emnist classes to load, but different labels
    end up getting returned because most classifiers require that the labels
    be in range(len(classes)). We return a dictionary `class_map` which maps from
    elements of `classes` down to range(len(classes)).

    Pixel values of returned images are integers in the range 0-255, but stored as float32.
    Returned X array has shape (n_images,) + shape.

    Parameters
    ----------
    path: str
        Path to data directory, assumed to contain a sub-directory called `emnist`.
    classes: list of character from the set (0-9, A-Z, a-z)
        Each character is the name of a class to load.
    balance: bool
        If True, will ensure that all classes are balanced by removing elements
        from classes that are larger than the minimu-size class.
    include_blank: bool
        If True, includes an additional class that consists of blank images.
    shape: (int, int)
        Shape of the images.
    n_examples: int
        Maximum number of examples returned. If not supplied, return all available data.
    example_range: pair of floats
        Pair of floats specifying, for each class, the range of examples that should be used.
        Each element of the pair is a number in (0, 1), and the second number should be larger.
    show: bool
        If True, prints out an image from each class.

    """
    emnist_dir = os.path.join(path, 'emnist')

    classes = list(classes) + []

    needs_reshape = False
    if shape and shape != (28, 28):
        resized_dir = os.path.join(path, 'emnist_{}_by_{}'.format(*shape))

        if _validate_emnist(resized_dir):
            emnist_dir = resized_dir
        else:
            needs_reshape = True

    if example_range is not None:
        assert 0.0 <= example_range[0] < example_range[1] <= 1.0

    x, y = [], []
    class_count = []
    classes = sorted([str(s) for s in classes])

    for i, cls in enumerate(classes):
        with gzip.open(os.path.join(emnist_dir, str(cls) + '.pklz'), 'rb') as f:
            _x = dill.load(f)

        if example_range is not None:
            low = int(example_range[0] * len(_x))
            high = int(example_range[1] * len(_x))
            _x = _x[low:high, ...]

        x.append(np.uint8(_x))
        y.extend([i] * _x.shape[0])

        if show:
            print(cls)
            print(image_to_string(x[-1]))

        class_count.append(_x.shape[0])

    x = np.concatenate(x, axis=0)

    if include_blank:
        min_class_count = min(class_count)

        blanks = np.zeros((min_class_count,) + x.shape[1:], dtype=np.uint8)
        x = np.concatenate((x, blanks), axis=0)

        blank_idx = len(classes)

        y.extend([blank_idx] * min_class_count)

        blank_symbol = ' '
        classes.append(blank_symbol)

    y = np.array(y)

    if balance:
        min_class_count = min(class_count)

        keep_x, keep_y = [], []
        for i, cls in enumerate(classes):
            keep_indices = np.nonzero(y == i)[0]
            keep_indices = keep_indices[:min_class_count]
            keep_x.append(x[keep_indices])
            keep_y.append(y[keep_indices])

        x = np.concatenate(keep_x, axis=0)
        y = np.concatenate(keep_y, axis=0)

    order = np.random.permutation(x.shape[0])
    x = x[order]
    y = y[order]

    if n_examples:
        x = x[:n_examples]
        y = y[:n_examples]

    if needs_reshape:
        if x.shape[0] > 10000:
            warnings.warn(
                "Performing an online resize of a large number of images ({}), "
                "consider creating and storing the resized dataset.".format(x.shape[0])
            )

        x = [resize_image(img, shape) for img in x]
        x = np.uint8(x)

    return x, y, classes


def omniglot_classes():
    omniglot_dir = os.path.join(cfg.data_dir, 'omniglot')
    alphabets = os.listdir(omniglot_dir)
    classes = []
    for ab in alphabets:
        n_characters = len(os.listdir(os.path.join(omniglot_dir, ab)))
        classes.extend(["{},{}".format(ab, i+1) for i in range(n_characters)])
    return classes


# Class spec: alphabet,character
def load_omniglot(
        path, classes, include_blank=False, shape=None, one_hot=False, indices=None, show=False):
    """ Load omniglot data from disk by class.

    Elements of `classes` pick out which omniglot classes to load, but different labels
    end up getting returned because most classifiers require that the labels
    be in range(len(classes)). We return a dictionary `class_map` which maps from
    elements of `classes` down to range(len(classes)).

    Returned images are arrays of floats in the range 0-255. White text on black background
    (with 0 corresponding to black). Returned X array has shape (n_images,) + shape.

    Parameters
    ----------
    path: str
        Path to data directory, assumed to contain a sub-directory called `omniglot`.
    classes: list of strings, each giving a class label
        Each character is the name of a class to load.
    balance: bool
        If True, will ensure that all classes are balanced by removing elements
        from classes that are larger than the minimu-size class.
    include_blank: bool
        If True, includes an additional class that consists of blank images.
    shape: (int, int)
        Shape of returned images.
    one_hot: bool
        If True, labels are one-hot vectors instead of integers.
    indices: list of int
        The image indices within the classes to include. For each class there are 20 images.
    show: bool
        If True, prints out an image from each class.

    """
    omniglot_dir = os.path.join(path, 'omniglot')
    classes = list(classes)[:]

    if not indices:
        indices = list(range(20))

    for idx in indices:
        assert 0 <= idx < 20

    x, y = [], []
    class_map, class_count = {}, {}

    for i, cls in enumerate(sorted(list(classes))):
        alphabet, character = cls.split(',')
        char_dir = os.path.join(omniglot_dir, alphabet, "character{:02d}".format(int(character)))
        files = os.listdir(char_dir)
        class_id = files[0].split("_")[0]

        for idx in indices:
            f = os.path.join(char_dir, "{}_{:02d}.png".format(class_id, idx + 1))
            _x = imageio.imread(f)

            # Convert to white-on-black
            _x = 255. - _x

            if shape:
                _x = resize_image(_x, shape)

            x.append(_x)
            y.append(i)

        if show:
            print(cls)
            print(image_to_string(x[-1]))

        class_map[cls] = i
        class_count[cls] = len(indices)

    x = np.array(x, dtype=np.uint8)

    if include_blank:
        min_class_count = min(class_count.values())
        blanks = np.zeros((min_class_count,) + shape, dtype=np.uint8)
        x = np.concatenate((x, blanks), axis=0)

        blank_idx = len(class_map)

        y.extend([blank_idx] * min_class_count)

        blank_symbol = ' '
        class_map[blank_symbol] = blank_idx
        classes.append(blank_symbol)

    y = np.array(y)

    order = np.random.permutation(x.shape[0])
    x = x[order]
    y = y[order]

    if one_hot:
        _y = np.zeros((y.shape[0], len(classes))).astype('f')
        _y[np.arange(y.shape[0]), y] = 1.0
        y = _y

    return x, y, class_map
