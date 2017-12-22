import dill
import gzip
import scipy
from skimage.transform import resize
import numpy as np
import os
import shutil
import warnings
from pathlib import Path

from dps import cfg
from dps.utils import image_to_string, cd


def emnist_classes():
    return (
        [str(i) for i in range(10)] +
        [chr(i + ord('A')) for i in range(26)] +
        [chr(i + ord('a')) for i in range(26)]
    )


emnist_filenames = [c + ".pklz" for c in emnist_classes()]


def _validate_emnist(path):
    path = str(path)
    if not os.path.isdir(path):
        return False

    with cd(path):
        return set(os.listdir(path)) == set(emnist_filenames)


def convert_emnist_and_store(path, new_image_shape):
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
        [str(i) for i in range(10)] +
        [chr(i + ord('A')) for i in range(26)] +
        [chr(i + ord('a')) for i in range(26)]
    )

    for i, cls in enumerate(sorted(classes)):
        with gzip.open(os.path.join(emnist_dir, str(cls) + '.pklz'), 'rb') as f:
            _x = dill.load(f)

            new_x = []
            for img in _x:
                img = resize(img, new_image_shape, mode='edge')
                new_x.append(img)

            print(cls)
            print(image_to_string(_x[0]))
            _x = np.array(new_x)
            print(image_to_string(_x[0]))

            path_i = os.path.join(new_dir, cls + '.pklz')
            with gzip.open(path_i, 'wb') as f:
                dill.dump(_x, f, protocol=dill.HIGHEST_PROTOCOL)


def load_emnist(
        path, classes, balance=False, include_blank=False,
        shape=None, one_hot=False, n_examples=None, show=False):
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
    balance: boolean
        If True, will ensure that all classes are balanced by removing elements
        from classes that are larger than the minimu-size class.
    include_blank: boolean
        If True, includes an additional class that consists of blank images.
    shape: (int, int)
        Shape of the images.
    one_hot: bool
        If True, labels are one-hot vectors instead of integers.
    n_examples: int
        Maximum number of examples returned. If not supplied, return all available data.
    show: bool
        If True, prints out an image from each class.

    """
    emnist_dir = os.path.join(path, 'emnist')

    needs_reshape = False
    if shape and shape != (28, 28):
        resized_dir = os.path.join(path, 'emnist_{}_by_{}'.format(*shape))

        if _validate_emnist(resized_dir):
            emnist_dir = resized_dir
        else:
            needs_reshape = True

    classes = list(classes)[:]
    y = []
    x = []
    class_map = {}
    for i, cls in enumerate(sorted(list(classes))):
        with gzip.open(os.path.join(emnist_dir, str(cls) + '.pklz'), 'rb') as f:
            _x = dill.load(f)
            x.append(np.float32(np.uint8(255*np.minimum(_x, 1))))
            y.extend([i] * x[-1].shape[0])
        if show:
            print(cls)
            print(image_to_string(x[-1]))
        class_map[cls] = i
    x = np.concatenate(x, axis=0)
    y = np.array(y).reshape(-1, 1)

    if include_blank:
        class_count = min([(y == class_map[c]).sum() for c in classes])
        blanks = np.zeros((class_count,) + x.shape[1:])
        x = np.concatenate((x, blanks), axis=0)
        blank_idx = len(class_map)
        y = np.concatenate((y, blank_idx * np.ones((class_count, 1), dtype=y.dtype)), axis=0)
        blank_symbol = ' '
        class_map[blank_symbol] = blank_idx
        classes.append(blank_symbol)

    if balance:
        class_count = min([(y == class_map[c]).sum() for c in classes])
        keep_x, keep_y = [], []
        for i, cls in enumerate(classes):
            keep_indices, _ = np.nonzero(y == class_map[cls])
            keep_indices = keep_indices[:class_count]
            keep_x.append(x[keep_indices, :])
            keep_y.append(y[keep_indices, :])
        x = np.concatenate(keep_x, 0)
        y = np.concatenate(keep_y, 0)

    order = np.random.permutation(x.shape[0])

    x = x[order, :]
    y = y[order, :]

    if n_examples is not None:
        x = x[:n_examples]
        y = y[:n_examples]

    if one_hot:
        _y = np.zeros((y.shape[0], len(classes))).astype('f')
        _y[np.arange(y.shape[0]), y.flatten()] = 1.0
        y = _y

    if needs_reshape:
        if x.shape[0] > 10000:
            warnings.warn(
                "Performing an online resize of a large number of images ({}), "
                "consider creating and storing the resized dataset.".format(x.shape[0])
            )

        x = [resize(img, shape, mode='edge') for img in np.uint8(x)]
        x = np.float32(np.uint8(255*np.minimum(x, 1)))

    return x, y, class_map


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
    balance: boolean
        If True, will ensure that all classes are balanced by removing elements
        from classes that are larger than the minimu-size class.
    include_blank: boolean
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
    y = []
    x = []
    class_map = {}
    for i, cls in enumerate(sorted(list(classes))):
        alphabet, character = cls.split(',')
        char_dir = os.path.join(omniglot_dir, alphabet, "character{:02d}".format(int(character)))
        files = os.listdir(char_dir)
        class_id = files[0].split("_")[0]

        for idx in indices:
            f = os.path.join(char_dir, "{}_{:02d}.png".format(class_id, idx + 1))
            _x = scipy.misc.imread(f)
            _x = 255. - _x
            if shape:
                _x = resize(_x, shape, mode='edge')

            x.append(np.float32(_x))
            y.append(i)
        if show:
            print(cls)
            print(image_to_string(x[-1]))
        class_map[cls] = i

    x = np.array(x)
    y = np.array(y).reshape(-1, 1)

    if include_blank:
        class_count = min([(y == class_map[c]).sum() for c in classes])
        blanks = np.zeros((class_count,) + shape)
        x = np.concatenate((x, blanks), axis=0)
        blank_idx = len(class_map)
        y = np.concatenate((y, blank_idx * np.ones((class_count, 1), dtype=y.dtype)), axis=0)
        blank_symbol = ' '
        class_map[blank_symbol] = blank_idx
        classes.append(blank_symbol)

    order = np.random.permutation(x.shape[0])

    x = x[order, :]
    y = y[order, :]

    if one_hot:
        _y = np.zeros((y.shape[0], len(classes))).astype('f')
        _y[np.arange(y.shape[0]), y.flatten()] = 1.0
        y = _y

    return x, y, class_map
