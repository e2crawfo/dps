import shutil
import numpy as np
import dill
import gzip
import os
import subprocess
import struct
from array import array
import warnings

from dps import cfg
from dps.utils import image_to_string, cd, resize_image


# This link seems not to work anymore...
# emnist_url = 'https://cloudstor.aarnet.edu.au/plus/index.php/s/54h3OuGJhFLwAlQ/download'

emnist_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'


template = 'emnist-byclass-{}-{}-idx{}-ubyte.gz'

emnist_gz_names = [
    template.format('test', 'images', 3),
    template.format('test', 'labels', 1),
    template.format('train', 'images', 3),
    template.format('train', 'labels', 1)
]


def emnist_classes():
    return (
        [str(i) for i in range(10)]
        + [chr(i + ord('A')) for i in range(26)]
        + [chr(i + ord('a')) for i in range(26)]
    )


emnist_filenames = [c + ".pklz" for c in emnist_classes()]


def _validate_emnist(path):
    if not os.path.isdir(path):
        return False
    return set(os.listdir(path)) == set(emnist_filenames)


def _download_emnist(data_dir):
    """
    Download the emnist data. Result is that a directory called "emnist_raw"
    is created inside `data_dir` which contains 4 files.

    Parameters
    ----------
    path: str
        Path to directory where files should be stored.

    """
    emnist_raw_dir = os.path.join(data_dir, "emnist_raw")
    os.makedirs(emnist_raw_dir, exist_ok=True)

    with cd(emnist_raw_dir):
        if not os.path.exists('gzip.zip'):
            print("Downloading...")
            command = "wget --output-document=gzip.zip {}".format(emnist_url).split()
            subprocess.run(command, check=True)
        else:
            print("Found existing copy of gzip.zip, not downloading.")

        print("Extracting...")
        for fname in emnist_gz_names:
            if not os.path.exists(fname):
                subprocess.run('unzip gzip.zip gzip/{}'.format(fname), shell=True, check=True)
                shutil.move('gzip/{}'.format(fname), '.')
            else:
                print("{} already exists, skipping extraction.".format(fname))

        try:
            shutil.rmtree('gzip')
        except FileNotFoundError:
            pass

    return emnist_raw_dir


def _emnist_load_helper(path_img, path_lbl):
    with gzip.open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic))

        labels = array("B", file.read())

    with gzip.open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic))

        image_data = array("B", file.read())

    images = np.zeros((size, rows * cols), dtype=np.uint8)

    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.uint8)


def maybe_convert_emnist_shape(path, shape):
    """ Create a version of emnist on disk that is reshaped to the desired shape.

        Images are stored on disk as uint8.

    """
    if shape == (28, 28):
        return

    shape_dir = os.path.join(path, 'emnist_{}_by_{}'.format(*shape))

    if os.path.isdir(shape_dir):
        return

    emnist_dir = os.path.join(path, 'emnist')

    print("Converting (28, 28) EMNIST dataset to {}...".format(shape))

    try:
        shutil.rmtree(shape_dir)
    except FileNotFoundError:
        pass

    os.makedirs(shape_dir, exist_ok=False)

    classes = ''.join(
        [str(i) for i in range(10)]
        + [chr(i + ord('A')) for i in range(26)]
        + [chr(i + ord('a')) for i in range(26)]
    )

    for i, cls in enumerate(sorted(classes)):
        with gzip.open(os.path.join(emnist_dir, str(cls) + '.pklz'), 'rb') as f:
            _x = dill.load(f)

            new_x = []
            for img in _x[:10]:
                img = resize_image(img, shape, preserve_range=True)
                new_x.append(img)

            print(cls)
            print(image_to_string(_x[0]))
            _x = np.array(new_x, dtype=_x.dtype)
            print(image_to_string(_x[0]))

            path_i = os.path.join(shape_dir, cls + '.pklz')
            with gzip.open(path_i, 'wb') as f:
                dill.dump(_x, f, protocol=dill.HIGHEST_PROTOCOL)


def maybe_download_emnist(data_dir, quiet=0, shape=None):
    """
    Download emnist data if it hasn't already been downloaded. Do some
    post-processing to put it in a more useful format. End result is a directory
    called `emnist-byclass` which contains a separate pklz file for each emnist
    class.

    Pixel values of stored images are uint8 values up to 255.
    Images for each class are put into a numpy array with shape (n_images_in_class, 28, 28).
    This numpy array is pickled and stored in a zip file with name <class char>.pklz.

    Parameters
    ----------
    data_dir: str
         Directory where files should be stored.

    """
    emnist_dir = os.path.join(data_dir, 'emnist')

    if _validate_emnist(emnist_dir):
        print("EMNIST data seems to be present already.")
    else:
        print("EMNIST data not found, downloading and processing...")
        try:
            shutil.rmtree(emnist_dir)
        except FileNotFoundError:
            pass

        raw_dir = _download_emnist(data_dir)

        with cd(raw_dir):
            images, labels = _emnist_load_helper(emnist_gz_names[0], emnist_gz_names[1])
            images1, labels1 = _emnist_load_helper(emnist_gz_names[2], emnist_gz_names[3])

        with cd(data_dir):
            os.makedirs('emnist', exist_ok=False)

            print("Processing...")
            with cd('emnist'):
                x = np.concatenate((images, images1), 0)
                y = np.concatenate((labels, labels1), 0)

                # Give images the right orientation so that plt.imshow(x[0]) just works.
                x = np.moveaxis(x.reshape(-1, 28, 28), 1, 2)

                for i in sorted(set(y.flatten())):
                    keep = y == i
                    x_i = x[keep.flatten(), :]
                    if i >= 36:
                        char = chr(i-36+ord('a'))
                    elif i >= 10:
                        char = chr(i-10+ord('A'))
                    else:
                        char = str(i)

                    if quiet >= 2:
                        pass
                    elif quiet == 1:
                        print(char)
                    elif quiet <= 0:
                        print(char)
                        print(image_to_string(x_i[0, ...]))

                    file_i = char + '.pklz'
                    with gzip.open(file_i, 'wb') as f:
                        dill.dump(x_i, f, protocol=dill.HIGHEST_PROTOCOL)

    if shape is not None:
        maybe_convert_emnist_shape(data_dir, shape)


def load_emnist(
        classes, balance=False, include_blank=False,
        shape=None, n_examples=None, example_range=None, show=False, path=None):
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
    if path is None:
        path = cfg.data_dir

    maybe_download_emnist(path, shape=shape)

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

        x.append(_x)
        y.extend([i] * _x.shape[0])

        if show:
            print(cls)
            indices_to_show = np.random.choice(len(_x), size=100)
            for i in indices_to_show:
                print(image_to_string(_x[i]))

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

    if show:
        indices_to_show = np.random.choice(len(x), size=200)
        for i in indices_to_show:
            print(y[i])
            print(image_to_string(x[i]))

    return x, y, classes
