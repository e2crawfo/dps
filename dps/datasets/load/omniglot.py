import imageio
import shutil
import numpy as np
import os
import subprocess
import zipfile

from dps import cfg
from dps.utils import image_to_string, cd, process_path, resize_image


def omniglot_classes():
    omniglot_dir = os.path.join(cfg.data_dir, 'omniglot')
    alphabets = os.listdir(omniglot_dir)
    classes = []
    for ab in alphabets:
        n_characters = len(os.listdir(os.path.join(omniglot_dir, ab)))
        classes.extend(["{},{}".format(ab, i+1) for i in range(n_characters)])
    return classes


omniglot_alphabets = [
    'Bengali',
    'Balinese',
    'Arcadian',
    'Tibetan',
    'Braille',
    'Burmese_(Myanmar)',
    'Malay_(Jawi_-_Arabic)',
    'Cyrillic',
    'Hebrew',
    'Atemayar_Qelisayer',
    'Latin',
    'Mkhedruli_(Georgian)',
    'Mongolian',
    'Asomtavruli_(Georgian)',
    'Oriya',
    'Kannada',
    'Futurama',
    'Inuktitut_(Canadian_Aboriginal_Syllabics)',
    'Tifinagh',
    'Old_Church_Slavonic_(Cyrillic)',
    'Grantha',
    'Greek',
    'N_Ko',
    'Blackfoot_(Canadian_Aboriginal_Syllabics)',
    'Malayalam',
    'Manipuri',
    'Gujarati',
    'Anglo-Saxon_Futhorc',
    'Korean',
    'Sanskrit',
    'Sylheti',
    'Aurek-Besh',
    'Ojibwe_(Canadian_Aboriginal_Syllabics)',
    'Gurmukhi',
    'Avesta',
    'Angelic',
    'Japanese_(katakana)',
    'Japanese_(hiragana)',
    'ULOG',
    'Early_Aramaic',
    'Tagalog',
    'Glagolitic',
    'Syriac_(Serto)',
    'Alphabet_of_the_Magi',
    'Armenian',
    'Syriac_(Estrangelo)',
    'Keble',
    'Atlantean',
    'Ge_ez',
    'Tengwar'
]


def _validate_omniglot(path):
    if not os.path.isdir(path):
        return False

    with cd(path):
        return set(os.listdir(path)) == set(omniglot_alphabets)


def process_omniglot(data_dir, quiet):
    try:
        omniglot_dir = process_path(os.path.join(data_dir, 'omniglot'))

        if _validate_omniglot(omniglot_dir):
            print("Omniglot data seems to be present already.")
            return
        else:
            try:
                shutil.rmtree(omniglot_dir)
            except FileNotFoundError:
                pass

        os.makedirs(omniglot_dir, exist_ok=False)

        with cd(omniglot_dir):
            subprocess.run("git clone https://github.com/brendenlake/omniglot --depth=1".split(), check=True)

            with cd('omniglot/python'):
                zip_ref = zipfile.ZipFile('images_evaluation.zip', 'r')
                zip_ref.extractall('.')
                zip_ref.close()

                zip_ref = zipfile.ZipFile('images_background.zip', 'r')
                zip_ref.extractall('.')
                zip_ref.close()

            subprocess.run('mv omniglot/python/images_background/* .', shell=True, check=True)
            subprocess.run('mv omniglot/python/images_evaluation/* .', shell=True, check=True)
        print("Done setting up Omniglot data.")
    finally:
        try:
            shutil.rmtree(os.path.join(omniglot_dir, 'omniglot'))
        except FileNotFoundError:
            pass


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