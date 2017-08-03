from pathlib import Path
import dill
import gzip

import matplotlib.pyplot as plt
import numpy as np

from dps import cfg
from dps.environment import RegressionDataset
from dps.utils import Param


# Character used for ascii art, sorted in order of increasing sparsity
ascii_art_chars = \
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "


def char_map(value):
    """ Maps a relative "sparsity" or "lightness" value in [0, 1) to a character. """
    if value >= 1:
        value = 1 - 1e-6
    n_bins = len(ascii_art_chars)
    bin_id = int(value * n_bins)
    return ascii_art_chars[bin_id]


def image_to_string(array):
    """ Convert an image stored as an array to an ascii art string """
    if array.ndim == 1:
        array = array.reshape(-1, int(np.sqrt(array.shape[0])))
    image = [char_map(value) for value in array.flatten()]
    image = np.reshape(image, array.shape)
    return '\n'.join(''.join(c for c in row) for row in image)


def view_emnist(x, y, n):
    m = int(np.ceil(np.sqrt(n)))
    fig, subplots = plt.subplots(m, m)
    for i, s in enumerate(subplots.flatten()):
        s.imshow(x[i, :].reshape(28, 28).transpose())
        s.set_title(str(y[i, 0]))


def load_emnist(classes, balance=False, include_blank=False):
    """ Load emnist data from disk by class.

    Elements of `classes` pick out which emnist classes to load, but different labels
    end up getting returned because most classifiers require that the labels
    be in range(len(classes)). We return a dictionary `class_map` which maps from
    elements of `classes` down to range(len(classes)).

    """
    classes = list(classes)[:]
    data_dir = Path(cfg.data_dir).expanduser()
    emnist_dir = data_dir / 'emnist/emnist-byclass'
    y = []
    x = []
    class_map = {}
    for i, cls in enumerate(sorted(list(classes))):
        with gzip.open(str(emnist_dir / (str(cls) + '.pklz')), 'rb') as f:
            x.append(dill.load(f))
            y.extend([i] * x[-1].shape[0])
        class_map[cls] = i
    x = np.concatenate(x, axis=0)
    y = np.array(y).reshape(-1, 1)

    if include_blank:
        class_count = min([(y == class_map[c]).sum() for c in classes])
        blanks = np.zeros((class_count, x.shape[1]))
        x = np.concatenate((x, blanks), axis=0)
        blank_idx = len(class_map)
        y = np.concatenate((y, blank_idx * np.ones((class_count, 1))), axis=0)
        blank_symbol = 62
        class_map[blank_symbol] = blank_idx
        classes.append(blank_symbol)

    order = np.random.permutation(x.shape[0])

    x = x[order, :]
    y = y[order, :]

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

    return x, y, class_map


class Rect(object):
    def __init__(self, x, y, w, h):
        self.left = x
        self.right = x+w
        self.top = y+h
        self.bottom = y

    def intersects(self, r2):
        r1 = self
        h_overlaps = (r1.left <= r2.right) and (r1.right >= r2.left)
        v_overlaps = (r1.bottom <= r2.top) and (r1.top >= r2.bottom)
        return h_overlaps and v_overlaps

    def __str__(self):
        return "<%d:%d %d:%d>" % (self.left, self.right, self.top, self.bottom)


class PatchesDataset(RegressionDataset):
    W = Param()
    max_overlap = Param()

    def __init__(self, **kwargs):
        x, y = self._make_dataset(self.n_examples)
        super(PatchesDataset, self).__init__(x, y)

    def _sample_patches(self):
        raise Exception()

    def _make_dataset(self, n_examples):
        max_overlap, W = self.max_overlap, self.W
        if n_examples == 0:
            return np.zeros((0, W, W)).astype('f'), np.zeros((0, 1)).astype('i')

        new_X, new_Y = [], []

        for j in range(n_examples):
            images, y = self._sample_patches()
            image_shapes = [img.shape for img in images]

            # Sample rectangles
            n_rects = len(images)
            i = 0
            while True:
                rects = [
                    Rect(
                        np.random.randint(0, W-m+1),
                        np.random.randint(0, W-n+1), m, n)
                    for m, n in image_shapes]
                area = np.zeros((W, W), 'f')

                for rect in rects:
                    area[rect.left:rect.right, rect.bottom:rect.top] += 1

                if (area >= 2).sum() < max_overlap:
                    break

                i += 1

                if i > 1000:
                    raise Exception(
                        "Could not fit rectangles. "
                        "(n_rects: {}, W: {}, max_overlap: {})".format(
                            n_rects, W, max_overlap))

            # Populate rectangles
            o = np.zeros((W, W), 'f')
            for image, rect in zip(images, rects):
                o[rect.left:rect.right, rect.bottom:rect.top] += image

            new_X.append(np.uint8(255*np.minimum(o, 1)))
            new_Y.append(y)

        new_X = np.array(new_X).astype('f')
        new_Y = np.array(new_Y).astype('i').reshape(-1, 1)
        return new_X, new_Y

    def visualize(self, n=9):
        m = int(np.ceil(np.sqrt(n)))
        fig, subplots = plt.subplots(m, m)
        size = int(np.sqrt(self.x.shape[1]))
        for i, s in enumerate(subplots.flatten()):
            s.imshow(self.x[i, :].reshape(size, size))
            s.set_title(str(self.y[i, 0]))


def idx_to_char(i):
    assert isinstance(i, int)
    if i >= 72:
        raise Exception()
    elif i >= 36:
        char = chr(i-36+ord('a'))
    elif i >= 10:
        char = chr(i-10+ord('A'))
    elif i >= 0:
        char = str(i)
    else:
        raise Exception()
    return char


def char_to_idx(c):
    assert isinstance(c, str)
    assert len(c) == 1

    if c.isupper():
        idx = ord(c) - ord('A') + 10
    elif c.islower():
        idx = ord(c) - ord('A') + 36
    elif c.isnumeric():
        idx = int(c)
    else:
        raise Exception()
    return idx


class MnistArithmeticDataset(PatchesDataset):
    min_digits = Param()
    max_digits = Param()
    reductions = Param()
    base = Param()

    def __init__(self, **kwargs):
        self.X, self.Y, self.class_map = load_emnist(list(range(self.base)))
        self.X = self.X.reshape(-1, 28, 28)

        # reductions is a list of pairs of the form (character, reduction function)

        # map each character to its index according to the emnist dataset
        reductions = {char_to_idx(s): f for s, f in self.reductions}

        self.eX, self.eY, _class_map = load_emnist(list(reductions.keys()))
        self.eX = self.eX.reshape(-1, 28, 28)
        self.class_map.update(_class_map)

        self.reductions = {self.class_map[k]: v for k, v in reductions.items()}

        super(MnistArithmeticDataset, self).__init__()

        del self.X
        del self.Y
        del self.eX
        del self.eY

    def _sample_patches(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        symbol_idx = np.random.randint(0, self.eY.shape[0])
        symbol_class = self.eY[symbol_idx, 0]
        func = self.reductions[symbol_class]
        digit_indices = np.random.randint(0, self.Y.shape[0], n)
        images = [self.eX[symbol_idx]] + [self.X[i] for i in digit_indices]
        y = func([self.Y[i] for i in digit_indices])
        return images, y


class TranslatedMnistDataset(PatchesDataset):
    min_digits = Param()
    max_digits = Param()
    reduction = Param()
    base = Param
    symbols = Param()
    include_blank = Param()

    def __init__(self, **kwargs):
        if self.reduction is None:
            self.reduction = lambda inputs: sum(inputs)
        self.symbols = self.symbols or list(range(10))

        self.X, self.Y, self.symbol_map = load_emnist(
            self.symbols, include_blank=self.include_blank)
        self.X = self.X.reshape(-1, 28, 28)

        super(TranslatedMnistDataset, self).__init__()

        del self.X
        del self.Y

    def _sample_patches(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        digit_indices = np.random.randint(0, self.Y.shape[0], n)
        images = [self.X[i] for i in digit_indices]
        y = self.reduction([self.Y[i] for i in digit_indices])
        return images, y
