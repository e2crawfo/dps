import numpy as np
from skimage.transform import resize

from dps import cfg
from dps.utils import image_to_string, Param, Parameterized, DataContainer
from dps.datasets import load_emnist, load_omniglot, emnist_classes, omniglot_classes


class Dataset(Parameterized):
    n_examples = Param(None)

    def __init__(self, *tracks, shuffle=True, **kwargs):
        length = len(tracks[0])
        assert all(len(t) == length for t in tracks[1:])

        self.tracks = list(tracks)
        self.obs_shape = np.array(self.tracks[0])
        self.n_examples = len(self.tracks[0])
        self.shuffle = shuffle

        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._reset_indices(self.shuffle)

        super(Dataset, self).__init__(**kwargs)

    @property
    def x(self):
        return self.tracks[0]

    @x.setter
    def x(self, value):
        self.tracks[0] = value

    @property
    def y(self):
        return self.tracks[-1]

    @y.setter
    def y(self, value):
        self.tracks[-1] = value

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def completion(self):
        return self.epochs_completed + self.index_in_epoch / self.n_examples

    def _reset_indices(self, shuffle):
        indices = np.arange(self.n_examples)
        if self.shuffle:
            np.random.shuffle(indices)
        self.indices = indices

    def post_process_batch(self, output):
        return tuple(np.array(t) for t in output)

    def next_batch(self, batch_size=None, advance=True):
        """ Return the next ``batch_size`` examples from this data set.

        If ``batch_size`` not specified, return rest of the examples in the current epoch.

        """
        start = self._index_in_epoch

        if batch_size is None:
            batch_size = self.n_examples - start

        if batch_size > self.n_examples:
            if advance:
                self._epochs_completed += batch_size / self.n_examples
                self._index_in_epoch = 0
            indices = np.random.choice(self.n_examples, batch_size, replace=True)

        if start + batch_size >= self.n_examples:
            # Finished epoch

            # Get the remaining examples in this epoch
            rest_indices = self.indices[start:]

            self._reset_indices(self.shuffle and advance)

            # Start next epoch
            end = batch_size - len(rest_indices)
            new_indices = self.indices[:end]

            indices = [*rest_indices, *new_indices]

            if advance:
                self._index_in_epoch = end
                self._epochs_completed += 1
        else:
            # Middle of epoch
            end = start + batch_size
            indices = self.indices[start:end]

            if advance:
                self._index_in_epoch = end

        output = [[t[i] for i in indices] for t in self.tracks]
        return self.post_process_batch(output)


class ImageDataset(Dataset):
    def post_process_batch(self, x):
        x = super(ImageDataset, self).post_process_batch(x)

        first, *rest = x
        assert first.dtype == np.uint8
        first = (first / 255.).astype('f')
        return tuple([first, *rest])


# EMNIST ***************************************


class EmnistDataset(ImageDataset):
    """
    Download and pre-process EMNIST dataset:
    python scripts/download.py emnist <desired location>

    """
    shape = Param((14, 14))
    include_blank = Param(True)
    one_hot = Param(True)
    balance = Param(False)
    classes = Param()

    class_pool = ''.join(
        [str(i) for i in range(10)] +
        [chr(i + ord('A')) for i in range(26)] +
        [chr(i + ord('a')) for i in range(26)]
    )

    @staticmethod
    def sample_classes(n_classes):
        classes = np.random.choice(len(EmnistDataset.class_pool), n_classes, replace=False)
        return [EmnistDataset.class_pool[i] for i in classes]

    def __init__(self, **kwargs):
        x, y, class_map = load_emnist(cfg.data_dir, **self.param_values())

        if x.shape[0] < self.n_examples:
            raise Exception(
                "Too few datapoints. Requested {}, "
                "only {} are available.".format(self.n_examples, x.shape[0]))
        super(EmnistDataset, self).__init__(x, y)


# VISUAL_ARITHMETIC ***************************************


class Rect(object):
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
        return "<%d:%d %d:%d>" % (self.top, self.bottom, self.left, self.right)


class PatchesDataset(ImageDataset):
    n_examples = Param()
    max_overlap = Param(10)
    image_shape = Param((100, 100))
    draw_shape = Param(None)
    draw_offset = Param((0, 0))
    depth = Param(None)
    sub_image_size_std = Param(None)

    def __init__(self, **kwargs):
        self.draw_shape = self.draw_shape or self.image_shape
        self.draw_offset = self.draw_offset or (0, 0)

        assert self.draw_offset[0] >= 0
        assert self.draw_offset[1] >= 0

        assert self.draw_offset[0] + self.draw_shape[0] <= self.image_shape[0]
        assert self.draw_offset[1] + self.draw_shape[1] <= self.image_shape[1]

        x, y, self.patch_centres = self._make_dataset(self.n_examples)
        super(PatchesDataset, self).__init__(x, y, **kwargs)

    def _sample_patches(self):
        raise Exception("AbstractMethod")

    def _sample_patch_locations(self, sub_image_shapes):
        """ Sample random locations within draw_shape. """

        sub_image_shapes = np.array(sub_image_shapes)
        n_rects = sub_image_shapes.shape[0]
        i = 0
        while True:
            if self.sub_image_size_std is None:
                shape_multipliers = 1.
            else:
                shape_multipliers = np.maximum(np.random.randn(n_rects, 2) * self.sub_image_size_std + 1.0, 0.5)

            _sub_image_shapes = np.ceil(shape_multipliers * sub_image_shapes[:, :2]).astype('i')

            rects = [
                Rect(
                    np.random.randint(0, self.draw_shape[0]-m+1),
                    np.random.randint(0, self.draw_shape[1]-n+1), m, n)
                for m, n in _sub_image_shapes]
            area = np.zeros(self.draw_shape, 'uint8')

            for rect in rects:
                area[rect.top:rect.bottom, rect.left:rect.right] += 1

            if (area >= 2).sum() < self.max_overlap:
                break

            i += 1

            if i > 1000:
                raise Exception(
                    "Could not fit rectangles. "
                    "(n_rects: {}, draw_shape: {}, max_overlap: {})".format(
                        n_rects, self.draw_shape, self.max_overlap))
        return rects

    def _post_process_label(self, label, rects):
        """ To be used in cases where the labels depend on the locations. """
        return label

    def _post_process(self, new_X, new_Y):
        new_X = np.array(new_X, dtype=np.uint8)
        new_Y = np.array(new_Y, dtype='i')
        if new_Y.ndim == 1:
            new_Y = new_Y[..., None]
        return new_X, new_Y

    def _make_dataset(self, n_examples):
        if n_examples == 0:
            return np.zeros((0,) + self.image_shape).astype('uint8'), np.zeros((0, 1)).astype('i')

        new_X, new_Y = [], []
        patch_centres = []

        for j in range(n_examples):
            sub_images, y = self._sample_patches()
            sub_image_shapes = [img.shape for img in sub_images]
            rects = self._sample_patch_locations(sub_image_shapes)
            y = self._post_process_label(y, rects)

            patch_centres.append([r.centre() for r in rects])

            draw_shape = self.draw_shape
            if self.depth is not None:
                draw_shape = draw_shape + (self.depth,)

            x = np.zeros(draw_shape, 'uint8')

            # Populate rectangles
            for image, rect in zip(sub_images, rects):
                rect_shape = (rect.h, rect.w)
                if image.shape[:2] != rect_shape:
                    image = resize(image, rect_shape, mode='edge', preserve_range=True)

                patch = x[rect.top:rect.bottom, rect.left:rect.right, ...]
                x[rect.top:rect.bottom, rect.left:rect.right, ...] = np.maximum(image, patch)

            if self.draw_shape != self.image_shape or self.draw_offset != (0, 0):
                image_shape = self.image_shape
                if self.depth is not None:
                    image_shape = image_shape + (self.depth,)

                _x = np.zeros(image_shape, 'uint8')
                y_start, x_start = self.draw_offset
                y_end, x_end = y_start + self.draw_shape[0], x_start + self.draw_shape[1]
                _x[y_start:y_end, x_start:x_end, ...] = x
                x = _x

            new_X.append(x)
            new_Y.append(y)

            if j % 10000 == 0:
                print(y)
                print(image_to_string(x))
                print("\n")

        new_X, new_Y = self._post_process(new_X, new_Y)

        return new_X, new_Y, patch_centres

    def visualize(self, n=9):
        import matplotlib.pyplot as plt
        m = int(np.ceil(np.sqrt(n)))
        fig, subplots = plt.subplots(m, m)
        size = int(np.sqrt(self.x.shape[1]))
        for i, s in enumerate(subplots.flatten()):
            s.imshow(self.x[i, :].reshape(size, size))
            s.set_title(str(self.y[i, 0]))


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
    sub_image_shape = Param((14, 14))
    n_sub_image_examples = Param(None)
    one_hot = Param(False)
    largest_digit = Param(1000)
    op_scale = Param(1.0)

    reductions_dict = {
        "sum": sum,
        "prod": np.product,
        "max": max,
        "min": min,
        "len": len,
    }

    def __init__(self, **kwargs):
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
                                                            shape=self.sub_image_shape, one_hot=False,
                                                            n_examples=self.n_sub_image_examples)
            emnist_y = np.squeeze(emnist_y, 1)

            self._remapped_reductions = {character_map[k]: v for k, v in reductions.items()}

            self.op_reps = DataContainer(emnist_x, emnist_y)
        else:
            assert callable(reductions)
            self.op_reps = None
            self.func = reductions

        mnist_x, mnist_y, classmap = load_emnist(cfg.data_dir, self.digits, balance=True,
                                                 shape=self.sub_image_shape, one_hot=False,
                                                 n_examples=self.n_sub_image_examples)
        mnist_y = np.squeeze(mnist_y, 1)
        inverted_classmap = {v: k for k, v in classmap.items()}
        mnist_y = np.array([inverted_classmap[y] for y in mnist_y])

        self.digit_reps = DataContainer(mnist_x, mnist_y)

        super(VisualArithmeticDataset, self).__init__(**kwargs)

        del self.digit_reps
        del self.op_reps

    def _sample_patches(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        digits = [self.digit_reps.get_random() for i in range(n)]
        digit_x, digit_y = zip(*digits)

        if self.op_reps is not None:
            op_x, op_y = self.op_reps.get_random()
            func = self._remapped_reductions[int(op_y)]
            images = [self.op_scale * op_x] + list(digit_x)
        else:
            func = self.func
            images = list(digit_x)

        y = func(digit_y)

        if self.one_hot:
            _y = np.zeros(self.largest_digit + 2)
            hot_idx = min(int(y), self.largest_digit + 1)
            _y[hot_idx] = 1.0
            y = _y
        else:
            y = np.minimum(y, self.largest_digit)

        return images, y


class VisualArithmeticDatasetColour(VisualArithmeticDataset):
    colours = Param('red green blue')

    def __init__(self, **kwargs):
        self.depth = 3
        colours = self.colours
        if isinstance(colours, str):
            colours = colours.split(' ')

        import matplotlib as mpl
        colour_map = mpl.colors.get_named_colors_mapping()
        self._colours = [np.array(mpl.colors.to_rgb(colour_map[cn]))[None, None, :] for cn in colours]

        super(VisualArithmeticDatasetColour, self).__init__(**kwargs)

    def colourize(self, img, colour_idx=None):
        if colour_idx is None:
            colour_idx = np.random.randint(len(self._colours))
        colour = self._colours[colour_idx]
        colourized = np.array(img[..., None] * colour, np.uint8)
        return colourized

    def _sample_patches(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        digits = [self.digit_reps.get_random() for i in range(n)]
        digit_x, digit_y = zip(*digits)

        digit_x = [self.colourize(dx) for dx in digit_x]

        if self.op_reps is not None:
            op_x, op_y = self.op_reps.get_random()
            op_x = self.colourize(op_x)
            func = self._remapped_reductions[int(op_y)]
            images = [op_x, *digit_x]
        else:
            func = self.func
            images = list(digit_x)

        y = func(digit_y)

        if self.one_hot:
            _y = np.zeros(self.largest_digit + 2)
            hot_idx = min(int(y), self.largest_digit + 1)
            _y[hot_idx] = 1.0
            y = _y
        else:
            y = np.minimum(y, self.largest_digit)

        return images, y


class GridArithmeticDataset(VisualArithmeticDataset):
    image_shape_grid = Param((2, 2))
    draw_shape_grid = Param((2, 2))
    op_loc = Param(None)

    def __init__(self, **kwargs):
        self.image_shape = tuple(gs*s for gs, s in zip(self.image_shape_grid, self.sub_image_shape))

        if self.draw_shape_grid is None:
            self.draw_shape_grid = self.image_shape_grid
        self.draw_shape = tuple(gs*s for gs, s in zip(self.draw_shape_grid, self.sub_image_shape))

        self.grid_size = np.product(self.draw_shape_grid)
        if self.op_loc is not None:
            self.op_loc_idx = np.ravel_multi_index(self.op_loc, self.draw_shape_grid)

        super(GridArithmeticDataset, self).__init__(**kwargs)

    def _sample_patch_locations(self, sub_image_shapes):
        """ Sample random locations within draw_shape. """
        n_images = len(sub_image_shapes)
        indices = np.random.choice(self.grid_size, n_images, replace=False)

        if sub_image_shapes and self.op_loc is not None and self.op_reps is not None:
            indices[indices == self.op_loc_idx] = indices[0]
            indices[0] = self.op_loc_idx

        grid_locs = list(zip(*np.unravel_index(indices, self.draw_shape_grid)))
        top_left = np.array(grid_locs) * self.sub_image_shape

        return [Rect(t, l, m, n) for (t, l), (m, n) in zip(top_left, sub_image_shapes)]


# OMNIGLOT ***************************************


class OmniglotDataset(ImageDataset):
    shape = Param()
    include_blank = Param()
    one_hot = Param()
    indices = Param()
    classes = Param()

    n_examples = Param(0)

    @staticmethod
    def sample_classes(n_classes):
        class_pool = omniglot_classes()
        classes = np.random.choice(len(class_pool), n_classes, replace=False)
        return [class_pool[i] for i in classes]

    def __init__(self, indices, **kwargs):
        pv = self.param_values()
        del pv['n_examples']
        x, y, class_map = load_omniglot(cfg.data_dir, **pv)
        super(OmniglotDataset, self).__init__(x, y)


class OmniglotCountingDataset(PatchesDataset):
    min_digits = Param(2)
    max_digits = Param(3)
    classes = Param()
    sub_image_shape = Param((14, 14))
    one_hot = Param(False)
    target_scale = Param(0.5)
    indices = Param(list(range(20)))

    def __init__(self, **kwargs):
        assert self.min_digits <= self.max_digits
        assert np.product(self.draw_shape) >= self.max_digits + 1

        omniglot_x, omniglot_y, character_map = load_omniglot(
            cfg.data_dir, self.classes, one_hot=False,
            indices=self.indices, shape=self.sub_image_shape
        )
        omniglot_y = np.squeeze(omniglot_y, 1)
        self.omni_reps = DataContainer(omniglot_x, omniglot_y)

        super(OmniglotCountingDataset, self).__init__(**kwargs)

        del self.omni_reps

    def _sample_patches(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        n_target_copies = np.random.randint(n)

        target_x, target_y = self.omni_reps.get_random()

        xs = [self.target_scale * target_x]

        for k in range(n-1):
            if k < n_target_copies:
                x, _ = self.omni_reps.get_random_with_label(target_y)
            else:
                x, _ = self.omni_reps.get_random_without_label(target_y)
            xs.append(x)

        y = n_target_copies

        if self.one_hot:
            _y = np.zeros(self.max_digits-1)
            _y[y] = 1.0
            y = _y

        return xs, y


class GridOmniglotDataset(OmniglotCountingDataset):
    image_shape_grid = Param()
    draw_shape_grid = Param()
    target_loc = Param()

    def __init__(self, **kwargs):
        self.image_shape = tuple(gs*s for gs, s in zip(self.image_shape_grid, self.sub_image_shape))

        if self.draw_shape_grid is None:
            self.draw_shape_grid = self.image_shape_grid
        self.draw_shape = tuple(gs*s for gs, s in zip(self.draw_shape_grid, self.sub_image_shape))

        self.grid_size = np.product(self.draw_shape_grid)
        if self.target_loc is not None:
            self.target_loc_idx = np.ravel_multi_index(self.target_loc, self.draw_shape_grid)

        super(GridOmniglotDataset, self).__init__(**kwargs)

    def _sample_patch_locations(self, sub_image_shapes):
        """ Sample random locations within draw_shape. """
        n_images = len(sub_image_shapes)
        indices = np.random.choice(self.grid_size, n_images, replace=False)

        if sub_image_shapes and self.target_loc is not None and self.op_reps is not None:
            indices[indices == self.target_loc_idx] = indices[0]
            indices[0] = self.target_loc_idx

        grid_locs = list(zip(*np.unravel_index(indices, self.draw_shape_grid)))
        top_left = np.array(grid_locs) * self.sub_image_shape

        return [Rect(t, l, m, n) for (t, l), (m, n) in zip(top_left, sub_image_shapes)]


# SALIENCE ***************************************


class SalienceDataset(PatchesDataset):
    """ A dataset for detecting salience.  """

    classes = Param(None)
    min_digits = Param(1)
    max_digits = Param(1)
    n_sub_image_examples = Param(None)
    sub_image_shape = Param((14, 14))
    output_shape = Param((14, 14))
    std = Param(0.1)
    flatten_output = Param(False)
    point = Param(False)

    def __init__(self, **kwargs):
        classes = self.classes or emnist_classes()
        self.X, _, _ = load_emnist(
            cfg.data_dir, classes, shape=self.sub_image_shape, n_examples=self.n_sub_image_examples)

        super(SalienceDataset, self).__init__(**kwargs)

        del self.X

        y = []

        for x, pc in zip(self.x, self.patch_centres):
            _y = np.zeros(self.output_shape)
            for centre in pc:
                if self.point:
                    pixel_y = int(centre[0] * self.output_shape[0] / self.image_shape[0])
                    pixel_x = int(centre[1] * self.output_shape[1] / self.image_shape[1])
                    _y[pixel_y, pixel_x] = 1.0
                else:
                    kernel = gaussian_kernel(
                        self.output_shape,
                        (centre[0]/self.image_shape[0], centre[1]/self.image_shape[1]),
                        self.std)
                    _y = np.maximum(_y, kernel)
            y.append(_y)

        y = np.array(y)
        if self.flatten_output:
            y = y.reshape(y.shape[0], -1)
        self.y = y

        for j in range(self.y.shape[0]):
            if j % 10000 == 0:
                print(image_to_string(self.y[j, ...]))
                print(image_to_string(self.x[j, ...]))
                print("\n")

    def _sample_patches(self):
        n = np.random.randint(self.min_digits, self.max_digits+1)
        indices = np.random.randint(0, self.X.shape[0], n)
        images = [self.X[i] for i in indices]
        y = 0
        return images, y

    def visualize(self, n=9):
        import matplotlib.pyplot as plt
        m = int(np.ceil(np.sqrt(n)))
        fig, axes = plt.subplots(m, 2 * m)
        for i, s in enumerate(axes[:, :m].flatten()):
            s.imshow(self.x[i, :].reshape(self.image_shape))
        for i, s in enumerate(axes[:, m:].flatten()):
            s.imshow(self.y[i, :].reshape(self.output_shape))
        plt.show()


def gaussian_kernel(shape, mu, std):
    """ creates gaussian kernel with side length l and a sigma of sig """
    axy = (np.arange(shape[0]) + 0.5) / shape[1]
    axx = (np.arange(shape[1]) + 0.5) / shape[1]
    yy, xx = np.meshgrid(axx, axy, indexing='ij')

    kernel = np.exp(-((xx - mu[1])**2 + (yy - mu[0])**2) / (2. * std**2))

    return kernel


# OBJECT_DETECTION ***************************************


class EMNIST_ObjectDetection(PatchesDataset):
    min_chars = Param(2)
    max_chars = Param(3)
    characters = Param(
        [str(i) for i in range(10)] +
        [chr(i + ord('A')) for i in range(26)] +
        [chr(i + ord('a')) for i in range(26)]
    )
    sub_image_shape = Param((14, 14))
    n_sub_image_examples = Param(None)
    colours = Param('red green blue')

    def __init__(self, **kwargs):
        self.depth = 3

        colours = self.colours
        if isinstance(colours, str):
            colours = colours.split(' ')

        import matplotlib as mpl
        colour_map = mpl.colors.get_named_colors_mapping()
        self._colours = [np.array(mpl.colors.to_rgb(colour_map[cn]))[None, None, :] for cn in colours]

        assert self.min_chars <= self.max_chars

        emnist_x, emnist_y, self.classmap = load_emnist(cfg.data_dir, self.characters, balance=True,
                                                        shape=self.sub_image_shape, one_hot=False,
                                                        n_examples=self.n_sub_image_examples)
        emnist_y = np.squeeze(emnist_y, 1)

        self.char_reps = DataContainer(emnist_x, emnist_y)

        super(EMNIST_ObjectDetection, self).__init__(**kwargs)

        del self.char_reps

    def _post_process_label(self, label, rects):
        """ To be used in cases where the labels depend on the locations. """
        return [[l, r.top, r.bottom, r.left, r.right] for l, r in zip(label, rects)]

    def _post_process(self, new_X, new_Y):
        new_X = np.array(new_X, dtype=np.uint8)
        return new_X, new_Y

    def colourize(self, img, colour_idx=None):
        if colour_idx is None:
            colour_idx = np.random.randint(len(self._colours))
        colour = self._colours[colour_idx]
        colourized = np.array(img[..., None] * colour, np.uint8)
        return colourized

    def _sample_patches(self):
        n = np.random.randint(self.min_chars, self.max_chars+1)
        chars = [self.char_reps.get_random() for i in range(n)]
        char_x, char_y = zip(*chars)
        char_x = [self.colourize(cx) for cx in char_x]
        return char_x, char_y

    def visualize(self, n=9):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        m = int(np.ceil(np.sqrt(n)))
        fig, subplots = plt.subplots(m, m)

        height = self.x.shape[1]

        for i, ax in enumerate(subplots.flatten()):
            ax.imshow(self.x[i, ...])
            for cls, top, bottom, left, right in self.y[i]:
                width = bottom - top
                height = right - left
                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()


if __name__ == "__main__":
    # dset = VisualArithmeticDataset(n_examples=10, draw_shape=(50, 50), draw_offset=(50, 50))
    # dset = GridArithmeticDataset(
    #     n_examples=10, draw_shape_grid=(3, 3), image_shape_grid=(3, 3), sub_image_shape=(28, 28), op_scale=0.5)

    # dset = OmniglotCountingDataset(classes=classes, n_examples=10, sub_image_shape=(28, 28))
    # dset = SalienceDataset(min_digits=1, max_digits=4, sub_image_n_exmaples=100, n_examples=10)
    dset = EMNIST_ObjectDetection(min_chars=1, max_chars=10, n_sub_image_examples=100, n_examples=10)
    dset.visualize()
