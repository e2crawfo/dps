import numpy as np
import imageio
import os
import itertools

import dps
from dps.datasets.base import PatchesDataset, LongVideoMixin
from dps.utils import Param, resize_image


class ShapesDataset(PatchesDataset):
    """ Display a specific set of shapes in random positions. """
    shape_specs = Param(help="space-separated string of shape specs. Each shape spec is of form colour,shape")
    patch_shape = Param()

    n_classes = 1

    def _make(self):
        self.patches = {}
        for spec in self.shape_specs:
            shape, colour = spec.split(",")

            f = os.path.join(os.path.dirname(dps.__file__), "datasets/shapes", "{}.png".format(shape))
            image = imageio.imread(f)
            image = resize_image(image[..., 3], self.patch_shape)
            image = self._colourize(image, colour)

            self.patches[spec] = image

        super(ShapesDataset, self)._make()

    def _sample_patches(self):
        return list(np.random.permutation(self.patches)), [0 for i in range(len(self.patches))], 0


class RandomShapesDataset(ShapesDataset):
    """ Display a random number of random shapes in random positions. """
    patch_shape = Param()
    min_shapes = Param()
    max_shapes = Param()
    shapes = Param()

    _shape_specs = None

    @property
    def shape_specs(self):
        if self._shape_specs is None:
            colours = self.colours
            if isinstance(colours, str):
                colours = colours.split()

            shapes = self.shapes
            if isinstance(shapes, str):
                shapes = shapes.split()

            shape_specs = []
            for colour in colours:
                for shape in shapes:
                    shape_specs.append("{},{}".format(shape, colour))
            self._shape_specs = shape_specs
        return self._shape_specs

    def _sample_patches(self):
        n_shapes = np.random.randint(self.min_shapes, self.max_shapes+1)
        specs = np.random.choice(self.shape_specs, size=n_shapes)
        patches = [self.patches[spec] for spec in specs]
        return patches, [0] * len(patches), 0


class LongVideoRandomShapes(LongVideoMixin, RandomShapesDataset):
    pass


class BlueXAboveRedCircle(ShapesDataset):
    patch_shape = Param()
    distractor_shapes = Param()
    n_distractor_shapes = Param()

    reference_shapes = "red,circle blue,x".split()
    shapes = None

    n_classes = 2

    def _make(self):
        if isinstance(self.distractor_shapes, str):
            self.distractor_shapes = self.distractor_shapes.split()
        self.distractor_shapes = [s for s in self.distractor_shapes if s not in self.reference_shapes]
        self.shapes = self.distractor_shapes + self.reference_shapes
        super(BlueXAboveRedCircle, self)._make()

    def _sample_image(self):
        label = np.random.randint(2)

        shape_specs = self.reference_shapes + []

        if label == 1:
            shape_specs.reverse()

        if self.n_distractor_shapes is None:
            distractor_shapes = self.distractor_shapes + []
        else:
            distractor_shapes = np.random.choice(self.distractor_shapes, size=self.n_distractor_shapes, replace=True)

        shape_specs.extend(distractor_shapes)
        patches = [self.patches[spec] for spec in shape_specs]
        patch_shapes = [self.patch_shape] * len(patches)

        ok = False
        while not ok:
            locs = self._sample_patch_locations(
                patch_shapes,
                max_overlap=self.max_overlap,
                size_std=self.patch_size_std)

            n_tries = 0
            while not ok:
                indices = np.random.permutation(range(len(locs)))
                if locs[indices[0]].top < locs[indices[1]].top:
                    ok = True

                n_tries += 1
                if n_tries > 10:
                    break

        locs = [locs[i] for i in indices]

        return locs, patches, [0 for i in indices], label


class SetThreeAttr(PatchesDataset):
    colours = Param()
    shapes = Param()
    digits = Param()
    digit_colour = Param()
    n_cards = Param()
    set_size = Param()
    patch_shape = Param()

    n_classes = 2

    @staticmethod
    def _generate_cards_and_label(cards, n_cards, set_size):
        shuffled_cards = np.random.permutation(cards)
        drawn_cards = [tuple(c) for c in shuffled_cards[:n_cards]]

        for _set in itertools.combinations(drawn_cards, set_size):
            is_set = True
            for attr_idx in range(len(_set[0])):
                attr_values = set(card[attr_idx] for card in _set)

                if len(attr_values) == 1 or len(attr_values) == set_size:
                    continue
                else:
                    is_set = False
                    break

            if is_set:
                return drawn_cards, _set

        return drawn_cards, None

    def _get_patch_for_card(self, card):
        patch = self.patches.get(card, None)
        if patch is None:
            colour, shape, digit = card

            f = os.path.join(os.path.dirname(dps.__file__), "datasets/shapes", "{}.png".format(shape))
            image = imageio.imread(f)
            image = resize_image(image[..., 3], self.patch_shape)
            image = self._colourize(image, colour)

            shape_colour = image[:, :, :3]
            shape_alpha = image[:, :, 3:4] / 255.

            f = os.path.join(os.path.dirname(dps.__file__), "datasets/digits", "{}.png".format(digit))
            digit_image = imageio.imread(f)
            digit_image = resize_image(digit_image[..., 3], self.patch_shape)
            digit_image = self._colourize(digit_image, self.digit_colour)

            digit_colour = digit_image[:, :, :3]
            digit_alpha = digit_image[:, :, 3:4] / 255.

            image_rgb = digit_alpha * digit_colour + (1-digit_alpha) * shape_colour
            image_alpha = (np.clip(digit_alpha + shape_alpha, 0, 1) * 255).astype(np.uint8)

            patch = self.patches[card] = np.concatenate([image_rgb, image_alpha], axis=2)

        return patch

    def _make(self):
        if isinstance(self.colours, str):
            self.colours = self.colours.split()
        if isinstance(self.shapes, str):
            self.shapes = self.shapes.split()
        if isinstance(self.digits, str):
            self.digits = self.digits.split()

        self.cards = list(itertools.product(self.colours, self.shapes, self.digits))
        self.patches = {}
        self.n_pos = 0
        self.n_neg = 0
        super(SetThreeAttr, self)._make()

    def _sample_patches(self):
        label = np.random.randint(2)
        blabel = bool(label)

        is_set = not blabel
        while is_set != blabel:
            cards, _set = self._generate_cards_and_label(self.cards, self.n_cards, self.set_size)
            is_set = bool(_set)
            if is_set:
                self.n_pos += 1
            else:
                self.n_neg += 1

        patches = [self._get_patch_for_card(card) for card in cards]
        return patches, [0] * len(patches), label


if __name__ == "__main__":
    import tensorflow as tf
    from scipy.stats.distributions import truncexpon

    dset = RandomShapesDataset(
        n_examples=20, shapes="circle diamond star x plus", colours="red green blue cyan magenta yellow",
        min_shapes=70, max_shapes=71, image_shape=(96, 96), patch_shape=(21, 21),
        max_overlap=98, n_frames=0,
        patch_shape_dist=truncexpon(b=0.2, loc=0.1, scale=0.6),
        patch_speed=10)

    # dset = LongVideoRandomShapes(
    #     n_examples=4, n_frames=20, n_batches=10, shapes="circle diamond star x plus", colours="red green blue cyan magenta yellow",
    #     min_shapes=5, max_shapes=10, image_shape=(96, 96), patch_shape=(21, 21),
    #     max_overlap=98, patch_size_std=0.4, patch_speed=10, backgrounds="hard")

    # shapes = "green,circle blue,circle orange,circle teal,circle red,circle black,circle"
    # dset = ShapesDataset(
    #     n_examples=20, background_colours="white", shapes=shapes,
    #     image_shape=(48, 48), patch_shape=(14, 14), max_overlap=98)

    # colours = "red blue green".split()
    # shapes = "circle triangle x".split()
    # specs = ["{},{}".format(c, s) for c, s in itertools.product(colours, shapes)]

    # dset = BlueXAboveRedCircle(
    #     n_examples=21, background_colours="white", distractor_shapes=specs,
    #     n_distractor_shapes=None, image_shape=(48, 48), patch_shape=(14, 14), max_overlap=98)

    # dset = SetThreeAttr(
    #     n_examples=16,
    #     background_colours="cyan magenta yellow",
    #     colours="red green blue",
    #     shapes="circle square diamond",
    #     digits="simple1 simple2 simple3",
    #     digit_colour="black",
    #     n_cards=7,
    #     set_size=3,
    #     image_shape=(48, 48),
    #     patch_shape=(14, 14),
    #     max_overlap=14*14/3)

    sess = tf.Session()
    with sess.as_default():
        dset.visualize(n=16)
