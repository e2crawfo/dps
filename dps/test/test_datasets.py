import shutil

from dps.utils import NumpySeed, remove
from dps.datasets import (
    EmnistDataset, VisualArithmeticDataset, GridArithmeticDataset, OmniglotDataset,
    GridEMNIST_ObjectDetection
)


def test_cache_dataset():
    with NumpySeed(100):

        kwargs = dict(
            shape=(14, 14), include_blank=True, one_hot=False,
            balance=True, classes=[1, 2, 3], n_examples=100)

        cache_dir = "/tmp/dps_test/cached_datasets"

        shutil.rmtree(cache_dir, ignore_errors=True)

        with remove(cache_dir):
            dataset = EmnistDataset(**kwargs, use_dataset_cache=cache_dir)
            assert not dataset.loaded

            assert set(dataset.y.flatten()) == set([0, 1, 2, 3])
            assert dataset.x.shape == (100, 14, 14)
            assert dataset.y.shape == (100, 1)
            assert dataset.x.min() == 0.0
            assert 0.0 <= dataset.x.min() <= 10.0
            assert 200.0 <= dataset.x.max() <= 255.0

            dataset2 = EmnistDataset(**kwargs, use_dataset_cache=cache_dir)
            assert dataset2.loaded

            assert set(dataset2.y.flatten()) == set([0, 1, 2, 3])
            assert dataset2.x.shape == (100, 14, 14)
            assert dataset2.y.shape == (100, 1)
            assert dataset2.x.min() == 0.0
            assert 0.0 <= dataset2.x.min() <= 10.0
            assert 200.0 <= dataset2.x.max() <= 255.0

            assert (dataset.x == dataset2.x).all()
            assert (dataset.y == dataset2.y).all()

            dataset3 = EmnistDataset(**kwargs, use_dataset_cache=False)

            assert set(dataset3.y.flatten()) == set([0, 1, 2, 3])
            assert dataset3.x.shape == (100, 14, 14)
            assert dataset3.y.shape == (100, 1)
            assert dataset3.x.min() == 0.0
            assert 0.0 <= dataset3.x.min() <= 10.0
            assert 200.0 <= dataset3.x.max() <= 255.0

            assert (dataset3.x != dataset2.x).any()
            assert (dataset3.y != dataset2.y).any()


def test_emnist_dataset():
    with NumpySeed(100):

        kwargs = dict(
            shape=(14, 14), include_blank=True, one_hot=False,
            balance=True, classes=[1, 2, 3], n_examples=100)

        dataset = EmnistDataset(**kwargs)
        assert set(dataset.y.flatten()) == set([0, 1, 2, 3])
        assert dataset.x.shape == (100, 14, 14)
        assert dataset.y.shape == (100, 1)
        assert dataset.x.min() == 0.0
        assert 0.0 <= dataset.x.min() <= 10.0
        assert 200.0 <= dataset.x.max() <= 255.0

        kwargs['one_hot'] = True
        dataset = EmnistDataset(**kwargs)
        assert dataset.x.shape == (100, 14, 14)
        assert dataset.y.shape == (100, 4)
        assert (dataset.y.sum(1) == 1).all()
        assert ((dataset.y == 0) | (dataset.y == 1)).all()
        assert 0.0 <= dataset.x.min() <= 10.0
        assert 200.0 <= dataset.x.max() <= 255.0


def test_visual_arithmetic_dataset():
    with NumpySeed(100):
        n_examples = 100

        kwargs = dict(
            reductions="sum", min_digits=2, max_digits=3, digits=list(range(5)),
            sub_image_shape=(14, 14), image_shape=(50, 50), largest_digit=1000,
            one_hot=False, n_sub_image_examples=100, n_examples=n_examples)

        dataset = VisualArithmeticDataset(**kwargs)
        assert dataset.x.shape == (n_examples, 50, 50)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))

        _kwargs = kwargs.copy()
        _kwargs.update(image_shape=(100, 100), draw_shape=(50, 50))
        dataset = VisualArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 100, 100)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))
        assert (dataset.x[:, 50:, :] == 0).all()
        assert (dataset.x[:, :, 50:] == 0).all()

        _kwargs = kwargs.copy()
        _kwargs.update(image_shape=(100, 100), draw_shape=(50, 50), draw_offset=(50, 50))
        dataset = VisualArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 100, 100)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))
        assert (dataset.x[:, :50, :] == 0).all()
        assert (dataset.x[:, :, :50] == 0).all()

        _kwargs = kwargs.copy()
        _kwargs.update(largest_digit=5)
        dataset = VisualArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 50, 50)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(6)))

        _kwargs = kwargs.copy()
        _kwargs.update(one_hot=True, largest_digit=5)
        dataset = VisualArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 50, 50)
        assert dataset.y.shape == (n_examples, 7)
        assert ((dataset.y == 0) | (dataset.y == 1)).all()

        _kwargs = kwargs.copy()
        _kwargs.update(reductions=sum)
        dataset = VisualArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 50, 50)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))

        _kwargs = kwargs.copy()
        _kwargs.update(reductions="A:sum")
        dataset = VisualArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 50, 50)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))

        _kwargs = kwargs.copy()
        _kwargs.update(reductions="A:sum,M:prod")
        dataset = VisualArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 50, 50)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(126)))
        assert not set(dataset.y.flatten()).issubset(set(range(20)))

        _kwargs = kwargs.copy()
        _kwargs.update(reductions="min")
        dataset = VisualArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 50, 50)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(10)))


def test_grid_arithmetic_dataset():
    with NumpySeed(100):
        n_examples = 100

        kwargs = dict(
            reductions="sum", min_digits=2, max_digits=3, digits=list(range(5)),
            sub_image_shape=(14, 14), image_shape_grid=(3, 3), largest_digit=1000,
            one_hot=False, n_sub_image_examples=100, n_examples=n_examples)

        dataset = GridArithmeticDataset(**kwargs)
        assert dataset.x.shape == (n_examples, 42, 42)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))

        _kwargs = kwargs.copy()
        _kwargs.update(image_shape_grid=(6, 6), draw_shape_grid=(3, 3))
        dataset = GridArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 84, 84)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))
        assert (dataset.x[:, 42:, :] == 0).all()
        assert (dataset.x[:, :, 42:] == 0).all()

        _kwargs = kwargs.copy()
        _kwargs.update(image_shape_grid=(6, 6), draw_shape_grid=(3, 3), draw_offset=(42, 42))
        dataset = GridArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 84, 84)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))
        assert (dataset.x[:, :42, :] == 0).all()
        assert (dataset.x[:, :, :42] == 0).all()

        _kwargs = kwargs.copy()
        _kwargs.update(largest_digit=5)
        dataset = GridArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 42, 42)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(6)))

        _kwargs = kwargs.copy()
        _kwargs.update(one_hot=True, largest_digit=5)
        dataset = GridArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 42, 42)
        assert dataset.y.shape == (n_examples, 7)
        assert ((dataset.y == 0) | (dataset.y == 1)).all()

        _kwargs = kwargs.copy()
        _kwargs.update(reductions=sum)
        dataset = GridArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 42, 42)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))

        _kwargs = kwargs.copy()
        _kwargs.update(reductions="A:sum")
        dataset = GridArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 42, 42)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(13)))

        _kwargs = kwargs.copy()
        _kwargs.update(reductions="A:sum,M:prod")
        dataset = GridArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 42, 42)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(126)))
        assert not set(dataset.y.flatten()).issubset(set(range(20)))

        _kwargs = kwargs.copy()
        _kwargs.update(reductions="min")
        dataset = GridArithmeticDataset(**_kwargs)
        assert dataset.x.shape == (n_examples, 42, 42)
        assert dataset.y.shape == (n_examples, 1)
        assert set(dataset.y.flatten()).issubset(set(range(10)))


def test_omniglot_dataset():
    with NumpySeed(100):

        classes = [
            'Cyrillic,17', 'Mkhedruli_(Georgian),5', 'Bengali,23', 'Mongolian,19',
            'Malayalam,3', 'Ge_ez,15', 'Glagolitic,33', 'Tagalog,11', 'Gujarati,23',
            'Old_Church_Slavonic_(Cyrillic),7']  # Chosen randomly from set of all omniglot characters.
        indices = [1, 3, 5, 7, 9]
        kwargs = dict(
            shape=(14, 14), include_blank=True, one_hot=False,
            indices=indices, classes=classes
        )

        n_classes = len(classes) + 1
        n_examples = len(indices) * n_classes

        dataset = OmniglotDataset(**kwargs)
        assert set(dataset.y.flatten()) == set(range(n_classes))
        assert dataset.x.shape == (n_examples, 14, 14)
        assert dataset.y.shape == (n_examples, 1)
        assert dataset.x.min() == 0.0
        assert 0.0 <= dataset.x.min() <= 10.0
        assert 200.0 <= dataset.x.max() <= 255.0

        kwargs['one_hot'] = True

        dataset = OmniglotDataset(**kwargs)
        assert dataset.x.shape == (n_examples, 14, 14)
        assert dataset.y.shape == (n_examples, n_classes)
        assert (dataset.y.sum(1) == 1).all()
        assert ((dataset.y == 0) | (dataset.y == 1)).all()
        assert 0.0 <= dataset.x.min() <= 10.0
        assert 200.0 <= dataset.x.max() <= 255.0


def test_tile_wrapper(show_plots):
    with NumpySeed(100):
        tile_shape = (28, 35)
        n_examples = 10
        dset = GridEMNIST_ObjectDetection(
            min_chars=25, max_chars=25, n_sub_image_examples=100, n_examples=n_examples,
            draw_shape_grid=(5, 5), image_shape=(4*14, 5*14), draw_offset=(0, 0), spacing=(-5, -5),
            characters=list(range(10)), colours="white", postprocessing="tile", tile_shape=tile_shape)

        assert dset.x[0].shape == tile_shape + (3,)
        assert dset.image_shape == tile_shape
        assert len(dset.x) == n_examples * 4
        assert dset.n_examples == n_examples * 4

        if show_plots:
            dset.visualize()


def test_random_wrapper(show_plots):
    with NumpySeed(100):
        tile_shape = (28, 35)
        n_examples = 10
        dset = GridEMNIST_ObjectDetection(
            min_chars=25, max_chars=25, n_sub_image_examples=100, n_examples=n_examples,
            draw_shape_grid=(5, 5), image_shape=(4*14, 5*14), draw_offset=(0, 0), spacing=(-5, -5),
            characters=list(range(10)), colours="white", postprocessing="random", tile_shape=tile_shape, n_samples_per_image=4)

        assert dset.x[0].shape == tile_shape + (3,)
        assert dset.image_shape == tile_shape
        assert len(dset.x) == n_examples * 4
        assert dset.n_examples == n_examples * 4

    if show_plots:
        dset.visualize()
