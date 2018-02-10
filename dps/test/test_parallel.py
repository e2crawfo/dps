import pytest
import os
import shutil
from contextlib import contextmanager
import numpy as np

from dps.parallel.object_store import FileSystemObjectStore, ZipObjectStore, ObjectFragment
from dps.parallel.hyper import build_and_submit
from dps.env.advanced import simple_addition
from dps.rl.algorithms import a2c
from dps.config import DEFAULT_CONFIG


def test_object_store_general():
    directory = '/tmp/test_fs_object_store/test'
    parent = os.path.dirname(directory)
    shutil.rmtree(parent, ignore_errors=True)

    store = FileSystemObjectStore(directory)
    obj1 = dict(x=1)
    obj2 = dict(x=2)
    obj3 = [3]
    obj4 = [4]
    obj5 = set([5])

    store.save_object('dict', 1, obj1)
    store.save_object('dict', 2, obj2)
    store.save_object('list', 3, obj3)

    new_key = store.get_unique_key('list')
    assert new_key == 4
    store.save_object('list', new_key, obj4)

    new_key = store.get_unique_key('set')
    assert new_key == 0
    store.save_object('set', new_key, obj5)

    def test_store(store):
        dicts = store.load_objects('dict')
        assert set(dicts.keys()) == set([1, 2])
        assert dicts[1] == obj1
        assert dicts[2] == obj2

        lists = store.load_objects('list')
        assert set(lists.keys()) == set([3, 4])
        assert lists[3] == obj3
        assert lists[4] == obj4

        sets = store.load_objects('set')
        assert set(sets.keys()) == set([0])
        assert sets[0] == obj5

    test_store(store)

    new_store = FileSystemObjectStore(directory)
    test_store(new_store)

    zip_path = os.path.join(parent, 'my_store.zip')

    _zip_path = store.zip(zip_path)

    assert os.path.realpath(_zip_path) == os.path.realpath(zip_path)

    zip_store = ZipObjectStore(_zip_path)
    test_store(zip_store)

    zip_store = ZipObjectStore(_zip_path)
    test_store(zip_store)


def test_object_store_no_overwrite():
    """ Test that objects cannot be overwritten. """

    directory = '/tmp/test_fs_object_store/test'
    shutil.rmtree(directory, ignore_errors=True)

    store = FileSystemObjectStore(directory)
    obj1 = dict(x=1)
    obj2 = dict(x=2)
    obj3 = [3]

    store.save_object('dict', 1, obj1)
    store.save_object('dict', 2, obj2)
    store.save_object('list', 1, obj3)

    with pytest.raises(ValueError):
        store.save_object('dict', 1, dict(x=4))


class ListFragment(ObjectFragment):
    def __init__(self, lst):
        self.lst = lst

    def combine(self, *others):
        lst = self.lst + []
        for _l in others:
            lst.extend(_l.lst)
        return lst


class DictFragment(ObjectFragment):
    def __init__(self, d):
        self.d = d

    def combine(self, *others):
        d = self.d.copy()
        for _d in others:
            d.update(_d.d)
        return d


def test_object_store_fragments():
    """ Test that object fragments can be stored and loaded. """

    directory = '/tmp/test_fs_object_store/test'
    parent = os.path.dirname(directory)
    shutil.rmtree(parent, ignore_errors=True)

    store = FileSystemObjectStore(directory)
    obj1 = dict(x=1)
    obj2 = dict(y=2)
    obj3 = dict(z=[0, 1])
    obj123 = dict(**obj1, **obj2, **obj3)

    obj4 = dict(a=1)
    obj5 = dict(b=2)
    obj45 = dict(**obj4, **obj5)

    obj6 = [1, 2]
    obj7 = [3, 4, 5]
    obj67 = obj6 + obj7

    store.save_object('dict', 1, DictFragment(obj1))
    store.save_object('dict', 1, DictFragment(obj2))
    store.save_object('dict', 1, DictFragment(obj3))

    new_key = store.get_unique_key('dict')
    assert new_key == 2
    store.save_object('dict', new_key, DictFragment(obj4))
    store.save_object('dict', new_key, DictFragment(obj5))

    new_key = store.get_unique_key('list')
    assert new_key == 0
    store.save_object('list', new_key, ListFragment(obj6))
    store.save_object('list', new_key, ListFragment(obj7))

    store.save_object('list', 1, obj6)

    with pytest.raises(ValueError):
        store.save_object('list', 1, obj7)

    store.save_object('list', 1, ListFragment(obj7))

    def test_store(store):
        dicts = store.load_objects('dict')
        assert set(dicts.keys()) == set([1, 2])
        assert dicts[1] == obj123
        assert dicts[2] == obj45

        list0 = store.load_object('list', 0)
        assert list0 == obj67

        with pytest.raises(ValueError):
            store.load_object('list', 1)

    test_store(store)

    new_store = FileSystemObjectStore(directory)
    test_store(new_store)

    zip_path = os.path.join(parent, 'my_store.zip')

    _zip_path = store.zip(zip_path)

    assert os.path.realpath(_zip_path) == os.path.realpath(zip_path)

    zip_store = ZipObjectStore(_zip_path)
    test_store(zip_store)

    zip_store = ZipObjectStore(_zip_path)
    test_store(zip_store)


@contextmanager
def remove_tree(path):
    path = os.path.abspath(path)
    try:
        yield
    finally:
        shutil.rmtree(str(path), ignore_errors=True)


@pytest.mark.slow
def test_hyper(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(a2c.config)
    config.update(simple_addition.config)
    config.update(test_config)
    config['max_steps'] = 101

    distributions = dict(n_train=2**np.array([5, 6, 7]))
    n_repeats = 2

    session = build_and_submit(
        name="test_hyper", config=config, distributions=distributions, n_repeats=n_repeats,
        kind='parallel', host_pool=':', wall_time='1year', cleanup_time='10mins',
        slack_time='10mins', ppn=2, load_avg_threshold=100000.0)

    path = session.exp_dir.path
    files = os.listdir(path)
    assert set(files) == set(
        ['orig.zip', 'experiments', 'os_environ.txt', 'results.zip', 'pip_freeze.txt',
         'dps_git_summary.txt', 'nodefile.txt', 'results.txt', 'job_log.txt']
    )
    experiments = os.listdir(os.path.join(path, 'experiments'))
    for exp in experiments:
        assert exp.startswith('exp_')
        assert os.path.isdir(os.path.join(path, 'experiments', exp))
    assert len(experiments) == n_repeats * len(distributions['n_train'])

    with open(os.path.join(path, 'results.txt'), 'r') as f:
        results = f.read()

    assert "n_ops: 6" in results
    assert "n_completed_ops: 6" in results
    assert "n_partially_completed_ops: 0" in results
    assert "n_ready_incomplete_ops: 0" in results
    assert "n_not_ready_incomplete_ops: 0" in results
