import pytest
import os
import shutil

from dps.parallel import (
    FileSystemObjectStore, ZipObjectStore, ObjectFragment, Job
)


root_test_dir = "/tmp/test_dps/"


def test_object_store_general():
    directory = os.path.join(root_test_dir, 'object_store_general')
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

    directory = os.path.join(root_test_dir, 'object_store_no_overwrite')
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

    directory = os.path.join(root_test_dir, "object_store_fragments")
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


def test_parallel_job():
    directory = os.path.join(root_test_dir, 'parallel_job')
    parent = os.path.dirname(directory)
    shutil.rmtree(parent, ignore_errors=True)

    job = Job(directory)
    x = range(10)
    z = job.map(lambda y: y + 1, x)
    job.reduce(lambda *inputs: sum(inputs), z)

    print(job.summary(verbose=4))

    # Run job
    job.simple_run("map", 0)
    job.simple_run("map", 1)
    job.simple_run("map", [2, 4])
    job.simple_run("map", [3, 5])
    job.simple_run("map", 6)
    job.simple_run("map", [7, 8])
    job.simple_run("map", [9])

    job.simple_run("reduce", None)

    result = job.get_ops("reduce")[0].get_outputs(job.objects)[0]
    assert result == sum(i + 1 for i in range(10))

    print(job.summary(verbose=4))
