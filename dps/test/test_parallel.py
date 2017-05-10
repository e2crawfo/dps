import pytest
import shutil
from pathlib import Path

from dps.parallel.base import FileSystemObjectStore, ZipObjectStore


def test_fs_object_store():
    directory = Path('/tmp/test_fs_object_store/test')
    try:
        shutil.rmtree(str(directory))
    except:
        pass
    store = FileSystemObjectStore(directory, force_fresh=True)
    obj1 = dict(x=1)
    obj2 = dict(x=2)
    obj3 = [3]

    store.save_object('dict', 1, obj1)
    store.save_object('dict', 2, obj2)
    store.save_object('list', 1, obj3)

    dicts = store.load_objects('dict')
    assert len(dicts) == 2

    lists = store.load_objects('list')
    assert len(lists) == 1

    zip_path = directory.parent / 'my_store'
    store.zip(zip_path)

    zip_store = ZipObjectStore("{}.zip".format(zip_path))

    dicts = zip_store.load_objects('dict')
    assert len(dicts) == 2

    lists = zip_store.load_objects('list')
    assert len(lists) == 1

    zip_store = ZipObjectStore("{}.zip".format(zip_path))

    dicts = zip_store.load_objects('dict')
    assert len(dicts) == 2

    lists = zip_store.load_objects('list')
    assert len(lists) == 1


def test_force_unique():
    directory = Path('/tmp/test_fs_object_store/test')
    try:
        shutil.rmtree(str(directory))
    except:
        pass
    store = FileSystemObjectStore(directory, force_fresh=True)
    obj1 = dict(x=1)
    obj2 = dict(x=2)
    obj3 = [3]

    store.save_object('dict', 1, obj1)
    store.save_object('dict', 2, obj2)
    store.save_object('list', 1, obj3)

    with pytest.raises(ValueError):
        store.save_object('dict', 1, dict(x=4))
