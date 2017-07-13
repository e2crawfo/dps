import pytest
import os
import shutil
from pathlib import Path
import subprocess
from contextlib import contextmanager

from dps.parallel.base import FileSystemObjectStore, ZipObjectStore


def test_fs_object_store():
    directory = Path('/tmp/test_fs_object_store/test')
    zip_path = directory.parent / 'my_store.zip'
    try:
        shutil.rmtree(str(directory))
    except:
        pass
    try:
        zip_path.unlink()
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

    store.zip(zip_path)

    zip_store = ZipObjectStore(zip_path)

    dicts = zip_store.load_objects('dict')
    assert len(dicts) == 2

    lists = zip_store.load_objects('list')
    assert len(lists) == 1

    zip_store = ZipObjectStore(zip_path)

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


@contextmanager
def remove_tree(path):
    path = os.path.abspath(path)
    try:
        yield
    finally:
        try:
            shutil.rmtree(path)
        except:
            pass


def run_cmd(cmd):
    try:
        process = subprocess.run(cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e.output.decode())
        raise e
    print(process.stdout.decode())
    return process


def test_hyper():
    path = '/tmp/dps/test_hyper'
    with remove_tree(path):
        cmd = 'dps-hyper build {} this_is_a_test 3 3 reinforce hello_world --use-gpu=0 --n-train=200'.format(path)
        process = run_cmd(cmd)

        cmd = 'dps-hyper run {}/latest _ --max-time=20'.format(path)
        process = run_cmd(cmd)

        cmd = 'dps-hyper view {}/latest'.format(path)
        process = run_cmd(cmd)
        output = process.stdout.decode()
        assert "n_ops: 9" in output
        assert "n_completed_ops: 9" in output
        assert "n_ready_incomplete_ops: 0" in output
        assert "n_not_ready_incomplete_ops: 0" in output
