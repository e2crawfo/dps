import pytest
import os
import shutil
from pathlib import Path
from contextlib import contextmanager
import numpy as np

from dps.parallel.base import FileSystemObjectStore, ZipObjectStore
from dps.parallel.hyper import build_and_submit
from dps.envs import simple_addition
from dps.rl.algorithms import a2c
from dps.config import DEFAULT_CONFIG


def test_fs_object_store():
    directory = Path('/tmp/test_fs_object_store/test')
    zip_path = directory.parent / 'my_store.zip'
    shutil.rmtree(str(directory), ignore_errors=True)

    try:
        zip_path.unlink()
    except Exception:
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
    shutil.rmtree(str(directory), ignore_errors=True)
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
        shutil.rmtree(str(path), ignore_errors=True)


def test_hyper(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(a2c.config)
    config.update(simple_addition.config)
    config.update(test_config)
    config['max_steps'] = 101

    distributions = dict(n_train=2**np.array([5, 6, 7]))
    session = build_and_submit(
        name="test_hyper", config=config, distributions=distributions, n_repeats=2,
        kind='parallel', host_pool=':', wall_time='1year', cleanup_time='10mins',
        slack_time='10mins', ppn=2)

    path = session.exp_dir.path
    files = os.listdir(path)
    assert set(files) == set(
        ['orig.zip', 'experiments', 'os_environ.txt', 'results.zip', 'pip_freeze.txt',
         'dps_git_summary.txt', 'nodefile.txt', 'results.txt', 'job_log.txt']
    )

    with open(os.path.join(path, 'results.txt'), 'r') as f:
        results = f.read()

    assert "n_ops: 6" in results
    assert "n_completed_ops: 6" in results
    assert "n_partially_completed_ops: 0" in results
    assert "n_ready_incomplete_ops: 0" in results
    assert "n_not_ready_incomplete_ops: 0" in results
