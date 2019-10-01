import pytest
import os
import numpy as np

from dps.train import training_loop
from dps.hyper import build_and_submit
from dps.config import DEFAULT_CONFIG
from dps.utils import remove
from dps.iris_example import iris_config, mlp_config


@pytest.mark.slow
def test_hyper_step_limited(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(iris_config)
    config.update(mlp_config)
    config.update(test_config)
    config['max_steps'] = 101
    config['checkpoint_step'] = 43

    with config:
        training_loop()

    distributions = dict(n_train=2**np.array([5, 6, 7]))
    n_repeats = 2

    session = build_and_submit(
        category="test_hyper", exp_name="test_hyper", config=config, distributions=distributions,
        n_repeats=n_repeats, kind='parallel', host_pool=':', wall_time='1year',
        cleanup_time='10mins', slack_time='10mins', ppn=2, load_avg_threshold=1e6)

    path = session.job_path
    with remove(path):
        files = os.listdir(path)
        assert set(files) == set(
            'config.json experiments nodefile.txt results.txt run_kwargs.json session.pkl stdout '
            'context job_log.txt orig.zip results.zip sampled_configs.txt stderr'.split()
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


@pytest.mark.slow
def test_hyper_time_limited(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(iris_config)
    config.update(mlp_config)
    config.update(test_config)

    config['max_steps'] = 101

    with config:
        training_loop()

    config['max_steps'] = 100000
    config['checkpoint_step'] = 43

    distributions = dict(n_train=2**np.array([5, 6, 7]))
    n_repeats = 2

    session = build_and_submit(
        category="test_hyper", exp_name="test_hyper", config=config, distributions=distributions,
        n_repeats=n_repeats, kind='parallel', host_pool=':', wall_time='1min',
        cleanup_time='10seconds', slack_time='5seconds', ppn=2, load_avg_threshold=1e6)

    path = session.job_path
    with remove(path):
        files = os.listdir(path)
        assert set(files) == set(
            'config.json experiments nodefile.txt results.txt run_kwargs.json session.pkl stdout '
            'context job_log.txt orig.zip results.zip sampled_configs.txt stderr'.split()
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
