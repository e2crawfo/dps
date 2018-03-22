import pytest
import os
import numpy as np

from dps.hyper import build_and_submit
from dps.env.advanced import simple_addition
from dps.rl.algorithms import a2c
from dps.config import DEFAULT_CONFIG


@pytest.mark.slow
def test_hyper(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(a2c.config)
    config.update(simple_addition.config)
    config.update(test_config)
    config['max_steps'] = 101
    config['checkpoint_step'] = 43

    distributions = dict(n_train=2**np.array([5, 6, 7]))
    n_repeats = 2

    session = build_and_submit(
        name="test_hyper", config=config, distributions=distributions,
        n_repeats=n_repeats, kind='parallel', host_pool=':', wall_time='1year',
        cleanup_time='10mins', slack_time='10mins', ppn=2, load_avg_threshold=1e6)

    path = session.job_path
    files = os.listdir(path)
    assert set(files) == set(
        ['orig.zip', 'experiments', 'os_environ.txt', 'results.zip', 'pip_freeze.txt',
         'dps_git_summary.txt', 'nodefile.txt', 'results.txt', 'job_log.txt', 'uname.txt', 'lscpu.txt']
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
