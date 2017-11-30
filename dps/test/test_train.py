import time

from dps.rl.algorithms.a2c import reinforce_config
from dps.train import training_loop
from dps.envs import simple_addition
from dps.config import DEFAULT_CONFIG


def test_time_limit():
    config = DEFAULT_CONFIG.copy()
    config.update(simple_addition.config)
    config.update(reinforce_config)
    config.update(max_time=10, max_steps=10000, seed=100)

    start = time.time()
    with config:
        list(training_loop())
    elapsed = start - time.time()
    assert elapsed < config.max_time + 1
