import time

from dps.train import training_loop
from dps.config import actor_configs, tasks


def test_time_limit():
    config = tasks['simple_addition']
    config.update(actor_configs['reinforce'])
    config.update(max_time=2, max_steps=10000, seed=100)

    start = time.time()
    with config:
        training_loop()
    elapsed = start - time.time()
    assert elapsed < config.max_time + 1
