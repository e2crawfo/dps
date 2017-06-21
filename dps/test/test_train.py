import time

from dps.test.config import algorithms, tasks


def test_time_limit():
    config = tasks['simple_addition']
    config.update(algorithms['reinforce'])
    config.update(max_time=2, max_steps=10000, seed=100)

    start = time.time()
    with config:
        config.trainer.train()
    elapsed = start - time.time()
    assert elapsed < config.max_time + 1
