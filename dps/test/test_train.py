import time

from dps.test.config import algorithms, tasks


def test_time_limit():
    config = tasks['simple_addition']
    config.update(algorithms['reinforce'])
    config.max_time = 2
    config.max_steps = 10000
    config.seed = 100

    start = time.time()
    config.trainer.train(config=config)
    elapsed = start - time.time()
    assert elapsed < config.max_time + 1
