import time
import subprocess
import pytest

from dps.train import training_loop, Hook
from dps.iris_example import iris_config, mlp_config
from dps.config import DEFAULT_CONFIG
from dps.utils import Alarm


def _run(config):
    with config:
        return training_loop()


@pytest.mark.slow
def test_time_limit(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(iris_config, **mlp_config)
    config.update(max_time=120, max_steps=10000, seed=100)
    config.update(test_config)

    start = time.time()
    with config:
        training_loop()
    elapsed = start - time.time()
    assert elapsed < config.max_time + 1


class AlarmHook(Hook):
    def __init__(self, start, stage_idx):
        self.start = start
        self.stage_idx = stage_idx
        super(AlarmHook, self).__init__()

    def start_stage(self, training_loop, updater, stage_idx):
        if self.start and stage_idx == self.stage_idx:
            raise Alarm("Raised by AlarmHook")

    def end_stage(self, training_loop, updater, stage_idx):
        if not self.start and stage_idx == self.stage_idx:
            raise Alarm("Raised by AlarmHook")


@pytest.mark.slow
def test_time_limit_between_stages(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(iris_config, **mlp_config)
    config.update(max_time=120, max_steps=10, seed=100)
    config.update(hooks=[AlarmHook(False, 0)])
    config.update(test_config)

    start = time.time()
    with config:
        result = training_loop()
    print(result)
    elapsed = start - time.time()
    assert elapsed < 20


class DummyHook(Hook):
    def __init__(self, n_stages, base_config):
        self.n_stages = n_stages
        self.base_config = base_config
        super(DummyHook, self).__init__()

    def _attrs(self):
        return "n_stages base_config".split()

    def end_stage(self, training_loop, updater, stage_idx):
        if stage_idx < self.n_stages - 1:
            training_loop.edit_remaining_stage(0, self.base_config)


def test_stage_hook(test_config):
    """ Test that we can safely use hooks to add new stages. """
    config = DEFAULT_CONFIG.copy()
    config.update(iris_config, **mlp_config)
    config.update(
        max_steps=11, eval_step=10, n_train=100, seed=100,
        hooks=[DummyHook(3, dict(max_steps=21))],
        curriculum=[dict()],
        width=1,
    )
    config.update(test_config)

    with config:
        data = training_loop()
        assert data.n_stages == 3
        assert not data.history[0]["stage_config"]
        assert data.history[1]["stage_config"]["max_steps"] == 21
        assert data.history[2]["stage_config"]["max_steps"] == 21


def grep(pattern, filename, options=""):
    return subprocess.check_output(
        'grep {} "{}" {}'.format(options, pattern, filename),
        shell=True).decode()


def test_train_data(test_config):
    config = test_config.copy()
    config.update(iris_config, **mlp_config)
    config.update(max_steps=101, checkpoint_step=43, eval_step=100)
    lr_schedule = config.lr_schedule
    config.update(
        curriculum=[dict(), dict(lr_schedule=.3*lr_schedule), dict(lr_schedule=.09*lr_schedule)]
    )

    frozen_data = _run(config)

    # train

    train0 = frozen_data.step_data('train', 0)
    assert train0.shape[0] == 2
    assert (train0['stage_idx'] == 0).all()

    train1 = frozen_data.step_data('train', 1)
    assert train1.shape[0] == 2
    assert (train1['stage_idx'] == 1).all()

    train2 = frozen_data.step_data('train', 2)
    assert train2.shape[0] == 2
    assert (train2['stage_idx'] == 2).all()

    trainNone = frozen_data.step_data('train', None)
    assert trainNone.shape[0] == 6

    assert trainNone.loc[0, 'stage_idx'] == 0
    assert trainNone.loc[0, 'local_step'] == 0
    assert trainNone.loc[0, 'global_step'] == 0

    assert trainNone.iloc[-1]['stage_idx'] == 2
    assert trainNone.iloc[-1]['local_step'] == 100
    assert trainNone.iloc[-1]['global_step'] == 302

    train03 = frozen_data.step_data('train', (0, 3))
    assert (trainNone == train03).all().all()

    trainSlice03 = frozen_data.step_data('train', slice(0, 3))
    assert (trainNone == trainSlice03).all().all()

    # val

    val0 = frozen_data.step_data('val', 0)
    assert val0.shape[0] == 2
    assert (val0['stage_idx'] == 0).all()

    val1 = frozen_data.step_data('val', 1)
    assert val1.shape[0] == 2
    assert (val1['stage_idx'] == 1).all()

    val2 = frozen_data.step_data('val', 2)
    assert val2.shape[0] == 2
    assert (val2['stage_idx'] == 2).all()

    valNone = frozen_data.step_data('val', None)
    assert valNone.shape[0] == 6

    assert valNone.loc[0, 'stage_idx'] == 0
    assert valNone.loc[0, 'local_step'] == 0
    assert valNone.loc[0, 'global_step'] == 0

    assert valNone.iloc[-1]['stage_idx'] == 2
    assert valNone.iloc[-1]['local_step'] == 100
    assert valNone.iloc[-1]['global_step'] == 302

    val03 = frozen_data.step_data('val', (0, 3))
    assert (valNone == val03).all().all()

    valSlice03 = frozen_data.step_data('val', slice(0, 3))
    assert (valNone == valSlice03).all().all()
