import time
import shutil
import subprocess
import pytest

from dps.run import _run
from dps.rl.algorithms.a2c import reinforce_config
from dps.train import training_loop, Hook
from dps.env.advanced import translated_mnist
from dps.env.advanced import simple_addition
from dps.config import DEFAULT_CONFIG
from dps.utils.tf import get_tensors_from_checkpoint_file


@pytest.mark.slow
def test_time_limit(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(simple_addition.config)
    config.update(reinforce_config)
    config.update(max_time=120, max_steps=10000, seed=100)
    config.update(test_config)

    start = time.time()
    with config:
        training_loop()
    elapsed = start - time.time()
    assert elapsed < config.max_time + 1


class DummyHook(Hook):
    def __init__(self, n_stages, base_config):
        self.n_stages = n_stages
        self.base_config = base_config
        super(DummyHook, self).__init__()

    def _attrs(self):
        return "n_stages base_config".split()

    def end_stage(self, training_loop, stage_idx):
        if stage_idx < self.n_stages - 1:
            training_loop.add_stage(self.base_config.copy())


def test_stage_hook(test_config):
    """ Test that we can safely use hooks to add new stages. """
    config = DEFAULT_CONFIG.copy()
    config.update(simple_addition.config)
    config.update(reinforce_config)
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
    config.update(max_steps=101, checkpoint_step=43, eval_step=100)

    frozen_data = _run('hello_world', 'a2c', _config=config)

    # train

    train0 = frozen_data.step_data('train', 0)
    assert train0.shape[0] == 101
    assert (train0['stage_idx'] == 0).all()

    train1 = frozen_data.step_data('train', 1)
    assert train1.shape[0] == 101
    assert (train1['stage_idx'] == 1).all()

    train2 = frozen_data.step_data('train', 2)
    assert train2.shape[0] == 101
    assert (train2['stage_idx'] == 2).all()

    trainNone = frozen_data.step_data('train', None)
    assert trainNone.shape[0] == 303

    assert trainNone.ix[0, 'stage_idx'] == 0
    assert trainNone.ix[0, 'local_step'] == 0
    assert trainNone.ix[0, 'global_step'] == 0

    assert trainNone.iloc[-1]['stage_idx'] == 2
    assert trainNone.iloc[-1]['local_step'] == 100
    assert trainNone.iloc[-1]['global_step'] == 302

    train03 = frozen_data.step_data('train', (0, 3))
    assert (trainNone == train03).all().all()

    trainSlice03 = frozen_data.step_data('train', slice(0, 3))
    assert (trainNone == trainSlice03).all().all()

    # off_policy

    off_policy = frozen_data.step_data('off_policy', 0)
    assert off_policy is None

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

    assert valNone.ix[0, 'stage_idx'] == 0
    assert valNone.ix[0, 'local_step'] == 0
    assert valNone.ix[0, 'global_step'] == 0

    assert valNone.iloc[-1]['stage_idx'] == 2
    assert valNone.iloc[-1]['local_step'] == 100
    assert valNone.iloc[-1]['global_step'] == 302

    val03 = frozen_data.step_data('val', (0, 3))
    assert (valNone == val03).all().all()

    valSlice03 = frozen_data.step_data('val', slice(0, 3))
    assert (valNone == valSlice03).all().all()

    # test

    test0 = frozen_data.step_data('test', 0)
    assert test0.shape[0] == 2
    assert (test0['stage_idx'] == 0).all()

    test1 = frozen_data.step_data('test', 1)
    assert test1.shape[0] == 2
    assert (test1['stage_idx'] == 1).all()

    test2 = frozen_data.step_data('test', 2)
    assert test2.shape[0] == 2
    assert (test2['stage_idx'] == 2).all()

    testNone = frozen_data.step_data('test', None)
    assert testNone.shape[0] == 6

    assert testNone.ix[0, 'stage_idx'] == 0
    assert testNone.ix[0, 'local_step'] == 0
    assert testNone.ix[0, 'global_step'] == 0

    assert testNone.iloc[-1]['stage_idx'] == 2
    assert testNone.iloc[-1]['local_step'] == 100
    assert testNone.iloc[-1]['global_step'] == 302

    test03 = frozen_data.step_data('test', (0, 3))
    assert (testNone == test03).all().all()

    testSlice03 = frozen_data.step_data('test', slice(0, 3))
    assert (testNone == testSlice03).all().all()


@pytest.mark.slow
def test_fixed_variables(test_config):
    """ Test that variables stay fixed when we use use `ScopedFunction.fix_variables`. """

    digits = [0, 1]
    config = translated_mnist.config.copy(
        log_name="test_fixed_variables", render_step=0, n_sub_image_examples=1000,
        value_weight=0.0, opt_steps_per_update=20, image_shape=(20, 20),
        max_steps=101, eval_step=10, use_gpu=False, seed=1034340,
        model_dir="/tmp/dps_test/models", n_train=100, digits=digits
    )
    config['emnist_config:threshold'] = 0.1
    config['emnist_config:n_train'] = 1000
    config['emnist_config:classes'] = digits
    config.emnist_config.update(test_config)
    try:
        shutil.rmtree(config.model_dir)
    except FileNotFoundError:
        pass

    config.update(test_config)
    config.update(fix_classifier=False, pretrain_classifier=True)

    # ------------- First run
    _config = config.copy(name="PART_1")
    output = _run("translated_mnist", "a2c", _config=_config)

    load_path1 = output.path_for('weights/best_of_stage_0')
    tensors1 = get_tensors_from_checkpoint_file(load_path1)

    prefix = "{}/digit_classifier".format(translated_mnist.AttentionClassifier.__name__)
    relevant_keys = [key for key in tensors1 if key.startswith(prefix)]
    assert len(relevant_keys) > 0

    # ------------- Second run, reload, no training
    _config = config.copy(
        name="PART_2",
        fix_classifier=True,
        pretrain_classifier=False,
        load_path=load_path1,
        do_train=False,
    )

    output = _run("translated_mnist", "a2c", _config=_config)
    load_path2 = output.path_for('weights/best_of_stage_0')
    tensors2 = get_tensors_from_checkpoint_file(load_path2)

    for key in relevant_keys:
        assert (tensors1[key] == tensors2[key]).all(), "Error on tensor with name {}".format(key)

    # ------------- Third run, reload with variables fixed, do some training, assert that variables haven't changed
    _config = config.copy(
        name="PART_3",
        fix_classifier=True,
        pretrain_classifier=False,
        load_path=load_path1,
        max_steps=101,
        do_train=True,
    )

    output = _run("translated_mnist", "a2c", _config=_config)
    load_path3 = output.path_for('weights/best_of_stage_0')
    tensors3 = get_tensors_from_checkpoint_file(load_path3)

    for key in relevant_keys:
        assert (tensors1[key] == tensors3[key]).all(), "Error on tensor with name {}".format(key)

    # ------------- Fourth run, reload with variables NOT fixed, do some training, assert that the variables are different
    _config = config.copy(
        name="PART_4",
        fix_classifier=False,
        pretrain_classifier=False,
        load_path=load_path1,
        max_steps=101,
        do_train=True,
    )

    output = _run("translated_mnist", "a2c", _config=_config)
    load_path4 = output.path_for('weights/best_of_stage_0')
    tensors4 = get_tensors_from_checkpoint_file(load_path4)

    for key in relevant_keys:
        assert (tensors1[key] != tensors4[key]).any(), "Error on tensor with name {}".format(key)
