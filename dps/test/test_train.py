import time
import os
import shutil
import subprocess

from dps.run import _run
from dps.rl.algorithms.a2c import reinforce_config
from dps.envs import translated_mnist
from dps.train import training_loop
from dps.envs import simple_addition
from dps.config import DEFAULT_CONFIG
from dps.utils.tf import get_tensors_from_checkpoint_file


def test_time_limit(test_config):
    config = DEFAULT_CONFIG.copy()
    config.update(simple_addition.config)
    config.update(reinforce_config)
    config.update(max_time=120, max_steps=10000, seed=100)
    config.update(test_config)

    start = time.time()
    with config:
        list(training_loop())
    elapsed = start - time.time()
    assert elapsed < config.max_time + 1


def grep(pattern, filename, options=""):
    return subprocess.check_output('grep {} "{}" {}'.format(options, pattern, filename), shell=True).decode()


def test_no_train(test_config):
    """ Test that variables stay fixed when we use use `ScopedFunction.fix_variables`. """

    digits = [0, 1]
    config = translated_mnist.config.copy(
        log_name="test_no_train", render_step=0, n_sub_image_examples=1000,
        value_weight=0.0, opt_steps_per_update=20, image_shape=(20, 20),
        max_steps=101, eval_step=10, use_gpu=False, seed=1034340,
        model_dir="/tmp/dps_test/models", n_train=100, digits=digits
    )
    config['emnist_config:threshold'] = 0.1
    config['emnist_config:n_train'] = 1000
    config['emnist_config:classes'] = digits
    try:
        shutil.rmtree(config.model_dir)
    except FileNotFoundError:
        pass

    config.update(test_config)
    config.update(fix_classifier=False, pretrain_classifier=True)

    # ------------- First run
    _config = config.copy(name="PART_1")
    output = _run("translated_mnist", "a2c", _config=_config)

    load_path1 = os.path.join(output['exp_dir'], 'best_of_stage_0')
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
        max_steps=2,
        do_train=False,
    )

    output = _run("translated_mnist", "a2c", _config=_config)
    load_path2 = os.path.join(output['exp_dir'], 'best_of_stage_0')
    tensors2 = get_tensors_from_checkpoint_file(load_path2)

    for key in relevant_keys:
        assert (tensors1[key] == tensors2[key]).all(), "Error on tensor with name {}".format(key)

    # ------------- Third run, reload, some training
    _config = config.copy(
        name="PART_3",
        fix_classifier=True,
        pretrain_classifier=False,
        load_path=load_path1,
        max_steps=101,
        do_train=True,
    )

    output = _run("translated_mnist", "a2c", _config=_config)
    load_path3 = os.path.join(output['exp_dir'], 'best_of_stage_0')
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
    load_path4 = os.path.join(output['exp_dir'], 'best_of_stage_0')
    tensors4 = get_tensors_from_checkpoint_file(load_path4)

    for key in relevant_keys:
        assert (tensors1[key] != tensors4[key]).any(), "Error on tensor with name {}".format(key)
