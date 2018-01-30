import os
import subprocess
from collections import defaultdict
import pytest

from dps.run import _raw_run
from dps.utils import Config
from dps.rl.policy import BuildEpsilonSoftmaxPolicy, BuildLstmController
from dps.rl.algorithms import a2c
from dps.env.advanced import simple_addition


def _get_deterministic_output(filename):
    """ Get values that we can count on to be the same on repeated runs with the same seed. """
    # pattern = ('best_01_loss\|best_2norm_loss\|best_reward_per_ep\|'
    #            'best_reward_per_ep_avg\|test_01_loss\|test_2_norm\|'
    #            'test_reward_per_ep\|constituting')
    pattern = 'best_01_loss'
    return subprocess.check_output(
        'grep "{}" {} | cat -n'.format(pattern, filename),
        shell=True).decode()


@pytest.mark.slow
def test_simple_add(test_config):
    # Fully specify the config here so that this test is not affected by config changes external to this file.
    config = Config(
        log_name="test_simple_add_a2c",
        name="test_simple_add_a2c",
        get_updater=a2c.A2C,
        n_controller_units=32,
        batch_size=16,
        optimizer_spec="adam",
        opt_steps_per_update=20,
        sub_batch_size=0,
        epsilon=0.2,
        lr_schedule=1e-4,

        max_steps=501,

        build_policy=BuildEpsilonSoftmaxPolicy(),
        build_controller=BuildLstmController(),

        exploration_schedule=0.1,
        val_exploration_schedule=0.0,
        actor_exploration_schedule=None,

        policy_weight=1.0,
        value_weight=0.0,
        value_reg_weight=0.0,
        entropy_weight=0.01,

        split=False,
        q_lmbda=1.0,
        v_lmbda=1.0,
        policy_importance_c=0,
        q_importance_c=None,
        v_importance_c=None,
        max_grad_norm=None,
        gamma=1.0,

        use_differentiable_loss=False,

        use_gpu=False,
        render_step=0,
        seed=1034340,

        # env-specific
        build_env=simple_addition.build_env,
        T=30,
        curriculum=[
            dict(width=1),
            dict(width=2),
            dict(width=3),
        ],
        base=10,
        final_reward=True,
    )

    config.update(test_config)

    n_repeats = 1  # Haven't made it completely deterministic yet, so keep it at 1.

    results = defaultdict(int)

    for i in range(n_repeats):
        config = config.copy()
        output = _raw_run(config)
        stdout = os.path.join(output['exp_dir'], 'stdout')
        result = _get_deterministic_output(stdout)
        results[result] += 1
        assert output['history'][-1]['best_01_loss'] < 0.1

    if len(results) != 1:
        for r in sorted(results):
            print("\n" + "*" * 80)
            print("The following occurred {} times:\n".format(results[r]))
            print(r)
        raise Exception("Results were not deterministic.")

    assert len(output['config'].curriculum) == 3
    config.load_path = os.path.join(output['exp_dir'], 'best_of_stage_2')
    assert os.path.exists(config.load_path + ".index")
    assert os.path.exists(config.load_path + ".meta")

    # Load one of the hypotheses, train it for a bit, make sure the accuracy is still high.
    config.curriculum = [output['config'].curriculum[-1]]
    config = config.copy()
    output = _raw_run(config)
    stdout = os.path.join(output['exp_dir'], 'stdout')
    result = _get_deterministic_output(stdout)
    results[result] += 1
    assert output['history'][-1]['best_01_loss'] < 0.1

    # Load one of the hypotheses, don't train it at all, make sure the accuracy is still high.
    config.do_train = False
    config = config.copy()
    output = _raw_run(config)
    stdout = os.path.join(output['exp_dir'], 'stdout')
    result = _get_deterministic_output(stdout)
    results[result] += 1
    assert output['history'][-1]['best_01_loss'] < 0.1
