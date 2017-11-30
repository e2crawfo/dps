import os
import subprocess
from collections import defaultdict

from dps.run import _run


def _get_deterministic_output(filename):
    """ Get values that we can count on to be the same on repeated runs with the same seed. """
    # pattern = ('best_01_loss\|best_2norm_loss\|best_reward_per_ep\|'
    #            'best_reward_per_ep_avg\|test_01_loss\|test_2_norm\|'
    #            'test_reward_per_ep\|constituting')
    pattern = 'best_01_loss'
    return subprocess.check_output(
        'grep "{}" {} | cat -n'.format(pattern, filename),
        shell=True).decode()


def test_simple_add(test_config):
    _config = dict(
        log_name="test_simple_add_a2c", render_step=0,
        value_weight=0.0, opt_steps_per_update=20,
        max_steps=501, use_gpu=False, seed=1034340)
    _config.update(test_config)

    n_repeats = 1  # Haven't made it completely deterministic yet, so keep it at 1.

    results = defaultdict(int)

    for i in range(n_repeats):
        config = _config.copy()
        output = _run("simple_addition", "a2c", _config=config)
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
