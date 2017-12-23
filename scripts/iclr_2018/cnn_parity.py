import numpy as np
import clify
import argparse

from config import cnn_config as config

grid = (
    [{'curriculum:-1:n_train': 1, 'curriculum:-1:do_train': False}] +
    [{'curriculum:-1:n_train': n} for n in 2**np.arange(0, 18, 2)]
)

config = config.copy(
    n_controller_units=512,
    reductions="sum",
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices="B C".split(), default='')

args, _ = parser.parse_known_args()

if args.task == "B":
    config.curriculum = [dict(parity='even', n_train=2**17), dict(parity='odd')]
elif args.task == "C":
    config.curriculum = [dict(parity='odd')]
else:
    raise Exception()

from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
