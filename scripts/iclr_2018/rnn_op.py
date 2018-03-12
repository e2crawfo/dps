import numpy as np
import clify
import argparse

from config import rnn_config as config

grid = (
    [{'curriculum:-1:n_train': 1, 'curriculum:-1:do_train': False}] +
    [{'curriculum:-1:n_train': n} for n in 2**np.arange(0, 18, 2)]
)

config = config.copy(
    n_controller_units=128,
)

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices="A B C D".split(), default='')

args, _ = parser.parse_known_args()

if args.task == "A":
    config.curriculum = [dict(reductions="sum", n_train=2**17), dict(reductions='prod')]
elif args.task == "B":
    config.curriculum = [dict(reductions="max", n_train=2**17), dict(reductions='prod')]
elif args.task == "C":
    config.curriculum = [dict(reductions="min", n_train=2**17), dict(reductions='prod')]
elif args.task == "D":
    config.curriculum = [dict(reductions='prod')]
else:
    raise Exception()

from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
