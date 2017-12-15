import numpy as np
import clify
import argparse

from config import cnn_config as config

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices="A B C D E F".split(), default='')

args, _ = parser.parse_known_args()

if args.task == "A":
    config.curriculum = [
        dict(draw_shape=(2, 2), n_train=2**17),
        dict()
    ]
elif args.task == "B":
    config.curriculum = [
        dict(draw_shape=(2, 2), n_train=2**17),
        dict(n_train=2**17),
        dict(min_digits=4, max_digits=4)
    ]
elif args.task == "C":
    config.curriculum = [
        dict(draw_shape=(2, 2), n_train=2**17),
        dict(n_train=2**17),
        dict(n_train=2**17, min_digits=4, max_digits=4),
        dict(min_digits=5, max_digits=5)
    ]
elif args.task == "D":
    config.curriculum = [
        dict()
    ]
elif args.task == "E":
    config.curriculum = [
        dict(min_digits=4, max_digits=4)
    ]
elif args.task == "F":
    config.curriculum = [
        dict(min_digits=5, max_digits=5)
    ]
else:
    raise Exception()

config.update(
    n_controller_units=512,
    reductions="sum",
    env_shape=(3, 3),
)

grid = [
    {'curriculum:-1:n_train': n} for n in 2**np.arange(6, 18, 2)
]

from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
