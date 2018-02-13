import numpy as np
import clify
import argparse

from config import rnn_config as config

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices="A B C D E F G H I".split(), default='')

args, _ = parser.parse_known_args()

config.update(
    n_controller_units=128,
    reductions="sum",
    image_shape_grid=(3, 3),
)

stage_0 = dict(draw_shape_grid=(2, 2))
stage_1 = dict()
stage_2 = dict(min_digits=4, max_digits=4)
stage_3 = dict(min_digits=5, max_digits=5)

grid = (
    [{'curriculum:-1:n_train': 1, 'curriculum:-1:do_train': False}] +
    [{'curriculum:-1:n_train': n} for n in 2**np.arange(0, 18, 2)]
)

if args.task == "A":
    config.curriculum = [
        stage_0.copy(),
        stage_1.copy()
    ]
elif args.task == "B":
    config.curriculum = [
        stage_0.copy(),
        stage_1.copy(),
        stage_2.copy()
    ]
elif args.task == "C":
    config.curriculum = [
    ]
    config.curriculum = [
        stage_0.copy(),
        stage_1.copy(),
        stage_2.copy(),
        stage_3.copy()
    ]
elif args.task == "D":
    config.curriculum = [
        stage_1.copy()
    ]
elif args.task == "E":
    config.curriculum = [
        stage_2.copy()
    ]
elif args.task == "F":
    config.curriculum = [
        stage_3.copy()
    ]
elif args.task == "G":
    config.curriculum = [
        dict(min_digits=1, max_digits=1),
        dict(min_digits=2, max_digits=2),
    ]
elif args.task == "H":
    config.curriculum = [
        dict(min_digits=2, max_digits=2),
    ]
elif args.task == "I":
    config.curriculum = [
        dict(min_digits=1, max_digits=1),
        dict(min_digits=2, max_digits=2),
        dict(min_digits=2, max_digits=3),
    ]
else:
    raise Exception()

for stage in config.curriculum[:-1]:
    stage['n_train'] = 2**17

from dps.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
