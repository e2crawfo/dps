import numpy as np
import os
import clify
import argparse

from config import rl_config as config

config.update(
    image_shape_grid=(3, 3),
    reductions="sum",
)

grid = [dict(n_train=1, do_train=False)] + [dict(n_train=x) for x in 2**np.arange(0, 18, 2)]

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices="A B C D E F 0".split(), default='')

args, _ = parser.parse_known_args()

stage_1 = dict()
stage_2 = dict(min_digits=4, max_digits=4)
stage_3 = dict(min_digits=5, max_digits=5)

if args.task == "0":
    grid = dict(n_train=2**np.arange(14, 18, 2))
    config.update(image_shape_grid=(2, 2))

elif args.task == "A":
    zero_dir = "/home/e2crawfo/rl_size_0/"
    config.load_path = [
        os.path.join(zero_dir, d, 'weights/best_of_stage_0') for d in os.listdir(zero_dir)
    ]
    config.update(stage_1)

elif args.task == "B":
    A_dir = "/home/e2crawfo/rl_size_A/"
    config.load_path = [
        os.path.join(A_dir, d, 'weights/best_of_stage_0') for d in os.listdir(A_dir)
    ]
    config.update(stage_2)

elif args.task == "C":
    B_dir = "/home/e2crawfo/rl_size_B/"
    config.load_path = [
        os.path.join(B_dir, d, 'weights/best_of_stage_0') for d in os.listdir(B_dir)
    ]
    config.update(stage_3)

elif args.task == "D":
    config.update(stage_1)
elif args.task == "E":
    config.update(stage_2)
elif args.task == "F":
    config.update(stage_3)
else:
    raise Exception()


from dps.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
