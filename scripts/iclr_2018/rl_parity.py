import numpy as np
import os
import clify
import argparse

from config import rl_config as config

config.update(
    image_shape_grid=(2, 2),
    reductions="sum",
)

grid = [dict(n_train=1, do_train=False)] + [dict(n_train=x) for x in 2**np.arange(0, 18, 2)]

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices="A B C".split(), default='')

args, _ = parser.parse_known_args()


if args.task == "A":
    grid = dict(n_train=2**np.arange(14, 18, 2))
    config.update(parity='even')

elif args.task == "B":
    A_dir = "/home/e2crawfo/rl_parity_A/"
    config.load_path = [
        os.path.join(A_dir, d, 'weights/best_of_stage_0') for d in os.listdir(A_dir)
    ]
    config.update(parity='odd')

elif args.task == "C":
    config.update(parity='odd')

else:
    raise Exception()


from dps.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
