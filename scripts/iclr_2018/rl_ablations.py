import numpy as np
import clify
import argparse

from config import rl_config as config

from dps.env import ga_no_modules, ga_no_classifiers, ga_no_transformations

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", default="no_modules", choices="no_modules no_classifiers no_transformations".split())
parser.add_argument("--search", action="store_true")
args, _ = parser.parse_known_args()

if args.ablation == 'no_modules':
    config.update(ga_no_modules.config_delta)

elif args.ablation == 'no_classifiers':
    config.update(ga_no_classifiers.config_delta)

elif args.ablation == 'no_transformations':
    config.update(ga_no_transformations.config_delta)

else:
    raise Exception("NotImplemented")

config.update(ablations=args.ablation, reductions="sum")


if args.search:
    config.n_train = 2**10
    grid = dict(lr_schedule=[1e-3, 1e-4, 1e-5], value_weight=[0, 1])
else:
    grid = dict(n_train=2**np.arange(6, 18, 2))


from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
