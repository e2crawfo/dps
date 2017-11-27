import numpy as np
import clify
import argparse

from config import rl_config as config

from dps.envs import ga_no_modules

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", default="no_modules", choices="no_modules no_classifiers no_ops".split())
args, _ = parser.parse_known_args()

if args.ablation == 'no_modules':
    config.update(ga_no_modules.config_delta)

elif args.ablation == 'no_classifiers':
    # B
    pass

elif args.ablation == 'no_ops':
    # C
    pass
else:
    raise Exception("NotImplemented")

config.update(ablations=args.ablation)


grid = dict(n_train=2**np.arange(6, 18, 2))


from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
