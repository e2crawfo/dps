import numpy as np
import os
import clify

from config import rl_config as config

A_dir = "/home/e2crawfo/checkpoints/grid_parity_checkpoints/"

A_load_paths = [
    os.path.join(A_dir, i, 'best_of_stage_0') for i in range(5)
]


A_curric = [dict(parity='even')]
B_curric = [dict(parity='odd', load_path=A_load_paths)]
C_curric = [dict(parity='odd')]

config.update(
    load_path="",
    curriculum=A_curric,
    reductions="sum"
)


grid = dict(n_train=2**np.arange(14, 18, 2))
# grid = dict(n_train=2**np.arange(6, 18, 2))


from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
