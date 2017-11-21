import numpy as np
import os
import clify

from config import rl_config as config

A_dir = "/home/e2crawfo/rl_size_zero/"

A_load_paths = [
    os.path.join(A_dir, str(i), 'best_of_stage_0') for i in range(8)
]

B_dir = "/home/e2crawfo/rl_size_A/"
B_load_paths = [
    os.path.join(B_dir, d, 'best_of_stage_0') for d in os.listdir(B_dir)
]

G_dir = "/home/e2crawfo/rl_size_B/"
G_load_paths = [
    os.path.join(G_dir, d, 'best_of_stage_0') for d in os.listdir(G_dir)
]

zero_curric = [dict(env_shape=(2, 2))]
A_curric = [dict(env_shape=(3, 3), load_path=A_load_paths)]
C_curric = [dict(env_shape=(3, 3))]
B_curric = [dict(env_shape=(3, 3), min_digits=4, max_digits=4, load_path=B_load_paths)]
G_curric = [dict(env_shape=(3, 3), min_digits=5, max_digits=5, load_path=G_load_paths)]
F_curric = [dict(env_shape=(3, 3), min_digits=4, max_digits=4)]

config.update(
    load_path="",
    curriculum=G_curric,
    reductions="sum",
)


# grid = dict(n_train=2**np.arange(14, 18, 2))
grid = dict(n_train=2**np.arange(6, 18, 2))


from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
