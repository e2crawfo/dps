import numpy as np
import clify

from config import cnn_config as config


A_curric = [dict(shape=(2, 2), n_train=2**17), dict(shape=(3, 3))]
B_curric = [dict(shape=(3, 3))]
C_curric = [
    dict(shape=(2, 2), n_train=2**17),
    dict(shape=(3, 3), n_train=2**17),
    dict(shape=(3, 3), min_digits=4, max_digits=4)]
F_curric = [dict(shape=(3, 3), min_digits=4, max_digits=4)]
G_curric = [
    dict(draw_shape=(2, 2), n_train=2**17),
    dict(draw_shape=(3, 3), n_train=2**17),
    dict(draw_shape=(3, 3), min_digits=4, max_digits=4, n_train=2**17),
    dict(draw_shape=(3, 3), min_digits=5, max_digits=5)
]

config.update(
    curriculum=G_curric,
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
