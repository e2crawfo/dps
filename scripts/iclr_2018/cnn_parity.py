import numpy as np
import clify

from config import cnn_config as config

A_curric = [dict(parity='even', n_train=2**17), dict(parity='odd')]
B_curric = [dict(parity='odd')]
config = config.copy(
    curriculum=A_curric,
    n_controller_units=512,
    reductions="sum",
)

grid = [
    {'curriculum:-1:n_train': n} for n in 2**np.arange(6, 18, 2)
]

from dps.parallel.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
