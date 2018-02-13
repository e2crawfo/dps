import numpy as np
import clify

from config import rnn_config as config

grid = dict(n_train=2**np.arange(6, 18, 2), n_controller_units=[32, 128, 512])

from dps.hyper import build_and_submit, default_host_pool
clify.wrap_function(build_and_submit)(
    config=config, distributions=grid, n_param_settings=None, host_pool=default_host_pool)
