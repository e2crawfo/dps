import numpy as np
import configparser
import socket
from pathlib import Path

import dps
from dps import cfg
from dps.utils import Config, process_path
from dps.rl import rl_render_hook


class SystemConfig(Config):
    def __init__(self, _d=None, **kwargs):
        config = _load_system_config()
        if _d:
            config.update(_d)
        config.update(kwargs)
        super(SystemConfig, self).__init__(**config)


def _load_system_config(key=None):
    _config = configparser.ConfigParser()
    location = Path(dps.__file__).parent
    _config.read(str(location / 'config.ini'))

    if not key:
        key = socket.gethostname()

    if 'travis' in key:
        key = 'travis'

    if key not in _config:
        key = 'DEFAULT'

    # Load default configuration from a file
    config = Config(
        hostname=socket.gethostname(),
        start_tensorboard=_config.getboolean(key, 'start_tensorboard'),
        reload_interval=_config.getint(key, 'reload_interval'),
        update_latest=_config.getboolean(key, 'update_latest'),
        save_summaries=_config.getboolean(key, 'save_summaries'),
        data_dir=process_path(_config.get(key, 'data_dir')),
        model_dir=process_path(_config.get(key, 'model_dir')),
        build_experiments_dir=process_path(_config.get(key, 'build_experiments_dir')),
        run_experiments_dir=process_path(_config.get(key, 'run_experiments_dir')),
        log_root=process_path(_config.get(key, 'log_root')),
        show_plots=_config.getboolean(key, 'show_plots'),
        save_plots=_config.getboolean(key, 'save_plots'),
        use_gpu=_config.getboolean(key, 'use_gpu'),
        tbport=_config.getint(key, 'tbport'),
        verbose=_config.getboolean(key, 'verbose'),
        per_process_gpu_memory_fraction=_config.getfloat(key, 'per_process_gpu_memory_fraction'),
        gpu_allow_growth=_config.getboolean(key, 'gpu_allow_growth'),
    )

    config.max_experiments = _config.getint(key, 'max_experiments')
    if config.max_experiments <= 0:
        config.max_experiments = np.inf
    return config


def get_experiment_name():
    name = []

    try:
        name.append('name={}'.format(cfg.name))
    except Exception:
        pass

    return '_'.join(name)


DEFAULT_CONFIG = SystemConfig(
    name="Default",
    seed=-1,

    curriculum=[{}],

    load_path="",  # Path to load variables from.
    do_train=True,
    preserve_policy=True,  # Whether to use the policy learned on the last stage of the curriculum for each new stage.
    preserve_env=False,
    power_through=True,  # Whether to complete the entire curriculum, even if threshold not reached.

    slim=False,  # If true, tries to use little disk space
    patience=np.inf,

    render_step=np.inf,
    display_step=100,
    eval_step=100,
    checkpoint_step=5000,

    n_train=10000,
    n_val=500,
    batch_size=16,
    opt_steps_per_update=1,
    reward_window=0.499,
    threshold=0.01,

    gamma=1.0,
    noise_schedule=None,
    max_grad_norm=None,

    standardize_advantage=True,
    reset_env=True,

    n_controller_units=64,

    max_time=0,
    max_steps=1000000,
    max_experiences=np.inf,

    deadline='',
    render_n_rollouts=10,
    render_hook=rl_render_hook,

    get_experiment_name=get_experiment_name,
    error_on_timeout=False,

    stopping_criteria_name="",
    tee=True,  # If True, output of training run (stdout and stderr) will is written to screen as
               # well as a file. If False, only written to the file.
)


RL_EXPERIMENT_CONFIG = DEFAULT_CONFIG.copy(
    name="RLExperiment",

    display_step=100,
    eval_step=100,
    patience=np.inf,
    max_steps=1000000,
    power_through=False,
    preserve_policy=True,

    error_on_timeout=False,
)


SL_EXPERIMENT_CONFIG = RL_EXPERIMENT_CONFIG.copy(
    name="SLExperiment",
    error_on_timeout=True,
    patience=5000,
)


cfg._stack.append(DEFAULT_CONFIG)
