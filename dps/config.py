import numpy as np

from dps import cfg
from dps.utils import DpsConfig
from dps.rl import rl_render_hook


def get_experiment_name():
    name = []

    try:
        name.append('name={}'.format(cfg.name))
    except:
        pass

    try:
        name.append('seed={}'.format(cfg.seed))
    except:
        pass

    return '_'.join(name)


DEFAULT_CONFIG = DpsConfig(
    name="Default",
    seed=None,

    preserve_policy=True,  # Whether to use the policy learned on the last stage of the curriculum for each new stage.
    power_through=True,  # Whether to complete the entire curriculum, even if threshold not reached.

    slim=False,  # If true, tries to use little disk space
    patience=np.inf,

    render_step=np.inf,
    display_step=1000,
    eval_step=10,

    n_train=10000,
    n_val=500,
    batch_size=16,
    opt_steps_per_update=1,
    reward_window=0.499,
    threshold=0.01,

    gamma=1.0,
    noise_schedule=None,
    max_grad_norm=None,

    normalize_obs=False,
    standardize_advantage=True,
    reset_env=True,

    n_controller_units=64,

    stopping_function=None,

    max_time=0,
    max_steps=1000000,
    max_experiences=np.inf,

    deadline='',
    render_hook=rl_render_hook,

    get_experiment_name=get_experiment_name,
)


cfg._stack.append(DEFAULT_CONFIG)
