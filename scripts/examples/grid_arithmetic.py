from dps.train import training_loop
from dps.config import DEFAULT_CONFIG
from dps.rl.policy import BuildEpsilonSoftmaxPolicy
from dps.rl.algorithms.a2c import config as alg_config
from dps.envs.grid_arithmetic import config as env_config

config = DEFAULT_CONFIG.copy()
config.update(alg_config)
config.update(env_config)

config.update(
    max_steps=1000000,
    threshold=-1,
    display_step=100,
    symbols=[('A', lambda x: sum(x))],
    opt_steps_per_update=10,
    n_controller_units=32,
    min_digits=2,
    max_digits=2,
    shape=(3, 1),
    visible_glimpse=True,
    build_policy=BuildEpsilonSoftmaxPolicy(),
    exploration_schedule='Poly(1.0, 1000000, end=0.01)',
    gamma=0.98,
    T=30,
    arithmetic_actions=[
        ('+', lambda acc, digit: acc + digit),
    ],
)

with config:
    training_loop()
