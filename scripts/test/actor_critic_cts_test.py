from dps.train import training_loop
from dps.config import DEFAULT_CONFIG
from dps.rl.algorithms import a2c
from dps.envs import room

config = DEFAULT_CONFIG.copy()
config.update(a2c.actor_critic_config)
config.update(room.config)

config.update(
    max_steps=100000,
    threshold=-1,
    policy_weight=1.0,
    value_weight=1.0,
    lmbda=1.0,
    opt_steps_per_update=1,
    exploration_schedule=1.0,
)

with config:
    training_loop()
