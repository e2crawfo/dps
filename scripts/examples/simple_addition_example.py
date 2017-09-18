from dps.train import training_loop
from dps.config import DEFAULT_CONFIG
from dps.rl.algorithms import a2c as alg
from dps.envs import simple_addition

config = DEFAULT_CONFIG.copy()
config.update(alg.config)
config.update(simple_addition.config)

config.update(
    max_steps=100000,
    threshold=-1,
    display_step=100,
)

with config:
    training_loop()
