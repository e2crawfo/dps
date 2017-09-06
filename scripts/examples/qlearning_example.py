from dps.train import training_loop
from dps.config import DEFAULT_CONFIG
from dps.rl.algorithms import qlearning
from dps.envs import grid

config = DEFAULT_CONFIG.copy()
config.update(qlearning.config)
config.update(grid.config)

config.update(
    max_steps=100000,
    threshold=-1,
)

with config:
    training_loop()
