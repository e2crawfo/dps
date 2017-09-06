import numpy as np

from dps.train import training_loop
from dps.config import DEFAULT_CONFIG
from dps.rl.algorithms import qlearning
from dps.envs import cliff_walk
from dps.rl.policy import BuildLinearController

config = DEFAULT_CONFIG.copy()
config.update(qlearning.config)
config.update(cliff_walk.config)

config.update(
    visualize=True,

    T=20,
    width=4,
    n_actions=2,

    build_controller=BuildLinearController(),
    max_steps=100000,
    threshold=-1000,

    steps_per_target_update=1,
    beta_schedule=0.0,
    alpha=0.0,
    double=False,
    reverse_double=False,
    exploration_schedule=1.0,
)


np.random.seed(10)
config.order = np.random.randint(config.n_actions, size=config.width)
np.random.seed()


with config:
    training_loop()
