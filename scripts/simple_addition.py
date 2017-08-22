import argparse
import numpy as np

import clify

from dps import cfg
from dps.train import training_loop
from dps.rl import QLearning, Retrace
from dps.config import DuelingLstmController, DEFAULT_CONFIG, epsilon_greedy
from dps.experiments import simple_addition
from dps.utils import pdb_postmortem, Config


config = DEFAULT_CONFIG.copy(
    # name="QLearning",
    # alg=QLearning,

    name="Retrace",
    alg=Retrace,

    controller=DuelingLstmController(),
    # controller=FeedforwardController(),

    lr_schedule="0.001",
    init_steps=5000,
    lmbda=1.0,
    test_time_explore=0.10,
    exploration_schedule="poly 1.0 5000 0.1",
    greedy_factor=100.0,
    beta_schedule=1.0,
    alpha=0.7,

    action_selection=epsilon_greedy,
    n_controller_units=64,
    double=False,

    optimizer_spec="adam",

    gamma=1.0,

    opt_steps_per_batch=10,
    steps_per_target_update=1000,
    target_update_rate=0.001,
    patience=np.inf,
    update_batch_size=32,  # Number of sample rollouts to use for each parameter update
    batch_size=32,  # Number of sample experiences per update

    actor_config=Config(),

    replay_max_size=20000,

    max_grad_norm=0.0,

    build_env=simple_addition.build_env,
    log_name='simple_addition',
    eval_step=10,

    dense_reward=True,
    T=30,
    curriculum=[
        dict(width=1, base=10, threshold=-1.0),
        dict(width=2, base=10, threshold=-1.0),
        dict(width=3, base=10, threshold=-1.0),
    ],

    max_steps=100000
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', action='store_true',
                        help="If supplied, enter post-mortem debugging on error.")
    args, _ = parser.parse_known_args()

    with config:
        cl_args = clify.wrap_object(cfg).parse()
        cfg.update(cl_args)

        if args.pdb:
            with pdb_postmortem():
                training_loop()
        else:
            training_loop()
