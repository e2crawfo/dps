import argparse
import numpy as np

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.experiments import alt_arithmetic
from dps.utils import Config, pdb_postmortem
from dps.rl import QLearning, rl_render_hook
from dps.train import training_loop
from dps.config import LstmController, epsilon_greedy, DuelingLstmController

max_steps = 100000

config = DEFAULT_CONFIG.copy(
    n_val=500,
    controller=LstmController(),
    exploration_schedule='poly 1.0 100000 1.0',
    get_experiment_name=lambda: "name={}_seed={}".format(cfg.actor_config.name, cfg.seed),
    batch_size=32,  # Number of sample experiences per update
    max_steps=max_steps,

    actor_config=Config(
        name="QLearning",
        alg=QLearning,

        action_selection=epsilon_greedy,
        n_controller_units=128,
        controller=DuelingLstmController(),
        double=True,

        lr_schedule="0.00025",
        exploration_schedule="poly 1.0 50000 0.1",
        test_time_explore="0.01",

        optimizer_spec="adam",

        gamma=1.0,

        init_steps=5000,

        opt_steps_per_batch=10,
        target_update_rate=0.01,
        steps_per_target_update=None,
        patience=np.inf,
        update_batch_size=32,  # Number of sample rollouts to use for each parameter update

        replay_max_size=5000,
        alpha=0.7,
        beta_schedule="0.5",
        n_partitions=100,

        max_grad_norm=0.0,
    ),

    build_env=alt_arithmetic.build_env,

    curriculum=[
        dict(T=30, min_digits=2, max_digits=3, shape=(2, 2), dense_reward=True),
    ],
    test_time_explore=1.0,
    display=False,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    force_2d=False,
    symbols=[('M', lambda r: np.product(r))],
    base=10,
    threshold=0.01,
    reward_window=0.4,
    ablation='',  # anything other than "bad_wiring", "no_classifiers", "no_ops", "no_modules" will use the default.
    log_name='alt_arithmetic',
    visualize=True,
    render_hook=rl_render_hook,
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
