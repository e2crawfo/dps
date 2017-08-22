import argparse
import numpy as np

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.experiments import alt_arithmetic
from dps.utils import Config, pdb_postmortem
from dps.rl import PPO, rl_render_hook
from dps.rl.value import TrustRegionPolicyEvaluation
from dps.train import training_loop
from dps.config import LstmController, softmax

max_steps = 100000

config = DEFAULT_CONFIG.copy(
    n_val=500,
    controller=LstmController(),
    action_selection=softmax,
    exploration_schedule='poly 10.0 100000 0.1',
    get_experiment_name=lambda: "name={}_seed={}".format(cfg.actor_config.name, cfg.seed),
    batch_size=16,
    max_steps=max_steps,

    actor_config=Config(
        name="PPO",
        alg=PPO,
        optimizer_spec='adam',
        entropy_schedule="0.0625",
        # entropy_schedule="poly 0.125 100000 1e-6 1",
        epsilon=0.2,
        opt_steps_per_batch=10,
        lr_schedule="1e-3",
        gamma=0.9,
        lmbda=1.0,
        n_controller_units=128,
    ),

    critic_config=Config(
        name="TRPE",
        alg=TrustRegionPolicyEvaluation,
        delta_schedule='1e-2',
        max_cg_steps=10,
        max_line_search_steps=10,
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
    classification_bonus=0.0,
    symbols=[
        # ('A', lambda r: sum(r)),
        ('M', lambda r: np.product(r)),
        # ('C', lambda r: len(r)),
    ],
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
