import tensorflow as tf
import argparse

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.experiments import alt_arithmetic
from dps.utils import CompositeCell, Config, MLP, pdb_postmortem
from dps.rl import PPO, REINFORCE
from dps.rl.value import TrustRegionPolicyEvaluation
from dps.rl.policy import EpsilonSoftmax, Softmax
from dps.train import training_loop

max_steps = 100000

config = DEFAULT_CONFIG.copy(
    n_val=500,
    n_controller_units=64,
    controller=lambda n_params, name: CompositeCell(
        tf.contrib.rnn.LSTMCell(num_units=cfg.n_controller_units), MLP(), n_params, name=name),
    action_selection=lambda env: Softmax(env.n_actions),
    # action_selection=lambda env: EpsilonSoftmax(env.n_actions),
    # exploration_schedule="poly 1.0 1000 0.1",
    exploration_schedule='poly 10.0 {} 1.0'.format(int(max_steps/10)),
    get_experiment_name=lambda: "name={}_seed={}".format(cfg.actor_config.name, cfg.seed),
    batch_size=16,
    max_steps=max_steps,

    # actor_config=Config(
    #     name="PPO",
    #     alg=PPO,
    #     entropy_schedule="0.0",
    #     epsilon=0.2,
    #     K=10,
    #     lr_schedule="1e-5",
    #     n_controller_units=64,
    #     test_time_explore=-1
    # ),

    actor_config=Config(
        name="REINFORCE",
        alg=REINFORCE,
        lr_schedule="poly 1e-3 {} 1e-6".format(max_steps),
        entropy_schedule='poly 0.25 {} 1e-6 1'.format(max_steps),
        n_controller_units=64,
        test_time_explore=0.01,
    ),

    critic_config=Config(
        name="TRPE",
        alg=TrustRegionPolicyEvaluation,
        delta_schedule='0.01',
        max_cg_steps=10,
        max_line_search_steps=10,
    ),

    build_env=alt_arithmetic.build_env,

    curriculum=[
        dict(T=20, min_digits=2, max_digits=3, shape=(2, 2)),
    ],
    display=False,
    mnist=False,
    op_loc=(0, 0),
    start_loc=(0, 0),
    base=10,
    threshold=0.01,
    reward_window=0.4,
    ablation='',  # anything other than "bad_wiring", "no_classifiers", "no_ops", "no_modules" will use the default.

    log_name='alt_arithmetic',
    render_rollouts=None
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
