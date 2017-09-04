from dps import cfg
from dps.utils import Config
from dps.rl import (
    RLContext, Agent, StochasticGradientDescent,
    BuildSoftmaxPolicy, BuildLstmController,
    PolicyGradient, RLUpdater, AdvantageEstimator,
    PolicyEntropyBonus, ValueFunction, PolicyEvaluation_State, Retrace
)


config = Config(
    name="ActorCritic",
    get_updater=ActorCritic,
    n_controller_units=64,
    build_policy=BuildSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    batch_size=16,
    optimizer_spec="adam",
    opt_steps_per_batch=10,
    lr_schedule="1e-4",
    exploration_schedule='poly 10.0 10000 1e-6 1.0',
    test_time_explore=0.1,
    policy_weight=1.0,
    value_weight=10.0,
    entropy_weight=0.0,
    lmbda=0.9,
    epsilon=0.2,
)
