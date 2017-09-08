from dps import cfg
from dps.utils import Config
from dps.rl import (
    RLContext, Agent, StochasticGradientDescent,
    PolicyGradient, BasicAdvantageEstimator, RLUpdater,
    BuildSoftmaxPolicy, BuildLstmController
)


def PPO(env):
    with RLContext(cfg.gamma) as context:
        policy = cfg.build_policy(env, name="actor")
        context.set_behaviour_policy(policy)

        # Build an agent with `policy` as one of its heads.
        agent = Agent("agent", cfg.build_controller, [policy])

        # Build an advantage estimator that estimates advantage from rollouts.
        advantage_estimator = BasicAdvantageEstimator(policy)

        # Add a term to the objective function encapsulated by `context`.
        # The surrogate objective function, when differentiated, yields the policy gradient.
        PolicyGradient(policy, advantage_estimator, epsilon=cfg.epsilon)

        # Optimize the objective function using stochastic gradient descent with respect
        # to the variables stored inside `agent`.
        optimizer = StochasticGradientDescent(
            agents=[agent], alg=cfg.optimizer_spec,
            lr_schedule=cfg.lr_schedule,
            opt_steps_per_update=cfg.opt_steps_per_update,
            max_grad_norm=cfg.max_grad_norm,
            noise_schedule=cfg.noise_schedule
        )
        context.set_optimizer(optimizer)

    return RLUpdater(env, context)


config = Config(
    name="PPO",
    get_updater=PPO,
    build_policy=BuildSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    batch_size=16,
    optimizer_spec="adam",
    opt_steps_per_update=10,
    lr_schedule="1e-4",
    n_controller_units=64,
    exploration_schedule='Poly(10.0, 10000, end=0.1)',
    test_time_explore=0.1,
    epsilon=0.2
)


reinforce_config = config.copy(
    epsilon=0.0,
    opt_steps_per_update=1,
    name="REINFORCE",
)
