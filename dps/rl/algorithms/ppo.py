from dps import cfg
from dps.utils import Config
from dps.rl import (
    RLContext, Agent, StochasticGradientDescent,
    PolicyGradient, BasicAdvantageEstimator, RLUpdater,
    BuildEpsilonSoftmaxPolicy, BuildLstmController, PolicyEntropyBonus,
)


def PPO(env):
    with RLContext(cfg.gamma) as context:
        if cfg.actor_exploration_schedule is not None:
            mu = cfg.build_policy(env, name="mu")
            context.set_behaviour_policy(mu)

            actor = cfg.build_policy(env, name="actor", exploration_schedule=cfg.actor_exploration_schedule)
            context.set_validation_policy(actor)

            agent = Agent("agent", cfg.build_controller, [actor])
            agent.add_head(mu, existing_head=actor)
        else:
            actor = cfg.build_policy(env, name="actor")
            context.set_behaviour_policy(actor)
            context.set_validation_policy(actor)

            agent = Agent("agent", cfg.build_controller, [actor])

        # Build an advantage estimator that estimates advantage from current set of rollouts.
        advantage_estimator = BasicAdvantageEstimator(
            actor, q_importance_c=cfg.q_importance_c, v_importance_c=cfg.v_importance_c)

        # Add a term to the objective function encapsulated by `context`.
        # The surrogate objective function, when differentiated, yields the actor gradient.
        PolicyGradient(
            actor, advantage_estimator, epsilon=cfg.epsilon,
            weight=cfg.policy_weight, importance_c=cfg.policy_importance_c)
        PolicyEntropyBonus(actor, weight=cfg.entropy_weight)

        # Optimize the objective function using stochastic gradient descent with respect
        # to the variables stored inside `agent`.
        optimizer = StochasticGradientDescent(
            agents=[agent], alg=cfg.optimizer_spec,
            lr_schedule=cfg.lr_schedule,
            max_grad_norm=cfg.max_grad_norm,
            noise_schedule=cfg.noise_schedule
        )
        context.set_optimizer(optimizer)

    return RLUpdater(env, context)


config = Config(
    name="PPO",
    get_updater=PPO,
    build_policy=BuildEpsilonSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    batch_size=8,
    optimizer_spec="adam",
    opt_steps_per_update=10,
    lr_schedule=1e-4,
    exploration_schedule=0.2,
    actor_exploration_schedule=None,
    n_controller_units=64,
    test_time_explore=-1.,
    epsilon=0.2,
    policy_weight=1.0,
    entropy_weight=0.0,

    policy_importance_c=10,
    q_importance_c=None,
    v_importance_c=None,
)


reinforce_config = config.copy(
    epsilon=None,
    opt_steps_per_update=1,
    name="REINFORCE",
)
