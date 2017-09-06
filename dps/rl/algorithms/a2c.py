from dps import cfg
from dps.utils import Config
from dps.rl import (
    RLContext, Agent, StochasticGradientDescent,
    BuildSoftmaxPolicy, BuildLstmController,
    PolicyGradient, RLUpdater, AdvantageEstimator,
    PolicyEntropyBonus, ValueFunction, PolicyEvaluation_State, Retrace
)


def A2C(env):
    with RLContext(cfg.gamma) as context:
        policy = cfg.build_policy(env, name="actor")

        context.set_behaviour_policy(policy)

        value_function = ValueFunction(1, policy, "critic")
        agent = Agent("agent", cfg.build_controller, [policy, value_function])

        action_values_from_returns = Retrace(
            policy, value_function, lmbda=cfg.lmbda,
            to_action_value=True, from_action_value=False,
            name="RetraceQ"
        )

        advantage_estimator = AdvantageEstimator(
            action_values_from_returns, value_function)

        # Policy optimization
        PolicyGradient(policy, advantage_estimator, epsilon=cfg.epsilon, weight=cfg.policy_weight)

        values_from_returns = Retrace(
            policy, value_function, lmbda=cfg.lmbda,
            to_action_value=False, from_action_value=False,
            name="RetraceV"
        )

        # Policy evaluation
        PolicyEvaluation_State(value_function, values_from_returns, weight=cfg.value_weight)

        # Entropy bonus
        PolicyEntropyBonus(policy, weight=cfg.entropy_weight)

        optimizer = StochasticGradientDescent(
            agents=[agent], alg=cfg.optimizer_spec,
            lr_schedule=cfg.lr_schedule,
            opt_steps_per_update=cfg.opt_steps_per_update)

        context.set_optimizer(optimizer)

    return RLUpdater(env, context)


config = Config(
    name="A2C",
    get_updater=A2C,
    n_controller_units=64,
    build_policy=BuildSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    batch_size=16,
    optimizer_spec="adam",
    opt_steps_per_update=10,
    lr_schedule="1e-4",
    exploration_schedule='Poly(10.0, 10000, 1e-6)',
    test_time_explore=0.1,
    policy_weight=1.0,
    value_weight=10.0,
    entropy_weight=0.0,
    lmbda=0.9,
    epsilon=0.2,
)


def ActorCritic(env):
    """ Similar to A2C, but here the actor and critic do not share any weights. """
    with RLContext(cfg.gamma) as context:
        policy = cfg.build_policy(env, name="actor")
        context.set_behaviour_policy(policy)
        actor_agent = Agent("actor_agent", cfg.build_controller, [policy])

        value_function = ValueFunction(1, policy, "critic")
        critic_agent = Agent("critic_agent", cfg.build_controller, [value_function])

        action_values_from_returns = Retrace(
            policy, value_function, lmbda=cfg.lmbda,
            to_action_value=True, from_action_value=False,
            name="RetraceQ"
        )

        advantage_estimator = AdvantageEstimator(
            action_values_from_returns, value_function)

        # Policy optimization
        PolicyGradient(policy, advantage_estimator, epsilon=cfg.epsilon, weight=cfg.policy_weight)

        values_from_returns = Retrace(
            policy, value_function, lmbda=cfg.lmbda,
            to_action_value=False, from_action_value=False,
            name="RetraceV"
        )

        # Policy evaluation
        PolicyEvaluation_State(value_function, values_from_returns, weight=cfg.value_weight)

        # Entropy bonus
        PolicyEntropyBonus(policy, weight=cfg.entropy_weight)

        optimizer = StochasticGradientDescent(
            agents=[actor_agent, critic_agent], alg=cfg.optimizer_spec,
            lr_schedule=cfg.lr_schedule,
            opt_steps_per_update=cfg.opt_steps_per_update)

        context.set_optimizer(optimizer)

    return RLUpdater(env, context)


actor_critic_config = config.copy(
    name="ActorCritic",
    get_updater=ActorCritic,
)
