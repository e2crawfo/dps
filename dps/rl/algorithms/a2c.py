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
        actor = cfg.build_policy(env, name="actor")
        context.set_behaviour_policy(actor)

        value_function = ValueFunction(1, actor, "critic")

        if cfg.split:
            actor_agent = Agent("actor_agent", cfg.build_controller, [actor])
            critic_agent = Agent("critic_agent", cfg.build_controller, [value_function])
            agents = [actor_agent, critic_agent]
        else:
            agent = Agent("agent", cfg.build_controller, [actor, value_function])
            agents = [agent]

        action_values_from_returns = Retrace(
            actor, value_function, lmbda=cfg.lmbda,
            to_action_value=True, from_action_value=False,
            name="RetraceQ"
        )

        advantage_estimator = AdvantageEstimator(
            action_values_from_returns, value_function)

        PolicyGradient(actor, advantage_estimator, epsilon=cfg.epsilon, weight=cfg.policy_weight)

        values_from_returns = Retrace(
            actor, value_function, lmbda=cfg.lmbda,
            to_action_value=False, from_action_value=False,
            name="RetraceV"
        )

        PolicyEvaluation_State(value_function, values_from_returns, weight=cfg.value_weight)
        PolicyEntropyBonus(actor, weight=cfg.entropy_weight)

        optimizer = StochasticGradientDescent(
            agents=agents, alg=cfg.optimizer_spec,
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
    exploration_schedule='Poly(10.0, 10000, end=0.1)',
    test_time_explore=0.1,
    policy_weight=1.0,
    value_weight=10.0,
    entropy_weight=0.0,
    lmbda=0.9,
    epsilon=0.2,
    split=False
)


actor_critic_config = config.copy(
    name="ActorCritic",
    split=True
)
