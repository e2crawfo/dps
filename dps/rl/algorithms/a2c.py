from dps import cfg
from dps.utils import Config
from dps.rl import (
    RLContext, Agent, StochasticGradientDescent,
    BuildSoftmaxPolicy, BuildEpsilonSoftmaxPolicy, BuildLstmController, PolicyGradient,
    RLUpdater, AdvantageEstimator, MonteCarloValueEstimator,
    PolicyEntropyBonus, ValueFunction, PolicyEvaluation_State, Retrace,
    ValueFunctionRegularization
)


def A2C(env):
    with RLContext(cfg.gamma) as context:

        if cfg.actor_exploration_schedule is not None:
            actor = cfg.build_policy(env, name="actor", exploration_schedule=cfg.actor_exploration_schedule)
            context.set_validation_policy(actor)

            mu = cfg.build_policy(env, name="mu")
            context.set_behaviour_policy(mu)
        else:
            actor = cfg.build_policy(env, name="actor")
            context.set_behaviour_policy(actor)
            context.set_validation_policy(actor)

        value_function = ValueFunction(1, actor, "critic")

        if cfg.split:
            actor_agent = Agent("actor_agent", cfg.build_controller, [actor])
            critic_agent = Agent("critic_agent", cfg.build_controller, [value_function])
            agents = [actor_agent, critic_agent]
        else:
            agent = Agent("agent", cfg.build_controller, [actor, value_function])
            agents = [agent]

        if cfg.actor_exploration_schedule is not None:
            agents[0].add_head(mu, existing_head=actor)

        action_values_from_returns = Retrace(
            actor, value_function, lmbda=cfg.q_lmbda, importance_c=cfg.q_importance_c,
            to_action_value=True, from_action_value=False,
            name="RetraceQ"
        )

        advantage_estimator = AdvantageEstimator(
            action_values_from_returns, value_function)

        PolicyGradient(
            actor, advantage_estimator, epsilon=cfg.epsilon,
            weight=cfg.policy_weight, importance_c=cfg.policy_importance_c)

        values_from_returns = Retrace(
            actor, value_function, lmbda=cfg.v_lmbda, importance_c=cfg.v_importance_c,
            to_action_value=False, from_action_value=False,
            name="RetraceV"
        )

        PolicyEntropyBonus(actor, weight=cfg.entropy_weight)

        policy_eval = PolicyEvaluation_State(value_function, values_from_returns, weight=cfg.value_weight)
        ValueFunctionRegularization(policy_eval, weight=cfg.value_reg_weight)

        optimizer = StochasticGradientDescent(
            agents=agents, alg=cfg.optimizer_spec,
            lr_schedule=cfg.lr_schedule,
            max_grad_norm=cfg.max_grad_norm,
            noise_schedule=cfg.noise_schedule
        )

        context.set_optimizer(optimizer)

    return RLUpdater(env, context)


config = Config(
    name="A2C",
    get_updater=A2C,
    n_controller_units=64,
    # build_policy=BuildSoftmaxPolicy(),
    build_policy=BuildEpsilonSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    batch_size=8,
    optimizer_spec="adam",
    opt_steps_per_update=50,
    lr_schedule="1e-4",
    exploration_schedule="Poly(1.0, 100000, end=0.05)",
    actor_exploration_schedule=None,
    test_time_explore=-1,
    policy_weight=1.0,
    value_weight=1.0,
    value_reg_weight=0.0,
    entropy_weight=0.0,
    epsilon=0.2,
    split=True,
    q_lmbda=1.0,
    v_lmbda=1.0,
    policy_importance_c=0,
    q_importance_c=None,
    v_importance_c=None,
    max_grad_norm=None,
    gamma=1.0
)


actor_critic_config = config.copy(
    name="ActorCritic",
    split=True
)
