from dps import cfg
from dps.utils import Config
from dps.rl import (
    RLContext, Agent, StochasticGradientDescent,
    BuildSoftmaxPolicy, BuildLstmController, PolicyGradient,
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
            actor, value_function, lmbda=cfg.lmbda,
            to_action_value=True, from_action_value=False,
            name="RetraceQ"
        )

        action_values_from_returns = MonteCarloValueEstimator(actor)

        advantage_estimator = AdvantageEstimator(
            action_values_from_returns, value_function)

        PolicyGradient(
            actor, advantage_estimator, epsilon=cfg.epsilon,
            weight=cfg.policy_weight, importance_c=cfg.importance_c)

        values_from_returns = Retrace(
            actor, value_function, lmbda=1.,
            to_action_value=False, from_action_value=False,
            name="RetraceV"
        )

        values_from_returns = MonteCarloValueEstimator(actor)

        PolicyEntropyBonus(actor, weight=cfg.entropy_weight)

        policy_eval = PolicyEvaluation_State(value_function, values_from_returns, weight=cfg.value_weight)
        ValueFunctionRegularization(policy_eval, weight=cfg.value_reg_weight)

        optimizer = StochasticGradientDescent(
            agents=agents, alg=cfg.optimizer_spec,
            lr_schedule=cfg.lr_schedule,
            opt_steps_per_update=cfg.opt_steps_per_update,
            max_grad_norm=cfg.max_grad_norm,
            noise_schedule=cfg.noise_schedule
        )

        context.set_optimizer(optimizer)

    return RLUpdater(env, context)


config = Config(
    name="A2C",
    get_updater=A2C,
    n_controller_units=64,
    build_policy=BuildSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    batch_size=8,
    optimizer_spec="adam",
    opt_steps_per_update=10,
    lr_schedule="1e-4",
    exploration_schedule=5.0,
    actor_exploration_schedule=5.0,
    test_time_explore=-1,
    policy_weight=1.0,
    value_weight=1.0,
    value_reg_weight=0.0,
    entropy_weight=0.0,
    lmbda=0.95,
    epsilon=0.2,
    split=True,
    importance_c=0,
    max_grad_norm=5.0,
    gamma=0.98,
)


actor_critic_config = config.copy(
    name="ActorCritic",
    split=True
)
