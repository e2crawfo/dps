from dps import cfg
from dps.utils import Config
from dps.rl import (
    RLUpdater, RLContext, Agent, StochasticGradientDescent,
    PolicyGradient, PolicyEvaluation_State, PolicyEntropyBonus,
    AdvantageEstimator, ValueFunction, Retrace,
    BuildSoftmaxPolicy, BuildLstmController, PrioritizedReplayBuffer,
    ValueFunctionRegularization
)
from dps.rl.algorithms.qlearning import MaxPriorityFunc


def ACER(env):
    with RLContext(cfg.gamma) as context:
        if cfg.separate_exploration_policy:
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

        if cfg.separate_exploration_policy:
            agents[0].add_head(mu, existing_head=actor)

        action_values_from_returns = Retrace(
            actor, value_function, lmbda=cfg.lmbda,
            to_action_value=True, from_action_value=False,
            name="RetraceQ"
        )

        advantage_estimator = AdvantageEstimator(
            action_values_from_returns, value_function)

        PolicyGradient(
            actor, advantage_estimator, epsilon=cfg.epsilon,
            importance_c=cfg.c, weight=cfg.policy_weight)

        # Entropy bonus
        PolicyEntropyBonus(actor, weight=cfg.entropy_weight)

        values_from_returns = Retrace(
            actor, value_function, lmbda=cfg.lmbda,
            to_action_value=False, from_action_value=False,
            name="RetraceV"
        )

        # Policy evaluation
        policy_eval = PolicyEvaluation_State(value_function, values_from_returns, weight=cfg.value_weight)
        ValueFunctionRegularization(policy_eval, weight=cfg.value_reg_weight)

        priority_func = MaxPriorityFunc(policy_eval)
        replay_buffer = PrioritizedReplayBuffer(
            cfg.replay_size, cfg.replay_n_partitions,
            priority_func, cfg.alpha, cfg.beta_schedule, cfg.min_experiences)
        context.set_replay_buffer(cfg.update_batch_size, replay_buffer)

        # Optimize the objective function using stochastic gradient descent with respect
        # to the variables stored inside `agent`.
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
    name="ACER",
    get_updater=ACER,
    build_policy=BuildSoftmaxPolicy(),
    build_controller=BuildLstmController(),
    optimizer_spec="adam",
    lr_schedule="1e-4",
    n_controller_units=64,

    split=False,

    exploration_schedule="Poly(10.0, 10000, end=0.1)",
    test_time_explore=0.1,

    batch_size=1,
    update_batch_size=32,
    opt_steps_per_update=1,
    updates_per_sample=10,
    epsilon=0.2,

    policy_weight=1.0,
    value_weight=10.0,
    value_reg_weight=0.0,
    entropy_weight=0.0,
    lmbda=1,

    c=1,

    min_experiences=1000,
    replay_size=20000,
    replay_n_partitions=100,
    alpha=0.0,
    beta_schedule=0.0,
)
