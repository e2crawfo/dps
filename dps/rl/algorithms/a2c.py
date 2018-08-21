from dps import cfg
from dps.utils import Config
from dps.rl import (
    RLContext, Agent, StochasticGradientDescent,
    BuildEpsilonSoftmaxPolicy, BuildLstmController,
    PolicyGradient, RLUpdater, AdvantageEstimator, PolicyEntropyBonus,
    ValueFunction, PolicyEvaluation_State, Retrace, ValueFunctionRegularization,
    BasicAdvantageEstimator, ConstrainedPolicyEvaluation_State, DifferentiableLoss
)


def A2C(env):
    with RLContext(cfg.gamma) as context:
        actor = cfg.build_policy(
            env, name="actor",
            exploration_schedule=cfg.exploration_schedule,
            val_exploration_schedule=cfg.val_exploration_schedule
        )

        context.set_behaviour_policy(actor)
        context.set_validation_policy(actor)

        if cfg.value_weight:
            value_function = ValueFunction(1, actor, "critic")

            if cfg.split:
                actor_agent = Agent("actor_agent", cfg.build_controller, [actor])
                critic_agent = Agent("critic_agent", cfg.build_controller, [value_function])
                agents = [actor_agent, critic_agent]
            else:
                agent = Agent("agent", cfg.build_controller, [actor, value_function])
                agents = [agent]

            values_from_returns = Retrace(
                actor, value_function, lmbda=cfg.v_lmbda, importance_c=cfg.v_importance_c,
                to_action_value=False, from_action_value=False,
                name="RetraceV"
            )

            if cfg.value_epsilon:
                ConstrainedPolicyEvaluation_State(
                    value_function, values_from_returns,
                    epsilon=cfg.value_epsilon, weight=cfg.value_weight,
                    n_samples=cfg.value_n_samples, direct=cfg.value_direct
                )
            else:
                policy_eval = PolicyEvaluation_State(value_function, values_from_returns, weight=cfg.value_weight)

                if cfg.value_reg_weight:
                    ValueFunctionRegularization(policy_eval, weight=cfg.value_reg_weight)

            action_values_from_returns = Retrace(
                actor, value_function, lmbda=cfg.q_lmbda, importance_c=cfg.q_importance_c,
                to_action_value=True, from_action_value=False,
                name="RetraceQ"
            )

            advantage_estimator = AdvantageEstimator(
                action_values_from_returns, value_function)
        else:
            agent = Agent("agent", cfg.build_controller, [actor])
            agents = [agent]

            # Build an advantage estimator that estimates advantage from current set of rollouts.
            advantage_estimator = BasicAdvantageEstimator(
                actor, q_importance_c=cfg.q_importance_c, v_importance_c=cfg.v_importance_c)

        PolicyGradient(
            actor, advantage_estimator, epsilon=cfg.epsilon,
            importance_c=cfg.policy_importance_c, weight=cfg.policy_weight)
        PolicyEntropyBonus(actor, weight=cfg.entropy_weight)

        if env.has_differentiable_loss and cfg.use_differentiable_loss:
            DifferentiableLoss(env, actor)

        optimizer = StochasticGradientDescent(agents=agents, alg=cfg.optimizer_spec)
        context.set_optimizer(optimizer)

    return RLUpdater(env, context)


config = Config(
    exp_name="A2C",
    get_updater=A2C,
    n_controller_units=64,
    batch_size=16,
    optimizer_spec="adam",
    opt_steps_per_update=1,
    sub_batch_size=0,
    epsilon=None,
    lr_schedule=1e-4,

    exploration_schedule=0.1,
    val_exploration_schedule=0.0,

    value_weight=1.0,
    value_epsilon=None,
    value_n_samples=0,
    value_direct=False,
    value_reg_weight=0.0,

    build_policy=BuildEpsilonSoftmaxPolicy(),
    build_controller=BuildLstmController(),

    policy_weight=1.0,
    entropy_weight=0.01,

    split=False,
    q_lmbda=1.0,
    v_lmbda=1.0,
    policy_importance_c=0,
    q_importance_c=None,
    v_importance_c=None,
    max_grad_norm=None,
    gamma=1.0,

    use_differentiable_loss=False,
    render_n_rollouts=4,
)


actor_critic_config = config.copy(
    exp_name="ActorCritic",
    split=True
)


ppo_config = config.copy(
    exp_name="PPO",
    opt_steps_per_update=10,
    sub_batch_size=2,
    epsilon=0.2,
    value_weight=0.0,
)


# Same config that is used in the test.
test_config = config.copy(
    exp_name="TestA2C",
    opt_steps_per_update=20,
    sub_batch_size=0,
    epsilon=0.2,
    n_controller_units=32,
    value_weight=0.0,
    split=False,
)


reinforce_config = config.copy(
    exp_name="REINFORCE",
    epsilon=0.0,
    opt_steps_per_update=1,
    value_weight=0.0,
)
