from dps.env import BatchGymEnv
from dps.config import DEFAULT_CONFIG
from dps.rl import rl_render_hook, BuildSoftmaxPolicy


def build_env():
    return BatchGymEnv(gym_env='CartPole-v0')


config = DEFAULT_CONFIG.copy()


config.update(
    log_name="cartpole",

    build_env=build_env,

    # Found Softmax to be more sample efficient AND more stable than EpsilonSoftmax

    build_policy=BuildSoftmaxPolicy(one_hot=False),
    exploration_schedule="1.0",
    val_exploration_schedule="0.1",

    # build_policy=BuildEpsilonSoftmaxPolicy(one_hot=False),
    # exploration_schedule="0.1",
    # val_exploration_schedule="0.0",

    n_controller_units=64,

    epsilon=0.2,
    opt_steps_per_update=10,
    sub_batch_size=0,

    value_weight=0.0,
    T=None,

    n_val=100,
    batch_size=3,
    render_hook=rl_render_hook,
    render_step=10,
    eval_step=10,
    display_step=10,
    stopping_criteria="reward_per_ep,max",
    threshold=200,
)
