from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.env import BatchGymEnv
from dps.utils.tf import MLP, FeedforwardCell
from dps.rl import rl_render_hook, BuildSoftmaxPolicy, BuildMlpController


def build_env():
    return BatchGymEnv(gym_env='MountainCar-v0')


controller = lambda params_dim, name: FeedforwardCell(
    lambda inp, output_size: MLP(
        [cfg.n_controller_units, cfg.n_controller_units])(inp, output_size),
    params_dim, name=name)


config = DEFAULT_CONFIG.copy()


# So far, have not been able to solve this with a policy gradient method, the exploration problem is quite hard.


config.update(
    env_name="mountain_car",

    build_env=build_env,

    build_controller=BuildMlpController(),
    build_policy=BuildSoftmaxPolicy(one_hot=False),
    exploration_schedule="1.0",
    val_exploration_schedule="0.1",

    n_controller_units=64,

    value_weight=10.0,
    reward_scale=200,
    split=False,
    v_lmbda=0.8,
    q_lmbda=0.8,

    T=None,

    n_val=100,
    batch_size=1,
    render_hook=rl_render_hook,
    render_step=10,
    eval_step=10,
    display_step=10,
    stopping_criteria="reward_per_ep,max",
    threshold=0,
)
