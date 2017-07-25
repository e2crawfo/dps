from gym.envs.classic_control import MountainCarEnv

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG, PPO_CONFIG, PE_CONFIG
from dps.gym_env import GymEnvWrapper
from dps.train import training_loop
from dps.rl.policy import Softmax
from dps.utils import MLP, FeedforwardCell


def build_env():
    return GymEnvWrapper(MountainCarEnv())


controller = lambda n_params, name: FeedforwardCell(
    lambda inp, output_size: MLP([128, 128])(inp, output_size), n_params, name=name)


config = DEFAULT_CONFIG.copy(
    action_selection=lambda env: Softmax(env.n_actions, one_hot=False),
    controller=controller,
    critic_config=PE_CONFIG,
    actor_config=PPO_CONFIG,
    build_env=build_env,
    log_name="mountain_car",
    T=None,
    threshold=-10,
    n_val=100,
    batch_size=1,
    visualize=False,
    render_step=10,
    display_step=10,
    test_time_explore=-1,
    exploration_schedule="100",
    reset_env=False
)


with config:
    cl_args = clify.wrap_object(cfg).parse()
    cfg.update(cl_args)

    training_loop()
