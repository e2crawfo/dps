from gym.envs.classic_control import CartPoleEnv

from dps import cfg
from dps.envs.gym_env import GymEnvWrapper
from dps.train import training_loop
from dps.config import DEFAULT_CONFIG
from dps.rl import rl_render_hook, BuildSoftmaxPolicy
from dps.rl.algorithms import ppo


def build_env():
    return GymEnvWrapper(CartPoleEnv())


config = DEFAULT_CONFIG.copy()

config.update(
    ppo.config,
    build_env=build_env,
    build_policy=BuildSoftmaxPolicy(one_hot=False),
    log_name="cartpole",
    T=None,
    threshold=-1000,
    n_val=100,
    batch_size=3,
    render_hook=rl_render_hook,
    render_step=100,
    seed=1,
    max_steps=100000,
)


with config:
    cfg.update_from_command_line()
    training_loop()
