from gym.envs.classic_control import CartPoleEnv

import clify

from dps import cfg
from dps.config import DEFAULT_CONFIG, PPO_CONFIG, PPE_CONFIG
from dps.gym_env import GymEnvWrapper
from dps.train import training_loop
from dps.rl.policy import Softmax


def build_env():
    return GymEnvWrapper(CartPoleEnv())


config = DEFAULT_CONFIG.copy(
    action_selection=lambda env: Softmax(env.n_actions, one_hot=False),
    critic_config=PPE_CONFIG.copy(
        K=100,
        epsilon=0.3
    ),
    actor_config=PPO_CONFIG,
    build_env=build_env,
    log_name="cartpole",
    T=None,
    threshold=-1000,
    n_val=100,
    batch_size=3,
    visualize=False,
    render_step=10
)


with config:
    cl_args = clify.wrap_object(cfg).parse()
    cfg.update(cl_args)

    training_loop()
