from dps.env import BatchGymEnv
from dps.config import DEFAULT_CONFIG
from dps.rl import rl_render_hook, BuildSoftmaxPolicy, BuildEpsilonSoftmaxPolicy
from dps.datasets.xo import XO_Env, XO_RenderHook
from dps.utils.tf import FeedforwardCell, MLP, ScopedFunction, RelationNetwork

from auto_yolo.models.core import Backbone


def build_env():
    gym_env = XO_Env(
        image_shape=(12, 12), background_colour="black", entity_colours="white white white",
        entity_sizes="3 3 3", min_entities=1, max_entities=4, max_overlap=0.0, step_size=3,
        grid=False, cross_prob=0.5, corner=None, max_episode_length=None, image_obs=False)
    # gym_env = XO_Env(
    #     image_shape=(36, 36), background_colour="black", entity_colours="white white white",
    #     entity_sizes="5 8 8", min_entities=1, max_entities=4, max_overlap=0.0, step_size=4,
    #     grid=False, cross_prob=0.5, corner=None, max_episode_length=None, image_obs=False)
    # gym_env = XO_Env(
    #     image_shape=(36, 36), background_colour="black", entity_colours="white white white",
    #     entity_sizes="5 10 10", min_entities=4, max_entities=8, overlap_factor=0.2, step_size=5,
    #     grid=False, cross_prob=0.5, corner=None, max_episode_length=10)

    return BatchGymEnv(gym_env=gym_env)


class XO_Backbone(ScopedFunction):
    backbone = None
    mlp = None

    def _call(self, inp, output_size, is_training):
        if self.backbone is None:
            self.backbone = Backbone()

        if self.mlp is None:
            self.mlp = MLP([100, 100])

        outp = self.backbone(inp, 0, is_training)
        outp = self.mlp(outp, output_size, is_training)
        return outp


def build_xo_controller(output_size, name):
    # ff = MLP([256, 256, 256], scope="xo_controller")
    # ff = XO_Backbone(scope="xo_controller")
    ff = RelationNetwork(scope="xo_controller")
    return FeedforwardCell(ff, output_size, name=name)


config = DEFAULT_CONFIG.copy()


config.update(
    env_name="xo",

    build_env=build_env,

    build_controller=build_xo_controller,
    build_relation_network_f=lambda scope: MLP([100, 100], scope=scope),
    build_relation_network_g=lambda scope: MLP([100, 100], scope=scope),
    f_dim=128,
    symmetric_op="max",

    # pixels_per_cell=(12, 12),
    # kernel_size=1,
    # n_channels=128,
    # n_final_layers=3,

    build_policy=BuildSoftmaxPolicy(one_hot=False),
    # build_policy=BuildEpsilonSoftmaxPolicy(one_hot=False),
    exploration_schedule=1.0,
    val_exploration_schedule=0.1,

    n_controller_units=64,

    epsilon=0.0,
    opt_steps_per_update=1,
    sub_batch_size=0,

    value_weight=1.0,
    T=20,

    n_val=100,
    batch_size=16,
    render_hook=XO_RenderHook(),
    render_step=1000,
    eval_step=100,
    display_step=100,
    stopping_criteria="reward_per_ep,max",
    threshold=1000,
)
