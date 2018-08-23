import numpy as np
import gym
import copy
from matplotlib import pyplot as plt
from matplotlib import animation
from collections import defaultdict

from dps import cfg
from dps.env.basic import game
from dps.env.env import BatchGymEnv
from dps.utils import Param, square_subplots
from dps.train import Hook
from dps.utils.tf import FeedforwardCell, MLP, ScopedFunction
from dps.rl.policy import Policy, ProductDist, SigmoidNormal, Softmax


class CollectBase(game.ObjectGame):
    agent_spec = Param()
    collectable_specs = Param()
    obstacles_specs = Param()
    step_size = Param()
    time_reward = Param()
    discrete_actions = Param()

    def __init__(self, **kwargs):
        self.agent_spec = copy.deepcopy(dict(self.agent_spec))
        self.agent_spec['collectable'] = False

        self.collectable_specs = copy.deepcopy(list(self.collectable_specs))
        self.collectable_specs = [dict(cs) for cs in self.collectable_specs]
        for spec in self.collectable_specs:
            spec['collectable'] = True

        self.obstacles_specs = copy.deepcopy(list(self.obstacles_specs))
        self.obstacles_specs = [dict(os) for os in self.obstacles_specs]
        for spec in self.obstacles_specs:
            spec['collectable'] = False

        self.entity_specs = [self.agent_spec] + self.collectable_specs + self.obstacles_specs

        for i, es in enumerate(self.entity_specs):
            es['idx'] = i

        if self.discrete_actions:
            action_space = gym.spaces.MultiDiscrete([8, 3])
        else:
            action_space = gym.spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        super(CollectBase, self).__init__(
            action_space=action_space,
            reward_range=(-10, 10),
            entity_feature_dim=len(self.entity_specs),
            **kwargs)

    def move_entities(self, action):
        if self.discrete_actions:
            angle_idx, magnitude_idx = action
            angle = angle_idx * 2 * np.pi / 8
            magnitude = [0.1, 0.5, 1.0][int(magnitude_idx)]
            y = self.step_size * magnitude * np.sin(angle)
            x = self.step_size * magnitude * np.cos(angle)
        else:
            y, x, magnitude = action
            y = np.clip(y, -1, 1)
            x = np.clip(x, -1, 1)
            magnitude = np.clip(magnitude, 0, 1)

            norm = np.sqrt(x**2 + y**2)
            if norm > 1e-6:
                y = self.step_size * magnitude * y / norm
                x = self.step_size * magnitude * x / norm
            else:
                y = x = 0

        return self._move_entity(self.entities[0], y, x)

    def resolve_collision(self, mover, other):
        """ Return (kill, stop, reward) """
        if isinstance(other, str):  # wall
            return (False, True, 0)
        else:
            if other.collectable:
                if self.time_reward:
                    return (True, False, 0)
                else:
                    return (True, False, 1/self.n_collectables)
            else:
                return (False, True, 0)

    def get_entity_features(self, entity):
        return [int(entity.idx == i) for i in range(len(self.entity_specs))]

    def compute_reward(self):
        if self.time_reward:
            return sum([-1 for entity in self.entities if entity.collectable and entity.alive]) / (self.n_collectables * cfg.T)
        else:
            return 0.0


class CollectA(CollectBase):
    n_collectables = Param()
    n_obstacles = Param()
    max_overlap = Param()

    max_entities = None

    def __init__(self, **kwargs):
        self.max_entities = 1 + self.n_collectables + self.n_obstacles
        assert self.n_collectables > 0

        super(CollectA, self).__init__(**kwargs)

    def setup_field(self):
        collectable_specs = list(np.random.choice(self.collectable_specs, size=self.n_collectables, replace=True))
        obstacles_specs = list(np.random.choice(self.obstacles_specs, size=self.n_obstacles, replace=True))
        specs = [self.agent_spec] + collectable_specs + obstacles_specs
        shapes = [spec['shape'] for spec in specs]

        rectangles = game.sample_entities(self.image_shape, shapes, self.max_overlap)
        entities = [game.Entity(**spec) for spec in specs]
        for rect, entity in zip(rectangles, entities):
            entity.top = rect.top
            entity.left = rect.left

        return entities


def build_env():
    gym_env = CollectA()
    return BatchGymEnv(gym_env=gym_env)


class CollectB(CollectBase):
    angle_sep = Param()
    n_dirs = Param()
    max_entities = None

    def __init__(self, **kwargs):
        self.max_entities = 2*self.n_dirs + 1
        self.n_collectables = 1
        super(CollectB, self).__init__(**kwargs)

    def setup_field(self):
        assert self.image_shape[0] == self.image_shape[1]
        start_angle = np.pi/4
        radius = int(np.floor(self.image_shape[0] / 2 - self.agent_spec['shape'][0]/2))
        center = (self.image_shape[0]/2, self.image_shape[1]/2)
        centers = []
        for i in range(self.n_dirs):
            angle = start_angle + 2*np.pi * i / self.n_dirs
            angle1 = angle - self.angle_sep
            angle2 = angle + self.angle_sep

            for angle in [angle1, angle2]:
                y = radius * np.sin(angle) + center[0]
                x = radius * np.cos(angle) + center[1]
                centers.append((y, x))

        collectable_spec = np.random.choice(self.collectable_specs)
        obstacles_specs = list(np.random.choice(self.obstacles_specs, size=2*self.n_dirs-1, replace=True))
        object_specs = np.random.permutation([collectable_spec] + obstacles_specs)

        agent = game.Entity(**self.agent_spec)
        agent.center = center

        objects = [game.Entity(**spec) for spec in object_specs]

        for center, obj in zip(centers, objects):
            obj.center = center

        return [agent, *objects]


class CollectC(CollectBase):
    max_overlap = Param()
    n_collectables = Param()
    max_entities = None

    def __init__(self, **kwargs):
        self.max_entities = 1 + self.n_collectables
        super(CollectC, self).__init__(**kwargs)

    def setup_field(self):
        collectable_specs = list(np.random.choice(self.collectable_specs, size=self.n_collectables, replace=True))
        specs = [self.agent_spec] + collectable_specs
        shapes = [spec['shape'] for spec in specs]

        rectangles = game.sample_entities(self.image_shape, shapes, self.max_overlap)
        entities = [game.Entity(**spec) for spec in specs]
        for rect, entity in zip(rectangles, entities):
            entity.top = rect.top
            entity.left = rect.left

        return entities


class RolloutsHook(Hook):
    def __init__(self, env_class, plot_step=None, env_kwargs=None, **kwargs):
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        kwarg_string = "_".join("{}={}".format(k, v) for k, v in self.env_kwargs.items())
        name = env_class.__name__ + ("_" + kwarg_string if kwarg_string else "")
        self.name = name.replace(" ", "_")
        self.plot_step = plot_step
        super(RolloutsHook, self).__init__(**kwargs)

    def start_stage(self, training_loop, stage_idx):
        gym_env = self.env_class(**self.env_kwargs)
        self.env = BatchGymEnv(gym_env=gym_env)

    def plot(self, updater, rollouts):
        plt.ion()

        if updater.env.gym_env.image_obs:
            obs = rollouts.obs
        else:
            obs = rollouts.image

        fig, axes = square_subplots(rollouts.batch_size, figsize=(5, 5))
        plt.subplots_adjust(top=0.95, bottom=0, left=0, right=1, wspace=0.1, hspace=0.1)

        images = []
        for i, ax in enumerate(axes.flatten()):
            ax.set_aspect("equal")
            ax.set_axis_off()
            image = ax.imshow(np.zeros(obs.shape[2:]))
            images.append(image)

        def animate(t):
            for i in range(rollouts.batch_size):
                images[i].set_array(obs[t, i, :, :, :])

        anim = animation.FuncAnimation(fig, animate, frames=len(rollouts), interval=500)

        path = updater.exp_dir.path_for('plots', '{}_animation.gif'.format(self.name))
        anim.save(path, writer='imagemagick')

        plt.close(fig)

    def step(self, training_loop, updater, step_idx):
        n_rollouts = cfg.n_val_rollouts
        batch_size = cfg.batch_size
        record = defaultdict(float)
        n_iters = int(np.ceil(n_rollouts / batch_size))

        for it in range(n_iters):
            n_remaining = n_rollouts - it * batch_size
            _batch_size = min(batch_size, n_remaining)

            for learner in updater.learners:
                with learner:
                    rollouts = self.env.do_rollouts(policy=learner.pi, n_rollouts=_batch_size, T=cfg.T, mode='val')
                    key = "{}-reward_per_ep".format(self.name)
                    record[key] += _batch_size * rollouts.rewards.sum(0).mean()

            if it == 0 and self.plot_step and step_idx % self.plot_step == 0:
                self.plot(updater, rollouts)

        return dict(val={k: v / n_rollouts for k, v in record.items()})


agent_size = (10, 10)
entity_size = (10, 10)
noise_res = None
# colors = "black"
colors = "red green blue"

agent_spec = dict(appearance="star", color=colors)  # color="black")

collectable_specs = [dict(appearance="x", color=colors)]

obstacles_specs = [
    dict(appearance="circle", color=colors),
    dict(appearance="ud_triangle", color=colors),
    dict(appearance="triangle", color=colors),
    dict(appearance="plus", color=colors),
    dict(appearance="diamond", color=colors),
]
entity_specs = [agent_spec] + collectable_specs + obstacles_specs

for es in entity_specs:
    es.update(shape=entity_size, noise_res=noise_res)
entity_specs[0]['shape'] = agent_size
entity_specs[0]['noise_res'] = noise_res

hook_step = 1000

# env config
config = game.config.copy(
    env_name="collect",

    n_collectables=5,
    n_obstacles=5,
    agent_spec=agent_spec,
    collectable_specs=collectable_specs,
    obstacles_specs=obstacles_specs,
    build_env=build_env,
    image_shape=(48, 48), background_colour="white", max_overlap=0.25, step_size=14,
    hooks=[
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectB, env_kwargs=dict(n_dirs=4)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectB, env_kwargs=dict(n_dirs=5)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectB, env_kwargs=dict(n_dirs=6)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectB, env_kwargs=dict(n_dirs=7)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectB, env_kwargs=dict(n_dirs=8)),

        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectC, env_kwargs=dict(n_collectables=5)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectC, env_kwargs=dict(n_collectables=10)),

        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(n_collectables=6, n_obstacles=6)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(n_collectables=7, n_obstacles=7)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(n_collectables=8, n_obstacles=8)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(n_collectables=9, n_obstacles=9)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(n_collectables=10, n_obstacles=10)),

        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(image_shape=(72, 72), n_collectables=5, n_obstacles=5)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(image_shape=(72, 72), n_collectables=6, n_obstacles=6)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(image_shape=(72, 72), n_collectables=7, n_obstacles=7)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(image_shape=(72, 72), n_collectables=8, n_obstacles=8)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(image_shape=(72, 72), n_collectables=9, n_obstacles=9)),
        RolloutsHook(n=hook_step, plot_step=hook_step, initial=True, env_class=CollectA, env_kwargs=dict(image_shape=(72, 72), n_collectables=10, n_obstacles=10)),

    ],
    angle_sep=np.pi/16,

    discrete_actions=True,
    time_reward=False,
    eval_step=1000,
    display_step=1000,

)


class Backbone(ScopedFunction):
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


def build_attentional_relation_network(output_size, name):
    from dps.utils.tf import AttentionalRelationNetwork
    ff = AttentionalRelationNetwork(n_repeats=2, scope="collection_controller")
    return FeedforwardCell(ff, output_size, name=name)


def build_object_network_controller(output_size, name):
    from dps.utils.tf import ObjectNetwork
    ff = ObjectNetwork(n_repeats=1, scope="collection_controller")
    return FeedforwardCell(ff, output_size, name=name)


def build_controller(output_size, name):
    if cfg.controller_type == "arn":
        return build_attentional_relation_network(output_size, name)
    elif cfg.controller_type == "obj":
        return build_object_network_controller(output_size, name)
    else:
        raise Exception("Unknown controller_type: {}".format(cfg.controller_type))


def build_policy(env, **kwargs):
    if cfg.discrete_actions:
        action_selection = ProductDist(
            Softmax(8, one_hot=False), Softmax(3, one_hot=False),
        )
    else:
        action_selection = ProductDist(
            SigmoidNormal(-1, 1, explore=cfg.explore),
            SigmoidNormal(-1, 1, explore=cfg.explore),
            SigmoidNormal(0, 1, explore=cfg.explore),)
    return Policy(action_selection, env.obs_shape, **kwargs)


# alg config
config.update(
    build_controller=build_controller,
    controller_type="obj",
    d=256,
    layer_norm=True,
    symmetric_op="max",
    use_mask=True,

    # For obj
    build_on_input_network=lambda scope: MLP([128, 128], scope=scope),
    build_on_object_network=lambda scope: MLP([128, 128], scope=scope),
    build_on_output_network=lambda scope: MLP([128, 128, 128], scope=scope),

    # For arn
    build_arn_network=lambda scope: MLP([128, 128], scope=scope),
    build_arn_object_network=lambda scope: MLP([128, 128], scope=scope),
    n_heads=1,

    exploration_schedule=1.0,
    val_exploration_schedule=0.1,

    build_policy=build_policy,
)


if __name__ == "__main__":
    with config:
        env = build_env().gym_env
        agent = game.RandomAgent(env.action_space)
        game.do_rollouts(
            env, agent, render=True,
            callback=lambda action, reward, **kwargs: print("Action: {}, Reward: {}".format(action, reward)))
