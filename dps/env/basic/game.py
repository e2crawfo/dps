import gym
from gym.utils import seeding
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import animation
import os
import imageio
from skimage.transform import resize
import warnings
import copy

import dps
from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.rl.policy import Policy, ProductDist, SigmoidNormal
from dps.utils import square_subplots
from dps.utils.tf import FeedforwardCell, MLP, ScopedFunction, RelationNetwork, RenderHook
from dps.env.env import BatchGymEnv


class Entity(object):
    masks = {}
    images = {}

    def __init__(self, appearance="cross", color="white", shape=(1, 1), position=(0, 0), z=None, reward=0, **kwargs):
        self.appearance = appearance
        self.color = color

        try:
            shape = tuple(shape)
        except (ValueError, TypeError):
            shape = (float(shape), float(shape))
        self.h, self.w = shape
        self.top, self.left = position

        self.alive = True
        self.z = np.random.rand() if z is None else z
        self.reward = reward

        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self):
        self.alive = True

    def get_appearance(self):
        key = (self.color, self.h, self.w)
        image = Entity.images.get(key, None)

        if image is None:
            color_array = np.array(to_rgb(self.color))[None, None, :]
            image = Entity.images[key] = np.tile(color_array, (self.h, self.w, 1))

        key = (self.appearance, self.h, self.w)
        mask = Entity.masks.get(key, None)

        if mask is None:
            f = os.path.join(os.path.dirname(dps.__file__), "shapes", "{}.png".format(self.appearance))
            mask = imageio.imread(f)
            mask = mask[:, :, 3:]
            mask = resize(mask, (self.h, self.w), mode='edge', preserve_range=True)
            mask = Entity.masks[key] = mask / 255.

        return image, mask

    @property
    def shape(self):
        return (self.h, self.w)

    @shape.setter
    def shape(self, _shape):
        self.h, self.w = _shape

    @property
    def position(self):
        return (self.top, self.left)

    @position.setter
    def position(self, _position):
        self.top, self.left = _position

    @property
    def bottom(self):
        return self.top + self.h

    @property
    def right(self):
        return self.left + self.w

    @property
    def y(self):
        return self.top + self.h / 2

    @y.setter
    def y(self, _y):
        self.top = _y - self.h / 2

    @property
    def x(self):
        return self.left + self.w / 2

    @x.setter
    def x(self, _x):
        self.left = _x - self.w / 2

    @property
    def center(self):
        return (self.y, self.x)

    @property
    def area(self):
        return self.h * self.w

    def intersects(self, r2):
        return self.overlap_area(r2) > 0

    def overlap_area(self, r2):
        overlap_bottom = np.minimum(self.bottom, r2.bottom)
        overlap_top = np.maximum(self.top, r2.top)

        overlap_right = np.minimum(self.right, r2.right)
        overlap_left = np.maximum(self.left, r2.left)

        area = np.maximum(overlap_bottom - overlap_top, 0) * np.maximum(overlap_right - overlap_left, 0)
        return area

    def __str__(self):
        return "<{} - ({}:{}, {}:{}), alive={}, z={}, appearance={}, color={}>".format(
            self.__class__, self.top, self.bottom, self.left, self.right, self.alive, self.z, self.appearance, self.color)

    def __repr__(self):
        return str(self)

    def act(self, game):
        pass


def liang_barsky(bottom, top, left, right, y0, x0, y1, x1):
    assert bottom < top
    assert left < right

    dx = x1 - x0
    dy = y1 - y0

    checks = ((-dx, -(left - x0)),
              (dx, right - x0),
              (-dy, -(bottom - y0)),
              (dy, top - y0))

    out_in = [0]
    in_out = [1]

    for p, q in checks:
        if p == 0 and q < 0:
            return None

        if p != 0:
            target_list = out_in if p < 0 else in_out
            target_list.append(q / p)

    _out_in = max(out_in)
    _in_out = min(in_out)

    if _out_in < _in_out:
        return _out_in, _in_out
    else:
        return None


NoAnswer = object()


def _test_liang_barsky(*args, ref_answer=NoAnswer):
    answer = liang_barsky(*args)
    print("{}: {}".format(args, answer))

    if ref_answer is not NoAnswer:
        assert answer == ref_answer


# if __name__ == "__main__":
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, 1.5, 2.5, ref_answer=(1/4, 3/4))
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, 1.5, .99, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, 1.5, 1, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, 1.5, 1.01, ref_answer=(0.5 / 0.51, 1))
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 0.5, -1.5, -2.5, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 2.5, 0.5, 2.5, 2.5, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 0.5, 2.5, 2.5, 2.5, ref_answer=None)
#     _test_liang_barsky(1, 2, 1, 2, 0, 0, 2, 2, ref_answer=(0.5, 1))
#     _test_liang_barsky(1, 2, 1, 2, 0, .99, 2, 2.99, ref_answer=(0.5, 0.505))
#     _test_liang_barsky(1, 2, 1, 2, 1.5, 1.5, 3, 3, ref_answer=(0, 1/3))


class ObjectGame(gym.Env):
    """

    Handles collision detection and scene rendering. Subclass takes care of the rest.

    Subclasses must implement:

    setup_field
    move_entities
    resolve_collision
    get_entity_features

    And optionally:

    compute_reward

    """
    metadata = {'render.modes': ['human']}

    def __init__(
            self, image_shape=(100, 100), background_colour="black", max_episode_length=100, image_obs=False,
            action_space=None, reward_range=None, max_entities=None, entity_feature_dim=None,
            **kwargs):

        self.image_shape = image_shape

        colour = to_rgb(background_colour)
        colour = np.array(colour)[None, None, :]
        self.background_colour = colour

        self.max_episode_length = max_episode_length

        assert action_space is not None
        self.action_space = action_space

        assert reward_range is not None
        self.reward_range = reward_range

        assert max_entities is not None
        self.max_entities = max_entities

        assert entity_feature_dim is not None
        self.entity_feature_dim = entity_feature_dim

        self.image_obs = image_obs
        if image_obs:
            self.observation_space = gym.spaces.Box(0, 1, shape=(*self.image_shape, 3))
        else:
            self.observation_space = gym.spaces.Box(0, 1, shape=(self.max_entities, 4 + self.entity_feature_dim))

        self.seed()
        self.reset()
        self.viewer = None

    def setup_field(self):
        """ Return a list of entities. """
        raise Exception("NotImplemented")

    def move_entities(self, action):
        """ Call `self._move_entity` for each movement. Return reward. """
        raise Exception("NotImplemented")

    def resolve_collision(self, mover, other):
        """ Return (kill, stop, reward) """
        raise Exception("NotImplemented")

    def get_entity_features(entity):
        """ Compute features for an entity. """
        raise Exception("NotImplemented")

    def compute_reward(self):
        """ Compute an additional reward based on final location of entities. """
        return 0.0

    def get_image(self):
        image = np.ones((*self.image_shape, 3)) * self.background_colour

        sorted_entities = sorted(self.entities, key=lambda x: x.z)

        for entity in sorted_entities:
            if not entity.alive:
                continue

            _image, _alpha = entity.get_appearance()

            top = int(entity.top)
            bottom = top + int(entity.h)

            left = int(entity.left)
            right = left + int(entity.w)

            image[top:bottom, left:right, ...] = (
                _alpha * _image + (1 - _alpha) * image[top:bottom, left:right, ...])

        return image

    def get_entities(self):
        height, width = self.image_shape
        representation = np.zeros((self.max_entities, self.entity_feature_dim + 4))

        i = 0
        for entity in self.entities:
            if not entity.alive:
                continue

            features = self.get_entity_features(entity)

            representation[i, :] = (
                (entity.top/height, entity.left/width, entity.h/height, entity.w/width,) +
                tuple(features)
            )

            i += 1

        return representation

    def _move_entity(self, entity, y_step, x_step):
        # For each entity, compute the time of intercept.

        h, w = entity.h, entity.w

        obstacles = []
        for other in self.entities:
            y, x = other.center

            # top, bottom, left, right
            obstacles.append((other, (y - h / 2, y + h / 2, x - w / 2, x + w / 2)))

        # walls
        obstacles.append(("top", (-np.inf, np.ceil(entity.h / 2), -np.inf, np.inf)))
        obstacles.append(("bottom", (self.image_shape[0] - np.ceil(entity.h)/2, np.inf, -np.inf, np.inf)))
        obstacles.append(("left", (-np.inf, np.inf, -np.inf, np.ceil(entity.w)/2)))
        obstacles.append(("right", (-np.inf, np.inf, self.image_shape[1] - np.ceil(entity.w)/2, np.inf)))

        y, x = entity.y, entity.x

        collisions = []
        for other, (top, bottom, left, right) in obstacles:
            if entity is other:
                continue

            alive = getattr(other, "alive", True)
            if not alive:
                continue

            collision = liang_barsky(top, bottom, left, right, y, x, y + y_step, x + x_step)

            if collision is not None:
                collisions.append((collision, other))

        collisions = sorted(collisions)

        total_reward = 0

        stop = False
        for (start, end), other in collisions:
            kill, stop, reward = self.resolve_collision(entity, other)

            if kill:
                other.alive = False

            total_reward += reward

            if stop:
                entity.y = y + (start-1e-6) * y_step
                entity.x = x + (start-1e-6) * x_step

                break

        if not stop:
            entity.y = y + y_step
            entity.x = x + x_step

        return total_reward

    def step(self, action):
        reward = self.move_entities(action)

        info = {}
        self._step += 1

        if self.image_obs:
            obs = self.get_image()
            info['entities'] = self.get_entities()
        else:
            obs = self.get_entities()
            info['image'] = self.get_image()

        done = bool(self.max_episode_length) and self._step >= self.max_episode_length

        return obs, reward, done, info

    def reset(self):
        '''Clear entities and state, call setup_field()'''
        self.entities = self.setup_field()
        self._step = 0
        if self.image_obs:
            obs = self.get_image()
        else:
            obs = self.get_entities()
        return obs

    def render(self, mode='human', close=False):
        if close:
            return

        plt.ion()
        if self.viewer is None:
            self.viewer = plt.imshow(self.get_image())
        self.viewer.set_data(self.get_image())
        plt.pause(1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class CollectionGame(ObjectGame):
    def __init__(
            self, agent_spec, entity_specs, step_size=None,
            min_entities=25, max_overlap=0.2, grid=False, corner=None, **kwargs):
        """
        agent_spec:
            h,w,shape,colour,z
        entity_specs:
            list of h,w,shape,colour,z,reward,logit

        """
        self.agent_spec = copy.deepcopy(agent_spec)
        self.agent_spec['idx'] = 0

        self.entity_specs = copy.deepcopy(entity_specs)
        for i, es in enumerate(self.entity_specs):
            es['idx'] = i + 1

        entity_logits = np.array([es.get('logit', 1) for es in self.entity_specs])
        self.entity_dist = np.exp(entity_logits) / np.exp(entity_logits).sum()

        self.step_size = step_size or min(self.agent_spec.y, self.agent_spec.x)
        self.min_entities = min_entities
        self.max_overlap = max_overlap
        self.grid = grid
        self.corner = corner

        super(CollectionGame, self).__init__(
            action_space=gym.spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1])),
            # action_space=gym.spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1])),
            reward_range=(-10, 10),
            entity_feature_dim=len(entity_specs)+1,
            **kwargs)

    def setup_field(self):
        n_entities = np.random.randint(self.min_entities-1, self.max_entities)

        if self.corner is not None:
            shape = (self.image_shape[0]/2, self.image_shape[1]/2)
            if self.corner == "top_left":
                mask_entity = Entity(position=(0, 0), shape=shape)
            elif self.corner == "top_right":
                mask_entity = Entity(position=(0, self.image_shape[1]/2), shape=shape)
            elif self.corner == "bottom_left":
                mask_entity = Entity(position=(self.image_shape[0]/2, 0), shape=shape)
            elif self.corner == "bottom_right":
                mask_entity = Entity(position=(self.image_shape[0]/2, self.image_shape[1]/2), shape=shape)

        if self.grid:
            n_per_row = int(round(np.sqrt(n_entities)))
            n_entities = n_per_row ** 2
            center_spacing_y = self.image_shape[0] / n_per_row
            center_spacing_x = self.image_shape[1] / n_per_row

            agent = Entity(**self.agent_spec)
            agent.top = np.random.randint(self.image_shape[0]-agent.h)
            agent.left = np.random.randint(self.image_shape[1]-agent.w)

            entities = [agent]

            for i in range(n_per_row):
                for j in range(n_per_row):
                    y = center_spacing_y / 2 + center_spacing_y * i
                    x = center_spacing_x / 2 + center_spacing_x * j

                    spec_idx = np.random.choice(len(self.entity_specs), p=self.entity_dist)
                    spec = self.entity_specs[spec_idx]
                    entity = Entity(**spec)
                    entity.top = y - entity.h / 2
                    entity.left = x - entity.w / 2

                    if self.corner is not None and not mask_entity.intersects(entity):
                        continue

                    self.entities.append(entity)
        else:
            spec_indices = np.random.choice(len(self.entity_specs), size=n_entities, replace=True, p=self.entity_dist)
            specs = [self.agent_spec] + [self.entity_specs[i] for i in spec_indices]
            shapes = [spec['shape'] for spec in specs]

            rectangles = self._sample_entities(shapes, self.max_overlap)
            entities = [Entity(**spec) for spec in specs]
            for rect, entity in zip(rectangles, entities):
                entity.top = rect.top
                entity.left = rect.left

            if self.corner is not None:
                entities = entities[0] + [e for e in entities[1:] if e.intersects(mask_entity)]

        return entities

    def _sample_entities(self, patch_shapes, max_overlap=None, size_std=None):
        if len(patch_shapes) == 0:
            return []

        patch_shapes = np.array(patch_shapes)
        n_rects = patch_shapes.shape[0]

        rects = []

        for i in range(n_rects):
            n_tries = 0
            while True:
                if size_std is None:
                    shape_multipliers = 1.
                else:
                    shape_multipliers = np.maximum(np.random.randn(2) * size_std + 1.0, 0.5)

                m, n = np.ceil(shape_multipliers * patch_shapes[i, :2]).astype('i')

                position = (
                    np.random.randint(0, self.image_shape[0]-m+1),
                    np.random.randint(0, self.image_shape[1]-n+1))

                rect = Entity(position=position, shape=(m, n))

                if max_overlap is None:
                    rects.append(rect)
                    break
                else:
                    violation = False
                    for r in rects:
                        min_area = min(rect.area, r.area)
                        if rect.overlap_area(r) / min_area > max_overlap:
                            violation = True
                            break

                    if not violation:
                        rects.append(rect)
                        break

                n_tries += 1

                if n_tries > 10000:
                    warnings.warn(
                        "Could not fit rectangles. "
                        "(n_rects: {}, image_shape: {}, max_overlap: {})".format(
                            n_rects, self.image_shape, max_overlap))
                    break

        return rects

    def move_entities(self, action):
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
            return (True, False, other.reward)

    def get_entity_features(self, entity):
        return [int(entity.idx == i) for i in range(len(self.entity_specs) + 1)]


class Collection_RenderHook(RenderHook):
    def __init__(self, N=16):
        self.N = N

    def __call__(self, updater):
        plt.ion()
        for learner in updater.learners:
            with learner:
                rollouts = updater.env.do_rollouts(
                    policy=learner.pi, n_rollouts=self.N, mode='val')

        if updater.env.gym_env.image_obs:
            obs = rollouts.obs
        else:
            obs = rollouts.image

        fig, axes = square_subplots(self.N, figsize=(5, 5))
        plt.subplots_adjust(top=0.95, bottom=0, left=0, right=1, wspace=0.1, hspace=0.1)

        images = []
        for i, ax in enumerate(axes.flatten()):
            ax.set_aspect("equal")
            ax.set_axis_off()
            image = ax.imshow(np.zeros(obs.shape[2:]))
            images.append(image)

        def animate(t):
            for i in range(self.N):
                images[i].set_array(obs[t, i, :, :, :])

        anim = animation.FuncAnimation(fig, animate, frames=len(rollouts), interval=500)

        path = updater.exp_dir.path_for('plots', 'animation.gif')  # Don't open gifs with VLC
        anim.save(path, writer='imagemagick')

        if cfg.show_plots:
            plt.show()

        plt.close(fig)


class RandomAgent(object):
    def __init__(self, action_space, persist_prob=0.0):
        self.action_space = action_space
        self.persist_prob = persist_prob
        self.action = None

    def act(self, observation, reward, done):
        if self.action is None or np.random.rand() > self.persist_prob:
            self.action = self.action_space.sample()

        return self.action


class Filter(object):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def __call__(self, n):
        return np.random.rand() < self.keep_prob


def do_rollouts(env, agent, render=False):
    reward = 0
    done = False

    ob = env.reset()

    while True:
        env.reset()

        if render:
            env.render()

        done = False
        while not done:
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)

            if render:
                env.render()

            print('Action:', action, 'Reward:', reward)
            if done:
                break


def build_env():
    entity_size = (3, 3)  # (14, 14)
    agent_spec = dict(appearance="cross", color="green", shape=entity_size)

    entity_specs = [
        dict(appearance="circle", color="blue", shape=entity_size, reward=-1),
        dict(appearance="square", color="red", shape=entity_size, reward=1)
    ]
    gym_env = CollectionGame(
        agent_spec=agent_spec, entity_specs=entity_specs,
        image_shape=(25, 25), background_colour="white", min_entities=10, max_entities=10,
        max_overlap=0.0, step_size=5)
    # gym_env = CollectionGame(
    #     agent_spec=agent_spec, entity_specs=entity_specs,
    #     image_shape=(100, 100), background_colour="white", min_entities=2, max_entities=4,
    #     max_overlap=0.0, step_size=7)

    return BatchGymEnv(gym_env=gym_env)


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


def build_collection_game_controller(output_size, name):
    # ff = MLP([256, 256, 256], scope="collection_game_controller")
    # ff = Backbone(scope="collection_game_controller")
    ff = MLP([100, 100], scope="collection_controller")
    # ff = RelationNetwork(scope="collection_controller")
    return FeedforwardCell(ff, output_size, name=name)


def build_policy(env, **kwargs):
    action_selection = ProductDist(
        SigmoidNormal(-1, 1, explore=True),
        SigmoidNormal(-1, 1, explore=True),
        SigmoidNormal(0, 1, explore=True),)
    return Policy(action_selection, env.obs_shape, **kwargs)


config = DEFAULT_CONFIG.copy()


config.update(
    env_name="collection_game",

    build_env=build_env,

    build_controller=build_collection_game_controller,
    build_relation_network_f=lambda scope: MLP([100, 100], scope=scope),
    build_relation_network_g=lambda scope: MLP([100, 100], scope=scope),
    f_dim=128,
    symmetric_op="max",

    build_policy=build_policy,
    exploration_schedule=0.1,
    val_exploration_schedule=0.01,

    n_controller_units=64,

    epsilon=0.0,
    opt_steps_per_update=1,
    sub_batch_size=0,

    value_weight=0.0,
    T=20,

    n_val=100,
    batch_size=16,
    render_hook=Collection_RenderHook(),
    render_step=1000,
    eval_step=100,
    display_step=100,
    stopping_criteria="reward_per_ep,max",
    threshold=1000,
)


if __name__ == "__main__":
    agent_spec = dict(appearance="cross", color="green", shape=(14, 14))

    entity_specs = [
        dict(appearance="circle", color="blue", shape=(14, 14), reward=1),
        dict(appearance="square", color="red", shape=(14, 14), reward=-1),
    ]

    gym_env = CollectionGame(
        agent_spec=agent_spec, entity_specs=entity_specs,
        image_shape=(100, 100), background_colour="white", min_entities=10, max_entities=20,
        max_overlap=0.0, step_size=25)
    agent = RandomAgent(gym_env.action_space)
    do_rollouts(gym_env, agent, render=True)
