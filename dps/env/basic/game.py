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
from dps.utils import square_subplots, generate_perlin_noise_2d, Config, Param, Parameterized
from dps.utils.tf import RenderHook
from dps.env.env import BatchGymEnv


class Entity(object):
    masks = {}
    images = {}

    def __init__(self, appearance="plus", color="white", shape=(1, 1), position=(0, 0), z=None, reward=0, **kwargs):
        self.appearance = appearance

        if isinstance(color, str):
            color = color.split()
        self.color = np.random.choice(color)

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

        noise_res = getattr(self, "noise_res", None)
        if noise_res is not None:
            noise = generate_perlin_noise_2d(self.shape, self.noise_res, normalize=True)
            mask = mask * noise[:, :, None]

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

    @center.setter
    def center(self, _center):
        self.y, self.x = _center

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


class Rectangle(object):
    def __init__(self, bottom, top, left, right):
        self.bottom = bottom
        self.top = top
        self.left = left
        self.right = right

    def collision(self, y0, x0, y1, x1):
        return liang_barsky(self.bottom, self.top, self.left, self.right, y0, x0, y1, x1)


class Ellipse(object):
    def __init__(self, cy, cx, y_radius, x_radius):
        self.cy = cy
        self.cx = cx
        self.y_radius_2 = y_radius**2
        self.x_radius_2 = x_radius**2

    def collision(self, y0, x0, y1, x1):
        y0 -= self.cy
        y1 -= self.cy
        x0 -= self.cx
        x1 -= self.cx

        a = (y1 - y0)**2 / self.y_radius_2 + (x1 - x0)**2 / self.x_radius_2
        b = 2 * ((y1 - y0) * y0 / self.y_radius_2 + (x1 - x0) * x0 / self.x_radius_2)
        c = y0**2 / self.y_radius_2 + x0**2 / self.x_radius_2 - 1

        disc = b**2 - 4 * a * c
        if disc <= 0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        if t2 * t1 < 0:  # start inside
            t1 = 0
        else:  # start outside
            if t1 < 0:  # Line segment points away from circle
                return None
            if t1 > 1:  # Line segment doesn't go far enough
                return None

        t2 = min(1, t2)

        return (t1, t2)


class ObjectGame(Parameterized, gym.Env):
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

    image_shape = Param()
    background_colour = Param()
    max_episode_length = Param()
    image_obs = Param()
    max_entities = Param()

    def __init__(
            self, action_space=None, reward_range=None, entity_feature_dim=None, **kwargs):

        colour = to_rgb(self.background_colour)
        colour = np.array(colour)[None, None, :]
        self.background_colour = colour

        assert action_space is not None
        self.action_space = action_space

        assert reward_range is not None
        self.reward_range = reward_range

        assert entity_feature_dim is not None
        self.entity_feature_dim = entity_feature_dim

        if self.image_obs:
            self.observation_space = gym.spaces.Box(0, 1, shape=(*self.image_shape, 3), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(
                0, 1, shape=(self.max_entities, 5 + self.entity_feature_dim), dtype=np.float32)

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
        representation = np.zeros((self.max_entities, 5 + self.entity_feature_dim))

        entities = np.random.permutation(self.entities)

        i = 0
        for entity in entities:
            if not entity.alive:
                continue

            features = self.get_entity_features(entity)

            representation[i, :] = (
                (1.0, entity.top/height, entity.left/width, entity.h, entity.w,) + tuple(features)
            )

            i += 1

        return representation

    def _move_entity(self, entity, y_step, x_step):
        # For each entity, compute the time of intercept.

        h, w = entity.h, entity.w

        obstacles = []
        for other in self.entities:
            y, x = other.center

            shape = Ellipse(y, x, h/2, w/2)

            # obstacles.append((other, (y - h / 2, y + h / 2, x - w / 2, x + w / 2)))
            obstacles.append((other, shape))

        # walls
        obstacles.append(("top", Rectangle(-np.inf, np.ceil(entity.h / 2), -np.inf, np.inf)))
        obstacles.append(("bottom", Rectangle(self.image_shape[0] - np.ceil(entity.h)/2, np.inf, -np.inf, np.inf)))
        obstacles.append(("left", Rectangle(-np.inf, np.inf, -np.inf, np.ceil(entity.w)/2)))
        obstacles.append(("right", Rectangle(-np.inf, np.inf, self.image_shape[1] - np.ceil(entity.w)/2, np.inf)))

        y, x = entity.y, entity.x

        collisions = []
        for other, shape in obstacles:
            if entity is other:
                continue

            alive = getattr(other, "alive", True)
            if not alive:
                continue

            collision = shape.collision(y, x, y + y_step, x + x_step)
            # collision = liang_barsky(top, bottom, left, right, y, x, y + y_step, x + x_step)

            if collision is not None:
                collisions.append((collision, other))

        collisions = sorted(collisions, key=lambda x: x[0])

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
        reward += self.compute_reward()

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


def sample_entities(image_shape, patch_shapes, max_overlap=None, size_std=None, masks=None):
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
                np.random.randint(0, image_shape[0]-m+1),
                np.random.randint(0, image_shape[1]-n+1))

            rect = Entity(position=position, shape=(m, n))

            mask_valid = True
            if masks is not None:
                mask_idx_y = int(masks[i].shape[0] * rect.center[0] / image_shape[0])
                mask_idx_x = int(masks[i].shape[1] * rect.center[1] / image_shape[1])
                mask_valid = masks[i][mask_idx_y, mask_idx_x] > 0.5

            if mask_valid:
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
                        n_rects, image_shape, max_overlap))
                break

    return rects


class CollectionGame(ObjectGame):
    agent_spec = Param()
    entity_specs = Param()

    step_size = Param()
    min_entities = Param()
    max_overlap = Param()
    grid = Param()
    corner = Param()

    def __init__(self, **kwargs):
        self.agent_spec = copy.deepcopy(self.agent_spec)
        self.agent_spec['idx'] = 0

        self.entity_specs = copy.deepcopy(self.entity_specs)
        for i, es in enumerate(self.entity_specs):
            es['idx'] = i + 1

        entity_logits = np.array([es.get('logit', 1) for es in self.entity_specs])
        self.entity_dist = np.exp(entity_logits) / np.exp(entity_logits).sum()

        super(CollectionGame, self).__init__(
            action_space=gym.spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32),
            reward_range=(-10, 10),
            entity_feature_dim=len(self.entity_specs)+1,
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
            else:
                raise Exception("Unknown value for `corner`: {}".format(self.corner))

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

            rectangles = sample_entities(self.image_shape, shapes, self.max_overlap)
            entities = [Entity(**spec) for spec in specs]
            for rect, entity in zip(rectangles, entities):
                entity.top = rect.top
                entity.left = rect.left

            if self.corner is not None:
                entities = entities[0] + [e for e in entities[1:] if e.intersects(mask_entity)]

        return entities

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
                rollouts = updater.env.do_rollouts(policy=learner.pi, n_rollouts=self.N, T=cfg.T, mode='val')

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

    def act(self, observation):
        if self.action is None or np.random.rand() > self.persist_prob:
            self.action = self.action_space.sample()

        return self.action


def do_rollouts(env, agent, render=False, callback=None, n_rollouts=None):
    idx = 0
    while n_rollouts is None or idx < n_rollouts:
        obs = env.reset()

        if render:
            env.render()

        done = False
        while not done:
            action = agent.act(obs)
            new_obs, reward, done, info = env.step(action)

            if callback:
                callback(obs=obs, action=action, reward=reward, info=info, done=done, new_obs=new_obs)

            obs = new_obs

            if render:
                env.render()

            if callback and hasattr(callback, "finished") and callback.finished():
                return

            if done:
                break

        idx += 1


def build_env():
    gym_env = CollectionGame()
    return BatchGymEnv(gym_env=gym_env)


entity_size = (5, 5)

config = Config(
    env_name="collection_game",

    build_env=build_env,
    T=20,
    max_episode_length=20,

    entropy_weight=0.0,
    batch_size=16,
    render_hook=Collection_RenderHook(N=16),
    render_step=1000,
    eval_step=100,
    display_step=100,
    stopping_criteria="reward_per_ep,max",
    threshold=1000,

    image_shape=(25, 25),
    background_colour="white",
    image_obs=False,
    max_entities=5,

    agent_spec=dict(appearance="plus", color="green", shape=entity_size),
    entity_specs=[
        dict(appearance="circle", color="blue", shape=entity_size, reward=-1),
        dict(appearance="square", color="red", shape=entity_size, reward=1)
    ],
    min_entities=5,
    max_overlap=0.0,
    step_size=5,
    corner=None,
    grid=False,

    explore=False,
    discrete_actions=False,
)


if __name__ == "__main__":
    with config:
        env = build_env().gym_env
        agent = RandomAgent(env.action_space)
        do_rollouts(env, agent, render=True)
