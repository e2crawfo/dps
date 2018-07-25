'''Base class for the DSRL paper toy game'''
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import to_rgb
import imageio
import os
from skimage.transform import resize
import inspect
from gym_recording.wrappers import TraceRecordingWrapper
import warnings

from dps import cfg
from dps.datasets.base import RawDataset
from dps.datasets.atari import RewardClassificationDataset
from dps.utils import Param, square_subplots
from dps.utils.tf import RenderHook


class Entity(object):
    def __init__(self, y, x, h, w, kind, center=False, z=None):
        if center:
            self.top = y - h / 2
            self.left = x - w / 2
        else:
            self.top = y
            self.left = x

        self.h = h
        self.w = w
        self.alive = True
        self.z = np.random.rand() if z is None else z
        self.kind = kind

    @property
    def right(self):
        return self.left + self.w

    @property
    def bottom(self):
        return self.top + self.h

    def intersects(self, r2):
        return self.overlap_area(r2) > 0

    def overlap_area(self, r2):
        overlap_bottom = np.minimum(self.bottom, r2.bottom)
        overlap_top = np.maximum(self.top, r2.top)

        overlap_right = np.minimum(self.right, r2.right)
        overlap_left = np.maximum(self.left, r2.left)

        area = np.maximum(overlap_bottom - overlap_top, 0) * np.maximum(overlap_right - overlap_left, 0)
        return area

    def centre(self):
        return (
            self.top + self.h / 2.,
            self.left + self.w / 2.
        )

    @property
    def area(self):
        return self.h * self.w

    def __str__(self):
        return "<{}:{} {}:{}, alive={}, z={}, kind={}>".format(
            self.top, self.bottom, self.left, self.right, self.alive, self.z, self.kind)

    def __repr__(self):
        return str(self)


ACTION_LOOKUP = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT'
}


class XO_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    entity_kinds = sorted("agent circle cross".split())

    def __init__(
            self, image_shape=(100, 100), background_colour="black", entity_colours="white white white",
            entity_sizes="7 10 10", min_entities=25, max_entities=50, max_overlap=0.2, collection_overlap=0.25,
            step_size=None, grid=False, cross_prob=0.5, corner=None, max_episode_length=100, image_obs=False):

        self.image_shape = image_shape
        self.background_colour = background_colour

        if isinstance(entity_sizes, str):
            entity_sizes = entity_sizes.split()
        try:
            entity_sizes = int(entity_sizes)
        except (ValueError, TypeError):
            pass
        else:
            entity_sizes = [entity_sizes] * 3

        self.entity_sizes = {
            entity_type: int(c)
            for entity_type, c in zip(self.entity_kinds, entity_sizes)}

        self.min_entities = min_entities
        self.max_entities = max_entities

        self.max_overlap = max_overlap
        self.collection_overlap = collection_overlap
        self.step_size = step_size or int(self.entity_sizes["agent"] / 2)

        self.grid = grid
        self.cross_prob = cross_prob
        self.corner = corner

        self.action_space = spaces.Discrete(4)
        self.image_obs = image_obs
        if image_obs:
            self.observation_space = spaces.Box(0, 1, shape=(*self.image_shape, 3))
        else:
            self.observation_space = spaces.Box(0, 1, shape=(self.max_entities+1, 7))
        self.reward_range = (-1, 1)

        self.entities = {'cross': [], 'circle': []}
        self.agent = None

        self.masks = {}
        for entity_type in self.entity_kinds:
            f = os.path.join(os.path.dirname(__file__), "xo_images", "{}.png".format(entity_type))
            mask = imageio.imread(f)
            entity_size = self.entity_sizes[entity_type]
            mask = resize(mask, (entity_size, entity_size), mode='edge', preserve_range=True)
            self.masks[entity_type] = np.tile(mask[..., 3:], (1, 1, 3)) / 255.

        self.background_colour = None
        if background_colour:
            colour = to_rgb(background_colour)
            colour = np.array(colour)[None, None, :]
            self.background_colour = colour

        self.entity_colours = None
        if entity_colours:
            if isinstance(entity_colours, str):
                entity_colours = entity_colours.split()

            self.entity_colours = {
                entity_type: np.array(to_rgb(c))[None, None, :]
                for entity_type, c in zip(self.entity_kinds, entity_colours)}

        self.max_episode_length = max_episode_length

        self.seed()
        self.reset()
        self.viewer = None

    def get_image(self):
        image = np.ones((*self.image_shape, 3)) * self.background_colour

        all_entities = []
        for entity_type, entities in self.entities.items():
            all_entities.extend(entities)
        all_entities.append(self.agent)

        all_entities = sorted(all_entities, key=lambda x: x.z)

        for entity in all_entities:
            if not entity.alive:
                continue

            _alpha = self.masks[entity.kind]
            entity_size = self.entity_sizes[entity.kind]
            if self.entity_colours is None:
                _image = np.random.rand(entity_size, entity_size, 3)
            else:
                _image = np.tile(self.entity_colours[entity.kind], (entity_size, entity_size, 1))

            top = int(entity.top)
            bottom = top + int(entity.h)

            left = int(entity.left)
            right = left + int(entity.w)

            image[top:bottom, left:right, ...] = _alpha * _image + (1 - _alpha) * image[top:bottom, left:right, ...]

        return image

    def get_entities(self):
        height, width = self.image_shape
        representation = np.zeros((self.max_entities+1, 7))
        agent = self.agent
        representation[0, :] = (agent.top/height, agent.left/width, agent.h/height, agent.w/width, 1, 0, 0)

        i = 1
        for kind, entities in self.entities.items():
            features = (0, 1, 0) if kind == "cross" else (0, 0, 1)

            for entity in entities:
                if not entity.alive:
                    continue

                representation[i, :] = (entity.top/height, entity.left/width, entity.h/height, entity.w/width,) + features
                i += 1

        return representation

    def step(self, action):
        action_type = ACTION_LOOKUP[int(action)]

        if action_type == 'UP':
            collision = self._move_agent(0, self.step_size)
        if action_type == 'DOWN':
            collision = self._move_agent(0, -self.step_size)
        if action_type == 'LEFT':
            collision = self._move_agent(-self.step_size, 0)
        if action_type == 'RIGHT':
            collision = self._move_agent(self.step_size, 0)

        reward = collision['cross'] - collision['circle']

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
        self.entities = {'cross': [], 'circle': []}
        self.agent = None
        self.setup_field()
        self._step = 0
        if self.image_obs:
            obs = self.get_image()
        else:
            obs = self.get_entities()
        return obs

    def setup_field(self):
        n_entities = np.random.randint(self.min_entities, self.max_entities+1)

        agent_size = self.entity_sizes["agent"]
        top = np.random.randint(self.image_shape[0]-agent_size)
        left = np.random.randint(self.image_shape[1]-agent_size)
        self.agent = Entity(top, left, agent_size, agent_size, kind="agent")

        if self.corner is not None:
            if self.corner == "top_left":
                mask_entity = Entity(0, 0, self.image_shape[0]/2, self.image_shape[1]/2, kind="mask")
            elif self.corner == "top_right":
                mask_entity = Entity(0, self.image_shape[1]/2, self.image_shape[0]/2, self.image_shape[1]/2, kind="mask")
            elif self.corner == "bottom_left":
                mask_entity = Entity(self.image_shape[0]/2, 0, self.image_shape[0]/2, self.image_shape[1]/2, kind="mask")
            elif self.corner == "bottom_right":
                mask_entity = Entity(
                    self.image_shape[0]/2, self.image_shape[1]/2, self.image_shape[0]/2, self.image_shape[1]/2, kind="mask")

        if self.grid:
            n_per_row = int(round(np.sqrt(n_entities)))
            n_entities = n_per_row ** 2
            center_spacing_y = self.image_shape[0] / n_per_row
            center_spacing_x = self.image_shape[1] / n_per_row

            for i in range(n_per_row):
                for j in range(n_per_row):
                    y = center_spacing_y / 2 + center_spacing_y * i
                    x = center_spacing_x / 2 + center_spacing_x * j

                    if np.random.rand() < self.cross_prob:
                        entity_kind = 'cross'
                    else:
                        entity_kind = 'circle'

                    entity_size = self.entity_sizes[entity_kind]
                    entity = Entity(y, x, entity_size, entity_size, center=True, kind=entity_kind)

                    if self.corner is not None and not mask_entity.intersects(entity):
                        continue

                    self.entities[entity_kind].append(entity)
        else:
            entity_kinds = np.random.choice(["cross", "circle"], n_entities, p=[self.cross_prob, 1-self.cross_prob])
            entity_shapes = [(self.entity_sizes[kind], self.entity_sizes[kind]) for kind in entity_kinds]
            entities = self._sample_entities(entity_shapes, self.max_overlap)

            for i, (e, kind) in enumerate(zip(entities, entity_kinds)):
                if self.corner is not None and not mask_entity.intersects(e):
                    continue

                e.kind = kind
                self.entities[kind].append(e)

        # Clear objects that overlap with the agent originally
        self._move_agent(0, 0)

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

                rect = Entity(
                    np.random.randint(0, self.image_shape[0]-m+1),
                    np.random.randint(0, self.image_shape[1]-n+1), m, n, kind=None)

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

    def _make_shape(self, entity_type):
        return self.masks[entity_type]

    def _move_agent(self, x_step, y_step):
        agent = self.agent
        collisions = {entity_type: 0 for entity_type in self.entities}

        for entity_type, entities in self.entities.items():
            for i, entity in enumerate(entities):
                min_area = min(agent.area, entity.area)
                if entity.alive and agent.overlap_area(entity) / min_area > self.collection_overlap:
                    collisions[entity_type] += 1
                    entity.alive = False

        n_segments = 10

        if not x_step and not y_step:
            return collisions

        for i in range(n_segments):
            new_x = agent.left + x_step / n_segments
            new_y = agent.top + y_step / n_segments

            wall_collision = (
                new_y + agent.h > self.image_shape[0] or
                new_y < 0 or
                new_x + agent.w > self.image_shape[1] or
                new_x < 0
            )

            if wall_collision:
                break
            else:
                agent.left = new_x
                agent.top = new_y

                for entity_type, entities in self.entities.items():
                    for i, entity in enumerate(entities):
                        min_area = min(agent.area, entity.area)
                        if entity.alive and agent.overlap_area(entity) / min_area > self.collection_overlap:
                            collisions[entity_type] += 1
                            entity.alive = False

        return collisions


class XO_RenderHook(RenderHook):
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


def do_rollouts(env, agent, n_examples, balanced, render=False):
    reward = 0
    done = False

    ob = env.reset()

    n_examples_per_class = None
    if balanced:
        n_examples_per_class = n_examples

    while True:
        env.reset()

        if render:
            env.render()

        done = False
        while not done:
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)

            if balanced and min(env.recording.keep_freqs.values()) >= n_examples_per_class:
                env.close()
                return
            elif not balanced and env.recording.n_recorded >= n_examples:
                env.close()
                return

            if render:
                env.render()

            print('Action:', action, 'Reward:', reward)
            if done:
                break


xo_env_params = dict(inspect.signature(XO_Env.__init__).parameters)
del xo_env_params['self']


class XO_RewardRawDataset(RawDataset):
    classes = Param()

    n_examples = Param()
    persist_prob = Param(0.0)
    keep_prob = Param(1)
    balanced = Param(True)

    def _make(self):
        env_kwargs = {k: v for k, v in self.param_values().items() if k in xo_env_params}
        env = XO_Env(**env_kwargs)

        reward_classes = self.classes if self.balanced else None

        env = TraceRecordingWrapper(
            env, directory=self.directory, episode_filter=Filter(1),
            frame_filter=Filter(self.keep_prob), reward_classes=reward_classes)
        env.seed(0)
        agent = RandomAgent(env.action_space, self.persist_prob)

        do_rollouts(env, agent, self.n_examples, self.balanced)


for name, p in xo_env_params.items():
    setattr(XO_RewardRawDataset, name, Param(p.default))


class XO_RewardClassificationDataset(RewardClassificationDataset):
    n_examples = Param()
    persist_prob = Param(0.0)
    keep_prob = Param(1)
    balanced = Param(True)

    rl_data_location = None

    def _make(self):
        raw_kwargs = {k: v for k, v in self.param_values().items() if k in XO_RewardRawDataset.param_names()}
        raw_dataset = XO_RewardRawDataset(**raw_kwargs)
        self.rl_data_location = raw_dataset.directory

        super(XO_RewardClassificationDataset, self)._make()


for name, p in xo_env_params.items():
    setattr(XO_RewardClassificationDataset, name, Param(p.default))


if __name__ == "__main__":
    dataset = XO_RewardClassificationDataset(
        classes=[-1, 0, 1], n_examples=100, persist_prob=0.3,
        max_episode_length=51, image_shape=(72, 72), min_entities=20, max_entities=30,
    )

    import tensorflow as tf
    sess = tf.Session()
    with sess.as_default():
        dataset.visualize()
