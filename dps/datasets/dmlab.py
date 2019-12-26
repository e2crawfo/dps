import numpy as np
import matplotlib.pyplot as plt

from dps.datasets.base import ImageDataset, ImageFeature, ArrayFeature, FloatFeature
from dps.utils import Param, Parameterized, gen_seed, animate


# class RandomAgent(Parameterized):
#     action_repeat = Param()
# 
#     def __init__(self, **kwargs):
#         self.reset()
# 
#     def set_env(self, env):
#         self.env = env
# 
#     def reset(self):
#         self.action = None
#         self.steps_since_sample = self.action_repeat
# 
#     def step(self, reward, obs):
#         if self.steps_since_sample == self.action_repeat:
#             self.action = self.env.sample()
#             self.steps_since_sample = 0
# 
#         self.steps_since_sample += 1
# 
#         return self.action


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class DiscretizedRandomAgent:
    """Simple agent for DeepMind Lab."""

    ACTIONS = {
        'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
        'look_right': _action(20, 0, 0, 0, 0, 0, 0),
        'look_up': _action(0, 10, 0, 0, 0, 0, 0),
        'look_down': _action(0, -10, 0, 0, 0, 0, 0),
        'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
        'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
        'forward': _action(0, 0, 0, 1, 0, 0, 0),
        'backward': _action(0, 0, 0, -1, 0, 0, 0),
        'fire': _action(0, 0, 0, 0, 1, 0, 0),
        'jump': _action(0, 0, 0, 0, 0, 1, 0),
        'crouch': _action(0, 0, 0, 0, 0, 0, 1)
    }

    def __init__(self, action_spec, **kwargs):
        self.ACTION_LIST = list(self.ACTIONS.items())

    def step(self, unused_reward, unused_image):
        """Gets an image state and a reward, returns an action."""
        return self.ACTION_LIST[np.random.choice(len(self.ACTION_LIST))][1]

    def reset(self):
        pass


class RotatingAgent(DiscretizedRandomAgent):
    ACTIONS = {
        'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
        'look_right': _action(20, 0, 0, 0, 0, 0, 0),
    }


class SpringAgent:
    """A random agent using spring-like forces for its action evolution."""

    def __init__(self, action_spec, **kwargs):
        self.action_spec = action_spec
        print('Starting random spring agent. Action spec:', action_spec)

        self.omega = np.array([
            0.1,  # look left-right
            0.1,  # look up-down
            0.1,  # strafe left-right
            0.1,  # forward-backward
            0.0,  # fire
            0.0,  # jumping
            0.0  # crouching
        ])

        self.velocity_scaling = np.array([2.5, 2.5, 0.01, 0.01, 1, 1, 1])

        self.indices = {a['name']: i for i, a in enumerate(self.action_spec)}
        self.mins = np.array([a['min'] for a in self.action_spec])
        self.maxs = np.array([a['max'] for a in self.action_spec])
        self.reset()

        self.rewards = 0

    def critically_damped_derivative(self, t, omega, displacement, velocity):
        """Critical damping for movement.
        I.e., x(t) = (A + Bt) \exp(-\omega t) with A = x(0), B = x'(0) + \omega x(0)
        See
          https://en.wikipedia.org/wiki/Damping#Critical_damping_.28.CE.B6_.3D_1.29
        for details.
        Args:
          t: A float representing time.
          omega: The undamped natural frequency.
          displacement: The initial displacement at, x(0) in the above equation.
          velocity: The initial velocity, x'(0) in the above equation
        Returns:
           The velocity x'(t).
        """
        a = displacement
        b = velocity + omega * displacement
        return (b - omega * t * (a + t * b)) * np.exp(-omega * t)

    def step(self, reward, unused_frame):
        """Gets an image state and a reward, returns an action."""
        self.rewards += reward

        action = (self.maxs - self.mins) * np.random.random_sample(
            size=[len(self.action_spec)]) + self.mins

        # Compute the 'velocity' 1 time unit after a critical damped force
        # dragged us towards the random `action`, given our current velocity.
        self.velocity = self.critically_damped_derivative(1, self.omega, action,
                                                          self.velocity)

        # Since walk and strafe are binary, we need some additional memory to
        # smoothen the movement. Adding half of action from the last step works.
        self.action = self.velocity / self.velocity_scaling + 0.5 * self.action

        # Fire with p = 0.01 at each step
        self.action[self.indices['FIRE']] = int(np.random.random() > 0.99)

        # Jump/crouch with p = 0.005 at each step
        self.action[self.indices['JUMP']] = int(np.random.random() > 0.995)
        self.action[self.indices['CROUCH']] = int(np.random.random() > 0.995)

        # Clip to the valid range and convert to the right dtype
        return self.clip_action(self.action)

    def clip_action(self, action):
        return np.clip(action, self.mins, self.maxs).astype(np.intc)

    def reset(self):
        self.velocity = np.zeros([len(self.action_spec)])
        self.action = np.zeros([len(self.action_spec)])


class LabDataset(ImageDataset):
    level = Param()
    level_config = Param()

    agent_class = Param()
    agent_params = Param()

    get_depth = Param()
    renderer = Param()
    image_shape = Param()
    crop = Param()

    fps = Param()
    n_frames = Param()

    n_episodes = Param()
    max_episode_length = Param()
    sample_density = Param()
    max_examples = Param()
    frame_skip = Param()

    _obs_shape = None
    depth = 3

    @property
    def features(self):
        if self._features is None:
            self._features = [
                ImageFeature("image", self.obs_shape),
                ArrayFeature("depth", (*self.obs_shape[:-1], 1)),
                ArrayFeature("action", (self.n_frames, 7,), dtype=np.int32),
                ArrayFeature("reward", (self.n_frames,), dtype=np.float32),
            ]

        return self._features

    @property
    def obs_shape(self):
        if self._obs_shape is None:
            if self.postprocessing:
                self._obs_shape = (self.n_frames, *self.tile_shape, self.depth,)
            elif self.crop is not None:
                t, b, l, r = self.crop
                crop_shape = (b - t, r - l)
                self._obs_shape = (self.n_frames, *crop_shape, self.depth,)
            else:
                self._obs_shape = (self.n_frames, *self.image_shape, self.depth,)

        return self._obs_shape

    def _make(self):
        import deepmind_lab  # noqa

        config = dict(
            height=self.image_shape[0],
            width=self.image_shape[1],
            fps=self.fps,
            mixerSeed=gen_seed(),
        )

        if 'dummy' not in self.level_config:
            config.update(self.level_config)
        config = {k: str(v) for k, v in config.items()}

        obs_key = 'RGBD_INTERLEAVED' if self.get_depth else 'RGB_INTERLEAVED'

        observations = [obs_key]

        env = deepmind_lab.Lab(self.level, observations,
                               config=config,
                               renderer=self.renderer)

        agent = self.agent_class(env.action_spec(), **self.agent_params)

        reward = 0

        self._n_examples = 0

        for n in range(self.n_episodes):
            env.reset()
            agent.reset()
            reward = 0
            ep = dict(o=[], a=[], r=[])

            for step in range(self.max_episode_length):
                print("Ep: {}, step: {}".format(n, step))

                obs = env.observations()[obs_key]

                if not env.is_running():
                    print('Environment stopped early')
                    ep['o'].append(obs)
                    break

                action = agent.step(reward, obs)
                reward = env.step(action, num_steps=1)

                ep['o'].append(obs)
                ep['a'].append(action)
                ep['r'].append(reward)

            do_break = self._process_ep(**ep)
            if do_break:
                break

    def _process_ep(self, o, a, r):
        """ process one episode """

        episode_length = len(a)  # o is one step longer than a and r

        n_frames_to_fetch = (self.n_frames - 1) * self.frame_skip + 1
        max_start_idx = episode_length - n_frames_to_fetch + 1

        n_samples = int(np.ceil(self.sample_density * max_start_idx))

        indices = np.random.choice(max_start_idx, size=n_samples, replace=False)

        step = self.frame_skip

        for start in indices:
            if self._n_examples % 100 == 0:
                print("Processing example {}".format(self._n_examples))

            end = start + n_frames_to_fetch

            _o = np.array(o[start:end:step])
            _a = np.array(a[start:end:step])
            _r = np.array(r[start:end:step])

            assert len(_o) == self.n_frames
            assert len(_a) == self.n_frames
            assert len(_r) == self.n_frames

            if self.crop is not None:
                top, bot, left, right = self.crop
                _o = _o[:, top:bot, left:right, ...]

            self._write_example(image=_o, action=_a, reward=_r)
            self._n_examples += 1

            if self._n_examples >= self.max_examples:
                print("Found maximum of {} examples, done.".format(self._n_examples))
                return True

    def _write_example(self, image, **kwargs):
        if self.get_depth:
            image, depth = image[..., :3], image[..., 3:4]
        else:
            depth = np.zeros_like(image[..., :1])
        super()._write_example(image=image, depth=depth, **kwargs)

    def visualize(self, n=4):
        sample = self.sample(n)
        images = sample["image"]
        depth = sample["depth"]
        actions = sample["action"]
        rewards = sample["reward"]

        labels = ["actions={}, rewards={}".format(a, r) for a, r in zip(actions, rewards)]

        fig, *_ = animate(images, depth, labels=labels)
        plt.subplots_adjust(top=0.95, bottom=0, left=0, right=1, wspace=0.05, hspace=0.1)

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    import tensorflow as tf
    from dps import cfg
    from dps.utils import Config

    config = Config(
        level='tests/empty_room_test',
        # level='stairway_to_melon',
        level_config=dict(minimalUI="true"),

        get_depth=True,
        agent_class=RotatingAgent,
        # agent_class=SpringAgent,
        agent_params=dict(dummy=1),

        renderer='hardware',
        image_shape=(200, 200),
        crop=None,

        fps=60,
        n_frames=10,

        n_episodes=10,
        max_episode_length=1000,
        sample_density=0.1,
        max_examples=100,
        frame_skip=5,
        N=16,
    )

    with config:
        config.update_from_command_line()
        dset = LabDataset()

        sess = tf.Session()
        with sess.as_default():
            dset.visualize(cfg.N)
