import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from dps.datasets.base import ImageDataset, ImageFeature, ArrayFeature, IntegerFeature
from dps.utils import Param, animate, map_structure, numpy_print_options, RunningStats


class RandomAgent:
    """Simple agent for DeepMind Lab."""

    _ACTIONS = dict(
        turn_left=0,
        turn_right=1,
        move_forward=2,
        move_back=3,
        pickup=4,
        drop=5,
        toggle=6,
        done=7
    )

    ACTIONS = None

    def __init__(self, action_repeat=1, **kwargs):
        if self.ACTIONS is None:
            self.actions = list(self._ACTIONS.values())
        else:
            self.actions = [self._ACTIONS[k] for k in self.ACTIONS.split()]

        self.action_repeat = action_repeat
        self.reset()

    def reset(self):
        self.action = None
        self.steps_since_sample = self.action_repeat

    def step(self, unused_reward, unused_image):
        """Gets an image state and a reward, returns an action."""

        if self.steps_since_sample == self.action_repeat:
            self.action = np.random.choice(self.actions)
            self.steps_since_sample = 0

        self.steps_since_sample += 1

        return self.action


class RandomMovementAgent(RandomAgent):
    """ Doesn't use any of the non-movement actions. """

    ACTIONS = "turn_left turn_right move_forward move_back"


class MiniworldDataset(ImageDataset):
    level = Param()
    level_config = Param()

    agent_class = Param()
    agent_params = Param()

    # image_shape = Param()
    image_shape = (60, 80)
    crop = Param()

    n_frames = Param()
    n_episodes = Param()
    max_episode_length = Param()
    sample_density = Param()
    max_examples = Param()
    frame_skip = Param()

    _artifact_names = ['pose_t_mean', 'pose_t_std', 'n_examples']
    angle_order = ['pitch', 'yaw', 'roll']
    angle_units = 'deg'

    _obs_shape = None
    depth = 3
    action_dim = 8

    @property
    def features(self):
        if self._features is None:
            self._features = [
                ImageFeature("image", self.obs_shape),
                ArrayFeature("depth", (*self.obs_shape[:-1], 1)),
                ArrayFeature("pose_r", (self.n_frames, 3,)),
                ArrayFeature("pose_t", (self.n_frames, 3,)),
                ArrayFeature("action", (self.n_frames, self.action_dim,), dtype=np.int32),
                ArrayFeature("reward", (self.n_frames,), dtype=np.float32),
                IntegerFeature("idx"),
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
        import gym  # noqa
        import gym_miniworld  # noqa

        env = gym.make(self.level)

        agent = self.agent_class(**self.agent_params)

        reward = 0

        self._n_examples = 0

        pose_t_stats = RunningStats()

        for n in range(self.n_episodes):
            env.reset()
            agent.reset()
            reward = 0
            done = False
            ep = defaultdict(list)

            for step in range(self.max_episode_length):
                print("Ep: {}, step: {}".format(n, step))

                """
                Keep in mind coordinate system of miniworld:
                y axis increases up, x and z increase horizontally (pos z is to the right of pos x, so it
                is a righthanded system). Angle 0 looks down positive x axis.

                env.agent.dir is a single number giving rotation around the positive (up-pointing) y axis
                (so pretty much the yaw).

                """
                obs = dict(
                    pose_r=np.array([0.0, env.agent.dir, 0.0]),
                    pose_t=np.array(env.agent.pos),
                    depth=env.render_depth(),
                    image=env.render_obs(),
                )

                pose_t = obs['pose_t']
                pose_t = pose_t.reshape(-1, pose_t.shape[-1])
                pose_t_stats.add(pose_t)

                if done:
                    print('Environment stopped early')
                    ep['o'].append(obs)
                    break

                action = agent.step(reward, obs)
                _, reward, done, _ = env.step(action)

                one_hot_action = np.zeros(self.action_dim)
                one_hot_action[action] = 1.0

                ep['o'].append(obs)
                ep['a'].append(one_hot_action)
                ep['r'].append(reward)

            do_break = self._process_ep(**ep)
            if do_break:
                break

        pose_t_mean, pose_t_var = pose_t_stats.get_stats()

        artifacts = dict(
            pose_t_mean=pose_t_mean,
            pose_t_std=np.sqrt(pose_t_var),
            n_examples=self._n_examples
        )

        return artifacts

    def _process_ep(self, o, a, r):
        """ process one episode """
        episode_length = len(a)  # o is one step longer than a and r, so use a

        n_frames_to_fetch = (self.n_frames - 1) * self.frame_skip + 1

        max_start_idx = episode_length - n_frames_to_fetch + 1

        if max_start_idx < 0:
            return False

        n_samples = int(np.ceil(self.sample_density * max_start_idx))

        indices = np.random.choice(max_start_idx, size=n_samples, replace=False)

        step = self.frame_skip

        for start in indices:
            if self._n_examples % 100 == 0:
                print("Processing example {}".format(self._n_examples))

            end = start + n_frames_to_fetch

            _o = o[start:end:step]
            assert len(_o) == self.n_frames

            _o = map_structure(
                lambda *v: np.stack(v, axis=0), *_o,
                is_leaf=lambda v: isinstance(v, np.ndarray))

            _a = np.array(a[start:end:step])
            _r = np.array(r[start:end:step])

            assert len(_a) == self.n_frames
            assert len(_r) == self.n_frames

            if self.crop is not None:
                top, bot, left, right = self.crop
                _o['image'] = _o['image'][:, top:bot, left:right, ...]

            self._write_example(idx=self._n_examples, action=_a, reward=_r, **_o)
            self._n_examples += 1

            if self._n_examples >= self.max_examples:
                print("Found maximum of {} examples, done.".format(self._n_examples))
                return True

    def visualize(self, n=4):
        sample = self.sample(n)
        images = sample["image"]
        depth = sample["depth"]
        normalized_depth = (depth - depth.min()) / ((depth.max() - depth.min()) + 1e-6)
        actions = sample["action"]
        rewards = sample["reward"]
        pose_r = sample["pose_r"]
        pose_t = sample["pose_t"]
        indices = sample["idx"]

        with numpy_print_options(precision=2):
            text = [
                ["idx={}\nt={}\npose_r={}\npose_t={}\nr={}\na={}".format(i, t, _pr, _pt, _r, _a)
                 for t, (_a, _r, _pr, _pt) in enumerate(zip(a, r, pr, pt))]
                for i, a, r, pr, pt in zip(indices, actions, rewards, pose_r, pose_t)
            ]
            text = np.array(text)

        fig, *_ = animate(images, normalized_depth, text=text)
        plt.subplots_adjust(top=0.95, bottom=0, left=0, right=1, wspace=0.05, hspace=0.1)

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    import tensorflow as tf
    from dps import cfg
    from dps.utils import Config

    config = Config(
        level='MiniWorld-CollectHealth-v0',
        # level='MiniWorld-PutNext-v0',
        # level='MiniWorld-Maze-v0',
        # level='MiniWorld-Sidewalk-v0',
        # level='MiniWorld-FourRooms-v0',
        # level='MiniWorld-Hallway-v0',
        level_config=dict(dummy=True),

        agent_class=RandomMovementAgent,
        agent_params=dict(dummy=1),

        # image_shape=(100, 100),
        crop=None,

        n_frames=20,
        n_episodes=10,
        max_episode_length=2000,
        sample_density=0.1,
        max_examples=100,
        frame_skip=1,
        N=16,
    )

    with config:
        config.update_from_command_line()
        dset = MiniworldDataset()

        sess = tf.Session()
        with sess.as_default():
            dset.visualize(cfg.N)