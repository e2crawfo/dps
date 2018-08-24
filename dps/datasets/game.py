import numpy as np

from dps.env.basic import game, collect
from dps.utils import Param
from dps.datasets.base import ImageDataset, ImageFeature, NestedListFeature, IntegerFeature


class RolloutCallback(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_kept = 0

    def __call__(self, new_obs, info, **kwargs):
        if np.random.rand() < self.dataset.keep_prob:
            if self.n_kept % 1000 == 0:
                print("Working on datapoint {}...".format(self.n_kept))

            if self.dataset.env.image_obs:
                image = new_obs
                entities = info['entities']
            else:
                image = info['image']
                entities = new_obs

            image = (image*255).astype('uint8')

            height, width = self.dataset.image_shape

            annotations = [
                [0, top * height, top * height + h, left * width, left * width + w]
                for _, top, left, h, w, *_ in entities]

            self.dataset._write_example(image=image, annotations=annotations, label=0)
            self.n_kept += 1

    def finished(self):
        return self.n_kept >= self.dataset.n_examples


class GameDataset(ImageDataset):
    env = Param()
    keep_prob = Param()

    depth = 3

    @property
    def features(self):
        if self._features is None:
            self._features = [
                ImageFeature("image", self.obs_shape),
                NestedListFeature("annotations", 5),
                IntegerFeature("label", None)]

        return self._features

    def _make(self):
        agent = game.RandomAgent(self.env.action_space)
        callback = RolloutCallback(self)
        game.do_rollouts(self.env, agent, callback=callback)


if __name__ == "__main__":
    with collect.config.copy(n_examples=100, keep_prob=0.1):
        env = collect.build_env().gym_env
        dataset = GameDataset(env=env)

        import tensorflow as tf
        sess = tf.Session()
        with sess.as_default():
            dataset.visualize(16)
