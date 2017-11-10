import tensorflow as tf
import numpy as np

from dps.utils import Parameterized, Param
from dps.utils.tf import build_gradient_train_op


class Optimizer(Parameterized):
    def __init__(self, agents):
        self.agents = agents

    def trainable_variables(self):
        return [v for agent in self.agents for v in agent.trainable_variables()]


class StochasticGradientDescent(Optimizer):
    opt_steps_per_update = Param(1)
    sub_batch_size = Param(0)
    lr_schedule = Param()
    max_grad_norm = Param(None)
    noise_schedule = Param(None)

    def __init__(self, agents, alg, **kwargs):
        super(StochasticGradientDescent, self).__init__(agents)
        self.alg = alg

    def build_update(self, context):
        tvars = self.trainable_variables()

        # `context.objective` is the quantity we want to maximize, but TF minimizes.
        self.train_op, train_summaries = build_gradient_train_op(
            -context.objective, tvars, self.alg, self.lr_schedule, self.max_grad_norm,
            self.noise_schedule)

        for s in train_summaries:
            context.add_train_summary(s)

    def update(self, n_rollouts, feed_dict):
        sess = tf.get_default_session()
        for epoch in range(self.opt_steps_per_update):
            if not self.sub_batch_size:
                sess.run(self.train_op, feed_dict=feed_dict)
            else:
                for fd in self.subsample_feed_dict(n_rollouts, feed_dict):
                    sess.run(self.train_op, feed_dict=fd)

    def subsample_feed_dict(self, n_rollouts, feed_dict):
        updates_per_epoch = int(np.floor(n_rollouts / self.sub_batch_size))
        permutation = np.random.permutation(n_rollouts)
        offset = 0
        for i in range(updates_per_epoch):
            indices = permutation[offset:offset+self.sub_batch_size]
            fd = {}
            for k, v in feed_dict.items():
                if isinstance(v, np.ndarray):
                    fd[k] = v[:, indices, ...]
                else:
                    fd[k] = v
            yield fd
            offset += self.sub_batch_size
