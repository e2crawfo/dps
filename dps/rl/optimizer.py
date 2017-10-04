import tensorflow as tf

from dps.utils.tf import build_gradient_train_op


class Optimizer(object):
    def __init__(self, agents):
        self.agents = agents

    def trainable_variables(self):
        return [v for agent in self.agents for v in agent.trainable_variables()]


class StochasticGradientDescent(Optimizer):
    def __init__(self, agents, alg, lr_schedule, max_grad_norm=None, noise_schedule=None):
        super(StochasticGradientDescent, self).__init__(agents)
        self.alg = alg
        self.lr_schedule = lr_schedule
        self.max_grad_norm = max_grad_norm
        self.noise_schedule = noise_schedule

    def build_update(self, context):
        tvars = self.trainable_variables()
        self.train_op, train_summaries = build_gradient_train_op(
            context.loss, tvars, self.alg, self.lr_schedule, self.max_grad_norm,
            self.noise_schedule)

        for s in train_summaries:
            context.add_train_summary(s)

    def update(self, feed_dict):
        sess = tf.get_default_session()
        sess.run(self.train_op, feed_dict=feed_dict)
