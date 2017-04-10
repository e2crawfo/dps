import abc
from future.utils import with_metaclass

import tensorflow as tf

from dps.production_system import ProductionSystemFunction, ProductionSystemEnv
from dps.utils import training, default_config
from dps.environment import DifferentiableEnv


class Updater(with_metaclass(abc.ABCMeta, object)):
    def __init__(self):
        self._n_experiences = 0

    @property
    def n_experiences(self):
        return self._n_experiences

    def update(self, batch_size, summary_op=None):
        self._n_experiences += batch_size
        return self._update(batch_size, summary_op)

    @abc.abstractmethod
    def _update(self, batch_size, summary_op=None):
        raise NotImplementedError()


class DifferentiableUpdater(Updater):
    """ Update parameters of a production system using vanilla gradient descent.

    All components of ``psystem`` (core network, controller, action selection) must be
    differentiable to apply this updater.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    psystem: ProductionSystem
        The production system to use to learn about the problem.
    exploration: scalar Tensor
        A scalar giving the amount of exploration to use at any point in time.
        Is passed to the action selection function.
    rl_alg: ReinforcementLearningAlgorithm (required iff env is not differentiable)
        The reinforcement learning algorithm to use for optimizing parameters
        of psystem when the environment is not fully differentiable (in this case,
        the differentiable function obtained by merging the core network, controller
        and action selection method is used as a parameterized policy whose
        parameters are learned by this RL algorithm.

    """
    def __init__(self, env, psystem, exploration, global_step, rl_alg=None):
        # This call has to take place in the context of both a default session and a default graph
        super(DifferentiableUpdater, self).__init__()
        self.env = env
        self.psystem = psystem

        self.exploration = exploration
        self.rl_alg = rl_alg

        self.ps_func = ProductionSystemFunction(psystem, exploration=exploration)
        self.loss, self.target_placeholders = env.loss(self.ps_func.get_register_values()[-1, :, :])
        tf.summary.scalar('loss', self.loss)

        self.train_op = training(self.loss, global_step)

    def _update(self, batch_size, summary_op=None):
        # This call has to take place in the context of both a default session and a default graph
        config = default_config()
        if isinstance(self.env, DifferentiableEnv):
            env, loss = self.env, self.loss
            train_op, ps_func, targets = self.train_op, self.ps_func, self.target_placeholders
            sess = tf.get_default_session()

            batch_x, batch_y = env.train.next_batch(batch_size)
            if config.debug:
                print("x", batch_x)
                print("y", batch_y)

            feed_dict = ps_func.build_feeddict(batch_x, self.psystem.T)
            feed_dict[targets] = batch_y

            if summary_op is not None:
                train_summary, train_loss, _ = sess.run([summary_op, loss, train_op], feed_dict=feed_dict)

                val_x, val_y = env.val.next_batch()
                val_feed_dict = ps_func.build_feeddict(val_x, self.psystem.T)
                val_feed_dict[targets] = val_y

                val_summary, val_loss = sess.run([summary_op, loss], feed_dict=val_feed_dict)
                return train_summary, train_loss, val_summary, val_loss
            else:
                train_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                return train_loss
        else:
            pass

    def checkpoint(self, path, saver):
        pass


class ReinforcementLearningUpdater(Updater):
    """ Update parameters of a production system using reinforcement learning.

    There are no restrictions on ``psystem`` for using this method.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    psystem: ProductionSystem
        The production system to use to solve the problem.
    exploration: scalar Tensor
        A scalar giving the amount of exploration to use at any point in time.
        Is passed to the action selection function.
    rl_alg: ReinforcementLearningAlgorithm
        The reinforcement learning algorithm to use for optimizing parameters
        of the controller.

    """
    def __init__(self, env, psystem, exploration, rl_alg):
        # This call has to take place in the context of both a default session and a default graph
        super(ReinforcementLearningUpdater, self).__init__()

        self.env = env
        self.psystem = psystem
        self.ps_env = ProductionSystemEnv(psystem, env)

        self.exploration = exploration
        self.rl_alg = rl_alg

        self.record_loss = tf.placeholder(tf.float32, shape=())
        tf.summary.scalar('loss', self.loss)

    def _update(self, batch_size, summary_op=None):
        # This call has to take place in the context of both a default session and a default graph
        config = default_config()
        env.set_mode('train', batch_size)
        done = False

        obs = env.reset()

        while not done:
            action = self.policy.act(obs)
            obs, reward, done, info = env.step(action)


        if isinstance(self.env, DifferentiableEnv):
            env, loss = self.env, self.loss
            train_op, ps_func, targets = self.train_op, self.ps_func, self.target_placeholders
            sess = tf.get_default_session()

            batch_x, batch_y = env.train.next_batch(batch_size)
            if config.debug:
                print("x", batch_x)
                print("y", batch_y)

            feed_dict = ps_func.build_feeddict(batch_x, self.psystem.T)
            feed_dict[targets] = batch_y

            if summary_op is not None:
                train_summary, train_loss, _ = sess.run([summary_op, loss, train_op], feed_dict=feed_dict)

                val_x, val_y = env.val.next_batch()
                val_feed_dict = ps_func.build_feeddict(val_x, self.psystem.T)
                val_feed_dict[targets] = val_y

                val_summary, val_loss = sess.run([summary_op, loss], feed_dict=val_feed_dict)
                return train_summary, train_loss, val_summary, val_loss
            else:
                train_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                return train_loss
        else:
            pass


class ReinforcementLearningAlgorithm(object):
    def __init__(self):
        self.policy = None

    def set_policy(policy):
        self.policy = policy

    def add_experience(self, experience, behaviour_policy=None):
        if not self.policy:
            raise ValueError("Policy has not been set using ``set_policy``.")
        # Make it so every argument can be a batch.
        state, action, reward, state_prime, action_prime = experience


