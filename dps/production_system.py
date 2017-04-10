import numpy as np
from collections import namedtuple
import abc
from future.utils import with_metaclass

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from gym import Env
from gym import spaces
from gym.utils import seeding

from dps.utils import training, default_config
from dps.environment import DifferentiableEnv


params = 'core_network controller action_selection use_act T'


class ProductionSystem(namedtuple('ProductionSystem', params.split())):
    """ A production system."""
    pass


class CoreNetwork(object):
    """ A core network inside a production system.

    A specification of the functionality of the core network; any state must be maintained externally.

    Parameters
    ----------
    n_actions: int > 0
        The number of actions recognized by the core network.
    body: callable
        Accepts a tensor representing action activations and an object
        created by calling ``instantiate`` on ``register_spec``
        (which stores tensors representing register values) and outputs tensors
        representing the new values of the registers after running the
        core network for one step.
    register_spec: instance of a subclass of RegisterSpec
        Provides information about the registers operated on by this core network.

    """
    def __init__(self, n_actions, body, register_spec, name):
        self.n_actions = n_actions
        self.body = body
        self.register_spec = register_spec
        self.name = name

        self._graph = None
        self._register_placeholders = None

    def _build_graph(self):
        with tf.namescope('core_network'):
            aa_placeholders = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='action_activations')
            self._register_placeholders = self.register_spec.build_placeholders()
            self._graph = self.body(aa_placeholders, self.register_placeholders)

    @property
    def graph(self):
        if self._graph is None:
            self._build_graph()
        return self._graph

    @property
    def register_placeholders(self):
        if self._register_placeholders is None:
            self._build_graph()
        return self._register_placeholders

    def run(self, action_activations, registers):
        feed_dict = {ph: v for ph, v in zip(self.register_placeholders, registers)}
        feed_dict['action_activations'] = action_activations

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
            output = sess.run(self.graph)
        return output


class ProductionSystemCell(RNNCell):
    def __init__(self, psystem, exploration, reuse=None):
        """ ``controller`` may be an RNN cell (if it has internal state) or a tensorflow function. """
        self.core_network, self.controller, self.action_selection, _, _ = psystem
        self.exploration = exploration

        if isinstance(self.controller, RNNCell):
            self._state_size = (
                self.controller.state_size,
                self.core_network.register_spec.state_size())
        else:
            self._state_size = self.core_network.register_spec.state_size()
        self._reuse = reuse

    def __call__(self, inputs, state, scope=None):
        """ Run the production system for one step.

        Has no proper inputs or outputs, everything is contained in the registers.

        Args:
            inputs: noise, passed to the core network and the action selection mechanism for doing reparamaterization trick
            state: if `self.state_size` is an integer, this should be a `2-D Tensor`
                with shape `[batch_size x (self.state_size]`.  Otherwise, if
                `self.state_size` is a tuple of integers, this should be a tuple
                with shapes `[batch_size x s] for s in self.state_size`.
            scope: VariableScope for the created subgraph; defaults to class name.
        Returns:
            A pair containing:
            - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
            - New state: Either a single `2-D` tensor, or a tuple of tensors matching
              the arity and shapes of `state`.
        """
        with tf.name_scope(scope or 'production_system_cell'):
            controller_state = None

            if isinstance(self.controller, RNNCell):
                controller_state = state[0]
                registers = state[1]
            else:
                registers = state

            with tf.name_scope('controller'):
                obs = self.core_network.register_spec.as_obs(registers, visible_only=1)
                utilities, new_controller_state = self.controller(obs, controller_state)

            with tf.name_scope('action_selection'):
                action_activations = self.action_selection(utilities, self.exploration)

            with tf.name_scope('core_network'):
                _, new_registers = self.core_network.body(action_activations, registers)

            if isinstance(self.controller, RNNCell):
                new_state = (new_controller_state, new_registers)
            else:
                new_state = new_registers

            # Return state as output since ProductionSystemCell has no other meaningful output,
            # and this has benefit of making all intermediate states accessible when using
            # used with tf function ``dynamic_rnn``
            return new_state, new_state

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def zero_state(self, batch_size, dtype):
        if isinstance(self.controller, RNNCell):
            initial_state = (
                self.controller.zero_state(batch_size, dtype),
                self.core_network.register_spec.build_placeholders(dtype))
        else:
            initial_state = self.core_network.register_spec.build_placeholders(dtype)
        return initial_state


class ProductionSystemFunction(object):
    """ Wrapper around a tensorflow function that implements a production system.

    Parameters
    ----------
    pystem: ProductionSystem
        The production system that we want to turn into a tensorflow function.
    exploration: scalar Tensor
        A scalar, passed to the action selection function, giving the amount
        of exploration to use at any point in time

    """
    def __init__(self, psystem, exploration, scope=None):

        self.core_network, self.controller, self.action_selection, self.use_act, self.T = psystem
        self.exploration = exploration

        self.inputs, self.initial_state, self.ps_cell, self.states, self.final_state = (
            self._build_ps_function(psystem, exploration, scope))

    def build_feeddict(self, registers, T=None):
        fd = {}

        if not isinstance(registers, self.core_network.register_spec.namedtuple):
            registers = self.core_network.register_spec.from_obs(registers)

        batch_size = registers[0].shape[0]

        if not self.use_act and T is None:
            raise Exception("Number of steps ``T`` must be supplied when not using ACT.")
        if not self.use_act:
            # The first dim of this dummy input determines the number of time steps.
            fd[self.inputs] = np.zeros((T, batch_size, 1))

        initial_registers = (
            self.initial_state[1] if isinstance(self.controller, RNNCell) else self.initial_state)
        for placeholder, value in zip(initial_registers, registers):
            fd[placeholder] = value

        return fd

    @staticmethod
    def _build_ps_function(psystem, exploration, scope=None):
        if psystem.use_act:
            raise NotImplemented()
            # Code for doing Adaptive Computation Time
            # with vs.variable_scope(scope or type(self).__name__):
            #     # define within cell constants/ counters used to control while loop for ACTStep
            #     prob = tf.constant(0.0, tf.float32, name="prob")
            #     prob_compare = tf.constant(0.0, tf.float32, name="prob_compare")
            #     counter = tf.constant(0.0, tf.float32, [self.batch_size], name="counter")
            #     acc_outputs = tf.zeros_like(state, tf.float32, name="output_accumulator")
            #     acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")
            #     batch_mask = tf.constant(True, tf.bool, [self.batch_size])

            #     # While loop stops when this predicate is FALSE.
            #     # Ie all (probability < 1-eps AND counter < N) are false.
            #     def halting_predicate(batch_mask, prob_compare, prob,
            #                   counter, state, input, acc_output, acc_state):
            #         return tf.reduce_any(tf.logical_and(
            #                 tf.less(prob_compare,self.one_minus_eps),
            #                 tf.less(counter,self.N)))

            #     # Do while loop iterations until predicate above is false.
            #     _,_,remainders,iterations,_,_,output,next_state = \
            #         tf.while_loop(halting_predicate, self.act_step,
            #         [batch_mask, prob_compare, prob,
            #          counter, state, inputs, acc_outputs, acc_states])
            # tf.while_loop(cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)

        else:
            with tf.name_scope(scope or "production_system_function"):
                # (max_time, batch_size, 1)
                inputs = tf.placeholder(tf.float32, shape=(None, None, 1), name="inputs")

                ps_cell = ProductionSystemCell(psystem, exploration)

                batch_size = tf.shape(inputs)[1]
                initial_state = ps_cell.zero_state(batch_size, tf.float32)

                # ps_cell is hacked to provide its internal state as output
                # so that we have access to internal states from every time
                # step, instead of just the final time step
                states, final_state = dynamic_rnn(
                    ps_cell, inputs, initial_state=initial_state,
                    parallel_iterations=1, swap_memory=False,
                    time_major=True)

        return inputs, initial_state, ps_cell, states, final_state

    def get_register_values(self, *names, as_obs=True):
        if not names:
            names = self.core_network.register_spec.names

        values = []
        for name in names:
            registers = self.states[1] if isinstance(self.controller, RNNCell) else self.states
            try:
                r = getattr(registers, name)
            except AttributeError:
                r = registers[self.core_network.register_spec.names.index(name)]
                if isinstance(r, tuple):
                    r = r[0]
            values.append(r)

        if as_obs:
            if len(values) > 1:
                return tf.concat(values, axis=2)
            return values[0]
        else:
            return tuple(values)

    def __str__(self):
        return ("<ProductionSystemFunction - core_network={}, controller={}, "
                "action_selection={}, use_act={}>".format(
                    self.core_network, self.controller, self.action_selection, self.use_act))

    def __call__(self, inputs, T=None):
        # The compiled function should act like this: if it was built with act=True,
        # and is given a T value of None, then it makes use of ACT. Otherwise it runs
        # for T steps no matter what, and the action activations for the stop action
        # are essentially ignored.
        T = self.T if T is None else T
        return self.function(inputs, T)


class ProductionSystemEnv(Env):
    """ An environment that combines a CoreNetwork with a gym environment.

    Parameters
    ----------
    psystem: ProductionSystem
        The production system whose core network we want to bind to the environment.
    env: gym Env
        The environment to bind ``psystem``'s core network to. The core network will
        be run "inside" this environment.

    """
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, psystem, env):
        self.core_network, _, _, self.use_act, self.T = psystem.core_network
        self.env = env

        # Extra action is for stopping. If T is not 0, this action will be effectively ignored.
        self.n_actions = self.core_network.n_actions + 1

        # TODO: define a more specific space for this
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_actions,))

        obs_dim = sum(self.core_network.register_spec.get_register_dims(visible_only=True))
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,))

        self.reward_range = env.reward_range

        self.reset()

    def __str__(self):
        return "<ProductionSystemEnv, core_network={}, env={}, T={}>".format(self.core_network, self.env, self.T)

    def _step(self, action):
        assert self.action_space.contains(action), (
            "{} ({}) is not a valid action for env {}." % (action, type(action), self))

        # TODO: need to decide how to stop when not using discrete actions. Maybe the
        # activation of the stop action should be interpreted as the probability of stopping,
        # from which we then sample. Definitely something like this.
        step_external = (
            (not self.T and action == self.core_network.n_actions) or
            (self.T and self.t % self.T == 0))

        # Either update external or internal environment, never both.
        if step_external:
            external_action = self.registers.get_output()

            external_obs, reward, done, info = self.env.step(external_action)
            self.registers.set_input(external_obs)
        else:
            reward, done, info = 0.0, False, {}

            action = action[:-1]  # The core network knows nothing about stopping computation, so cut off the stop action.
            self.registers = self.core_network(action, self.registers)

        self.t += 1

        return self.core_network.register_spec.as_obs(self.registers, visible_only=True), reward, done, info

    def _reset(self):
        self.t = 0

        external_obs = self.env.reset()

        self.registers = self.core_network.register_spec.instantiate(
            batch_size=1, np_random=self.np_random, input=external_obs)
        return self.core_network.register_spec.as_obs(self.registers, visible_only=True)

    def _render(self, mode='human', close=False):
        raise NotImplementedError()

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


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

        self._n_experiences += batch_size

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
        The production system to use to learn about the problem.
    exploration: scalar Tensor
        A scalar giving the amount of exploration to use at any point in time.
        Is passed to the action selection function.
    rl_alg: ReinforcementLearningAlgorithm
        The reinforcement learning algorithm to use for optimizing parameters
        of the controller.

    """
    def __init__(self, env, psystem, exploration, rl_alg):
        super(ReinforcementLearningUpdater, self).__init__()

        self.env = env
        self.psystem = psystem
        self.ps_env = ProductionSystemEnv(psystem, env)

        self.exploration = exploration
        self.rl_alg = rl_alg

    def _update(self, batch_size, summary_op=None):
        pass
