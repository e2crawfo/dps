import numpy as np
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from gym.utils import seeding

from dps.environment import BatchBox, Env


params = 'core_network policy use_act T'


class ProductionSystem(namedtuple('ProductionSystem', params.split())):
    """ A production system."""
    pass


class CoreNetwork(object):
    """ A core network inside a production system.

    A specification of the functionality of the core network; state must be maintained externally.

    Parameters
    ----------
    n_actions: int > 0
        The number of actions recognized by the core network. Doesn't include a stopping actions.
    __call__:
        Accepts a tensor representing action activations and an object
        created by calling ``instantiate`` on ``register_spec``
        (which stores tensors representing register values) and outputs tensors
        representing the new values of the registers after running the
        core network for one step.
    register_spec: instance of a subclass of RegisterSpec
        Provides information about the registers operated on by this core network.

    """
    _n_actions = None
    _register_spec = None

    def __init__(self):
        self._graph = None
        self._action_activations_ph = None
        self._register_ph = None

    def assert_defined(self, attr):
        assert getattr(self, attr) is not None, (
            "Instances of subclasses of CoreNetwork must "
            "specify a value for attr {}.".format(attr))

    @property
    def n_actions(self):
        self.assert_defined('_n_actions')
        return self._n_actions

    @property
    def obs_dim(self):
        return sum(shape[1] for shape in self.register_spec.shapes(visible_only=True))

    @property
    def register_spec(self):
        self.assert_defined('_register_spec')
        return self._register_spec

    def __call__(self, action_activations, registers):
        """ Returns: Tensors representing action_activations, new_registers. """
        raise NotImplementedError()

    def _build_graph(self):
        with tf.name_scope('core_network'):
            self._action_activations_ph = tf.placeholder(
                tf.float32, shape=[None, self.n_actions], name='action_activations')
            self._register_ph = self.register_spec.build_placeholders()
            self._graph = self(self._action_activations_ph, self._register_ph)

    def run(self, action_activations, registers):
        if self._graph is None:
            self._build_graph()

        feed_dict = {ph: v for ph, v in zip(self._register_ph, registers)}
        feed_dict[self._action_activations_ph] = action_activations

        with tf.Session() as sess:
            output = sess.run(self._graph, feed_dict=feed_dict)
        return output


class ProductionSystemCell(RNNCell):
    def __init__(self, psystem, reuse=None):
        self.core_network, self.policy, _, _ = psystem

        self._state_size = (
            self.policy.state_size,
            self.core_network.register_spec.state_size())

        self._reuse = reuse

    def __call__(self, inputs, state, scope=None):
        with tf.name_scope(scope or 'production_system_cell'):
            policy_state = None

            policy_state = state[0]
            registers = state[1]

            with tf.name_scope('policy'):
                obs = self.core_network.register_spec.as_obs(registers, visible_only=1)
                action_activations, new_policy_state = self.policy(obs, policy_state)

            with tf.name_scope('core_network'):
                # Strip off the last action, which is the stopping action.
                new_registers = self.core_network(action_activations[:, :-1], registers)

            new_state = (new_policy_state, new_registers)

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
        initial_state = (
            self.policy.zero_state(batch_size, dtype),
            self.core_network.register_spec.build_placeholders(dtype))
        return initial_state


class ProductionSystemFunction(object):
    """ Wrapper around a tensorflow function that implements a production system.

    Parameters
    ----------
    psystem: ProductionSystem
        The production system that we want to turn into a tensorflow function.

    """
    def __init__(self, psystem, scope=None):

        self.core_network, self.policy, self.use_act, self.T = psystem

        self.inputs, self.initial_state, self.ps_cell, self.states, self.final_state = (
            self._build_ps_function(psystem, scope))

    def build_feeddict(self, inp, T=None):
        batch_size = inp.shape[0]

        registers = self.core_network.register_spec.instantiate(batch_size=batch_size)
        self.core_network.register_spec.set_input(registers, inp)

        fd = {}

        T = T or self.T
        if not self.use_act:
            # first dim of this dummy input determines the number of time steps.
            fd[self.inputs] = np.zeros((T, batch_size, 1))

        register_ph = self.initial_state[1]
        for ph, value in zip(register_ph, registers):
            fd[ph] = value

        return fd

    def get_output(self):
        output = self.core_network.register_spec.get_output(self.states[1])
        return output[-1, :, :]

    @staticmethod
    def _build_ps_function(psystem, scope=None):
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

                ps_cell = ProductionSystemCell(psystem)

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
        registers = self.states[1]
        return self.core_network.register_spec.get_register_values(registers, *names, as_obs=as_obs, axis=2)

    def __str__(self):
        return ("<ProductionSystemFunction - core_network={}, policy={}, use_act={}>".format(
            self.core_network, self.policy, self.policy, self.use_act))

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
        self.core_network, _, self.use_act, self.T = psystem
        self.env = env

        # Extra action is for stopping. If T is not 0, this action will be effectively ignored.
        self.n_actions = self.core_network.n_actions + 1
        self.action_space = BatchBox(low=0.0, high=1.0, shape=(None, self.n_actions))

        obs_dim = sum(
            shape[1] for shape
            in self.core_network.register_spec.shapes(visible_only=True))
        self.observation_space = BatchBox(
            low=-np.inf, high=np.inf, shape=(None, obs_dim))

        self.reward_range = env.reward_range

        self._seed()
        self.reset()

    def set_mode(self, kind, batch_size):
        self.env.set_mode(kind, batch_size)

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
            (self.T and self.t != 0 and self.t % self.T == 0))

        # Either update external or internal environment, never both.
        if step_external:
            external_action = self.core_network.register_spec.get_output(self.registers)
            external_obs, reward, done, info = self.env.step(external_action)
            self.core_network.register_spec.set_input(self.registers, external_obs)
        else:
            reward, done, info = np.zeros(action.shape[0]), False, {}
            action = action[:, :-1]  # The core network knows nothing about stopping computation, so cut off the stop action.
            self.registers = self.core_network.run(action, self.registers)
        self.t += 1

        return self.core_network.register_spec.as_obs(self.registers, visible_only=True), reward, done, info

    def _reset(self):
        self.t = 0

        external_obs = self.env.reset()

        self.registers = self.core_network.register_spec.instantiate(
            batch_size=external_obs.shape[0], np_random=self.np_random)
        self.core_network.register_spec.set_input(self.registers, external_obs)
        return self.core_network.register_spec.as_obs(self.registers, visible_only=True)

    def _render(self, mode='human', close=False):
        if not close:
            raise NotImplementedError()

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
