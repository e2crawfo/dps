import numpy as np
from collections import namedtuple
import os
import copy
import pandas as pd
from tabulate import tabulate

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from gym.utils import seeding

from dps.environment import BatchBox, Env
from dps.updater import DifferentiableUpdater
from dps.rl import REINFORCE
from dps.utils import default_config, build_decaying_value
from dps.train import Curriculum


params = 'env core_network policy use_act T'


class ProductionSystem(namedtuple('ProductionSystem', params.split())):
    """ A production system."""

    def build_psystem_env(self):
        return ProductionSystemEnv(self)

    def build_psystem_func(self, sample=False):
        return ProductionSystemFunction(self, sample=sample)

    def visualize(self, mode, n_rollouts, sample):
        ps_func = self.build_psystem_func(sample=sample)
        env = self.env
        env.set_mode(mode, n_rollouts)

        obs = env.reset()

        final_registers = None

        registers, actions, rewards = [], [], []

        done = False
        while not done:
            if final_registers is None:
                fd = ps_func.build_feeddict(inp=obs)
            else:
                start_registers = copy.deepcopy(final_registers)
                self.core_network.register_spec.set_input(start_registers, obs)
                fd = ps_func.build_feeddict(registers=start_registers)

            sess = tf.get_default_session()
            reg, a, final = sess.run(
                [ps_func.registers,
                 ps_func.action_activations,
                 ps_func.final_registers],
                feed_dict=fd)

            external_action = self.core_network.register_spec.get_output(final)
            new_obs, reward, done, info = env.step(external_action)

            registers.append(reg)
            actions.append(a)

            _reward = np.zeros(a.shape[:-1])
            _reward[-1, :] = reward
            rewards.append(_reward)

            obs = new_obs

        external_step_lengths = [a.shape[0] for a in actions]

        registers = self.core_network.register_spec.concatenate(registers, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)

        self._pprint_rollouts(actions, registers, rewards, external_step_lengths)

    def _pprint_rollouts(self, action_activations, registers, rewards, external_step_lengths):
        """ Prints a single rollout, which may consists of multiple external time steps.

            Each of ``action_activations`` and ``registers`` has as many entries as there are external time steps.
            Each of THOSE entries has as many entries as there were internal time steps for that external time step.

        """
        total_internal_steps = sum(external_step_lengths)
        register_spec = self.core_network.register_spec
        row_names = ['t=', 'i=', ''] + self.core_network.action_names + ['stop']
        reg_ranges = {}
        for n, s in zip(register_spec.names, register_spec.shapes()):
            row_names.append('')
            start = len(row_names)
            for k in range(s[-1]):
                row_names.append('{}[{}]'.format(n, k))
            end = len(row_names)
            reg_ranges[n] = (start, end)
        row_names.extend(['', 'reward'])

        n_timesteps, batch_size, n_actions = action_activations.shape

        for b in range(batch_size):
            print("\nElement {} of batch ".format(b) + "-" * 40)
            values = np.zeros((total_internal_steps, len(row_names)))
            external_t, internal_t = 0, 0
            for i in range(action_activations.shape[0]):
                values[i, 0] = external_t
                values[i, 1] = internal_t
                values[i, 3:3+n_actions] = action_activations[i, b, :]
                for n, v in zip(register_spec.names, registers):
                    rr = reg_ranges[n]
                    values[i, rr[0]:rr[1]] = v[i, b, :]
                values[i, -1] = rewards[i, b]

                internal_t += 1

                if internal_t == external_step_lengths[external_t]:
                    external_t += 1
                    internal_t = 0

            values = pd.DataFrame(values.T)
            values.insert(0, 'name', row_names)
            values = values.set_index('name')
            print(tabulate(values, headers='keys', tablefmt='fancy_grid'))


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
    _action_names = None
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
    def action_names(self):
        self.assert_defined('_action_names')
        return self._action_names

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
    def __init__(self, psystem, reuse=None, sample=False):
        _, self.core_network, self.policy, _, _ = psystem

        self._state_size = (
            self.policy.state_size,
            self.core_network.register_spec.state_size())
        self._output_size = (
            self.core_network.n_actions+1,
            self.policy.state_size,
            self.core_network.register_spec.state_size())

        self._reuse = reuse
        self._sample = sample

    def __call__(self, inputs, state, scope=None):
        with tf.name_scope(scope or 'production_system_cell'):
            policy_state, registers = state

            with tf.name_scope('policy'):
                obs = self.core_network.register_spec.as_obs(registers, visible_only=1)
                action_activations, new_policy_state = self.policy(obs, policy_state, sample=self._sample)

            with tf.name_scope('core_network'):
                # Strip off the last action, which is the stopping action.
                new_registers = self.core_network(action_activations[:, :-1], registers)

            # Return state as output since ProductionSystemCell has no other meaningful output,
            # and this has benefit of making all intermediate states accessible when using
            # used with tf function ``dynamic_rnn``
            return (action_activations, policy_state, registers), (new_policy_state, new_registers)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

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
    def __init__(self, psystem, scope=None, sample=False):
        self.psystem = psystem
        _, self.core_network, self.policy, self.use_act, self.T = psystem
        self._sample = sample

        output = self._build_ps_function(psystem, scope, sample)
        (self.inputs, self.initial_state, self.ps_cell,
            self.action_activations, self.policy_states, self.registers,
            self.final_policy_states, self.final_registers) = output

    def build_feeddict(self, inp=None, registers=None, T=None):
        """ Can either specify just input (inp) or all values for all registers via ``registers``. """
        if (inp is None) == (registers is None):
            raise Exception("Must supply exactly one of ``inp``, ``registers``.")

        if inp is not None:
            batch_size = inp.shape[0]
            registers = self.core_network.register_spec.instantiate(batch_size=batch_size)
            self.core_network.register_spec.set_input(registers, inp)
        else:
            if not isinstance(registers, self.core_network.register_spec._namedtuple):
                registers = self.core_network.register_spec.from_obs(registers)

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
        return self.core_network.register_spec.get_output(self.final_registers)

    @staticmethod
    def _build_ps_function(psystem, scope=None, sample=False):
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

                ps_cell = ProductionSystemCell(psystem, sample=sample)

                batch_size = tf.shape(inputs)[1]
                initial_state = ps_cell.zero_state(batch_size, tf.float32)

                # ps_cell gives its internal state (at beginning of each time step)
                # as output so that we have access to internal states from every time
                # step, instead of just the final time step
                output = dynamic_rnn(
                    ps_cell, inputs, initial_state=initial_state,
                    parallel_iterations=1, swap_memory=False,
                    time_major=True)

                ((action_activations, policy_states, registers),
                    (final_policy_states, final_registers)) = output

        return (
            inputs, initial_state, ps_cell,
            action_activations, policy_states, registers,
            final_policy_states, final_registers)

    def get_register_values(self, *names, as_obs=True):
        return self.core_network.register_spec.get_register_values(
            self.registers, *names, as_obs=as_obs, axis=2)

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

    def __init__(self, psystem):
        self.psystem = psystem
        self.env, self.core_network, _, self.use_act, self.T = psystem
        self.sampler = None

        # Extra action is for stopping. If T is not 0, this action will be effectively ignored.
        self.n_actions = self.core_network.n_actions + 1
        self.action_space = BatchBox(low=0.0, high=1.0, shape=(None, self.n_actions))

        obs_dim = sum(
            shape[1] for shape
            in self.core_network.register_spec.shapes(visible_only=True))
        self.observation_space = BatchBox(
            low=-np.inf, high=np.inf, shape=(None, obs_dim))

        self.reward_range = self.env.reward_range

        self._seed()
        self.reset()

    @property
    def completion(self):
        return self.env.completion

    def set_mode(self, kind, batch_size):
        self.env.set_mode(kind, batch_size)

    def __str__(self):
        return "<ProductionSystemEnv, core_network={}, env={}, T={}>".format(self.core_network, self.env, self.T)

    def _step(self, action):
        assert self.action_space.contains(action), (
            "{} ({}) is not a valid action for env {}.".format(action, type(action), self))

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

    def do_rollouts(self, alg, policy, mode, n_rollouts=None):
        assert policy is self.psystem.policy

        if self.sampler is None:
            self.sampler = self.psystem.build_psystem_func(sample=True)
        self.set_mode(mode, n_rollouts)

        obs = self.reset()

        alg.start_episode()
        final_registers = None

        done = False
        while not done:
            if final_registers is None:
                fd = self.sampler.build_feeddict(inp=obs)
            else:
                start_registers = copy.deepcopy(final_registers)
                self.core_network.register_spec.set_input(start_registers, obs)
                fd = self.sampler.build_feeddict(registers=start_registers)

            sess = tf.get_default_session()
            registers, actions, final_registers = sess.run(
                [self.sampler.registers,
                 self.sampler.action_activations,
                 self.sampler.final_registers],
                feed_dict=fd)

            external_action = self.core_network.register_spec.get_output(final_registers)
            new_obs, reward, done, info = self.env.step(external_action)

            # record the trajectory
            obs = self.core_network.register_spec.as_obs(registers, visible_only=True)
            for t, (o, a) in enumerate(zip(obs, actions)):
                r = np.zeros(reward.shape) if t < obs.shape[0]-1 else reward
                alg.remember(o, a, r)

            obs = new_obs

        alg.end_episode()


def build_diff_updater(psystem):
    config = default_config()
    ps_func = psystem.build_psystem_func()
    updater = DifferentiableUpdater(
        psystem.env, ps_func, config.optimizer_class,
        config.lr_schedule, config.noise_schedule, config.max_grad_norm)

    return updater


def build_reinforce_updater(psystem):
    config = default_config()
    ps_env = psystem.build_psystem_env()
    updater = REINFORCE(
        ps_env, psystem.policy, config.optimizer_class,
        config.lr_schedule, config.noise_schedule, config.max_grad_norm,
        config.gamma, config.l2_norm_param)

    return updater


class ProductionSystemCurriculum(Curriculum):
    def __init__(self, base_kwargs, curriculum, build_env, build_core_network, build_policy):
        super(ProductionSystemCurriculum, self).__init__(base_kwargs, curriculum)
        self.build_env = build_env
        self.build_core_network = build_core_network
        self.build_policy = build_policy

    def __call__(self):
        if self.stage == self.prev_stage:
            raise Exception("Need to call member function ``end_stage`` before getting next stage.")

        if self.stage == len(self.curriculum):
            raise StopIteration()

        kw = self.curriculum[self.stage]
        kw = kw.copy()
        kw.update(self.base_kwargs)

        config = default_config()
        T = kw.pop('T', config.T)

        env = self.build_env(**kw)
        core_network = self.build_core_network(env)
        exploration = build_decaying_value(config.exploration_schedule, 'exploration')
        policy = self.policy = self.build_policy(core_network, exploration)

        if self.stage != 0:
            g = tf.get_default_graph()
            policy_variables = g.get_collection('trainable_variables', scope=self.policy.scope.name)
            saver = tf.train.Saver(policy_variables)
            saver.restore(tf.get_default_session(), os.path.join(default_config().path, 'policy.chk'))

        psystem = ProductionSystem(env, core_network, policy, False, T)
        self.current_psystem = psystem

        if config.use_rl:
            updater = build_reinforce_updater(psystem)
        else:
            updater = build_diff_updater(psystem)
        self.prev_stage = self.stage
        return updater

    def end_stage(self):
        self.current_psystem.visualize('train', 5, default_config().use_rl)

        # Occurs inside the same default graph, session and config as the previous call to __call__.
        g = tf.get_default_graph()
        policy_variables = g.get_collection('trainable_variables', scope=self.policy.scope.name)
        saver = tf.train.Saver(policy_variables)
        saver.save(tf.get_default_session(), os.path.join(default_config().path, 'policy.chk'))
        self.stage += 1
