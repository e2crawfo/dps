import numpy as np
from collections import namedtuple
import os
import copy
import pandas as pd
from tabulate import tabulate
from pprint import pprint

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from gym.utils import seeding

from dps.environment import BatchBox, Env
from dps.updater import DifferentiableUpdater
from dps.reinforce import REINFORCE
from dps.qlearning import QLearning
from dps.utils import default_config, build_decaying_value
from dps.train import Curriculum, training_loop
from dps.policy import Policy


class ProductionSystemTrainer(object):
    def __init__(self):
        pass

    def build_policy(self, cn, exploration):
        config = default_config()
        return Policy(
            config.controller_func(cn.n_actions), config.action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="{}_policy".format(cn.__class__.__name__))

    def build_env(self, **kwargs):
        raise NotImplementedError("Abstract method.")

    def build_core_network(self, env):
        raise NotImplementedError("Abstract method.")

    def train(self, config, seed=-1):
        config.seed = config.seed if seed < 0 else seed
        np.random.seed(config.seed)

        base_kwargs = dict(n_train=config.n_train, n_val=config.n_val, n_test=config.n_test)
        if hasattr(config, 'base_kwargs'):
            base_kwargs.update(config.base_kwargs)

        curriculum = ProductionSystemCurriculum(
            base_kwargs, config.curriculum, config.updater_class,
            self.build_env, self.build_core_network, self.build_policy, verbose=config.verbose)

        exp_name = "selection={}_updater={}".format(
            config.action_selection.__class__.__name__, config.updater_class.__name__)
        return training_loop(curriculum, config, exp_name=exp_name)


params = 'env core_network policy use_act T'


class ProductionSystem(namedtuple('ProductionSystem', params.split())):
    """ A production system."""

    def build_psystem_env(self):
        return ProductionSystemEnv(self)

    def build_psystem_func(self, sample=False):
        return ProductionSystemFunction(self, sample=sample)

    def visualize(self, mode, n_rollouts, sample, render_rollouts=None):
        """ Visualize rollouts from a production system.

        Parameters
        ----------
        mode: str
            One of 'train', 'val', 'test', specifies mode for environment.
        n_rollouts: int
            Number of rollouts.
        sample: bool
            Whether to perform additional sampling step when rolling out.
        render_rollouts: fn (optional)
            Accepts actions, registers, rewards, and external_step_lengths and
            performs visualization.

        """
        ps_func = self.build_psystem_func(sample=sample)
        env = self.env
        env.set_mode(mode, n_rollouts)

        obs = env.reset()

        final_registers = None

        registers, actions, rewards = [], [], []
        t = 0

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

            print("info: ")
            info.update(t=t)
            pprint(info)

        external_step_lengths = [a.shape[0] for a in actions]

        rspec = self.core_network.register_spec
        registers = rspec.concatenate(registers + [rspec.expand_dims(final, 0)], axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)

        self._pprint_rollouts(actions, registers, rewards, external_step_lengths)

        if render_rollouts is not None:
            render_rollouts(self, actions, registers, rewards, external_step_lengths)

    def _pprint_rollouts(self, action_activations, registers, rewards, external_step_lengths):
        """ Prints a single rollout, which may consists of multiple external time steps.

            Each of ``action_activations`` and ``registers`` has as many entries as there are external time steps.
            Each of THOSE entries has as many entries as there were internal time steps for that external time step.

        """
        total_internal_steps = sum(external_step_lengths)

        row_names = ['t=', 'i=', ''] + self.core_network.action_names

        register_spec = self.core_network.register_spec
        omit = set(getattr(self.core_network.register_spec, 'omit', []))
        reg_ranges = {}

        for n, s in zip(register_spec.names, register_spec.shapes()):
            if n in omit:
                continue

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
            values = np.zeros((total_internal_steps+1, len(row_names)))
            external_t, internal_t = 0, 0

            for i in range(total_internal_steps):
                values[i, 0] = external_t
                values[i, 1] = internal_t
                values[i, 3:3+n_actions] = action_activations[i, b, :]
                for n, v in zip(register_spec.names, registers):
                    if n in omit:
                        continue
                    rr = reg_ranges[n]
                    values[i, rr[0]:rr[1]] = v[i, b, :]
                values[i, -1] = rewards[i, b]

                internal_t += 1

                if internal_t == external_step_lengths[external_t]:
                    external_t += 1
                    internal_t = 0

            # Print final values for the registers
            for n, v in zip(register_spec.names, registers):
                if n in omit:
                    continue
                rr = reg_ranges[n]
                values[-1, rr[0]:rr[1]] = v[-1, b, :]

            values = pd.DataFrame(values.T)
            values.insert(0, 'name', row_names)
            values = values.set_index('name')
            print(tabulate(values, headers='keys', tablefmt='fancy_grid'))


class CoreNetworkMeta(type):
    def __init__(cls, *args, **kwargs):
        super(CoreNetworkMeta, cls).__init__(*args, **kwargs)
        if hasattr(cls.action_names, '__len__'):
            cls.n_actions = len(cls.action_names)


class CoreNetwork(object, metaclass=CoreNetworkMeta):
    """ A core network inside a production system.

    A specification of the functionality of the core network;
    state must be maintained externally.

    """
    action_names = None
    register_spec = None

    def __init__(self):
        self._graph = None
        self._action_activations_ph = None
        self._register_ph = None

        self.assert_defined('action_names')

    def assert_defined(self, attr):
        assert getattr(self, attr) is not None, (
            "Subclasses of CoreNetwork must "
            "specify a value for attr {}.".format(attr))

    @property
    def obs_dim(self):
        return sum(shape[1] for shape in self.register_spec.shapes(visible_only=True))

    def __call__(self, action_activations, egisters):
        """ Accepts a tensor representing action activations and an object
        created by calling ``instantiate`` on ``register_spec``
        (which stores tensors representing register values) and outputs tensors
        representing the new values of the registers after running the
        core network for one step.

        """
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
    def __init__(self, psystem, sample):
        _, self.core_network, self.policy, _, _ = psystem

        self._state_size = (
            self.policy.state_size,
            self.core_network.register_spec.state_size())

        self._output_size = (
            self.core_network.n_actions,
            self.policy.state_size,
            self.core_network.register_spec.state_size())

        self._sample = sample

    def __call__(self, inputs, state, scope=None):
        with tf.name_scope(scope or 'production_system_cell'):
            policy_state, registers = state

            with tf.name_scope('policy'):
                obs = self.core_network.register_spec.as_obs(registers, visible_only=1)
                action_activations, new_policy_state = self.policy(obs, policy_state, sample=self._sample)

            with tf.name_scope('core_network'):
                new_registers = self.core_network(action_activations, registers)

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
        policy_state = self.policy.zero_state(batch_size, dtype)
        registers = self.core_network.register_spec.build_placeholders(dtype)
        return (policy_state, registers)


class ProductionSystemFunction(object):
    """ Wrapper around a tensorflow function that implements a production system.

    Parameters
    ----------
    psystem: ProductionSystem
        The production system that we want to turn into a tensorflow function.

    """
    def __init__(self, psystem, scope=None, sample=False, initialize=True):
        self.psystem = psystem
        _, self.core_network, self.policy, self.use_act, self.T = psystem
        self._sample = sample
        self._initialize = initialize

        output = self._build_ps_function(psystem, scope, sample, initialize)
        (self.inputs, self.register_ph, self.ps_cell,
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

        for ph, value in zip(self.register_ph, registers):
            fd[ph] = value

        return fd

    def get_output(self):
        return self.core_network.register_spec.get_output(self.final_registers)

    @staticmethod
    def _build_ps_function(psystem, scope, sample, initialize):
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
                policy_state, register_ph = ps_cell.zero_state(batch_size, tf.float32)

                if initialize:
                    no_op = tf.concat(
                        (tf.fill((batch_size, psystem.core_network.n_actions-1), 0.0),
                         tf.fill((batch_size, 1), 1.0)),
                        axis=1)
                    initial_registers = psystem.core_network(no_op, register_ph)
                else:
                    initial_registers = register_ph

                # ps_cell gives its internal state (at beginning of each time step)
                # as output so that we have access to internal states from every time
                # step, instead of just the final time step
                output = dynamic_rnn(
                    ps_cell, inputs, initial_state=(policy_state, initial_registers),
                    parallel_iterations=1, swap_memory=False,
                    time_major=True)

                ((action_activations, policy_states, registers),
                    (final_policy_states, final_registers)) = output

        return (
            inputs, register_ph, ps_cell,
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
        self.n_actions = self.core_network.n_actions
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

        obs = self.env.reset()

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

            fd[alg.is_training] = mode == 'train'

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


class ProductionSystemCurriculum(Curriculum):
    def __init__(self, base_kwargs, curriculum, updater_class, build_env, build_core_network, build_policy, verbose=False):
        super(ProductionSystemCurriculum, self).__init__(base_kwargs, curriculum)
        self.updater_class = updater_class
        self.build_env = build_env
        self.build_core_network = build_core_network
        self.build_policy = build_policy
        self.verbose = verbose

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

        is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

        exploration = build_decaying_value(config.exploration_schedule, 'exploration')
        if config.test_time_explore is not None:
            testing_exploration = tf.constant(config.test_time_explore, tf.float32, name='testing_exploration')
            exploration = tf.cond(is_training, lambda: exploration, lambda: testing_exploration)
        policy = self.policy = self.build_policy(core_network, exploration)
        policy.capture_scope()

        target_policy = policy.deepcopy("target_policy")
        target_policy.capture_scope()

        if self.stage != 0:
            # So that there exist variables to load into
            policy.maybe_build_act()

            g = tf.get_default_graph()

            policy_variables = g.get_collection('trainable_variables', scope=policy.scope.name)
            saver = tf.train.Saver(policy_variables)
            saver.restore(tf.get_default_session(), os.path.join(default_config().path, 'policy.chk'))

        psystem = ProductionSystem(env, core_network, policy, False, T)
        self.current_psystem = psystem

        config = default_config()
        ps_func = psystem.build_psystem_func()
        ps_env = psystem.build_psystem_env()

        if self.updater_class is DifferentiableUpdater:
            updater = DifferentiableUpdater(
                psystem.env, ps_func, config.optimizer_class,
                config.lr_schedule, config.noise_schedule, config.max_grad_norm)
        elif self.updater_class is REINFORCE:
            updater = REINFORCE(
                ps_env, psystem.policy, config.optimizer_class,
                config.lr_schedule, config.noise_schedule, config.max_grad_norm,
                config.gamma, config.l2_norm_param)
        elif self.updater_class is QLearning:
            updater = QLearning(
                ps_env, psystem.policy, target_policy, config.double, config.replay_max_size, config.target_update_rate,
                config.recurrent, config.optimizer_class, config.lr_schedule, config.noise_schedule, config.max_grad_norm,
                config.gamma, config.l2_norm_param)
        else:
            raise NotImplementedError()
        updater.is_training = is_training

        self.prev_stage = self.stage
        return updater

    def end_stage(self):
        sample = self.updater_class in [REINFORCE, QLearning]
        if self.verbose:
            self.current_psystem.visualize('train', 5, sample)

        # Occurs inside the same default graph, session and config as the previous call to __call__.
        g = tf.get_default_graph()
        policy_variables = g.get_collection('trainable_variables', scope=self.policy.scope.name)
        saver = tf.train.Saver(policy_variables)
        saver.save(tf.get_default_session(), os.path.join(default_config().path, 'policy.chk'))
        self.stage += 1
