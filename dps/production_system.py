import numpy as np
from collections import namedtuple
import os
import copy
import pandas as pd
from tabulate import tabulate
from pprint import pformat
import time
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from dps.environment import BatchBox
from dps.updater import DifferentiableUpdater
from dps.reinforce import REINFORCE
from dps.qlearning import QLearning
from dps.utils import default_config, build_decaying_value, uninitialized_variables_initializer, gen_seed
from dps.train import Curriculum, training_loop
from dps.policy import Policy


params = 'env core_network policy use_act T'


class ProductionSystem(namedtuple('ProductionSystem', params.split())):
    """ A production system."""

    def __init__(self, env, core_network, policy, use_act, T):
        super(ProductionSystem, self).__init__()
        self.rb = core_network.register_bank

        self.n_actions = self.core_network.n_actions
        self.action_space = BatchBox(low=0.0, high=1.0, shape=(None, self.n_actions))

        self.obs_dim = self.rb.visible_width
        self.observation_space = BatchBox(low=-np.inf, high=np.inf, shape=(None, self.obs_dim))

        self.reward_range = env.reward_range

        self.sampler = None

    def build_psystem_func(self, sample=False):
        return ProductionSystemFunction(self, sample=sample)

    @property
    def completion(self):
        return self.env.completion

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

        self.env.set_mode(mode, n_rollouts)
        external_obs = self.env.reset()

        start_registers = self.rb.new_array(n_rollouts)
        final_registers = None

        registers, actions, rewards = [], [], []
        external = [external_obs]

        done = False
        while not done:
            fd = ps_func.build_feeddict(inp=external_obs, registers=start_registers)

            sess = tf.get_default_session()

            try:
                sess.run(uninitialized_variables_initializer())
                sess.run(tf.assert_variables_initialized())
            except TypeError:
                pass

            reg, a, final_registers = sess.run(
                [ps_func.registers,
                 ps_func.action_activations,
                 ps_func.final_registers],
                feed_dict=fd)

            external_action = self.rb.get_output(final_registers)
            external_obs, reward, done, info = self.env.step(external_action)

            registers.append(reg)
            actions.append(a)
            external.append(external_obs)

            _reward = np.zeros(a.shape[:-1] + (1,))
            _reward[-1, :, :] = reward
            rewards.append(_reward)
            start_registers = final_registers.copy()

        external_step_lengths = [a.shape[0] for a in actions]

        registers = np.concatenate(list(registers) + [np.expand_dims(final_registers, 0)], axis=0)

        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)

        self._pprint_rollouts(actions, registers, rewards, external, external_step_lengths)

        if render_rollouts is not None:
            render_rollouts(self, actions, registers, rewards, external, external_step_lengths)

    def _pprint_rollouts(self, action_activations, registers, rewards, external, external_step_lengths):
        """ Prints a single rollout, which may consists of multiple external time steps.

            Each of ``action_activations`` and ``registers`` has as many entries as there are external time steps.
            Each of THOSE entries has as many entries as there were internal time steps for that external time step.

        """
        total_internal_steps = sum(external_step_lengths)

        row_names = ['t=', 'i=', ''] + self.core_network.action_names

        omit = set(self.rb.no_display)
        reg_ranges = {}

        for n, s in zip(self.rb.names, self.rb.shapes):
            if n in omit:
                continue

            row_names.append('')
            start = len(row_names)
            for k in range(s):
                row_names.append('{}[{}]'.format(n, k))
            end = len(row_names)
            reg_ranges[n] = (start, end)
        row_names.extend(['', 'reward'])

        n_timesteps, batch_size, n_actions = action_activations.shape

        registers = self.rb.as_tuple(registers)

        for b in range(batch_size):
            print("\nElement {} of batch ".format(b) + "-" * 40)

            if external[0].shape[-1] < 40:
                print("External observations: ")
                for i, e in enumerate(external):
                    print("{}: {}".format(i, e[b, :]))

            values = np.zeros((total_internal_steps+1, len(row_names)))
            external_t, internal_t = 0, 0

            for i in range(total_internal_steps):
                values[i, 0] = external_t
                values[i, 1] = internal_t
                values[i, 3:3+n_actions] = action_activations[i, b, :]
                for n, v in zip(self.rb.names, registers):
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
            for n, v in zip(self.rb.names, registers):
                if n in omit:
                    continue
                rr = reg_ranges[n]
                values[-1, rr[0]:rr[1]] = v[-1, b, :]

            values = pd.DataFrame(values.T)
            values.insert(0, 'name', row_names)
            values = values.set_index('name')
            print(tabulate(values, headers='keys', tablefmt='fancy_grid'))

    def do_rollouts(self, alg, policy, mode, n_rollouts=None):
        # For now...
        assert policy is self.policy

        if self.sampler is None:
            self.sampler = self.build_psystem_func(sample=True)

        self.env.set_mode(mode, n_rollouts)
        external_obs = self.env.reset()
        batch_size = external_obs.shape[0]

        alg.start_episode()

        start_registers = self.rb.new_array(batch_size)
        final_registers = None

        done = False
        while not done:
            fd = self.sampler.build_feeddict(inp=external_obs, registers=start_registers)
            fd[alg.is_training] = mode == 'train'

            sess = tf.get_default_session()
            registers, actions, final_registers = sess.run(
                [self.sampler.registers,
                 self.sampler.action_activations,
                 self.sampler.final_registers],
                feed_dict=fd)

            external_action = self.rb.get_output(final_registers)
            external_obs, reward, done, info = self.env.step(external_action)

            # record the trajectory
            for t, (o, a) in enumerate(zip(self.rb.visible(registers), actions)):
                r = np.zeros(reward.shape) if t < actions.shape[0]-1 else reward
                alg.remember(o, a, r)

            start_registers = final_registers.copy()

        alg.end_episode()


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
    register_bank = None

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
        return self.register_bank.visible_width

    @property
    def input_dim(self):
        """ Return a boolean. """
        raise NotImplementedError()

    @property
    def make_input_available(self):
        """ A boolean, whether to make pass the input as the third argument when calling self.__call__. """
        raise NotImplementedError()

    def init(self, registers, inp):
        """ Note: this is done every time the function is applied, even on subsequent calls within the same episode. """
        raise NotImplementedError()

    def __call__(self, action_activations, registers, inp=None):
        """ Accepts a tensor representing action activations, and an instance
        of ``self.register_bank`` storing register contents, and
        return a new instance of ``self.register_bank`` storing the
        register contents for the next time step.


        ``__call__`` is required to accept an ``inp`` argument iff
        ``self.make_input_available`` is True.

        """
        raise NotImplementedError()


class ProductionSystemCell(RNNCell):
    def __init__(self, psystem, sample):
        _, self.core_network, self.policy, _, _ = psystem
        self.rb = psystem.rb

        self._state_size = (self.policy.state_size, self.rb.width)

        self._output_size = (
            self.core_network.n_actions,
            self.policy.state_size,
            self.rb.width)

        self._sample = sample
        self._inp_tensor = None

    def set_input(self, inp_tensor):
        """ Provide a batched tensor that is the same every time step, and which can be accessed
            by the core network.

        """
        self._inp_tensor = inp_tensor

    def __call__(self, inputs, state, scope=None):
        with tf.name_scope(scope or 'production_system_cell'):
            policy_state, registers = state

            with tf.name_scope('policy'):
                obs = self.rb.visible(registers)
                action_activations, new_policy_state = self.policy(
                    obs, policy_state, sample=self._sample)

            if self._inp_tensor is not None:
                with tf.name_scope('core_network'):
                    new_registers = self.core_network(action_activations, registers, self._inp_tensor)
            else:
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
        ph = tf.placeholder(tf.float32, (None, self.rb.width), name='ps_cell_register')
        return policy_state, ph


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
        self.rb = psystem.rb
        self._sample = sample

        output = self._build_ps_function(psystem, scope, sample)
        (self.inputs, self.inp_ph, self.register_ph, self.ps_cell,
         self.action_activations, self.policy_states, self.registers,
         self.final_policy_states, self.final_registers) = output

    def build_feeddict(self, inp, registers, T=None):
        batch_size = inp.shape[0]
        fd = {self.register_ph: registers,
              self.inp_ph: inp}
        T = T or self.T
        if not self.use_act:
            fd[self.inputs] = np.zeros((T, batch_size, 1))
        return fd

    def get_output(self):
        return self.final_registers.get_output()

    @staticmethod
    def _build_ps_function(psystem, scope, sample):
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
                inputs = tf.placeholder(tf.float32, shape=(None, None, 1), name="timestep")
                ps_cell = ProductionSystemCell(psystem, sample=sample)

                batch_size = tf.shape(inputs)[1]
                policy_state, register_ph = ps_cell.zero_state(batch_size, tf.float32)
                inp_ph = tf.placeholder(tf.float32, (None, psystem.core_network.input_dim), name='input')

                if psystem.core_network.make_input_available:
                    ps_cell.set_input(inp_ph)

                initial_registers = psystem.core_network.init(register_ph, inp_ph)

                output = dynamic_rnn(
                    ps_cell, inputs, initial_state=(policy_state, initial_registers),
                    parallel_iterations=1, swap_memory=False,
                    time_major=True)

                ((action_activations, policy_states, registers),
                    (final_policy_states, final_registers)) = output

        return (
            inputs, inp_ph, register_ph, ps_cell,
            action_activations, policy_states, registers,
            final_policy_states, final_registers)

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


class ProductionSystemCurriculum(Curriculum):
    def __init__(self, config, build_env, build_core_network, build_policy, verbose=False):
        super(ProductionSystemCurriculum, self).__init__(config)
        self.build_env = build_env
        self.build_core_network = build_core_network
        self.build_policy = build_policy

    def __call__(self):
        super(ProductionSystemCurriculum, self).__call__()
        if self.stage == self.prev_stage:
            raise Exception("Need to call member function ``end_stage`` before getting next stage.")

        if self.stage == len(self.config.curriculum):
            raise StopIteration()

        config = copy.copy(self.config)
        config.update(self.config.curriculum[self.stage])

        print("\nStarting stage {} of the curriculum. "
              "New config values for this stage are: \n{}\n".format(
                  self.stage, pformat(self.config.curriculum[self.stage])))

        with config.as_default():
            updater = self._build_updater()
        self.prev_stage = self.stage
        return config, updater

    def _build_updater(self):
        config = default_config()
        env = self.build_env()
        core_network = self.build_core_network(env)

        is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

        exploration = build_decaying_value(config.schedule('exploration'), 'exploration')
        if config.test_time_explore is not None:
            testing_exploration = tf.constant(config.test_time_explore, tf.float32, name='testing_exploration')
            exploration = tf.cond(is_training, lambda: exploration, lambda: testing_exploration)
        policy = self.policy = self.build_policy(core_network, exploration)
        policy.capture_scope()

        target_policy = policy.deepcopy("target_policy")
        target_policy.capture_scope()

        if self.stage != 0 and config.preserve_policy:
            policy.maybe_build_act()

            g = tf.get_default_graph()

            policy_variables = g.get_collection('trainable_variables', scope=policy.scope.name)
            saver = tf.train.Saver(policy_variables)
            saver.restore(tf.get_default_session(), os.path.join(default_config().path, 'policy.chk'))

        psystem = ProductionSystem(env, core_network, policy, False, config.T)
        self.current_psystem = psystem

        ps_func = psystem.build_psystem_func()

        if self.config.updater_class is DifferentiableUpdater:
            updater = DifferentiableUpdater(
                psystem.env, ps_func, config.optimizer_class,
                config.schedule('lr'), config.schedule('noise'), config.max_grad_norm)
        elif self.config.updater_class is REINFORCE:
            updater = REINFORCE(
                psystem, psystem.policy, config.optimizer_class,
                config.schedule('lr'), config.schedule('noise'), config.max_grad_norm,
                config.gamma, config.l2_norm_penalty, config.schedule('entropy'))
        elif self.config.updater_class is QLearning:
            updater = QLearning(
                psystem, psystem.policy, target_policy, config.double,
                config.replay_max_size, config.replay_threshold, config.replay_proportion,
                config.target_update_rate, config.recurrent, config.optimizer_class,
                config.schedule('lr'), config.schedule('noise'), config.max_grad_norm,
                config.gamma, config.l2_norm_penalty)
        else:
            raise NotImplementedError()
        updater.is_training = is_training
        return updater

    def end_stage(self):
        super(ProductionSystemCurriculum, self).end_stage()
        sample = self.config.updater_class in [REINFORCE, QLearning]
        if self.config.verbose:
            self.current_psystem.visualize('train', 5, sample)

        # Occurs inside the same default graph, session and config as the previous call to __call__.
        g = tf.get_default_graph()
        policy_variables = g.get_collection('trainable_variables', scope=self.policy.scope.name)
        saver = tf.train.Saver(policy_variables)
        saver.save(tf.get_default_session(), os.path.join(default_config().path, 'policy.chk'))
        self.stage += 1


class ProductionSystemTrainer(object):
    def __init__(self):
        pass

    def build_policy(self, cn, exploration):
        config = default_config()
        return Policy(
            config.controller_func(cn.n_actions), config.action_selection, exploration,
            cn.n_actions, cn.obs_dim, name="{}_policy".format(cn.__class__.__name__))

    def build_env(self):
        raise NotImplementedError("Abstract method.")

    def build_core_network(self, env):
        raise NotImplementedError("Abstract method.")

    def train(self, config, seed=-1):
        config.seed = config.seed if seed < 0 else seed
        np.random.seed(config.seed)

        curriculum = ProductionSystemCurriculum(
            config, self.build_env, self.build_core_network, self.build_policy)

        exp_name = "selection={}_updater={}_seed={}".format(
            config.action_selection.__class__.__name__, config.updater_class.__name__, config.seed)
        return training_loop(curriculum, config, exp_name=exp_name)


def build_and_visualize(sample=False):
    config = default_config()
    with ExitStack() as stack:

        graph = tf.Graph()

        if not default_config().use_gpu:
            stack.enter_context(graph.device("/cpu:0"))

        sess = tf.Session(graph=graph)

        stack.enter_context(graph.as_default())
        stack.enter_context(sess)
        stack.enter_context(sess.as_default())

        tf_seed = gen_seed()
        tf.set_random_seed(tf_seed)

        env = config.trainer.build_env()
        cn = config.trainer.build_core_network(env)

        exploration = tf.constant(0.0)
        policy = Policy(
            config.controller_func(cn.n_actions), config.action_selection, exploration,
            cn.n_actions, cn.obs_dim)
        psystem = ProductionSystem(env, cn, policy, False, config.T)

        render_rollouts = getattr(config, 'render_rollouts', None)

        start_time = time.time()
        psystem.visualize('train', config.batch_size, sample, render_rollouts)
        duration = time.time() - start_time

        print("Took {} seconds.".format(duration))
