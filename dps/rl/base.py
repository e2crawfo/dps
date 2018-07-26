import numpy as np
import tensorflow as tf
import abc
from contextlib import ExitStack
import time

from dps import cfg
from dps.utils import Param, Parameterized, shift_fill
from dps.utils.tf import masked_mean, tf_discount_matrix, build_scheduled_value
from dps.updater import Updater


def rl_render_hook(updater):
    if hasattr(updater, 'learners'):
        render_rollouts = getattr(cfg, 'render_rollouts', None)
        for learner in updater.learners:
            with learner:
                updater.env.visualize(
                    policy=learner.pi,
                    n_rollouts=cfg.render_n_rollouts, T=cfg.T, mode='val',
                    render_rollouts=render_rollouts)
    else:
        print("Not rendering.")


class RLObject(object, metaclass=abc.ABCMeta):
    def __new__(cls, *args, **kwargs):
        new = super(RLObject, cls).__new__(cls)

        current_context = get_active_context()
        if current_context is not None:
            current_context.add_rl_object(new)
        return new

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def build_core_signals(self, context):
        # "core" signals are those which are generated before all other signals.
        # We can think of them as "leaves" in a tree where the nodes are signals,
        # edges are ops. They must not depend on signals built by other RLObject
        # instances. Not all leaves have to be created in "build_core_signals",
        # but doing so can can ensure the signal is not built in a weird context
        # (e.g. tf.while or tf.cond) which can cause problems.
        pass

    def generate_signal(self, signal_key, context):
        pass

    def pre_update(self, feed_dict, context):
        pass

    def post_update(self, feed_dict, context):
        pass

    def pre_eval(self, feed_dict, context):
        pass

    def post_eval(self, feed_dict, context):
        pass


class ObjectiveFunctionTerm(RLObject):
    def __init__(self, *, use_weights=False, weight=1.0, name=None):
        self.use_weights = use_weights
        self.weight_schedule = weight
        super(ObjectiveFunctionTerm, self).__init__(name)

    def build_core_signals(self, context):
        self.weight = build_scheduled_value(self.weight_schedule, "{}-weight".format(self.name))

    @abc.abstractmethod
    def build_graph(self, context):
        pass


def get_active_context():
    if RLContext.active_context is None:
        raise Exception("No context is currently active.")
    return RLContext.active_context


class RLContext(Parameterized):
    """
        truncated_rollouts: bool
            If True, then our rollouts are sub-trajectories of longer trajectories. The consequence
            is that in general we cannot say that the value after final state-action-reward triple
            is 0. This implies that T-step backups (assuming the rollout is given by
            (x_0, a_0, r_0, ..., x_(T-1), a_(T-1), r_(T-1)) will not be valid, since we don't have
            a proper value estimate past the end of the trajectory. When not using truncated trajectories,
            we assume that the value past the end of the trajectory is 0, and hence T-step backups are OK.

    """
    active_context = None

    replay_updates_per_sample = Param(1)
    on_policy_updates = Param(True)

    def __init__(self, gamma, truncated_rollouts=False, name=""):
        self.mu = None
        self.gamma = gamma
        self.name = name
        self.terms = []
        self.plugins = []
        self._signals = {}
        self.truncated_rollouts = truncated_rollouts
        self.optimizer = None
        self.train_summaries = []
        self.summaries = []
        self.recorded_values = []
        self.update_batch_size = None
        self.replay_buffer = None
        self.objective_fn_terms = []
        self.agents = []
        self.rl_objects = []

    def __enter__(self):
        if RLContext.active_context is not None:
            raise Exception("May not have multiple instances of RLContext active at once.")
        RLContext.active_context = self
        return self

    def __exit__(self, type_, value, tb):
        RLContext.active_context = None

    def trainable_variables(self, for_opt):
        return [v for agent in self.agents for v in agent.trainable_variables(for_opt=for_opt)]

    def add_rl_object(self, obj):
        if isinstance(obj, ObjectiveFunctionTerm):
            self.objective_fn_terms.append(obj)
        from dps.rl.agent import Agent
        if isinstance(obj, Agent):
            self.agents.append(obj)
        assert isinstance(obj, RLObject)
        self.rl_objects.append(obj)

    def set_behaviour_policy(self, mu):
        self.mu = mu

    def set_validation_policy(self, pi):
        self.pi = pi

    def set_optimizer(self, opt):
        self.optimizer = opt

    def set_replay_buffer(self, update_batch_size, replay_buffer):
        self.update_batch_size = update_batch_size
        self.replay_buffer = replay_buffer

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_train_summary(self, summary):
        """ Add a summary that should only be evaluated during training. """
        self.train_summaries.append(summary)

    def add_summary(self, summary):
        self.summaries.append(summary)

    def add_recorded_value(self, name, tensor):
        self.recorded_values.append((name, tensor))

    def set_mode(self, mode):
        for obj in self.rl_objects:
            if hasattr(obj, 'set_mode'):
                obj.set_mode(mode)

    def build_graph(self, env):
        self.env = env
        self.obs_shape = env.obs_shape
        self.action_shape = env.action_shape

        with ExitStack() as stack:
            if self.name:
                stack.enter_context(tf.name_scope(self.name))

            stack.enter_context(self)

            self.build_core_signals()

            objective = None
            for term in self.objective_fn_terms:
                if objective is None:
                    objective = term.weight * term.build_graph(self)
                else:
                    objective += term.weight * term.build_graph(self)
            self.objective = objective
            self.add_recorded_value("rl_objective", self.objective)

            self.optimizer.build_update(self)

            self.recorded_tensors = {}
            for name in getattr(env, 'recorded_names', []):
                ph = tf.placeholder(tf.float32, (), name=name+"_ph")
                self.recorded_tensors[name] = ph
                self.add_summary(tf.summary.scalar("env_" + name, ph))

            self.train_summary_op = tf.summary.merge(self.train_summaries + self.summaries)
            self.summary_op = tf.summary.merge(self.summaries)

    def build_core_signals(self):
        self._signals['mask'] = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_mask")
        self._signals['obs'] = tf.placeholder(
            tf.float32, shape=(cfg.T, None) + self.obs_shape, name="_obs")

        if hasattr(self.env, 'rb'):
            self._signals['hidden'] = tf.placeholder(
                tf.float32, shape=(cfg.T, None, self.env.rb.hidden_width,), name="_hidden")

        self._signals['actions'] = tf.placeholder(
            tf.float32, shape=(cfg.T, None) + self.action_shape, name="_actions")
        self._signals['gamma'] = tf.constant(self.gamma)
        self._signals['batch_size'] = tf.shape(self._signals['obs'])[1]
        self._signals['batch_size_float'] = tf.cast(self._signals['batch_size'], tf.float32)

        self._signals['rewards'] = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_rewards")
        self._signals['returns'] = tf.cumsum(
            self._signals['rewards'], axis=0, reverse=True, name="_returns")
        self._signals['reward_per_ep'] = tf.reduce_mean(
            tf.reduce_sum(self._signals['rewards'], axis=0), name="_reward_per_ep")

        self.add_summary(tf.summary.scalar("reward_per_ep", self._signals['reward_per_ep']))
        self.add_recorded_value("reward_per_ep", self._signals['reward_per_ep'])

        self._signals['train_n_episodes'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._signals['train_cumulative_reward'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._signals['train_reward_per_ep_avg'] = (
            self._signals['train_cumulative_reward'] / self._signals['train_n_episodes'])

        self._signals['off_policy_n_episodes'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._signals['off_policy_cumulative_reward'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._signals['off_policy_reward_per_ep_avg'] = (
            self._signals['off_policy_cumulative_reward'] / self._signals['off_policy_n_episodes'])

        self._signals['val_n_episodes'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._signals['val_cumulative_reward'] = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._signals['val_reward_per_ep_avg'] = (
            self._signals['val_cumulative_reward'] / self._signals['val_n_episodes'])

        mode = self._signals['mode'] = tf.placeholder(tf.string, ())

        self.update_stats_op = tf.case(
            [
                (tf.equal(mode, 'train'), lambda: tf.group(
                    tf.assign_add(self._signals['train_n_episodes'], self._signals['batch_size_float']),
                    tf.assign_add(self._signals['train_cumulative_reward'], tf.reduce_sum(self._signals['rewards']))
                )),
                (tf.equal(mode, 'off_policy'), lambda: tf.group(
                    tf.assign_add(self._signals['off_policy_n_episodes'], self._signals['batch_size_float']),
                    tf.assign_add(self._signals['off_policy_cumulative_reward'], tf.reduce_sum(self._signals['rewards']))
                )),
                (tf.equal(mode, 'val'), lambda: tf.group(
                    tf.assign_add(self._signals['val_n_episodes'], self._signals['batch_size_float']),
                    tf.assign_add(self._signals['val_cumulative_reward'], tf.reduce_sum(self._signals['rewards']))
                )),
            ],
            default=lambda: tf.group(tf.ones((), dtype=tf.float32), tf.ones((), dtype=tf.float32)),
            exclusive=True
        )

        self._signals['reward_per_ep_avg'] = tf.case(
            [
                (tf.equal(mode, 'train'), lambda: self._signals['train_reward_per_ep_avg']),
                (tf.equal(mode, 'off_policy'), lambda: self._signals['off_policy_reward_per_ep_avg']),
                (tf.equal(mode, 'val'), lambda: self._signals['val_reward_per_ep_avg']),
            ],
            default=lambda: tf.zeros_like(self._signals['train_reward_per_ep_avg']),
            exclusive=True
        )

        self.add_summary(tf.summary.scalar("reward_per_ep_avg", self._signals['reward_per_ep_avg']))
        self.add_recorded_value("reward_per_ep_avg", self._signals['reward_per_ep_avg'])

        self._signals['weights'] = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_weights")

        T = tf.shape(self._signals['mask'])[0]
        discount_matrix = tf_discount_matrix(self.gamma, T)
        discounted_returns = tf.tensordot(
            discount_matrix, self._signals['rewards'], axes=1, name="_discounted_returns")
        self._signals['discounted_returns'] = discounted_returns

        mask = self._signals['mask']
        mean_returns = masked_mean(discounted_returns, mask, axis=1, keepdims=True)
        mean_returns += tf.zeros_like(discounted_returns)
        self._signals['average_discounted_returns'] = mean_returns

        # off-policy
        self._signals['mu_utils'] = tf.placeholder(
            tf.float32, shape=(cfg.T, None,) + self.mu.param_shape, name="_mu_log_probs")
        self._signals['mu_exploration'] = tf.placeholder(
            tf.float32, shape=(None,), name="_mu_exploration")
        self._signals['mu_log_probs'] = tf.placeholder(
            tf.float32, shape=(cfg.T, None, 1), name="_mu_log_probs")

        for obj in self.rl_objects:
            obj.build_core_signals(self)

    def make_feed_dict(self, rollouts, mode, weights=None):
        if weights is None:
            weights = np.ones((rollouts.T, rollouts.batch_size, 1))
        elif weights.ndim == 1:
            weights = np.tile(weights.reshape(1, -1, 1), (rollouts.T, 1, 1))

        feed_dict = {
            self._signals['mask']: (1-shift_fill(rollouts.done, 1)).astype('f'),
            self._signals['obs']: rollouts.o,
            self._signals['actions']: rollouts.a,
            self._signals['rewards']: rollouts.r,
            self._signals['weights']: weights,
            self._signals['mu_log_probs']: rollouts.log_probs,
            self._signals['mode']: mode,
        }

        if hasattr(rollouts, 'hidden'):
            feed_dict[self._signals['hidden']] = rollouts.hidden

        if hasattr(rollouts, 'utils'):
            # utils are not always stored in the rollouts as they can occupy a lot of memory
            feed_dict.update({
                self._signals['mu_utils']: rollouts.utils,
                self._signals['mu_exploration']: rollouts.get_static('exploration')
            })

        return feed_dict

    def get_signal(self, key, generator=None, gradient=False, masked=True, memoize=True, **kwargs):
        """ Memoized signal retrieval and generation. """
        if generator is None:
            signal = self._signals[key]
        else:
            try:
                gen_key = hash(generator)
            except TypeError:
                gen_key = id(generator)
            gen_key = str(gen_key)
            signal_key = key
            key = [gen_key, signal_key]

            for k in sorted(kwargs):
                key.append("{}={}".format(k, kwargs[k]))

            key = '|'.join(key)

            if memoize:
                signal = self._signals.get(key, None)
                if signal is None:
                    signal = generator.generate_signal(signal_key, self, **kwargs)
                    self._signals[key] = signal
            else:
                signal = generator.generate_signal(signal_key, self, **kwargs)

        maskable = len(signal.shape) >= 2
        if masked and maskable:
            mask = self._signals['mask']
            diff = len(signal.shape) - len(mask.shape)
            if diff > 0:
                new_shape = tf.concat([tf.shape(mask), [1] * diff], axis=0)
                mask = tf.reshape(mask, new_shape)
            signal *= mask

        if not gradient:
            signal = tf.stop_gradient(signal)

        return signal

    def _run_and_record(self, rollouts, mode, weights, do_update, summary_op, collect_summaries):
        assert do_update or collect_summaries, (
            "Both `do_update` and `collect_summaries` are False, no point in calling `_run_and_record`.")

        sess = tf.get_default_session()
        feed_dict = self.make_feed_dict(rollouts, mode, weights)
        self.set_mode(mode)
        sess.run(self.update_stats_op, feed_dict=feed_dict)

        for obj in self.rl_objects:
            if do_update:
                obj.pre_update(feed_dict, self)
            else:
                obj.pre_eval(feed_dict, self)

        record = {}
        for name in getattr(self.env, 'recorded_names', []):
            v = rollouts._metadata[name]
            feed_dict[self.recorded_tensors[name]] = v
            record[name] = v

        if do_update:
            self.optimizer.update(rollouts.batch_size, feed_dict)

        summaries = b""
        if collect_summaries:
            summaries, *values = (
                sess.run(
                    [summary_op] + [v for _, v in self.recorded_values],
                    feed_dict=feed_dict))
        else:
            values = sess.run(
                [v for _, v in self.recorded_values],
                feed_dict=feed_dict)

        for obj in self.rl_objects:
            if do_update:
                obj.post_update(feed_dict, self)
            else:
                obj.post_eval(feed_dict, self)

        record.update({k: v for v, (k, _) in zip(values, self.recorded_values)})

        return summaries, record

    def update(self, batch_size, collect_summaries):
        assert self.mu is not None, "A behaviour policy must be set using `set_behaviour_policy` before calling `update`."
        assert self.optimizer is not None, "An optimizer must be set using `set_optimizer` before calling `update`."

        with self:
            start = time.time()
            rollouts = self.env.do_rollouts(self.mu, batch_size, mode='train')
            train_rollout_duration = time.time() - start

            train_summaries = b""
            train_record = {}

            start = time.time()
            if self.replay_buffer is None or self.on_policy_updates:
                train_summaries, train_record = self._run_and_record(
                    rollouts, mode='train', weights=None, do_update=True,
                    summary_op=self.train_summary_op,
                    collect_summaries=collect_summaries)
            elif collect_summaries:
                train_summaries, train_record = self._run_and_record(
                    rollouts, mode='train', weights=None, do_update=False,
                    summary_op=self.summary_op, collect_summaries=True)
            train_step_duration = time.time() - start
            train_record.update(step_duration=train_step_duration, rollout_duration=train_rollout_duration)

            off_policy_summaries = b""
            off_policy_record = {}

            if self.replay_buffer is not None:
                start = time.time()

                self.replay_buffer.add_rollouts(rollouts)
                for i in range(self.replay_updates_per_sample):
                    off_policy_rollouts, weights = self.replay_buffer.get_batch(self.update_batch_size)
                    if off_policy_rollouts is None:
                        # Most common reason for `rollouts` being None
                        # is there not being enough experiences in replay memory.
                        break

                    _collect_summaries = (
                        collect_summaries and
                        i == self.replay_updates_per_sample-1
                    )

                    off_policy_summaries, off_policy_record = self._run_and_record(
                        off_policy_rollouts, mode='off_policy', weights=weights, do_update=True,
                        summary_op=self.train_summary_op,
                        collect_summaries=_collect_summaries)

                off_policy_duration = time.time() - start
                off_policy_record['step_duration'] = off_policy_duration

            return train_summaries, off_policy_summaries, train_record, off_policy_record

    def evaluate(self, batch_size, mode):
        assert self.pi is not None, "A validation policy must be set using `set_validation_policy` before calling `evaluate`."

        with self:
            start = time.time()
            rollouts = self.env.do_rollouts(self.pi, batch_size, mode=mode)
            eval_rollout_duration = time.time() - start

            start = time.time()
            eval_summaries, eval_record = self._run_and_record(
                rollouts, mode=mode, weights=None, do_update=False,
                summary_op=self.summary_op,
                collect_summaries=True)
            eval_duration = time.time() - start

            eval_record.update(
                eval_duration=eval_duration, rollout_duration=eval_rollout_duration)

        return eval_summaries, eval_record


class RLUpdater(Updater):
    """ Update parameters of objects (mainly policies and value functions)
        based on sequences of interactions between a behaviour policy and
        an environment.

    Must be used in context of a default graph, session and config.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    learners: RLContext instance or list thereof
        Objects that learn from the trajectories.

    """
    stopping_criteria = "reward_per_ep,max"

    def __init__(self, env, learners=None, **kwargs):
        self.env = env

        learners = learners or []
        try:
            self.learners = list(learners)
        except (TypeError, ValueError):
            self.learners = [learners]

        learner_names = [l.name for l in self.learners]
        assert len(learner_names) == len(set(learner_names)), (
            "Learners must have unique names. Names are: {}".format(learner_names))

        super(RLUpdater, self).__init__(env, **kwargs)

    def trainable_variables(self, for_opt):
        return [v for learner in self.learners for v in learner.trainable_variables(for_opt=for_opt)]

    def _build_graph(self):
        for learner in self.learners:
            learner.build_graph(self.env)

    def _update(self, batch_size, collect_summaries):
        train_record, off_policy_record = {}, {}
        train_summary, off_policy_summary = [], []

        for learner in self.learners:
            ts, us, tr, ur = learner.update(batch_size, collect_summaries)

            for k, v in tr.items():
                if learner.name:
                    s = learner.name + ":" + k
                else:
                    s = k
                train_record[s] = v

            for k, v in ur.items():
                if learner.name:
                    s = learner.name + ":" + k
                else:
                    s = k
                off_policy_record[s] = v

            train_summary.append(ts)
            off_policy_summary.append(us)

        train_summary = (b'').join(train_summary)
        off_policy_summary = (b'').join(off_policy_summary)

        return {
            'train': (train_record, train_summary),
            'off_policy': (off_policy_record, off_policy_summary),
        }

    def _evaluate(self, batch_size, mode):
        record = {}
        summary = []

        for learner in self.learners:
            s, r = learner.evaluate(batch_size, mode)
            summary.append(s)

            for k, v in r.items():
                if learner.name:
                    s = learner.name + ":" + k
                else:
                    s = k
                record[s] = v

        summary = (b'').join(summary)

        return record, summary
