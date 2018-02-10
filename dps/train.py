from __future__ import absolute_import
from __future__ import division
import time
from contextlib import ExitStack
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from pprint import pformat
import datetime
import shutil
import os
import socket
import pandas as pd

import dps
from dps import cfg
from dps.utils import (
    gen_seed, time_limit, memory_usage, ExperimentStore, memory_limit,
    du, Config, ClearConfig, redirect_stream, NumpySeed
)
from dps.utils.tf import (
    restart_tensorboard, uninitialized_variables_initializer, trainable_variables
)
from dps.parallel.object_store import ObjectFragment


class TrainingDataFragment(ObjectFragment):
    def __init__(self, global_step, data):
        # data is a mapping from stage_idx to data
        self.global_step = global_step
        self.data = data

    def combine(self, *others):
        final = None
        fragments = [self] + others
        history = {}

        for fragment in sorted(fragments, key=lambda f: f.global_step):
            if final is None:
                final = dict(
                    config=fragment.data['config'],
                    host=fragment.data['host'],
                    exp_dir=fragment.data['exp_dir'],
                    history=history,
                )

            for stage_idx, _data in sorted(fragment.data.history.items()):
                if stage_idx not in history:
                    history[stage_idx] = _data
                else:
                    for key, value in _data.items():
                        history[stage_idx][key].append(value)
        return final


def stepped_training_loop(exp_name='', start_time=None):
    """ A generator that runs a training loop. Every `checkpoint_step` steps,
        this generator yields an output that summarizes the state of the
        training run so far, acting as a checkpointing mechanism.

    """
    loop = TrainingLoop(exp_name)
    yield from loop.run(start_time)


def training_loop(exp_name='', start_time=None):
    """ Run a training loop without checkpointing,
        returning a summary only at the end of training. """
    return list(stepped_training_loop(exp_name, start_time))[-1]


class EarlyStopHook(object):
    def __init__(self, patience, maximize):
        self.patience = patience
        self.maximize = maximize
        self.reset()

    def _check_trigger(self, sc):
        if self._best_stopping_criteria is None:
            return True

        if self.maximize:
            return sc > self._best_stopping_criteria
        else:
            return sc < self._best_stopping_criteria

    def check(self, stopping_criteria, step, record):
        new_best = self._check_trigger(stopping_criteria)
        if new_best:
            self._best_stopping_criteria = stopping_criteria
            self._best_step = step
            self._best_record = record.copy()

        self._early_stopped = (
            self._early_stopped or
            (step - self._best_step > self.patience))
        return new_best, self._early_stopped

    @property
    def best(self):
        best = self._best_record.copy()
        best.update(stopping_criteria=self._best_stopping_criteria, local_step=self._best_step)
        return best

    def reset(self):
        self._best_stopping_criteria = None
        self._best_record = None
        self._best_step = None
        self._early_stopped = 0


def load_or_train(train_config, var_scope, path, target_var_scope=None, sess=None):
    """ Attempts to load variables into ``var_scope`` from checkpoint stored at ``path``.

    If said checkpoint is not found, trains a model using the function
    ``train`` and stores the resulting variables for future use.

    Returns True iff model was successfully loaded, False otherwise.

    If `target_var_scope` is not None, look for the variables under that scope name in the file
    that we load from, instead of `var_scope`.

    """
    sess = sess or tf.get_default_session()

    to_be_loaded = trainable_variables(var_scope, for_opt=False)
    if target_var_scope is not None:
        _tbl = {}
        for var in to_be_loaded:
            assert var.name.startswith(var_scope.name)
            bare_name = var.name[len(var_scope.name):]
            while bare_name.startswith('/'):
                bare_name = bare_name[1:]
            name_in_file = target_var_scope + '/' + bare_name
            _tbl[name_in_file] = var
        to_be_loaded = _tbl
    else:
        to_be_loaded = {v.name: v for v in to_be_loaded}

    saver = tf.train.Saver(to_be_loaded)

    if path is not None:
        # Make sure that the location we want to save the result exists
        os.mkdirs(os.path.dirname(path), exist_ok=True)

    success = False
    try:
        saver.restore(sess, path)
        success = True
    except tf.errors.NotFoundError:
        with ExitStack() as stack:
            stack.enter_context(ClearConfig())
            stack.enter_context(train_config.copy(save_path=path))

            output = training_loop(var_scope.name)

            stem = os.path.splitext(path)[0]
            shutil.copyfile(os.path.join(output['exp_dir'], 'stdout'), stem + '.stdout')
            shutil.copyfile(os.path.join(output['exp_dir'], 'stderr'), stem + '.stderr')

        saver.restore(sess, path)
    return success


class Hook(object):
    """ Hook called throughout training.

    Parameters
    ----------
    n: int
        Hook is called every n steps throughout training.
    mode: str
        The mode in which this hook is called (essentially determines where its summaries end up).
    initial: bool
        If True, this hook is called on the first step of a stage.

    """
    def __init__(self, n, mode, initial=False):
        self.n = n
        self.mode = mode
        self.initial = initial

    def start_stage(self, stage_idx):
        """ Called at the beginning of every stage. """
        pass

    def end_stage(self):
        """ Called at the end of every stage. """
        pass

    def step(self, updater, step_idx):
        """ May return a list of summaries and a dictionary of recorded values, similar to an updater. """
        pass


class TrainingLoop(object):
    """ A training loop.

    The behaviour of the training loop depends on the context stack that is active when it is
    run (i.e. `run` method is called), not when it is created.

    Parameters
    ----------
    exp_name: str
        Name of the experiment, used as a prefix when creating a directory for the training loop.
    hooks: list of Hook instances
        Hooks to run throughout training.

    """
    def __init__(self, exp_name='', hooks=None):
        self.exp_name = exp_name or cfg.get_experiment_name()
        self.hooks = hooks or []
        self.history = {}
        self.start_time = None

    @property
    def current_stage_record(self):
        return self.history[self.stage_idx]

    def record_values(self, d=None, **kwargs):
        """ Record values for the current stage. """
        d = d or {}
        self.current_stage_record.update(d)
        self.current_stage_record.update(kwargs)

    def summarize(self, current_stage_only=False):
        """
        Parameters
        ----------
        current_stage_only: bool
            If True, only output info for the current/most recent stage.

        """
        s = "\n"
        if current_stage_only:
            itr = [(self.stage_idx, self.current_stage_record)]
        else:
            s += "Stage-by-stage summary " + ">" * 30 + "\n"
            itr = sorted(self.history.items())

        for stage_idx, record in itr:
            if current_stage_only:
                s += "Step(g: {}, l: {}): ".format(self.global_step, self.local_step)
            else:
                s += "Stage {} ".format(stage_idx)

            s += "*" * 30 + '\n'

            for k, v in sorted(record.items()):
                if k in 'train_data off_policy_data val_data test_data'.split() and len(v) > 0:

                    if isinstance(v, pd.DataFrame):
                        s += "* {} (final_step): {}\n".format(k, v.iloc[-1].to_dict())

                    elif isinstance(v, str):
                        # CSV representation of a data-frame
                        try:
                            lines = [l for l in v.split('\n') if l]
                            keys = lines[0].split(',')
                            values = lines[-1].split(',')
                            d = dict(zip(keys, values))
                            s += "* {} (final_step): {}\n".format(k, sorted(d.items()))
                        except IndexError:
                            pass

                    else:
                        # A list of records
                        final_value = v[-1]
                        if isinstance(final_value, dict):
                            s += "* {} (final_step): {}\n".format(k, sorted(final_value.items()))
                        else:
                            s += "* {} (final_step): {}\n".format(k, final_value)
                else:
                    s += "* {}: {}\n".format(k, v)

            s += "* new cfg values: {}\n\n".format(pformat(self.curriculum[stage_idx]))
        return s

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def time_remaining(self):
        if cfg.max_time is None or cfg.max_time <= 0:
            return np.inf
        else:
            return cfg.max_time - self.elapsed_time

    def _build_return_value(self):
        stored_history = {}

        for stage_idx, record in self.history.items():
            new_record = {}

            for kind in 'train off_policy val test'.split():
                key = kind + '_data'
                if key in record:
                    # new_record[key] = pd.DataFrame.from_records(record[key]).to_csv(index=False)
                    new_record[key] = record[key]
                    record[key] = []

            stored_history[stage_idx] = new_record

        return_value = dict(
            config=cfg.freeze(remove_callable=True),
            history=stored_history,
            host=socket.gethostname(),
            exp_dir=self.exp_dir.path
        )

        return TrainingDataFragment(global_step=self.global_step, data=return_value)

    def run(self, start_time):
        """ Run the training loop.

        Parameters
        ----------
        start_time: int
            Start time (in seconds since epoch) for measuring elapsed time for
            purposes of interrupting the training loop.

        """
        self.curriculum = cfg.curriculum + []

        if cfg.seed is None or cfg.seed < 0:
            cfg.seed = gen_seed()

        if cfg.start_tensorboard:
            restart_tensorboard(str(cfg.log_dir), cfg.tbport, cfg.reload_interval)

        with ExitStack() as stack:
            # Create a directory to store the results of the training session.
            es = ExperimentStore(str(cfg.log_dir), max_experiments=cfg.max_experiments, delete_old=1)
            self.exp_dir = exp_dir = es.new_experiment(
                self.exp_name, cfg.seed, add_date=1, force_fresh=1, update_latest=cfg.update_latest)

            # Record training session environment for post-mortem diagnostic purposes
            exp_dir.record_environment(config=cfg.freeze(), git_modules=[dps])

            # Tee stdout and stderr so that it gets written to a file
            stack.enter_context(redirect_stream('stdout', exp_dir.path_for('stdout'), tee=cfg.tee))
            stack.enter_context(redirect_stream('stderr', exp_dir.path_for('stderr'), tee=cfg.tee))

            exp_dir.make_directory('plots')

            if start_time is None:
                start_time = time.time()
            self.start_time = start_time

            print("\n\n" + "=" * 80)
            print("Starting training run (name={}) at {}, {} seconds after given "
                  "start time.".format(cfg.name, datetime.datetime.now(), time.time() - self.start_time))

            print("\nScratch directory for this training run is {}.".format(exp_dir.path))
            cfg.path = exp_dir.path

            stack.enter_context(NumpySeed(cfg.seed))

            print("Set numpy random seed to {}.".format(cfg.seed))

            yield from self._run()

            print("Done training run (name={}) at {}, {} seconds after given "
                  "start time.".format(cfg.name, datetime.datetime.now(), time.time() - self.start_time))
            print("=" * 80)
            print("\n\n")

    def _run(self):
        print(cfg.to_string())

        threshold_reached = True
        self.global_step = 0

        for stage_idx, stage_config in enumerate(self.curriculum):
            print("\n" + "=" * 50)
            print("Starting stage {} at {}, {} seconds after given "
                  "start time.\n".format(stage_idx, datetime.datetime.now(), time.time() - self.start_time))
            print("\n")

            stage_config = Config(stage_config)

            if self.time_remaining <= 1:
                print("Time limit exceeded.")
                break

            stage_start_time = time.time()

            # Initial record for the stage.
            stage_record = dict(
                stage=stage_idx,
                train_data=[],
                off_policy_data=[],
                val_data=[],
                test_data=[]
            )

            self.stage_idx = stage_idx
            self.history[self.stage_idx] = stage_record

            with ExitStack() as stack:

                # --------------- Stage set-up -------------------

                # Configure (number of threads and gpu) and create and create session and graph for stage.
                session_config = tf.ConfigProto()
                session_config.intra_op_parallelism_threads = cfg.get('intra_op_parallelism_threads', 0)
                session_config.inter_op_parallelism_threads = cfg.get('inter_op_parallelism_threads', 0)

                if cfg.use_gpu:
                    per_process_gpu_memory_fraction = getattr(cfg, 'per_process_gpu_memory_fraction', None)
                    if per_process_gpu_memory_fraction:
                        session_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction

                    gpu_allow_growth = getattr(cfg, 'gpu_allow_growth', None)
                    if gpu_allow_growth:
                        session_config.gpu_options.allow_growth = gpu_allow_growth

                print("Available devices: ")
                print(device_lib.list_local_devices())
                print("\n")

                if cfg.use_gpu:
                    print("Using GPU if available.")
                    print("Using {}% of GPU memory.".format(
                        100 * session_config.gpu_options.per_process_gpu_memory_fraction))
                    print("Allowing growth of GPU memory: {}".format(session_config.gpu_options.allow_growth))

                graph = tf.Graph()
                sess = tf.Session(graph=graph, config=session_config)

                if not cfg.use_gpu:
                    print("Not using GPU.")
                    stack.enter_context(graph.device("/cpu:0"))

                # Initialize summary writers
                if cfg.save_summaries:
                    self.train_writer = tf.summary.FileWriter(
                        self.exp_dir.path_for('train'), graph, flush_secs=cfg.reload_interval)
                    self.off_policy_writer = tf.summary.FileWriter(
                        self.exp_dir.path_for('off_policy'), flush_secs=cfg.reload_interval)
                    self.val_writer = tf.summary.FileWriter(
                        self.exp_dir.path_for('val'), flush_secs=cfg.reload_interval)
                    self.test_writer = tf.summary.FileWriter(
                        self.exp_dir.path_for('test'), flush_secs=cfg.reload_interval)

                    print("Writing summaries to {}.".format(self.exp_dir.path))

                stack.enter_context(graph.as_default())
                stack.enter_context(sess)
                stack.enter_context(sess.as_default())

                # Set the seed for the stage. Notice we generate a new tf seed for each stage.
                tf_seed = gen_seed()
                tf.set_random_seed(tf_seed)

                # Set limit on CPU RAM for the stage
                cpu_ram_limit_mb = cfg.get("cpu_ram_limit_mb", None)
                if cpu_ram_limit_mb is not None:
                    stack.enter_context(memory_limit(cfg.cpu_ram_limit_mb))

                print("New config values for this stage are: \n{}\n".format(pformat(stage_config)))

                stack.enter_context(stage_config)

                # Build env
                if stage_idx == 0 or not cfg.preserve_env:
                    if getattr(self, 'env', None):
                        self.env.close()

                    self.env = cfg.build_env()

                # Build updater
                updater = cfg.get_updater(self.env)
                updater.build_graph()

                # Optionally initialize policy weights
                if cfg.load_path:
                    # Initialize weights from specified check point file
                    if isinstance(cfg.load_path, list):
                        repeat = getattr(cfg, 'repeat', 0)
                        path = cfg.load_path[repeat % len(cfg.load_path)]
                    else:
                        path = cfg.load_path
                    path = os.path.realpath(path)
                    assert isinstance(path, str)
                    print("Loading hypothesis from {}.".format(path))
                    updater.restore(sess, path)
                elif stage_idx > 0 and cfg.preserve_policy:
                    # Initialize weights from best hypothesis discovered on the previous stage
                    updater.restore(sess, self.history[stage_idx-1]['best_path'])

                self.summary_op = tf.summary.merge_all()
                tf.train.get_or_create_global_step()
                sess.run(uninitialized_variables_initializer())
                sess.run(tf.assert_variables_initialized())

                # --------------- Run stage -------------------

                yield from self.run_stage(stage_idx, updater)

                # --------------- Record stage results -------------------

                threshold_reached, reason = self.threshold_reached, self.reason
                self.record_values(reason=reason)

                print("Optimization complete. Reason: {}.".format(reason))
                print("Best hypothesis for this stage was found on "
                      "step (g: {best_global_step}, l: {best_local_step}) "
                      "with stopping criteria ({sc_name}) of {best_stopping_criteria}.".format(
                          sc_name=self.stopping_criteria_name, **self.current_stage_record))

                best_path = self.current_stage_record['best_path']
                print("Loading best hypothesis for this stage "
                      "from file {}...".format(best_path))
                updater.restore(sess, best_path)

                _, test_record = updater.evaluate(cfg.n_val, 'test')
                print("Results on test dataset: ")
                print(test_record)
                self.record_values(**{'test_' + k: v for k, v in test_record.items()})

                print(self.summarize(current_stage_only=True))

                if cfg.start_tensorboard:
                    restart_tensorboard(str(cfg.log_dir), cfg.tbport, cfg.reload_interval)

                if cfg.render_step > 0 and cfg.render_hook is not None:
                    cfg.render_hook(updater)

                if not (threshold_reached or cfg.power_through):
                    print("Failed to reach error threshold on stage {} "
                          "of the curriculum, terminating.".format(stage_idx))
                    break

                self.record_values(stage_duration=time.time()-stage_start_time)

                print("Done stage {} at {}, {} seconds after given "
                      "start time.".format(stage_idx, datetime.datetime.now(), time.time() - self.start_time))
                print("=" * 50)

        print(self.summarize(current_stage_only=False))

        if cfg.slim:
            print("`slim` is True, so deleting experiment directory {}.".format(self.exp_dir.path))
            print("Size of {} before delete: {}.".format(cfg.log_dir, du(cfg.log_dir)))
            try:
                shutil.rmtree(self.exp_dir.path)
            except FileNotFoundError:
                pass
            print("Size of {} after delete: {}.".format(cfg.log_dir, du(cfg.log_dir)))

        yield self._build_return_value()

    def run_stage(self, stage_idx, updater):
        self.threshold_reached = False
        self.reason = ""

        # Parse stopping criteria
        stopping_criteria = cfg.get("stopping_criteria", "")
        if not stopping_criteria:
            stopping_criteria = updater.stopping_criteria

        if isinstance(stopping_criteria, str):
            stopping_criteria = stopping_criteria.split(",")

        self.stopping_criteria_name = stopping_criteria[0]
        if "max" in stopping_criteria[1]:
            self.maximize_sc = True
        elif "min" in stopping_criteria[1]:
            self.maximize_sc = False
        else:
            raise Exception("Ambiguous stopping criteria specification: {}".format(stopping_criteria[1]))

        # Set-up hooks
        early_stop = EarlyStopHook(patience=cfg.patience, maximize=self.maximize_sc)
        for hook in self.hooks:
            hook.start_stage(stage_idx)

        # Run stage with time and memory limits
        print("{} seconds left at the beginning of stage {}.".format(self.time_remaining, stage_idx))

        phys_memory_before = memory_usage(physical=True)

        with time_limit(self.time_remaining, verbose=True) as limiter:
            try:
                yield from self._run_stage(stage_idx, updater, early_stop)
            except KeyboardInterrupt:
                self.threshold_reached = False
                self.reason = "User interrupt"
            except NotImplementedError as e:
                # There is a bug that prevents instances of `NotImplementedError`
                # from being handled properly, so replace it with an instance of `Exception`.
                raise Exception("NotImplemented") from e

        phys_memory_after = memory_usage(physical=True)

        self.record_values(
            stage_duration=limiter.elapsed_time,
            phys_memory_before_mb=phys_memory_before,
            phys_memory_after_mb=phys_memory_after,
            phys_memory_delta_mb=phys_memory_after - phys_memory_before
        )

        for hook in self.hooks:
            hook.end_stage()

        if limiter.ran_out:
            self.reason = "Time limit exceeded"
            if cfg.error_on_timeout:
                raise Exception("Timed out.")

    def _run_stage(self, stage_idx, updater, early_stop):
        """ Run main training loop for a stage of the curriculum. """
        self.local_step = 0
        threshold_reached = False
        reason = None
        total_train_time = 0.0
        time_per_example = 0.0
        time_per_batch = 0.0

        while True:
            # Check whether to keep training
            if self.local_step >= cfg.max_steps:
                reason = "Maximum number of steps reached"
                break

            if updater.n_experiences >= cfg.max_experiences:
                reason = "Maximum number of experiences reached"
                break

            # Temporarily yield results gathered so far (for checkpointing purposes)
            if self.local_step > 0 and self.local_step % cfg.checkpoint_step == 0:
                yield self._build_return_value()

            evaluate = self.local_step % cfg.eval_step == 0
            display = self.local_step % cfg.display_step == 0
            render = cfg.render_step > 0 and (self.local_step % cfg.render_step == 0) and self.local_step > 0

            # --------------- Perform an update -------------------

            train_summaries, off_policy_summaries = b"", b""
            train_record, off_policy_record = {}, {}

            update_start_time = time.time()

            if cfg.do_train:
                train_summaries, off_policy_summaries, train_record, off_policy_record = updater.update(
                    cfg.batch_size, collect_summaries=evaluate and cfg.save_summaries)

            update_duration = time.time() - update_start_time

            self.current_stage_record['train_data'].append(train_record)
            self.current_stage_record['off_policy_data'].append(off_policy_record)

            # --------------- Possibly evaluate -------------------

            if evaluate or display:
                val_summaries, val_record = updater.evaluate(cfg.n_val, 'val')
                self.current_stage_record['val_data'].append(val_record)

                test_summaries, test_record = updater.evaluate(cfg.n_val, 'test')
                self.current_stage_record['test_data'].append(test_record)

                if evaluate and cfg.save_summaries:
                    self.train_writer.add_summary(train_summaries, (self.global_step + 1) * cfg.batch_size)
                    self.off_policy_writer.add_summary(off_policy_summaries, (self.global_step + 1) * cfg.batch_size)
                    self.val_writer.add_summary(val_summaries, (self.global_step + 1) * cfg.batch_size)
                    self.test_writer.add_summary(test_summaries, (self.global_step + 1) * cfg.batch_size)

                stopping_criteria = val_record[self.stopping_criteria_name]
                new_best, stop = early_stop.check(stopping_criteria, self.local_step, val_record)

                if new_best:
                    print("Storing new best on (local, global) step ({}, {}), "
                          "constituting {} local experiences, "
                          "with stopping criteria ({}) of {}.".format(
                              self.local_step, self.global_step, updater.n_experiences,
                              self.stopping_criteria_name, stopping_criteria))

                    path = cfg.get('save_path', '')
                    path = path or self.exp_dir.path_for('best_of_stage_{}'.format(stage_idx))

                    best_path = updater.save(tf.get_default_session(), path)

                    self.record_values(best_path=best_path, best_global_step=self.global_step)
                    self.record_values(**{'best_' + k: v for k, v in early_stop.best.items()})

                if stop:
                    print("Early stopping triggered.")
                    reason = "Early stopping triggered"
                    break

                if self.maximize_sc:
                    threshold_reached = stopping_criteria > cfg.threshold
                else:
                    threshold_reached = stopping_criteria < cfg.threshold

                if threshold_reached:
                    reason = "Stopping criteria threshold reached"
                    break

                self.record_values(
                    time_per_example=time_per_example,
                    time_per_batch=time_per_batch,
                    n_steps=self.local_step,
                    n_experiences=self.local_step*cfg.batch_size,
                    epoch=updater.env.completion
                )

                if display:
                    print(self.summarize(current_stage_only=True))
                    print("\nPhysical memory use: {}mb".format(memory_usage(physical=True)))
                    print("Virtual memory use: {}mb".format(memory_usage(physical=False)))

            # Run training hooks
            for hook in self.hooks:
                run_hook = self.local_step == 0 and hook.initial
                run_hook |= self.local_step > 0 and self.local_step % hook.n == 0

                if run_hook:
                    result = hook.step(updater)

                    if result:
                        # TODO: currently nothing is done with the record
                        summaries, record = result
                        writer = getattr(self, "{}_writer".format(hook.mode))
                        writer.add_summary(summaries, (self.global_step + 1) * cfg.batch_size)

            # Possibly render
            if render and cfg.render_hook is not None:
                cfg.render_hook(updater)

            # If `do_train` is False, we do no training and evaluate
            # exactly once, so only one iteration is required.
            if not cfg.do_train:
                reason = "`do_train` set to False"
                break

            total_train_time += update_duration
            time_per_example = total_train_time / ((self.local_step+1) * cfg.batch_size)
            time_per_batch = total_train_time / (self.local_step+1)

            self.local_step += 1
            self.global_step += 1

        self.threshold_reached = threshold_reached
        self.reason = reason
