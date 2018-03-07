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
import pandas as pd
import dill
from collections import defaultdict

import dps
from dps import cfg
from dps.utils import (
    gen_seed, time_limit, memory_usage, ExperimentStore, ExperimentDirectory,
    memory_limit, du, Config, ClearConfig, redirect_stream, NumpySeed, make_symlink
)
from dps.utils.tf import (
    restart_tensorboard, uninitialized_variables_initializer, trainable_variables
)


def training_loop(exp_name='', start_time=None):
    loop = TrainingLoop(exp_name)
    return loop.run(start_time)


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

        if self.patience > 0:
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
        os.makedirs(os.path.dirname(path), exist_ok=True)

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
            shutil.copyfile(output.path_for('stdout'), stem + '.stdout')
            shutil.copyfile(output.path_for('stderr'), stem + '.stderr')

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
        self.start_time = None

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def time_remaining(self):
        if cfg.max_time is None or cfg.max_time <= 0:
            return np.inf
        else:
            return cfg.max_time - self.elapsed_time

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
            restart_tensorboard(cfg.log_dir, cfg.tbport, cfg.reload_interval)

        with ExitStack() as stack:
            # Create a directory to store the results of the training session.
            es = ExperimentStore(cfg.log_dir, max_experiments=cfg.max_experiments, delete_old=1)
            exp_dir = es.new_experiment(
                self.exp_name, cfg.seed, add_date=1, force_fresh=1, update_latest=cfg.update_latest)
            self.exp_dir = exp_dir

            make_symlink(exp_dir.path, os.path.join(os.getenv("HOME"), "dps-latest"))

            self.data = _TrainingLoopData(exp_dir)
            self.data.setup()

            # Tee stdout and stderr to files
            stack.enter_context(redirect_stream('stdout', self.data.path_for('stdout'), tee=cfg.tee))
            stack.enter_context(redirect_stream('stderr', self.data.path_for('stderr'), tee=cfg.tee))

            if start_time is None:
                start_time = time.time()
            self.start_time = start_time

            print("\n\n" + "=" * 80)
            print("Starting training run (name={}) at {}, {} seconds after given "
                  "start time.".format(cfg.name, datetime.datetime.now(), time.time() - self.start_time))

            print("\nScratch directory for this training run is {}.".format(self.data.path))
            cfg.path = self.data.path

            stack.enter_context(NumpySeed(cfg.seed))

            print("Set numpy random seed to {}.".format(cfg.seed))

            self._run()

            print("Done training run (name={}) at {}, {} seconds after given "
                  "start time.".format(cfg.name, datetime.datetime.now(), time.time() - self.start_time))
            print("=" * 80)
            print("\n\n")

            return self.data.freeze(self.local_step)

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

            self.data.start_stage(stage_idx)

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
                updater.stage_idx = stage_idx
                updater.exp_dir = self.exp_dir

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
                    updater.restore(sess, self.data.history[-2]['best_path'])

                self.summary_op = tf.summary.merge_all()
                tf.train.get_or_create_global_step()
                sess.run(uninitialized_variables_initializer())
                sess.run(tf.assert_variables_initialized())

                # --------------- Run stage -------------------

                self.run_stage(stage_idx, updater)

                threshold_reached, reason = self.threshold_reached, self.reason
                self.data.record_values_for_stage(reason=reason)

                # --------------- Evaluate the best hypothesis -------------------

                print("\n" + "-" * 10 + " Optimization complete " + "-" * 10)
                print("\nReason: {}.\n".format(reason))

                print("Best hypothesis for this stage was found on "
                      "step (l: {best_local_step}, g: {best_global_step}) "
                      "with stopping criteria ({sc_name}) of {best_stopping_criteria}.".format(
                          sc_name=self.stopping_criteria_name, **self.data.current_stage_record))

                best_path = self.data.current_stage_record['best_path']
                print("Loading best hypothesis for this stage "
                      "from file {}...".format(best_path))
                updater.restore(sess, best_path)

                eval_results = updater.evaluate(cfg.n_val)

                print("\n" + "-" * 10 + " Final evaluation " + "-" * 10)
                for mode, (record, _) in eval_results.items():
                    if record:
                        print("\n-- {} -- \n".format(mode))
                        for k, v in sorted(record.items()):
                            print("* {}: {}".format(k, v))

                print()

                # --------------- Optionally render performance of best hypothesis -------------------

                if cfg.render_step > 0 and cfg.render_hook is not None:
                    cfg.render_hook(updater)

                # --------------- Finish up the stage -------------------

                self.data.end_stage(self.local_step)

                if cfg.start_tensorboard:
                    restart_tensorboard(cfg.log_dir, cfg.tbport, cfg.reload_interval)

                if not (threshold_reached or cfg.power_through):
                    print("Failed to reach error threshold on stage {} "
                          "of the curriculum, terminating.".format(stage_idx))
                    break

                print("Done stage {} at {}, {} seconds after given start time.".format(
                    stage_idx, datetime.datetime.now(), time.time() - self.start_time))
                print("=" * 50)

        print(self.data.summarize())

        if cfg.slim:
            print("`slim` is True, so deleting experiment directory {}.".format(self.data.path))
            print("Size of {} before delete: {}.".format(cfg.log_dir, du(cfg.log_dir)))
            shutil.rmtree(self.data.path, ignore_errors=True)
            print("Size of {} after delete: {}.".format(cfg.log_dir, du(cfg.log_dir)))

    def run_stage(self, stage_idx, updater):
        self.threshold_reached = False
        self.reason = ""

        # Parse stopping criteria
        stopping_criteria = cfg.get("stopping_criteria", updater.stopping_criteria)

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
                self._run_stage(stage_idx, updater, early_stop)
            except KeyboardInterrupt:
                self.threshold_reached = False
                self.reason = "User interrupt"
            except NotImplementedError as e:
                # There is a bug that prevents instances of `NotImplementedError`
                # from being handled properly, so replace it with an instance of `Exception`.
                raise Exception("NotImplemented") from e

        phys_memory_after = memory_usage(physical=True)

        self.data.record_values_for_stage(
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

            if self.local_step > 0 and self.local_step % cfg.checkpoint_step == 0:
                self.data.dump_data(self.local_step)

            evaluate = (self.local_step % cfg.eval_step) == 0
            display = (self.local_step % cfg.display_step) == 0
            render = (cfg.render_step > 0 and
                      (self.local_step % cfg.render_step) == 0 and
                      self.local_step > 0)

            n_global_experiences = (self.global_step + 1) * cfg.batch_size

            # --------------- Perform an update -------------------

            update_start_time = time.time()

            if cfg.do_train:
                collect_summaries = evaluate and cfg.save_summaries

                update_output = updater.update(
                    cfg.batch_size, collect_summaries=collect_summaries)

                self.data.store_step_data_and_summaries(
                    stage_idx, self.local_step, self.global_step, n_global_experiences,
                    **update_output)

            update_duration = time.time() - update_start_time

            # --------------- Possibly evaluate -------------------

            if evaluate or display:
                eval_results = updater.evaluate(cfg.n_val)

                self.data.store_step_data_and_summaries(
                    stage_idx, self.local_step, self.global_step, n_global_experiences,
                    **eval_results)

                record = eval_results[cfg.eval_mode][0]

                stopping_criteria = record[self.stopping_criteria_name]
                new_best, stop = early_stop.check(
                    stopping_criteria, self.local_step, record)

                if new_best:
                    print("Storing new best on (local, global) step ({}, {}), "
                          "constituting {} local experiences, "
                          "with stopping criteria ({}) of {}.".format(
                              self.local_step, self.global_step,
                              updater.n_experiences, self.stopping_criteria_name,
                              stopping_criteria))

                    best_path = self.data.path_for(
                        'weights/best_of_stage_{}'.format(stage_idx))
                    best_path = cfg.get('save_path', best_path)
                    best_path = updater.save(tf.get_default_session(), best_path)

                    self.data.record_values_for_stage(
                        best_path=best_path, best_global_step=self.global_step)
                    self.data.record_values_for_stage(
                        **{'best_' + k: v for k, v in early_stop.best.items()})

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

                self.data.record_values_for_stage(
                    time_per_example=time_per_example,
                    time_per_batch=time_per_batch,
                    n_steps=self.local_step,
                    n_experiences=self.local_step*cfg.batch_size,
                    epoch=updater.completion
                )

                if display:
                    print(self.data.summarize(self.local_step, self.global_step))
                    print("\nPhysical memory use: "
                          "{}mb".format(memory_usage(physical=True)))
                    print("Virtual memory use: "
                          "{}mb".format(memory_usage(physical=False)))

            # Run training hooks
            for hook in self.hooks:
                run_hook = self.local_step == 0 and hook.initial
                run_hook |= self.local_step > 0 and self.local_step % hook.n == 0

                if run_hook:
                    result = hook.step(updater)

                    if result:
                        # TODO: currently nothing is done with the record
                        record, summary = result
                        self.data.add_summary(n_global_experiences, hook.mode, summary)

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


class FrozenTrainingLoopData(ExperimentDirectory):
    """ Interface for the on-disk data generated by a training loop.

    Parameters
    ----------
    path: str
        Path to the the directory for the experiment whose data we want
        to anaylze.  Should contain a sub-directory for each data-collection
        mode (e.g. train, test).

    """
    def __init__(self, path):
        self.path = path.path if isinstance(path, ExperimentDirectory) else path
        self._config = None
        self._history = None

    def get_summary_path(self, mode):
        return self.path_for('summaries/' + mode, is_dir=True)

    def get_data_path(self, mode, stage_idx, local_step):
        local_path = 'data/{}/stage{}/localstep={}.csv'.format(mode, stage_idx, local_step)
        return self.path_for(local_path)

    def step_data(self, mode, stage_slice=None):
        indices = range(self.n_stages)
        if stage_slice is None:
            pass
        elif isinstance(stage_slice, int):
            indices = [indices[stage_slice]]
        elif isinstance(stage_slice, slice):
            indices = indices[stage_slice]
        else:
            start, end, *step = stage_slice
            step = step[0] if step else 1
            indices = indices[start:end:step]

        data = {}

        for stage_idx in indices:
            local_path = 'data/{}/stage{}'.format(mode, stage_idx)
            path = self.path_for(local_path)
            files = os.listdir(path) if os.path.isdir(path) else []
            for f in files:
                local_step = int(f.split('=')[1].split('.')[0])
                data[(stage_idx, local_step)] = pd.read_csv(os.path.join(path, f))

        data_frames = [d[1] for d in sorted(data.items())]
        if data_frames:
            return pd.concat(data_frames, axis=0, ignore_index=True)
        else:
            return None

    @property
    def config(self):
        if self._config is None:
            with open(self.path_for('config.pkl'), 'rb') as f:
                self._config = dill.load(f)
        return self._config

    @property
    def n_stages(self):
        return len(self.history)

    @property
    def history(self):
        if self._history is None:
            with open(self.path_for('history.pkl'), 'rb') as f:
                self._history = dill.load(f)
        return self._history

    @property
    def modes(self):
        pass


class _TrainingLoopData(FrozenTrainingLoopData):
    """ Data structure used by a TrainingLoop to manage data
        throughout the experiment.

    """
    def setup(self):
        # Record training session environment for later diagnostic purposes
        self.record_environment(config=cfg.freeze(), git_modules=[dps])
        self.curriculum = cfg.curriculum + []

        self.make_directory('weights')
        self.make_directory('plots')
        self.make_directory('data')
        self.make_directory('summaries')

        self._history = []

        self.data = defaultdict(list)
        self.summary_writers = {}

        self.stage_idx = -1

    @property
    def history(self):
        return self._history

    def start_stage(self, stage_idx):
        self.history.append(dict(stage_idx=stage_idx))
        self.stage_idx = stage_idx
        self.summary_writers = {}

    def end_stage(self, local_step):
        self.dump_data(local_step)
        for writer in self.summary_writers.values():
            writer.close()

    def dump_data(self, local_step):
        for mode, data in self.data.items():
            if data:
                path = self.get_data_path(mode, self.stage_idx, local_step)

                with open(path, 'w') as f:
                    pd.DataFrame.from_records(data).to_csv(f, index=False)

                self.data[mode] = []

    def record_values_for_stage(self, d=None, **kwargs):
        """ Record values for the current stage. """
        d = d or {}
        self.current_stage_record.update(d)
        self.current_stage_record.update(kwargs)

    def store_step_records(self, stage_idx, local_step, global_step, **step_records):
        for mode, record in step_records.items():
            if record:
                record.update(
                    local_step=local_step,
                    global_step=global_step,
                    stage_idx=stage_idx)

                self.data[mode].append(record)

    def store_step_data_and_summaries(self, stage_idx, local_step, global_step, n_global_experiences, **data):
        for mode, (record, summary) in data.items():
            if record and cfg.store_step_data:
                record = record.copy()
                record.update(
                    local_step=local_step,
                    global_step=global_step,
                    stage_idx=stage_idx)

                self.data[mode].append(record)

            if summary:
                self.add_summary(mode, n_global_experiences, summary)

    def _get_summary_writer(self, mode):
        if mode not in self.summary_writers:
            self.summary_writers[mode] = tf.summary.FileWriter(
                self.get_summary_path(mode), flush_secs=cfg.reload_interval)
        return self.summary_writers[mode]

    def add_summary(self, mode, n_global_experiences, summary):
        writer = self._get_summary_writer(mode)
        writer.add_summary(summary, n_global_experiences)

    @property
    def current_stage_record(self):
        return self.history[-1]

    def _finalize(self, local_step):
        """ Write all stored data to disk. """
        self.dump_data(local_step)

        with open(self.path_for('history.pkl'), 'wb') as f:
            dill.dump(self.history, f, protocol=dill.HIGHEST_PROTOCOL, recurse=False)

    def freeze(self, local_step):
        self._finalize(local_step)
        return FrozenTrainingLoopData(self.path)

    def summarize(self, *steps):
        """ Summarize the training data.

        Parameters
        ----------
        steps: pair of ints
            local_step, global_step

        """
        s = "\n"
        current_stage_only = bool(steps)
        if current_stage_only:
            local_step, global_step = steps
            history = [self.current_stage_record]
        else:
            s += "Stage-by-stage summary " + ">" * 30 + "\n"
            history = self.history

        for record in history:
            stage_idx = record['stage_idx']

            if current_stage_only:
                s += "\nStep(l: {}, g: {}, stage: {}): ".format(local_step, global_step, stage_idx)
            else:
                s += "\nStage {} ".format(stage_idx)

            s += "*" * 30 + '\n'

            for k, v in sorted(record.items()):
                s += "* {}: {}\n".format(k, v)

            s += "* new cfg values:\n{}\n".format(pformat(self.curriculum[stage_idx]))

        for mode, data in self.data.items():
            if data:
                record = data[-1]

                if record:
                    s += "\n-- {} -- \n".format(mode)
                    for k, v in sorted(record.items()):
                        s += "* {}: {}\n".format(k, v)

        return s
