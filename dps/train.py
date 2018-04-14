from __future__ import absolute_import
from __future__ import division
import time
from contextlib import ExitStack
import tensorflow as tf
import numpy as np
from pprint import pformat
import datetime
import shutil
import os
import pandas as pd
import dill
from collections import defaultdict
import traceback

import dps
from dps import cfg
from dps.utils import (
    gen_seed, time_limit, Alarm, memory_usage, ExperimentStore, ExperimentDirectory, nvidia_smi,
    memory_limit, Config, ClearConfig, redirect_stream, NumpySeed, make_symlink
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


class TrainingLoop(object):
    """ A training loop.

    The behaviour of the training loop depends on the context stack that is active when it is
    run (i.e. `run` method is called), not when it is created.

    Parameters
    ----------
    exp_name: str
        Name of the experiment, used as a prefix when creating a directory for the training loop.

    """
    def __init__(self, exp_name=''):
        self.exp_name = exp_name or cfg.log_name
        self.start_time = None

    @property
    def time_remaining(self):
        if cfg.max_time is None or cfg.max_time <= 0:
            return np.inf
        else:
            elapsed_time = time.time() - self.start_time
            return cfg.max_time - elapsed_time

    def edit_remaining_stage(self, idx, stage_config):
        if len(self.curriculum_remaining) < idx+1:
            for i in range(idx+1 - len(self.curriculum_remaining)):
                self.curriculum_remaining.append(dict())

        self.curriculum_remaining[idx].update(stage_config)

    def run(self, start_time):
        """ Run the training loop.

        Parameters
        ----------
        start_time: int
            Start time (in seconds since epoch) for measuring elapsed time for
            purposes of interrupting the training loop.

        """
        prepare_func = cfg.get("prepare_func", None)
        if callable(prepare_func):
            prepare_func()  # Modify the config in arbitrary ways before training

        self.curriculum = cfg.curriculum + []

        if cfg.seed is None or cfg.seed < 0:
            cfg.seed = gen_seed()

        if cfg.start_tensorboard:
            restart_tensorboard(cfg.log_dir, cfg.tbport, cfg.reload_interval)

        # Create a directory to store the results of the training session.
        es = ExperimentStore(cfg.log_dir, max_experiments=cfg.max_experiments, delete_old=1)
        exp_dir = es.new_experiment(
            self.exp_name, cfg.seed, add_date=1, force_fresh=1, update_latest=cfg.update_latest)
        self.exp_dir = exp_dir
        cfg.path = exp_dir.path

        if cfg.update_latest:
            make_symlink(exp_dir.path, os.path.join(os.getenv("HOME"), "dps-latest-experiment"))

        breaker = "-" * 40
        header = "{}\nREADME.md - {}\n{}\n\n\n".format(breaker, os.path.basename(exp_dir.path), breaker)
        readme = header + (cfg.readme if cfg.readme else "") + "\n\n"

        with open(exp_dir.path_for('README.md'), 'w') as f:
            f.write(readme)

        self.data = _TrainingLoopData(exp_dir)
        self.data.setup()

        frozen_data = None

        with ExitStack() as stack:
            # Tee stdout and stderr to files
            stack.enter_context(redirect_stream('stdout', self.data.path_for('stdout'), tee=cfg.tee))
            stack.enter_context(redirect_stream('stderr', self.data.path_for('stderr'), tee=cfg.tee))

            if start_time is None:
                start_time = time.time()
            self.start_time = start_time

            print("\n\n" + "=" * 80)
            print("Starting training run (name={}) at {}, {} seconds after given "
                  "start time.".format(self.exp_name, datetime.datetime.now(), time.time() - self.start_time))

            print("\nDirectory for this training run is {}.".format(exp_dir.path))

            stack.enter_context(NumpySeed(cfg.seed))
            print("\nSet numpy random seed to {}.\n".format(cfg.seed))

            limiter = time_limit(
                self.time_remaining, verbose=True,
                timeout_callback=lambda limiter: print("Training run exceeded its time limit."))

            try:
                with limiter:
                    self._run()

            finally:
                print(self.data.summarize())

                print("Done training run (name={}) at {}, {} seconds after given "
                      "start time.".format(self.exp_name, datetime.datetime.now(), time.time() - self.start_time))
                print("=" * 80)
                print("\n\n")

                frozen_data = self.data.freeze()

        return frozen_data

    def _run(self):
        print(cfg.to_string())

        threshold_reached = True
        self.global_step = 0
        self.n_global_experiences = 0
        self.curriculum_remaining = self.curriculum + []
        self.curriculum_complete = []

        stage_idx = 0
        while self.curriculum_remaining:
            print("\n" + "=" * 50)
            print("Starting stage {} at {}, {} seconds after given "
                  "start time.\n".format(stage_idx, datetime.datetime.now(), time.time() - self.start_time))
            print("\n")

            stage_config = self.curriculum_remaining.pop(0)
            stage_config = Config(stage_config)

            self.data.start_stage(stage_idx, stage_config)

            with ExitStack() as stack:

                # --------------- Stage set-up -------------------
                print("\n" + "-" * 10 + " Stage set-up " + "-" * 10)

                print("\nNew config values for this stage are: \n{}\n".format(pformat(stage_config)))
                stack.enter_context(stage_config)

                for hook in cfg.hooks:
                    assert isinstance(hook, Hook)
                    hook.start_stage(self, stage_idx)

                # Configure and create session and graph for stage.
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

                if cfg.use_gpu:
                    print("Using GPU if available.")
                    print("Using {}% of GPU memory.".format(
                        100 * session_config.gpu_options.per_process_gpu_memory_fraction))
                    print("Allowing growth of GPU memory: {}".format(session_config.gpu_options.allow_growth))

                graph = tf.Graph()
                sess = tf.Session(graph=graph, config=session_config)

                # This HAS to come after the creation of the session, otherwise
                # it allocates all GPU memory if using the GPU.
                print("\nAvailable devices: ")
                from tensorflow.python.client import device_lib
                print(device_lib.list_local_devices())
                print("\n")

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

                # Optionally build env
                if stage_idx == 0 or not cfg.preserve_env:
                    if getattr(self, 'env', None):
                        self.env.close()

                    self.env = cfg.build_env()

                # Build updater
                updater = cfg.get_updater(self.env)
                updater.stage_idx = stage_idx
                updater.exp_dir = self.exp_dir
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
                    updater.restore(sess, self.data.history[stage_idx-1]['best_path'])

                self.summary_op = tf.summary.merge_all()
                tf.train.get_or_create_global_step()
                sess.run(uninitialized_variables_initializer())
                sess.run(tf.assert_variables_initialized())

                threshold_reached = False
                reason = None

                try:
                    # --------------- Run stage -------------------

                    start = time.time()
                    phys_memory_before = memory_usage(physical=True)

                    threshold_reached, reason = self._run_stage(stage_idx, updater)

                except KeyboardInterrupt:
                    reason = "User interrupt"

                except NotImplementedError as e:
                    # There is a bug in pdb_postmortem that prevents instances of `NotImplementedError`
                    # from being handled properly, so replace it with an instance of `Exception`.
                    if cfg.robust:
                        traceback.print_exc()
                        reason = "Exception occurred ({})".format(e)
                    else:
                        raise Exception("NotImplemented") from e

                except Exception as e:
                    if cfg.robust:
                        traceback.print_exc()
                        reason = "Exception occurred ({})".format(e)
                    else:
                        raise

                except Alarm:
                    reason = "Time limit exceeded"
                    raise

                finally:
                    phys_memory_after = memory_usage(physical=True)
                    self.data.record_values_for_stage(
                        stage_duration=time.time()-start,
                        phys_memory_before_mb=phys_memory_before,
                        phys_memory_delta_mb=phys_memory_after - phys_memory_before
                    )

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

                    eval_results = updater.evaluate(cfg.batch_size)

                    print("\n" + "-" * 10 + " Final evaluation " + "-" * 10)
                    for mode, (record, _) in eval_results.items():
                        if record:
                            print("\n-- {} -- \n".format(mode))
                            for k, v in sorted(record.items()):
                                print("* {}: {}".format(k, v))
                    print()

                    # --------------- Optionally render performance of best hypothesis -------------------

                    if cfg.render_step > 0 and cfg.render_hook is not None:
                        print("Rendering...")
                        cfg.render_hook(updater)
                        print("Done rendering.")

                    # --------------- Finish up the stage -------------------

                    if cfg.start_tensorboard:
                        restart_tensorboard(cfg.log_dir, cfg.tbport, cfg.reload_interval)

                    self.data.end_stage(updater.n_updates)

                    print("\n" + "-" * 10 + " Running end-of-stage hooks " + "-" * 10 + "\n")
                    for hook in cfg.hooks:
                        hook.end_stage(self, stage_idx)

                    stage_idx += 1
                    self.curriculum_complete.append(stage_config)

                    print("\nDone stage {} at {}, {} seconds after given start time.".format(
                        stage_idx, datetime.datetime.now(), time.time() - self.start_time))
                    print("=" * 50)

                if not (threshold_reached or cfg.power_through):
                    print("Failed to reach stopping criteria threshold on stage {} "
                          "of the curriculum, terminating.".format(stage_idx))
                    break

    def _run_stage(self, stage_idx, updater):
        """ Run main training loop for a stage of the curriculum. """

        threshold_reached = False
        reason = "NotStarted"

        # Parse stopping criteria, set up early stopping
        stopping_criteria = cfg.get("stopping_criteria", None)
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

        early_stop = EarlyStopHook(patience=cfg.patience, maximize=self.maximize_sc)

        # Start stage
        print("{} seconds left at the beginning of stage {}.".format(self.time_remaining, stage_idx))
        print("\n" + "-" * 10 + " Training begins " + "-" * 10 + "\n")

        total_train_time = 0.0
        time_per_example = 0.0
        time_per_batch = 0.0

        while True:
            # Check whether to keep training
            if updater.n_updates >= cfg.max_steps:
                reason = "Maximum number of steps-per-stage reached"
                break

            if updater.n_experiences >= cfg.max_experiences:
                reason = "Maximum number of experiences-per-stage reached"
                break

            local_step = updater.n_updates
            global_step = self.global_step

            if local_step > 0 and local_step % cfg.checkpoint_step == 0:
                self.data.dump_data(local_step)

            evaluate = (local_step % cfg.eval_step) == 0
            display = (local_step % cfg.display_step) == 0
            render = (cfg.render_step > 0 and
                      (local_step % cfg.render_step) == 0 and
                      local_step > 0)

            # --------------- Perform an update -------------------

            update_start_time = time.time()

            if cfg.do_train:
                collect_summaries = evaluate and cfg.save_summaries

                _old_n_experiences = updater.n_experiences

                update_output = updater.update(
                    cfg.batch_size, collect_summaries=collect_summaries)

                n_experiences_delta = updater.n_experiences - _old_n_experiences
                self.n_global_experiences += n_experiences_delta

                self.data.store_step_data_and_summaries(
                    stage_idx, local_step, global_step,
                    updater.n_experiences, self.n_global_experiences,
                    **update_output)

            n_local_experiences = updater.n_experiences
            n_global_experiences = self.n_global_experiences

            update_duration = time.time() - update_start_time

            # --------------- Possibly evaluate -------------------

            if evaluate or display:
                eval_results = updater.evaluate(cfg.batch_size)

                self.data.store_step_data_and_summaries(
                    stage_idx, local_step, global_step,
                    n_local_experiences, n_global_experiences,
                    **eval_results)

                record = eval_results[cfg.eval_mode][0]

                stopping_criteria = record[self.stopping_criteria_name]
                new_best, stop = early_stop.check(stopping_criteria, local_step, record)

                if new_best:
                    print("Storing new best on step (l={}, g={}), "
                          "constituting (l={}, g={}) experiences, "
                          "with stopping criteria ({}) of {}.".format(
                              local_step, global_step,
                              n_local_experiences, n_global_experiences,
                              self.stopping_criteria_name, stopping_criteria))

                    best_path = self.data.path_for(
                        'weights/best_of_stage_{}'.format(stage_idx))
                    best_path = cfg.get('save_path', best_path)
                    best_path = updater.save(tf.get_default_session(), best_path)

                    self.data.record_values_for_stage(
                        best_path=best_path, best_global_step=global_step)
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
                    n_steps=local_step,
                    n_experiences=n_local_experiences,
                    epoch=updater.completion
                )

                if display:
                    print(self.data.summarize(local_step, global_step, n_local_experiences, n_global_experiences))
                    print("\nMy PID: {}".format(os.getpid()))
                    print("\nPhysical memory use: "
                          "{}mb".format(memory_usage(physical=True)))
                    print("Virtual memory use: "
                          "{}mb".format(memory_usage(physical=False)))
                    print("Avg time per batch: {}s".format(time_per_batch))
                    print("Most recent time per batch: {}s".format(update_duration))

                    if cfg.use_gpu:
                        print(nvidia_smi())

            for hook in cfg.hooks:
                if hook.call_per_timestep:
                    run_hook = local_step == 0 and hook.initial
                    run_hook |= local_step > 0 and local_step % hook.n == 0

                    if run_hook:
                        result = hook.step(self, updater)

                        if result:
                            # TODO: currently nothing is done with the record
                            record, summary = result
                            self.data.add_summary(n_global_experiences, hook.mode, summary)

            # Possibly render
            if render and cfg.render_hook is not None:
                print("Rendering...")
                cfg.render_hook(updater)
                print("Done rendering.")

            # If `do_train` is False, we do no training and evaluate
            # exactly once, so only one iteration is required.
            if not cfg.do_train:
                reason = "`do_train` set to False"
                break

            total_train_time += update_duration
            time_per_example = total_train_time / updater.n_experiences
            time_per_batch = total_train_time / updater.n_updates

            self.global_step += 1

        return threshold_reached, reason


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
        return os.listdir(self.path_for('summaries'))


class _TrainingLoopData(FrozenTrainingLoopData):
    """ Data structure used by a TrainingLoop to manage data
        throughout the experiment.

    """
    def setup(self):
        # Record training session environment for later diagnostic purposes
        self.record_environment(config=cfg.freeze(), git_modules=[dps])
        self.curriculum = []

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

    def start_stage(self, stage_idx, stage_config):
        self.history.append(dict(stage_idx=stage_idx, stage_config=stage_config))
        self.stage_idx = stage_idx
        self.summary_writers = {}

    def end_stage(self, local_step):
        self.dump_data(local_step)
        for writer in self.summary_writers.values():
            writer.close()

    def dirty(self):
        return any(bool(v) for v in self.data.values())

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

    def store_step_data_and_summaries(
            self, stage_idx, local_step, global_step, n_local_experiences, n_global_experiences, **data):

        for mode, (record, summary) in data.items():
            if record and cfg.store_step_data:
                record = record.copy()
                record.update(
                    local_step=local_step,
                    global_step=global_step,
                    n_local_experiences=n_local_experiences,
                    n_global_experiences=n_global_experiences,
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

    def _finalize(self):
        """ Write all stored data to disk. """
        assert not self.dirty(), "_TrainingLoopData is finalizing, but still contains data that needs to be dumped."

        with open(self.path_for('history.pkl'), 'wb') as f:
            dill.dump(self.history, f, protocol=dill.HIGHEST_PROTOCOL, recurse=False)

    def freeze(self):
        self._finalize()
        return FrozenTrainingLoopData(self.path)

    def summarize(self, *steps):
        """ Summarize the training data.

        Parameters
        ----------
        steps: quadtuple of ints
            local_step, global_step, local_experience, global_experiences

        """
        s = "\n"
        current_stage_only = bool(steps)
        if current_stage_only:
            local_step, global_step, n_local_experiences, n_global_experiences = steps
            history = [self.current_stage_record]
        else:
            s += "Stage-by-stage summary " + ">" * 30 + "\n"
            history = self.history

        for record in history:
            stage_idx = record['stage_idx']

            if current_stage_only:
                s += "\nStage={}, Step(l={}, g={}), Experiences(l={}, g={}): ".format(
                    stage_idx, local_step, global_step, n_local_experiences, n_global_experiences)
            else:
                s += "\nStage {} ".format(stage_idx)

            s += "*" * 30 + '\n'

            for k, v in sorted(record.items()):
                if isinstance(v, dict):
                    v = "\n" + pformat(v, indent=2)
                s += "* {}: {}\n".format(k, v)

        for mode, data in self.data.items():
            if data:
                record = data[-1]

                if record:
                    s += "\n-- {} -- \n".format(mode)
                    for k, v in sorted(record.items()):
                        if isinstance(v, dict):
                            v = "\n" + pformat(v, indent=2)
                        s += "* {}: {}\n".format(k, v)

        return s


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
    def __init__(self, n=None, mode=None, initial=False):
        self.n = n
        self.mode = mode
        self.initial = initial

    @property
    def call_per_timestep(self):
        return not (self.n is None or self.mode is None)

    def _attrs(self):
        return "n mode initial".split()

    def __str__(self):
        attr_string = ", ".join("{}={}".format(k, getattr(self, k)) for k in self._attrs())
        return("{}({})".format(self.__class__.__name__, attr_string))

    def __repr__(self):
        return str(self)

    def start_stage(self, training_loop, stage_idx):
        """ Called at the beginning of every stage. """
        pass

    def end_stage(self, training_loop, stage_idx):
        """ Called at the end of every stage. """
        pass

    def step(self, training_loop, updater, step_idx):
        """ May return a list of summaries and a dictionary of recorded values, similar to an updater. """
        pass

    def _print(self, s):
        print("{}: {}".format(self.__class__.__name__, s))


class ScheduleHook(Hook):
    def __init__(self, attr_name, query_name, initial_value=0.0, tolerance=0.05, base_configs=None):
        self.attr_name = attr_name
        self.query_name = query_name
        self.initial_value = initial_value

        if tolerance is None:
            tolerance = np.inf
        self.tolerance = tolerance

        if isinstance(base_configs, dict):
            base_configs = [base_configs]
        self.base_configs = base_configs or [{}]

        self.n_fragments_added = 0

        super(ScheduleHook, self).__init__()

    def _attrs(self):
        return "attr_name query_name initial_value tolerance base_configs".split()

    def _attr_value_for_fragment(self):
        raise Exception("NotImplemented")

    def start_stage(self, training_loop, stage_idx):
        if stage_idx == 0:
            self.final_orig_stage = len(training_loop.curriculum) - 1

    def end_stage(self, training_loop, stage_idx):

        if stage_idx >= self.final_orig_stage:
            attr_value = self._attr_value_for_fragment(self.n_fragments_added)
            new_stages = [{self.attr_name: attr_value, **bc} for bc in self.base_configs]

            if stage_idx == self.final_orig_stage:
                self.original_performance = training_loop.data.history[-1][self.query_name]
                self._print("End of original stages, adding 1st curriculum fragment:\n{}".format(new_stages))
                for i, ns in enumerate(new_stages):
                    training_loop.edit_remaining_stage(i, ns)
                self.n_fragments_added = 1

            elif not training_loop.curriculum_remaining:
                # Check whether performance achieved on most recent stage was near that of the first stage.
                current_stage_performance = training_loop.data.history[-1][self.query_name]
                threshold = (1 + self.tolerance) * self.original_performance

                self._print("End of {}-th curriculum fragment.".format(self.n_fragments_added))
                self._print("Original <{}>: {}".format(self.query_name, self.original_performance))
                self._print("<{}> for fragment {}: {}".format(
                    self.query_name, self.n_fragments_added, current_stage_performance))
                self._print("<{}> threshold: {}".format(self.query_name, threshold))

                if current_stage_performance <= threshold:
                    self.n_fragments_added += 1
                    self._print("Threshold reached, adding {}-th "
                                "curriculum fragment:\n{}".format(self.n_fragments_added, new_stages))

                    for i, ns in enumerate(new_stages):
                        training_loop.edit_remaining_stage(i, ns)

                else:
                    self._print("Threshold not reached, ending training run")
            else:
                self._print("In the middle of the {}-th curriculum fragment.".format(self.n_fragments_added))
        else:
            self._print("Still running initial stages.")


class GeometricScheduleHook(ScheduleHook):
    def __init__(self, *args, multiplier=2.0, **kwargs):
        super(GeometricScheduleHook, self).__init__(*args, **kwargs)
        self.multiplier = multiplier

    def _attrs(self):
        return super(GeometricScheduleHook, self)._attrs() + ["multiplier"]

    def _attr_value_for_fragment(self, fragment_idx):
        return self.initial_value * (self.multiplier ** fragment_idx)


class PolynomialScheduleHook(ScheduleHook):
    def __init__(self, *args, scale=10.0, power=1.0, **kwargs):
        super(PolynomialScheduleHook, self).__init__(*args, **kwargs)
        self.scale = scale
        self.power = power

    def _attrs(self):
        return super(PolynomialScheduleHook, self)._attrs() + ["scale", "power"]

    def _attr_value_for_fragment(self, fragment_idx):
        return self.initial_value + self.scale * (fragment_idx ** self.power)
