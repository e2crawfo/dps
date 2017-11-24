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
import dill
import os
import socket
import pandas as pd
from pathlib import Path
import copy

from dps import cfg
from dps.utils import (
    gen_seed, time_limit, memory_usage, ExperimentStore, dps_git_summary,
    memory_limit, du, Config, ClearConfig, parse_date, redirect_stream,
    NumpySeed
)
from dps.utils.tf import (
    restart_tensorboard, uninitialized_variables_initializer,
    trainable_variables
)


def training_loop(exp_name='', start_time=None):
    loop = TrainingLoop(exp_name, start_time)
    yield from loop.run()


class TrainingLoop(object):
    def __init__(self, exp_name='', start_time=None):
        self.exp_name = exp_name or cfg.get_experiment_name()
        self.start_time = start_time
        self.history = []

    def record(self, d=None, **kwargs):
        d = d or {}
        self.history[-1].update(d)
        self.history[-1].update(kwargs)

    @property
    def latest(self):
        return self.history[-1]

    def summarize(self, latest=False):
        s = "\n"
        if latest:
            history = [self.latest]
        else:
            history = self.history

        for stage, d in enumerate(history):
            if latest:
                s += "Step(g: {}, l: {}): ".format(self.global_step, self.local_step)
            else:
                s += "Stage {} ".format(stage)

            s += "*" * 30 + '\n'

            items = sorted(d.items(), key=lambda x: x[0])
            for k, v in items:
                if k in 'train_data update_data val_data test_data'.split() and len(v) > 0:
                    if isinstance(v, pd.DataFrame):
                        s += "* {} (final_step): {}\n".format(k, v.iloc[-1].to_dict())
                    elif isinstance(v, str):
                        try:
                            lines = [l for l in v.split('\n') if l]
                            keys = lines[0].split(',')
                            values = lines[-1].split(',')
                            d = {_k: _v for _k, _v in zip(keys, values)}
                            s += "* {} (final_step): {}\n".format(k, d)
                        except IndexError:
                            pass
                    else:
                        s += "* {} (final_step): {}\n".format(k, v[-1])
                else:
                    s += "* {}: {}\n".format(k, v)
            if not latest:
                s += "* new config values: {}\n\n".format(pformat(self.curriculum[stage]))
        return s

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def time_remaining(self):
        """ Prioritize `deadline`, fall back on `max_time`. """
        try:
            deadline = parse_date(cfg.deadline)
            return (deadline - datetime.datetime.now()).total_seconds()
        except Exception:
            if cfg.max_time is None or cfg.max_time <= 0:
                return np.inf
            else:
                return cfg.max_time - self.elapsed_time

    def _build_return_value(self):
        history = copy.deepcopy(self.history)
        for kind in 'train update val test'.split():
            key = kind + '_data'
            if key in self.latest:
                history[-1][key] = pd.DataFrame.from_records(self.latest[key]).to_csv(index=False)

        result = dict(
            config=cfg.freeze(remove_callable=True),
            history=history,
            host=socket.gethostname()
        )
        return result

    def run(self):
        self.curriculum = cfg.curriculum

        if cfg.seed is None or cfg.seed < 0:
            cfg.seed = gen_seed()

        with NumpySeed(cfg.seed):
            yield from self._run()

    def _run(self):
        if cfg.start_tensorboard:
            restart_tensorboard(str(cfg.log_dir), cfg.tbport, cfg.reload_interval)
        if self.start_time is None:
            self.start_time = time.time()
        print("Starting training {} seconds after given start time.".format(time.time() - self.start_time))

        es = ExperimentStore(str(cfg.log_dir), max_experiments=cfg.max_experiments, delete_old=1)
        self.exp_dir = exp_dir = es.new_experiment(
            self.exp_name, cfg.seed, add_date=1, force_fresh=1, update_latest=cfg.update_latest)

        print("Scratch directory for this training run is {}.".format(exp_dir.path))
        cfg.path = exp_dir.path

        print(cfg)

        with open(exp_dir.path_for('config.pkl'), 'wb') as f:
            dill.dump(cfg.freeze(), f, protocol=dill.HIGHEST_PROTOCOL)

        with open(exp_dir.path_for('config.txt'), 'w') as f:
            f.write(str(cfg.freeze()))

        git_summary = dps_git_summary()
        with open(exp_dir.path_for('git_summary.txt'), 'w') as f:
            f.write(git_summary.summarize(diff=True))

        with open(exp_dir.path_for('os_environ.txt'), 'w') as f:
            f.write(pformat(os.environ._data))

        threshold_reached = True
        self.global_step = 0

        for stage, stage_config in enumerate(self.curriculum):
            stage_config = Config(stage_config)

            if self.time_remaining <= 1:
                print("Time limit exceeded.")
                break

            stage_start = time.time()

            self.history.append(dict(stage=stage, train_data=[], update_data=[], val_data=[], test_data=[]))

            with ExitStack() as stack:
                session_config = tf.ConfigProto()

                if cfg.use_gpu:
                    per_process_gpu_memory_fraction = getattr(cfg, 'per_process_gpu_memory_fraction', None)
                    if per_process_gpu_memory_fraction:
                        session_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction

                    gpu_allow_growth = getattr(cfg, 'gpu_allow_growth', None)
                    if gpu_allow_growth:
                        session_config.gpu_options.allow_growth = gpu_allow_growth

                graph = tf.Graph()
                sess = tf.Session(graph=graph, config=session_config)

                print("Available devices: ")
                print(device_lib.list_local_devices())
                print("\n")

                if not cfg.use_gpu:
                    print("Not using GPU.")
                    stack.enter_context(graph.device("/cpu:0"))
                else:
                    print("Using GPU if available.")
                    print("Using {}% of GPU memory.".format(100 * session_config.gpu_options.per_process_gpu_memory_fraction))
                    print("Allowing growth of GPU memory: {}".format(session_config.gpu_options.allow_growth))

                if cfg.save_summaries:
                    self.train_writer = tf.summary.FileWriter(
                        exp_dir.path_for('train'), graph,
                        flush_secs=cfg.reload_interval)
                    self.update_writer = tf.summary.FileWriter(
                        exp_dir.path_for('update'), flush_secs=cfg.reload_interval)
                    self.val_writer = tf.summary.FileWriter(
                        exp_dir.path_for('val'), flush_secs=cfg.reload_interval)
                    self.test_writer = tf.summary.FileWriter(
                        exp_dir.path_for('test'), flush_secs=cfg.reload_interval)

                    print("Writing summaries to {}.".format(exp_dir.path))

                stack.enter_context(graph.as_default())
                stack.enter_context(sess)
                stack.enter_context(sess.as_default())

                tf_seed = gen_seed()
                tf.set_random_seed(tf_seed)

                memory_limit_mb = cfg.get("memory_limit_mb", None)
                if memory_limit_mb is not None:
                    stack.enter_context(memory_limit(cfg.memory_limit_mb))

                print("\nStarting stage {} of the curriculum at {}.\n"
                      "New config values for this stage are: \n{}\n".format(
                          stage, datetime.datetime.now(), pformat(stage_config)))

                stack.enter_context(stage_config)

                if stage == 0 or not cfg.preserve_env:
                    if getattr(self, 'env', None):
                        self.env.close()

                    self.env = cfg.build_env()

                updater = cfg.get_updater(self.env)
                updater.build_graph()

                if cfg.load_path:
                    if isinstance(cfg.load_path, list):
                        repeat = getattr(cfg, 'repeat', 0)
                        path = cfg.load_path[repeat % len(cfg.load_path)]
                    else:
                        path = cfg.load_path
                    path = os.path.realpath(path)
                    assert isinstance(path, str)
                    print("Loading hypothesis from {}.".format(path))
                    updater.restore(sess, path)
                elif stage > 0 and cfg.preserve_policy:
                    updater.restore(sess, self.history[-2]['best_path'])

                self.summary_op = tf.summary.merge_all()
                tf.train.get_or_create_global_step()
                sess.run(uninitialized_variables_initializer())
                sess.run(tf.assert_variables_initialized())

                yield from self.run_stage(stage, updater)
                threshold_reached, reason = self.threshold_reached, self.reason

                self.record(reason=reason)

                print("Optimization complete. Reason: {}.".format(reason))
                print("Best hypothesis for this stage was found on "
                      "step (g: {best_global_step}, l: {best_local_step}) "
                      "with stopping criteria ({sc_name}) of {best_stopping_criteria}.".format(
                          sc_name=updater.stopping_criteria, **self.latest))

                best_path = self.latest['best_path']
                print("Loading best hypothesis for this stage "
                      "from file {}...".format(best_path))
                updater.restore(sess, best_path)

                _, test_record = updater.evaluate(cfg.n_val, 'test')
                print("Results on test dataset: ")
                print(test_record)
                self.record(**{'test_' + k: v for k, v in test_record.items()})

                print(self.summarize(latest=True))

                if cfg.start_tensorboard:
                    restart_tensorboard(
                        str(cfg.log_dir), cfg.tbport, cfg.reload_interval)

                if cfg.render_hook is not None:
                    cfg.render_hook(updater)

                if not (threshold_reached or cfg.power_through):
                    print("Failed to reach error threshold on stage {} "
                          "of the curriculum, terminating.".format(stage))
                    break

                self.record(stage_duration=time.time()-stage_start)

        print(self.summarize(latest=False))

        if cfg.slim:
            print("`slim` is True, so deleting experiment directory {}.".format(self.exp_dir.path))
            print("Size of {} before delete: {}.".format(cfg.log_dir, du(cfg.log_dir)))
            try:
                shutil.rmtree(self.exp_dir.path)
            except FileNotFoundError:
                pass
            print("Size of {} after delete: {}.".format(cfg.log_dir, du(cfg.log_dir)))

        yield self._build_return_value()

    def run_stage(self, stage, updater):
        self.threshold_reached = False
        self.reason = ""
        early_stop = EarlyStopHook(patience=cfg.patience, maximize=updater.maximize)
        time_remaining = self.time_remaining

        print("{} seconds left "
              "at the beginning of stage {}.".format(time_remaining, stage))

        phys_memory_before = memory_usage(physical=True)

        with time_limit(self.time_remaining, verbose=True) as limiter:
            try:
                yield from self._run_stage(stage, updater, early_stop)
            except KeyboardInterrupt:
                self.threshold_reached = False
                self.reason = "User interrupt"

        phys_memory_after = memory_usage(physical=True)

        self.record(
            stage_duration=limiter.elapsed_time,
            phys_memory_before_mb=phys_memory_before,
            phys_memory_after_mb=phys_memory_after,
            phys_memory_delta_mb=phys_memory_after - phys_memory_before
        )

        if limiter.ran_out:
            self.reason = "Time limit exceeded"
            if cfg.error_on_timeout:
                raise Exception("Timed out.")

    def _run_stage(self, stage, updater, early_stop):
        """ Run a stage of a curriculum. """

        self.local_step = 0
        threshold_reached = False
        reason = None
        total_train_time = 0.0
        time_per_example = 0.0
        time_per_batch = 0.0

        while True:
            if self.local_step >= cfg.max_steps:
                reason = "Maximum number of steps reached"
                break

            if updater.n_experiences >= cfg.max_experiences:
                reason = "Maximum number of experiences reached"
                break

            if self.local_step % cfg.checkpoint_step == 0:
                yield self._build_return_value()

            evaluate = self.local_step % cfg.eval_step == 0
            display = self.local_step % cfg.display_step == 0
            render = (self.local_step % cfg.render_step == 0) and self.local_step > 0

            start_time = time.time()
            train_summaries = b""
            update_summaries = b""
            train_record = {}
            update_record = {}
            if cfg.do_train:
                train_summaries, update_summaries, train_record, update_record = updater.update(
                    cfg.batch_size, collect_summaries=evaluate and cfg.save_summaries)
            update_duration = time.time() - start_time

            self.latest['train_data'].append(train_record)
            self.latest['update_data'].append(update_record)

            if evaluate or display:
                val_summaries, val_record = updater.evaluate(cfg.n_val, 'val')
                self.latest['val_data'].append(val_record)

                test_summaries, test_record = updater.evaluate(cfg.n_val, 'test')
                self.latest['test_data'].append(test_record)

                if evaluate and cfg.save_summaries:
                    self.train_writer.add_summary(train_summaries, (self.global_step + 1) * cfg.batch_size)
                    self.update_writer.add_summary(update_summaries, (self.global_step + 1) * cfg.batch_size)
                    self.val_writer.add_summary(val_summaries, (self.global_step + 1) * cfg.batch_size)
                    self.test_writer.add_summary(test_summaries, (self.global_step + 1) * cfg.batch_size)

                stopping_criteria = val_record[updater.stopping_criteria]
                new_best, stop = early_stop.check(stopping_criteria, self.local_step, val_record)

                if new_best:
                    print("Storing new best on (local, global) step ({}, {}), "
                          "constituting {} local experiences, "
                          "with stopping criteria ({}) of {}.".format(
                              self.local_step, self.global_step, updater.n_experiences,
                              updater.stopping_criteria, stopping_criteria))

                    try:
                        path = cfg.save_path
                        assert path
                    except (AttributeError, AssertionError):
                        path = self.exp_dir.path_for('best_of_stage_{}'.format(stage))
                    best_path = updater.save(tf.get_default_session(), path)

                    self.record(best_path=best_path, best_global_step=self.global_step)
                    self.record(**{'best_' + k: v for k, v in early_stop.best.items()})

                if stop:
                    print("Early stopping triggered.")
                    reason = "Early stopping triggered"
                    break

                if updater.maximize:
                    threshold_reached = stopping_criteria > cfg.threshold
                else:
                    threshold_reached = stopping_criteria < cfg.threshold
                if threshold_reached:
                    reason = "Stopping criteria threshold reached"
                    break

                self.record(
                    time_per_example=time_per_example,
                    time_per_batch=time_per_batch,
                    n_steps=self.local_step,
                    n_experiences=self.local_step*cfg.batch_size,
                    epoch=updater.env.completion
                )

                if display:
                    print(self.summarize(latest=True))

            if render and cfg.render_hook is not None:
                cfg.render_hook(updater)

            if not cfg.do_train:
                reason = "`do_train` set to False"
                break

            total_train_time += update_duration
            time_per_example = total_train_time / ((self.local_step+1) * cfg.batch_size)
            time_per_batch = total_train_time / (self.local_step+1)

            if self.global_step % 100 == 0:
                print("\nPhysical memory use: {}mb".format(memory_usage(physical=True)))
                print("Virtual memory use: {}mb".format(memory_usage(physical=False)))

            self.local_step += 1
            self.global_step += 1

        self.threshold_reached = threshold_reached
        self.reason = reason


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


def load_or_train(train_config, var_scope, path, sess=None, redirect=False):
    """ Attempts to load variables into ``var_scope`` from checkpoint stored at ``path``.

    If said checkpoint is not found, trains a model using the function
    ``train`` and stores the resulting variables for future use.

    Returns True iff model was successfully loaded, False otherwise.

    """
    sess = sess or tf.get_default_session()

    to_be_loaded = trainable_variables(var_scope.name)
    saver = tf.train.Saver(var_list=to_be_loaded)

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    success = False
    try:
        print("Trying to load variables for variable scope {} "
              "from checkpoint {}...".format(var_scope.name, path))
        saver.restore(sess, path)
        success = True
        print("Load successful.")
    except tf.errors.NotFoundError:
        print("Loading failed, training a model...")

        with ExitStack() as stack:
            if redirect:
                stem = os.path.splitext(str(path))[0]
                stack.enter_context(redirect_stream('stdout', stem + '.stdout', tee=True))
                stack.enter_context(redirect_stream('stderr', stem + '.stderr', tee=True))

            stack.enter_context(ClearConfig())
            stack.enter_context(train_config.copy(save_path=path))

            list(training_loop(var_scope.name))

        saver.restore(sess, path)
        print("Training successful.")
    return success
