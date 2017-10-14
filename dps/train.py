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

from dps import cfg
from dps.utils import (
    gen_seed, time_limit, memory_usage, ExperimentStore,
    memory_limit, du, Config, ClearConfig, parse_date
)
from dps.utils.tf import (
    restart_tensorboard, uninitialized_variables_initializer,
    trainable_variables
)


def training_loop(exp_name='', start_time=None):
    np.random.seed(cfg.seed)

    exp_name = exp_name or cfg.get_experiment_name()
    try:
        curriculum = cfg.curriculum
    except AttributeError:
        curriculum = [{}]

    loop = TrainingLoop(curriculum, exp_name, start_time)
    return loop.run()


class TrainingLoop(object):
    def __init__(self, curriculum, exp_name='', start_time=None):
        self.curriculum = curriculum
        self.exp_name = exp_name
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
                if k in 'train_data update_data val_data'.split() and len(v) > 0:
                    if isinstance(v, pd.DataFrame):
                        s += "* {} (final_step): {}\n".format(k, v.iloc[-1].to_dict())
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
        except:
            if cfg.max_time is None or cfg.max_time <= 0:
                return np.inf
            else:
                return cfg.max_time - self.elapsed_time

    def run(self):
        print("CUDA_VISIBLE_DEVICES: ", os.getenv("CUDA_VISIBLE_DEVICES"))

        if cfg.start_tensorboard:
            restart_tensorboard(str(cfg.log_dir), cfg.tbport, cfg.reload_interval)

        value = self._run_core()

        if cfg.slim:
            print("`slim` is True, so deleting experiment directory {}.".format(self.exp_dir.path))
            print("Size of {} before delete: {}.".format(cfg.log_dir, du(cfg.log_dir)))
            try:
                shutil.rmtree(self.exp_dir.path)
            except FileNotFoundError:
                pass
            print("Size of {} after delete: {}.".format(cfg.log_dir, du(cfg.log_dir)))

        return value

    def _run_core(self):
        if self.start_time is None:
            self.start_time = time.time()
        print("Starting training {} seconds after given start time.".format(time.time() - self.start_time))
        print("Max experiments: {}".format(cfg.max_experiments))

        es = ExperimentStore(str(cfg.log_dir), max_experiments=cfg.max_experiments, delete_old=1)
        self.exp_dir = exp_dir = es.new_experiment(
            self.exp_name, add_date=1, force_fresh=1, update_latest=cfg.update_latest)

        print("Scratch directory is {}.".format(exp_dir.path))
        cfg.path = exp_dir.path

        print(cfg)

        with open(exp_dir.path_for('config.txt'), 'w') as f:
            f.write(str(cfg.freeze()))
        with open(exp_dir.path_for('config.pkl'), 'wb') as f:
            dill.dump(cfg.freeze(), f, protocol=dill.HIGHEST_PROTOCOL)

        threshold_reached = True
        self.global_step = 0

        for stage, stage_config in enumerate(self.curriculum):
            stage_config = Config(stage_config)

            if self.time_remaining <= 1:
                print("Time limit exceeded.")
                break

            stage_start = time.time()

            self.history.append(dict(stage=stage, train_data=[], val_data=[], update_data=[]))

            with ExitStack() as stack:
                session_config = tf.ConfigProto()
                session_config.gpu_options.allow_growth = True

                if cfg.use_gpu:

                    per_process_gpu_memory_fraction = getattr(cfg, 'per_process_gpu_memory_fraction', None)
                    if per_process_gpu_memory_fraction:
                        session_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
                        print("Using {}% of GPU memory.".format(100 * per_process_gpu_memory_fraction))

                    gpu_allow_growth = getattr(cfg, 'gpu_allow_growth', None)
                    if gpu_allow_growth:
                        session_config.gpu_options.allow_growth = gpu_allow_growth
                        print("Allowing growth of GPU memory: {}".format(gpu_allow_growth))

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

                if cfg.save_summaries:
                    self.train_writer = tf.summary.FileWriter(
                        exp_dir.path_for('train'), graph,
                        flush_secs=cfg.reload_interval)
                    self.update_writer = tf.summary.FileWriter(
                        exp_dir.path_for('update'), graph,
                        flush_secs=cfg.reload_interval)
                    self.val_writer = tf.summary.FileWriter(
                        exp_dir.path_for('val'), flush_secs=cfg.reload_interval)

                    print("Writing summaries to {}.".format(exp_dir.path))

                stack.enter_context(graph.as_default())
                stack.enter_context(sess)
                stack.enter_context(sess.as_default())

                memory_limit_mb = cfg.get("memory_limit_mb", None)
                if memory_limit_mb is not None:
                    stack.enter_context(memory_limit(cfg.memory_limit_mb))

                print("\nStarting stage {} of the curriculum at {}.\n"
                      "New config values for this stage are: \n{}\n".format(
                          stage, datetime.datetime.now(), pformat(stage_config)))

                stack.enter_context(stage_config)

                if stage == 0 or not cfg.preserve_env:
                    self.env = cfg.build_env()

                updater = cfg.get_updater(self.env)
                updater.build_graph()

                if stage > 0 and cfg.preserve_policy:
                    updater.restore(sess, self.history[-2]['best_path'])

                tf_seed = gen_seed()
                tf.set_random_seed(tf_seed)

                self.summary_op = tf.summary.merge_all()
                tf.contrib.framework.get_or_create_global_step()
                sess.run(uninitialized_variables_initializer())
                sess.run(tf.assert_variables_initialized())

                threshold_reached, reason = self.run_stage(stage, updater)

                self.record(reason=reason)

                print("Optimization complete. Reason: {}.".format(reason))
                print("Best hypothesis for this stage was found on "
                      "step (g: {best_global_step}, l: {best_local_step}) "
                      "with validation loss = {best_loss}.".format(**self.latest))

                best_path = self.latest['best_path']
                print("Loading best hypothesis for this stage "
                      "from file {}...".format(best_path))
                updater.restore(sess, best_path)
                test_loss, _, test_record = updater.evaluate(cfg.n_val, 'test')
                print("Results on test dataset: ")
                print("Test loss: {}".format(test_loss))
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
        result = dict(
            config=cfg.freeze(remove_callable=True),
            history=self.history,
            host=socket.gethostname()
        )

        return result

    def run_stage(self, stage, updater):
        threshold_reached = False
        reason = ""
        early_stop = EarlyStopHook(patience=cfg.patience)
        time_remaining = self.time_remaining

        print("{} seconds left "
              "at the beginning of stage {}.".format(time_remaining, stage))

        memory_before = memory_usage()

        with time_limit(self.time_remaining, verbose=True) as limiter:
            try:
                threshold_reached, reason = self._run_stage(stage, updater, early_stop)
            except KeyboardInterrupt:
                threshold_reached = False
                reason = "User interrupt"

        memory_after = memory_usage()

        self.record(
            stage_duration=limiter.elapsed_time,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_after - memory_before
        )

        self.latest['train_data'] = pd.DataFrame.from_records(self.latest['train_data'])
        self.latest['update_data'] = pd.DataFrame.from_records(self.latest['update_data'])
        self.latest['val_data'] = pd.DataFrame.from_records(self.latest['val_data'])

        if limiter.ran_out:
            reason = "Time limit reached"
        return threshold_reached, reason

    def _run_stage(self, stage, updater, early_stop):
        """ Run a stage of a curriculum. """

        self.local_step = 0
        threshold_reached = False
        val_loss = np.inf
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

            evaluate = self.global_step % cfg.eval_step == 0
            display = self.global_step % cfg.display_step == 0
            render = (self.global_step % cfg.render_step == 0) and self.global_step > 0

            start_time = time.time()
            train_summaries, update_summaries, train_record, update_record = updater.update(
                cfg.batch_size, collect_summaries=evaluate and cfg.save_summaries)
            update_duration = time.time() - start_time

            self.latest['train_data'].append(train_record)
            self.latest['update_data'].append(update_record)

            if evaluate or display:
                val_loss, val_summaries, val_record = updater.evaluate(cfg.n_val, 'val')

                self.latest['val_data'].append(val_record)

                if evaluate and cfg.save_summaries:
                    self.train_writer.add_summary(train_summaries, (self.global_step + 1) * cfg.batch_size)
                    self.update_writer.add_summary(update_summaries, (self.global_step + 1) * cfg.batch_size)
                    self.val_writer.add_summary(val_summaries, (self.global_step + 1) * cfg.batch_size)

                if cfg.stopping_function is not None:
                    stopping_criteria = cfg.stopping_function(val_record)
                else:
                    stopping_criteria = val_loss
                new_best, stop = early_stop.check(stopping_criteria, self.local_step, val_record)

                if new_best:
                    print("Storing new best on (local, global) step ({}, {}), "
                          "constituting {} local experiences, "
                          "with stopping criteria of {} and validation loss of {}.".format(
                              self.local_step, self.global_step, updater.n_experiences, stopping_criteria, val_loss))

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

                if stopping_criteria < cfg.threshold:
                    reason = "Stopping criteria threshold reached"
                    threshold_reached = True
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

            total_train_time += update_duration
            time_per_example = total_train_time / ((self.local_step+1) * cfg.batch_size)
            time_per_batch = total_train_time / (self.local_step+1)

            self.local_step += 1
            self.global_step += 1

        return threshold_reached, reason


class EarlyStopHook(object):
    def __init__(self, patience):
        self.patience = patience
        self.reset()

    def check(self, loss, step, record):
        new_best = self._best_loss is None or loss < self._best_loss
        if new_best:
            self._best_loss = loss
            self._best_step = step
            self._best_record = record.copy()

        self._early_stopped = (
            self._early_stopped or
            (step - self._best_step > self.patience))
        return new_best, self._early_stopped

    @property
    def best(self):
        best = self._best_record.copy()
        best.update(loss=self._best_loss, local_step=self._best_step)
        return best

    def reset(self):
        self._best_loss = None
        self._best_record = None
        self._best_step = None
        self._early_stopped = 0


def load_or_train(train_config, var_scope, path, sess=None):
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
        with ClearConfig():
            with train_config.copy(save_path=path):
                training_loop(var_scope.name)
        saver.restore(sess, path)
        print("Training successful.")
    return success
