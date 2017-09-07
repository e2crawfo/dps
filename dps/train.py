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

from spectral_dagger.utils.experiment import ExperimentStore
from dps import cfg
from dps.utils import (
    restart_tensorboard, gen_seed, time_limit, memory_usage,
    uninitialized_variables_initializer, du, Config, parse_date)


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

    def summarize(self):
        s = "\n"
        for stage, d in enumerate(self.history):
            s += "Stage {} ".format(stage) + "*" * 30 + '\n'
            items = sorted(d.items(), key=lambda x: x[0])
            for k, v in items:
                s += "* {}: {}\n".format(k, v)
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
            self.exp_name, use_time=1, force_fresh=1, update_latest=cfg.update_latest)

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

            if stage > 0:
                stage_config.init_steps = 0

            if self.time_remaining <= 1:
                print("Time limit exceeded.")
                break

            stage_start = time.time()

            self.history.append(dict(stage=stage))

            with ExitStack() as stack:
                graph = tf.Graph()

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
                    self.val_writer = tf.summary.FileWriter(
                        exp_dir.path_for('val'), flush_secs=cfg.reload_interval)

                    print("Writing summaries to {}.".format(exp_dir.path))

                sess = tf.Session(graph=graph)

                stack.enter_context(graph.as_default())
                stack.enter_context(sess)
                stack.enter_context(sess.as_default())

                print("\nStarting stage {} of the curriculum at {}.\n"
                      "New config values for this stage are: \n{}\n".format(
                          stage, datetime.datetime.now(), pformat(stage_config)))

                stack.enter_context(stage_config)

                self.env = env = cfg.build_env()
                updater = cfg.get_updater(env)
                updater.build_graph()

                if stage > 0 and cfg.preserve_policy:
                    updater.restore(
                        tf.get_default_session(), self.history[-2]['best_path'])

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

                s = "~" * 40
                for k, v in self.latest.items():
                    s += '\n{}: {}'.format(k, v)
                s += "\n" + "~" * 40
                print(s)

                best_path = self.latest['best_path']
                print("Loading best hypothesis for this stage "
                      "from file {}...".format(best_path))
                updater.restore(tf.get_default_session(), best_path)

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

        print(self.summarize())
        result = dict(
            config=cfg.freeze(),
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

        if limiter.ran_out:
            reason = "Time limit reached"
        return threshold_reached, reason

    def _run_stage(self, stage, updater, early_stop):
        """ Run a stage of a curriculum. """

        local_step = 0
        threshold_reached = False
        val_loss = np.inf
        reason = None
        total_train_time = 0.0
        time_per_example = 0.0
        time_per_batch = 0.0

        while True:
            if local_step >= cfg.max_steps:
                reason = "Maximum number of steps reached"
                break

            evaluate = self.global_step % cfg.eval_step == 0
            display = self.global_step % cfg.display_step == 0
            render = (self.global_step % cfg.render_step == 0) and self.global_step > 0

            start_time = time.time()
            update_summaries = updater.update(
                cfg.batch_size, collect_summaries=evaluate)
            update_duration = time.time() - start_time

            if evaluate or display:
                _, train_summaries, train_record = \
                    updater.evaluate(cfg.batch_size, mode='train_eval')
                val_loss, val_summaries, val_record = \
                    updater.evaluate(cfg.batch_size, mode='val')

                record = {k + '(train)': v for k, v in train_record.items()}
                record.update({k + '(val)': v for k, v in val_record.items()})

                if evaluate and cfg.save_summaries:
                    self.train_writer.add_summary(
                        update_summaries + train_summaries, updater.n_experiences)
                    self.val_writer.add_summary(val_summaries, updater.n_experiences)

                if cfg.stopping_function is not None:
                    stopping_criteria = cfg.stopping_function(val_record)
                else:
                    stopping_criteria = val_loss
                new_best, stop = early_stop.check(stopping_criteria, local_step, **record)

                if new_best:
                    print("Storing new best on (local, global) step ({}, {}), "
                          "constituting {} local experiences, "
                          "with validation loss of {}.".format(
                              local_step, self.global_step, updater.n_experiences, val_loss))

                    filename = self.exp_dir.path_for('best_of_stage_{}'.format(stage))
                    best_path = updater.save(tf.get_default_session(), filename)

                    self.record(best_path=best_path, best_global_step=self.global_step)
                    self.record(**{'best_' + k: v for k, v in early_stop.best.items()})

                if stop:
                    print("Early stopping triggered.")
                    reason = "Early stopping triggered"
                    break

                if val_loss < cfg.threshold:
                    reason = "Validation loss threshold reached"
                    threshold_reached = True
                    break

                self.record(
                    time_per_example=time_per_example,
                    time_per_batch=time_per_batch,
                    n_steps=local_step,
                    epoch=updater.env.completion
                )

                if display:
                    s = "~" * 40
                    s += "\nStep(g: {}, l: {}): ".format(self.global_step, local_step)
                    for k, v in self.latest.items():
                        s += '\n{}: {}'.format(k, v)
                    s += "\n" + "~" * 40
                    print(s)

            if render and cfg.render_hook is not None:
                cfg.render_hook(updater)

            total_train_time += update_duration
            time_per_example = total_train_time / ((local_step+1) * cfg.batch_size)
            time_per_batch = total_train_time / (local_step+1)

            local_step += 1
            self.global_step += 1

        return threshold_reached, reason


class EarlyStopHook(object):
    def __init__(self, patience):
        self.patience = patience
        self.reset()

    def check(self, loss, step, **kwargs):
        new_best = self._best_loss is None or loss < self._best_loss
        if new_best:
            self._best_loss = loss
            self._best_step = step
            self._best_record = kwargs.copy()

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
