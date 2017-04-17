from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
from contextlib import ExitStack

import tensorflow as tf
import numpy as np

from spectral_dagger.utils.experiment import ExperimentStore
from dps.utils import restart_tensorboard, EarlyStopHook, gen_seed


def _training_loop(
        env, build_updater, log_dir, config,
        max_experiments=5, start_tensorboard=True, exp_name=''):

    es = ExperimentStore(log_dir, max_experiments=max_experiments, delete_old=1)
    exp_dir = es.new_experiment(exp_name, use_time=1, force_fresh=1)

    print(config)

    with open(exp_dir.path_for('config'), 'w') as f:
        f.write(str(config))

    batches_per_epoch = int(np.ceil(config.n_train / config.batch_size))
    max_epochs = int(np.ceil(config.max_steps / batches_per_epoch))

    early_stop = EarlyStopHook(patience=config.patience)
    val_loss = np.inf

    tf.reset_default_graph()
    graph = tf.Graph()
    sess = tf.Session()
    with ExitStack() as stack:
        stack.enter_context(graph.as_default())
        stack.enter_context(sess)
        stack.enter_context(sess.as_default())
        stack.enter_context(config.as_default())

        tf.set_random_seed(gen_seed())

        updater = build_updater()

        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter(exp_dir.path_for('train'), sess.graph)
        val_writer = tf.summary.FileWriter(exp_dir.path_for('val'))
        print("Writing session summaries to {}.".format(exp_dir.path))

        sess.run(init)
        sess.run(tf.assert_variables_initialized())
        step = 1

        while True:
            n_epochs = updater.n_experiences / config.n_train
            if n_epochs >= max_epochs:
                print("Optimization complete, maximum number of epochs reached.")
                break
            if val_loss < config.threshold:
                print("Optimization complete, validation loss threshold reached.")
                break

            evaluate = step % config.eval_step == 0 or (step + 1) == config.max_steps
            display = step % config.display_step == 0 or (step + 1) == config.max_steps

            if evaluate or display:
                start_time = time.time()
                train_summary, train_loss, val_summary, val_loss = updater.update(
                    config.batch_size, summary_op if evaluate else None)
                duration = time.time() - start_time

                if evaluate:
                    train_writer.add_summary(train_summary, step)
                    val_writer.add_summary(val_summary, step)

                if display:
                    print("Step({}): Minibatch Loss={:06.4f}, Validation Loss={:06.4f}, "
                          "Minibatch Duration={:06.4f} seconds, Epoch={:04.2f}.".format(
                              step, train_loss, val_loss, duration, env.completion))

                new_best, stop = early_stop.check(val_loss)

                if new_best:
                    print("Storing new best on step {} with validation loss of {}.".format(step, val_loss))
                    checkpoint_file = exp_dir.path_for('best.checkpoint')
                    saver.save(sess, checkpoint_file, global_step=step)

                if stop:
                    print("Optimization complete, early stopping triggered.")
                    break
            else:
                updater.update(config.batch_size)

            if step % config.checkpoint_step == 0:
                print("Checkpointing on step {}.".format(step))
                checkpoint_file = exp_dir.path_for('model.checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

            step += 1


def training_loop(
        env, build_updater, log_dir, config,
        max_experiments=5, start_tensorboard=True, exp_name=''):

    try:
        _training_loop(
            env, build_updater, log_dir, config,
            max_experiments=5, exp_name='')
    finally:
        if start_tensorboard:
            restart_tensorboard(log_dir)
