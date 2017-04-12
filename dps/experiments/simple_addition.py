from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import time
from contextlib import ExitStack

import tensorflow as tf
import numpy as np

from spectral_dagger.utils.experiment import ExperimentStore

from dps import ProductionSystem, CoreNetwork, RegisterSpec, DifferentiableUpdater
from dps.environment import RegressionDataset, RegressionEnv
from dps.action_selection import (
    softmax_selection, gumbel_softmax_selection, relu_selection)
from dps.utils import (
    restart_tensorboard, Config, EarlyStopHook, default_config)


class AdditionDataset(RegressionDataset):
    def __init__(self, order, n_examples, for_eval=False, shuffle=True):
        self.order = order

        x = np.random.randn(n_examples, 2)
        x = np.concatenate((x.copy(), np.zeros((x.shape[0], 1))), axis=1)
        y = x.copy()
        for i in order:
            if i == 0:
                y[:, 0] = y[:, 0] + y[:, 1]
            else:
                y[:, 1] = y[:, 0] * y[:, 1]

        super(AdditionDataset, self).__init__(x, y, for_eval, shuffle)


class AdditionEnv(RegressionEnv):
    def __init__(self, order, n_train, n_val, n_test):
        super(AdditionEnv, self).__init__(
            train=AdditionDataset(order, n_train, for_eval=False),
            val=AdditionDataset(order, n_val, for_eval=True),
            test=AdditionDataset(order, n_test, for_eval=True))


# Define at top-level to enable pickling
addition_nt = namedtuple('AdditionRegSpec', 'r0 r1 r2'.split())


class AdditionRegSpec(RegisterSpec):
    @property
    def visible(self):
        return [1, 1, 1]

    @property
    def initial_values(self):
        return [np.array([v], dtype='f') for v in [1.0, 0.0, 0.0]]

    @property
    def namedtuple(self):
        return addition_nt

    @property
    def input_names(self):
        return self.names

    @property
    def output_names(self):
        return self.names


reg_spec = AdditionRegSpec()


def _inference(global_step):
    n_actions = 3
    config = default_config()

    def addition(action_activations, r):
        """ Action 0: add the variables in the registers, store in r0.
            Action 1: multiply the variables in the registers, store in r1.
            Action 2: no-op """
        if config.debug:
            action_activations = tf.Print(action_activations, [r], "registers", summarize=20)
            action_activations = tf.Print(
                action_activations, [action_activations], "action activations", summarize=20)

        a0, a1, a2 = tf.split(action_activations, n_actions, axis=1)
        r0 = a0 * (r.r0 + r.r1) + (1 - a0) * r.r0
        r1 = a1 * (r.r0 * r.r1) + (1 - a1) * r.r1

        if config.debug:
            r0 = tf.Print(r0, [r0], "r0", summarize=20)
            r1 = tf.Print(r1, [r1], "r1", summarize=20)
        new_registers = reg_spec.wrap(r0=r0, r1=r1, r2=r.r2+0)

        return action_activations, new_registers

    adder = CoreNetwork(n_actions=n_actions,
                        body=addition,
                        register_spec=reg_spec,
                        name="Addition")

    controller = tf.contrib.rnn.LSTMCell(num_units=config.lstm_size, num_proj=n_actions)

    start, decay_steps, decay_rate, staircase = config.exploration_schedule
    exploration = tf.train.exponential_decay(
        start, global_step, decay_steps, decay_rate, staircase=staircase)
    tf.summary.scalar('exploration', exploration)

    # ps_func = ProductionSystemFunction(psystem, exploration=exploration)
    # return ps_func
    psystem = ProductionSystem(adder, controller, config.action_selection, False, config.T)
    return psystem, exploration


class DefaultConfig(Config):
    seed = 10

    T = 4
    order = [0, 0, 0, 1]

    lstm_size = 64
    optimizer_class = tf.train.RMSPropOptimizer

    max_steps = 10000
    batch_size = 100
    n_train = 1000
    n_val = 100
    n_test = 0

    threshold = 1e-3
    patience = 100

    display_step = 100
    eval_step = 10
    checkpoint_step = 1000

    action_selection = staticmethod([
        softmax_selection,
        gumbel_softmax_selection(hard=0),
        relu_selection][1])

    # start, decay_steps, decay_rate, staircase
    lr_schedule = (0.1, 1000, 0.96, False)
    noise_schedule = (0.0, 10, 0.96, False)
    exploration_schedule = (10.0, 100, 0.96, False)

    max_grad_norm = 0.0

    debug = False


class DebugConfig(DefaultConfig):
    debug = True

    max_steps = 4
    n_train = 2
    batch_size = 2
    eval_step = 1
    display_step = 1
    checkpoint_step = 1
    exploration_schedule = (0.5, 100, 0.96, False)


def get_config(name):
    try:
        return dict(
            default=DefaultConfig(),
            debug=DebugConfig()
        )[name]
    except KeyError:
        raise KeyError("Unknown config name {}.".format(name))


def train(log_dir, config="default", max_experiments=5):
    es = ExperimentStore(log_dir, max_experiments=max_experiments, delete_old=1)
    exp_dir = es.new_experiment('', use_time=1, force_fresh=1)

    config = get_config(config)
    print(config)
    np.random.seed(config.seed)

    with open(exp_dir.path_for('config'), 'w') as f:
        f.write(str(config))

    env = AdditionEnv(config.order, config.n_train, config.n_val, config.n_test)

    batches_per_epoch = int(np.ceil(config.n_train / config.batch_size))
    max_epochs = int(np.ceil(config.max_steps / batches_per_epoch))

    early_stop = EarlyStopHook(patience=config.patience)
    val_loss = np.inf

    graph = tf.Graph()
    sess = tf.Session()
    with ExitStack() as stack:
        stack.enter_context(graph.as_default())
        stack.enter_context(sess)
        stack.enter_context(sess.as_default())
        stack.enter_context(config.as_default())

        global_step = tf.contrib.framework.get_or_create_global_step()
        psystem, exploration = _inference(global_step)
        updater = DifferentiableUpdater(env, psystem, exploration, global_step)

        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter(exp_dir.path_for('train'), sess.graph)
        val_writer = tf.summary.FileWriter(exp_dir.path_for('val'))
        print("Writing session summaries to {}.".format(exp_dir.path))

        sess.run(init)
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

    restart_tensorboard(log_dir)


def main(argv=None):
    from clify import command_line
    command_line(train)(log_dir='/tmp/dps/addition')


if __name__ == '__main__':
    tf.app.run()
