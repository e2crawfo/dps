import abc
from future.utils import with_metaclass

import tensorflow as tf

from dps.utils import Parameterized, Param
from dps.utils.tf import build_gradient_train_op, trainable_variables, get_scheduled_values


class Updater(with_metaclass(abc.ABCMeta, Parameterized)):
    def __init__(self, env, scope=None, **kwargs):
        self.scope = scope
        self.env = env
        self._n_experiences = 0
        self._n_updates = 0

    @property
    def n_experiences(self):
        return self._n_experiences

    @property
    def n_updates(self):
        return self._n_updates

    @property
    def completion(self):
        return self.env.completion

    def build_graph(self):
        with tf.name_scope(self.scope or self.__class__.__name__) as scope:
            self._scope = scope

            self._build_graph()

            global_step = tf.train.get_or_create_global_step()
            self.inc_global_step_op = tf.assign_add(global_step, 1)
            global_step_input = tf.placeholder(tf.int64, ())
            assign_global_step = tf.assign(global_step, global_step_input)
            tf.get_default_session().run(assign_global_step, feed_dict={global_step_input: 0})

    @abc.abstractmethod
    def _build_graph(self):
        raise Exception("NotImplemented")

    def update(self, batch_size):
        update_result = self._update(batch_size)

        sess = tf.get_default_session()
        sess.run(self.inc_global_step_op)
        self._n_experiences += batch_size
        self._n_updates += 1

        return update_result

    @abc.abstractmethod
    def _update(self, batch_size):
        raise Exception("NotImplemented")

    def evaluate(self, batch_size, mode="val"):
        assert mode in "val test".split()
        return self._evaluate(batch_size, mode)

    @abc.abstractmethod
    def _evaluate(self, batch_size, mode):
        raise Exception("NotImplemented")

    def trainable_variables(self, for_opt):
        raise Exception("AbstractMethod")

    def save(self, session, filename):
        updater_variables = {v.name: v for v in self.trainable_variables(for_opt=False)}
        saver = tf.train.Saver(updater_variables)
        path = saver.save(tf.get_default_session(), filename)
        return path

    def restore(self, session, path):
        updater_variables = {v.name: v for v in self.trainable_variables(for_opt=False)}
        saver = tf.train.Saver(updater_variables)
        saver.restore(tf.get_default_session(), path)


class DummyUpdater(Updater):
    """ For when you just want to build datasets. Much faster than most normal updaters. """

    def trainable_variables(self, for_opt):
        return []

    def _build_graph(self):
        pass

    def _update(self, batch_size):
        return dict(train={})

    def _evaluate(self, batch_size, mode):
        return {}, b''

    def save(self, session, filename):
        return ''

    def restore(self, session, path):
        pass


class DifferentiableUpdater(Updater):
    """ Update parameters of a differentiable function `f` using gradient-based algorithm.

    Must be used in context of a default graph, session and config.

    Parameters
    ----------
    env: gym Env
        The environment we're trying to learn about.
    f: An instance of ScopedFunction
        Accepts a tensor (input), returns a tensor (inference).

    """
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()
    l2_weight = Param(None)
    stopping_criteria = "loss,min"

    def __init__(self, env, f, **kwargs):
        assert hasattr(env, 'build'), (
            "Environments used with DifferentiableUpdater must possess "
            "a method called `build` which builds returns a dictionary of scalar tensors."
        )
        self.f = f

        super(DifferentiableUpdater, self).__init__(env, **kwargs)

    def trainable_variables(self, for_opt):
        return trainable_variables(self.f.scope, for_opt=for_opt)

    def _build_graph(self):
        self.recorded_tensors = self.env.build(self.f)
        self.loss = self.recorded_tensors['loss']

        tvars = self.trainable_variables(for_opt=True)
        if self.l2_weight is not None:
            self.loss += self.l2_weight * sum(tf.nn.l2_loss(v) for v in tvars if 'weights' in v.name)

        self.train_op, self.train_recorded_tensors = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.recorded_tensors.update(get_scheduled_values())

    def _update(self, batch_size):
        feed_dict = self.env.data_manager.do_train()

        sess = tf.get_default_session()
        _, record, train_record = sess.run(
            [self.train_op, self.recorded_tensors, self.train_recorded_tensors], feed_dict=feed_dict)
        record.update(train_record)

        return dict(train=record)

    def _evaluate(self, batch_size, mode):
        if mode == "val":
            feed_dict = self.env.data_manager.do_val()
        elif mode == "test":
            feed_dict = self.env.data_manager.do_test()
        else:
            raise Exception("Unknown evaluation mode: {}".format(mode))

        sess = tf.get_default_session()
        return sess.run(self.recorded_tensors, feed_dict=feed_dict)


class DataManager(Parameterized):
    shuffle_buffer_size = Param(1000)

    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, **kwargs):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size

    def build_graph(self):
        # --- train ---

        train_dataset = tf.data.TFRecordDataset(self.train_dataset.filename)

        shuffle_and_repeat = tf.contrib.data.shuffle_and_repeat(self.shuffle_buffer_size)
        train_dataset = (train_dataset.apply(shuffle_and_repeat)
                                      .batch(self.batch_size)
                                      .map(self.train_dataset.parse_example_batch)
                                      .prefetch(10))

        self.train_iterator = train_dataset.make_one_shot_iterator()

        sess = tf.get_default_session()
        self.train_handle = sess.run(self.train_iterator.string_handle())

        # --- val --

        val_dataset = tf.data.TFRecordDataset(self.val_dataset.filename)

        val_dataset = (val_dataset.batch(self.batch_size)
                                  .map(self.val_dataset.parse_example_batch)
                                  .prefetch(10))

        self.val_iterator = val_dataset.make_initializable_iterator()

        self.val_handle = sess.run(self.val_iterator.string_handle())

        # --- test --

        test_dataset = tf.data.TFRecordDataset(self.test_dataset.filename)

        test_dataset = (test_dataset.batch(self.batch_size)
                                    .map(self.test_dataset.parse_example_batch)
                                    .prefetch(10))

        self.test_iterator = test_dataset.make_initializable_iterator()

        self.test_handle = sess.run(self.test_iterator.string_handle())

        # --- outputs ---

        self.handle = tf.placeholder(tf.string, shape=())
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, train_dataset.output_types, train_dataset.output_shapes)
        self.is_training = tf.placeholder(tf.bool, shape=())

    def do_train(self):
        return {self.handle: self.train_handle, self.is_training: True}

    def do_val(self):
        sess = tf.get_default_session()
        sess.run(self.val_iterator.initializer)
        return {self.handle: self.val_handle, self.is_training: False}

    def do_test(self):
        sess = tf.get_default_session()
        sess.run(self.test_iterator.initializer)
        return {self.handle: self.test_handle, self.is_training: False}
