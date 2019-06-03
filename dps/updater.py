import abc
from future.utils import with_metaclass

import tensorflow as tf

from dps import cfg
from dps.utils import Parameterized, Param
from dps.utils.tf import build_gradient_train_op, trainable_variables, get_scheduled_values


class Updater(with_metaclass(abc.ABCMeta, Parameterized)):
    build_saver = True

    def __init__(self, env, scope=None, mpi_context=None, **kwargs):
        self.scope = scope
        self.env = env
        self.mpi_context = mpi_context
        self._n_experiences = 0
        self._n_updates = 0
        self._saver = None

    @property
    def n_experiences(self):
        return self._n_experiences

    @property
    def n_updates(self):
        return self._n_updates

    def build_graph(self):
        with tf.name_scope(self.scope or self.__class__.__name__) as scope:
            self._scope = scope

            self._build_graph()

            global_step = tf.train.get_or_create_global_step()
            self.inc_global_step_op = tf.assign_add(global_step, 1)
            global_step_input = tf.placeholder(tf.int64, ())
            assign_global_step = tf.assign(global_step, global_step_input)
            tf.get_default_session().run(assign_global_step, feed_dict={global_step_input: 0})

            if self.build_saver:
                updater_variables = {v.name: v for v in self.trainable_variables(for_opt=False)}
                self.saver = tf.train.Saver(updater_variables)

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
        path = self.saver.save(tf.get_default_session(), filename)
        return path

    def restore(self, session, path):
        self.saver.restore(tf.get_default_session(), path)


class DummyUpdater(Updater):
    """ For when you just want to build datasets. Much faster than most normal updaters. """

    build_saver = False

    def trainable_variables(self, for_opt):
        return []

    def _build_graph(self):
        pass

    def _update(self, batch_size):
        return dict(train={})

    def _evaluate(self, batch_size, mode):
        return dict()

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
    prefetch_buffer_size_in_batches = Param(10)
    prefetch_to_device = Param(False)

    train_initialized = False

    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, batch_size=1, **kwargs):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size

    def build_graph(self):
        sess = tf.get_default_session()

        datasets = []

        # --- train ---

        if self.train_dataset is not None:
            train_dataset = tf.data.TFRecordDataset(self.train_dataset.filename)

            try:
                shuffle_and_repeat_func = tf.data.experimental.shuffle_and_repeat
            except AttributeError:
                shuffle_and_repeat_func = tf.contrib.data.shuffle_and_repeat

            shuffle_and_repeat = shuffle_and_repeat_func(self.shuffle_buffer_size)
            train_dataset = (train_dataset.apply(shuffle_and_repeat)
                                          .batch(self.batch_size)
                                          .map(self.train_dataset.parse_example_batch))

            if self.prefetch_to_device:
                train_dataset = (train_dataset.apply(tf.data.experimental.copy_to_device('/gpu:0'))
                                              .prefetch(self.prefetch_buffer_size_in_batches))
                # prefetch = tf.data.experimental.prefetch_to_device('/gpu:0', self.prefetch_buffer_size_in_batches)
                # train_dataset = train_dataset.apply(prefetch)
            else:
                train_dataset = train_dataset.prefetch(self.prefetch_buffer_size_in_batches)

            datasets.append(train_dataset)

            # self.train_iterator = train_dataset.make_one_shot_iterator()
            self.train_iterator = train_dataset.make_initializable_iterator()
            self.train_handle = sess.run(self.train_iterator.string_handle(name="train_string_handle"))

        # --- val --

        if self.val_dataset is not None:
            val_dataset = tf.data.TFRecordDataset(self.val_dataset.filename)

            val_dataset = (val_dataset.batch(self.batch_size)
                                      .map(self.val_dataset.parse_example_batch))

            if self.prefetch_to_device:
                # Suggested here: https://github.com/tensorflow/tensorflow/issues/18947#issuecomment-407778515
                val_dataset = (val_dataset.apply(tf.data.experimental.copy_to_device('/gpu:0'))
                                          .prefetch(self.prefetch_buffer_size_in_batches))
                # prefetch = tf.data.experimental.prefetch_to_device('/gpu:0', self.prefetch_buffer_size_in_batches)
                # val_dataset = val_dataset.apply(prefetch)
            else:
                val_dataset = val_dataset.prefetch(self.prefetch_buffer_size_in_batches)

            datasets.append(val_dataset)

            self.val_iterator = val_dataset.make_initializable_iterator()
            self.val_handle = sess.run(self.val_iterator.string_handle(name="val_string_handle"))

        # --- test --

        if self.test_dataset is not None:
            test_dataset = tf.data.TFRecordDataset(self.test_dataset.filename)

            test_dataset = (test_dataset.batch(self.batch_size)
                                        .map(self.test_dataset.parse_example_batch))

            if self.prefetch_to_device:
                test_dataset = (test_dataset.apply(tf.data.experimental.copy_to_device('/gpu:0'))
                                            .prefetch(self.prefetch_buffer_size_in_batches))
                # prefetch = tf.data.experimental.prefetch_to_device('/gpu:0', self.prefetch_buffer_size_in_batches)
                # test_dataset = test_dataset.apply(prefetch)
            else:
                test_dataset = test_dataset.prefetch(self.prefetch_buffer_size_in_batches)

            datasets.append(test_dataset)

            self.test_iterator = test_dataset.make_initializable_iterator()
            self.test_handle = sess.run(self.test_iterator.string_handle(name="test_string_handle"))

        assert datasets, "Must supply at least one of train_dataset, val_dataset, test_dataset"
        dataset = datasets[0]

        # --- outputs ---

        self.handle = tf.placeholder(tf.string, shape=(), name="dataset_handle")

        if cfg.use_gpu and self.prefetch_to_device:
            # In tensorflow 1.13 (at least), tf wants to put this op on CPU, not sure why. This results in an error like:
            #
            # InvalidArgumentError: Attempted create an iterator on device "/job:localhost/replica:0/task:0/device:CPU:0"
            #                       from handle defined on device "/job:localhost/replica:0/task:0/device:GPU:0"
            #
            # And the error explicitly references IteratorFromStringHandleV2 built here. The reason is that the
            # resources that are pointed to by self.handle are all on the GPU, but, unless we are explicit,
            # the iterator created from that handle will be on the CPU, which is apparently not allowed.

            with tf.device("/gpu:0"):
                self.iterator = tf.data.Iterator.from_string_handle(
                    self.handle, dataset.output_types, dataset.output_shapes)
        else:
            self.iterator = tf.data.Iterator.from_string_handle(
                self.handle, dataset.output_types, dataset.output_shapes)

        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

    def do_train(self, is_training=True):
        if not self.train_initialized:
            sess = tf.get_default_session()
            sess.run(self.train_iterator.initializer)
            self.train_initialized = True
        return {self.handle: self.train_handle, self.is_training: is_training}

    def do_val(self, is_training=False):
        sess = tf.get_default_session()
        sess.run(self.val_iterator.initializer)
        return {self.handle: self.val_handle, self.is_training: is_training}

    def do_test(self, is_training=False):
        sess = tf.get_default_session()
        sess.run(self.test_iterator.initializer)
        return {self.handle: self.test_handle, self.is_training: is_training}
