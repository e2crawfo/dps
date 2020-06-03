import abc
import json
from future.utils import with_metaclass
from collections import defaultdict

import numpy as np
import tensorflow as tf

from dps import cfg
from dps.utils import Parameterized, Param
from dps.utils.tf import build_gradient_train_op, trainable_variables, get_scheduled_values, ScopedFunction
from dps.datasets.base import Dataset


class Updater(with_metaclass(abc.ABCMeta, Parameterized)):
    build_saver = True

    def __init__(self, env, scope=None, mpi_context=None, **kwargs):
        self.scope = scope
        self.env = env
        self.mpi_context = mpi_context
        self._n_experiences = 0
        self.step = 0
        self._saver = None

    @property
    def n_experiences(self):
        return self._n_experiences

    def build_graph(self):
        # with tf.name_scope(self.scope or self.__class__.__name__) as scope:
        #     self._scope = scope

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

    def update(self, batch_size, step):
        update_result = self._update(batch_size)

        sess = tf.get_default_session()
        sess.run(self.inc_global_step_op)
        self._n_experiences += batch_size

        return update_result

    @abc.abstractmethod
    def _update(self, batch_size):
        raise Exception("NotImplemented")

    def evaluate(self, batch_size, step, mode="val"):
        assert mode in "val test".split()
        return self._evaluate(batch_size, mode)

    @abc.abstractmethod
    def _evaluate(self, batch_size, mode):
        raise Exception("NotImplemented")

    def trainable_variables(self, for_opt):
        raise Exception("AbstractMethod")

    def save(self, filename):
        path = self.saver.save(tf.get_default_session(), filename)
        return path

    def restore(self, path):
        self.saver.restore(tf.get_default_session(), path)


class DummyUpdater(Updater):
    """ For when you just want to build datasets. Much faster than most normal updaters. """

    build_saver = False

    def trainable_variables(self, for_opt):
        return []

    def _build_graph(self):
        pass

    def _update(self, batch_size):
        return dict()

    def _evaluate(self, batch_size, mode):
        return dict()

    def save(self, session, filename):
        return ''

    def restore(self, path):
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

        return record

    def _evaluate(self, batch_size, mode):
        if mode == "val":
            feed_dict = self.env.data_manager.do_val()
        elif mode == "test":
            feed_dict = self.env.data_manager.do_test()
        else:
            raise Exception("Unknown evaluation mode: {}".format(mode))

        sess = tf.get_default_session()
        return sess.run(self.recorded_tensors, feed_dict=feed_dict)


class VideoUpdater(Updater):
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()
    grad_n_record_groups = Param(None)

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.obs_shape
        *other, self.image_height, self.image_width, self.image_depth = self.obs_shape
        self.n_frames = other[0] if other else 0
        self.network = cfg.build_network(env, self, scope="network")

        super(VideoUpdater, self).__init__(env, scope=scope, **kwargs)

    def trainable_variables(self, for_opt):
        return self.network.trainable_variables(for_opt)

    def _update(self, batch_size):
        if cfg.get('no_gradient', False):
            return dict()

        feed_dict = self.data_manager.do_train()

        sess = tf.get_default_session()
        _, record, train_record = sess.run(
            [self.train_op, self.recorded_tensors, self.train_records], feed_dict=feed_dict)
        record.update(train_record)

        return record

    def _evaluate(self, _batch_size, mode):
        return self.evaluator.eval(self.recorded_tensors, self.data_manager, mode)

    def _build_graph(self):
        self.data_manager = DataManager(datasets=self.env.datasets)
        self.data_manager.build_graph()

        data = self.data_manager.iterator.get_next()
        self.inp = data["image"]
        network_outputs = self.network(data, self.data_manager.is_training)

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]

        self.tensors = network_tensors

        self.recorded_tensors = recorded_tensors = dict(global_step=tf.train.get_or_create_global_step())

        # --- loss ---

        self.loss = tf.constant(0., tf.float32)
        for name, tensor in network_losses.items():
            self.loss += tensor
            recorded_tensors['loss_' + name] = tensor
        recorded_tensors['loss'] = self.loss

        # --- train op ---

        if cfg.do_train and not cfg.get('no_gradient', False):
            tvars = self.trainable_variables(for_opt=True)

            self.train_op, self.train_records = build_gradient_train_op(
                self.loss, tvars, self.optimizer_spec, self.lr_schedule,
                self.max_grad_norm, self.noise_schedule, grad_n_record_groups=self.grad_n_record_groups)

        sess = tf.get_default_session()
        for k, v in getattr(sess, 'scheduled_values', None).items():
            if k in recorded_tensors:
                recorded_tensors['scheduled_' + k] = v
            else:
                recorded_tensors[k] = v

        # --- recorded values ---

        intersection = recorded_tensors.keys() & network_recorded_tensors.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)
        recorded_tensors.update(network_recorded_tensors)

        intersection = recorded_tensors.keys() & self.network.eval_funcs.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)

        if self.network.eval_funcs:
            eval_funcs = self.network.eval_funcs
        else:
            eval_funcs = {}

        # For running functions, during evaluation, that are not implemented in tensorflow
        self.evaluator = Evaluator(eval_funcs, network_tensors, self)


class TensorRecorder(ScopedFunction):
    _recorded_tensors = None

    def record_tensors(self, **kwargs):
        for k, v in kwargs.items():
            self.recorded_tensors[k] = tf.reduce_mean(tf.to_float(v))

    @property
    def recorded_tensors(self):
        if self._recorded_tensors is None:
            self._recorded_tensors = {}
        return self._recorded_tensors


class DataManager(Parameterized):
    """ Manages a collection of datasets (of type dps/datasets/base.py:Dataset) and iterators accessing them.

        Datasets of type Dataset are passed into the constructor. At least one of those must be called
        'train', 'val' or 'test'. When build_graph is called, iterators accessing those datasets
        are created, and a special string-handle iterator is created. (Note: an iterator is a tensorflow
        operations which is used to stream data from a file stored on disk). The string-handle iterator
        can switch between datasets; which dataset it accesses is controlled by the value of a string tensor.
        This allows us to build a single model (i.e. a single tensorflow graph), but feed it different data.
        For example, we can easily switch from feeding the model training data to feeding it evaluation data.

        Note: all datasets collected under a single DataManager instance must return data with the same structure.
        (i.e. they should have the same set of Features; see dps/datasets/base.py:Dataset).

        Convenience functions do_train, do_val and do_test are provided. When called, they return feed_dicts
        when can be used to set the string handle to the appropriate value for the desired dataset.

        Additional iterators can be provided by directly calling `build_iterator`, after `build_graph` has
        been called. Indeed, this MUST be done in order to access datasets other than 'train', 'val', 'test',
        as `build_graph` does not create iterators for these non-standard datasets.

        Example use:

            dm = DataManager(
                train=MyTrainDataset(),
                val=MyValDataset(),
                test=MyTestDataset(),
            )
            input_data = dm.iterator.get_next()

        The form of input_data will depend on the Features of the datasets; most often if will be a dictionary of tensors.

    """
    shuffle_buffer_size = Param()

    prefetch_buffer_size_in_batches = Param(10)
    prefetch_to_device = Param(False)

    batch_size = Param()

    train_initialized = False

    def __init__(self, train=None, val=None, test=None, datasets=None, **kwargs):
        self.datasets = {}
        self.datasets.update(train=train, val=val, test=test)
        self.datasets.update(datasets)

        assert (
            self.datasets['train'] is not None
            or self.datasets['val'] is not None
            or self.datasets['test'] is not None), (
                'Must provide at least one dataset with name "train", "val", or "test".')

        self.iterators_and_handles = {}

    def build_graph(self):
        tf_dsets = []

        train_dataset = self.datasets.get('train', None)
        if train_dataset is not None:
            train_dset, _, _ = self.build_iterator('train', 'train', self.batch_size, True, self.shuffle_buffer_size)
            tf_dsets.append(train_dset)

        val_dataset = self.datasets.get('val', None)
        if val_dataset is not None:
            val_dset, _, _ = self.build_iterator('val', 'val', self.batch_size, False, 0)
            tf_dsets.append(val_dset)

        test_dataset = self.datasets.get('test', None)
        if test_dataset is not None:
            test_dset, _, _ = self.build_iterator('test', 'test', self.batch_size, False, 0)
            tf_dsets.append(test_dset)

        # --- outputs ---

        self.handle = tf.placeholder(tf.string, shape=(), name="dataset_handle")
        tf_dset = tf_dsets[0]

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
                    self.handle, tf_dset.output_types, tf_dset.output_shapes)
        else:
            self.iterator = tf.data.Iterator.from_string_handle(
                self.handle, tf_dset.output_types, tf_dset.output_shapes)

        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

    def build_iterator(self, name, base_dataset_name, batch_size, repeat, shuffle_buffer_size):
        base_dataset = self.datasets[base_dataset_name]

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(base_dataset, tf.data.Dataset):
            dset = base_dataset
        elif isinstance(base_dataset, Dataset):
            dset = tf.data.TFRecordDataset(base_dataset.filename)
        else:
            raise Exception("Unknown dataset type: {}.".format(base_dataset))

        # --- possibly repeat and/or shuffle --

        if repeat and shuffle_buffer_size > 0:
            try:
                shuffle_and_repeat_func = tf.data.experimental.shuffle_and_repeat
            except AttributeError:
                shuffle_and_repeat_func = tf.contrib.data.shuffle_and_repeat
            shuffle_and_repeat = shuffle_and_repeat_func(self.shuffle_buffer_size)

            dset = dset.apply(shuffle_and_repeat)

        elif shuffle_buffer_size > 0:
            dset = dset.shuffle(self.shuffle_buffer_size)

        # --- batch and parse ---

        dset = dset.batch(batch_size)

        if hasattr(base_dataset, 'parse_example_batch'):
            dset = dset.map(base_dataset.parse_example_batch)

        # --- possibly prefetch to improve performance ---

        if self.prefetch_buffer_size_in_batches > 0:
            if cfg.use_gpu and self.prefetch_to_device:
                # Suggested here: https://github.com/tensorflow/tensorflow/issues/18947#issuecomment-407778515
                dset = (dset.apply(tf.data.experimental.copy_to_device('/gpu:0'))
                            .prefetch(self.prefetch_buffer_size_in_batches))
            else:
                dset = dset.prefetch(self.prefetch_buffer_size_in_batches)

        # --- finalize ---

        iterator = dset.make_initializable_iterator()

        sess = tf.get_default_session()
        handle = sess.run(iterator.string_handle(name="{}_string_handle".format(name)))

        self.iterators_and_handles[name] = (iterator, handle)

        return dset, iterator, handle

    def do_train(self, is_training=True):
        return self.do('train', is_training)

    def do_val(self, is_training=False):
        return self.do('val', is_training)

    def do_test(self, is_training=False):
        return self.do('test', is_training)

    def do(self, name, is_training=False):
        """ Initialize iterator (unless it's the `train` iterator, which is handled slightly differently)
            and return a feed_dict populated with the appropriate handle for the requested iterator. """
        iterator, handle = self.iterators_and_handles[name]

        sess = tf.get_default_session()

        if name == 'train':
            if not self.train_initialized:
                sess.run(iterator.initializer)
                self.train_initialized = True
        else:
            sess.run(iterator.initializer)

        return {self.handle: handle, self.is_training: is_training}


class DummyFunc:
    keys_accessed = ""

    def __call__(self, fetched, updater):
        return {}


class Evaluator:
    """ A helper object for running a list of functions on a collection of evaluated tensors.

    Parameters
    ----------
    functions: a dict (name-> function). Each function as assumed to have an attribute `keys_accessed`
               listing the keys (into `tensors`) that will be accessed by that function.
    tensors: a (possibly nested) dictionary of tensors which will provide the input to the functions
    updater: the updater object, passed into the functions at eval time

    """
    def __init__(self, functions, tensors, updater):
        self._functions = functions
        self._tensors = tensors

        # Force evaluation to happen at with the default feed_dict
        functions["dummy"] = DummyFunc()

        self.updater = updater

        self.functions = defaultdict(list)
        self.feed_dicts = {}
        fetch_keys = defaultdict(set)

        for name, func in functions.items():
            if hasattr(func, 'get_feed_dict'):
                feed_dict = func.get_feed_dict(updater)
            else:
                feed_dict = {}

            fd_key = {str(k): str(v) for k, v in feed_dict.items()}
            fd_key = json.dumps(fd_key, default=str, indent=4, sort_keys=True)

            self.functions[fd_key].append((name, func))
            self.feed_dicts[fd_key] = feed_dict

            # store for the function

            keys_accessed = func.keys_accessed

            if isinstance(keys_accessed, str):
                keys_accessed = keys_accessed.split()

            for key in keys_accessed:
                fetch_keys[fd_key].add(key)

        self.fetches = {}

        for fd_key, _fetch_keys in fetch_keys.items():
            fetches = self.fetches[fd_key] = {}

            for key in _fetch_keys:
                dst = fetches
                src = tensors
                subkeys = key.split(":")

                for i, _key in enumerate(subkeys):
                    if i == len(subkeys)-1:
                        dst[_key] = src[_key]
                    else:
                        if _key not in dst:
                            dst[_key] = dict()
                        dst = dst[_key]
                        src = src[_key]

    def _check_continue(self, record):
        return True

    def eval(self, recorded_tensors, data_manager, mode):
        final_record = {}

        for key, functions in self.functions.items():
            if mode == "val":
                feed_dict = data_manager.do_val()
            elif mode == "test":
                feed_dict = data_manager.do_test()
            else:
                raise Exception("Unknown evaluation mode: {}".format(mode))

            extra_feed_dict = self.feed_dicts[key]
            feed_dict.update(extra_feed_dict)

            sess = tf.get_default_session()

            n_points = 0
            record = defaultdict(float)
            fetches = self.fetches.get(key, {})

            while True:
                try:
                    if extra_feed_dict:
                        _recorded_tensors = dict(batch_size=recorded_tensors['batch_size'])
                        _record, fetched = sess.run([_recorded_tensors, fetches], feed_dict=feed_dict)
                    else:
                        # Only get values from recorded_tensors when using the default feed dict.
                        _record, fetched = sess.run([recorded_tensors, fetches], feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break

                for name, func in functions:
                    result = func(fetched, self.updater)

                    if isinstance(result, dict):
                        for k, v in result.items():
                            _record["{}:{}".format(name, k)] = np.mean(v)
                    else:
                        _record[name] = np.mean(result)

                batch_size = _record['batch_size']

                # Assumes that each record entry is an average over the batch
                for k, v in _record.items():
                    record[k] += batch_size * v

                n_points += batch_size

                do_continue = self._check_continue(_record)
                if not do_continue:
                    break

            record = {k: v / n_points for k, v in record.items()}

            intersection = record.keys() & final_record.keys() - set(['batch_size'])
            assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)

            final_record.update(record)

        return final_record
