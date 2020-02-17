import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import numpy as np
import time

from dps import cfg
from dps.datasets.base import Dataset
from dps.train import TrainingLoop, TrainingLoopData
from dps.utils import Parameterized, Param, flush_print as _print, gen_seed, map_structure, timed_block
from dps.utils.pytorch import walk_variable_scopes, to_np


pytorch_device = None


def set_pytorch_device(device):
    global pytorch_device
    pytorch_device = device


def get_pytorch_device():
    return pytorch_device


class PyTorchTrainingLoopData(TrainingLoopData):
    def store_scalar_summaries(self, mode, path, record, n_global_experiences):
        if mode not in self.writers:
            self.writers[mode] = SummaryWriter(path)

        for k, v in record.items():
            self.writers[mode].add_scalar("all/"+k, float(v), n_global_experiences)


class PyTorchTrainingLoop(TrainingLoop):
    training_loop_data_class = PyTorchTrainingLoopData

    def framework_initialize_stage(self, stack):
        # Set the seed for the stage.
        torch_seed = gen_seed()
        _print("Setting pytorch seed to generated seed: {}\n".format(torch_seed))
        torch.manual_seed(torch_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if cfg.use_gpu:
            _print("Using GPU.")

            _print("Device count: {}".format(torch.cuda.device_count()))
            device = torch.cuda.current_device()
            _print("Device idx: {}".format(device))
            _print("Device name: {}".format(torch.cuda.get_device_name(device)))
            _print("Device capability: {}".format(torch.cuda.get_device_capability(device)))

            set_pytorch_device('cuda')
        else:
            _print("Not using GPU.")
            set_pytorch_device('cpu')

    def framework_finalize_stage_initialization(self):
        self.framework_print_variables()
        self.framework_load_weights()

    def framework_print_variables(self):
        walk_variable_scopes(self.updater.model, max_depth=cfg.variable_scope_depth)

    def framework_load_weights(self):
        """
        Adapted from the tensorflow version, roughly treats a pytorch module as equivalant
        to a tensorflow variable scope. Currently, similar to the tensorflow version, This
        assumes that that all loaded modules lie on the same path in the on disk file as
        they do in the current module.

        """
        for module_path, path in self.get_load_paths():
            _print("Loading submodule \"{}\" from {}.".format(module_path, path))

            start = time.time()

            device = get_pytorch_device()

            loaded_state_dict = torch.load(path, map_location=device)['model']

            if module_path:
                loaded_state_dict = type(loaded_state_dict)(
                    {k: v for k, v in loaded_state_dict.items() if k.startswith(module_path)}
                )

            module = self.updater.model

            state_dict = module.state_dict()
            state_dict.update(loaded_state_dict)

            module.load_state_dict(state_dict)

            _print("Done loading weights for module {}, took {} seconds.".format(module_path, time.time() - start))


class PyTorchUpdater(Parameterized):
    build_model = Param()
    lr_schedule = Param()
    build_optimizer = Param()

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
        print("Building model...")
        self.model = self.build_model(env=self.env)

        print("Moving model to device...")
        self.model.to(get_pytorch_device())

        print("Building optimizer...")
        self.optimizer = self.build_optimizer(self.model.parameters(), lr=self.lr_schedule)

        print("Building data manager...")
        data_manager = self.data_manager = PyTorchDataManager(datasets=self.env.datasets)
        data_manager.build_graph()
        self.train_iterator = data_manager.do_train()

        self._build_graph()

    def _build_graph(self):
        pass

    def update(self, batch_size):
        self.model.train()

        data = next(self.train_iterator)
        tensors, recorded_tensors, losses = self.model(data, plot=False, is_training=True)

        # --- loss ---

        loss = 0.0
        for name, tensor in losses.items():
            loss += tensor
            recorded_tensors['loss_' + name] = tensor
        recorded_tensors['loss'] = loss

        print_time = self._n_updates % 100 == 0
        with timed_block('zero_grad', print_time):
            self.optimizer.zero_grad()

        with timed_block('loss backward', print_time):
            loss.backward()

        with timed_block('optimizer step', print_time):
            self.optimizer.step()

        update_result = self._update(batch_size)

        if isinstance(update_result, dict):
            recorded_tensors.update(update_result)

        self._n_experiences += batch_size
        self._n_updates += 1

        return recorded_tensors

    def _update(self, batch_size):
        pass

    def evaluate(self, batch_size, mode="val"):
        assert mode in "val test".split()

        self.model.eval()
        if mode == 'val':
            data_iterator = self.data_manager.do_val()
        elif mode == 'test':
            data_iterator = self.data_manager.do_test()
        else:
            raise Exception("Unknown data mode: {}".format(mode))

        n_points = 0
        n_batches = 0
        record = None

        start = time.time()

        for data in data_iterator:

            with torch.no_grad():
                tensors, recorded_tensors, losses = self.model.evaluate(data)

                loss = 0.0
                for name, tensor in losses.items():
                    loss += tensor
                    recorded_tensors['loss_' + name] = tensor
                recorded_tensors['loss'] = loss

                recorded_tensors = map_structure(
                    lambda t: to_np(t) if isinstance(t, torch.Tensor) else t,
                    recorded_tensors, is_leaf=lambda rec: not isinstance(rec, dict))

            batch_size = recorded_tensors['batch_size']

            if record is None:
                record = recorded_tensors
            else:
                record = map_structure(
                    lambda rec, rec_t: rec + batch_size * np.mean(rec_t), record, recorded_tensors,
                    is_leaf=lambda rec: not isinstance(rec, dict))

            n_points += batch_size
            n_batches += 1

        record = map_structure(
            lambda rec: rec / n_points, record, is_leaf=lambda rec: not isinstance(rec, dict))

        record['eval_duration_per_item'] = (time.time() - start) / n_points
        record['eval_duration_per_batch'] = (time.time() - start) / n_batches
        print("Evaluation took {} seconds per item, {} seconds per batch.".format(
            record['eval_duration_per_item'], record['eval_duration_per_batch']))

        return record

    def save(self, path):
        whole_dict = dict(model=self.model.state_dict())
        torch.save(whole_dict, path)
        return path

    def restore(self, path):
        device = get_pytorch_device()
        whole_dict = torch.load(path, map_location=device)
        state = self.model.state_dict()
        state.update(whole_dict['model'])
        self.model.load_state_dict(state)


class PyTorchDataManager(Parameterized):
    shuffle_buffer_size = Param()
    prefetch_buffer_size_in_batches = Param()
    batch_size = Param()

    def __init__(self, train=None, val=None, test=None, datasets=None, **kwargs):
        self.datasets = {}
        self.datasets.update(train=train, val=val, test=test)
        self.datasets.update(datasets)

        assert (
            self.datasets['train'] is not None
            or self.datasets['val'] is not None
            or self.datasets['test'] is not None), (
                'Must provide at least one dataset with name "train", "val", or "test".')

        self.iterators = {}
        self.train_initialized = False

        self.graph = tf.Graph()

        # Don't use GPU at all for this. There's no point in using the GPU here,
        # as the data is going to come back through the CPU anyway.
        session_config = tf.ConfigProto()
        session_config.device_count['GPU'] = 0
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.0
        self.sess = tf.Session(graph=self.graph, config=session_config)

    def build_graph(self):
        tf_dsets = []

        with self.graph.as_default(), self.graph.device("/cpu:0"):
            train_dataset = self.datasets.get('train', None)
            if train_dataset is not None:
                train_dset, __ = self.build_iterator('train', 'train', self.batch_size, True, self.shuffle_buffer_size)
                tf_dsets.append(train_dset)

            val_dataset = self.datasets.get('val', None)
            if val_dataset is not None:
                val_dset, _ = self.build_iterator('val', 'val', self.batch_size, False, 0)
                tf_dsets.append(val_dset)

            test_dataset = self.datasets.get('test', None)
            if test_dataset is not None:
                test_dset, _ = self.build_iterator('test', 'test', self.batch_size, False, 0)
                tf_dsets.append(test_dset)

        self.graph.finalize()

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
            dset = dset.prefetch(self.prefetch_buffer_size_in_batches)

        # --- finalize ---

        tf_iterator = dset.make_initializable_iterator()
        iterator = DataIterator(self.sess, tf_iterator)
        self.iterators[name] = iterator

        return dset, iterator

    def do_train(self, is_training=True):
        return self.do('train', is_training)

    def do_val(self, is_training=False):
        return self.do('val', is_training)

    def do_test(self, is_training=False):
        return self.do('test', is_training)

    def do(self, name, is_training=False):
        """ Initialize iterator (unless it's the `train` iterator, which is handled slightly differently)
            and return a feed_dict populated with the appropriate handle for the requested iterator. """

        iterator = self.iterators[name]

        if name == 'train':
            if not self.train_initialized:
                iterator.reset(is_training)
                self.train_initialized = True
        else:
            iterator.reset(is_training)
        return iterator


class DataIterator:
    def __init__(self, sess, tf_iterator):
        self.sess = sess
        self.tf_iterator = tf_iterator
        self.get_next_op = self.tf_iterator.get_next()
        self.is_training = None

    def reset(self, is_training):
        self.sess.run(self.tf_iterator.initializer)
        self.is_training = is_training

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.sess.run(self.get_next_op)
            result['is_training'] = True
        except tf.errors.OutOfRangeError:
            raise StopIteration("TensorFlow iterator raised tf.errors.OutOfRangeError.")

        device = get_pytorch_device()

        result = map_structure(
            lambda v: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v,
            result, is_leaf=lambda v: not isinstance(v, dict)
        )

        return result