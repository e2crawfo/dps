import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import numpy as np
import time
from torch._six import inf
import matplotlib.pyplot as plt
import os
import shutil
import traceback as tb

from dps import cfg
from dps.datasets.base import Dataset
from dps.train import TrainingLoop, TrainingLoopData
from dps.utils import (
    AttrDict, Parameterized, Param, flush_print as _print, gen_seed, map_structure, timed_block
)
from dps.utils.pytorch import walk_variable_scopes, to_np, describe_structure, GradNormRecorder, pad_and_concatenate


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

        for k, v in AttrDict(record).flatten().items():
            self.writers[mode].add_scalar("all/"+k, float(v), n_global_experiences)


class PyTorchTrainingLoop(TrainingLoop):
    training_loop_data_class = PyTorchTrainingLoopData

    def framework_initialize_stage(self, stack):
        # Set the seed for the stage.
        torch_seed = gen_seed()
        _print("Setting pytorch seed to generated seed: {}\n".format(torch_seed))
        torch.manual_seed(torch_seed)

        torch.backends.cudnn.enabled = True

        torch.backends.cudnn.benchmark = cfg.pytorch_cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.pytorch_cudnn_deterministic

        if cfg.use_gpu:
            _print("Trying to use GPU...")
            try:
                device = torch.cuda.current_device()
                use_gpu = True
            except AssertionError:
                tb.print_exc()
                use_gpu = False
        else:
            use_gpu = False

        if use_gpu:
            _print("Using GPU.")

            _print("Device count: {}".format(torch.cuda.device_count()))
            _print("Device idx: {}".format(device))
            _print("Device name: {}".format(torch.cuda.get_device_name(device)))
            _print("Device capability: {}".format(torch.cuda.get_device_capability(device)))

            set_pytorch_device('cuda')
        else:
            _print("Not using GPU.")
            set_pytorch_device('cpu')

        torch.set_printoptions(profile='full')

    def framework_finalize_stage_initialization(self):
        self.framework_print_variables()
        self.framework_load_weights()

        if cfg.initial_step is not None and cfg.initial_step > 0:
            self.updater.model.update_global_step(cfg.initial_step)

    def framework_print_variables(self):
        walk_variable_scopes(self.updater.model, max_depth=cfg.variable_scope_depth)

    def framework_load_weights(self):
        """
        Adapted from the tensorflow version, roughly treats a pytorch module as equivalant
        to a tensorflow variable scope.

        Most general form a dictionary entry is: {"<dest_module_path>": "<source_module_path>:<file_path>"}
        Maps tensors located at module path `source_module_path` in file `file_path` to module path `dest_module_path`
        in the current model.

        """
        omit_modules = cfg.get('omit_modules_from_loading', [])

        for dest_module_path, path in self.get_load_paths():
            _print("Loading submodule \"{}\" from {}.".format(dest_module_path, path))

            if ":" in path:
                source_module_path, source_path = path.split(':')
            else:
                source_path = path
                source_module_path = dest_module_path

            start = time.time()

            device = get_pytorch_device()

            loaded_state_dict = torch.load(source_path, map_location=device)['model']

            if source_module_path:
                source_module_path_with_sep = source_module_path + '.'

                loaded_state_dict = type(loaded_state_dict)(
                    {k: v for k, v in loaded_state_dict.items() if k.startswith(source_module_path_with_sep)}
                )

                assert loaded_state_dict, (
                    f"File contains no tensors with prefix `{source_module_path_with_sep}` (file: {source_path})"
                )

            if dest_module_path != source_module_path:
                # Rename variables from the loaded state dict by replacing `source_module_path` with `dest_module_path`.

                _source_module_path = source_module_path + '.' if source_module_path else source_module_path
                _dest_module_path = dest_module_path + '.' if dest_module_path else dest_module_path

                loaded_state_dict = {
                    k.replace(_source_module_path, _dest_module_path, 1): v
                    for k, v in loaded_state_dict.items()
                }

            module = self.updater.model

            state_dict = module.state_dict()

            intersection = set(state_dict.keys()) & set(loaded_state_dict.keys())

            if not intersection:
                raise Exception(
                    f"Loading variables with spec ({dest_module_path}, {path}) "
                    f"would have no effect (no variables found)."
                )
            loaded_state_dict = {k: loaded_state_dict[k] for k in intersection}

            if omit_modules:
                omitted_variables = {
                    k: v for k, v in loaded_state_dict.items()
                    if any(k.startswith(o) for o in omit_modules)
                }

                print("Omitting the following variables from loading:")
                describe_structure(omitted_variables)

                loaded_state_dict = {
                    k: v for k, v in loaded_state_dict.items()
                    if k not in omitted_variables
                }

            _print("Loading variables:")
            describe_structure(loaded_state_dict)

            state_dict.update(loaded_state_dict)

            module.load_state_dict(state_dict, strict=True)

            _print("Done loading weights for module {}, took {} seconds.".format(dest_module_path, time.time() - start))

    def get_load_paths(self):

        load_path = cfg.load_path
        _print("\nMaybe loading weights, load_path={} ...".format(load_path))

        if load_path:
            if isinstance(load_path, str) or isinstance(load_path, int):
                load_path = {"": load_path}

            load_path = dict(load_path)

            # Sort in increasing order, so that it if one variable scope lies within another scope,
            # the outer scope gets loaded before the inner scope, rather than having the outer scope
            # wipe out the inner scope.
            items = sorted(load_path.items())
            return items

        else:
            _print("`load_path` is null, using a fresh set of weights.")
            return []


def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    if norm_type == inf:
        return max(p.grad.data.abs().max() for p in parameters)

    total_norm = 0
    for i, p in enumerate(parameters):
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class PyTorchUpdater(Parameterized):
    build_model = Param()
    lr_schedule = Param()
    build_optimizer = Param()
    max_grad_norm = Param()
    print_grad_norm_step = Param(0)

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

    def path_for(self, name, ext):
        if ext is None:
            basename = 'stage={:0>4}_local_step={}'.format(self.stage_idx, self.step)
        else:
            basename = 'stage={:0>4}_local_step={}.{}'.format(self.stage_idx, self.step, ext)
        return self.exp_dir.path_for('plots', name, basename)

    def savefig(self, name, fig, ext, is_dir=True):
        if is_dir:
            path = self.path_for(name, ext)
            fig.savefig(path)
            plt.close(fig)

            shutil.copyfile(
                path,
                os.path.join(
                    os.path.dirname(path),
                    f'latest_stage{self.stage_idx:0>4}.{ext}')
            )
        else:
            path = self.exp_dir.path_for('plots', f"{name}.{ext}")
            fig.savefig(path)
            plt.close(fig)

    def build_graph(self):
        print("Building model...")
        self.model = self.build_model(env=self.env)

        print("Built model:")
        print(repr(self.model))

        device = get_pytorch_device()
        print(f"\nMoving model to device '{device}'...")
        self.model.to(device)

        print("Building optimizer...")

        if isinstance(self.lr_schedule, float):
            lr = self.lr_schedule
            lr_schedule = None
        else:
            lr = 1e-4
            lr_schedule = self.lr_schedule

        self.optimizer = self.build_optimizer(self.model.parameters(), lr=lr)

        if lr_schedule is not None:
            self.scheduler = lr_schedule(self.optimizer)
        else:
            self.scheduler = None

        print("Building data manager...")
        data_manager = self.data_manager = PyTorchDataManager(datasets=self.env.datasets)
        data_manager.build_graph()
        self.train_iterator = data_manager.do_train()

        if self.print_grad_norm_step > 0:
            self.grad_norm_recorder = GradNormRecorder(self.model)
        else:
            self.grad_norm_recorder = None

    def update(self, batch_size, step):
        print_time = step % 100 == 0

        self.model.train()

        data = AttrDict(next(self.train_iterator))

        self.model.update_global_step(step)

        detect_grad_anomalies = cfg.get('detect_grad_anomalies', False)
        with torch.autograd.set_detect_anomaly(detect_grad_anomalies):

            profile_step = cfg.get('pytorch_profile_step', 0)
            if profile_step > 0 and step % profile_step == 0:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    tensors, data, recorded_tensors, losses = self.model(data, step)
                print(prof)
            else:
                with timed_block('forward', print_time):
                    tensors, data, recorded_tensors, losses = self.model(data, step)

            # --- loss ---

            losses = AttrDict(losses)

            loss = 0.0
            for name, tensor in losses.flatten().items():
                loss += tensor
                recorded_tensors['loss_' + name] = tensor
            recorded_tensors['loss'] = loss

            with timed_block('zero_grad', print_time):
                # Apparently this is faster, according to https://www.youtube.com/watch?v=9mS1fIYj1So, 10:37
                for param in self.model.parameters():
                    param.grad = None
                # self.optimizer.zero_grad()

            with timed_block('loss backward', print_time):
                loss.backward()

        with timed_block('process grad', print_time):
            if self.grad_norm_recorder is not None:
                self.grad_norm_recorder.update()

                if step % self.print_grad_norm_step == 0:
                    self.grad_norm_recorder.display()

            parameters = list(self.model.parameters())
            pure_grad_norm = grad_norm(parameters)

            if self.max_grad_norm is not None and self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

            clipped_grad_norm = grad_norm(parameters)

        with timed_block('optimizer step', print_time):
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        update_result = self._update(batch_size)

        if isinstance(update_result, dict):
            recorded_tensors.update(update_result)

        self._n_experiences += batch_size

        recorded_tensors.update(
            grad_norm_pure=pure_grad_norm,
            grad_norm_clipped=clipped_grad_norm
        )

        scheduled_values = self.model.get_scheduled_values()
        recorded_tensors.update(scheduled_values)

        recorded_tensors = map_structure(
            lambda t: t.mean() if isinstance(t, torch.Tensor) else t,
            recorded_tensors, is_leaf=lambda rec: not isinstance(rec, dict))

        return recorded_tensors

    def _update(self, batch_size):
        pass

    def evaluate(self, _, step, mode="val"):
        return self.model.evaluate(self, step, mode=mode)

    def get_eval_tensors(self, step, mode="val", data_exclude=None, tensors_exclude=None):
        """ Run `self.model` on either val or test dataset, return data and tensors. """

        assert mode in "val test".split()

        if tensors_exclude is None:
            tensors_exclude = []
        if isinstance(tensors_exclude, str):
            tensors_exclude = tensors_exclude.split()

        if data_exclude is None:
            data_exclude = []
        if isinstance(data_exclude, str):
            data_exclude = data_exclude.split()

        self.model.eval()
        if mode == 'val':
            data_iterator = self.data_manager.do_val()
        elif mode == 'test':
            data_iterator = self.data_manager.do_test()
        else:
            raise Exception("Unknown data mode: {}".format(mode))

        _tensors = []
        _data = []

        n_points = 0
        n_batches = 0
        record = None

        with torch.no_grad():
            for data in data_iterator:
                data = AttrDict(data)

                tensors, data, recorded_tensors, losses = self.model(data, step)

                losses = AttrDict(losses)

                loss = 0.0
                for name, tensor in losses.flatten().items():
                    loss += tensor
                    recorded_tensors['loss_' + name] = tensor
                recorded_tensors['loss'] = loss

                recorded_tensors = map_structure(
                    lambda t: to_np(t.mean()) if isinstance(t, (torch.Tensor, np.ndarray)) else t,
                    recorded_tensors, is_leaf=lambda rec: not isinstance(rec, dict))

                batch_size = recorded_tensors['batch_size']

                n_points += batch_size
                n_batches += 1

                if record is None:
                    record = recorded_tensors
                else:
                    record = map_structure(
                        lambda rec, rec_t: rec + batch_size * np.mean(rec_t), record, recorded_tensors,
                        is_leaf=lambda rec: not isinstance(rec, dict))

                data = AttrDict(data)
                for de in data_exclude:
                    try:
                        del data[de]
                    except (KeyError, AttributeError):
                        pass
                data = map_structure(
                    lambda t: to_np(t) if isinstance(t, torch.Tensor) else t,
                    data, is_leaf=lambda rec: not isinstance(rec, dict))

                tensors = AttrDict(tensors)
                for te in tensors_exclude:
                    try:
                        del tensors[te]
                    except (KeyError, AttributeError):
                        pass
                tensors = map_structure(
                    lambda t: to_np(t) if isinstance(t, torch.Tensor) else t,
                    tensors, is_leaf=lambda rec: not isinstance(rec, dict))

                _tensors.append(tensors)
                _data.append(data)

        def postprocess(*t):
            return pad_and_concatenate(t, axis=0)

        _tensors = map_structure(postprocess, *_tensors, is_leaf=lambda rec: not isinstance(rec, dict))
        _data = map_structure(postprocess, *_data, is_leaf=lambda rec: not isinstance(rec, dict))

        record = map_structure(
            lambda rec: rec / n_points, record, is_leaf=lambda rec: not isinstance(rec, dict))
        record = AttrDict(record)

        return _data, _tensors, record

    def save(self, path):
        whole_dict = dict(model=self.model.state_dict())
        torch.save(whole_dict, path)
        return path

    def restore(self, path):
        device = get_pytorch_device()
        whole_dict = torch.load(path, map_location=device)
        state = self.model.state_dict()
        state.update(whole_dict['model'])
        self.model.load_state_dict(state, strict=False)


class DummyUpdater(PyTorchUpdater):
    """ For when you just want to build datasets. Much faster than most normal updaters. """

    @property
    def n_experiences(self):
        return 0

    def build_graph(self):
        pass

    def update(self, batch_size, step):
        return dict()

    def evaluate(self, batch_size, mode):
        return dict()

    def save(self, session, filename):
        return ''

    def restore(self, path):
        pass


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
        iterator.batch_size = batch_size
        self.iterators[name] = iterator

        return dset, iterator

    def do_train(self):
        return self.do('train')

    def do_val(self):
        return self.do('val')

    def do_test(self):
        return self.do('test')

    def do(self, name):
        """ Initialize iterator (unless it's the `train` iterator, which is handled slightly differently)
            and return a feed_dict populated with the appropriate handle for the requested iterator. """

        iterator = self.iterators[name]

        if name == 'train':
            if not self.train_initialized:
                iterator.reset()
                self.train_initialized = True
        else:
            iterator.reset()
        return iterator


class DataIterator:
    def __init__(self, sess, tf_iterator):
        self.sess = sess
        self.tf_iterator = tf_iterator
        self.get_next_op = self.tf_iterator.get_next()

    def reset(self):
        self.sess.run(self.tf_iterator.initializer)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.sess.run(self.get_next_op)
        except tf.errors.OutOfRangeError:
            raise StopIteration("TensorFlow iterator raised tf.errors.OutOfRangeError.")

        device = get_pytorch_device()

        result = map_structure(
            lambda v: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v,
            result, is_leaf=lambda v: not isinstance(v, dict)
        )

        return result


class PyTorchDummyDataManager(Parameterized):
    """ A dummy DataManager which only loads a single batch, when it is created, and yields that same batch every
        timestep. Using this data manager in place of the real one can be used to determine the runtime taken up by
        loading data and putting it on the GPU (this dummy data manager doesn't do loading every timestep ,so the
        difference in runtime per timestep between using this DataManager vs the real one is the amount of time used
        to load the data each timestep).

    """
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
                train_dset, __ = self.build_iterator('train', 'train', self.batch_size, True, 0)
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

        dset = dset.batch(batch_size)
        if hasattr(base_dataset, 'parse_example_batch'):
            dset = dset.map(base_dataset.parse_example_batch)
        tf_iterator = dset.make_initializable_iterator()
        iterator = DummyDataIterator(self.sess, tf_iterator)
        self.iterators[name] = iterator

        return dset, iterator

    def do_train(self):
        return self.do('train')

    def do_val(self):
        return self.do('val')

    def do_test(self):
        return self.do('test')

    def do(self, name):
        """ Initialize iterator (unless it's the `train` iterator, which is handled slightly differently)
            and return a feed_dict populated with the appropriate handle for the requested iterator. """

        iterator = self.iterators[name]

        if name == 'train':
            if not self.train_initialized:
                iterator.reset()
                self.train_initialized = True
        else:
            iterator.reset()
        return iterator


class DummyDataIterator:
    def __init__(self, sess, tf_iterator):
        self.sess = sess
        self.tf_iterator = tf_iterator
        self.get_next_op = self.tf_iterator.get_next()
        self.result = None

    def reset(self):
        self.sess.run(self.tf_iterator.initializer)
        try:
            result = self.sess.run(self.get_next_op)
        except tf.errors.OutOfRangeError:
            raise StopIteration("TensorFlow iterator raised tf.errors.OutOfRangeError.")

        device = get_pytorch_device()
        self.result = map_structure(
            lambda v: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v,
            result, is_leaf=lambda v: not isinstance(v, dict)
        )

    def __iter__(self):
        return self

    def __next__(self):
        return self.result
