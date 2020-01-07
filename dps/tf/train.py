import tensorflow as tf
import time

from dps import cfg
from dps.train import TrainingLoop, TrainingLoopData
from dps.utils import flush_print as _print, gen_seed
from dps.utils.tf import (
    uninitialized_variables_initializer, trainable_variables, walk_variable_scopes
)


class TensorFlowTrainingLoopData(TrainingLoopData):
    def store_scalar_summaries(self, mode, path, record, n_global_experiences):
        if mode not in self.writers:
            self.writers[mode] = tf.summary.FileWriter(path, flush_secs=cfg.reload_interval)

        # Build a summary using the Summary protocol buffer
        # See https://stackoverflow.com/questions/37902705/how-to-manually-create-a-tf-summary
        summary_values = [tf.Summary.Value(tag="all/"+k, simple_value=float(v)) for k, v in record.items()]
        summary = tf.Summary(value=summary_values)

        self.writers[mode].add_summary(summary, n_global_experiences)


class TensorFlowTrainingLoop(TrainingLoop):
    training_loop_data_class = TensorFlowTrainingLoopData

    def framework_initialize_stage(self, stack):
        # Configure and create session and graph for stage.
        session_config = tf.ConfigProto()
        session_config.intra_op_parallelism_threads = cfg.get('intra_op_parallelism_threads', 0)
        session_config.inter_op_parallelism_threads = cfg.get('inter_op_parallelism_threads', 0)
        session_config.log_device_placement = cfg.get('log_device_placement', 0)

        if cfg.use_gpu:
            per_process_gpu_memory_fraction = getattr(cfg, 'per_process_gpu_memory_fraction', None)
            if per_process_gpu_memory_fraction:
                session_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction

            gpu_allow_growth = getattr(cfg, 'gpu_allow_growth', None)
            if gpu_allow_growth:
                session_config.gpu_options.allow_growth = gpu_allow_growth

            _print("Using GPU if available.")
            _print("Using {}% of GPU memory.".format(
                100 * session_config.gpu_options.per_process_gpu_memory_fraction))
            _print("Allowing growth of GPU memory: {}".format(session_config.gpu_options.allow_growth))

        graph = tf.Graph()
        sess = tf.Session(graph=graph, config=session_config)

        # This HAS to come after the creation of the session, otherwise
        # it allocates all GPU memory if using the GPU.
        _print("\nAvailable devices: ")
        from tensorflow.python.client import device_lib
        _print(device_lib.list_local_devices())

        if not cfg.use_gpu:
            _print("Not using GPU.")
            stack.enter_context(graph.device("/cpu:0"))

        stack.enter_context(graph.as_default())
        stack.enter_context(sess)
        stack.enter_context(sess.as_default())

        # Set the seed for the stage.
        tf_seed = gen_seed()
        _print("Setting tensorflow seed to generated seed: {}\n".format(tf_seed))
        tf.set_random_seed(tf_seed)

        tf.logging.set_verbosity(tf.logging.ERROR)

    def framework_finalize_stage_initialization(self):
        self.framework_print_variables()
        self.framework_load_weights()

        sess = tf.get_default_session()

        tf_step = tf.train.get_or_create_global_step()

        if cfg.initial_step is not None and cfg.initial_step > 0:
            sess.run(tf_step.assign(cfg.initial_step))

        sess.run(uninitialized_variables_initializer())
        sess.run(tf.assert_variables_initialized())

        # Prevent memory leaks, no ops can be added to the graph after this point
        tf.get_default_graph().finalize()

    def framework_print_variables(self):
        walk_variable_scopes(max_depth=cfg.variable_scope_depth)

    def framework_load_weights(self):
        for var_scope, path in self.get_load_paths():
            _print("Loading var scope \"{}\" from {}.".format(var_scope, path))

            start = time.time()
            variables = {v.name: v for v in trainable_variables(var_scope, for_opt=False)}
            if not variables:
                _print("No variables to load in scope {}.".format(str(var_scope)))
                continue

            saver = tf.train.Saver(variables)
            saver.restore(tf.get_default_session(), path)

            _print("Done loading var scope, took {} seconds.".format(time.time() - start))
