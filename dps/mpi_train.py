""" Entry point for workers spawned by the training loop. """

import tensorflow as tf
from contextlib import ExitStack
import sys

from dps import cfg
from dps.utils import memory_limit, NumpySeed, gen_seed, redirect_stream, ExperimentDirectory
from dps.utils.tf import uninitialized_variables_initializer

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def run_stage(mpi_context, env, stage_idx, exp_dir):
    config, seed = mpi_context.start_stage()

    with ExitStack() as stack:
        stack.enter_context(config)
        stack.enter_context(NumpySeed(seed))

        # Accept config for new stage
        print("\n" + "-" * 10 + " Stage set-up " + "-" * 10)

        print(cfg.to_string())

        # Configure and create session and graph for stage.
        session_config = tf.ConfigProto()
        session_config.intra_op_parallelism_threads = cfg.get('intra_op_parallelism_threads', 0)
        session_config.inter_op_parallelism_threads = cfg.get('inter_op_parallelism_threads', 0)

        # if cfg.use_gpu:
        #     per_process_gpu_memory_fraction = getattr(cfg, 'per_process_gpu_memory_fraction', None)
        #     if per_process_gpu_memory_fraction:
        #         session_config.gpu_options.per_process_gpu_memory_fraction = \
        #             per_process_gpu_memory_fraction

        #     gpu_allow_growth = getattr(cfg, 'gpu_allow_growth', None)
        #     if gpu_allow_growth:
        #         session_config.gpu_options.allow_growth = gpu_allow_growth

        # if cfg.use_gpu:
        #     print("Using GPU if available.")
        #     print("Using {}% of GPU memory.".format(
        #         100 * session_config.gpu_options.per_process_gpu_memory_fraction))
        #     print("Allowing growth of GPU memory: {}".format(session_config.gpu_options.allow_growth))

        graph = tf.Graph()
        sess = tf.Session(graph=graph, config=session_config)

        # This HAS to come after the creation of the session, otherwise
        # it allocates all GPU memory if using the GPU.
        print("\nAvailable devices:")
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

        # if not cfg.use_gpu:
        #     print("Not using GPU.")
        #     stack.enter_context(graph.device("/cpu:0"))

        stack.enter_context(graph.device("/cpu:0"))

        stack.enter_context(graph.as_default())
        stack.enter_context(sess)
        stack.enter_context(sess.as_default())

        tf_seed = gen_seed()
        print("Setting tensorflow seed to generated seed: {}\n".format(tf_seed))
        tf.set_random_seed(tf_seed)

        # Set limit on CPU RAM for the stage
        cpu_ram_limit_mb = cfg.get("cpu_ram_limit_mb", None)
        if cpu_ram_limit_mb is not None:
            stack.enter_context(memory_limit(cfg.cpu_ram_limit_mb))

        print("Building env...\n")

        # Maybe build env
        if stage_idx == 0 or not cfg.preserve_env:
            if env is not None:
                env.close()
            env = cfg.build_env()

        if hasattr(env, "print_memory_footprint"):
            env.print_memory_footprint()

        print("\nDone building env.\n")
        print("Building updater...\n")

        updater = cfg.get_updater(env, mpi_context=mpi_context)
        updater.stage_idx = stage_idx
        updater.exp_dir = exp_dir

        updater.build_graph()
        print("\nDone building updater.\n")

        # walk_variable_scopes(max_depth=3)

        tf.train.get_or_create_global_step()
        sess.run(uninitialized_variables_initializer())
        sess.run(tf.assert_variables_initialized())

        updater.worker_code()

        stage_idx += 1

    return env


class MPI_Context(object):

    def _populate(self):
        self.n_procs = self.merged_comm.Get_size()
        self.rank = self.merged_comm.Get_rank()

    def __enter__(self):
        pass

    def __exit__(self, type_, value, tb):
        pass


class MPI_MasterContext(MPI_Context):
    def __init__(self, n_procs, exp_dir):
        self.n_procs = n_procs
        if n_procs > 1:
            exp_dir.path_for('subproc_stdout', is_dir=True)
            exp_dir.path_for('subproc_stderr', is_dir=True)
            self.launch_workers(n_procs, exp_dir.path)
        else:
            self.n_procs = 1
            self.inter_comm = None
            self.merged_comm = None
            self.rank = 0

    def launch_workers(self, n_procs, exp_path):
        script_path = __file__
        self.inter_comm = MPI.COMM_SELF.Spawn(sys.executable, args=[script_path], maxprocs=n_procs-1)
        print("done spawn")
        self.merged_comm = self.inter_comm.Merge(False)
        self._populate()
        print("done merge")
        self.inter_comm.bcast(exp_path, MPI.ROOT)
        print("done bcast")

    def start_stage(self):
        if self.n_procs > 1:
            config = cfg.freeze()
            self.inter_comm.bcast(config, root=MPI.ROOT)

            seeds = [gen_seed() for _ in range(self.n_procs-1)]
            self.inter_comm.scatter(seeds, root=MPI.ROOT)


class MPI_WorkerContext(MPI_Context):
    def __init__(self):
        self.inter_comm = MPI.Comm.Get_parent()
        self.merged_comm = self.inter_comm.Merge(True)
        self._populate()

        print("process {} of {}".format(self.rank, self.n_procs))

        exp_path = self.inter_comm.bcast(None, root=0)
        print(exp_path)
        self.exp_dir = ExperimentDirectory(exp_path, force_fresh=False)
        print("made exp dir")

    def start_stage(self):
        print("waiting for config.")
        config = self.inter_comm.bcast(None, root=0)
        print("got config.")
        print("waiting for seed config.")
        seed = self.inter_comm.scatter(None, root=0)
        print("got seed")
        return config, seed


if __name__ == "__main__":
    mpi_context = MPI_WorkerContext()

    exp_dir = mpi_context.exp_dir
    env = None
    stage_idx = 0

    with ExitStack() as stack:
        stack.enter_context(
            redirect_stream('stdout', exp_dir.path_for('subproc_stdout', 'proc_{}'.format(mpi_context.rank)), tee=False))
        stack.enter_context(
            redirect_stream('stderr', exp_dir.path_for('subproc_stderr', 'proc_{}'.format(mpi_context.rank)), tee=False))

        # stack.enter_context(mpi_context)

        print("Should be writing to files! rank: {}".format(mpi_context.rank))

        stage_idx = 0
        while True:
            env = run_stage(mpi_context, env, stage_idx, exp_dir)
            stage_idx += 1
