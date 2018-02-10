import shutil
import signal
import numpy as np
import os
import types
import sys
import argparse

from dps.parallel.object_store import FileSystemObjectStore, ZipObjectStore
from dps.utils import SigTerm, KeywordMapping, pdb_postmortem, redirect_stream, modify_env


def raise_sigterm(*args, **kwargs):
    raise SigTerm()


def vprint(s, v, threshold=0):
    if v > threshold:
        print(s)


class Operator(object):
    """ A processing element of a computation graph.

    Needs to be fully serializable since we're saving and loading these all the time.

    Parameters
    ----------
    idx: int
    name: str
    func_key: key
    inp_keys: list of key
    outp_keys: list of key
    pass_store: bool
        Whether to pass the store object as the first argument.
    metadata: dict

    """
    def __init__(
            self, idx, name, func_key, inp_keys, outp_keys,
            pass_store=False, metadata=None):

        self.idx = idx
        self.name = name
        self.func_key = func_key
        self.inp_keys = inp_keys
        self.outp_keys = outp_keys
        self.pass_store = pass_store
        self.metadata = metadata

    def __str__(self):
        from pprint import pformat
        return """\
Operator(
    idx={},
    name={},
    func_key={},
    inp_keys={},
    outp_keys={},
    pass_store={},
    metadata={})""".format(
            pformat(self.idx), pformat(self.name), pformat(self.func_key),
            pformat(self.inp_keys), pformat(self.outp_keys),
            pformat(self.pass_store), pformat(self.metadata))

    def __repr__(self):
        return str(self)

    def status(self, store, verbose=False):
        s = ["\n" + ("-" * 40)]
        s.append("Status report for\n {}.".format(self))
        is_ready = self.is_ready(store)
        s.append("Ready? {}".format(is_ready))
        if is_ready and verbose:
            s.append("Input values:")
            inputs = [store.load_object('data', ik) for ik in self.inp_keys]
            for inp in inputs:
                s.append(str(inp))

        is_complete = self.is_complete(store)
        s.append("Complete? {}".format(is_complete))
        if is_complete and verbose:
            s.append("Output values:")
            outputs = [store.load_object('data', ok) for ok in self.outp_keys]
            for outp in outputs:
                s.append(str(outp))
        return '\n'.join(s)

    def is_complete(self, store, partial=False):
        complete = store.object_exists('complete', self.idx)
        part_complete = all([store.object_exists('data', ok) for ok in self.outp_keys])
        return complete or (part_complete and partial)

    def is_ready(self, store):
        return all([store.object_exists('data', ik) for ik in self.inp_keys])

    def call_function(self, func, store, inputs, verbose):
        outputs = func(*inputs)

        if not isinstance(outputs, types.GeneratorType):
            outputs = [outputs]

        for outp in outputs:
            print("Checkpointing...")
            vprint("\n\n" + ("-" * 40), verbose)
            vprint("Function for op {} has returned".format(self.name), verbose)
            vprint("Saving output for op {}".format(self.name), verbose)

            if len(self.outp_keys) == 1:
                outp = [outp]

            for key, obj in zip(self.outp_keys, outp):
                store.save_object('data', key, obj, recurse=False)

    def run(self, store, force, output_to_files=True, verbose=False):
        vprint("\n\n" + ("*" * 80), verbose)
        vprint("Checking whether to run op {}".format(self.name), verbose)
        old_sigterm_handler = signal.signal(signal.SIGTERM, raise_sigterm)

        try:
            if self.is_complete(store, partial=False):
                vprint("Skipping op {}, already complete.".format(self.name), verbose)
                return False
            else:
                vprint("Op {} is not complete, so we SHOULD run it.".format(self.name), verbose)

            if not self.is_ready(store):
                vprint("Skipping op {}, deps are not met.".format(self.name), verbose)
                return False
            else:
                vprint("Op {} is ready, so we CAN run it.".format(self.name), verbose)

            vprint("Running op {}".format(self.name), verbose)

            vprint("Loading objects for op {}".format(self.name), verbose)
            inputs = [store.load_object('data', ik) for ik in self.inp_keys]
            func = store.load_object('function', self.func_key)

            for inp in inputs:
                if hasattr(inp, 'log_dir'):
                    inp.log_dir = str(store.directory)

            if self.pass_store:
                inputs.insert(0, store)

            vprint("Calling function for op {}".format(self.name), verbose)
            vprint("\n\n" + ("-" * 40), verbose)
            if output_to_files:
                stdout_path = store.path_for_kind('stdout')
                os.makedirs(stdout_path, exist_ok=True)

                stderr_path = store.path_for_kind('stderr')
                os.makedirs(stderr_path, exist_ok=True)

                with redirect_stream('stdout', os.path.join(stdout_path, self.name)):
                    with redirect_stream('stderr', os.path.join(stderr_path, self.name)):
                        self.call_function(func, store, inputs, verbose)
            else:
                self.call_function(func, store, inputs, verbose)

            store.save_object('complete', self.idx, self.idx)

            vprint("op {} complete.".format(self.name), verbose)
            vprint("\n\n" + ("*" * 80), verbose)

            return True
        finally:
            signal.signal(signal.SIGTERM, old_sigterm_handler)

    def get_inputs(self, store):
        if not self.is_ready(store):
            raise Exception("Cannot get inputs, op is not ready to run.")
        return [store.load_object('data', ik) for ik in self.inp_keys]

    def get_outputs(self, store):
        if not self.is_complete(store, partial=True):
            raise Exception("Cannot get outputs, op is not complete.")
        return [store.load_object('data', ok) for ok in self.outp_keys]


class Signal(object):
    """ Representation of a piece of data used when constructing computation graphs. """

    def __init__(self, key, name):
        self.key = key
        self.name = name

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Signal(key={}, name={})".format(self.key, self.name)


class ReadOnlyJob(object):
    """ A job that can only be examined. """

    def __init__(self, path):
        if os.path.splitext(path)[1] == '.zip':
            self.objects = ZipObjectStore(path)
        else:
            self.objects = FileSystemObjectStore(path)

    def get_ops(self, pattern=None, sort=True, ready=None, complete=None, partial=False):
        operators = list(self.objects.load_objects('operator').values())
        if pattern is not None:
            selected = KeywordMapping.batch([op.name for op in operators], pattern)
            operators = (op for i, op in enumerate(operators) if selected[i])

        if ready is not None:
            pred = lambda b: b if ready else lambda b: not b
            operators = [op for op in operators if pred(op.is_ready(self.objects))]

        if complete is not None:
            pred = lambda b: b if complete else lambda b: not b
            operators = [op for op in operators if pred(op.is_complete(self.objects, partial=partial))]

        if sort:
            operators = sorted(operators, key=lambda op: op.idx)

        return operators

    def _print_ops(self, ops, verbose):
        s = []

        for op in ops:
            if verbose == 1:
                s.append(op.name)
            elif verbose == 2:
                s.append(op.status(self.objects, verbose=False))
            elif verbose == 3:
                s.append(op.status(self.objects, verbose=True))

        return '\n'.join(s)

    def summary(self, pattern=None, verbose=0):
        operators = self.get_ops(pattern, sort=True)
        operators = list(self.objects.load_objects('operator').values())
        s = ["\nJob Summary\n-----------"]
        s.append("\nn_ops: {}".format(len(operators)))

        is_complete = [op.is_complete(self.objects) for op in operators]
        is_partially_complete = [op.is_complete(self.objects, partial=True) for op in operators]
        completed_ops = [op for i, op in enumerate(operators) if is_complete[i]]
        partially_completed_ops = [op for i, op in enumerate(operators) if is_partially_complete[i] and not is_complete[i]]
        incomplete_ops = [op for i, op in enumerate(operators) if not is_complete[i]]

        s.append("\nn_completed_ops: {}".format(len(completed_ops)))
        s.append(self._print_ops(completed_ops, verbose))

        s.append("n_partially_completed_ops: {}".format(len(partially_completed_ops)))
        s.append(self._print_ops(partially_completed_ops, verbose))

        is_ready = [op.is_ready(self.objects) for op in operators]
        ready_incomplete_ops = [op for i, op in enumerate(incomplete_ops) if is_ready[i]]
        not_ready_incomplete_ops = [op for i, op in enumerate(incomplete_ops) if not is_ready[i]]

        s.append("n_ready_incomplete_ops: {}".format(len(ready_incomplete_ops)))
        s.append(self._print_ops(ready_incomplete_ops, verbose))

        s.append("n_not_ready_incomplete_ops: {}".format(len(not_ready_incomplete_ops)))
        s.append(self._print_ops(not_ready_incomplete_ops, verbose))

        return '\n'.join(s)

    def completion(self, pattern=None):
        operators = list(self.get_ops(pattern, sort=True))
        is_complete = [op.is_complete(self.objects) for op in operators]
        completed_ops = [op for i, op in enumerate(operators) if is_complete[i]]
        incomplete_ops = [op for i, op in enumerate(operators) if not is_complete[i]]

        is_ready = [op.is_ready(self.objects) for op in incomplete_ops]
        ready_incomplete_ops = [op for i, op in enumerate(incomplete_ops) if is_ready[i]]
        not_ready_incomplete_ops = [op for i, op in enumerate(incomplete_ops) if not is_ready[i]]

        return dict(
            n_ops=len(operators),
            n_complete=len(completed_ops),
            n_incomplete=len(incomplete_ops),
            n_ready_incomplete=len(ready_incomplete_ops),
            n_not_ready_incomplete=len(not_ready_incomplete_ops),
            ready_incomplete_ops=ready_incomplete_ops)

    def ready_incomplete_ops(self, sort=False):
        return list(self.get_ops(None, sort=sort, complete=False, ready=True))

    def completed_ops(self, sort=False, partial=False):
        return list(self.get_ops(None, sort=sort, complete=True, partial=partial))


class Job(ReadOnlyJob):
    def __init__(self, path):
        self.objects = FileSystemObjectStore(path)  # A store for functions, data and operators.
        self.map_idx = 0
        self.reduce_idx = 0
        self.op_idx = 0
        self.n_signals = 0

    def save_object(self, kind, key, obj, force_unique=True, recurse=True):
        exists = self.objects.object_exists(kind, key)
        if force_unique and exists:
            raise Exception("Object with kind {} and key {} already exists, but `force_unique` is True.")

        self.objects.save_object(kind, key, obj, recurse=True)

    def add_op(self, name, func, inputs, n_outputs, pass_store):
        """ `func` can either be a function, the key to a function that has already been saved. """

        if not callable(func):
            func_key = func
        else:
            func_key = self.objects.get_unique_key('function')
            self.save_object('function', func_key, func, recurse=True)

        inp_keys = []
        for inp in inputs:
            if not isinstance(inp, Signal):
                key = self.objects.get_unique_key('data')
                self.save_object('data', key, inp, recurse=False)
            else:
                key = inp.key
            inp_keys.append(key)

        outputs = [
            Signal(key=self.objects.get_unique_key('data'), name="{}[{}]".format(name, i))
            for i in range(n_outputs)]
        outp_keys = [outp.key for outp in outputs]

        op = Operator(
            idx=self.op_idx, name=name, func_key=func_key,
            inp_keys=inp_keys, outp_keys=outp_keys,
            pass_store=pass_store)
        self.op_idx += 1
        op_key = self.objects.get_unique_key('operator')
        self.save_object('operator', op_key, op, recurse=True)

        return outputs

    def map(self, func, inputs, name=None, pass_store=False):
        """ Currently restricted to fns with one input and one output. """
        op_name = name or 'map:{}'.format(self.map_idx)
        results = []

        func_key = self.objects.get_unique_key('function')
        self.save_object('function', func_key, func, recurse=True)

        for idx, inp in enumerate(inputs):
            op_result = self.add_op(
                '{},app:{}'.format(op_name, idx), func_key, [inp], 1, pass_store)
            results.append(op_result[0])

        self.map_idx += 1

        return results

    def reduce(self, func, inputs, name=None, pass_store=False):
        op_name = name or 'reduce:{}'.format(self.reduce_idx)
        op_result = self.add_op(op_name, func, inputs, 1, pass_store)
        self.reduce_idx += 1

        return op_result

    def run(self, pattern, indices, force, output_to_files, verbose,
            idx_in_node, ppn, gpu_set, ignore_gpu):
        """ Run selected operators in the job.

        Only ops that meet constraints imposed by both `pattern` and `indices` will be executed.

        Parameters
        ----------
        pattern: str
            Only ops whose name matches this pattern will be executed.
        indices: set-like of int
            Only ops whose indices are in this set will be executed.

        """
        operators = self.get_ops(pattern)

        if not operators:
            return False

        # When using SLURM, not sure how to pass different arguments to different tasks. So instead, we
        # pass ALL arguments to all tasks, and then let each task select the argument we intend for it
        # by using SLURM_PROCID, which basically gives the ID of the task within the job.
        # `idx_in_job` is not used when not using slurm, as we can pass the indices to run directly to each job.
        idx_in_job = int(os.environ.get("SLURM_PROCID", -1))
        if idx_in_job != -1:
            try:
                indices = [indices[idx_in_job]]
            except IndexError:
                print("Process with index {} was not provided with an argument, exiting.".format(idx_in_job))
                sys.exit(0)
            print("My idx in the job is {}, the idx of the task I'm running is {}.".format(idx_in_job, indices[0]))
            print("My value of CUDA_VISIBLE_DEVICES is {}.".format(os.environ.get('CUDA_VISIBLE_DEVICES', None)))
            print("My value of GPU_DEVICE_ORDINAL is {}.".format(os.environ.get('GPU_DEVICE_ORDINAL', None)))

        print("My value of OMP_NUM_THREADS is {}.".format(os.environ.get('OMP_NUM_THREADS', None)))

        if int(idx_in_node) == -1:
            idx_in_node = int(os.environ.get("SLURM_LOCALID", -1))
        else:
            # When getting `idx_in_node` from gnu-parallel, it starts from 1 instead of 0.
            idx_in_node -= 1

        if ignore_gpu:
            env = dict(CUDA_VISIBLE_DEVICES="-1")
        elif idx_in_node != -1 and gpu_set:
            # `ppn` and `idx_in_node` are only used when using gpus.
            # Manually choose GPU to use based on our index in the node, the number of GPUs available on the node,
            # and the number of processors running on the node.
            assert ppn > 0
            gpus = [int(i) for i in gpu_set.split(',')]
            n_gpus = len(gpus)
            assert ppn % n_gpus == 0
            assert ppn >= n_gpus
            procs_per_gpu = ppn // n_gpus
            gpu_idx = int(np.floor(idx_in_node / procs_per_gpu))
            gpu_to_use = gpus[gpu_idx]

            env = dict(CUDA_VISIBLE_DEVICES=str(gpu_to_use))
        else:
            env = {}

        with modify_env(**env):
            return [op.run(self.objects, force, output_to_files, verbose)
                    for op in operators if op.idx in indices]

    def zip(self, archive_name=None, delete=False):
        return self.objects.zip(archive_name, delete=delete)


def run_command(args):
    """ Implements the `run` sub-command, which executes some of the job's operators. """
    job = Job(args.path)
    job.run(
        args.pattern, args.indices, args.force, args.redirect, args.verbose,
        args.idx_in_node, args.ppn, args.gpu_set, args.ignore_gpu)


def view_command(args):
    """ Implements the `view` sub-command, which prints a summary of a job. """
    job = ReadOnlyJob(args.path)
    print(job.summary(verbose=args.verbose))


def parallel_cl(desc, additional_cmds=None):
    """ Entry point for command-line utility to which additional sub-commands can be added.

    Parameters
    ----------
    desc: str
        Description for the script.
    additional_cmds: list
        Each element has the form (name, help, func, *(arg_name, kwargs)).

    """
    desc = desc or 'Run jobs and view their statuses.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--pdb', action='store_true', help="If supplied, enter post-mortem debugging on error.")
    parser.add_argument(
        '-v', '--verbose', action='count', default=0, help="Increase verbosity.")

    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help='Run a job.')
    run_parser.add_argument('path', type=str)
    run_parser.add_argument('pattern', type=str)
    run_parser.add_argument('indices', nargs='*', type=int)
    run_parser.add_argument('--idx-in-node', type=int, default=-1)
    run_parser.add_argument('--idx-in-job', type=int, default=-1)
    run_parser.add_argument('--ppn', type=int, default=-1)
    run_parser.add_argument('--gpu-set', type=str, default="")
    run_parser.add_argument('--ignore-gpu', action="store_true")
    run_parser.add_argument(
        '--force', action='store_true', help="If supplied, run the selected operators "
                                             "even if they've already been completed.")
    run_parser.add_argument(
        '--redirect', action='store_true', help="If supplied, output is redirected to files rather than being printed.")

    run_parser.set_defaults(func=run_command)

    view_parser = subparsers.add_parser('view', help='View status of a job.')
    view_parser.add_argument('path', type=str)

    view_parser.set_defaults(func=view_command)

    subparser_names = ['run', 'view']

    additional_cmds = additional_cmds or []
    for cmd in additional_cmds:
        subparser_names.append(cmd[0])
        cmd_parser = subparsers.add_parser(cmd[0], help=cmd[1])
        for arg_name, kwargs in cmd[3:]:
            cmd_parser.add_argument(arg_name, **kwargs)
        cmd_parser.set_defaults(func=cmd[2])

    args, _ = parser.parse_known_args()

    try:
        func = args.func
    except AttributeError:
        raise ValueError("Missing ``command`` argument to script. Should be one of {}.".format(subparser_names))

    if args.pdb:
        with pdb_postmortem():
            func(args)
    else:
        func(args)


if __name__ == "__main__":
    directory = '/tmp/test_job/test'
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        pass

    # Build job
    job = Job(directory)
    x = range(10)
    z = job.map(lambda y: y + 1, x)
    final = job.reduce(lambda *inputs: sum(inputs), z)
    print(job.summary())

    # Run job
    for i in range(10):
        job.run("map", i)
    job.run("reduce", 0)
    print(job.summary())
