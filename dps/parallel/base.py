import shutil
import dill
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import argparse
from contextlib import contextmanager
import sys
import signal

from dps.utils import SigTerm, KeywordMapping, pdb_postmortem


@contextmanager
def redirect_stream(stream, filename, mode='w'):
    assert stream in ['stdout', 'stderr']
    with open(str(filename), mode=mode) as f:
        old = getattr(sys, stream)
        setattr(sys, stream, f)

        try:
            yield
        finally:
            setattr(sys, stream, old)


def raise_sigterm(*args, **kwargs):
    raise SigTerm()


def vprint(s, v, threshold=0):
    if v > threshold:
        print(s)


class Operator(object):
    """
    Needs to be fully serializable since we're saving and loading these all the time.

    Parameters
    ----------
    name: str
    func_key: key
    inp_keys: list of key
    outp_keys: list of key
    pass_store: bool
        Whether to pass the store object as the first argument.
    metadata: dict

    """
    def __init__(self, name, func_key, inp_keys, outp_keys, pass_store=False, metadata=None):
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
    name={},
    func_key={},
    inp_keys={},
    outp_keys={},
    pass_store={},
    metadata={})""".format(
            pformat(self.name), pformat(self.func_key), pformat(self.inp_keys),
            pformat(self.outp_keys), pformat(self.pass_store), pformat(self.metadata))

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

    def is_complete(self, store):
        return all([store.object_exists('data', ok) for ok in self.outp_keys])

    def is_ready(self, store):
        return all([store.object_exists('data', ik) for ik in self.inp_keys])

    def run(self, store, force, output_to_files=True, verbose=False):
        vprint("\n\n" + ("*" * 80), verbose)
        vprint("Checking whether to run op {}".format(self.name), verbose)
        old_sigterm_handler = signal.signal(signal.SIGTERM, raise_sigterm)

        try:
            force_unique = True
            if self.is_complete(store):
                if force:
                    force_unique = False
                    vprint("Op {} is already complete, but ``force`` is True, so we're running it anyway.".format(self.name), verbose)
                else:
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
                stdout = store.path_for('stdout')
                stdout.mkdir(parents=True, exist_ok=True)
                stderr = store.path_for('stderr')
                stderr.mkdir(parents=True, exist_ok=True)
                with redirect_stream('stdout', stdout / self.name):
                    with redirect_stream('stderr', stderr / self.name):
                        outputs = func(*inputs)
            else:
                outputs = func(*inputs)
            vprint("\n\n" + ("-" * 40), verbose)
            vprint("Function for op {} has returned".format(self.name), verbose)
            vprint("Saving output for op {}".format(self.name), verbose)
            if len(self.outp_keys) == 1:
                store.save_object('data', self.outp_keys[0], outputs, force_unique=force_unique, clobber=True)
            else:
                for o, ok in zip(outputs, self.outp_keys):
                    store.save_object('data', ok, o, force_unique=force_unique, clobber=True)

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
        if not self.is_complete(store):
            raise Exception("Cannot get outputs, op is not complete.")
        return [store.load_object('data', ok) for ok in self.outp_keys]


class Signal(object):
    def __init__(self, key, name):
        self.key = key
        self.name = name

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Signal(key={}, name={})".format(self.key, self.name)


class ReadOnlyJob(object):
    def __init__(self, path):
        # A store for functions, data and operators.
        path = Path(path)
        if path.suffix == '.zip':
            self.objects = ZipObjectStore(path)
        else:
            self.objects = FileSystemObjectStore(path)

    def get_ops(self, pattern=None, sort=True):
        operators = list(self.objects.load_objects('operator').values())
        if pattern is not None:
            selected = KeywordMapping.batch([op.name for op in operators], pattern)
            operators = (op for i, op in enumerate(operators) if selected[i])
        if sort:
            operators = sorted(operators, key=lambda op: op.name)
        return operators

    def _print_ops(self, ops, verbose):
        s = []
        if verbose == 1:
            for op in ops:
                s.append(op.name)
        elif verbose == 2:
            for op in ops:
                s.append(op.status(self.objects, verbose=False))
        elif verbose == 3:
            for op in ops:
                s.append(op.status(self.objects, verbose=True))
        return '\n'.join(s)

    def summary(self, pattern=None, verbose=0):
        operators = self.get_ops(pattern, sort=True)
        operators = list(self.objects.load_objects('operator').values())
        s = ["\nJob Summary\n-----------"]
        s.append("\nn_ops: {}".format(len(operators)))

        is_complete = [op.is_complete(self.objects) for op in operators]
        completed_ops = [op for i, op in enumerate(operators) if is_complete[i]]
        incomplete_ops = [op for i, op in enumerate(operators) if not is_complete[i]]

        s.append("\nn_completed_ops: {}".format(len(completed_ops)))
        s.append(self._print_ops(completed_ops, verbose))

        is_ready = [op.is_ready(self.objects) for op in operators]
        ready_incomplete_ops = [op for i, op in enumerate(incomplete_ops) if is_ready[i]]
        not_ready_incomplete_ops = [op for i, op in enumerate(incomplete_ops) if not is_ready[i]]

        s.append("\nn_ready_incomplete_ops: {}".format(len(ready_incomplete_ops)))
        s.append(self._print_ops(ready_incomplete_ops, verbose))

        s.append("\nn_not_ready_incomplete_ops: {}".format(len(not_ready_incomplete_ops)))
        s.append(self._print_ops(not_ready_incomplete_ops, verbose))

        return '\n'.join(s)

    def completion(self, pattern=None):
        operators = list(self.get_ops(pattern, sort=False))
        is_complete = [op.is_complete(self.objects) for op in operators]
        completed_ops = [op for i, op in enumerate(operators) if is_complete[i]]
        incomplete_ops = [op for i, op in enumerate(operators) if not is_complete[i]]

        is_ready = [op.is_ready(self.objects) for op in operators]
        ready_incomplete_ops = [op for i, op in enumerate(incomplete_ops) if is_ready[i]]
        not_ready_incomplete_ops = [op for i, op in enumerate(incomplete_ops) if not is_ready[i]]

        return dict(
            n_ops=len(operators),
            n_complete=len(completed_ops),
            n_incomplete=len(incomplete_ops),
            n_ready_incomplete=len(ready_incomplete_ops),
            n_not_ready_incomplete=len(not_ready_incomplete_ops))

    def completed_ops(self):
        ops = list(self.get_ops(None, sort=False))
        return [op for i, op in enumerate(ops) if op.is_complete(self.objects)]


class Job(ReadOnlyJob):
    def __init__(self, path):
        self.objects = FileSystemObjectStore(path)  # A store for functions, data and operators.
        self.map_idx = 0
        self.reduce_idx = 0
        self.n_signals = 0

    def save_object(self, kind, key, obj, force_unique=True, clobber=False):
        self.objects.save_object(kind, key, obj, force_unique, clobber)

    def add_op(self, name, func, inputs, n_outputs, pass_store):
        if not callable(func):
            func_key = func
        else:
            func_key = self.objects.get_unique_key('function')
            self.save_object('function', func_key, func, force_unique=False)

        inp_keys = []
        for inp in inputs:
            if not isinstance(inp, Signal):
                key = self.objects.get_unique_key('data')
                self.save_object('data', key, inp, force_unique=True)
            else:
                key = inp.key
            inp_keys.append(key)

        outputs = [
            Signal(key=self.objects.get_unique_key('data'), name="{}[{}]".format(name, i))
            for i in range(n_outputs)]
        outp_keys = [outp.key for outp in outputs]

        op = Operator(
            name=name, func_key=func_key,
            inp_keys=inp_keys, outp_keys=outp_keys,
            pass_store=pass_store)
        op_key = self.objects.get_unique_key('operator')
        self.save_object('operator', op_key, op, force_unique=True)

        return outputs

    def map(self, func, inputs, name=None, pass_store=False):
        """ Currently restricted to fns with one input and one output. """
        op_name = name or 'map:{}'.format(self.map_idx)
        results = []

        func_key = self.objects.get_unique_key('function')
        self.save_object('function', func_key, func, force_unique=True)

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

    def run(self, pattern, indices, force, output_to_files, verbose):
        operators = self.get_ops(pattern)

        if not operators:
            return False

        if not indices:
            indices = set(range(len(operators)))

        return [op.run(self.objects, force, output_to_files, verbose) for i, op in enumerate(operators) if i in indices]

    def zip(self, archive_name=None, delete=False):
        return self.objects.zip(archive_name, delete=delete)


class ObjectStore(object):
    def __init__(self):
        pass

    def object_exists(self, kind, key):
        raise Exception("Abstract method.")

    def save_object(self, kind, key, obj, force_unique=True, clobber=False):
        raise Exception("Abstract method.")

    def load_object(self, kind, key):
        raise Exception("Abstract method.")

    def load_objects(self, kind):
        raise Exception("Abstract method.")

    def n_objects(self, kind=None):
        raise Exception("Abstract method.")


def split_path(path, root):
    _path = Path(path).relative_to(root)
    kind = str(_path.parent)
    key = _path.stem
    return kind, key


class FileSystemObjectStore(ObjectStore):
    def __init__(self, directory, force_fresh=False):
        if str(directory).endswith('.zip'):
            raise ValueError("Cannot create a FileSystemObjectStore from a zip file.")
        self.used_keys = defaultdict(list)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=not force_fresh)
        super(FileSystemObjectStore, self).__init__()

    def path_for(self, kind=None, key=None):
        path = self.directory
        if kind:
            path /= kind
        if key:
            path /= '{}.key'.format(key)
        return path

    def get_unique_key(self, kind):
        unique_key = max(self.used_keys[kind], default=0) + 1
        assert unique_key not in self.used_keys[kind]
        self.used_keys[kind].append(unique_key)
        return unique_key

    def object_exists(self, kind, key):
        return self.path_for(kind, key).exists()

    def save_object(self, kind, key, obj, force_unique=True, clobber=False):
        if self.object_exists(kind, key):
            if force_unique:
                raise ValueError("Trying to save object {} with kind {} and key {}, "
                                 "but an object ({}) already exists at that location and "
                                 "``force_unique`` is True.".format(obj, kind, key, self.load_object(kind, key)))
            if not clobber:
                return

        path = self.path_for(kind, key)
        path.parent.mkdir(exist_ok=True, parents=True)

        with path.open('wb') as f:
            dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

    def delete_object(self, kind, key):
        path = self.path_for(kind, key)
        try:
            shutil.rmtree(str(path))
        except:
            pass

    def load_object(self, kind, key):
        path = self.path_for(kind, key)
        if not self.object_exists(kind, key):
            raise ValueError("No object with kind {} and key {}.".format(kind, key))
        with path.open('rb') as f:
            obj = dill.load(f)
        return obj

    def load_objects(self, kind):
        directory = self.path_for(kind)
        objects = {}
        for obj_path in directory.glob('**/*.key'):
            with obj_path.open('rb') as f:
                obj = dill.load(f)
            objects[split_path(obj_path, self.directory)] = obj
        return objects

    def n_objects(self, kind=None):
        return len(self.keys(kind))

    def keys(self, kind):
        """ Returns list of tuples of form (kind, key) """
        directory = self.path_for(kind)
        return list(split_path(p, self.directory) for p in directory.glob('**/*.key'))

    def zip(self, archive_name=None, delete=False):
        if not archive_name:
            archive_name = Path(self.directory).name
        archive_name = Path(archive_name)
        archive_name = archive_name.parent / archive_name.stem

        # Within the archive, all entries are contained inside
        # a directory with a name given by ``base_dir``.
        archive_path = shutil.make_archive(
            str(archive_name), 'zip', root_dir=str(self.directory.parent),
            base_dir=str(self.directory.relative_to(self.directory.parent)))

        archive_path = Path(archive_path).resolve()
        if not str(archive_path).startswith(str(self.directory.parent)):
            archive_path = shutil.move(str(archive_path), str(self.directory.parent))

        print("Zipped {} as {}.".format(self.directory, archive_path))
        if delete:
            shutil.rmtree(str(self.directory))
        return archive_path


class ZipObjectStore(ObjectStore):
    """ A read-only object store based on zip file. Avoids ever unzipping the entire file. """
    def __init__(self, zip_path):
        self._zip = ZipFile(str(zip_path), 'r')
        self._zip_root = Path(zip_root(zip_path))

    def __enter__(self):
        pass

    def __exit__(self):
        self._zip.close()

    def path_for(self, kind=None, key=None):
        path = self._zip_root
        if kind:
            path /= kind
        if key:
            path /= '{}.key'.format(key)
        return path

    def object_exists(self, kind, key):
        return str(self.path_for(kind, key)) in self._zip.namelist()

    def save_object(self, kind, key, obj, force_unique, clobber):
        raise Exception("Read-only object store.")

    def load_object(self, kind, key):
        path = self.path_for(kind, key)
        if not self.object_exists(kind, key):
            raise ValueError("No object with kind {} and key {}.".format(kind, key))
        with self._zip.open(str(path), 'r') as f:
            obj = dill.load(f)
        return obj

    def load_objects(self, kind):
        directory = str(self.path_for(kind))
        object_files = [
            s for s in self._zip.namelist()
            if s.startswith(directory) and s.endswith('.key')]
        objects = {}
        for o in object_files:
            with self._zip.open(o, 'r') as f:
                obj = dill.load(f)
            objects[split_path(o, self._zip_root)] = obj
        return objects

    def n_objects(self, kind=None):
        raise Exception("Abstract method.")

    def keys(self, kind):
        directory = str(self.path_for(kind))
        _keys = [
            split_path(s, self._zip_root) for s in self._zip.namelist()
            if s.startswith(directory) and s.endswith('.key')]
        return _keys


def zip_root(zipfile):
    """ Get the name of the root directory of a zip file, if it has one. """
    if not isinstance(zipfile, ZipFile):
        zipfile = ZipFile(str(zipfile), 'r')
    zip_root = min(
        [z.filename for z in zipfile.infolist()],
        key=lambda s: len(s))
    return zip_root


def run_command(args):
    job = Job(args.path)
    job.run(args.pattern, args.indices, args.force, args.redirect, args.verbose)


def view_command(args):
    job = ReadOnlyJob(args.path)
    print(job.summary(verbose=args.verbose))


def parallel_cl(desc, additional_cmds=None):
    """ Entry point for the `dps-parallel` command-line utility.

    Parameters
    ----------
    desc: str
        Description for the script.
    additional_cmds: list
        Each element has form (name, help, func, *(arg_name, kwargs)).

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
    directory = Path('/tmp/test_job/test')
    try:
        shutil.rmtree(str(directory))
    except:
        pass
    job = Job(directory)
    x = range(10)
    z = job.map(lambda y: y + 1, x)
    final = job.reduce(lambda *inputs: sum(inputs), z)
    print(job.summary())
    # for i in range(10):
    #     job.run("map", i)
    # job.run("reduce", 0)
    # print(job.summary())
