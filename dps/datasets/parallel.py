import os
import subprocess
import inspect
import pprint

import clify

from dps import cfg
from dps.utils import gen_seed, Config, NumpySeed, ExperimentStore, cd, get_param_hash
from dps.parallel import Job
from dps.hyper.parallel_session import submit_job, ParallelSession


class _BuildDataset(object):
    """ Entry point for each process. """

    def __init__(self, cls, params):
        self.cls = cls
        self.params = params

    def __call__(self, inp):
        import os
        import datetime
        import dps
        from dps import cfg  # noqa
        from dps.utils import ExperimentStore, get_default_config
        os.nice(10)

        print("Entered _BuildDataset at: ")
        print(datetime.datetime.now())

        idx, seed, n_examples = inp
        print("idx: {}, seed: {}, n_examples: {}".format(idx, seed, n_examples))

        dps.reset_config()
        params = self.params.copy()
        params.update(seed=seed, n_examples=n_examples)

        default_config = get_default_config()

        with default_config:
            cfg.update_from_command_line()
            print(cfg)

            exp_store_name = 'env={}'.format(cfg.env_name)
            experiment_store = ExperimentStore(os.path.join(cfg.local_experiments_dir, exp_store_name))
            exp_dir = experiment_store.new_experiment("", seed, add_date=1, force_fresh=1, update_latest=False)

            params["data_dir"] = exp_dir.path

            print(params)

            self.cls(**params)

        print("Leaving _BuildDataset at: ")
        print(datetime.datetime.now())


def make_dataset_in_parallel(run_kwargs, dataset_cls, param_values=None):
    """ Uses dps.hyper.parallel_session.ParallelSession to create a dataset in parallel. """

    # Get run_kwargs from command line
    sig = inspect.signature(ParallelSession.__init__)
    default_run_kwargs = sig.bind_partial()
    default_run_kwargs.apply_defaults()
    cl_run_kwargs = clify.command_line(default_run_kwargs.arguments).parse()
    run_kwargs.update(cl_run_kwargs)

    param_values = param_values or dataset_cls._capture_param_values()
    param_values = Config(param_values)
    seed = param_values["seed"]
    if seed is None or seed < 0:
        seed = gen_seed()

    n_examples = param_values["n_examples"]
    n_examples_per_shard = run_kwargs["n_examples_per_shard"]

    experiment_store = ExperimentStore(
        cfg.parallel_experiments_build_dir, prefix="build_{}".format(dataset_cls.__name__))

    count = 0
    name = "attempt=0"
    has_built = False
    while not has_built:
        try:
            exp_dir = experiment_store.new_experiment(name, seed, add_date=True, force_fresh=True)
            has_built = True
        except FileExistsError:
            count += 1
            name = "attempt_{}".format(count)

    print("Building dataset.")

    job = Job(exp_dir.path)
    n_examples_remaining = n_examples

    with NumpySeed(seed):
        inputs = []
        idx = 0
        while n_examples_remaining:
            seed = gen_seed()
            cur_n_examples = min(n_examples_remaining, n_examples_per_shard)
            n_examples_remaining -= cur_n_examples

            inputs.append((idx, seed, cur_n_examples))
            idx += 1

        job.map(_BuildDataset(dataset_cls, param_values), inputs)
        job.save_object('metadata', 'param_values', param_values)

    print(job.summary())
    archive_path = job.zip(delete=True)
    print("Zipped {} as {}.".format(exp_dir.path, archive_path))

    run_kwargs = run_kwargs.copy()

    del run_kwargs['n_examples_per_shard']

    run_kwargs.update(
        archive_path=archive_path, name=name, kind="parallel",
        parallel_exe=cfg.parallel_exe)
    parallel_session = submit_job(**run_kwargs)

    with cd(os.path.join(parallel_session.job_path, 'experiments')):
        dataset_files = []
        for dir_path, dirs, files in os.walk('.'):
            if not dir_path.startswith("./exp__seed="):
                continue

            df = [f for f in files if not f.endswith('.cfg')]
            assert len(df) == 1
            dataset_files.append(os.path.join(dir_path, df[0]))

        cached_filename = os.path.join(
            cfg.data_dir, "cached_datasets",
            dataset_cls.__name__, str(get_param_hash(param_values)))

        command = "cat " + " ".join(dataset_files) + " > " + cached_filename
        print("Running command: \n" + command)
        subprocess.run(command, shell=True, check=True)
        print("Done.")

        with open(cached_filename + ".cfg", 'w') as f:
            f.write(pprint.pformat(param_values))

    return parallel_session
