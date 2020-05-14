from __future__ import print_function
import os
import datetime
import subprocess
import numpy as np
import time
import shutil
from collections import defaultdict
import sys
import dill
from zipfile import ZipFile
from contextlib import ExitStack
import json

from dps import cfg
from dps.parallel import ReadOnlyJob
from dps.utils import (
    cd, parse_timedelta, ExperimentStore,
    zip_root, process_path, path_stem, redirect_stream
)


DEFAULT_HOST_POOL = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]


class ParallelSession(object):
    """ Run a Job in parallel using gnu-parallel.

    A directory for this job execution is created in `scratch`, and results are saved there.

    Parameters
    ----------
    name: str
        Name for the experiment.
    input_zip: str
        Path to a zip archive storing the Job.
    pattern: str
        Pattern to use to select which ops to run within the Job.
    scratch: str
        Path to location where the results of running the selected ops will be
        written. Must be writeable by the master process.
    local_scratch_prefix: str
        Path to scratch directory that is local to each remote host.
    ppn: int
        Number of processors per node.
    wall_time: str
        String specifying the maximum wall-time allotted to running the selected ops.
    cleanup_time: str
        String specifying the amount of cleanup time to allow per step. Affects the time-limit
        that we pass to `gnu-parallel`, as well as the time limit passed to the python script.
    slack_time: float
        String specifying the amount of slack time to allow per step. Corresponds to
        time allotted to each process to respond to the signal that the step's time is up.
        Affects the time limit that we pass to the python script.
    add_date: bool
        Whether to add current date/time to the name of the directory where results are stored.
    dry_run: bool
        If True, control script will be generated but not executed/submitted.
    parallel_exe: str
        Path to the `gnu-parallel` executable to use.
    host_pool: list of str
        A list of names of hosts to use to execute the job.
    load_avg_threshold: float
        If a host exhibits a load average greater than this, it will not be used.
    max_hosts: int
        Maximum number of hosts to use.
    env_vars: dict (str -> str)
        Dictionary mapping environment variable names to values. These will be accessible
        by the submit script, and will also be sent to the worker nodes.
    output_to_files: bool
        If True, stderr and stdout of jobs is saved in files rather than being printed to screen.
    n_retries: int
        Number of retries per job.
    gpu_set: str
        Comma-separated list of indices of gpus to use.
    copy_venv: bool
        If True, copy the virtualenv from the launching environment and use it to run the simulation.
    python_startup: bool
        If True, source script located at "$HOME/python_startup.sh" before running step command.
    step_time_limit: str
        String specifying time limit for each step. If not supplied, a time limit is inferred
        automatically based on wall_time and number of steps (giving each step an equal amount
        of time).
    ignore_gpu: bool
        If True, GPUs will be requested by as part of the job, but will not be used at run time.
    ssh_options: string
        String of options to pass to ssh.
    loud_output: bool
        Whether to capture stdout for the main execution command.

    """
    def __init__(
            self, name, input_zip, pattern, scratch, local_scratch_prefix='/tmp/dps/hyper/', ppn=12, cpp=1,
            pmem=None, wall_time="1hour", cleanup_time="1min", slack_time="1min", add_date=True, dry_run=0,
            parallel_exe=None, kind="parallel", host_pool=None, load_avg_threshold=8., min_hosts=None,
            max_hosts=1, env_vars=None, output_to_files=True, n_retries=0, gpu_set="", copy_venv="",
            python_startup=False, step_time_limit=None, ignore_gpu=False, ssh_options=None, loud_output=True,
            rsync_verbosity=0, copy_locally=True):

        args = locals().copy()
        del args['self']

        print("\nParallelSession args:")
        print(args)

        launch_venv = os.getenv('VIRTUAL_ENV')
        if launch_venv:
            launch_venv = os.path.split(launch_venv)[1]

        if not parallel_exe:
            parallel_exe = "$HOME/.local/bin/parallel"

        if ssh_options is None:
            ssh_options = (
                "-oPasswordAuthentication=no "
                "-oStrictHostKeyChecking=no "
                "-oConnectTimeout=5 "
                "-oServerAliveInterval=2"
            )

        if kind == "pbs":
            local_scratch_prefix = "\\$RAMDISK"

        assert kind in "parallel pbs slurm slurm-local".split()
        hpc = kind != "parallel"

        # Create directory to run the job from - should be on scratch.
        scratch = os.path.abspath(os.path.expandvars(scratch))

        es = ExperimentStore(scratch, prefix="run")

        job_dir = es.new_experiment(name, 0, add_date=add_date, force_fresh=1)
        job_dir.record_environment()

        with open(job_dir.path_for('run_kwargs.json'), 'w') as f:
            json.dump(args, f, default=str, indent=4, sort_keys=True)
        del f
        del args

        job_path = job_dir.path
        job_dir.make_directory('experiments')

        input_zip_stem = path_stem(input_zip)
        input_zip = shutil.copy(input_zip, job_dir.path_for("orig.zip"))
        input_zip_abs = process_path(input_zip)
        input_zip_base = os.path.basename(input_zip)
        archive_root = zip_root(input_zip)

        self.copy_files(
            job_dir, input_zip, archive_root,
            ["README.md", "sampled_configs.txt", "config.json", "config.pkl"])

        # storage local to each node, from the perspective of that node
        local_scratch = os.path.join(local_scratch_prefix, os.path.basename(job_path))

        output_to_files = "--output-to-files" if output_to_files else ""

        env = os.environ.copy()

        env_vars = env_vars or {}

        env.update({e: str(v) for e, v in env_vars.items()})
        env_vars = ' '.join('--env ' + k for k in env_vars)

        rsync_verbosity = "" if not rsync_verbosity else "-" + "v" * rsync_verbosity

        ro_job = ReadOnlyJob(input_zip)
        indices_to_run = sorted([op.idx for op in ro_job.ready_incomplete_ops(sort=False)])
        del ro_job
        n_jobs_to_run = len(indices_to_run)
        if n_jobs_to_run == 0:
            print("All jobs are finished! Exiting.")
            return

        dirty_hosts = set()

        if hpc:
            host_pool = []
            n_nodes = max_hosts
            n_procs = n_nodes * ppn
            n_steps = int(np.ceil(n_jobs_to_run / n_procs))
        else:
            self.__dict__.update(locals())

            host_pool = host_pool or DEFAULT_HOST_POOL
            if isinstance(host_pool, str):
                host_pool = host_pool.split()

            # Get an estimate of the number of hosts we'll have available.
            with cd(job_path):
                hosts, n_procs = self.recruit_hosts(
                    hpc, min_hosts, max_hosts, host_pool,
                    ppn, max_procs=np.inf)
            n_nodes = len(hosts)

            if n_jobs_to_run < n_procs:
                n_steps = 1
                n_nodes = int(np.ceil(n_jobs_to_run / ppn))
                n_procs = n_nodes * ppn
                hosts = hosts[:n_nodes]
            else:
                n_steps = int(np.ceil(n_jobs_to_run / n_procs))

        node_file = " --sshloginfile nodefile.txt "

        wall_time_seconds, total_seconds_per_step, parallel_seconds_per_step, python_seconds_per_step = \
            self.compute_time_limits(wall_time, cleanup_time, slack_time, step_time_limit, n_steps)

        self.__dict__.update(locals())

        self.print_time_limits()

    def get_load_avg(self, host):
        return_code, stdout, stderr = self.ssh_execute("uptime", host, robust=True)
        print(stdout)
        if return_code:
            return 1000.0, 1000.0, 1000.0
        return [float(s) for s in stdout.split(':')[-1].split(',')]

    def print_time_limits(self):
        print("\n" + "~" * 40)
        print("We have {wall_time_seconds} seconds to complete {n_jobs_to_run} "
              "sub-jobs (grouped into {n_steps} steps) using {n_procs} processors.".format(**self.__dict__))
        print("Each step, we are allowing {slack_time} as slack and "
              "{cleanup_time} for cleanup.".format(**self.__dict__))
        print("Total time per step is {total_seconds_per_step} seconds.".format(**self.__dict__))
        print("Time-limit passed to parallel is {parallel_seconds_per_step} seconds.".format(**self.__dict__))
        print("Time-limit passed to dps-hyper is {python_seconds_per_step} seconds.".format(**self.__dict__))

    @staticmethod
    def compute_time_limits(wall_time, cleanup_time_per_step, slack_time_per_step, step_time_limit, n_steps):
        if isinstance(wall_time, str):
            wall_time = int(parse_timedelta(wall_time).total_seconds())
        assert isinstance(wall_time, int)
        assert wall_time > 0

        if isinstance(cleanup_time_per_step, str):
            cleanup_time_per_step = int(parse_timedelta(cleanup_time_per_step).total_seconds())
        assert isinstance(cleanup_time_per_step, int)
        assert cleanup_time_per_step > 0

        if isinstance(slack_time_per_step, str):
            slack_time_per_step = int(parse_timedelta(slack_time_per_step).total_seconds())
        assert isinstance(slack_time_per_step, int)
        assert slack_time_per_step > 0

        if step_time_limit is None:
            total_seconds_per_step = int(np.floor(wall_time / n_steps))
        else:
            if isinstance(step_time_limit, str):
                step_time_limit = int(parse_timedelta(step_time_limit).total_seconds())
            assert isinstance(step_time_limit, int)
            assert step_time_limit > 0

            total_seconds_per_step = step_time_limit

        # Subtract cleanup time and wall time.
        parallel_seconds_per_step = int(total_seconds_per_step - cleanup_time_per_step)
        python_seconds_per_step = int(
            total_seconds_per_step - cleanup_time_per_step - slack_time_per_step)

        assert total_seconds_per_step > 0
        assert parallel_seconds_per_step > 0
        assert python_seconds_per_step > 0

        return wall_time, total_seconds_per_step, parallel_seconds_per_step, python_seconds_per_step

    @staticmethod
    def copy_files(job_dir, input_zip, archive_root, filenames):
        # Copy files from archive
        with ZipFile(input_zip, 'r') as _input_zip:
            for filename in filenames:
                name_in_zip = os.path.join(archive_root, filename)
                text = None
                try:
                    text = _input_zip.read(name_in_zip).decode()
                except Exception:
                    print("No {} found in zip file.".format(filename))

                if text is not None:
                    with open(job_dir.path_for(filename), 'w') as f:
                        f.write(text)

    def recruit_hosts(self, hpc, min_hosts, max_hosts, host_pool, ppn, max_procs):
        if not hpc and getattr(self, 'candidate_hosts', None) is None:
            print("Ranking hosts by suitability...")
            candidate_hosts = {}
            for host in host_pool:
                if host != ':':
                    print("\n" + "~" * 40)
                    print("Testing connection to host {}...".format(host))
                    failed, _, _ = self.ssh_execute("echo Connected to \$HOSTNAME", host, robust=True)
                    if failed:
                        print("Could not connect.")
                        continue

                load_avg, _, _ = self.get_load_avg(host)
                print("1 minute load average: {}".format(load_avg))

                if load_avg < self.load_avg_threshold:
                    candidate_hosts[host] = load_avg
                else:
                    print("`load_avg` above threshold of {}, discarding host.".format(self.load_avg_threshold))

            self.candidate_hosts = candidate_hosts

        hosts = []

        if hpc:
            candidate_hosts = host_pool
        else:
            candidate_hosts = sorted(self.candidate_hosts, key=self.candidate_hosts.__getitem__)

        for host in candidate_hosts:
            n_hosts_recruited = len(hosts)
            if n_hosts_recruited >= max_hosts:
                break

            if n_hosts_recruited * ppn >= max_procs:
                break

            print("\n" + ("~" * 40))
            print("Recruiting host {}...".format(host))

            if not hpc:
                load_avg, _, _ = self.get_load_avg(host)
                print("Previous 1 minute load average: {}".format(self.candidate_hosts[host]))
                print("Recalculated 1 minute load average: {}".format(load_avg))
                self.candidate_hosts[host] = load_avg

            print("Preparing host...")
            try:
                command = "stat {local_scratch}"
                create_local_scratch, _, _ = self.ssh_execute(command, host, robust=True, output="quiet")

                if create_local_scratch:
                    print("Creating local scratch directory...")
                    command = "mkdir -p {local_scratch}"
                    self.ssh_execute(command, host, robust=False)
                    self.dirty_hosts.add(host)

                command = "cd {local_scratch} && stat {archive_root}"
                missing_archive, _, _ = self.ssh_execute(command, host, robust=True, output="quiet")

                if missing_archive:
                    command = "cd {local_scratch} && stat {input_zip_base}"
                    missing_zip, _, _ = self.ssh_execute(command, host, robust=True, output="quiet")

                    if missing_zip:
                        print("Copying zip to local scratch...")
                        if host == ':':
                            command = "cp {input_zip_abs} {local_scratch}".format(**self.__dict__)
                        else:
                            command = (
                                "rsync -a {rsync_verbosity} --timeout=300 -e \"ssh {ssh_options}\" "
                                "{input_zip_abs} {host}:{local_scratch}".format(host=host, **self.__dict__)
                            )
                        self.execute_command(command, frmt=False, robust=False)

                    print("Unzipping...")
                    command = "cd {local_scratch} && unzip -ouq {input_zip_base}"
                    self.ssh_execute(command, host, robust=False)

                print("Host successfully prepared.")
                hosts.append(host)

            except subprocess.CalledProcessError as e:
                print("Preparation of host failed.")
                print("Command output:\n{}".format(e.output))

        if min_hosts is not None and len(hosts) < min_hosts:
            raise Exception(
                "Found only {} usable hosts, but minimum "
                "required hosts is {}.".format(len(hosts), min_hosts))

        n_procs = ppn * len(hosts)

        print("\nProceeding with {} usable hosts, translates into {} procs total "
              "(max_procs: {}, max_hosts: {}).".format(
                  len(hosts), n_procs, max_procs, max_hosts))

        with open('nodefile.txt', 'w') as f:
            f.write('\n'.join(hosts))

        return hosts, n_procs

    def execute_command(
            self, command, frmt=True, shell=True, max_seconds=None, robust=False, output=None):
        """ Uses `subprocess` to execute `command`. Has a few added bells and whistles.

        if command returns non-zero exit status:
            if robust:
                returns as normal
            else:
                raise CalledProcessError

        Parameters
        ----------
        command: str
            The command to execute.


        Returns
        -------
        returncode, stdout, stderr

        """
        p = None
        try:
            assert isinstance(command, str)
            if frmt:
                command = command.format(**self.__dict__)

            if output == "loud":
                print("\nExecuting command: " + (">" * 40) + "\n")
                print(command)

            if not shell:
                command = command.split()

            stdout = None if output == "loud" else subprocess.PIPE
            stderr = None if output == "loud" else subprocess.PIPE

            start = time.time()

            sys.stdout.flush()
            sys.stderr.flush()

            p = subprocess.Popen(command, shell=shell, universal_newlines=True,
                                 stdout=stdout, stderr=stderr)

            interval_length = 1
            while True:
                try:
                    p.wait(interval_length)
                except subprocess.TimeoutExpired:
                    pass

                if p.returncode is not None:
                    break

            if output == "loud":
                print("\nCommand took {} seconds.\n".format(time.time() - start))

            _stdout = "" if p.stdout is None else p.stdout.read()
            _stderr = "" if p.stderr is None else p.stderr.read()

            if p.returncode != 0:
                if isinstance(command, list):
                    command = ' '.join(command)

                print("The following command returned with non-zero exit code "
                      "{}:\n    {}".format(p.returncode, command))

                if output is None or (output == "quiet" and not robust):
                    print("\n" + "-" * 20 + " stdout " + "-" * 20 + "\n")
                    print(_stdout)

                    print("\n" + "-" * 20 + " stderr " + "-" * 20 + "\n")
                    print(_stderr)

                if robust:
                    return p.returncode, _stdout, _stderr
                else:
                    raise subprocess.CalledProcessError(p.returncode, command, _stdout, _stderr)

            return p.returncode, _stdout, _stderr

        except BaseException as e:
            if p is not None:
                p.terminate()
                p.kill()
            raise e

    def ssh_execute(self, command, host, **kwargs):
        if host == ":":
            cmd = command
        else:
            cmd = "ssh {ssh_options} -T {host} \"{command}\"".format(
                ssh_options=self.ssh_options, host=host, command=command)
        return self.execute_command(cmd, **kwargs)

    def _step(self, i, indices_for_step):
        if not indices_for_step:
            print("No jobs left to run on step {}.".format(i))
            return

        _ignore_gpu = "--ignore-gpu" if self.ignore_gpu else ""
        _copy_locally = "--copy-dataset-to={}".format(self.local_scratch_prefix) if self.copy_locally else ""
        _backup_dir = self.job_dir.path_for('experiments')

        indices = ' '.join(str(i) for i in indices_for_step)

        if "slurm" in self.kind:
            parallel_command = (
                "cd {local_scratch} && "
                "dps-hyper run {archive_root} {pattern} {indices} --max-time {python_seconds_per_step} "
                "--local-experiments-dir {local_scratch} --backup-dir {_backup_dir} --env-name experiments "
                "--gpu-set={gpu_set} --ppn={ppn} {_ignore_gpu} {_copy_locally} {output_to_files}"
            )

            bind = "--accel-bind=g" if self.gpu_set else ""
            mem = "--mem-per-cpu={}mb".format(self.pmem) if self.pmem else ""

            command = ('timeout --signal=INT {parallel_seconds_per_step} srun --cpus-per-task {cpp} --ntasks {n_tasks} '
                       '{bind} {mem} --no-kill --quit-on-interrupt sh -c "{parallel_command}"'.format(
                           parallel_seconds_per_step=self.parallel_seconds_per_step,
                           cpp=self.cpp,
                           n_tasks=len(indices_for_step),
                           bind=bind,
                           mem=mem,
                           parallel_command=parallel_command))
        else:
            workon = "workon {launch_venv} && " if (self.copy_venv and self.launch_venv) else ""
            python_startup = "source \$HOME/python_startup.sh && " if self.python_startup else ""
            parallel_command = (
                python_startup +
                workon +
                "cd {local_scratch} && "
                "dps-hyper run {archive_root} {pattern} {{}} --max-time {python_seconds_per_step} "
                "--local-experiments-dir {local_scratch} -backup-dir {_backup_dir} --env-name experiments "
                "--idx-in-node={{%}} --gpu-set={gpu_set} --ppn={ppn} {_ignore_gpu} {_copy_locally} {output_to_files}"
            )

            command = (
                '{parallel_exe} --timeout {parallel_seconds_per_step} --no-notice -j{ppn} \\\n'
                '    --joblog {job_path}/job_log.txt {node_file} \\\n'
                '    {env_vars} -v \\\n'
                # '    --env PATH --env LD_LIBRARY_PATH {env_vars} -v \\\n'
                '    "' + parallel_command + '" \\\n'
                '    ::: {indices}'
            )

        command = command.format(
            indices=indices, _ignore_gpu=_ignore_gpu,
            _copy_locally=_copy_locally, _backup_dir=_backup_dir,
            **self.__dict__)

        self.execute_command(
            command, frmt=False, robust=True,
            max_seconds=self.parallel_seconds_per_step,
            output='loud' if self.loud_output else None)

    def _checkpoint(self, i):
        print("Fetching results of step {} at: ".format(i))
        print(datetime.datetime.now())

        for i, host in enumerate(self.hosts):
            if host == ':':
                command = "mv {local_scratch}/experiments/* ./experiments"
                self.execute_command(command, robust=True)

                command = "rm -rf {local_scratch}/experiments"
                self.execute_command(command, robust=True)

                command = "cp -ru {local_scratch}/{archive_root} ."
                self.execute_command(command, robust=True)
            else:
                command = (
                    "rsync -az {rsync_verbosity} --timeout=300 -e \"ssh {ssh_options}\" "
                    "{host}:{local_scratch}/experiments/ ./experiments".format(
                        host=host, **self.__dict__)
                )
                self.execute_command(command, frmt=False, robust=True, output="loud")

                command = "rm -rf {local_scratch}/experiments"
                self.ssh_execute(command, host, robust=True, output="loud")

                command = (
                    "rsync -az {rsync_verbosity} --timeout=300 -e \"ssh {ssh_options}\" "
                    "{host}:{local_scratch}/{archive_root} .".format(
                        host=host, **self.__dict__)
                )
                self.execute_command(command, frmt=False, robust=True, output="loud")

        self.execute_command("zip -rq results {archive_root}", robust=True)

        try:
            from dps.hyper import HyperSearch
            search = HyperSearch('.')
            with redirect_stream('stdout', 'results.txt', tee=False):
                search.print_summary(print_config=False, verbose=False)
            print(search.job.summary(verbose=False))
        except Exception:
            job_path = 'results.zip' if os.path.exists('results.zip') else 'orig.zip'
            assert os.path.exists(job_path)
            job = ReadOnlyJob(job_path)
            print(job.summary(verbose=False))

    def get_slurm_var(self, var_name):
        parallel_command = "printenv | grep {}".format(var_name)
        command = 'srun --ntasks 1 --no-kill sh -c "{parallel_command}"'.format(parallel_command=parallel_command)
        returncode, stdout, stderr = self.execute_command(command, frmt=False, robust=False)
        split = stdout.split('=')

        if len(split) != 2:
            raise Exception(
                "Unparseable output while getting SLURM environment "
                "variable {}: {}".format(var_name, stdout))

        _var_name, value = split
        _var_name = _var_name.strip()
        value = value.strip()

        if _var_name != var_name:
            raise Exception(
                "Got wrong variable. Wanted {}, got {} with value {}".format(var_name, _var_name, value))
        return value

    def run(self):
        with ExitStack() as stack:
            if not self.hpc:
                stack.enter_context(redirect_stream('stdout', self.job_dir.path_for('stdout'), tee=True))
                stack.enter_context(redirect_stream('stderr', self.job_dir.path_for('stderr'), tee=True))

            self._run()

    def _run(self):
        if self.dry_run:
            print("Dry run, so not running.")
            return

        if "slurm" in self.kind:
            # Have to jump through a hoop to get the proper node-local storage on cedar/graham.
            self.local_scratch_prefix = self.get_slurm_var("SLURM_TMPDIR")
            self.local_scratch = os.path.join(
                self.local_scratch_prefix,
                os.path.basename(self.job_path))

            # Compute new time limits based on the actual time remaining (protect against e.g. job starting late)

            print("Time limits before adjustment:")
            self.print_time_limits()

            job_id = os.getenv("SLURM_JOBID")
            command = 'squeue -h -j {} -o "%L"'.format(job_id)
            returncode, stdout, stderr = self.execute_command(command, frmt=False, robust=False)
            days = 0
            if "-" in stdout:
                days, time = stdout.split("-")
                days = int(days)
            else:
                time = stdout

            time = time.split(":")

            hours = int(time[-3]) if len(time) > 2 else 0
            minutes = int(time[-2]) if len(time) > 1 else 0
            seconds = int(time[-1])

            wall_time_delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
            wall_time_seconds = int(wall_time_delta.total_seconds())

            print("Actual remaining walltime: {}".format(wall_time_delta))
            print("Time limits after adjustment:")

            (self.wall_time_seconds, self.total_seconds_per_step,
             self.parallel_seconds_per_step, self.python_seconds_per_step) = \
                self.compute_time_limits(
                    wall_time_seconds, self.cleanup_time, self.slack_time, self.step_time_limit, self.n_steps)

        self.print_time_limits()

        with cd(self.job_path):
            print("\n" + ("=" * 80))
            job_start = datetime.datetime.now()
            print("Starting job at {}".format(job_start))

            job = ReadOnlyJob(self.input_zip)
            subjobs_remaining = sorted([op.idx for op in job.ready_incomplete_ops(sort=False)])

            n_failures = defaultdict(int)
            dead_jobs = set()

            i = 0
            while subjobs_remaining:
                step_start = datetime.datetime.now()

                print("\nStarting step {} at: ".format(i) + "=" * 90)
                print("{} ({} since start of job)".format(step_start, step_start - job_start))

                if not self.host_pool:
                    if self.kind == "pbs":
                        with open(os.path.expandvars("$PBS_NODEFILE"), 'r') as f:
                            self.host_pool = list(set([s.strip() for s in iter(f.readline, '')]))
                            print(self.host_pool)
                    elif "slurm" in self.kind:
                        p = subprocess.run(
                            'scontrol show hostnames $SLURM_JOB_NODELIST', stdout=subprocess.PIPE, shell=True)
                        self.host_pool = list(set([host.strip() for host in p.stdout.decode().split('\n') if host]))
                    else:
                        raise Exception("NotImplemented")

                self.hosts, self.n_procs = self.recruit_hosts(
                    self.hpc, self.min_hosts, self.max_hosts, self.host_pool,
                    self.ppn, max_procs=len(subjobs_remaining))

                indices_for_step = subjobs_remaining[:self.n_procs]
                self._step(i, indices_for_step)
                self._checkpoint(i)

                job = ReadOnlyJob(self.archive_root)

                subjobs_remaining = set([op.idx for op in job.ready_incomplete_ops(sort=False)])

                for j in indices_for_step:
                    if j in subjobs_remaining:
                        n_failures[j] += 1
                        if n_failures[j] > self.n_retries:
                            print("All {} attempts at completing job with index {} have failed, "
                                  "permanently removing it from set of eligible jobs.".format(n_failures[j], j))
                            dead_jobs.add(j)

                subjobs_remaining = [idx for idx in subjobs_remaining if idx not in dead_jobs]
                subjobs_remaining = sorted(subjobs_remaining)

                i += 1

                print("Step duration: {}.".format(datetime.datetime.now() - step_start))

            self.execute_command("rm -rf {archive_root}", robust=True)

        print("Cleaning up dirty hosts...")
        command = "rm -rf {local_scratch}"
        for host in self.dirty_hosts:
            print("Cleaning host {}...".format(host))
            self.ssh_execute(command, host, robust=True)


install_venv_and_run = """

deactivate
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r {requirements}
python {python_entry_point}

"""



def submit_job(
        archive_path, category, exp_name, wall_time="1year", ppn=1, cpp=1, pmem=0,
        queue="", kind="local", gpu_set="", project="rpp-bengioy", **run_kwargs):

    assert kind in "pbs slurm slurm-local parallel".split()

    if "slurm" in kind and not pmem:
        raise Exception("Must supply a value for pmem (per-process-memory in mb) when using SLURM")

    run_kwargs.update(
        wall_time=wall_time, ppn=ppn, cpp=cpp, kind=kind,
        gpu_set=gpu_set, pmem=pmem)

    run_kwargs['env_vars'] = dict(TF_CPP_MIN_LOG_LEVEL=3, CUDA_VISIBLE_DEVICES='-1')
    run_kwargs['dry_run'] = False

    scratch = os.path.join(cfg.parallel_experiments_run_dir, category)

    session = ParallelSession(exp_name, archive_path, 'map', scratch=scratch, **run_kwargs)

    job_path = session.job_path

    # Not strictly required if kind == "parallel", but do it anyway for completeness.
    with open(os.path.join(job_path, "session.pkl"), 'wb') as f:
        dill.dump(session, f, protocol=dill.HIGHEST_PROTOCOL, recurse=True)

    if kind in "parallel slurm-local".split():
        session.run()
        return session

    python_script = """#!{}
import datetime
start = datetime.datetime.now()
print("Starting job at " + str(start))
import dill
with open("./session.pkl", "rb") as f:
    session = dill.load(f)
session.run()
end = datetime.datetime.now()
print("Finishing job at " + str(end))
print(str((end - start).total_seconds()) + " seconds elapsed between start and finish.")

""".format(sys.executable)
    with open(os.path.join(job_path, "run.py"), 'w') as f:
        f.write(python_script)

    if kind == "pbs":
        resources = "nodes={}:ppn={},walltime={}".format(session.n_nodes, session.ppn, session.wall_time_seconds)
        if pmem:
            resources = "{},pmem={}mb".format(resources, pmem)

        email = "eric.crawford@mail.mcgill.ca"
        if queue:
            queue = "-q " + queue
        command = (
            "qsub -N {exp_name} -d {job_path} -w {job_path} -m abe -M {email} "
            "-A {project} {queue} -V -l {resources} "
            "-j oe output.txt run.py".format(
                exp_name=exp_name, job_path=job_path, email=email, project=project,
                queue=queue, resources=resources
            )
        )

    elif kind == "slurm":
        wall_time_minutes = int(np.ceil(session.wall_time_seconds / 60))
        resources = "--nodes={} --ntasks-per-node={} --cpus-per-task={} --time={}".format(
            session.n_nodes, session.ppn, cpp, wall_time_minutes)

        if pmem:
            resources = "{} --mem-per-cpu={}mb".format(resources, pmem)

        if gpu_set:
            n_gpus = len([int(i) for i in gpu_set.split(',')])
            resources = "{} --gres=gpu:{}".format(resources, n_gpus)

        email = "eric.crawford@mail.mcgill.ca"
        if queue:
            queue = "-p " + queue
        command = (
            "sbatch --job-name {exp_name} -D {job_path} --mail-type=ALL --mail-user=wiricon@gmail.com "
            "-A {project} {queue} --export=ALL {resources} "
            "-o stdout -e stderr run.py".format(
                exp_name=exp_name, job_path=job_path, email=email, project=project,
                queue=queue, resources=resources
            )
        )

    else:
        raise Exception()

    print("\n" + "~" * 40)
    print(command)

    with cd(job_path):
        subprocess.run(command.split())
    return session
