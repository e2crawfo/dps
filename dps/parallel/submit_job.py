from __future__ import print_function
import os
import datetime
import subprocess
from future.utils import raise_with_traceback
from pathlib import Path
import numpy as np
import time
import progressbar
import shutil
from collections import defaultdict
import sys

import dps
from dps.parallel.base import ReadOnlyJob, zip_root
from dps.utils import cd, parse_timedelta, make_symlink, ExperimentStore


DEFAULT_HOST_POOL = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 33)]


class ParallelSession(object):
    """ Run a Job in parallel using gnu-parallel.

    A directory for this Job execution is created in `scratch`, and results are saved there.

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
        String specifying the amount of cleanup time to allow per step.
    slack_time: float
        String specifying the amount of slack time to allow per step. Corresponds to
        time allotted to each process to respond to the signal that the step's time is up.
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
    redirect: bool
        If True, stderr and stdout of jobs is saved in files rather than being printed to screen.
    n_retries: int
        Number of retries per job.
    gpu_set: str
        Comma-separated list of indices of gpus to use.
    step_time_limit: str
        String specifying time limit for each step. If not supplied, a time limit is inferred
        automatically based on wall_time and number of steps (giving each step an equal amount
        of time).
    ignore_gpu: bool
        If True, GPUs will be requested by as part of the job, but will not be used at run time.
    store_experiments: bool
        If True, after each step we fetch experiment data from each of the nodes and store it locally
        along with the results. Either way, experiment data is deleted from nodes after each step.
    ssh_options: string
        String of options to pass to ssh.

    """
    def __init__(
            self, name, input_zip, pattern, scratch, local_scratch_prefix='/tmp/dps/hyper/', ppn=12, cpp=1,
            pmem=None, wall_time="1hour", cleanup_time="1min", slack_time="1min", add_date=True, dry_run=0,
            parallel_exe=None, kind="parallel", host_pool=None, load_avg_threshold=8.,
            min_hosts=None, max_hosts=1, env_vars=None, redirect=False, n_retries=0, gpu_set="",
            step_time_limit="", ignore_gpu=False, store_experiments=True, ssh_options=None, readme=""):

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
        elif "slurm" in kind:
            local_scratch_prefix = "/tmp/dps/hyper"

        assert kind in "parallel pbs slurm slurm-local".split()
        hpc = kind != "parallel"

        # Create directory to run the job from - should be on scratch.
        scratch = os.path.abspath(os.path.expandvars(str(scratch)))
        es = ExperimentStore(str(scratch), max_experiments=None, delete_old=False)
        exp_dir = es.new_experiment(name, 0, add_date=add_date, force_fresh=1)
        exp_dir.record_environment(git_modules=[dps])

        job_directory = str(exp_dir.path)

        os.makedirs(os.path.realpath(job_directory + "/experiments"))

        if readme:
            with open(os.path.join(job_directory, 'README.md'), 'w') as f:
                f.write(readme)
            del f

        input_zip_stem = Path(input_zip).stem
        input_zip = shutil.copy(str(input_zip), os.path.join(job_directory, "orig.zip"))
        input_zip = Path(input_zip)
        input_zip_abs = input_zip.resolve()
        input_zip_base = input_zip.name
        archive_root = zip_root(input_zip)

        # storage local to each node, from the perspective of that node
        local_scratch = str(Path(local_scratch_prefix) / Path(job_directory).name)

        redirect = "--redirect" if redirect else ""

        env = os.environ.copy()

        env_vars = env_vars or {}

        env.update({e: str(v) for e, v in env_vars.items()})
        env_vars = ' '.join('--env ' + k for k in env_vars)

        ro_job = ReadOnlyJob(input_zip)
        indices_to_run = sorted([op.idx for op in ro_job.ready_incomplete_ops(sort=False)])
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
            with cd(job_directory):
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

        wall_time = parse_timedelta(wall_time)
        wall_time_seconds = int(wall_time.total_seconds())

        cleanup_time_per_step = parse_timedelta(cleanup_time)
        cleanup_seconds = int(cleanup_time_per_step.total_seconds())

        slack_time_per_step = parse_timedelta(slack_time)
        slack_seconds = int(slack_time_per_step.total_seconds())

        assert wall_time_seconds > 0
        assert cleanup_seconds > 0
        assert slack_seconds > 0

        if step_time_limit:
            total_seconds_per_step = int(parse_timedelta(step_time_limit).total_seconds())
        else:
            total_seconds_per_step = int(np.floor(wall_time_seconds / n_steps))

        # Subtract cleanup time and wall time.
        parallel_seconds_per_step = int(total_seconds_per_step - cleanup_seconds)
        python_seconds_per_step = int(
            total_seconds_per_step - cleanup_seconds - slack_seconds)

        self.__dict__.update(locals())

        self.print_time_limits()

        assert total_seconds_per_step > 0
        assert parallel_seconds_per_step > 0
        assert python_seconds_per_step > 0

        # Create convenience `latest` symlinks
        make_symlink(job_directory, os.path.join(scratch, 'latest'))

    def get_load_avg(self, host, ):
        return_code, stdout = self.ssh_execute("uptime", host, output='get')
        return [float(s) for s in stdout.split(':')[-1].split(',')]

    def print_time_limits(self):
        print("We have {wall_time_seconds} seconds to complete {n_jobs_to_run} "
              "sub-jobs (grouped into {n_steps} steps) using {n_procs} processors.".format(**self.__dict__))
        print("Total time per step is {total_seconds_per_step}.".format(**self.__dict__))
        print("Each step, we are allowing {slack_seconds} seconds as slack and {cleanup_seconds} seconds for cleanup.".format(**self.__dict__))
        print("Time-limit passed to parallel is {parallel_seconds_per_step}.".format(**self.__dict__))
        print("Time-limit passed to dps-hyper is {python_seconds_per_step}.".format(**self.__dict__))

    def recruit_hosts(self, hpc, min_hosts, max_hosts, host_pool, ppn, max_procs):

        if getattr(self, 'candidate_hosts', None) is None:
            print("Ranking hosts by suitability...")
            candidate_hosts = {}
            for host in host_pool:
                if host is not ':':
                    print("\n" + "~" * 40)
                    print("Testing connection to host {}...".format(host))
                    failed = self.ssh_execute("echo Connected to \$HOSTNAME", host, robust=True)
                    if failed:
                        print("Could not connect.")
                        continue

                    # print("\nTOP:")
                    # return_code = self.ssh_execute("top -bn2 | head -n 5", host, output='loud')
                    _, load_avg, _ = self.get_load_avg(host)
                    print("5 minute load average: {}".format(load_avg))

                    if load_avg < self.load_avg_threshold:
                        candidate_hosts[host] = load_avg
                    else:
                        print("`load_avg` above threshold of {}, discarding host.".format(self.load_avg_threshold))
                else:
                    candidate_hosts[host] = 0.0
            self.candidate_hosts = candidate_hosts

        hosts = []
        candidate_hosts = sorted(self.candidate_hosts, key=self.candidate_hosts.__getitem__)

        for host in candidate_hosts:
            n_hosts_recruited = len(hosts)
            if n_hosts_recruited >= max_hosts:
                break

            if n_hosts_recruited * ppn >= max_procs:
                break

            print("\n" + ("~" * 40))
            print("Recruiting host {}...".format(host))

            _, load_avg, _ = self.get_load_avg(host)
            print("Previous 5 minute load average: {}".format(self.candidate_hosts[host]))
            print("Recalculated 5 minute load average: {}".format(load_avg))

            self.candidate_hosts[host] = load_avg

            print("Preparing host...")
            try:
                if host is ':':
                    command = "stat {local_scratch}"
                    create_local_scratch = self.execute_command(command, robust=True)

                    if create_local_scratch:
                        print("Creating local scratch directory...")
                        self.dirty_hosts.add(host)
                        command = "mkdir -p {local_scratch}"
                        self.execute_command(command, robust=False)

                    command = "cd {local_scratch} && stat {archive_root}"
                    missing_archive = self.execute_command(command, robust=True)

                    if missing_archive:
                        command = "cd {local_scratch} && stat {input_zip_base}"
                        missing_zip = self.execute_command(command, robust=True)

                        if missing_zip:
                            print("Copying zip to local scratch...")
                            command = "cp {input_zip_abs} {local_scratch}".format(**self.__dict__)
                            self.execute_command(command, frmt=False, robust=False)

                        print("Unzipping...")
                        command = "cd {local_scratch} && unzip -ouq {input_zip_base}"
                        self.execute_command(command, robust=False)

                else:
                    command = "stat {local_scratch}"
                    create_local_scratch = self.ssh_execute(command, host, robust=True)

                    if create_local_scratch:
                        print("Creating local scratch directory...")
                        command = "mkdir -p {local_scratch}"
                        self.dirty_hosts.add(host)
                        self.ssh_execute(command, host, robust=False)

                    command = "cd {local_scratch} && stat {archive_root}"
                    missing_archive = self.ssh_execute(command, host, robust=True)

                    if missing_archive:
                        command = "cd {local_scratch} && stat {input_zip_base}"
                        missing_zip = self.ssh_execute(command, host, robust=True)

                        if missing_zip:
                            print("Copying zip to local scratch...")
                            command = (
                                "rsync -av -e \"ssh {ssh_options}\" "
                                "{input_zip_abs} {host}:{local_scratch}".format(host=host, **self.__dict__)
                            )
                            self.execute_command(command, frmt=False, robust=False)

                        print("Unzipping...")
                        command = "cd {local_scratch} && unzip -ouq {input_zip_base}"
                        self.ssh_execute(command, host, robust=False)

                print("Host successfully prepared.")
                hosts.append(host)

            except subprocess.CalledProcessError:
                print("Preparation of host failed.")

        if len(hosts) < (min_hosts if min_hosts is not None else max_hosts):
            raise Exception(
                "Found only {} usable hosts, but minimum "
                "required hosts is {}.".format(len(hosts), min_hosts))

        n_procs = ppn * len(hosts)
        print("Proceeding with {} usable hosts, translates into {} procs "
              "(max_procs: {}, max_hosts: {}).".format(
                  len(hosts), n_procs, max_procs, max_hosts))

        with open('nodefile.txt', 'w') as f:
            f.write('\n'.join(hosts))

        return hosts, n_procs

    def execute_command(
            self, command, frmt=True, shell=True, robust=False, max_seconds=None,
            progress=False, verbose=False, output='quiet'):
        """ Uses `subprocess` to execute `command`. """

        p = None
        try:
            assert isinstance(command, str)
            if frmt:
                command = command.format(**self.__dict__)

            if verbose:
                print("\nExecuting command: " + (">" * 40) + "\n")
                print(command)

            if not shell:
                command = command.split()

            if output == 'quiet':
                stdout = subprocess.DEVNULL
            elif output == 'get':
                stdout = subprocess.PIPE
            elif output == 'loud':
                stdout = None

            stderr = subprocess.STDOUT

            start = time.time()

            sys.stdout.flush()
            sys.stderr.flush()

            p = subprocess.Popen(command, shell=shell, universal_newlines=True, stdout=stdout, stderr=stderr)

            if progress:
                progress = progressbar.ProgressBar(
                    widgets=['[', progressbar.Timer(), '] ', '(', progressbar.ETA(), ') ', progressbar.Bar()],
                    max_value=max_seconds or progressbar.UnknownLength, redirect_stdout=True)
            else:
                progress = None

            interval_length = 1
            while True:
                try:
                    p.wait(interval_length)
                except subprocess.TimeoutExpired:
                    if progress is not None:
                        progress.update(min(int(time.time() - start), max_seconds))

                if p.returncode is not None:
                    break

            if progress is not None:
                progress.finish()

            if verbose:
                print("\nCommand took {} seconds.\n".format(time.time() - start))

            if p.returncode != 0:
                if isinstance(command, list):
                    command = ' '.join(command)

                print("The following command returned with non-zero exit code {}:\n    {}".format(p.returncode, command))

                if robust:
                    if output == 'get':
                        stdout = p.stdout.read()
                        return p.returncode, stdout
                    else:
                        return p.returncode
                else:
                    raise subprocess.CalledProcessError(p.returncode, command)

            if output == 'get':
                stdout = p.stdout.read()
                return p.returncode, stdout
            else:
                return p.returncode
        except BaseException as e:
            if p is not None:
                p.terminate()
                p.kill()
            if progress is not None:
                progress.finish()
            raise_with_traceback(e)

    def ssh_execute(self, command, host, **kwargs):
        cmd = "ssh {ssh_options} -T {host} \"{command}\"".format(
            ssh_options=self.ssh_options, host=host, command=command)
        return self.execute_command(cmd, **kwargs)

    def _step(self, i, indices_for_step):
        if not indices_for_step:
            print("No jobs left to run on step {}.".format(i))
            return

        _ignore_gpu = "--ignore-gpu" if self.ignore_gpu else ""

        indices = ' '.join(str(i) for i in indices_for_step)

        if "slurm" in self.kind:
            parallel_command = (
                "cd {local_scratch} && "
                "dps-hyper run {archive_root} {pattern} {indices} --max-time {python_seconds_per_step} "
                "--log-root {local_scratch} --log-name experiments --gpu-set={gpu_set} --ppn={ppn} {_ignore_gpu} {redirect}"
            )

            bind = "--accel-bind=g" if self.gpu_set else ""
            mem = "--mem-per-cpu={}mb".format(self.pmem) if self.pmem else ""
            command = 'srun --cpus-per-task {cpp} --ntasks {n_tasks} {bind} {mem} --no-kill sh -c "{parallel_command}"'.format(
                cpp=self.cpp, n_tasks=len(indices_for_step), bind=bind, mem=mem, parallel_command=parallel_command)
            print(command)
        else:
            parallel_command = (
                "cd {local_scratch} && "
                "dps-hyper run {archive_root} {pattern} {{}} --max-time {python_seconds_per_step} "
                "--log-root {local_scratch} --log-name experiments "
                "--idx-in-node={{%}} --gpu-set={gpu_set} --ppn={ppn} {_ignore_gpu} {redirect}"
            )

            command = (
                '{parallel_exe} --timeout {parallel_seconds_per_step} --no-notice -j{ppn} \\\n'
                '    --joblog {job_directory}/job_log.txt {node_file} \\\n'
                '    --env PATH --env LD_LIBRARY_PATH {env_vars} -v \\\n'
                '    "' + parallel_command + '" \\\n'
                '    ::: {indices}'
            )

        command = command.format(
            indices=indices, _ignore_gpu=_ignore_gpu, **self.__dict__)

        self.execute_command(
            command, frmt=False, robust=True,
            max_seconds=self.parallel_seconds_per_step, progress=not self.hpc,
            output='loud', verbose=True)

    def _checkpoint(self, i):
        print("Fetching results of step {} at: ".format(i))
        print(datetime.datetime.now())

        for i, host in enumerate(self.hosts):
            if host is ':':
                if self.store_experiments:
                    command = "mv {local_scratch}/experiments/* ./experiments"
                    self.execute_command(command, robust=True)

                command = "rm -rf {local_scratch}/experiments"
                self.execute_command(command, robust=True)

                command = "cp -ru {local_scratch}/{archive_root} ."
                self.execute_command(command, robust=True)
            else:
                if self.store_experiments:
                    command = (
                        "rsync -avz -e \"ssh {ssh_options}\" "
                        "{host}:{local_scratch}/experiments/ ./experiments".format(
                            host=host, **self.__dict__)
                    )
                    self.execute_command(command, frmt=False, robust=True)

                command = "rm -rf {local_scratch}/experiments"
                self.ssh_execute(command, host, robust=True)

                command = (
                    "rsync -avz -e \"ssh {ssh_options}\" "
                    "{host}:{local_scratch}/{archive_root} .".format(
                        host=host, **self.__dict__)
                )
                self.execute_command(command, frmt=False, robust=True)

        self.execute_command("zip -rq results {archive_root}", robust=True)
        self.execute_command(
            "dps-hyper summary --no-config results.zip | tee results.txt", robust=True, output='loud')

    def run(self):
        if self.dry_run:
            print("Dry run, so not running.")
            return

        self.print_time_limits()

        with cd(self.job_directory):
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

                print("Starting step {} at: ".format(i) + "=" * 90)
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
            if host is ':':
                self.execute_command(command, robust=True)
            else:
                self.ssh_execute(command, host, robust=True)
