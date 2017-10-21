from __future__ import print_function
import os
import datetime
import subprocess
from future.utils import raise_with_traceback
from pathlib import Path
import numpy as np
import glob
import time
import progressbar
import shutil
from collections import defaultdict
import sys

from dps.parallel.base import ReadOnlyJob, zip_root
from dps.utils import cd, parse_timedelta, make_filename, make_symlink


DEFAULT_HOST_POOL = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 32)]


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
        String specifying the amount of time required to clean-up. Job execution will be
        halted when there is this much time left in the overall wall_time, at which point
        results computed so far will be collected.
    add_date: bool
        Whether to add current date/time to the name of the directory where results are stored.
    time_slack: int
        Number of extra seconds to allow per job.
    dry_run: bool
        If True, control script will be generated but not executed/submitted.
    parallel_exe: str
        Path to the `gnu-parallel` executable to use.
    host_pool: list of str
        A list of names of hosts to use to execute the job.
    max_hosts: int
        Maximum number of hosts to use.
    env_vars: dict (str -> str)
        Dictionary mapping environment variable names to values. These will be accessible
        by the submit script, and will also be sent to the worker nodes.
    redirect: bool
        If True, stderr and stdout of jobs is saved in files rather than being printed to screen.
    n_retries: int
        Number of retries per job.
    gpu_set: string
        Comma-separated list of indices of gpus to use.
    enforce_time_limit: bool
        If True, each step has a limited amount of time to finish execution. It is halted if it
        doesn't finish in time. Otherwise, each step just runs until it is finished. Not
        enforcing the time limit can be useful if different tasks/steps will take different amounts of time
        to complete.

    """
    def __init__(
            self, name, input_zip, pattern, scratch, local_scratch_prefix='/tmp/dps/hyper/', ppn=12,
            wall_time="1hour", cleanup_time="15mins", time_slack=0, add_date=True, dry_run=0,
            parallel_exe="$HOME/.local/bin/parallel", kind="parallel", host_pool=None,
            min_hosts=1, max_hosts=1, env_vars=None, redirect=False, n_retries=0, gpu_set="",
            enforce_time_limit=True):

        if kind == "pbs":
            local_scratch_prefix = "\\$RAMDISK"
        elif kind == "slurm":
            local_scratch_prefix = "/tmp/dps/hyper"

        assert kind in "parallel pbs slurm".split()
        hpc = kind in "pbs slurm".split()
        clean_pattern = pattern.replace(' ', '_')

        # Create directory to run the job from - should be on scratch.
        scratch = os.path.abspath(os.path.expandvars(str(scratch)))
        job_directory = make_filename(
            '{}_{}'.format(name, clean_pattern),
            directory=scratch,
            add_date=add_date)
        os.makedirs(os.path.realpath(job_directory + "/results"))

        if hpc:
            input_zip = shutil.copy(input_zip, job_directory)

        input_zip = Path(input_zip)
        input_zip_abs = input_zip.resolve()
        input_zip_base = input_zip.name
        input_zip_stem = input_zip.stem
        archive_root = zip_root(input_zip)

        # storage local to each node, from the perspective of that node
        local_scratch = str(Path(local_scratch_prefix) / Path(job_directory).name)

        wall_time = parse_timedelta(wall_time)
        cleanup_time = parse_timedelta(cleanup_time)

        wall_time_seconds = int(wall_time.total_seconds())
        cleanup_time_seconds = int(cleanup_time.total_seconds())

        assert wall_time_seconds > 0
        assert cleanup_time_seconds > 0

        redirect = "--redirect" if redirect else ""

        env = os.environ.copy()

        env_vars = env_vars or {}
        env_vars["OMP_NUM_THREADS"] = 1

        env.update({e: str(v) for e, v in env_vars.items()})
        env_vars = ' '.join('--env ' + k for k in env_vars)

        ro_job = ReadOnlyJob(input_zip)
        indices_to_run = sorted([op.idx for op in ro_job.ready_incomplete_ops(sort=False)])
        n_jobs_to_run = len(indices_to_run)
        if n_jobs_to_run == 0:
            print("All jobs are finished! Exiting.")
            return

        if hpc:
            host_pool = []
            n_nodes = max_hosts
            n_procs = n_nodes * ppn
            n_steps = int(np.ceil(n_jobs_to_run / n_procs))
        else:
            self.__dict__.update(locals())
            host_pool = host_pool or DEFAULT_HOST_POOL

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

        execution_time = int((wall_time - cleanup_time).total_seconds())
        abs_seconds_per_step = int(np.floor(execution_time / n_steps))
        seconds_per_step = abs_seconds_per_step - time_slack

        assert execution_time > 0
        assert abs_seconds_per_step > 0
        assert seconds_per_step > 0

        staged_hosts = set()

        self.__dict__.update(locals())

        # Create convenience `latest` symlinks
        make_symlink(job_directory, os.path.join(scratch, 'latest'))

        print("We have {wall_time_seconds} seconds to complete {n_jobs_to_run} "
              "sub-jobs (grouped into {n_steps} steps) using {n_procs} processors.".format(**self.__dict__))
        print("{execution_time} seconds have been reserved for job execution, "
              "and {cleanup_time_seconds} seconds have been reserved for cleanup.".format(**self.__dict__))
        print("Each step has been allotted {abs_seconds_per_step} seconds, "
              "{seconds_per_step} seconds of which is pure computation time.\n".format(**self.__dict__))

    def recruit_hosts(self, hpc, min_hosts, max_hosts, host_pool, ppn, max_procs):
        hosts = []
        for host in host_pool:
            n_hosts_recruited = len(hosts)
            if n_hosts_recruited >= max_hosts:
                break

            if n_hosts_recruited * ppn >= max_procs:
                break

            print("\n" + ("~" * 40))
            print("Recruiting host {}...".format(host))

            if host is not ':':
                print("Testing connection...")
                failed = self.ssh_execute("echo Connected to \$HOSTNAME", host, robust=True)
                if failed:
                    print("Could not connect.")
                    continue

            print("Preparing host...")
            try:
                if host is ':':
                    command = "stat {local_scratch}"
                    create_local_scratch = self.execute_command(command, robust=True)

                    if create_local_scratch:
                        print("Creating local scratch directory...")
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
                        self.ssh_execute(command, host, robust=False)

                    command = "cd {local_scratch} && stat {archive_root}"
                    missing_archive = self.ssh_execute(command, host, robust=True)

                    if missing_archive:
                        command = "cd {local_scratch} && stat {input_zip_base}"
                        missing_zip = self.ssh_execute(command, host, robust=True)

                        if missing_zip:
                            print("Copying zip to local scratch...")
                            command = (
                                "scp -q -oPasswordAuthentication=no -oStrictHostKeyChecking=no "
                                "-oConnectTimeout=5 -oServerAliveInterval=2 "
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

        if len(hosts) < min_hosts:
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
            progress=False, verbose=False, quiet=True):
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

            stdout = subprocess.DEVNULL if quiet else None
            stderr = subprocess.DEVNULL if quiet else None

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
                    return p.returncode
                else:
                    raise subprocess.CalledProcessError(p.returncode, command)

            return p.returncode
        except BaseException as e:
            if p is not None:
                p.terminate()
                p.kill()
            if progress is not None:
                progress.finish()
            raise_with_traceback(e)

    def ssh_execute(self, command, host, **kwargs):
        return self.execute_command(
            "ssh -oPasswordAuthentication=no -oStrictHostKeyChecking=no "
            "-oConnectTimeout=5 -oServerAliveInterval=2 "
            "-T {host} \"{command}\"".format(host=host, command=command), **kwargs)

    def fetch(self):
        for i, host in enumerate(self.hosts):
            if host is ':':
                command = "cd {local_scratch} && zip -rq results {archive_root}"
                self.execute_command(command, robust=True)

                command = "cp {local_scratch}/results.zip ./results/{i}.zip".format(i=i, **self.__dict__)
                self.execute_command(command, frmt=False, robust=True)
            else:
                command = "cd {local_scratch} && zip -rq results {archive_root}"
                self.ssh_execute(command, host, robust=True)

                command = (
                    "scp -q -oPasswordAuthentication=no -oStrictHostKeyChecking=no "
                    "-oConnectTimeout=5 -oServerAliveInterval=2 "
                    "{host}:{local_scratch}/results.zip ./results/{i}.zip".format(
                        host=host, i=i, **self.__dict__)
                )
                self.execute_command(command, frmt=False, robust=True)

    def step(self, i, indices_for_step):
        if not indices_for_step:
            print("No jobs left to run on step {}.".format(i))
            return

        indices_for_step = ' '.join(str(i) for i in indices_for_step)

        if self.enforce_time_limit:
            dps_hyper_timeout = " --max-time {} ".format(self.seconds_per_step)
            parallel_timeout = " --timeout {} ".format(self.abs_seconds_per_step)
        else:
            dps_hyper_timeout = ""
            parallel_timeout = ""

        parallel_command = (
            "cd {local_scratch} && "
            "dps-hyper run {archive_root} {pattern} {{}} " + dps_hyper_timeout +
            "--log-root {local_scratch} --log-name experiments "
            "--idx-in-node={{%}} --gpu-set={gpu_set} --ppn={ppn} {redirect}"
        )

        command = (
            '{parallel_exe} ' + parallel_timeout + ' --no-notice -j{ppn} \\\n'
            '    --joblog {job_directory}/job_log.txt {node_file} \\\n'
            '    --env PATH --env LD_LIBRARY_PATH {env_vars} -v \\\n'
            '    "' + parallel_command + '" \\\n'
            '    ::: {indices_for_step}'
        )

        command = command.format(
            indices_for_step=indices_for_step, **self.__dict__)

        self.execute_command(
            command, frmt=False, robust=True,
            max_seconds=self.abs_seconds_per_step, progress=not self.hpc,
            quiet=False, verbose=True)

    def checkpoint(self, i):
        print("Fetching results of step {} at: ".format(i))
        print(datetime.datetime.now())

        self.fetch()

        print("Unzipping results from nodes...")
        results_files = glob.glob("results/*.zip")

        for f in results_files:
            self.execute_command("unzip -nuq {} -d results".format(f), robust=True)

        for f in results_files:
            self.execute_command("rm -rf {}".format(f), robust=True)

        with cd('results'):
            self.execute_command("zip -rq ../results.zip {archive_root}", robust=True)

        self.execute_command("dps-hyper summary results.zip", robust=True, quiet=False)
        self.execute_command("dps-hyper view results.zip", robust=True, quiet=False)

    def run(self):
        if self.dry_run:
            print("Dry run, so not running.")
            return

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
                    elif self.kind == "slurm":
                        p = subprocess.run('scontrol show hostnames $SLURM_JOB_NODELIST', stdout=subprocess.PIPE, shell=True)
                        self.host_pool = list(set([host.strip() for host in p.stdout.decode().split('\n') if host]))
                        print(self.host_pool)
                    else:
                        raise Exception("NotImplemented")

                self.hosts, self.n_procs = self.recruit_hosts(
                    self.hpc, self.min_hosts, self.max_hosts, self.host_pool,
                    self.ppn, max_procs=len(subjobs_remaining))

                indices_for_step = subjobs_remaining[:self.n_procs]
                self.step(i, indices_for_step)
                self.checkpoint(i)

                job = ReadOnlyJob('results.zip')

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
