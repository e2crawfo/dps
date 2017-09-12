from __future__ import print_function
import os
import datetime
import subprocess
from future.utils import raise_with_traceback
from datetime import timedelta
from pathlib import Path
import numpy as np
import glob
from contextlib import contextmanager
import time
from collections import defaultdict

from spectral_dagger.utils.misc import make_symlink

from dps.parallel.base import ReadOnlyJob, zip_root
from dps.utils import parse_date


def make_directory_name(experiments_dir, network_name, add_date=True):
    if add_date:
        working_dir = os.path.join(experiments_dir, network_name + "_")
        dts = str(datetime.datetime.now()).split('.')[0]
        for c in [":", " ", "-"]:
            dts = dts.replace(c, "_")
        working_dir += dts
    else:
        working_dir = os.path.join(experiments_dir, network_name)

    return working_dir


def parse_timedelta(s):
    """ ``s`` should be of the form HH:MM:SS """
    args = [int(i) for i in s.split(":")]
    return timedelta(hours=args[0], minutes=args[1], seconds=args[2])


@contextmanager
def cd(path):
    """ A context manager that changes into given directory on __enter__,
        change back to original_file directory on exit. Exception safe.

    """
    old_dir = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(old_dir)


def submit_job(*args, **kwargs):
    session = ParallelSession(*args, **kwargs)
    session.run()


HOST_POOL = ['ecrawf6@cs-{}.cs.mcgill.ca'.format(i) for i in range(1, 32)]


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
        Path to scratch directory that is local to each remote process.
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
        Number of retries.

    """
    def __init__(
            self, name, input_zip, pattern, scratch, local_scratch_prefix='/tmp/dps/hyper/', ppn=12,
            wall_time="1:00:00", cleanup_time="00:15:00", time_slack=0,
            add_date=True, dry_run=0, parallel_exe="$HOME/.local/bin/parallel",
            host_pool=None, min_hosts=1, max_hosts=1, env_vars=None, redirect=False, n_retries=10):
        input_zip = Path(input_zip)
        input_zip_abs = input_zip.resolve()
        input_zip_base = input_zip.name
        input_zip_stem = input_zip.stem
        archive_root = zip_root(input_zip)
        clean_pattern = pattern.replace(' ', '_')

        # Create directory to run the job from - should be on scratch.
        scratch = os.path.abspath(scratch)
        job_directory = make_directory_name(
            scratch,
            '{}_{}'.format(name, clean_pattern),
            add_date=add_date)
        os.makedirs(os.path.realpath(job_directory + "/results"))

        # storage local to each node, from the perspective of that node
        local_scratch = str(Path(local_scratch_prefix) / Path(job_directory).name)

        cleanup_time = parse_timedelta(cleanup_time)
        try:
            wall_time = parse_timedelta(wall_time)
        except:
            deadline = parse_date(wall_time)
            wall_time = deadline - datetime.datetime.now()
            if int(wall_time.total_seconds()) < 0:
                raise Exception("Deadline {} is in the past!".format(deadline))

        if cleanup_time > wall_time:
            raise Exception("Cleanup time {} is larger than wall_time {}!".format(cleanup_time, wall_time))

        wall_time_seconds = int(wall_time.total_seconds())
        cleanup_time_seconds = int(cleanup_time.total_seconds())

        redirect = "--redirect" if redirect else ""

        env = os.environ.copy()

        env_vars = env_vars or {}
        env_vars["OMP_NUM_THREADS"] = 1

        env.update({e: str(v) for e, v in env_vars.items()})
        env_vars = ' '.join('--env ' + k for k in env_vars)

        ro_job = ReadOnlyJob(input_zip)
        completion = ro_job.completion(pattern)
        indices_to_run = [i for i, op in completion['ready_incomplete_ops']]
        n_jobs_to_run = len(indices_to_run)
        if n_jobs_to_run == 0:
            print("All jobs are finished! Exiting.")
            return

        self.min_hosts = min_hosts
        self.max_hosts = max_hosts
        hosts = []
        host_pool = host_pool or HOST_POOL
        bad_hosts = []

        self.__dict__.update(locals())

        with cd(self.job_directory):
            self.filter_hosts(check_current=False, add_new=True)

        node_file = " --sshloginfile nodefile.txt "

        n_nodes = len(hosts)
        n_procs = ppn * n_nodes

        if n_jobs_to_run < n_procs:
            n_steps = 1
            n_nodes = int(np.ceil(n_jobs_to_run / ppn))
            n_procs = n_nodes * ppn
            hosts = hosts[:n_nodes]
        else:
            n_steps = int(np.ceil(n_jobs_to_run / n_procs))

        execution_time = int((wall_time - cleanup_time).total_seconds())
        abs_seconds_per_step = int(np.floor(execution_time / n_steps))
        seconds_per_step = abs_seconds_per_step - time_slack

        assert execution_time > 0
        assert abs_seconds_per_step > 0
        assert seconds_per_step > 0

        n_attempts = defaultdict(int)
        staged_hosts = set()

        self.__dict__.update(locals())

        # Create convenience `latest` symlinks
        make_symlink(job_directory, os.path.join(scratch, 'latest'))

    def filter_hosts(self, check_current=True, add_new=False):
        if check_current:
            print("Check current hosts...")
            for host in self.hosts + []:
                if host is not ':':
                    print("Testing connection to host {}...".format(host))
                    failed = self.ssh_execute("echo Connected to \$HOSTNAME", host, robust=True)
                    if failed:
                        print("Could not connect to houst {}, removing it.".format(host))
                        self.hosts.remove(host)

        if add_new:
            print("Adding new hosts...")
            for host in self.host_pool:
                if len(self.hosts) >= self.max_hosts:
                    break

                if host in self.hosts or host in self.bad_hosts:
                    continue

                print("\n" + ("~" * 40))
                print("\nPreparing host {}...".format(host))

                try:
                    if host is ':':
                        print("Creating local scratch directory...")
                        command = "mkdir -p {local_scratch}"
                        self.execute_command(command, robust=False)

                        print("Printing local scratch directory...")
                        command = (
                            "cd {local_scratch} && "
                            "echo Local scratch on host \\$(hostname) is {local_scratch}, "
                            "working directory is \\$(pwd)."
                        )
                        self.execute_command(command, robust=False)

                        print("Unzipping...")
                        command = "cd {local_scratch} && unzip -ouq {input_zip_base} && echo \\$(hostname)"
                        self.execute_command(command, robust=False)

                        command = "cp {input_zip_abs} {local_scratch}".format(**self.__dict__)
                        self.execute_command(command, frmt=False, robust=False)

                        self.hosts.append(host)
                    else:
                        self.ssh_execute("echo Connected to \$HOSTNAME", host, robust=False)
                        print("Connection to host {} succeeded, now trying to stage it...".format(host))

                        print("Creating local scratch directory...")
                        self.ssh_execute("mkdir -p {local_scratch}", host, robust=False)

                        print("Printing local scratch directory...")
                        command = (
                            "cd {local_scratch} && "
                            "echo Local scratch on host \\$(hostname) is {local_scratch}, "
                            "working directory is \\$(pwd)."
                        )
                        self.ssh_execute(command, host, robust=False)

                        command = "scp -q {input_zip_abs} {host}:{local_scratch}".format(host=host, **self.__dict__)
                        self.execute_command(command, frmt=False, robust=False)

                        print("Unzipping...")
                        command = "cd {local_scratch} && unzip -ouq {input_zip_base} && echo \\$(hostname)"
                        self.ssh_execute(command, host, robust=False)

                        print("Host {} successfully prepared.".format(host))
                        self.hosts.append(host)

                except subprocess.CalledProcessError:
                    print("Preparation of host {} failed, not adding it.".format(host))
                    self.bad_hosts.append(host)

            if len(self.hosts) < self.min_hosts:
                raise Exception(
                    "Found only {} usable hosts, but minimum "
                    "required hosts is {}.".format(len(self.hosts), self.min_hosts))

            if len(self.hosts) < self.max_hosts:
                print("{} hosts were requested, "
                      "but only {} usable hosts could be found.".format(self.max_hosts, len(self.hosts)))

        with open('nodefile.txt', 'w') as f:
            f.write('\n'.join(self.hosts))

    def execute_command(self, command, frmt=True, shell=True, robust=False):
        """ Uses `subprocess` to execute `command`. """
        if frmt:
            command = command.format(**self.__dict__)

        print("\nExecuting command: ")
        print(command)

        if not shell:
            command = command.split()

        start = time.time()
        try:
            process = subprocess.run(command, check=True, universal_newlines=True, shell=shell)
            print("Command took {} seconds.\n".format(time.time() - start))
        except subprocess.CalledProcessError as e:
            if isinstance(command, list):
                command = ' '.join(command)
            print("CalledProcessError has been raised while executing command: {}.".format(command))

            if robust:
                return e.returncode
            else:
                raise_with_traceback(e)

        return process.returncode

    def ssh_execute(self, command, host, **kwargs):
        return self.execute_command(
            "ssh -oPasswordAuthentication=no -oStrictHostKeyChecking=no "
            "-oConnectTimeout=5 -T {host} \"{command}\"".format(host=host, command=command), **kwargs)

    def fetch(self):
        for i, host in enumerate(self.hosts):
            try:
                if host is ':':
                    command = (
                        "cd {local_scratch} && echo Zipping results on node \\$(hostname). "
                        "&& zip -rq results {archive_root}"
                    )
                    self.execute_command(command, robust=False)
                    command = "cp {local_scratch}/results.zip ./results/{i}.zip".format(i=i, **self.__dict__)
                    self.execute_command(command, frmt=False, robust=True)
                else:
                    command = (
                        "cd {local_scratch} && echo Zipping results on node \\$(hostname). "
                        "&& zip -rq results {archive_root}"
                    )
                    self.ssh_execute(command, host, robust=False)

                    command = "scp -q {host}:{local_scratch}/results.zip ./results/{i}.zip".format(
                        host=host, i=i, **self.__dict__)
                    self.execute_command(command, frmt=False, robust=True)

            except subprocess.CalledProcessError:
                print("Preparation of host {} failed, not adding it.".format(host))
                self.bad_hosts.append(host)

    def step(self, i, indices_for_step):
        print("Beginning step {} at: ".format(i) + "=" * 90)
        print(datetime.datetime.now())

        if not indices_for_step:
            print("No jobs left to run on step {}.".format(i))
            return

        indices_for_step = ' '.join(str(i) for i in indices_for_step)

        parallel_command = (
            "cd {local_scratch} && "
            "dps-hyper run {archive_root} {pattern} {{}} --max-time {seconds_per_step} "
            "--log-root {local_scratch} --log-name experiments {redirect}")

        command = (
            '{parallel_exe} --timeout {abs_seconds_per_step} --no-notice -j {ppn} --retries {n_retries} \\\n'
            '    --joblog {job_directory}/job_log.txt {node_file} \\\n'
            '    --env PATH --env LD_LIBRARY_PATH {env_vars} -v \\\n'
            '    "' + parallel_command + '" \\\n'
            '    ::: {indices_for_step}')
        command = command.format(
            indices_for_step=indices_for_step, **self.__dict__)

        n_failed = self.execute_command(command, frmt=False, robust=True)
        if n_failed:
            self.filter_hosts(check_current=True, add_new=False)

    def checkpoint(self, i):
        print("Fetching results of step {} at: ".format(i))
        print(datetime.datetime.now())

        self.fetch()

        print("Unzipping results from nodes...")
        results_files = glob.glob("results/*.zip")

        if not results_files:
            raise Exception("Did not find any results files from nodes on step {}.".format(i))

        for f in results_files:
            self.execute_command("unzip -nuq {} -d results".format(f))

        with cd('results'):
            self.execute_command("zip -rq ../results.zip {archive_root}")

        self.execute_command("dps-hyper summary results.zip")
        self.execute_command("dps-hyper view results.zip")

    def run(self):
        if self.dry_run:
            print("Dry run, so not running.")
            return

        with cd(self.job_directory):
            print("Starting job at {}".format(datetime.datetime.now()))

            print("We have {wall_time_seconds} seconds to complete {n_jobs_to_run} "
                  "sub-jobs (grouped into {n_steps} steps) using {n_procs} processors.".format(**self.__dict__))
            print("{execution_time} seconds have been reserved for job execution, "
                  "and {cleanup_time_seconds} seconds have been reserved for cleanup.".format(**self.__dict__))
            print("Each step has been allotted {abs_seconds_per_step} seconds, "
                  "{seconds_per_step} seconds of which is pure computation time.\n".format(**self.__dict__))

            job = ReadOnlyJob(self.input_zip)
            completion = job.completion(self.pattern)
            jobs_remaining = [op.idx for op in completion['ready_incomplete_ops']]

            i = 0
            while jobs_remaining:
                self.filter_hosts(check_current=True, add_new=True)

                indices_for_step = jobs_remaining[:self.n_procs]
                self.step(i, indices_for_step)
                self.checkpoint(i)

                for j in indices_for_step:
                    self.n_attempts[j] += 1

                job = ReadOnlyJob('results.zip')
                completion = job.completion(self.pattern)
                jobs_remaining = [
                    op.idx for op in completion['ready_incomplete_ops']
                    if self.n_attempts[j] <= self.n_retries]

                i += 1

            self.execute_command("cp -f {input_zip_abs} {input_zip_abs}.bk")
            self.execute_command("cp -f results.zip {input_zip_abs}")


def submit_job_pbs(
        input_zip, pattern, scratch, local_scratch_prefix='/tmp/dps/hyper/', n_nodes=1, ppn=12,
        wall_time="1:00:00", cleanup_time="00:15:00", time_slack=0,
        add_date=True, show_script=0, dry_run=0, parallel_exe="$HOME/.local/bin/parallel",
        queue=None, hosts=None, env_vars=None):
    """ Submit a Job to be executed in parallel.

    A directory for this Job execution is created in `scratch`, and results are saved there.

    Parameters
    ----------
    input_zip: str
        Path to a zip archive storing the Job.
    pattern: str
        Pattern to use to select which ops to run within the Job.
    scratch: str
        Path to location where the results of running the selected ops will be
        written. Must be writeable by the master process.
    local_scratch_prefix: str
        Path to scratch directory that is local to each remote process.
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
    show_script: bool
        Whether to print to console the control script that is generated.
    dry_run: bool
        If True, control script will be generated but not executed/submitted.
    parallel_exe: str
        Path to the `gnu-parallel` executable to use.
    queue: str
        The queue to submit job to.
    env_vars: dict (str -> str)
        Dictionary mapping environment variable names to values. These will be accessible
        by the submit script, and will also be sent to the worker nodes.

    """
    input_zip = Path(input_zip)
    input_zip_abs = input_zip.resolve()
    input_zip_base = input_zip.name
    archive_root = zip_root(input_zip)
    name = Path(input_zip).stem
    queue = "#PBS -q {}".format(queue) if queue is not None else ""
    clean_pattern = pattern.replace(' ', '_')
    stderr = "| tee -a /dev/stderr"

    # Create directory to run the job from - should be on scratch.
    scratch = os.path.abspath(scratch)
    job_directory = make_directory_name(
        scratch,
        '{}_{}'.format(name, clean_pattern),
        add_date=add_date)
    os.makedirs(os.path.realpath(job_directory))

    # storage local to each node, from the perspective of that node
    local_scratch = '\\$RAMDISK'

    cleanup_time = parse_timedelta(cleanup_time)
    wall_time = parse_timedelta(wall_time)
    if cleanup_time > wall_time:
        raise Exception("Cleanup time {} is larger than wall_time {}!".format(cleanup_time, wall_time))

    wall_time_seconds = int(wall_time.total_seconds())
    cleanup_time_seconds = int(cleanup_time.total_seconds())

    node_file = " --sshloginfile $PBS_NODEFILE "

    idx_file = 'job_indices.txt'

    kwargs = locals().copy()

    env = os.environ.copy()
    if env_vars is not None:
        env.update({e: str(v) for e, v in env_vars.items()})
        kwargs['env_vars'] = ' '.join('--env ' + k for k in env_vars)
    else:
        kwargs['env_vars'] = ''

    ro_job = ReadOnlyJob(input_zip)
    completion = ro_job.completion(pattern)
    n_jobs_to_run = completion['n_ready_incomplete']
    if n_jobs_to_run == 0:
        print("All jobs are finished! Exiting.")
        return
    kwargs['n_jobs_to_run'] = n_jobs_to_run

    execution_time = int((wall_time - cleanup_time).total_seconds())
    n_procs = ppn * n_nodes
    total_compute_time = n_procs * execution_time
    abs_seconds_per_job = int(np.floor(total_compute_time / n_jobs_to_run))
    seconds_per_job = abs_seconds_per_job - time_slack

    assert execution_time > 0
    assert total_compute_time > 0
    assert abs_seconds_per_job > 0
    assert seconds_per_job > 0

    kwargs['abs_seconds_per_job'] = abs_seconds_per_job
    kwargs['seconds_per_job'] = seconds_per_job
    kwargs['execution_time'] = execution_time
    kwargs['n_procs'] = n_procs

    stage_data_code = '''
{parallel_exe} --no-notice {node_file} --nonall {env_vars} \\
"cp {input_zip_abs} {local_scratch}
'''

    fetch_results_code = '''
{parallel_exe} --no-notice {node_file} --nonall {env_vars} \\
"cp {local_scratch}/results.zip {job_directory}/results/\\$(hostname).zip
'''

    code = '''#!/bin/bash
# MOAB/Torque submission script for multiple, dynamically-run serial jobs
#
#PBS -V
#PBS -l nodes={n_nodes}:ppn={ppn},wall_time={wall_time}
#PBS -N {name}_{clean_pattern}
#PBS -M eric.crawford@mail.mcgill.ca
#PBS -m abe
#PBS -A jim-594-aa
#PBS -e stderr.txt
#PBS -o stdout.txt
{queue}

# Turn off implicit threading in Python
export OMP_NUM_THREADS=1

cd {job_directory}
mkdir results

echo Starting job at {stderr}
date {stderr}

echo Printing local scratch directories... {stderr}

{parallel_exe} --no-notice {node_file} --nonall {env_vars} -k \\
    "cd {local_scratch} && echo Local scratch on host \\$(hostname) is {local_scratch}, working directory is \\$(pwd)."

echo Staging input archive... {stderr}

''' + stage_data_code + '''

echo Unzipping... {stderr}

{parallel_exe} --no-notice {node_file} --nonall {env_vars} -k \\
    "cd {local_scratch} && unzip -ouq {input_zip_base} && echo \\$(hostname) && ls"

echo We have {wall_time_seconds} seconds to complete {n_jobs_to_run} sub-jobs using {n_procs} processors.
echo {execution_time} seconds have been reserved for job execution, and {cleanup_time_seconds} seconds have been reserved for cleanup.
echo Each sub-job has been allotted {abs_seconds_per_job} seconds, {seconds_per_job} seconds of which is pure computation time.

echo Launching jobs at {stderr}
date {stderr}

start=$(date +%s)

# Requires a newish version of parallel, has to accept --timeout
{parallel_exe} --timeout {abs_seconds_per_job} --no-notice -j {ppn} --retries 10 \\
    --joblog {job_directory}/job_log.txt {node_file} \\
    --env OMP_NUM_THREADS --env PATH --env LD_LIBRARY_PATH {env_vars} \\
    "cd {local_scratch} && dps-hyper run {archive_root} {pattern} {{}} --max-time {seconds_per_job}" < {idx_file}

end=$(date +%s)

runtime=$((end-start))

echo Executing jobs took $runtime seconds.

echo Fetching results at {stderr}
date {stderr}

{parallel_exe} --no-notice {node_file} --nonall {env_vars} -k \\
    "cd {local_scratch} && echo Zipping results on node \\$(hostname). && zip -rq results {archive_root} && ls"

''' + fetch_results_code + '''

cd results

echo Unzipping results from nodes... {stderr}

if test -n "$(find . -maxdepth 1 -name '*.zip' -print -quit)"; then
    echo Results files exist: {stderr}
    ls
else
    echo Did not find any results files from nodes. {stderr}
    echo Contents of results directory is: {stderr}
    ls
    exit 1
fi

for f in *zip
do
    echo Storing contents of $f {stderr}
    unzip -nuq $f
done

echo Zipping final results... {stderr}
zip -rq {name} {archive_root}
mv {name}.zip ..
cd ..

dps-hyper summary {name}.zip
dps-hyper view {name}.zip
mv {input_zip_abs} {input_zip_abs}.bk
cp -f {name}.zip {input_zip_abs}

'''
    code = code.format(**kwargs)
    if show_script:
        print("\n")
        print("-" * 20 + " BEGIN SCRIPT " + "-" * 20)
        print(code)
        print("-" * 20 + " END SCRIPT " + "-" * 20)
        print("\n")

    # Create convenience `latest` symlinks
    make_symlink(job_directory, os.path.join(scratch, 'latest'))
    make_symlink(
        job_directory,
        os.path.join(scratch, 'latest_{}_{}'.format(name, clean_pattern)))

    os.chdir(job_directory)
    with open(idx_file, 'w') as f:
        [f.write('{}\n'.format(u)) for u in range(completion['n_ops'])]

    submit_script = "submit_script.sh"
    with open(submit_script, 'w') as f:
        f.write(code)

    if dry_run:
        print("Dry run, so not submitting.")
    else:
        try:
            command = ['qsub', submit_script]
            print("Submitting.")
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, env=env)
            output = output.decode()
            print(output)

            # Create a file in the directory with the job_id as its name
            job_id = output.split('.')[0]
            open(job_id, 'w').close()
            print("Job ID: {}".format(job_id))

        except subprocess.CalledProcessError as e:
            print("CalledProcessError has been raised while executing command: {}.".format(' '.join(command)))
            print("Output of process: ")
            print(e.output.decode())
            raise_with_traceback(e)


def _submit_job():
    """ Entry point for `dps-submit` command-line utility. """
    from clify import command_line
    command_line(submit_job, collect_kwargs=1, verbose=True)()
