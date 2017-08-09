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
    walltime: str
        String specifying the maximum wall-time allotted to running the selected ops.
    cleanup_time: str
        String specifying the amount of time required to clean-up. Job execution will be
        halted when there is this much time left in the overall walltime, at which point
        results computed so far will be collected.
    add_date: bool
        Whether to add current date/time to the name of the directory where results are stored.
    time_slack: int
        Number of extra seconds to allow per job.
    dry_run: bool
        If True, control script will be generated but not executed/submitted.
    parallel_exe: str
        Path to the `gnu-parallel` executable to use.
    hosts: list of str
        List of names of hosts to use to execute the job.
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
            walltime="1:00:00", cleanup_time="00:15:00", time_slack=0,
            add_date=True, dry_run=0, parallel_exe="$HOME/.local/bin/parallel",
            hosts=None, env_vars=None, redirect=False, n_retries=10):
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
        os.makedirs(job_directory + "/results")

        # storage local to each node, from the perspective of that node
        local_scratch = str(Path(local_scratch_prefix) / Path(job_directory).name)

        cleanup_time = parse_timedelta(cleanup_time)
        try:
            walltime = parse_timedelta(walltime)
        except:
            deadline = parse_date(walltime)
            walltime = deadline - datetime.datetime.now()
            if int(walltime.total_seconds()) < 0:
                raise Exception("Deadline {} is in the past!".format(deadline))

        if cleanup_time > walltime:
            raise Exception("Cleanup time {} is larger than walltime {}!".format(cleanup_time, walltime))

        walltime_seconds = int(walltime.total_seconds())
        cleanup_time_seconds = int(cleanup_time.total_seconds())

        if not hosts:
            hosts = [":"]  # Only localhost
        with (Path(job_directory) / 'nodefile.txt').open('w') as f:
            f.write('\n'.join(hosts))
        node_file = " --sshloginfile nodefile.txt "
        n_nodes = len(hosts)

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

        n_procs = min(ppn * n_nodes, n_jobs_to_run)
        n_steps = int(np.ceil(n_jobs_to_run / n_procs))

        execution_time = int((walltime - cleanup_time).total_seconds())
        abs_seconds_per_step = int(np.floor(execution_time / n_steps))
        seconds_per_step = abs_seconds_per_step - time_slack

        assert execution_time > 0
        assert abs_seconds_per_step > 0
        assert seconds_per_step > 0

        self.__dict__.update(locals())

        # Create convenience `latest` symlinks
        make_symlink(job_directory, os.path.join(scratch, 'latest'))

    def on_all_execute(self, parallel_command, arguments=False):
        """ Execute a command once on each host. """
        command = '{parallel_exe} --no-notice {node_file} --nonall {env_vars} -k "' + parallel_command + '"'
        self.execute_command(command)

    def stage(self):
        for host in self.hosts:
            if host == ":":
                command = "cp {input_zip_abs} {local_scratch}".format(**self.__dict__)
            else:
                command = "scp -q {input_zip_abs} {host}:{local_scratch}".format(host=host, **self.__dict__)
            self.execute_command(command, frmt=False)

    def fetch(self):
        for i, host in enumerate(self.hosts):
            if host == ":":
                command = "cp {local_scratch}/results.zip ./results/{i}.zip".format(i=i, **self.__dict__)
            else:
                command = "scp -q {host}:{local_scratch}/results.zip ./results/{i}.zip".format(
                    host=host, i=i, **self.__dict__)
            self.execute_command(command, frmt=False)

    def step(self, i):
        print("Beginning step {} at: ".format(i) + "=" * 90)
        print(datetime.datetime.now())

        indices_for_step = self.indices_to_run[i*self.n_procs:(i+1)*self.n_procs]
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

        self.execute_command(command, frmt=False)

    def checkpoint(self, i):
        print("Fetching results of step {} at: ".format(i))
        print(datetime.datetime.now())

        command = "cd {local_scratch} && echo Zipping results on node \\$(hostname). && zip -rq results {archive_root}"
        self.on_all_execute(command, arguments=False)

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

    def execute_command(self, command, frmt=True, shell=True):
        """ Uses `subprocess` to execute `command`. """
        if frmt:
            command = command.format(**self.__dict__)

        print("\nExecuting command: ")
        print(command)

        if not shell:
            command = command.split()

        start = time.time()
        try:
            subprocess.run(command, check=True, universal_newlines=True, shell=shell)
            print("Command took {} seconds.\n".format(time.time() - start))
        except subprocess.CalledProcessError as e:
            if isinstance(command, list):
                command = ' '.join(command)
            print("CalledProcessError has been raised while executing command: {}.".format(command))
            raise_with_traceback(e)

    def run(self):
        if self.dry_run:
            print("Dry run, so not running.")
            return

        with cd(self.job_directory):
            print("Starting job at {}".format(datetime.datetime.now()))
            print("Creating local scratch directories...")

            self.on_all_execute("mkdir -p {local_scratch}", arguments=False)

            print("Printing local scratch directories...")

            command = (
                "cd {local_scratch} && "
                "echo Local scratch on host \\$(hostname) is {local_scratch}, working directory is \\$(pwd).")
            self.on_all_execute(command, arguments=False)

            print("Staging input archive...")
            self.stage()

            print("Unzipping...")
            command = "cd {local_scratch} && unzip -ouq {input_zip_base} && echo \\$(hostname)"
            self.on_all_execute(command, arguments=False)

            print("We have {walltime_seconds} seconds to complete {n_jobs_to_run} "
                  "sub-jobs (grouped into {n_steps} steps) using {n_procs} processors.".format(**self.__dict__))
            print("{execution_time} seconds have been reserved for job execution, "
                  "and {cleanup_time_seconds} seconds have been reserved for cleanup.".format(**self.__dict__))
            print("Each step has been allotted {abs_seconds_per_step} seconds, "
                  "{seconds_per_step} seconds of which is pure computation time.\n".format(**self.__dict__))

            for i in range(self.n_steps):
                self.step(i)
                self.checkpoint(i)

            self.execute_command("cp -f {input_zip_abs} {input_zip_abs}.bk")
            self.execute_command("cp -f results.zip {input_zip_abs}")


def submit_job_pbs(
        input_zip, pattern, scratch, local_scratch_prefix='/tmp/dps/hyper/', n_nodes=1, ppn=12,
        walltime="1:00:00", cleanup_time="00:15:00", time_slack=0,
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
    walltime: str
        String specifying the maximum wall-time allotted to running the selected ops.
    cleanup_time: str
        String specifying the amount of time required to clean-up. Job execution will be
        halted when there is this much time left in the overall walltime, at which point
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
    os.makedirs(job_directory)

    # storage local to each node, from the perspective of that node
    local_scratch = '\\$RAMDISK'

    cleanup_time = parse_timedelta(cleanup_time)
    walltime = parse_timedelta(walltime)
    if cleanup_time > walltime:
        raise Exception("Cleanup time {} is larger than walltime {}!".format(cleanup_time, walltime))

    walltime_seconds = int(walltime.total_seconds())
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

    execution_time = int((walltime - cleanup_time).total_seconds())
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
#PBS -l nodes={n_nodes}:ppn={ppn},walltime={walltime}
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

echo We have {walltime_seconds} seconds to complete {n_jobs_to_run} sub-jobs using {n_procs} processors.
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
