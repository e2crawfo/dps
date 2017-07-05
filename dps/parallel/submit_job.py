from __future__ import print_function
import os
import datetime
import subprocess
from future.utils import raise_with_traceback
from datetime import timedelta
from pathlib import Path
import numpy as np

from spectral_dagger.utils.misc import make_symlink

from dps.parallel.base import ReadOnlyJob, zip_root


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


def submit_job(
        input_zip, pattern, scratch, local_scratch_prefix='/tmp/dps/hyper/', n_nodes=1, ppn=12,
        walltime="1:00:00", cleanup_time="00:15:00", time_slack=200,
        add_date=True, show_script=0, dry_run=0, parallel_exe="$HOME/.local/bin/parallel",
        pbs=True, queue=None, hosts=None, env_vars=None):
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
    n_nodes: int
        Number of computational nodes to employ.
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
    pbs: bool
        Whether to submit using PBS (i.e. whether we're in an HPC environment).
    queue: str
        If `pbs` set to True, the queue to submit job to.
    hosts: list of str
        If `pbs` is False, names of hosts to use to execute the job. If `pbs` is True,
        then this is ignored and host names are retrieved from `$PBS_NODEFILE`.
    env_vars: dict (str -> str)
        Dictionary mapping environment variable names to values. These will be accessible
        by the submit script, and will also be sent to the worker nodes.

    """
    input_zip = Path(input_zip)
    input_zip_abs = input_zip.resolve()
    input_zip_rsync = str(input_zip.parent) + '/./' + input_zip.name
    input_zip_base = input_zip.name
    archive_root = zip_root(input_zip)
    name = Path(input_zip).stem
    queue = "#PBS -q {}".format(queue) if queue is not None else ""
    clean_pattern = pattern.replace(' ', '_')

    if pbs:
        stderr = "{stderr}"
    else:
        stderr = ""

    # Create directory to run the job from - should be on scratch.
    scratch = os.path.abspath(scratch)
    job_directory = make_directory_name(
        scratch,
        '{}_{}'.format(name, clean_pattern),
        add_date=add_date)
    os.makedirs(job_directory)

    # storage local to each node, from the perspective of that node
    local_scratch = '\\$RAMDISK' if pbs else str(Path(local_scratch_prefix) / Path(job_directory).name)

    cleanup_time = parse_timedelta(cleanup_time)
    walltime = parse_timedelta(walltime)
    if cleanup_time > walltime:
        raise Exception("Cleanup time {} is larger than walltime {}!".format(cleanup_time, walltime))

    if not pbs:
        if not hosts:
            hosts = [":"]  # Only localhost
        with (Path(job_directory) / 'nodefile.txt').open('w') as f:
            f.write('\n'.join(hosts))
        node_file = " --sshloginfile nodefile.txt "
    else:
        node_file = " --sshloginfile $PBS_NODEFILE "

    idx_file = 'job_indices.txt'

    kwargs = locals().copy()

    ro_job = ReadOnlyJob(input_zip)
    completion = ro_job.completion(pattern)
    n_jobs_to_run = completion['n_ready_incomplete']
    if n_jobs_to_run == 0:
        print("All jobs are finished! Exiting.")
        return
    execution_time = int((walltime - cleanup_time).total_seconds())
    n_procs = ppn * n_nodes
    total_compute_time = n_procs * execution_time
    abs_seconds_per_job = int(np.floor(total_compute_time / n_jobs_to_run))
    seconds_per_job = abs_seconds_per_job - time_slack

    env = os.environ.copy()
    if env_vars is not None:
        env.update({e: str(v) for e, v in env_vars.items()})
        kwargs['env_vars'] = ' '.join('--env ' + k for k in env_vars)
    else:
        kwargs['env_vars'] = ''

    kwargs['abs_seconds_per_job'] = abs_seconds_per_job
    kwargs['seconds_per_job'] = seconds_per_job
    kwargs['execution_time'] = execution_time
    kwargs['n_procs'] = n_procs

    stage_data_code = '\n'.join(
        ("scp -q {{input_zip_abs}} {}:{{local_scratch}}".format(h)
            if h != ":" else "cp {input_zip_abs} {local_scratch}")
        for h in hosts)

    fetch_results_code = '\n'.join(
        ("scp -q {}:{{local_scratch}}/results.zip ./results/{}.zip".format(h, i)
            if h != ":" else "cp {{local_scratch}}/results.zip ./results/{}.zip".format(i))
        for i, h in enumerate(hosts))

    code = '''#!/bin/bash '''

    if pbs:
        code += '''
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
''' + queue

    code += '''
# Turn off implicit threading in Python
export OMP_NUM_THREADS=1

cd {job_directory}
mkdir results

echo Starting job at {stderr}
date {stderr}

'''
    if not pbs:
        code += '''
echo Creating local scratch directories... {stderr}

{parallel_exe} --no-notice {node_file} --nonall {env_vars} -k \\
    "mkdir -p {local_scratch}"
'''

    code += '''
echo Printing local scratch directories... {stderr}

{parallel_exe} --no-notice {node_file} --nonall {env_vars} -k \\
    "cd {local_scratch} && echo Local scratch on host \\$(hostname) is {local_scratch}, working directory is \\$(pwd)."

echo Staging input archive... {stderr}

''' + stage_data_code + '''

echo Unzipping... {stderr}

{parallel_exe} --no-notice {node_file} --nonall {env_vars} -k \\
    "cd {local_scratch} && unzip -ouq {input_zip_base} && echo \\$(hostname) && ls"

echo Each job is alloted {abs_seconds_per_job} seconds, {seconds_per_job} seconds of which is pure computation time.
echo Since we have {n_procs} processors, this batch should take at most {execution_time} seconds.

echo Running jobs... {stderr}

start=`date +%s`

# Requires a newish version of parallel, has to accept --timeout
{parallel_exe} --timeout {abs_seconds_per_job} -k --no-notice -j {ppn} \\
    --joblog {job_directory}/job_log.txt {node_file} \\
    --env OMP_NUM_THREADS --env PATH --env LD_LIBRARY_PATH {env_vars} \\
    "cd {local_scratch} && dps-hyper run {archive_root} {pattern} {{}} --max-time {seconds_per_job} " < {idx_file}

end=`date +%s`

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
    # rm $f
done

echo Zipping final results... {stderr}
zip -rq {name} {archive_root}
mv {name}.zip ..
cd ..

dps-hyper view {name}.zip
mv {input_zip_abs} {input_zip_abs}.bk
cp -f {name}.zip {input_zip_abs}

'''
    code = code.format(**kwargs)
    if show_script:
        print(code)

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
            if pbs:
                command = ['qsub', submit_script]
                print("Submitting.")
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, env=env)
                output = output.decode()
                print(output)

                # Create a file in the directory with the job_id as its name
                job_id = output.split('.')[0]
                open(job_id, 'w').close()
                print("Job ID: {}".format(job_id))
            else:
                command = ['sh', submit_script]
                output = subprocess.check_output(command, stderr=subprocess.STDOUT, env=env)
                output = output.decode()
                print(output)

        except subprocess.CalledProcessError as e:
            print("CalledProcessError has been raised while executing command: {}.".format(' '.join(command)))
            print("Output of process: ")
            print(e.output.decode())
            raise_with_traceback(e)


def _submit_job():
    """ Entry point for `dps-submit` command-line utility. """
    from clify import command_line
    command_line(submit_job, collect_kwargs=1, verbose=True)()


if __name__ == "__main__":
    _submit_job()
