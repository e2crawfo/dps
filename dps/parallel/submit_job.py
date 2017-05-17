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
        input_zip, pattern, scratch, n_jobs=-1, n_nodes=1, ppn=12, walltime="1:00:00",
        cleanup_time="00:15:00", add_date=True, test=0,
        show_script=0, dry_run=0, queue=None, time_slack=200,
        parallel_exe="$HOME/.local/bin/parallel",
        sdbin='$HOME/.virtualenvs/dps/bin/'):

    input_zip = Path(input_zip)
    input_zip_abs = input_zip.resolve()
    input_zip_base = input_zip.name
    archive_root = zip_root(input_zip)
    name = Path(input_zip).stem
    dps_hyper = os.path.join(sdbin, 'dps-hyper')
    local_scratch = 'LSCRATCH' if test else '$LSCRATCH'
    queue = "#PBS -q {}".format(queue) if queue is not None else ""
    clean_pattern = pattern.replace(' ', '_')

    # Create directory to run the job from - should be on scratch.
    scratch = os.path.abspath(scratch)
    job_directory = make_directory_name(
        scratch,
        '{}_{}'.format(name, clean_pattern),
        add_date=add_date)
    os.makedirs(job_directory)

    cleanup_time = parse_timedelta(cleanup_time)
    walltime = parse_timedelta(walltime)
    if cleanup_time > walltime:
        raise Exception("Cleanup time {} is larger than walltime {}!".format(cleanup_time, walltime))

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

    kwargs['abs_seconds_per_job'] = abs_seconds_per_job
    kwargs['seconds_per_job'] = seconds_per_job
    kwargs['execution_time'] = execution_time
    kwargs['n_procs'] = n_procs

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
''' + queue + '''

# Turn off implicit threading in Python
export OMP_NUM_THREADS=1

cd {job_directory}
echo PWD $PWD | tee -a /dev/stderr
mkdir results
echo "Starting job at - " | tee -a /dev/stderr
date

echo "Printing RAMDISK..." | tee -a /dev/stderr
{parallel_exe} --no-notice {node_file} --nonall \\
    "cd \\$RAMDISK && echo Ramdisk on host \\$HOSTNAME is \\$RAMDISK, working directory is \\$PWD."

echo "Staging input archive..." | tee -a /dev/stderr
{parallel_exe} --no-notice {node_file} --nonall \\
    "cd \\$RAMDISK && cp {input_zip_abs} ."

echo "Listing staging results..." | tee -a /dev/stderr
{parallel_exe} --no-notice {node_file} --nonall \\
    "cd \\$RAMDISK && echo ls on node \\$HOSTNAME && ls"

echo "Unzipping..." | tee -a /dev/stderr
{parallel_exe} --no-notice {node_file} --nonall \\
    "cd \\$RAMDISK && unzip -ouq {input_zip_base} && ls"

echo "Running parallel..." | tee -a /dev/stderr
echo Each job is alloted {abs_seconds_per_job} seconds, {seconds_per_job} seconds of which is pure computation time
echo Since we have {n_procs} processors, this batch should take roughly {execution_time} seconds

start=`date +%s`

# Requires a new-ish version of parallel, has to have --timeout
{parallel_exe} --timeout {abs_seconds_per_job} -k --no-notice -j {ppn} \\
    --joblog {job_directory}/job_log.txt \\
    --env OMP_NUM_THREADS --env PATH --env LD_LIBRARY_PATH \\
    {node_file} \\
    "cd \\$RAMDISK && dps-hyper run {archive_root} {pattern} {{}} --max-time {seconds_per_job} " < {idx_file}

end=`date +%s`

runtime=$((end-start))

echo Executing jobs took $runtime seconds.

echo "Cleaning up at - " | tee -a /dev/stderr
date | tee -a /dev/stderr

{parallel_exe} --no-notice {node_file} --nonall \\
    "echo Retrieving results from node \\$HOSTNAME && cd \\$RAMDISK && zip -rq \\$HOSTNAME {archive_root} && mv \\$HOSTNAME.zip {job_directory}/results"

cd {job_directory}/results
echo In job_directory/results: $PWD | tee -a /dev/stderr
ls

echo "Unzipping results from different nodes..." | tee -a /dev/stderr

if test -n "$(find . -maxdepth 1 -name '*.zip' -print -quit)"; then
    echo Files exist | tee -a /dev/stderr
    ls
else
    echo "No zip files from nodes" | tee -a /dev/stderr
    echo "Contents of results directory is:" | tee -a /dev/stderr
    ls
    exit 1
fi

for f in *zip
do
    echo "Storing results from node "$f | tee -a /dev/stderr
    unzip -nuq $f
    rm $f
done

echo "Zipping final results..." | tee -a /dev/stderr
zip -rq {name} {archive_root}
mv {name}.zip ..
cd ..
echo Should be in job directory now: $PWD | tee -a /dev/stderr
ls

# echo "Removing results directory..." | tee -a /dev/stderr
# rm -rf results

# dps-hyper view {name}.zip
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
        os.path.join(scratch,
                     'latest_{}_{}'.format(name, clean_pattern)))

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
            if test:
                command = ['sh', submit_script]
                print("Testing.")
                output = subprocess.check_output(command, stderr=subprocess.STDOUT)
                print(output)
            else:
                command = ['qsub', submit_script]
                print("Submitting.")
                output = subprocess.check_output(command, stderr=subprocess.STDOUT)
                print(output)

                # Create a file in the directory with the job_id as its name
                job_id = output.decode().split('.')[0]
                open(job_id, 'w').close()
                print("Job ID: {}".format(job_id))
        except subprocess.CalledProcessError as e:
            print("CalledProcessError has been raised while executing command: {}.".format(' '.join(command)))
            print("Output of process: ")
            print(e.output.decode())
            raise_with_traceback(e)


def _submit_job():
    from clify import command_line
    command_line(submit_job, collect_kwargs=1, verbose=True)()


if __name__ == "__main__":
    _submit_job()
