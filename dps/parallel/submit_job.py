from __future__ import print_function
import os
import datetime
import subprocess
from future.utils import raise_with_traceback
from datetime import timedelta
from pathlib import Path

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
    """ s should be of the form HH:MM:SS """
    args = [int(i) for i in s.split(":")]
    return timedelta(hours=args[0], minutes=args[1], seconds=args[2])


def submit_job(
        input_zip, pattern, scratch, n_jobs=-1, n_nodes=1, ppn=12, walltime="1:00:00",
        cleanup_time="00:15:00", add_date=True, test=0,
        show_script=0, dry_run=0, queue=None,
        parallel_exe="parallel", sdbin='$HOME/.virtualenvs/dps/bin/'):

    idx_file = 'job_indices.txt'
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
    execution_time = int((walltime - cleanup_time).total_seconds())

    node_file = " --sshloginfile $PBS_NODEFILE "

    kwargs = locals().copy()

    code = '''
#!/bin/bash

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
mkdir results
echo "Starting job at - "
date

echo "Printing RAMDISK..."
{parallel_exe} --no-notice {node_file} --nonall \\
    echo Ramdisk on host \\$HOSTNAME is \\$RAMDISK, working directory is \\$PWD.

echo "Staging input archive..."
{parallel_exe} --no-notice {node_file} --nonall \\
    cp {input_zip_abs} \\$RAMDISK

echo "Listing staging results..."
{parallel_exe} --no-notice {node_file} --nonall \\
    "echo ls on node \\$HOSTNAME && ls"

echo "Unzipping..."
{parallel_exe} --no-notice {node_file} --nonall \\
    "unzip \\$RAMDISK/{input_zip_base}"

echo "Running parallel..."
timeout --signal=TERM {execution_time}s \\
    {parallel_exe} --no-notice -j{ppn} --workdir $PWD \\
        --joblog {job_directory}/job_log.txt --env OMP_NUM_THREADS --env PATH \\
        {node_file}\\
        dps-hyper run {pattern} {{}} --verbose < {idx_file}

if [ "$?" -eq 124 ]; then
    echo Timed out after {execution_time} seconds.
fi

echo "Cleaning up at - "
date

{parallel_exe} --no-notice {node_file} --nonall \\
    "echo Retrieving results from node \\$HOSTNAME &&
     cd \\$RAMDISK &&
     dps-hyper zip {archive_root} \\$HOSTNAME.zip &&
     mv \\$HOSTNAME.zip {job_directory}/results"

cd {job_directory}/results
echo In job_directory/results: $PWD
ls

echo "Unzipping results from different nodes..."
for f in *zip
do
    echo "Storing results from node "$f
    unzip -nuq $f
    rm $f
done

echo "Zipping final results..."
dps-hyper zip {name} {archive_root}
mv {name}.zip ..
cd ..
echo Should be in job directory now: $PWD
ls

echo "Removing results directory..."
rm -rf results

dps-hyper view {name}.zip
mv {input_zip} {input_zip}.bk
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

    ro_job = ReadOnlyJob(input_zip)
    completion = ro_job.completion(pattern)
    if completion['n_complete'] == completion['n_ops']:
        print("All jobs are finished! Exiting.")
        return

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
                # Submit to queue
                output = subprocess.check_output(command, stderr=subprocess.STDOUT)
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
    from clify import command_line
    command_line(submit_job, collect_kwargs=1, verbose=True)()


if __name__ == "__main__":
    _submit_job()
