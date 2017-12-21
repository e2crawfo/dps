import subprocess


def grep(pattern, filename, options=""):
    return subprocess.check_output('grep {} "{}" {}'.format(options, pattern, filename), shell=True).decode()


for n_threads in range(5):
    for salience_model in [True, False]:
        command = "dps-run grid_arithmetic a2c --max-steps=2001 --salience-model={salience_model} --intra-op-parallelism-threads={n_threads} --inter-op-parallelism-threads={n_threads}".format(n_threads=n_threads, salience_model=salience_model)
        subprocess.run(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
        stdout = grep("time_per_batch", "/data/dps_data/logs/grid_arithmetic/latest/stdout")
        lines = [l for l in stdout.split('\n') if l]
        time_per_batch = float(lines[-1].split()[-1])
        print("n_threads: {}, salience_model: {}, time_per_batch: {}".format(n_threads, salience_model, time_per_batch))