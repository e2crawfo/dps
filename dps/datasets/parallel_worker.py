def build_dataset(directory, dataset_cls, params, inp, verbose):
    import os
    import datetime
    from dps.utils import get_default_config, redirect_stream
    from pyvirtualdisplay import Display
    from contextlib import ExitStack

    with ExitStack() as stack:
        pid = os.getpid()
        stdout = os.path.join(directory, f"{pid}.stdout")
        stderr = os.path.join(directory, f"{pid}.stderr")

        stack.enter_context(redirect_stream('stdout', stdout, tee=verbose))
        stack.enter_context(redirect_stream('stderr', stderr, tee=True))

        os.nice(10)

        print("Entered build_dataset at: ")
        print(datetime.datetime.now())

        idx, seed, n_examples = inp
        print(f"pid: {pid}, idx: {idx}, seed: {seed}, n_examples: {n_examples}")

        # del os.environ['DISPLAY']
        os.environ['DISPLAY'] = f":{idx+1}"
        print(f"Process with pid {pid} using display {os.environ['DISPLAY']}.")

        params = params.copy()
        params.update(seed=seed, n_examples=n_examples)
        params.build_dataset_n_workers = 0  # Don't spawn new processes from this worker process.

        default_config = get_default_config()
        default_config.cache_dir = directory
        default_config.visualize_dataset = False

        with default_config:
            print("Before opening display...")
            with Display(visible=False):
                print("Building dataset.")
                dataset = dataset_cls(**params, worker_idx=idx)

        print("Leaving build_dataset at: ")
        print(datetime.datetime.now())

        return dataset.param_hash, dataset.n_examples_written
