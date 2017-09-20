import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dps.utils import Config
from dps.parallel.base import ReadOnlyJob


def plot(path):
    job = ReadOnlyJob(path)

    distributions = job.objects.load_object('metadata', 'distributions')
    distributions = Config(distributions)
    keys = ['epsilon', 'opt_steps_per_update']

    records = []

    for i, op in enumerate(job.completed_ops()):
        if 'map' in op.name:
            try:
                r = op.get_outputs(job.objects)[0]
            except BaseException as e:
                print("Exception thrown when accessing output of op {}:\n    {}".format(op.name, e))
        idx = r['config']['idx']
        record = dict(idx=idx)
        record.update({key: r['config'][key] for key in keys})
        record['loss'] = r['history'][-1]['val_data'][-1]['PPO:avg_cumulative_reward']
        records.append(record)
    df = pd.DataFrame.from_records(records)
    for key in keys:
        df[key] = df[key].fillna(-np.inf)

    # epsilon x opt_steps_per_update
    data = -np.inf * np.ones((len(distributions['epsilon']), len(distributions['opt_steps_per_update'])))
    groups = df.groupby(keys)

    for k, _df in groups:
        if k[0] == -np.inf:
            k = (None, k[1])
        epsilon_idx = list(distributions['epsilon']).index(k[0])
        step_idx = list(distributions['opt_steps_per_update']).index(k[1])
        data[epsilon_idx, step_idx] = _df['loss'].median()

    data[data == -np.inf] = data[data != -np.inf].mean()

    with plt.style.context({'axes.grid': False}):
        plt.imshow(data)
        plt.xlabel("epsilon")
        plt.ylabel("opt_steps_per_update")
    plt.show()


if __name__ == "__main__":
    plot("/tmp/dps/search/execution/PPOExperiment_map_2017_09_17_17_27_50/results.zip")
