import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dps.utils import Config
from dps.parallel.base import ReadOnlyJob


def _summarize_search():
    """ Get all completed jobs, get their outputs. Summarize em. """
    job = ReadOnlyJob('results.zip')
    distributions = job.objects.load_object('metadata', 'distributions')
    distributions = Config(distributions)
    keys = list(distributions.keys())

    records = []
    for op in job.completed_ops():
        if 'map' in op.name:
            try:
                r = op.get_outputs(job.objects)[0]
            except BaseException as e:
                print("Exception thrown when accessing output of op {}:\n    {}".format(op.name, e))

        record = r['history'][-1].copy()
        record['host'] = r['host']
        record['op_name'] = op.name
        del record['best_path']

        if len(record['train_data']) > 0:
            for k, v in record['train_data'].iloc[-1].items():
                record[k + '_train'] = v
        if len(record['update_data']) > 0:
            for k, v in record['update_data'].iloc[-1].items():
                record[k + '_update'] = v
        if len(record['val_data']) > 0:
            for k, v in record['val_data'].iloc[-1].items():
                record[k + '_val'] = v

        del record['train_data']
        del record['update_data']
        del record['val_data']

        config = Config(r['config'])
        for k in keys:
            record[k] = config[k]

        record.update(
            latest_stage=r['history'][-1]['stage'],
            total_steps=sum(s['n_steps'] for s in r['history']),
        )

        record['seed'] = r['config']['seed']
        records.append(record)

    df = pd.DataFrame.from_records(records)
    for key in keys:
        df[key] = df[key].fillna(-np.inf)

    groups = df.groupby('n_controller_units')

    for k, _df in groups:
        _groups = _df.groupby('n_train')
        values = list(_groups)
        x = [v[0] for v in values]
        y = [v[1]['best_loss'].mean() for v in values]

        y_lower = y - np.array([v[1]['best_loss'].quantile(0.25) for v in values])
        y_upper = np.array([v[1]['best_loss'].quantile(0.75) for v in values]) - y

        yerr = np.vstack((y_lower, y_upper))

        plt.plot(x, y, label="n_hidden_units={}".format(k))
        plt.gca().errorbar(x, y, yerr=yerr)

    plt.ylabel("test_error")
    plt.xlabel("n_training_examples")
    plt.legend()
    plt.show()
    plt.savefig('cnn_results.png')


if __name__ == "__main__":
    _summarize_search()