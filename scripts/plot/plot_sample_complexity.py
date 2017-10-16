import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.stats.api as sms
from scipy.stats import sem
plt.style.use('ggplot')

from dps.utils import Config
from dps.parallel.base import ReadOnlyJob


def plot(spread_measure='std'):
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

    field_to_plot = 'test_reward'
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    groups = sorted(groups, key=lambda x: x[0])

    label_order = []

    for i, (k, _df) in enumerate(groups):
        _groups = _df.groupby('n_train')
        values = list(_groups)
        x = [v[0] for v in values]
        ys = [-100 * v[1][field_to_plot] for v in values]

        y = [_y.mean() for _y in ys]

        if spread_measure == 'std_dev':
            y_upper = y_lower = [_y.std() for _y in ys]
        elif spread_measure == 'conf_int':
            conf_int = [sms.DescrStatsW(_y.values).tconfint_mean() for _y in ys]
            y_lower = y - np.array([ci[0] for ci in conf_int])
            y_upper = np.array([ci[1] for ci in conf_int]) - y
        elif spread_measure == 'std_err':
            y_upper = y_lower = [sem(_y.values) for _y in ys]
        else:
            pass

        yerr = np.vstack((y_lower, y_upper))

        c = colours[i % len(colours)]
        label = "n_hidden_units={}".format(k)
        plt.semilogx(x, y, label=label, c=c, basex=2)
        label_order.append(label)
        plt.gca().errorbar(x, y, yerr=yerr, c=c)

    ax = plt.gca()
    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]
    ax.legend(ordered_handles, label_order, loc='upper right', ncol=1)
    plt.grid(True)

    plt.ylim((0.0, 100.0))

    plt.ylabel("% Incorrect on Test Set")
    plt.xlabel("# Training Examples")
    plt.show()
    plt.savefig('cnn_results.png')


if __name__ == "__main__":
    plot('std_dev')
