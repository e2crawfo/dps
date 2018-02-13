from cycler import cycler
import os
import argparse
import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
from dps.hyper import extract_data_from_job
from dps.utils import process_path, Config, sha_cache, set_clear_cache


cache_dir = process_path('/home/eric/.cache/dps_plots')
plot_dir = './plots'

single_op_paths = Config()
single_op_paths['sum:dps'] = 'sample_efficiency/dps_new/sum/results.zip'
single_op_paths['sum:cnn_pretrained'] = 'sample_efficiency/cnn_pretrained/sum/results.zip'
single_op_paths['sum:cnn'] = 'sample_efficiency/cnn/sum/exp_cnn_sum_12hours_2017_10_21_00_54_52.zip'
single_op_paths['sum:cnn_pretrained_pure'] = 'sample_efficiency/cnn_pretrained_pure/sum/results.zip'
single_op_paths['sum:rnn_pure'] = 'sample_efficiency/rnn_pure/sum/results.zip'
single_op_paths['sum:rnn_pretrained_pure'] = 'sample_efficiency/rnn_pretrained_pure/sum/results.zip'

single_op_paths['prod:dps'] = 'sample_efficiency/dps_new/prod/results.zip'
single_op_paths['prod:cnn_pretrained'] = 'sample_efficiency/cnn_pretrained/prod/results.zip'
single_op_paths['prod:cnn'] = 'sample_efficiency/cnn/prod/exp_cnn_prod_12hours_2017_10_21_00_55_50.zip'
single_op_paths['prod:cnn_pretrained_pure'] = 'sample_efficiency/cnn_pretrained_pure/prod/results.zip'
single_op_paths['prod:rnn_pure'] = 'sample_efficiency/rnn_pure/prod/results.zip'
single_op_paths['prod:rnn_pretrained_pure'] = 'sample_efficiency/rnn_pretrained_pure/prod/results.zip'

single_op_paths['max:dps'] = 'sample_efficiency/dps_new/max/results.zip'
single_op_paths['max:cnn_pretrained'] = 'sample_efficiency/cnn_pretrained/max/results.zip'
single_op_paths['max:cnn'] = 'sample_efficiency/cnn/max/exp_cnn_max_12hours_2017_10_21_00_56_53.zip'
single_op_paths['max:cnn_pretrained_pure'] = 'sample_efficiency/cnn_pretrained_pure/max/results.zip'
single_op_paths['max:rnn_pure'] = 'sample_efficiency/rnn_pure/max/results.zip'
single_op_paths['max:rnn_pretrained_pure'] = 'sample_efficiency/rnn_pretrained_pure/max/results.zip'

single_op_paths['min:dps'] = 'sample_efficiency/dps_new/min/results.zip'
single_op_paths['min:cnn_pretrained'] = 'sample_efficiency/cnn_pretrained/min/results.zip'
single_op_paths['min:cnn'] = 'sample_efficiency/cnn/min/exp_cnn_min_12hours_2017_10_21_00_57_32.zip'
single_op_paths['min:cnn_pretrained_pure'] = 'sample_efficiency/cnn_pretrained_pure/min/results.zip'
single_op_paths['min:rnn_pure'] = 'sample_efficiency/rnn_pure/min/results.zip'
single_op_paths['min:rnn_pretrained_pure'] = 'sample_efficiency/rnn_pretrained_pure/min/results.zip'

combined_paths = Config()
combined_paths['dps'] = 'sample_efficiency/dps_new/combined/results.zip'
combined_paths['cnn_pretrained'] = 'sample_efficiency/cnn_pretrained/combined/results.zip'
combined_paths['cnn'] = 'sample_efficiency/cnn/combined/exp_cnn_all_2017_10_23_01_22_58.zip'
combined_paths['cnn_pretrained_pure'] = 'sample_efficiency/cnn_pretrained_pure/combined/results.zip'
combined_paths['rnn_pure'] = 'sample_efficiency/rnn_pure/combined/results.zip'
combined_paths['rnn_pretrained_pure'] = 'sample_efficiency/rnn_pretrained_pure/combined/results.zip'

parity_paths = Config()
parity_paths['B:dps'] = 'curriculum_parity/dps_new/B/results.zip'
parity_paths['B:cnn'] = 'curriculum_parity/cnn/A/exp_final_cnn_parity_A_2017_10_26_23_08_20.zip'
parity_paths['B:cnn_pretrained'] = 'curriculum_parity/cnn_pretrained/B/results.zip'

parity_paths['C:dps'] = 'curriculum_parity/dps_new/C/results.zip'
parity_paths['C:cnn'] = 'curriculum_parity/cnn/B/exp_final_cnn_parity_B_2017_10_26_23_07_11.zip'
parity_paths['C:cnn_pretrained'] = 'curriculum_parity/cnn_pretrained/C/results.zip'

size_paths = Config()
size_paths["A:dps"] = 'curriculum_size/dps_new/A/results.zip'
size_paths["A:cnn"] = 'curriculum_size/cnn/A/results.zip'
size_paths['A:cnn_pretrained'] = 'curriculum_size/cnn_pretrained/A/results.zip'
size_paths['A:cnn_pretrained_pure'] = 'curriculum_size/cnn_pretrained_pure/A/results.zip'
size_paths['A:rnn_pure'] = 'curriculum_size/rnn_pure/A/results.zip'
size_paths['A:rnn_pretrained_pure'] = 'curriculum_size/rnn_pretrained_pure/A/results.zip'

size_paths["B:dps"] = 'curriculum_size/dps_new/B/results.zip'
size_paths["B:cnn"] = 'curriculum_size/cnn/B/results.zip'
size_paths['B:cnn_pretrained'] = 'curriculum_size/cnn_pretrained/B/results.zip'
size_paths['B:cnn_pretrained_pure'] = 'curriculum_size/cnn_pretrained_pure/B/results.zip'
size_paths['B:rnn_pure'] = 'curriculum_size/rnn_pure/B/results.zip'
size_paths['B:rnn_pretrained_pure'] = 'curriculum_size/rnn_pretrained_pure/B/results.zip'

size_paths["C:dps"] = 'curriculum_size/dps_new/C/results.zip'
size_paths["C:cnn"] = 'curriculum_size/cnn/C/results.zip'
size_paths['C:cnn_pretrained'] = 'curriculum_size/cnn_pretrained/C/results.zip'
size_paths['C:cnn_pretrained_pure'] = 'curriculum_size/cnn_pretrained_pure/C/results.zip'
size_paths['C:rnn_pure'] = 'curriculum_size/rnn_pure/C/results.zip'
size_paths['C:rnn_pretrained_pure'] = 'curriculum_size/rnn_pretrained_pure/C/results.zip'

size_paths["D:dps"] = 'curriculum_size/dps_new/D/results.zip'
size_paths["D:cnn"] = 'curriculum_size/cnn/D/results.zip'
size_paths['D:cnn_pretrained'] = 'curriculum_size/cnn_pretrained/D/results.zip'
size_paths['D:cnn_pretrained_pure'] = 'curriculum_size/cnn_pretrained_pure/D/results.zip'
size_paths['D:rnn_pure'] = 'curriculum_size/rnn_pure/D/results.zip'
size_paths['D:rnn_pretrained_pure'] = 'curriculum_size/rnn_pretrained_pure/D/results.zip'

size_paths["E:dps"] = 'curriculum_size/dps_new/E/results.zip'
size_paths["E:cnn"] = 'curriculum_size/cnn/E/results.zip'
size_paths['E:cnn_pretrained'] = 'curriculum_size/cnn_pretrained/E/results.zip'
size_paths['E:cnn_pretrained_pure'] = 'curriculum_size/cnn_pretrained_pure/E/results.zip'
size_paths['E:rnn_pure'] = 'curriculum_size/rnn_pure/E/results.zip'
size_paths['E:rnn_pretrained_pure'] = 'curriculum_size/rnn_pretrained_pure/E/results.zip'

size_paths["F:dps"] = 'curriculum_size/dps_new/F/results.zip'
size_paths["F:cnn"] = 'curriculum_size/cnn/F/results.zip'
size_paths['F:cnn_pretrained'] = 'curriculum_size/cnn_pretrained/F/results.zip'
size_paths['F:cnn_pretrained_pure'] = 'curriculum_size/cnn_pretrained_pure/F/results.zip'
size_paths['F:rnn_pure'] = 'curriculum_size/rnn_pure/F/results.zip'
size_paths['F:rnn_pretrained_pure'] = 'curriculum_size/rnn_pretrained_pure/F/results.zip'

size_paths['G:cnn_pretrained'] = 'curriculum_size/cnn_pretrained/G/results.zip'
size_paths['G:cnn_pretrained_pure'] = 'curriculum_size/cnn_pretrained_pure/G/results.zip'

size_paths['H:cnn_pretrained'] = 'curriculum_size/cnn_pretrained/H/results.zip'
size_paths['G:cnn_pretrained_pure'] = 'curriculum_size/cnn_pretrained_pure/G/results.zip'

op_paths = Config()
op_paths['B:dps'] = 'curriculum_op/dps_new/B/results.zip'
op_paths['B:cnn'] = 'curriculum_op/cnn/A/exp_final_cnn_op_A_2017_10_26_23_08_20.zip'
op_paths['B:cnn_pretrained'] = 'curriculum_op/cnn_pretrained/B/results.zip'

op_paths['C:dps'] = 'curriculum_op/dps_new/C/results.zip'
op_paths['C:cnn'] = 'curriculum_op/cnn/B/exp_final_cnn_op_B_2017_10_26_23_07_11.zip'
op_paths['C:cnn_pretrained'] = 'curriculum_op/cnn_pretrained/C/results.zip'

ablation_paths = Config()
ablation_paths['full_interface'] = 'sample_efficiency/dps/combined/results.zip'
ablation_paths['no_modules'] = 'ablations/no_modules/results.zip'
ablation_paths['no_transformations'] = 'ablations/no_transformations/results.zip'
ablation_paths['no_classifiers'] = 'ablations/no_classifiers/results.zip'


def std_dev(ys):
    y_upper = y_lower = [_y.std() for _y in ys]
    return y_upper, y_lower


def ci(data, coverage):
    return stats.t.interval(
        coverage, len(data)-1, loc=np.mean(data), scale=stats.sem(data))


def ci95(ys):
    conf_int = [ci(_y.values, 0.95) for _y in ys]
    y = [_y.mean() for _y in ys]
    y_lower = y - np.array([ci[0] for ci in conf_int])
    y_upper = np.array([ci[1] for ci in conf_int]) - y
    return y_upper, y_lower


def std_err(ys):
    y_upper = y_lower = [stats.sem(_y.values) for _y in ys]
    return y_upper, y_lower


spread_measures = {func.__name__: func for func in [std_dev, ci95, std_err]}


def test_01_loss(r):
    return 100 * r['test_01_loss']


@sha_cache(cache_dir)
def _extract_cnn_data(f, n_controller_units, spread_measure, y_func, groupby='n_train'):
    flat = False
    if isinstance(n_controller_units, int):
        n_controller_units = [n_controller_units]
        flat = True
    if isinstance(groupby, str):
        groupby = [groupby]

    data = {}
    df = extract_data_from_job(f, ['n_controller_units', 'curriculum:-1:do_train'] + groupby)
    df.loc[df['curriculum:-1:do_train'] == False, 'curriculum:-1:n_train'] = 0

    groups = df.groupby('n_controller_units')
    for i, (k, _df) in enumerate(groups):
        if k in n_controller_units:
            _groups = _df.groupby(groupby)
            values = [g for g in _groups]
            x = [v[0] for v in values]
            ys = [y_func(v[1]) for v in values]

            y = [_y.mean() for _y in ys]
            y_upper, y_lower = spread_measures[spread_measure](ys)

            data[k] = np.stack([x, y, y_upper, y_lower])

    if flat:
        return next(iter(data.values()))
    return data


@sha_cache(cache_dir)
def _extract_rl_data(f, spread_measure, y_func):
    data = {}
    df = extract_data_from_job(f, 'n_train')
    df.loc[df['total_steps'] == 0, 'n_train'] = 0
    groups = df.groupby('n_train')
    values = list(groups)
    x = [v[0] for v in values]
    ys = [y_func(v[1]) for v in values]

    y = [_y.mean() for _y in ys]
    y_upper, y_lower = spread_measures[spread_measure](ys)

    data = np.stack([x, y, y_upper, y_lower])
    return data


def gen_sample_efficiency_single_op():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 7))

    fig.text(0.52, 0.01, '# Training Examples', ha='center', fontsize=12)
    fig.text(0.01, 0.51, '% Test Error', va='center', rotation='vertical', fontsize=12)

    cnn_n_train = [32, 128, 512]
    spread_measure = 'std_err'

    pp = [
        dict(title='Sum', key='sum'),
        dict(title='Product', key='prod'),
        dict(title='Maximum', key='max'),
        dict(title='Minimum', key='min'),
    ]

    for i, (ax, p) in enumerate(zip(axes.flatten(), pp)):
        label_order = []

        x, y, *yerr = _extract_rl_data(single_op_paths[p['key']]['dps'], spread_measure, test_01_loss)
        label = 'RL + Interface'
        ax.errorbar(x, y, yerr=yerr, label=label, ls='--')
        label_order.append(label)

        data = _extract_cnn_data(
            single_op_paths[p['key']]['cnn_pretrained'], cnn_n_train, spread_measure, test_01_loss, 'n_train')

        for k, v in data.items():
            x, y, *yerr = v
            label = "CNN Pretrained - {} hidden units".format(k)
            ax.errorbar(x, y, yerr=yerr, label=label)
            label_order.append(label)

        data = _extract_cnn_data(
            single_op_paths[p['key']]['cnn_pretrained_pure'], cnn_n_train, spread_measure, test_01_loss, 'n_train')

        for k, v in data.items():
            x, y, *yerr = v
            label = "CNN Pretrained Pure - {} hidden units".format(k)
            ax.errorbar(x, y, yerr=yerr, label=label)
            label_order.append(label)

        data = _extract_cnn_data(
            single_op_paths[p['key']]['rnn_pure'], 128, spread_measure, test_01_loss, 'n_train')

        x, y, *yerr = data
        label = "RNN - 128 hidden units".format(k)
        ax.errorbar(x, y, yerr=yerr, label=label)
        label_order.append(label)

        data = _extract_cnn_data(
            single_op_paths[p['key']]['rnn_pretrained_pure'], 128, spread_measure, test_01_loss, 'n_train')

        x, y, *yerr = data
        label = "RNN Pretrained - 128 hidden units".format(k)
        ax.errorbar(x, y, yerr=yerr, label=label)
        label_order.append(label)

        ax.set_title(p['title'])
        ax.tick_params(axis='both', labelsize=14)

    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)
    ax.set_xticks(x)

    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    ax.legend(ordered_handles, label_order, loc='best', ncol=1)
    plt.subplots_adjust(
        left=0.07, bottom=0.08, right=0.98, top=0.95, wspace=0.05, hspace=0.15)

    fig.savefig(os.path.join(plot_dir, 'sample_efficiency.pdf'))
    return fig


def gen_sample_efficiency_combined():
    fig = plt.figure(figsize=(5, 3.5))

    ax = plt.gca()

    cnn_n_train = [32, 128, 512]
    spread_measure = 'std_err'

    label_order = []

    x, y, *yerr = _extract_rl_data(combined_paths['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--')
    label_order.append(label)

    data = _extract_cnn_data(combined_paths['cnn_pretrained'], cnn_n_train, spread_measure, test_01_loss, 'n_train')

    for k, v in data.items():
        x, y, *yerr = v
        label = "CNN - {} hidden units".format(k)
        label_order.append(label)
        ax.errorbar(x, y, yerr=yerr, label=label)

    data = _extract_cnn_data(
        combined_paths['cnn_pretrained_pure'], cnn_n_train, spread_measure, test_01_loss, 'n_train')

    for k, v in data.items():
        x, y, *yerr = v
        label = "CNN Pretrained Pure - {} hidden units".format(k)
        ax.errorbar(x, y, yerr=yerr, label=label)
        label_order.append(label)

    data = _extract_cnn_data(
        combined_paths['rnn_pure'], 128, spread_measure, test_01_loss, 'n_train')
    x, y, *yerr = data
    label = "RNN - 128 hidden units".format(k)
    ax.errorbar(x, y, yerr=yerr, label=label)
    label_order.append(label)

    data = _extract_cnn_data(
        combined_paths['rnn_pretrained_pure'], 128, spread_measure, test_01_loss, 'n_train')
    x, y, *yerr = data
    label = "RNN Pretrained - 128 hidden units".format(k)
    ax.errorbar(x, y, yerr=yerr, label=label)
    label_order.append(label)

    ax.set_ylabel('% Test Error', fontsize=12)
    ax.set_xlabel('# Training Examples', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)
    ax.set_xticks(x)

    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    ax.legend(ordered_handles, label_order, loc='best', ncol=1, fontsize=8)
    plt.subplots_adjust(left=0.16, bottom=0.15, right=0.97, top=0.96)
    fig.savefig(os.path.join(plot_dir, 'sample_efficiency_combined.pdf'))
    return fig


def gen_super_sample_efficiency():
    fig, _ = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10.7, 5.5))

    fig.text(0.52, 0.01, '# Training Examples', ha='center', fontsize=12)
    fig.text(0.01, 0.51, '% Test Error', va='center', rotation='vertical', fontsize=12)

    shape = (2, 4)
    indi_axes = [
        plt.subplot2grid(shape, (0, 0)),
        plt.subplot2grid(shape, (0, 1)),
        plt.subplot2grid(shape, (1, 0)),
        plt.subplot2grid(shape, (1, 1))
    ]
    combined_ax = plt.subplot2grid(shape, (0, 2), colspan=2, rowspan=2)

    pp = [
        dict(title='Sum', key='sum'),
        dict(title='Product', key='prod'),
        dict(title='Maximum', key='max'),
        dict(title='Minimum', key='min'),
    ]
    cnn_n_train = [32, 128, 512]
    spread_measure = 'std_err'

    for i, (ax, p) in enumerate(zip(indi_axes, pp)):
        x, y, *yerr = _extract_rl_data(single_op_paths[p['key']]['dps'], spread_measure, test_01_loss)
        label = 'RL + Interface'
        ax.errorbar(x, y, yerr=yerr, label=label, ls='--')

        cnn_sum_data = _extract_cnn_data(single_op_paths[p['key']]['cnn'], cnn_n_train, spread_measure, test_01_loss, 'n_train')

        for k, v in cnn_sum_data.items():
            x, y, *yerr = v
            label = "CNN - {} hidden units".format(k)
            ax.errorbar(x, y, yerr=yerr, label=label)

        ax.set_title(p['title'])
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylim((0.0, 100.0))
        ax.set_xscale('log', basex=2)

    # Combined
    combined_ax.set_title("Combined Task")
    combined_ax.tick_params(axis='both', labelsize=14)
    combined_ax.set_ylim((0.0, 100.0))
    combined_ax.set_xscale('log', basex=2)

    label_order = []

    x, y, *yerr = _extract_rl_data(combined_paths['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface'
    combined_ax.errorbar(x, y, yerr=yerr, label=label, ls='--')
    label_order.append(label)

    cnn_all_data = _extract_cnn_data(combined_paths['cnn'], cnn_n_train, spread_measure, test_01_loss, 'n_train')

    for k, v in cnn_all_data.items():
        x, y, *yerr = v
        label = "CNN - {} hidden units".format(k)
        label_order.append(label)
        combined_ax.errorbar(x, y, yerr=yerr, label=label)

    legend_handles = {l: h for h, l in zip(*combined_ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    combined_ax.legend(ordered_handles, label_order, loc='best', ncol=1)

    plt.subplots_adjust(
        left=0.09, bottom=0.11, right=0.97, top=0.95, wspace=0.13, hspace=0.20)

    fig.savefig(os.path.join(plot_dir, 'sample_efficiency_super.pdf'))
    return fig


def gen_size_curriculum():
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

    spread_measure = 'std_err'
    fig.text(0.52, 0.01, '# Training Examples on Test Task', ha='center', fontsize=12)

    # ********************************************************************************
    ax = axes[0]

    xticks = set()
    label_order = []

    x, y, *yerr = _extract_rl_data(size_paths['A']['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)
    xticks |= set(x)

    rl_colour = line.get_c()

    x, y, *yerr = _extract_rl_data(size_paths['D']['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=rl_colour)
    label_order.append(label)
    xticks |= set(x)

    x, y, *yerr = _extract_cnn_data(size_paths['A']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)
    xticks |= set(x)

    cnn_colour = line.get_c()

    x, y, *yerr = _extract_cnn_data(size_paths['D']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=cnn_colour)
    label_order.append(label)
    xticks |= set(x)


    x, y, *yerr = _extract_cnn_data(size_paths['A']['cnn_pretrained_pure'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN Pretrained Pure - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)
    xticks |= set(x)

    cnn_pretrained_pure_colour = line.get_c()

    x, y, *yerr = _extract_cnn_data(size_paths['D']['cnn_pretrained_pure'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN Pretrained Pure - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=cnn_pretrained_pure_colour)
    label_order.append(label)
    xticks |= set(x)


    x, y, *yerr = _extract_cnn_data(size_paths['A']['rnn_pure'], 128, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'RNN Pure - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)
    xticks |= set(x)

    rnn_pure_colour = line.get_c()

    x, y, *yerr = _extract_cnn_data(size_paths['D']['rnn_pure'], 128, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'RNN Pure - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=rnn_pure_colour)
    label_order.append(label)
    xticks |= set(x)


    x, y, *yerr = _extract_cnn_data(size_paths['A']['rnn_pretrained_pure'], 128, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'RNN Pretrained Pure - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)
    xticks |= set(x)

    rnn_pretrained_pure_colour = line.get_c()

    x, y, *yerr = _extract_cnn_data(size_paths['D']['rnn_pretrained_pure'], 128, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'RNN Pretrained Pure - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=rnn_pretrained_pure_colour)
    label_order.append(label)
    xticks |= set(x)


    if not args.paper:
        x, y, *yerr = _extract_cnn_data(size_paths['G']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
        label = 'CNN - Alternate Curric 1'
        ax.errorbar(x, y, yerr=yerr, label=label, ls='-.', c=cnn_colour)
        label_order.append(label)
        xticks |= set(x)

        x, y, *yerr = _extract_cnn_data(size_paths['H']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
        label = 'CNN - Alternate Curric 2'
        ax.errorbar(x, y, yerr=yerr, label=label, ls=':', c=cnn_colour)
        label_order.append(label)
        xticks |= set(x)

    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    ax.legend(ordered_handles, label_order, loc='best', ncol=1)

    ax.set_title('3x3, 2-3 digits')
    ax.set_ylabel('% Test Error', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('symlog', basex=2)
    ax.set_xticks(sorted(xticks))

    # ********************************************************************************
    ax = axes[1]
    xticks = set()

    x, y, *yerr = _extract_rl_data(size_paths['B']['dps'], spread_measure, test_01_loss)
    ax.errorbar(x, y, yerr=yerr, ls='-', c=rl_colour)
    xticks |= set(x)

    x, y, *yerr = _extract_rl_data(size_paths['E']['dps'], spread_measure, test_01_loss)
    ax.errorbar(x, y, yerr=yerr, ls='--', c=rl_colour)
    xticks |= set(x)


    x, y, *yerr = _extract_cnn_data(size_paths['B']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='-', c=cnn_colour)
    xticks |= set(x)

    x, y, *yerr = _extract_cnn_data(size_paths['E']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='--', c=cnn_colour)
    xticks |= set(x)


    x, y, *yerr = _extract_cnn_data(size_paths['B']['cnn_pretrained_pure'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='-', c=cnn_pretrained_pure_colour)
    xticks |= set(x)

    x, y, *yerr = _extract_cnn_data(size_paths['E']['cnn_pretrained_pure'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='--', c=cnn_pretrained_pure_colour)
    xticks |= set(x)


    x, y, *yerr = _extract_cnn_data(size_paths['B']['rnn_pure'], 128, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='-', c=rnn_pure_colour)
    xticks |= set(x)

    x, y, *yerr = _extract_cnn_data(size_paths['E']['rnn_pure'], 128, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='--', c=rnn_pure_colour)
    xticks |= set(x)


    x, y, *yerr = _extract_cnn_data(size_paths['B']['rnn_pretrained_pure'], 128, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='-', c=rnn_pretrained_pure_colour)
    xticks |= set(x)

    x, y, *yerr = _extract_cnn_data(size_paths['E']['rnn_pretrained_pure'], 128, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='--', c=rnn_pretrained_pure_colour)
    xticks |= set(x)

    ax.set_title('3x3, 4 digits')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('symlog', basex=2)
    ax.set_xticks(sorted(xticks))

    # ********************************************************************************
    # ax = axes[2]
    # xticks = set()

    # x, y, *yerr = _extract_rl_data(size_paths['C']['dps'], spread_measure, test_01_loss)
    # ax.errorbar(x, y, yerr=yerr, ls='-', c=rl_colour)
    # xticks |= set(x)

    # x, y, *yerr = _extract_rl_data(size_paths['F']['dps'], spread_measure, test_01_loss)
    # ax.errorbar(x, y, yerr=yerr, ls='--', c=rl_colour)
    # xticks |= set(x)

    # x, y, *yerr = _extract_cnn_data(size_paths['C']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    # ax.errorbar(x, y, yerr=yerr, ls='-', c=cnn_colour)
    # xticks |= set(x)

    # x, y, *yerr = _extract_cnn_data(size_paths['F']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    # ax.errorbar(x, y, yerr=yerr, ls='--', c=cnn_colour)
    # xticks |= set(x)

    # ax.set_title('3x3, 5 digits')
    # ax.tick_params(axis='both', labelsize=14)
    # ax.set_ylim((0.0, 100.0))
    # ax.set_xscale('symlog', basex=2)
    # ax.set_xticks(sorted(xticks))

    plt.subplots_adjust(
        left=0.09, bottom=0.11, right=0.97, top=0.95, wspace=0.13, hspace=0.20)

    fig.savefig(os.path.join(plot_dir, 'size_curriculum.pdf'))
    return fig


def gen_parity_curriculum():
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 5))
    spread_measure = 'std_err'
    fig.text(0.52, 0.01, '# Training Examples on Test Task', ha='center', fontsize=12)

    ax.set_ylabel('% Test Error', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)

    label_order = []

    x, y, *yerr = _extract_rl_data(parity_paths['B']['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    rl_colour = line.get_c()

    x, y, *yerr = _extract_rl_data(parity_paths['C']['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=rl_colour)
    label_order.append(label)

    x, y, *yerr = _extract_cnn_data(parity_paths['B']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    cnn_colour = line.get_c()

    x, y, *yerr = _extract_cnn_data(parity_paths['C']['cnn_pretrained'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=cnn_colour)
    label_order.append(label)

    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    ax.legend(ordered_handles, label_order, loc='best', ncol=1)

    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('symlog', basex=2)
    plt.subplots_adjust(left=0.09, bottom=0.11, right=0.97, top=0.95, wspace=0.13, hspace=0.20)

    fig.savefig(os.path.join(plot_dir, 'parity_curriculum.pdf'))
    return fig


def gen_curric():
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 3.7))
    spread_measure = 'std_err'
    fig.text(0.52, 0.01, '# Training Examples on Test Task', ha='center', fontsize=12)

    ax = axes[0]

    ax.set_title('3x3, 2-3 digits')
    ax.set_ylabel('% Test Error', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)

    label_order = []

    x, y, *yerr = _extract_rl_data(size_paths['A']['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    rl_colour = line.get_c()

    x, y, *yerr = _extract_rl_data(size_paths['C']['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=rl_colour)
    label_order.append(label)

    x, y, *yerr = _extract_cnn_data(size_paths['A']['cnn'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    cnn_colour = line.get_c()

    x, y, *yerr = _extract_cnn_data(size_paths['C']['cnn'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=cnn_colour)
    label_order.append(label)

    ax = axes[1]
    ax.set_title('3x3, 4 digits')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)

    x, y, *yerr = _extract_rl_data(size_paths['B']['dps'], spread_measure, test_01_loss)
    ax.errorbar(x, y, yerr=yerr, ls='-', c=rl_colour)

    x, y, *yerr = _extract_rl_data(size_paths['F']['dps'], spread_measure, test_01_loss)
    ax.errorbar(x, y, yerr=yerr, ls='--', c=rl_colour)

    x, y, *yerr = _extract_cnn_data(size_paths['B']['cnn'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='-', c=cnn_colour)

    x, y, *yerr = _extract_cnn_data(size_paths['F']['cnn'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    ax.errorbar(x, y, yerr=yerr, ls='--', c=cnn_colour)

    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)

    ax = axes[2]
    ax.set_title('even -> odd')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)

    label_order = []

    x, y, *yerr = _extract_rl_data(parity_paths['A']['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    rl_colour = line.get_c()

    x, y, *yerr = _extract_rl_data(parity_paths['C']['dps'], spread_measure, test_01_loss)
    label = 'RL + Interface - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=rl_colour)
    label_order.append(label)

    x, y, *yerr = _extract_cnn_data(parity_paths['A']['cnn'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN - With Curric'
    line, _, _ = ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    cnn_colour = line.get_c()

    x, y, *yerr = _extract_cnn_data(parity_paths['C']['cnn'], 512, spread_measure, test_01_loss, 'curriculum:-1:n_train')
    label = 'CNN - No Curric'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='--', c=cnn_colour)
    label_order.append(label)

    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)

    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    ax.legend(ordered_handles, label_order, loc='best', ncol=1, fontsize=8)

    plt.subplots_adjust(
        left=0.08, bottom=0.15, right=0.98, top=0.91, wspace=0.13, hspace=0.20)

    fig.savefig(os.path.join(plot_dir, 'curriculum.pdf'))
    return fig


def gen_ablations():
    plt.figure(figsize=(5, 3.5))

    ax = plt.gca()

    ax.set_ylabel('% Test Error', fontsize=12)
    ax.set_xlabel('# Training Examples', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim((0.0, 100.0))
    ax.set_xscale('log', basex=2)

    spread_measure = 'std_err'

    label_order = []

    x, y, *yerr = _extract_rl_data(ablation_paths['full_interface'], spread_measure, lambda r: 100 * r['test_loss'])
    label = 'Full Interface'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    x, y, *yerr = _extract_rl_data(ablation_paths['no_modules'], spread_measure, test_01_loss)
    label = 'No Modules'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    x, y, *yerr = _extract_rl_data(ablation_paths['no_transformations'], spread_measure, test_01_loss)
    label = 'No Transformations'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    x, y, *yerr = _extract_rl_data(ablation_paths['no_classifiers'], spread_measure, test_01_loss)
    label = 'No Classifiers'
    ax.errorbar(x, y, yerr=yerr, label=label, ls='-')
    label_order.append(label)

    legend_handles = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ordered_handles = [legend_handles[l] for l in label_order]

    ax.legend(ordered_handles, label_order, loc='best', ncol=1, fontsize=8)
    plt.subplots_adjust(left=0.16, bottom=0.15, right=0.97, top=0.96)
    fig.savefig(os.path.join(plot_dir, 'ablations.pdf'))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plots", nargs='+')
    parser.add_argument("--style", default="bmh")
    parser.add_argument("--no-block", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()
    plt.rc('lines', linewidth=1)

    color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']
    os.makedirs(plot_dir, exist_ok=True)

    if args.clear_cache:
        set_clear_cache(True)

    with plt.style.context(args.style):
        plt.rc('axes', prop_cycle=(cycler('color', color_cycle)))

        funcs = {
            "single_op": gen_sample_efficiency_single_op,
            "combined": gen_sample_efficiency_combined,
            "super": gen_super_sample_efficiency,
            "size": gen_size_curriculum,
            "parity": gen_parity_curriculum,
            "curriculum": gen_curric,
            "ablations": gen_ablations,
        }

        for name, do_plot in funcs.items():
            if name in args.plots:
                fig = do_plot()

                if args.show:
                    plt.show(block=not args.no_block)
