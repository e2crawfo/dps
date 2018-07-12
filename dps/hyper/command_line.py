import numpy as np
import pandas as pd
from pprint import pprint
import subprocess
import matplotlib.pyplot as plt
import inspect
import os
import json
import dill

import clify

from dps.utils import Config, process_path, confidence_interval, standard_error
from dps.parallel.command_line import SubCommand, parallel_cl
from dps.hyper import HyperSearch
from dps.hyper.submit_job import DEFAULT_HOST_POOL, submit_job, ParallelSession


def _print_config_cmd(path):
    search = HyperSearch(path)

    print("BASE CONFIG")
    print(search.objects.load_object('metadata', 'config'))

    dist = search.objects.load_object('metadata', 'dist')
    dist = Config(dist)

    print('\n' + '*' * 100)
    print("PARAMETER DISTRIBUTION")
    pprint(dist)


def _summarize_search_cmd(path, no_config, verbose, criteria, maximize):
    search = HyperSearch(path)
    search.print_summary(print_config=not no_config, verbose=verbose, criteria=criteria, maximize=maximize)


def _value_plot_cmd(path, mode, field, stage, x_axis, ylim, style):
    """ Plot the trajectory of a single value, specified by field, for each parameter
        setting in a hyperparameter search.

    Example:
         dps-hyper value_plot . rl_val COST_negative_mAP None --x-axis=stages --ylim="-1,0"

    Parameters
    ----------
    path: str
        Path passed to `HyperSearch`.
    mode: str
        Run mode to plot from (e.g. train, val).
    field: str
        Name of value to plot.
    stage: str
        String that is eval-ed to get an object specifying the stages to plot data from.
    x_axis: str, one of {steps, experiences, stages}
        Specifiation of value to use as x-axis for plots. If `stages` is used, then only
        the value obtained by the "chosen" hypothesis for that stage is used.
    ylim: str
        String that is eval-ed to get a tuple specifying y-limits for plots.
    style: str
        Matplotlib style to use for plot.

    """
    print("Plotting {} value of field {} from experiments stored at `{}`.".format(mode, field, path))

    assert x_axis in "steps experiences stages"
    x_axis_key = dict(
        steps="global_step",
        experiences="n_global_experiences",
        stages="stage_idx")[x_axis]

    search = HyperSearch(path)

    stage = eval(stage) if stage else ""
    ylim = eval(ylim) if ylim else ""

    fields = [field, x_axis_key]

    if x_axis == "stages":
        data = search.extract_stage_data(fields, bare=True)
    else:
        data = search.extract_step_data(mode, fields, stage)

    n_plots = len(data) + 1
    w = int(np.ceil(np.sqrt(n_plots)))
    h = int(np.ceil(n_plots / w))

    with plt.style.context(style):
        fig, axes = plt.subplots(h, w, sharex=True, sharey=True, figsize=(15, 10))
        fig.suptitle("{} vs {}".format(field, x_axis_key))

        axes = np.atleast_2d(axes)
        final_ax = axes[-1, -1]

        label_order = []

        for i, (key, value) in enumerate(sorted(data.items())):
            label = ",".join("{}={}".format(*kv) for kv in zip(key._fields, key))
            i = int(key.idx / w)
            j = key.idx % w
            ax = axes[i, j]

            if ylim:
                ax.set_ylim(ylim)

            ax.set_title(label)
            label_order.append(label)

            for (repeat, seed), _data in value.items():
                ax.plot(_data[x_axis_key], _data[field])

            to_concat = [_data.set_index(x_axis_key) for _data in value.values()]
            concat = pd.concat(to_concat, axis=1, ignore_index=True)
            mean = concat.mean(axis=1)
            final_ax.plot(mean, label=label)

        legend_handles = {l: h for h, l in zip(*final_ax.get_legend_handles_labels())}
        ordered_handles = [legend_handles[l] for l in label_order]

        final_ax.legend(
            ordered_handles, label_order, loc='center left',
            bbox_to_anchor=(1.05, 0.5), ncol=1)

        if ylim:
            final_ax.set_ylim(ylim)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.90, wspace=0.05, hspace=0.18)

        plt.savefig('value_plot_mode={}_field={}_stage={}'.format(mode, field, stage))
        plt.show()


def _search_plot_cmd(
        path, y_field, x_field, groupby, spread_measure,
        style, do_legend=False, **axes_kwargs):

    path = process_path(path)
    print("Plotting searches stored at {}.".format(path))

    search = HyperSearch(path)

    with plt.style.context(style):
        ax = plt.axes(xlabel=x_field, ylabel=y_field, **axes_kwargs)

        dist = search.objects.load_object('metadata', 'dist')
        dist = Config(dist)

        df = search.extract_summary_data()

        groups = sorted(df.groupby(groupby))

        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

        legend = []

        for i, (k, _df) in enumerate(groups):
            values = list(_df.groupby(x_field))
            x = [v[0] for v in values]
            ys = [v[1][y_field] for v in values]

            y = [_y.mean() for _y in ys]

            if spread_measure == 'std_dev':
                y_upper = y_lower = [_y.std() for _y in ys]
            elif spread_measure == 'conf_int':
                conf_int = [confidence_interval(_y.values, 0.95) for _y in ys]
                y_lower = y - np.array([ci[0] for ci in conf_int])
                y_upper = np.array([ci[1] for ci in conf_int]) - y
            elif spread_measure == 'std_err':
                y_upper = y_lower = [standard_error(_y.values) for _y in ys]
            else:
                raise Exception("NotImplemented")

            yerr = np.vstack((y_lower, y_upper))

            c = colours[i % len(colours)]

            ax.semilogx(x, y, c=c, basex=2)
            handle = ax.errorbar(x, y, yerr=yerr, c=c)
            label = "{} = {}".format(groupby, k)

            legend.append((handle, label))

        if do_legend:
            handles, labels = zip(*legend)
            ax.legend(
                handles, labels, loc='center left',
                bbox_to_anchor=(1.05, 0.5), ncol=1)

        # plt.subplots_adjust(
        #     left=0.09, bottom=0.13, right=0.7, top=0.93, wspace=0.05, hspace=0.18)

    filename = "value_plot.pdf"
    print("Saving plot as {}".format(filename))
    plt.savefig(filename)


def _ssh_execute(command, host):
    ssh_options = (
        "-oPasswordAuthentication=no "
        "-oStrictHostKeyChecking=no "
        "-oConnectTimeout=5 "
        "-oServerAliveInterval=2"
    )
    cmd = "ssh {ssh_options} -T {host} \"{command}\"".format(ssh_options=ssh_options, host=host, command=command)
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _probe_hosts(**_):
    """ Check the status of the hosts in the host pool. """
    n_connected = 0
    n_idle = 0

    for host in DEFAULT_HOST_POOL:
        if host is not ':':
            print("\n" + "*" * 80)
            print("Testing connection to host {}...".format(host))
            p = _ssh_execute("echo Connected to \$HOSTNAME", host)
            if p.returncode:
                print("Could not connect:")
                print(p.stdout.decode())
            else:
                n_connected += 1
                print("\nTOP:")
                p = _ssh_execute("top -bn2 | head -n 5", host)
                top_output = p.stdout.decode()
                print(top_output)

                cpu = top_output.split('\n')[2]
                start = cpu.find('ni')
                end = cpu.find('id')
                idle_cpu = float(cpu[start:end].split()[1])

                if idle_cpu > 95:
                    n_idle += 1

                print("\nWHO:")
                p = _ssh_execute("who", host)
                print(p.stdout.decode())

    print("Was able to connect to {} hosts.".format(n_connected))
    print("{} of those hosts have idle cpu percent > 95.".format(n_idle))


def _resubmit_cmd(path, name=""):
    """ Note the resubmitting still has a limitation: experiments are not copied over
        from the previous submission. Couldn't find a clean way to do this, so just do it manually
        for now. In the future we should revamp the build/run process so that the possibility of
        multiple runs is taken into account, and the results of the runs can be easily combined.

    """
    # Get run_kwargs from command line
    search = HyperSearch(path)
    archive_path = search.job.path

    try:
        with open(os.path.join(search.path, 'run_kwargs.json'), 'r') as f:
            reference = json.load(f)
    except FileNotFoundError:
        with open(os.path.join(search.path, 'session.pkl'), 'rb') as f:
            reference = dill.load(f).__dict__

    sig = inspect.signature(ParallelSession.__init__)
    _run_kwargs = sig.bind_partial()
    _run_kwargs.apply_defaults()

    run_kwargs = {}
    for k, v in _run_kwargs.arguments.items():
        run_kwargs[k] = reference[k]

    cl_run_kwargs = clify.command_line(run_kwargs).parse()
    run_kwargs.update(cl_run_kwargs)

    submit_job(archive_path, "resubmit", **run_kwargs)


def dps_hyper_cl():
    config_cmd = SubCommand(
        'config', 'Print config of a hyper-parameter search.',
        _print_config_cmd)

    config_cmd.add_argument('path', help="Location of directory for hyper-parameter search.", type=str)

    summary_cmd = SubCommand(
        'summary', 'Summarize results of a hyper-parameter search.',
        _summarize_search_cmd)

    summary_cmd.add_argument('path', help="Location of directory for hyper-parameter search.", type=str)
    summary_cmd.add_argument('--no-config', help="If supplied, don't print out config.", action='store_true')
    summary_cmd.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity.")
    summary_cmd.add_argument('--criteria', type=str, help="Criteria to order parameter settings by.")
    summary_cmd.add_argument('--maximize', action='store_true', help="Specify that the goal is to maximize the criteria.")

    style_list = ['default', 'classic'] + sorted(style for style in plt.style.available if style != 'classic')

    value_plot_cmd = SubCommand(
        'value_plot', 'Plot the trajectory of a value over time for all training runs (for comparing trajectories).',
        _value_plot_cmd)

    value_plot_cmd.add_argument('path', help="Path to directory for search.", type=str, default="results.zip")
    value_plot_cmd.add_argument('mode', help="Data-collection mode to plot data from.", default="val")
    value_plot_cmd.add_argument('field', help="Field to plot.", default="")
    value_plot_cmd.add_argument('stage', help="Stages to plot.", default="")
    value_plot_cmd.add_argument('--x-axis', help="Type of x-axis.", default="steps")
    value_plot_cmd.add_argument('--ylim', help="Y limits.", default="")
    value_plot_cmd.add_argument('--style', help="Style for plot.", choices=style_list, default="ggplot")

    search_plot_cmd = SubCommand(
        'search_plot', 'Plot values against one another.',
        _search_plot_cmd)

    search_plot_cmd.add_argument('path', help="Path to directory for search.", type=str)
    search_plot_cmd.add_argument('x_field', help="Name of field used as plot x-values.", type=str)
    search_plot_cmd.add_argument('y_field', help="Name of field used as plot y-values.", type=str)
    search_plot_cmd.add_argument('groupby', help="Field by which points are grouped into lines", type=str)
    search_plot_cmd.add_argument('--style', help="Style for plot.", choices=style_list, default="ggplot")
    search_plot_cmd.add_argument('--spread-measure', help="Measure of spread to use for error bars.",
                                 choices="std_dev conf_int std_err".split(), default="std_err")

    probe_hosts_cmd = SubCommand(
        'probe_hosts', 'Check the status of the hosts in host_pool.', _probe_hosts)

    resubmit_cmd = SubCommand('resubmit', 'Resubmit a job.', _resubmit_cmd)
    resubmit_cmd.add_argument('path', help="Path to directory for search.", type=str)

    parallel_cl(
        'Build, run, plot and view results of hyper-parameter searches.',
        [config_cmd, summary_cmd, value_plot_cmd, search_plot_cmd, probe_hosts_cmd, resubmit_cmd])
