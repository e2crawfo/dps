import numpy as np
import pandas as pd
from pprint import pprint
import subprocess
import matplotlib.pyplot as plt


from dps.utils import Config, process_path, confidence_interval, standard_error
from dps.parallel.command_line import SubCommand, parallel_cl
from dps.hyper import HyperSearch
from dps.hyper.submit_job import DEFAULT_HOST_POOL


def _print_config_cmd(path):
    search = HyperSearch(path)

    print("BASE CONFIG")
    print(search.objects.load_object('metadata', 'config'))

    dist = search.objects.load_object('metadata', 'dist')
    dist = Config(dist)

    print('\n' + '*' * 100)
    print("PARAMETER DISTRIBUTION")
    pprint(dist)


def _summarize_search_cmd(path, no_config, verbose):
    search = HyperSearch(path)
    search.print_summary(print_config=not no_config, verbose=verbose)


def _value_plot_cmd(path, mode, field, style):
    """ Plot the trajectory of a single value, specified by field, for each parameter
        setting in a hyperparameter search.

    """
    path = process_path(path)

    assert mode in "train val test update".split()

    print("Plotting {} value of field {} from experiments "
          "stored at {}.".format(mode, field, path))

    search = HyperSearch(path)
    keys = search.dist_keys()

    data = search.extract_step_data(mode, field, -1)

    n_plots = len(data) + 1
    w = int(np.ceil(np.sqrt(n_plots)))
    h = int(np.ceil(n_plots / w))

    with plt.style.context(style):
        fig, axes = plt.subplots(h, w, sharex=True, sharey=True, figsize=(15, 10))
        axes = np.atleast_2d(axes)
        final_ax = axes[-1, -1]

        label_order = []

        for n, key in enumerate(sorted(data)):
            label = ",".join("{}={}".format(k, v) for k, v in zip(keys, key))
            label_order.append(label)

            i = int(n / w)
            j = n % w
            ax = axes[i, j]
            for vd in data[key]:
                ax.plot(vd)
            ax.set_title(label)
            mean = pd.concat(data[key], axis=1).mean(axis=1)
            final_ax.plot(mean, label=label)

        legend_handles = {l: h for h, l in zip(*final_ax.get_legend_handles_labels())}
        ordered_handles = [legend_handles[l] for l in label_order]

        final_ax.legend(
            ordered_handles, label_order, loc='center left',
            bbox_to_anchor=(1.05, 0.5), ncol=1)

        plt.subplots_adjust(
            left=0.05, bottom=0.05, right=0.86, top=0.97, wspace=0.05, hspace=0.18)

        plt.savefig('valueplot_mode={}_field={}'.format(mode, field))
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
    summary_cmd.add_argument(
        '-v', '--verbose', action='count', default=0, help="Increase verbosity.")

    style_list = ['default', 'classic'] + sorted(style for style in plt.style.available if style != 'classic')

    value_plot_cmd = SubCommand(
        'value_plot', 'Plot the trajectory of a value throughout all training runs.',
        _value_plot_cmd)

    value_plot_cmd.add_argument('path', help="Path to directory for search.", type=str, default="results.zip", nargs='+')
    value_plot_cmd.add_argument('mode', help="Data-collection mode to plot data from.", choices="train val test update".split(), default="")
    value_plot_cmd.add_argument('field', help="Field to plot.", default="")
    value_plot_cmd.add_argument('--style', help="Style for plot.", choices=style_list, default="ggplot")

    search_plot_cmd = SubCommand(
        'search_plot', 'Plot a the search.', _search_plot_cmd)
    search_plot_cmd.add_argument('path', help="Path to directory for search.", type=str)
    search_plot_cmd.add_argument('x_field', help="Name of field used as plot x-values.", type=str)
    search_plot_cmd.add_argument('y_field', help="Name of field used as plot y-values.", type=str)
    search_plot_cmd.add_argument('groupby', help="Field by which points are grouped into lines", type=str)
    search_plot_cmd.add_argument('--style', help="Style for plot.", choices=style_list, default="ggplot")
    search_plot_cmd.add_argument('--spread-measure', help="Measure of spread to use for error bars.",
                                 choices="std_dev conf_int std_err".split(), default="std_err")

    probe_hosts_cmd = SubCommand(
        'probe_hosts', 'Check the status of the hosts in host_pool.', _probe_hosts)

    parallel_cl(
        'Build, run, plot and view results of hyper-parameter searches.',
        [config_cmd, summary_cmd, value_plot_cmd, search_plot_cmd, probe_hosts_cmd])
