import argparse

from dps.parallel import Job, ReadOnlyJob
from dps.utils import ipdb_postmortem


def run_command(path, **kwargs):
    """ Implements the `run` sub-command, which executes some of the job's operators. """
    job = Job(path)
    job.run(**kwargs)


def view_command(path, verbose):
    """ Implements the `view` sub-command, which prints a summary of a job. """
    job = ReadOnlyJob(path)
    print(job.summary(verbose=verbose))


class SubCommand(object):
    def __init__(self, name, help, function):
        self.name = name
        self.help = help
        self.function = function
        self.arguments = []

    def add_argument(self, *args, **kwargs):
        self.arguments.append((args, kwargs))


def parallel_cl(desc, additional_cmds=None):
    """ Entry point for command-line utility to which additional sub-commands
        can be added.

    Parameters
    ----------
    desc: str
        Description for the script.
    additional_cmds: list
        Each element is an instance of SubCommand.

    """
    desc = desc or 'Run jobs and view their statuses.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--pdb', action='store_true',
        help="If supplied, enter post-mortem debugging on error.")

    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help='Run a job.')
    run_parser.add_argument('path', type=str)
    run_parser.add_argument('pattern', type=str)
    run_parser.add_argument('indices', nargs='*', type=int)
    run_parser.add_argument('--idx-in-node', type=int, default=-1)
    run_parser.add_argument('--tasks-per-node', type=int, default=-1)
    run_parser.add_argument('--gpu-set', type=str, default="")
    run_parser.add_argument('--ignore-gpu', action="store_true")
    run_parser.add_argument(
        '--force', action='store_true',
        help="If supplied, run the selected operators even "
             "if they've already been completed.")
    run_parser.add_argument(
        '--output-to-files', action='store_true',
        help="If supplied, output is stored in files rather than being printed.")
    run_parser.add_argument(
        '-v', '--verbose', action='count', default=0, help="Increase verbosity.")

    run_parser.set_defaults(func=run_command)

    view_parser = subparsers.add_parser('view', help='View status of a job.')
    view_parser.add_argument('path', type=str)
    view_parser.add_argument(
        '-v', '--verbose', action='count', default=0, help="Increase verbosity.")

    view_parser.set_defaults(func=view_command)

    subparser_names = ['run', 'view']

    additional_cmds = additional_cmds or []
    for sub_command in additional_cmds:
        subparser_names.append(sub_command.name)
        cmd_parser = subparsers.add_parser(sub_command.name, help=sub_command.help)
        for args, kwargs in sub_command.arguments:
            cmd_parser.add_argument(*args, **kwargs)
        cmd_parser.set_defaults(func=sub_command.function)

    args, _ = parser.parse_known_args()

    try:
        func = args.func
    except AttributeError:
        raise ValueError(
            "Missing ``command`` argument to script. Should be one of "
            "{}.".format(subparser_names))

    pdb = args.pdb
    del args.pdb
    del args.func
    print(vars(args))

    if pdb:
        with ipdb_postmortem():
            func(**vars(args))
    else:
        func(**vars(args))
