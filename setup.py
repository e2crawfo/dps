import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install as install_command
from setuptools.command.develop import develop as develop_command
from setuptools.command.egg_info import egg_info as egg_info_command

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

links = []
requires = []

try:
    requirements = list(parse_requirements('requirements.txt'))
except Exception:
    # new versions of pip requires a session
    requirements = list(parse_requirements('requirements.txt', session=PipSession()))

to_manually_install = []

for item in requirements:
    # we want to handle package names and also repo urls
    link = None
    if getattr(item, 'url', None):   # older pip has url
        link = str(item.url)
    elif getattr(item, 'link', None):  # newer pip has link
        link = str(item.link)

    if link is not None and item.editable:
        to_manually_install.append(link)
        continue

    if link is not None:
        links.append(link)

    if item.req:
        requires.append(str(item.req))

manual_install_done = False


def manual_install():
    global to_manually_install, manual_install_done

    if not manual_install_done:
        for link in to_manually_install:
            print("Manually installing editable url: {}".format(link))
            command = 'pip install -v -e {}'.format(link)
            subprocess.run(command.split())


class InstallCommand(install_command):
    def run(self):
        print("$$$ In install command $$$")
        manual_install()
        super().run()


class DevelopCommand(develop_command):
    def run(self):
        print("$$$ In develop command $$$")
        manual_install()
        super().run()


class EggInfoCommand(egg_info_command):
    def run(self):
        print("$$$ In egginfo command $$$")
        manual_install()
        super().run()


setup(
    name='dps',
    author="Eric Crawford",
    author_email="eric.crawford@mail.mcgill.ca",
    version='0.1',
    packages=find_packages(),
    install_requires=requires,
    dependency_links=links,
    cmdclass={'install': InstallCommand, 'develop': DevelopCommand, 'egg_info': EggInfoCommand},
    entry_points={
        'console_scripts': ['dps-hyper=dps.hyper.command_line:dps_hyper_cl',
                            'dps-run=dps.run:run',
                            'readme=dps.utils.base:view_readme_cl',
                            'tf-inspect=dps.utils.tf:tf_inspect_cl',
                            'git-summary=dps.utils.base:git_summary_cl',
                            'report-to-videos=dps.utils.html_report:report_to_videos_cl']
    }
)

raise Exception()
