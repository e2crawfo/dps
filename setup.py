try:
    import setuptools
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    setuptools = use_setuptools()

from os.path import dirname, realpath
from setuptools import find_packages, setup  # noqa: F811


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f if not line.startswith('git+')]


setup(
    name='dps',
    author="Eric Crawford",
    author_email="eric.crawford@mail.mcgill.ca",
    version='0.1',
    packages=find_packages(),
    setup_requires=['numpy>=1.7'],
    install_requires=_read_requirements_file(),
    entry_points={
        'console_scripts': ['dps-hyper=dps.hyper.command_line:dps_hyper_cl',
                            'dps-run=dps.run:run',
                            'readme=dps.utils.base:view_readme_cl',
                            'tf-inspect=dps.utils.tf:tf_inspect_cl']
    }
)
