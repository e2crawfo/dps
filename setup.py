try:
    import setuptools
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    setuptools = use_setuptools()

from setuptools import find_packages, setup  # noqa: F811

setup(
    name='dps',
    author="Eric Crawford",
    author_email="eric.crawford@mail.mcgill.ca",
    version='0.1',
    packages=find_packages(),
    setup_requires=['numpy>=1.7'],
    install_requires=[
        "numpy",
        "pytest",
        "future",
        "six",
        "tensorflow",
        "gym",
        "tabulate"
    ],
    entry_points={
        'console_scripts': ['dps_parallel=dps.parallel:parallel_cl']}
)
