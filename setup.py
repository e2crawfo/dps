import subprocess
try:
    import setuptools
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    setuptools = use_setuptools()

from setuptools import find_packages, setup  # noqa: F811

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

for item in requirements:
    # we want to handle package names and also repo urls
    print(dir(item))
    print(item.req)
    print(item.link)

    link = None
    if getattr(item, 'url', None):   # older pip has url
        link = str(item.url)
    elif getattr(item, 'link', None):  # newer pip has link
        link = str(item.link)

    if link is not None:
        if item.editable:
            command = 'pip install -e {}'.format(link)
            print("Installing editable repo with command: {}".format(command))
            subprocess.run(command.split())
        else:
            links.append(link)

    if item.req:
        requires.append(str(item.req))


setup(
    name='dps',
    author="Eric Crawford",
    author_email="eric.crawford@mail.mcgill.ca",
    version='0.1',
    packages=find_packages(),
    setup_requires=['numpy>=1.7'],
    install_requires=requires,
    dependency_links=links,
    entry_points={
        'console_scripts': ['dps-hyper=dps.hyper.command_line:dps_hyper_cl',
                            'dps-run=dps.run:run',
                            'readme=dps.utils.base:view_readme_cl',
                            'tf-inspect=dps.utils.tf:tf_inspect_cl',
                            'git-summary=dps.utils.base:git_summary_cl',
                            'report-to-videos=dps.utils.html_report:report_to_videos_cl']
    }
)
