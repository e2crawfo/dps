from setuptools import setup, find_packages
setup(
    name='dps',
    author="Eric Crawford",
    author_email="eric.crawford@mail.mcgill.ca",
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['dps-hyper=dps.hyper.command_line:dps_hyper_cl',
                            'dps-run=dps.run:run',
                            'readme=dps.utils.base:view_readme_cl',
                            'tf-inspect=dps.utils.tf:tf_inspect_cl',
                            'git-summary=dps.utils.base:git_summary_cl',
                            'report-to-videos=dps.utils.html_report:report_to_videos_cl']
    }
)
