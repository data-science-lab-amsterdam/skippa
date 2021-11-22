#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

from skippa import __version__


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = ['poandas', 'scikit-learn', 'dill']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Robert van Straalen",
    author_email='tech@datasciencelab.nl',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    description="SciKIt-learn Pipeline in PAndas",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='preprocessing pipeline pandas sklearn',
    name='skippa',
    packages=find_packages(include=['skippa', 'skippa.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://dev.azure.com/data-science-lab/Intern/_git/skippa',
    version=__version__,
    zip_safe=False,
)
