#! /usr/bin/env python
####################################################################################################

from setuptools import setup

setup(
    name='pimms',
    version='0.1.2',
    description='Python immutable data structures library',
    keywords='persistent immutable functional',
    author='Noah C. Benson',
    author_email='nben@nyu.edu',
    url='https://github.com/noahbenson/pimms/',
    license='GPLv3',
    packages=['pimms', 'pimms.test'],
    package_data={'': ['LICENSE.txt']},
    include_package_data=True,
    install_requires=['pysistence>=0.4'])
