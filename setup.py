#! /usr/bin/env python
####################################################################################################

from setuptools import (setup, Extension)

setup(
    name='pimms',
    version='0.3.24',
    description='Python immutable data structures library',
    keywords='persistent immutable functional',
    author='Noah C. Benson',
    author_email='nben@nyu.edu',
    url='https://github.com/noahbenson/pimms/',
    license='GPLv3',
    packages=['pimms', 'pimms.test'],
    #ext_modules=[Extension('pimms.c_util', ['pimms/c_util.c'])],
    package_data={'': ['LICENSE.txt']},
    include_package_data=True,
    install_requires=['pyrsistent>=0.11',
                      'six>=1.10',
                      'numpy>=1.2',
                      'scipy>=1.2',
                      'pint>=0.7'])
