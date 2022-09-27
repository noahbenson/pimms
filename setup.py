#! /usr/bin/env python
################################################################################

from setuptools import setup
import os

base_path = os.path.dirname(__file__)
with open(os.path.join(base_path, 'requirements.txt'), 'r') as fl:
    requirements = fl.read().split('\n')
with open(os.path.join(base_path, 'pimms', '__init__.py'), 'r') as fl:
    init_lines = fl.read().split('\n')
version = None
desc = None
for ln in init_lines:
    if ln.startswith('__version__ = '):
        version = ln.split("'")
        if len(version) != 3: version = ln.split('"')
        version = version[1]
    elif ln.startswith('description = '):
        desc = ln.split("'")
        if len(desc) != 3: desc = ln.split('"')
        desc = desc[1]
setup(
    name='pimms',
    version=version,
    description=desc,
    keywords='persistent immutable functional scientific workflow',
    author='Noah C. Benson',
    author_email='nben@uw.edu',
    url='https://github.com/noahbenson/pimms',
    download_url='https://github.com/noahbenson/pimms',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'],
    license='MIT',
    packages=['pimms.doc', 'pimms.util', 'pimms.lazydict', 'pimms.workflow',
              'pimms', 'pimms.test'],
    package_data={'': ['LICENSE.txt']},
    include_package_data=True,
    install_requires=requirements)
