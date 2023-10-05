#! /usr/bin/env python
################################################################################

from setuptools import setup
import os

base_path = os.path.dirname(__file__)
with open(os.path.join(base_path, 'requirements.txt'), 'r') as fl:
    requirements = fl.read().split('\n')
    requirements = [k for k in requirements if k.strip() != '']
with open(os.path.join(base_path, 'pyproject.toml'), 'r') as fl:
    toml_lines = fl.read().split('\n')
version = None
for ln in toml_lines:
    ln = ln.strip()
    if ln.startswith('version = '):
        version = ln.split('"')[1]
        break
with open(os.path.join(base_path, 'pimms', '__init__.py'), 'r') as fl:
    init_text = fl.read()
desc = init_text.split("'''")[1]

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
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
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
    packages=['pimms.doc', 'pimms.util', 'pimms.workflow',
              'pimms.pathlib', 'pimms.iolib',
              'pimms',
              'pimms.test.doc', 'pimms.test.iolib', 'pimms.test.pathlib',
              'pimms.test.util', 'pimms.test.workflow',
              'pimms.test'],
    package_data={'': ['LICENSE.txt']},
    include_package_data=True,
    install_requires=requirements)
