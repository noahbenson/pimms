# -*- coding: utf-8 -*-
################################################################################
# pimms/_version.py
# Loads the version and splits it into meaningful pieces.

import os
from collections import namedtuple

try:
    from importlib.metadata import version as get_version
except ModuleNotFoundError:
    try:
        from importlib_metadata import version as get_version
    except ModuleNotFoundError:
        pass


# PimmsVersion Type ############################################################

_PimmsVersionBase = namedtuple(
    '_PimmsVersionBase',
    ['string', 'major', 'minor', 'micro', 'rc', 'tuple'])
class PimmsVersion(_PimmsVersionBase):
    """The type that manages the version information for the pimms package."""
    __slots__ = ()
    def __new__(cls, string=None):
        if string is None:
            # Try to deduce the version string.
            try:
                string = get_version(__package__)
            except PackageNotFoundError:
                # Probably pimms isn't installed via pip or poetry so it doesn't
                # show up as a registered module. As a backup, we can grab it
                # from the pyproject.toml file.
                pyproject_toml_path = os.path.join(
                    os.path.split(__file__)[0], '..', 'pyproject.toml')
                with open(os.path.join(base_path, 'pyproject.toml'), 'r') as fl:
                    pyproject_toml_lines = fl.read().split('\n')
                for ln in toml_lines:
                    ln = ln.strip()
                    if ln.startswith('version = '):
                        string = ln.split('"')[1]
                        break
        if string is None:
            from warnings import warn
            warn("pimms could not detect its version number")
            (major, minor, micro, rc) = (None, None, None, None)
        else:
            s = string
            if 'rc' in string:
                (s,rc) = s.split('rc')
                rc = int(rc)
            else:
                rc = None
            (major, minor, micro) = s.split('.')
            major = int(major)
            minor = int(minor)
            micro = int(micro)
        tup = tuple(u for u in (major, minor, micro, rc) if u is not None)
        return super(PimmsVersion, cls).__new__(
            cls, string=string, major=major, minor=minor, micro=micro, rc=rc, tuple=tup)
    def __str__(self):
        return self.string
    def __repr__(self):
        return f"PimmsVersion({repr(self.string)})"
    def __iter__(self):
        return iter(self.tuple)
    def __reversed__(self):
        return reversed(self.tuple)
    def __contains__(self, k):
        if isinstance(k, str):
            return k in self.string
        else:
            return k in self.tuple
# Declare the version object; this will automatically detect the version.
version = PimmsVersion()
