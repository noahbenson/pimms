# -*- coding: utf-8 -*-
################################################################################
# pimms/lazydict/__init__.py
#
# Lazy persistent dictionaries, related types, and utility functions.
#
# @author Noah C. Benson

"""Declarations of the lazydict type.

The `pimms.lazydict` module contains the definition of the lazydict (or ldict)
type and the various utility functions that go with it.
"""

from ._core import (delay, is_delay, is_dask_delayed, is_delayed, undelay,
                    frozendict, lazydict, ldict, is_lazydict, is_ldict,
                    lazykeymap, lazyvalmap, lazyitemmap,
                    keymap, valmap, itemmap,
                    dictmap, frozendictmap, fdictmap, lazydictmap, ldictmap,
                    merge, rmerge, assoc, dissoc)
# We want frozendict to have an fdict alias.
from ._core import frozendict as fdict

