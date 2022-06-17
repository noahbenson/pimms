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

from ._core import (Delay, delay, weak_delay, wdelay, is_delay, undelay,
                    lazydict, ldict, is_lazydict, is_ldict,
                    keymap, valmap, itemmap, merge, rmerge)

