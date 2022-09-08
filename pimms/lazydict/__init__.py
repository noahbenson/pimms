# -*- coding: utf-8 -*-
################################################################################
# pimms/lazydict/__init__.py
#
# Lazy persistent dictionaries, related types, and utility functions.
#
# Copyright 2022 Noah C. Benson
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Declarations of the lazydict type.

The `pimms.lazydict` module contains the definition of the lazydict (or ldict)
type and the various utility functions that go with it.
"""

from ._core import (delay, is_delay, is_dask_delayed, is_delayed, undelay,
                    frozendict, fdict, lazydict, ldict, is_lazydict, is_ldict,
                    lazykeymap, lazyvalmap, lazyitemmap,
                    keymap, valmap, itemmap,
                    dictmap, frozendictmap, fdictmap, lazydictmap, ldictmap,
                    merge, rmerge, assoc, dissoc,
                    lambdadict)

