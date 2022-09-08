# -*- coding: utf-8 -*-
################################################################################
# pimms/__init__.py
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

'''The Python Immutables and scientific utilities library.

The `pimms` library is the **P**ython **imm**utable **s**cientific toolkit. It
is a library designed to enable immutable data structures and lazy computation
in a scientific context, and it works primarily via a collection of utility
functions and through the use of decorators, which are generally applied to
classes and their members to declare how an immutable data-structure's members
are related.  Taken together, these utilities form a DSL-like system for
declaring workflows and immutable data-structures with full inheritance support.
'''

# Imports ######################################################################
#
# The doc namespace contains docwrap, docproc, and make_docproc, of which we
# only want to import the first two.
from .doc import (docwrap, docproc)
# The types, lazydict, and calculation namespaces all contain clean slates of
# functions to import; the private state for these is all hidden in submodules.
from .types       import *
from .lazydict    import *
from .calculation import *
from .plantype    import *
# Import the Global UnitRegistry object to the global pimms scope. This is the
# value that gets updated when one runs `pimms.default_ureg()`, and this is the
# UnitRegistry that is used as the default registry for all `pimms` functions.
from .types._quantity import _initial_global_ureg as units
"""UnitRegistry: the registry for units tracked by pimms.

`pimms.units` is a global `pint`-module unit registry that can be used as a
single global place for tracking units. Pimms functions that interact with units
generally take an argument `ureg` that can be used to modify this registry.
Additionally, the default registry (this object, `pimms.units`) can be
temporarily changed in a local block using `with pimms.default_ureg(ureg): ...`.
"""

# Modules/Reloading ############################################################
submodules = ('pimms.doc._core',
              'pimms.doc',
              'pimms.types._core',
              'pimms.types._numeric',
              'pimms.types._quantity',
              'pimms.types',
              'pimms.lazydict._core',
              'pimms.lazydict',
              'pimms.calculation._core',
              'pimms.calculation',
              'pimms.plantype._core',
              'pimms.plantype')
"""tuple: a list of all pimms subpackage names in load-order.

`pimms.submodules` is a tuple of strings, each of which is the name of one of
the sub-submodules in `pimms`. The modules are listed in load-order and all
`pimms` submodules are included.
"""
def reload_pimms():
    """Reload and return the entire `pimms` package.

    `pimms.reload_pimms()` reloads every submodule in the `pimms` package then
    reloads `pimms` itself, and returns the reloaded package.

    This function exists primarily for debugging purposes; its use is not
    generally needed or advised.
    """
    import sys, importlib
    for mod in modules:
        reload(sys.modules[mod])
    return reload(sys.modules['pimms'])

# Package Meta-Code ############################################################
__version__ = '1.0.0rc1'
__all__ = tuple([k
                 for k in locals()
                 if k[0] != '_'
                 if k != 'reload_pimms'
                 if k != 'description'
                 if k != 'submodules'
                 if ('pimms.' + k) not in submodules])
