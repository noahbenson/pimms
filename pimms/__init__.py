# -*- coding: utf-8 -*-
################################################################################
# pimms/__init__.py
#
# This source-code file is part of the pimms library.
#
# The pimms library is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

'''The Python Immutables and scientific utilities library.

The `pimms` is the **P**ython **imm**utable **s**cientific toolkit. It is a
library designed to enable immutable data structures and lazy computation in a
scientific context, and it works primarily via a collection of utility functions
and through the use of decorators, which are generally applied to classes and
their members to declare how an immutable data-structure's members are related.
Taken together, these utilities form a DSL-like system for declaring immutable
data-structures with full inheritance support.
'''

# Imports ######################################################################
#
# The doc namespace contains docwrap, docproc, and make_docproc, of which we
# only want to import the first two.
from .doc import (docwrap, docproc)
# The types, lazydict, and calculation namespaces all contain clean slates of
# functions to import; the private state for these is all hidden in submodules.
from .types import *
from .lazydict import *
from .calculation import *
from .plantype import *
#from .immutable   import (immutable, require, value, param, option, is_imm,
#                          is_imm_type, imm_copy,
#                          imm_persist, imm_transient, imm_params, imm_values,
#                          imm_dict,
#                          imm_is_persistent, imm_is_transient)
#from .cmdline     import (argv_parse, argv_parser, to_argv_schema,
#                          CommandLineParser,
#                          WorkLog, worklog)
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
