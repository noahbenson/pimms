# -*- coding: utf-8 -*-
################################################################################
# pimms/__init__.py

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

from .doc      import *
from .util     import *
from .pathlib  import *
from .workflow import *
# We want the version object from the ._version namespace.
from ._version import version
# Import the Global UnitRegistry object to the global pimms scope. This is the
# value that gets updated when one runs `pimms.default_ureg()`, and this is the
# UnitRegistry that is used as the default registry for all `pimms` functions.
from .util._quantity import _initial_global_ureg as units
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
              'pimms.util._core',
              'pimms.util._numeric',
              'pimms.util._quantity',
              'pimms.util',
              'pimms.pathlib._osf',
              'pimms.pathlib._cache',
              'pimms.pathlib._core',
              'pimms.pathlib',
              'pimms.workflow._core',
              'pimms.workflow._plantype',
              'pimms.workflow',
              'pimms._version')
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
    for mod in submodules:
        importlib.reload(sys.modules[mod])
    return importlib.reload(sys.modules['pimms'])


# Package Meta-Code ############################################################

__version__ = version.string
__all__ = [
    k for k in locals()
    if k[0] != '_'
    if k != 'reload_pimms'
    if k != 'submodules'
    if ('pimms.' + k) not in submodules
]
