# -*- coding: utf-8 -*-
####################################################################################################
# pimms/types/_core.py
# Core implementation of the utility classes for the various types that are managed by pimms.
# By Noah C. Benson

# Dependencies #####################################################################################
import inspect, types, sys, pint, os, numbers, warnings, base64
import collections as colls, numpy as np, pyrsistent as pyr
import scipy.sparse as sps

from ..doc import docwrap

# Units ############################################################################################
# Units are fundamentally treated as part of the pimms type-system. Pimms functios that deal with
# an object's type typically take an option `unit` that can be used to change the function's
# behavior depending on the units attached to an object.
# Setup pint / units:
from pint import UnitRegistry
@docwrap
def is_ureg(obj):
    """Returns `True` if an object is a `ping.UnitRegistry` object.

    `is_ureg(obj)` returns `True` if the given object `obj` is an instance
    of the `pint.UnitRegistry` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `UnitRegistry` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `UnitRegistry`, otherwise `False`.
    """
    return isinstance(obj, UnitRegistry)
from pint import Unit
@docwrap
def is_unit(q, ureg=Ellipsis):
    """Returns `True` if `q` is a `pint` unit and `False` otherwise.

    `is_unit(q)` returns `True` if `q` is a `pint` unit and `False` otherwise.

    Parameters
    ----------
    q : object
        The object whose quality as a `pint` unit is to be assessed.
    ureg : UnitRegistry or None, optional
        The `pint` `UnitRegistry` object that the given unit object must belong
        to. If `Ellipsis` (the default), then any unit registry is allowed. If
        `None`, then the `pimms.units` registry is used. Otherwise, this must be
        a specific `UnitRegistry` object.

    Returns
    -------
    boolean
        `True` if `q` is a `pint` unit and `False` otherwise.

    Raises
    ------
    TypeError
        If the `ureg` parameter is not a `UnitRegistry`, `Ellipsis`, or `None`.
    """
    if ureg is Ellipsis:
        return isinstance(q, Unit)
    elif ureg is None:
        from ._quantity import units
        return isinstance(q, units.Unit)
    elif is_ureg(ureg):
        return isinstance(q, ureg.Unit)
    else:
        raise TypeError("parameter ureg must be a UnitRegistry")
@docwrap
def is_quant(obj, unit=None, ureg=None):
    """Returns `True` if given a `pint` quantity and `False` otherwise.

    `is_quant(q)` returns `True` if `q` is a `pint` quantity and `False`
    otherwise. The optional parameter `unit` may additionally specify a unit
    that `obj` must be compatible with.

    Parameters
    ----------
    obj : object
        The object whose quality as a quantity is to be assessed.
    unit : UnitLike or None, optional
        The unit that the object must have in order to be considered valid. This
        may be a `pint` unit or unit-name (see also `pimms.unit`), a list or
        tuple of such units/unit-names, or `None`. If `None` (the default), then
        the object must be a `Quantity` object, but it doesn't matter what the
        unit of the object is. Otherwise, the object must have a unit equivalent
        to the unit or to one of the units given. The `UnitRegistry` objects for
        the units given via this parameter are ignored; only the `ureg`
        parameter influences the `UnitRegistry` requirements.
    ureg : pint.UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use for units. If `Ellipsis`, then
        `pimms.units` is used. If `ureg` is `None` (the default), then a
        specific unit registry is not checked.

    Returns
    -------
    boolean
        `True` if `obj` is a `pint` quantity whose units are compatible with the
        requested `unit` and `False` otherwise.

    Raises
    ------
    %(pimms.types._core.is_unit.raises)s
    """
    if ureg is None:
        if not isinstance(obj, pint.Quantity):
            return False
    else:
        if ureg is Ellipsis:
            from ._quantity import units
            ureg = units
        elif not is_reg(ureg):
            raise TypeError("parameter ureg must be a UnitRegistry")
        if not isinstance(obj, ureg.Quantity):
            return False
    return True if unit is None else obj.is_compatible_with(unit)

# Numpy Arrays #####################################################################################
from numpy import ndarray
@docwrap
def is_ndarray(obj, dtype=None, ureg=None, unit=Ellipsis):
    """Returns `True` if an object is a `numpy.ndarray` object, else `False`.

    `is_ndarray(obj)` returns `True` if the given object `obj` is an instance of
    the `numpy.ndarray`. Additional constraints may be placed on the object via
    the optional argments.

    Note that the `is_ndarray()` function is intended as a simple test of
    whether an object is a NumPy array; it does not recognize `Quantity`
    objects; for that, see `is_array` and `is_numeric`.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy array object is to be assessed.
    dtype : NumPy dtype-like or None, optional
        The NumPy `dtype` that is required of the `obj` in order to be
        considered a valid `ndarray`. The `obj.dtype` matches the given `dtype`
        parameter if either `dtype` is `None` (the default) or if `obj.dtype` is
        a sub-dtype of `dtype` according to `numpy.issubdtype`. Alternately,
        `dtype` can be a tuple, in which case, `obj` is considered valid if its
        dtype is a sub-dtype of any of the dtypes in `dtype`.

    Returns
    -------
    boolean
        `True` if `obj` is a valid numpy array, otherwise `False`.
    """
    # First, let's do the easy things
    if not isinstance(obj, ndarray): return False
    if dtype is None: return True
    dt = obj.dtype
    from numpy import issubdtype, dtype as to_dtype
    if is_tuple(dtype):
        for dtype in dtype:
            if issubdtype(dt, to_dtype(dtype)): return True
        return False
    else:
        return issubdtype(dt, to_dtype(dtype))

# Strings ##########################################################################################
@docwrap
def is_str(obj):
    """Returns `True` if an object is a string and `False` otherwise.

    `is_str(obj)` returns `True` if the given object `obj` is an instance
    of the `str` type or is a scalar `numpy` array of type with a `dtype` that
    is a subtype of `numpy.unicode_`.

    Parameters
    ----------
    obj : object
        The object whose quality as a string object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is a string, otherwise `False`.
    """
    return isinstance(obj, str) or is_ndarray(obj, dtype=str, unit=False)

# Builtin Python Abstract Types ####################################################################
from collections.abc import Callable
@docwrap
def is_callable(obj):
    """Returns `True` if an object is a callable object like a function.

    `is_callable(obj)` returns `True` if the given object `obj` is an instance
    of the `collections.abc.Callable` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Callable` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Callable`, otherwise `False`.
    """
    return isinstance(obj, Callable)
from collections.abc import Sized
@docwrap
def is_sized(obj, unit=Ellipsis):
    """Returns `True` if an object implements `len()`, otherwise `False`.

    `is_sized(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.abc.Sized` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Sized` object is to be assessed.
    unit : UnitLike, optional
        The unit that the object must have in order to be considered valid. This
        may be a `pint` unit or unit-name (see also `pimms.unit`), a list or
        tuple of such units/unit-names, `True`, `False`, `None`, or
        `Ellipsis`. If `True`, then the object must have a unit, but it doesn't
        matter what the unit is. If `False`, then the object must not have a
        unit (note that for `True` and `False` values, a dimesionless object is
        different from an object without an attached unit). If `Ellipsis`, then
        either an object with a unit or without a unit is accepted (this is the
        default).  Otherwise, the object must have a unit equivalent to the unit
        or to one of the units given. A value of `None` is interpreted as a unit
        that can either be dimensionless or an object without any unit; i.e., it
        is equivalent to `(False, 'dimensionless')`.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Sized`, otherwise `False`.
    """
    return isinstance(obj, Sized)
from collections.abc import Container
@docwrap
def is_container(obj):
    """Returns `True` if an object implements `__contains__`, otherwise `False`.

    `is_container(obj)` returns `True` if the given object `obj` is an instance
    of the `collections.abc.Container` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Container` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Container`, otherwise `False`.
    """
    return isinstance(obj, Container)
from collections.abc import Iterable
@docwrap
def is_iterable(obj):
    """Returns `True` if an object implements `__iter__`, otherwise `False`.

    `is_iterable(obj)` returns `True` if the given object `obj` is an instance
    of the `collections.abc.Iterable` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Iterable` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Iterable`, otherwise `False`.
    """
    return isinstance(obj, Iterable)
from collections.abc import Iterator
@docwrap
def is_iterator(obj):
    """Returns `True` if an object is an instance of `collections.abc.Iterator`.

    `is_iterable(obj)` returns `True` if the given object `obj` is an instance
    of the `collections.abc.Iterator` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Iterator` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Iterator`, otherwise `False`.
    """
    return isinstance(obj, Iterator)
from collections.abc import Reversible
@docwrap
def is_reversible(obj):
    """Returns `True` if an object is an instance of `Reversible`.

    `is_reversible(obj)` returns `True` if the given object `obj` is an instance
    of the `collections.abc.Reversible` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Reversible` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Reversible`, otherwise `False`.
    """
    return isinstance(obj, Reversible)
from collections.abc import Collection
@docwrap
def is_coll(obj):
    """Returns `True` if an object is a collection (a sized iterable container).

    `is_coll(obj)` returns `True` if the given object `obj` is an instance
    of the `collections.abc.Collection` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Collection` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Collection`, otherwise `False`.
    """
    return isinstance(obj, Collection)
from collections.abc import Sequence
@docwrap
def is_seq(obj):
    """Returns `True` if an object is a sequence, otherwise `False`.

    `is_seq(obj)` returns `True` if the given object `obj` is an instance of the
    `collections.abc.Sequence` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Sequence` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Sequence`, otherwise `False`.
    """
    return isinstance(obj, Sequence)
from collections.abc import MutableSequence
@docwrap
def is_mseq(obj):
    """Returns `True` if an object is a mutable sequence, otherwise `False`.

    `is_mseq(obj)` returns `True` if the given object `obj` is an instance of the
    `collections.abc.MutableSequence` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `MutableSequence` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `MutableSequence`, otherwise `False`.
    """
    return isinstance(obj, MutableSequence)
from collections.abc import ByteString
@docwrap
def is_bytestr(obj):
    """Returns `True` if an object is a byte-string, otherwise `False`.

    `is_bytestr(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.abc.ByteString` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `ByteString` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `ByteString`, otherwise `False`.
    """
    return isinstance(obj, ByteString)
from collections.abc import Set
@docwrap
def is_set(obj):
    """Returns `True` if an object is a set, otherwise `False`.

    `is_set(obj)` returns `True` if the given object `obj` is an instance of the
    `collections.abc.Set` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Set` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Set`, otherwise `False`.
    """
    return isinstance(obj, Set)
from collections.abc import MutableSet
@docwrap
def is_mset(obj):
    """Returns `True` if an object is a mutable set, otherwise `False`.

    `is_mset(obj)` returns `True` if the given object `obj` is an instance of the
    `collections.abc.MutableSet` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `MutableSet` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `MutableSet`, otherwise `False`.
    """
    return isinstance(obj, MutableSet)
from collections.abc import Mapping
@docwrap
def is_map(obj):
    """Returns `True` if an object is a mapping, otherwise `False`.

    `is_map(obj)` returns `True` if the given object `obj` is an instance of the
    `collections.abc.Mapping` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Mapping` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Mapping`, otherwise `False`.
    """
    return isinstance(obj, Mapping)
from collections.abc import MutableMapping
@docwrap
def is_mmap(obj):
    """Returns `True` if an object is a mutable mapping, otherwise `False`.

    `is_mmap(obj)` returns `True` if the given object `obj` is an instance of the
    `collections.abc.MutableMapping` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `MutableMapping` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `MutableMapping`, otherwise `False`.
    """
    return isinstance(obj, MutableMapping)
from collections.abc import Hashable
@docwrap
def is_hashable(obj):
    """Returns `True` if an object is a hashable object, otherwise `False`.

    `is_hashable(obj)` returns `True` if the given object `obj` is an instance
    of the `collections.abc.Hashable` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Hashable` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Hashable`, otherwise `False`.
    """
    return isinstance(obj, Hashable)

# Builtin Python Concrete Types ####################################################################
@docwrap
def is_list(obj):
    """Returns `True` if an object is a `list` object.

    `is_list(obj)` returns `True` if the given object `obj` is an instance of
    the `list` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `list` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `list`, otherwise `False`.
    """
    return isinstance(obj, list)
@docwrap
def is_tuple(obj):
    """Returns `True` if an object is a `tuple` object.

    `is_tuple(obj)` returns `True` if the given object `obj` is an instance of
    the `tuple` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `tuple` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `tuple`, otherwise `False`.
    """
    return isinstance(obj, tuple)
@docwrap
def is_pyset(obj):
    """Returns `True` if an object is a `set` object.

    `is_pyset(obj)` returns `True` if the given object `obj` is an instance of
    the `set` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `set` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `set`, otherwise `False`.
    """
    return isinstance(obj, set)
@docwrap
def is_frozenset(obj):
    """Returns `True` if an object is a `frozenset` object.

    `is_frozenset(obj)` returns `True` if the given object `obj` is an instance
    of the `frozenset` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `frozenset` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `frozenset`, otherwise `False`.
    """
    return isinstance(obj, frozenset)
@docwrap
def is_dict(obj):
    """Returns `True` if an object is a `dict` object.

    `is_dict(obj)` returns `True` if the given object `obj` is an instance of
    the `dict` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `dict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `dict`, otherwise `False`.

    """
    return isinstance(obj, dict)
from collections import OrderedDict
@docwrap
def is_odict(obj):
    """Returns `True` if an object is an `OrderedDict` object.

    `is_odict(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.OrderedDict` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `OrderedDict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `OrderedDict`, otherwise `False`.

    """
    return isinstance(obj, OrderedDict)
from collections import defaultdict
@docwrap
def is_ddict(obj):
    """Returns `True` if an object is an `defaultdict` object.

    `is_ddict(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.defaultdict` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `defaultdict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `defaultdict`, otherwise `False`.

    """
    return isinstance(obj, defaultdict)

# Persistent Types #################################################################################
from pyrsistent import PVector
@docwrap
def is_pvector(obj):
    """Returns `True` if an object is an `PVector` object.

    `is_pvector(obj)` returns `True` if the given object `obj` is an instance of
    the `pyrsistent.PVector` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `PVector` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `PVector`, otherwise `False`.
    """
    return isinstance(obj, PVector)
from pyrsistent import PList
@docwrap
def is_plist(obj):
    """Returns `True` if an object is an `PList` object.

    `is_plist(obj)` returns `True` if the given object `obj` is an instance of
    the `pyrsistent.PList` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `PList` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `PList`, otherwise `False`.
    """
    return isinstance(obj, PList)
from pyrsistent import PSet
@docwrap
def is_pset(obj):
    """Returns `True` if an object is an `PSet` object.

    `is_pset(obj)` returns `True` if the given object `obj` is an instance of
    the `pyrsistent.PSet` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `PSet` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `PSet`, otherwise `False`.
    """
    return isinstance(obj, PSet)
from pyrsistent import PMap
@docwrap
def is_pmap(obj):
    """Returns `True` if an object is an `PMap` object.

    `is_pmap(obj)` returns `True` if the given object `obj` is an instance of
    the `pyrsistent.PMap` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `PMap` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `PMap`, otherwise `False`.
    """
    return isinstance(obj, PMap)
