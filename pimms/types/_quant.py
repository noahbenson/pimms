# -*- coding: utf-8 -*-
####################################################################################################
# pimms/types/_core.py
# Core implementation of the utility classes for the various types that are managed by pimms.
# By Noah C. Benson

# Dependencies #####################################################################################
import inspect, types, sys, pint, os, numbers, warnings, base64
import collections as colls, numpy as np, pyrsistent as pyr
import scipy.sparse as sps
import docrep

from pint import DimensionalityError
from ..doc import docwrap
from ._core import (is_ureg, is_unit, is_quant, is_numpyarray, is_torchtensor)

# Quantity and UnitRegistry Classes ################################################################
# Because we want a type of quantity that plays nicely with pytorch tensors as well as numpy arrays,
# we overload the Quantity and UnitRegistry types from pint.
# In the Quantity overload class, we overload all of the mathematical operations so that they work
# with either numpy- or torch-based Quantities.

class Quantity(pint.Quantity):
    # Addition and subtraction are handled via these functions:
    def _iadd_sub(self, other, op):
        """Perform addition or subtraction operation in-place and return the result.

        The `pimms.Quantity` class makes this function compatible with NumPy arrays
        and with PyTorch tensors.

        Parameters
        ----------
        other : pint.Quantity or any type accepted by :func:`_to_magnitude`
            object to be added to / subtracted from self
        op : function
            operator function (e.g. operator.add, operator.isub)
        """
        # We need to promote the types.

    def __mul__(self, other):
        return pint.Quantity.__mul__(self, other)
def _build_quantity_class(reg, BaseClass):
    class Quantity(BaseClass):
        _REGISTRY = reg
    return Quantity
class UnitRegistry(pint.UnitRegistry):
    def _init_dynamic_classes(self) -> None:
        """Generate subclasses on the fly and attach them to self"""
        # We do what is in the BaseRegistry._init_dynamic_classes method first:
        from pint.unit import build_unit_class
        self.Unit = build_unit_class(self)
        from pint.measurement import build_measurement_class
        self.Measurement = build_measurement_class(self)
        # Then we do our own version for Quantity:
        self.Quantity: Type["Quantity"] = _build_quantity_class(self, Quantity)
        # Then we do the SystemRegistry._init_dynamic_classes method:
        self.Group = pint.systems.build_group_class(self)
        self.System = pint.systems.build_system_class(self)
units = UnitRegistry()
"""UnitRegistry: the registry for units tracked by pimms.

`pimms.units` is a global `pint`-module unit registry that can be used as a
single global place for tracking units. Pimms functions that interact with units
generally take an argument `ureg` that can be used to modify this registry.
Additionally, the default registry (this object, `pimms.units`) can be
temporarily changed in a local block using `with pimms.default_ureg(ureg): ...`.
"""
class default_ureg(object):
    """Context manager for setting the default `pimms` unit registry.

    The following code-block can be used to evaluate the code represented by
    `...` using the unit-registry `ureg` as the default `pimms.units` registry:

    ```python
    with pimms.default_ureg(ureg):
        ...
    ```
    """
    def __init__(self, ureg):
        if not is_ureg(ureg): raise TypeError("ureg must be a pint.UnitRegistry")
        object.__setattr__(self, 'original', None)
    def __enter__(self):
        global units
        object.__setattr__(self, 'original', units)
        units = ureg
        return ureg
    def __exit__(self, exc_type, exc_val, exc_tb):
        global units
        units = self.original
        return False
    def __setattr__(self, name, val):
        raise TypeError("cannot change the original units registry")
# We want to disable the awful pint warning fo numpy if it's present:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    units.Quantity([])
# Make sure there's a pixel unit
if not hasattr(units, 'pixels'):
    units.define('pixel = [image_length] = px')
def like_unit(q, ureg=None):
    """Returns `True` if `q` is or names a `pint` unit and `False` otherwise.

    `like_unit(q)` returns `True` if `q` is a `pint` unit or a string that names
    a `pint` unit and `False` otherwise.

    Parameters
    ----------
    q : object
        The object whose quality as a `pint` unit is to be assessed.
    ureg : UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use. If `None` (the default), then
        the `pimms.units` registry is used.

    Returns
    -------
    boolean
        `True` if `q` is a `pint` unit or a string naming a `pint` unit and
        `False` otherwise.
    """
    return isinstance(q, Unit) or (is_str(q) and hasattr(units, q))
def to_unit(obj):
    """Converts the argument into a a `Unit` object.

    `unit(u)` returns the `pimms`-library unit object for the given unit object
    `u` (which may be from a separate `pint.UnitRegistry` instance).

    `unit(uname)` returns the unit object for the given unit name string
    `uname`.

    `unit(None)` returns `pimms.units.dimensionless`.

    `unit(q) returns the unit of the given quantity object `q`.

    Parameters
    ----------
    obj : object
        The object that is to be converted to a unit.

    Returns
    -------
    pint.Unit
        The `Unit` object associated with the given argument.

    Raises
    ------
    TypeError
        When the argument cannot be converted to a `Unit` object.
    """
    if u is None:    return None
    elif is_unit(u): return getattr(units, str(u))
    elif is_quantity(u):
        if isinstance(u, tuple): return getattr(units, str(u[1]))
        else: return getattr(units, str(u.u))
    else:
        raise ValueError('unrecotnized unit argument')
def to_unit(u):
    '''
    unit(u) yields the pimms-library unit object for the given unit object u (which may be from a
      separate pint.UnitRegistry instance).
    unit(uname) yields the unit object for the given unit name uname.
    unit(None) yields None.
    unit(q) yields the unit of the given quantity q.
    '''
    if u is None:    return None
    elif is_unit(u): return getattr(units, str(u))
    elif is_quantity(u):
        if isinstance(u, tuple): return getattr(units, str(u[1]))
        else: return getattr(units, str(u.u))
    else:
        raise ValueError('unrecotnized unit argument')
def alike_units(a, b, ureg=None):
    """Returns `True` if the arguments are alike units, otherwise `False`.

    `alike_units(a,b)` returns `True` if `a` and `b` can be cast to each other
    in terms of units and `False` otherwise. Both `a` and `b` can either be
    units, unit names, or quantities with units. If either `a` or `b` is neither
    a unit nor a quantity, then it is considered equivalent to the dimensionless
    unit.

    Parameters
    ----------
    a : UnitLike
        A unit object or the name of a unit or a quantity.
    b : UnitLike
        A unit object or the name of a unit or a quantity.
    ureg : UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use. If `None` (the default), then
        the `pimms.units` registry is used.

    Returns
    -------
    boolean
        `True` if the units `a` and `b` are alike and `False` otherwise.
    """
    if ureg is None: ureg = units
    return ureg.is_compatible_with(a, b)
def like_quant(obj, unit=Ellipsis, ureg=Ellipsis):
    """Returns `True` for objects that can be converted into `pint` quantities.

    `like_quant(q)` returns `True` if `q` is a `pint` quantity or an object that
    can be converted into a `pint` quantity and `False` otherwise. The optional
    parameter `unit` may additionally specify a unit that `obj` must be
    compatible with.

    The following kinds of objects can be converted into quantities:
     * `Quantity` objects are already quantities;
     * any number, set, array, or tensor can be represented as a dimensionless
       quantity; or
     * a 2-tuple `(mag, unit)` where `mag` is a numeric magnitude value and
       `unit` is a string representing a valid unit.

    Parameters
    ----------
    obj : object
        The object whose quality as a quantity is to be assessed.
    unit : UnitLike or True or False or None or Ellipsis, optional
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
    ureg : pint.UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use for units. If `None`, then
        `pimms.units` is used. If the value `Ellipsis` is given (the default),
        then `Quantity` objects from any unit registry are considered valid.

    Returns
    -------
    boolean
        `True` if `obj` is a `pint` quantity or quantity-like object whose units
        are compatible with the requested `unit` and `False` otherwise.
    """
    if ureg is Ellipsis:
        ureg = units
        # In this case, we allow any quantity type.
        if isinstance(obj, pint.Quantity): return True
    else:
        if ureg is None: ureg = units
        if isinstance(obj, ureg.Quantity): return True
    if (is_tuple(obj)      and
        len(obj) == 2      and
        is_numeric(obj[0]) and
        is_str(obj[1])     and
        hasattr(ureg, obj[1])
    ): return True
    if is_numeric(obj): return True
    if is_set(obj): return True
    return False
def mag(val, u=Ellipsis):
    '''
    mag(scalar) yields scalar for the given scalar (numeric value without a unit).
    mag(quant) yields the scalar magnitude of the quantity quant.
    mag(scalar, u=u) yields scalar.
    mag(quant, u=u) yields the given magnitude of the given quant after being translated into the
      given unit u.

    The quant arguments in the above list may be replaced with (scalar, unit) as in (10, 'degrees')
    or (14, 'mm'). Note that mag always translates all lists and tuples into numpy ndarrays.
    '''
    if not is_quantity(val): return val
    if isinstance(val, tuple):
        val = units.Quantity(val[0], val[1])
    return val.m if u is Ellipsis else val.to(unit(u)).m
