# -*- coding: utf-8 -*-
################################################################################
# pimms/types/_core.py
#
# Core implementation of the utility classes for the various types that are
# managed by pimms.
#
# By Noah C. Benson

# Dependencies #################################################################
import pint, warnings
import numpy as np
import scipy.sparse as sps
import docrep

from pint   import DimensionalityError
from ..doc  import docwrap
from ._core import (alttorch, checktorch, scipy__is_sparse,
                    torch__is_tensor, torch,
                    is_set, is_str, is_ureg, is_unit, is_quant,
                    is_array, is_tensor, is_numeric, is_sparse, to_sparse,
                    to_array, to_tensor, to_numeric, to_sparse, to_dense)

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
        if not is_ureg(ureg):
            raise TypeError("ureg must be a pint.UnitRegistry")
        object.__setattr__(self, 'original', None)
    def __enter__(self):
        import pimms
        object.__setattr__(self, 'original', pimms.units)
        pimms.units = ureg
        return ureg
    def __exit__(self, exc_type, exc_val, exc_tb):
        import pimms
        pimms.units = self.original
        return False
    def __setattr__(self, name, val):
        raise TypeError("cannot change the original units registry")
_initial_global_ureg = pint.UnitRegistry()
# We want to disable the awful pint warning for numpy if it's present:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _initial_global_ureg.Quantity([])
# Make sure there's a pixel unit
if not hasattr(_initial_global_ureg, 'pixels'):
    _initial_global_ureg.define('pixel = [image_length] = px')
@docwrap
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
    if isinstance(q, Unit): return True
    if not is_str(q): return False
    if ureg is None or ureg is Ellipsis: from pimms import units as ureg
    return hasattr(ureg, q)
@docwrap
def unit(obj, ureg=Ellipsis):
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
    ureg : UnitRegistry or None or Ellipsis, optional
        The unit registry to convert the object into. If `Ellipsis` (the
        default), then `pimms.units` is used. If `None`, then the unit registry
        for `obj` is used if `obj` is a quantity or unit already, and
        `pimms.units` is used if not. Otherwise, must be a unit registry.

    Returns
    -------
    pint.Unit
        The `Unit` object associated with the given argument.

    Raises
    ------
    TypeError
        When the argument cannot be converted to a `Unit` object.
    """
    if ureg is Ellipsis: from pimms import units as ureg
    if obj is None: return ureg.dimensionless
    if is_quant(obj): u = u.u
    if is_unit(obj):
        if ureg is None or ureg is obj._REGISTRY:
            return obj
        else:
            return getattr(ureg, str(obj))
    elif is_str(obj):
        if ureg is None: from pimms import units as ureg
        return getattr(ureg, obj)
    else:
        raise ValueError(f'unrecognized unit argument: {obj}')
@docwrap
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
    if ureg is None: from pimms import units as ureg
    return ureg.is_compatible_with(a, b)
@docwrap
def quant(mag, units=Ellipsis, ureg=None):
    """Returns a `Quantity` object with the given magnitude and units.

    `quant(mag, units)` returns a `Quantity` object with the given magnitude and
    units. If `mag` is alreaady a `Quantity`, then it is converted into the
    given units and returned (a copy of `mag` is made only if necessary); if the
    units of `mag` in this case are not compatible with `units`, then an error
    is raised. If `mag` is not a quantity, then the given `units` are used to
    create the quantity.

    The value `units=None` is equivalent to `units='dimensionless'`. 

    `quant(mag)` is equivalent to `quant(mag, Ellipsis)`. Both return `mag` if
    `mag` is already a `Quantity`; otherwise they return a quantity with
    dimensionless units.

    Parameters
    ----------
    mag : object
        The magnitude to be given a unit.
    units : unit-like or None or Ellipsis, optional
        The units to use in the returned quantity. If `Ellipsis` (the default),
        then dimensionless units are assumed unless the `mag` argument already
        is a quantity with its own units.
    ureg : UnitRegistry or None or Ellipsis, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `pimms.units` is used. If `ureg` is `None` (the
        default), then no specific coersion to a `UnitRegistry` is performed.

    Returns
    -------
    pint.Quantity
        A quantity object representing the given magnitude and units.
    """
    if ureg is Ellipsis: from pimms import units as ureg
    if is_quant(mag):
        if ureg is None: ureg = mag._REGISTRY
        q = mag if units is Ellipsis else mag.to(units)
    else:
        if units is Ellipsis: units = None
        if ureg is None: from pimms import units as ureg
        q = ureg.Quantity(mag, units)
    if q._REGISTRY is not ureg:
        return ureg.Quantity(q.m, q.u)
    else:
        return q
@docwrap
def to_quants(*args, units=Ellipsis, ureg=None):
    """Converts each argument in a list to a quantity.

    `to_quants(*lst, **kw)` returns `[quant(el, **kw) for el in lst]`.

    Parameters
    ----------
    *args
        The list of arguments to be converted into quantities.
    %(pimms.types._quantity.quant.parameters.units)s
    %(pimms.types._quantity.quant.parameters.ureg)s

    Returns
    -------
    list of Quantity objects
        A list of the arguments, each converted into quantities.
    """
    return [quant(el, units=units, ureg=ureg) for el in args]
@docwrap
def mag(val, units=Ellipsis):
    """Returns the magnitude of the given object.

    `mag(quantity)` returns the magnitude of the given quantity, regardless of
    the quantity's units.

    `mag(obj)`, for a non-quantity object `obj`, simply returns `obj`.

    `mag(arg, units)` returns `arg.to(units)` of `arg` is a quantity and returns
    `arg` if `arg` is not a quantity.

    `mag(arg, Ellipsis)` is equivalent to `mag(arg)`.

    If `mag(quantity, units)` is given a quantity not compatible with the given
    units, then an error is raised.

    Parameters
    ----------
    val : object
        The object that is to be converted into a magnitude.
    units : unit-like or None or Ellipsis, optional
        The units in which the magnitude of the argument `val` should be
        returned. The default argument of `Ellipsis` indicates that the value's
        native units, if any, should be used.

    Returns
    -------
    object
        The magnitude of `val` in the requested units, if `val` is a quantity,
        or `val` itself, if it is not a quantity.

    Raises
    ------
    DimensionaltyError
        If the given `val` is a quantity whose units are not compatible with the
        `units` parameter.
    """
    if is_quant(val): 
        return val.m if units is Ellipsis else val.m_as(units)
    else:
        return val


# Promotion ####################################################################
def _array_promote(*args, ureg=None):
    return [to_array(el, quant=True, ureg=ureg) for el in args]
@alttorch(_array_promote)
@docwrap
def promote(*args, ureg=None):
    """Promotes all arguments into quantities with compatible magnitudes.

    `promote(a, b, c...)` converts all of the passed arguments into numerical
    quantity objects and returns them as a tuple. The returned arguments will
    all have compatible (promoted) types or magnitude types.

    Promotion is determined based on the object type. If any of the objects are
    PyTorch tensors or quantities with tensor magnitudes, then all of the
    returned quantities will have for their magnitudes PyTorch tensors with the
    same profile (e.g, device) as the tensor argument(s). Otherwise, the
    returned quantities will converted into array types. The purpose of this
    promotion is to ensure that all arguments can be combined into an expression
    (PyTorch tensor operations generally require that all arguments are
    tensors).

    Parameters
    ----------
    *args
        The arguments that are to be promoted.
    ureg : pint.UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `pimms.units` is used. If `ureg` is `None` (the
        default), then no specific coersion to a `UnitRegistry` is performed.

    Returns
    -------
    list of numeric objects
        A list of the arguments after each has been promoted.
    """
    if ureg is None: from pimms import units as ureg
    # We can start by converting the args into a list of quantities.
    qs = to_quants(args, ureg=ureg)
    # Basic question: are any of them tensors?
    first_tensor = next((q for q in qs if torch.is_tensor(q.m)), None)
    # If there isn't any, qs is fine as-is.
    if first_tensor is None: return qs
    device = first_tensor.device
    # Otherwise, we need to turn them all into tensors like this one.
    for (ii,q) in enumerate(qs):
        if q is first_tensor: continue
        mag = q.m
        mag = to_tensor(mag, device=device)
        if mag is not q.m:
            if q is args[ii]:
                qs[ii] = q.__class__(mag, q.u)
            else:
                q.m = mag
    # That's all that is needed.
    return qs
