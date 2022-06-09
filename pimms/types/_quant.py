# -*- coding: utf-8 -*-
################################################################################
# pimms/types/_core.py
#
# Core implementation of the utility classes for the various types that are
# managed by pimms.
#
# By Noah C. Benson
#
# Notes
# -----
#
# * A lot of the methods of the Quantity class (which overloads the pint.Quantity class) have been
#   overloaded for the sole purpose of extending their particular functionality to the PyTorch
#   tensor and SciPy sparse matrix types.
# * In many cases, these methods only need to overloaded to handle a single case in which the normal
#   operation doesn't work. For example, __abs__ must be overloaded because, while abs(np_array),
#   abs(sp_sparse_matrix) and abs(dense_tensor) all work fine, abs(sparse_tensor) currently raises
#   an exception. In this case, the default pint.Quantity.__abs__ method would be just fine except
#   for the sparse tensor type, However, PyTorch will likely implement the abs() method for sparse
#   tensors soon, and when that happens, we should just replace delete the overloaded method.
# * Deficiencies in particular libraries, as described in the above bullet, should be tagged with
#   one or more of the following hashtags in a comment: #numpy-fix, #scipy-fix, #torch-fix,
#   #torch-sparse-fix, #torch-dense-fix.

# Dependencies #####################################################################################
import inspect, types, sys, pint, os, numbers, warnings, base64, operator
import collections as colls, numpy as np, pyrsistent as pyr
import scipy.sparse as sps
import docrep

from pint   import DimensionalityError
from ..doc  import docwrap
from ._core import (alttorch, checktorch, scipy__is_sparse,
                    torch__is_tensor, torch,
                    is_set, is_str, is_ureg, is_unit, is_quant,
                    is_array, is_tensor, is_numeric, is_sparse, to_sparse,
                    to_array, to_tensor, to_numeric, to_sparse, to_dense)

# Quantity and UnitRegistry Classes ################################################################
# Because we want a type of quantity that plays nicely with pytorch tensors as well as numpy arrays,
# we overload the Quantity and UnitRegistry types from pint.
# In the Quantity overload class, we overload all of the mathematical operations so that they work
# with either numpy- or torch-based Quantities.
class Quantity(pint.Quantity):
    @property
    def T(self):
        mag = self._magnitude
        # #torch-sparse-fix #here #TODO
        return self.__class__(self._magnitude.T, self._units)
    def __abs__(self):
        mag = self._magnitude
        # #torch-sparse-fix
        # We need to handle the sparse-tensor case specially. Since abs() doesn't change zeros,
        # however, it's not terribly hard.
        if not torch__is_tensor(mag) or not mag.is_sparse:
            return pint.Quantity.__abs__(self)
        new_mag = mag.clone().detach().coalesce()
        vals = new_mag.values()
        torch.abs(vals, out=vals)
        return new_mag
        
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
        # We may need to promote the types or change the operation somewhat depending on whether the
        # arguments are tensors, scipy sparse matrices, or not.
        self_istensor = is_tensor(self)
        other_istensor = is_tensor(other)
        if self_istensor:
            if not other_istensor:
                (self, other) = promote(self, other)
        # Having put the arguments in appropriate formats, the operation can proceed.
        return pint.Quantity._iadd_sub(self, other, op)
    def _add_sub(self, other, op):
        """Perform addition or subtraction operation and return the result.

        The `pimms.Quantity` class makes this function compatible with NumPy arrays
        and with PyTorch tensors.

        Parameters
        ----------
        other : pint.Quantity or any type accepted by :func:`_to_magnitude`
            object to be added to / subtracted from self
        op : function
            operator function (e.g. operator.add, operator.isub)
        """
        # We may need to promote the types or change the operation somewhat depending on whether the
        # arguments are tensors, scipy sparse matrices, or not.
        self_istensor = is_tensor(self)
        other_istensor = is_tensor(other)
        if self_istensor:
            if not other_istensor:
                (self, other) = promote(self, other)
        # Having put the arguments in appropriate formats, the operation can proceed.
        res = pint.Quantity._add_sub(self, other, op)
        # If res is a numpy.matrix, then we fix that into an array.
        if is_quant(res):
            if isinstance(res.m, np.matrix):
                res = res.__class__(np.asarray(res.m), res.u)
        elif isinstance(res.m, np.matrix):
            res.m = np.asarray(res.m)
        # Return this corrected object.
        return res
    # Addition and subtraction are handled via these functions:
    def _imul_div(self, other, mag_op, unit_op=None):
        """Perform multiplication or division operations in-place and return
        the result.

        The `pimms.Quantity` class makes this function compatible with NumPy
        arrays and with PyTorch tensors.

        Parameters
        ----------
        other : pint.Quantity or any type accepted by :func:`_to_magnitude`
            Object that `self` should be multiplied / divided by.
        mag_op : function
            Operator function to perform on the magnitudes (e.g.,
            `operator.mul`).
        unit_op : function or None
            Operator function to perform on the units; if `None`, then the
            `mag_op` value is used.
        """
        if   mag_op is operator.mul: mag_op = operator.imul
        elif mag_op is operator.truediv: mag_op = operator.itruediv
        # We may need to promote the types or change the operation somewhat depending on whether the
        # arguments are tensors, scipy sparse matrices, or not.
        self_istensor = is_tensor(self)
        other_istensor = is_tensor(other)
        if self_istensor:
            if not other_istensor:
                (self, other) = promote(self, other)
        elif is_sparse(self):
            raise TypeError("scipy sparse matrices do not support in-place"
                            " multiplication or division")
        elif is_sparse(other) and mag_op is operator.itruediv:
            other = 1 / to_dense(other)
            return pint.Quantity._imul_div(self. other, operator.imul)
        # We can just perform the normal _imul_div once the above filters have been applied.
        return pint.Quantity._imul_div(self, other, mag_op, unit_op)
    def _mul_div(self, other, mag_op, unit_op=None):
        """Perform multiplication or division operations and return the result.

        The `pimms.Quantity` class makes this function compatible with NumPy
        arrays and with PyTorch tensors.

        Parameters
        ----------
        other : pint.Quantity or any type accepted by :func:`_to_magnitude`
            Object that `self` should be multiplied / divided by.
        mag_op : function
            Operator function to perform on the magnitudes (e.g.,
            `operator.mul`).
        unit_op : function or None
            Operator function to perform on the units; if `None`, then the
            `mag_op` value is used.
        """
        if   mag_op is operator.imul: mag_op = operator.mul
        elif mag_op is operator.itruediv: mag_op = operator.truediv
        # We may need to promote the types or change the operation somewhat depending on whether the
        # arguments are tensors, scipy sparse matrices, or not.
        self_istensor = is_tensor(self)
        other_istensor = is_tensor(other)
        if self_istensor or other_istensor:
            if not (self_istensor and other_istensor):
                (self, other) = promote(self, other)
            return pint.Quantity._mul_div(self, other, op)
        elif is_sparse(other) and mag_op is operator.truediv:
            # If we are dividing by a sparse matrix, so we need to make the matrix dense then recur.
            other = 1 / to_dense(other)
            return self._mul_div(other, operator.mul)
        elif is_sparse(self):
            # We run this multiplication here because it requires the .multiply() method instead of
            # the basic * / mul operator, which is performed by pint.
            if is_quant(other):
                omag = other.m
                ounit = other.u
            else:
                omag = other
                ounit = self._REGISTRY.dimensionless
            # Perform the operation.
            if mag_op is operator.mul:
                resmag = self.m.multiply(omag)
            elif mag_op is operator.truediv:
                resmag = self.m.multiply(1 / omag)
            else:
                raise TypeError("op must be operator.mul or operator.div")
            resunit = mag_op(self.u, ounit)
            # If the magnitude is a numpy matrix, fix that.
            if isinstance(resmag, np.matrix): resmag = np.asarray(resmag)
            # Return the new quantity.
            return self._REGISTRY.Quantity(resmag, self.u)
        else:
            return pint.Quantity._mul_div(self, other, op)
    def __matmul__(self, other):
        # We need to handle tensors, but if we have scipy sparse matrices and arrays mixed together,
        # that is ultimately just fine.
        cls = self.__class__
        (self, other) = promote(self, other)
        units = self.u * other.u
        if is_tensor(self):
            if is_sparse(other):
                raise TypeError("PyTorch does not currently support tensor multiplication with "
                                " sparse tensors on the right of the multiplication symbol")
            mag = torch.matmul(self.m, other.m)
        else:
            mag = self.m @ other.m
            if isinstance(mag, np.matrix): mag = np.asarray(mag)
        return cls(mag, units)
    def __rmatmul__(self, other):
        cls = self.__class__
        (self, other) = promote(self, other)
        units = self.u * other.u
        if is_tensor(self):
            if is_sparse(self):
                raise TypeError("PyTorch does not currently support tensor multiplication with "
                                " sparse tensors on the right of the multiplication symbol")
            mag = torch.matmul(other.m, self.m)
        else:
            mag = other.m @ self.m
            if isinstance(mag, np.matrix): mag = np.asarray(mag)
        return cls(mag, units)

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
_initial_global_ureg = UnitRegistry()
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
    ureg : UnitRegistry or None or Ellipsis
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
    %(pimms.types._quant.quant.units)s
    %(pimms.types._quant.quant.ureg)s

    Returns
    -------
    list of Quantity objects
        A list of the arguments, each converted into quantities.
    """
    return [quant(el, units=units, ureg=ureg) for el in args]
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

# Array-type Promotion #############################################################################
# Promotion ########################################################################################
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
