# -*- coding: utf-8 -*-
####################################################################################################
# pimms/types/_core.py
# Core implementation of the utility classes for the various types that are managed by pimms.
# By Noah C. Benson

# Dependencies #####################################################################################
import pint, os
import numpy as np
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
    ureg : pint.UnitRegistry or Ellipsis or None, optional
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

# Numerical Collection Suport ######################################################################
# Numerical collections include numpy arrays and torch tensors. These objects are handled similarly
# due to their overall functional similarity, and certain support functions are used for both.
def _numcoll_match(numcoll_shape, numcol_dtype, ndim, shape, dtype):
    """Checks that the actual numcoll shape and the actual numcol dtype match
    the requirements of the ndim, shape, and dtype parameters.
    """
    # Parse the shape int front and back requirements and whether middle values are allowed.
    shape_sh = np.shape(shape)
    if shape is None:
        (sh_pre, sh_mid, sh_suf) = ((), True, ())
    elif shape == ():
        (sh_pre, sh_mid, sh_suf) = ((), False, ())
    elif shape_sh == ():
        (sh_pre, sh_mid, sh_suf) = ((shape,), False, ())
    else:
        # We add things to the prefix until we get to an ellipsis...
        sh_pre = []
        for d in shape:
            if d is Ellipsis: break
            sh_pre.append(d)
        sh_pre = tuple(sh_pre)
        # We might have finished with just that; otherwise, note the ellipsis and move on.
        if len(sh_pre) == len(shape):
            (sh_mid, sh_suf) = (False, ())
        else:
            sh_suf = []
            for d in reversed(shape):
                if d is Ellipsis: break
                sh_suf.append(d)
            sh_suf = tuple(sh_suf) # We leave this reversed!
            sh_mid = len(sh_suf) + len(sh_pre) < len(shape)
        assert len(sh_suf) + len(sh_pre) + int(sh_mid) == len(shape), \
            "only one Ellipsis may be used in the shape filter"
    # Parse ndim.
    if not (is_tuple(ndim) or is_set(ndim) or is_list(ndim) or ndim is None):
        ndim = (ndim,)
    # See if we match in terms of ndim and shape
    sh = numcoll_shape
    if ndim is not None and len(sh) not in ndim:
        return False
    ndim = len(sh)
    if ndim < len(sh_pre) + len(sh_suf):
        return False
    (npre, nsuf) = (0,0)
    for (s,p) in zip(sh, sh_pre):
        if p != -1 and p != s: return False
        npre += 1
    for (s,p) in zip(reversed(sh), sh_suf):
        if p != -1 and p != s: return False
        nsuf += 1
    # If there are extras in the middle and we don't allow them, we fail the match.
    if not sh_mid and nsuf + npre != ndim: return False
    # See if we match the dtype.
    if dtype is not None:
        if is_numpydtype(numcoll_dtype):
            if is_seq(dtype) or is_set(dtype): dtype = [to_numpydtype(dt) for dt in dtype]
            else: dtype = [to_numpydtype(dtype)]
        elif is_torchdtype(numcoll_dtype):
            if is_seq(dtype) or is_set(dtype): dtype = [to_torchdtype(dt) for dt in dtype]
            else: dtype = [to_torchdtype(dtype)]
        if numcoll_dtype not in dtype: return False
    # We match everything!
    return True

# Numpy Arrays #####################################################################################
# For testing whether numpy arrays or pytorch tensors have the appropriate dimensionality, shape,
# and dtype, we use some helper functions.
from numpy import dtype as numpy_dtype
@docwrap
def is_numpydtype(dt):
    """Returns `True` for a NumPy dtype object and `False` otherwise.

    `is_numpydtype(obj)` returns `True` if the given object `obj` is an instance
    of the `numpy.dtype` class.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy `dtype` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is a valid `numpy.dtype`, otherwise `False`.
    """
    return isinstance(dt, numpy_dtype)
@docwrap
def like_numpydtype(dt):
    """Returns `True` for any object that can be converted into a numpy `dtype`.

    `like_numpydtype(obj)` returns `True` if the given object `obj` is an
    instance of the `numpy.dtype` class, is a string that can be used to
    construct a `numpy.dtype` object, or is a `torch.dtype` object.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy `dtype` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` can be converted into a valid numpy `dtype`, otherwise
        `False`.
    """
    if   is_numpydtype(dt): return True
    elif is_torchdtype(dt): return True
    else:
        try: return is_numpydtype(np.dtype(dt))
        except TypeError: return False
@docwrap
def to_numpydtype(dt):
    """Returns a `numpy.dtype` object equivalent to the given argument `dt`.

    `to_numpydtype(obj)` attempts to coerce the given `obj` into a `numpy.dtype`
    object. If `obj` is already a `numpy.dtype` object, then `obj` itself is
    returned. If the object cannot be converted into a `numpy.dtype` object,
    then an error is raised.

    The following kinds of objects can be converted into a `numpy.dtype` (see
    also `like_numpydtype()`):
     * `numpy.dtype` objects;
     * `torch.dtype` objects;
     * `None` (the default `numpy.dtype`);
     * strings that name `numpy.dtype` objects; or
     * any object that can be passed to `numpy.dtype()`, such as `numpy.int32`.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy `dtype` object is to be assessed.

    Returns
    -------
    numpy.dtype
        The `numpy.dtype` object that is equivalent to the argument `dt`.

    Raises
    ------
    TypeError
        If the given argument `dt` cannot be converted into a `numpy.dtype`
        object.
    """
    if   is_numpydtype(dt): return dt
    elif is_torchdtype(dt): return torch.as_tensor((), dtype=dt).numpy().dtype
    else: return np.dtype(dt)
from numpy import ndarray
@docwrap
def is_array(obj,
             dtype=None, shape=None, ndim=None, readonly=None,
             sparse=None, quant=None, units=Ellipsis, ureg=None):
    """Returns `True` if an object is a `numpy.ndarray` object, else `False`.

    `is_array(obj)` returns `True` if the given object `obj` is an instance of
    the `numpy.ndarray` class or is a `scipy.sparse` matrix or if `obj` is a
    `pint.Quantity` object whose magnitude is one of these. Additional
    constraints may be placed on the object via the optional argments.

    Note that to `pimms`, both `numpy.ndarray` arrays and `scipy.sparse` matrices are
    considered "arrays". This behavior can be changed with the `sparse` parameter.

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
        dtype is any of the dtypes in `dtype`.
    ndim : int or tuple or ints or None, optional
        The number of dimensions that the object must have in order to be
        considered a valid numpy array. If `None`, then any number of dimensions
        is acceptable (this is the default). If this is an integer, then the
        number of dimensions must be exactly that integer. If this is a list or
        tuple of integers, then the dimensionality must be one of these numbers.
    shape : int or tuple of ints or None, optional
        If the `shape` parameter is not `None`, then the given `obj` must have a
        shape shape that matches the parameter value `sh`. The value `sh` must
        be a tuple that is equal to the `obj`'s shape tuple with the following
        additional rules: a `-1` value in the `sh` tuple will match any value in
        the `obj`'s shape tuple, and a single `Ellipsis` may appear in `sh`,
        which matches any number of values in the `obj`'s shape tuple. The
        default value of `None` indicates that no restriction should be applied
        to the `obj`'s shape.
    readonly : boolean or None, optional
        If `None`, then no restrictions are placed on the `'WRITEABLE'` flag of
        `obj`. If `True`, then the data in `obj` must be read-only in order for
        `obj` to be considered a valid array. If `False`, then the data in `obj`
        must not be read-only.
    sparse : boolean or False, optional
        If the `sparse` parameter is `None`, then no requirements are placed on
        the sparsity of `obj` for it to be considered a valid array. If `sparse`
        is `True` or `False`, then `obj` must either be sparse or not be sparse,
        respectively, for `obj` to be considered valid. If `sparse` is a string,
        then it must be either `'coo'`, `'lil'`, `'csr'`, or `'csr'`, indicating
        the required sparse array type. Only `scipy.sparse` matrices are
        considered valid sparse arrays.
    quant : boolean, optional
        Whether `Quantity` objects should be considered valid arrays or not.  If
        `quant=True` then `obj` is considered a valid array only when `obj` is a
        quantity object with a `numpy` array as the magnitude. If `False`, then
        `obj` must be a `numpy` array itself and not a `Quantity` to be
        considered valid. If `None` (the default), then either quantities or
        `numpy` arrays are considered valid arrays.
    units : unit-like or Ellipsis, optional
        A unit with which the object obj's units must be compatible in order
        for `obj` to be considered a valid array. An `obj` that is not a
        quantity is considered to have dimensionless units. If `units=Ellipsis`
        (the default), then the object's units are ignored.
    ureg : UnitRegistry or None or Ellipsis, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `pimms.units` is used. If `ureg` is `None` (the
        default), then the registry of `obj` is used if `obj` is a quantity, and
        `pimms.units` is used if not.

    Returns
    -------
    boolean
        `True` if `obj` is a valid numpy array, otherwise `False`.
    """
    if ureg is Ellipsis: from pimms import units as ureg
    # If this is a quantity, just extract the magnitude.
    if is_quant(obj):
        if quant is False: return False
        if ureg is None: ureg = obj._REGISTRY
        u = obj.u
        obj = obj.m
    else:
        if quant is True: return False
        if ureg is None: from pimms import units as ureg
        u = None
    # At this point we want to check if this is a valid numpy array or scipy sparse matrix; however
    # how we handle the answer to this question depends on the sparse parameter.
    if sparse is True:
        if not scipy_issparse(obj): return False
    elif sparse is False:
        if not isinstance(obj, ndarray): return False
    elif sparse is None:
        if not (isinstance(obj, ndarray) or scipy_issparse(obj)): return False
    elif is_str(sparse):
        sparse = strnorm(sparse.strip(), case=True, unicode=False)
        if sparse == 'bsr':
            if not isinstance(obj, sps.bsr_matrix): return False
        elif sparse == 'coo':
            if not isinstance(obj, sps.coo_matrix): return False
        elif sparse == 'csc':
            if not isinstance(obj, sps.csc_matrix): return False
        elif sparse == 'csr':
            if not isinstance(obj, sps.csr_matrix): return False
        elif sparse == 'dia':
            if not isinstance(obj, sps.dia_matrix): return False
        elif sparse == 'dok':
            if not isinstance(obj, sps.dok_matrix): return False
        elif sparse == 'lil':
            if not isinstance(obj, sps.lil_matrix): return False
        else:
            raise ValueErroor(f"invalid sparse matrix type: {sparse}")
    else:
        raise ValueErroor(f"invalid sparse parameter: {sparse}")
    # Check that the object is read-only
    if readonly is not None:
        if readonly is True:
            if scipy_issparse(obj) or obj.flags['WRITEABLE']: return False
        elif readonly is False:
            if not scipy_issparse(obj) and not obj.flags['WRITEABLE']: return False
        else:
            raise ValueError(f"invalid value for parameter readonly: {readonly}")
    # Next, check compatibility of the units.
    if units is not Ellipsis and not ureg.is_compatible_with(u, units): return False
    # Check the match to the numeric collection last.
    if dtype is None and shape is None and ndim is None: return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, dtype)
def to_array(obj,
             dtype=None, order=None, copy=False, sparse=None, readonly=None,
             quant=None, ureg=None, units=Ellipsis):
    """Reinterprets `obj` as a NumPy array or quantity with an array magnitude.

    `pimms.to_array` is roughly equivalent to the `numpy.asarray` function with
    a few exceptions:
      * `to_array(obj)` allows quantities for `obj` and, in such a case, will
        return a quantity whose magnitude has been reinterpreted as an array,
        though this behavior can be altered with the `quant` parameter;
      * `to_array(obj)` can extract the `numpy` array from `torch` tensor
         objects.

    Parameters
    ----------
    obj : object
        The object that is to be reinterpreted as, or if necessary covnerted to,
        a NumPy array object.
    dtype : data-type, optional
        The dtype that is passed to `numpy.asarray()`.
    order : {'C', 'F'}, optional
        The array order that is passed to `numpy.asarray()`.
    copy : boolean, optional
        Whether to copy the data in `obj` or not. If `False`, then `obj` is only
        copied if doing so is required by the optional parameters. If `True`,
        then `obj` is always copied if possible.
    sparse : boolean or {'csr','coo'} or None, optional
        If `None`, then the sparsity of `obj` is the same as the sparsity of the
        array that is returned. Otherwise, the return value will always be
        either a `scipy.spase` matrix (`sparse=True`) or a `numpy.ndarray`
        (`sparse=False`) based on the given value of `sparse`. The `sparse`
        parameter may also be set to `'bsr'`, `'coo'`, `'csc'`, `'csr'`,
        `'dia'`, or `'dok'` to return specific sparse matrix types.
    readonly : boolean or None, optional
        Whether the return value should be read-only or not. If `None`, then no
        changes are made to the return value; if a new array is allocated in the
        `to_array()` function call, then it is returned in a writeable form. If
        `True`, then the return value is always a read-only array; if `obj` is
        not already read-only, then a copy of `obj` is always returned in this
        case. If `False`, then the return-value is never read-only. Note that
        `scipy.sparse` arrays do not support read-only mode, and thus an
        `ValueError` is raised if a sparse matrix is requested in read-only
        format.
    quant : boolean or None, optional
        Whether the return value should be a `Quantity` object wrapping the
        array (`quant=True`) or the array itself (`quant=False`). If `quant` is
        `None` (the default) then the return value is a quantity if `obj` is a
        quantity and is not a quantity if `obj` is not a quantity.
    ureg : pint.UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `pimms.units` is used. If `ureg` is `None` (the
        default), then no specific coersion to a `UnitRegistry` is performed
        (i.e., the same quantity class is returned).
    units : unit-like or boolean or Ellipsis, optional
        The units that should be used in the return value. When the return value
        of this function is a `Quantity` (see the `quant` parameter), the
        returned quantity always has units matching the `units` parameter; if
        the provided `obj` is not a quantity, then its units are presumed to be
        those requested by `units`. When the return value of this function is
        not a `Quantity` object and is instead a NumPy array object, then when
        `obj` is not a quantity the `units` parameter is ignored, and when `obj`
        is a quantity, its magnitude is returned after conversion into
        `units`. The default value of `units`, `Ellipsis`, indicates that, if
        `obj` is a quantity, its units should be used, and `units` should be
        considered dimensionless otherwise.

    Returns
    -------
    NumPy array or Quantity
        Either a NumPy array equivalent to `obj` or a `Quantity` whose magnitude
        is a NumPy array equivalent to `obj`.

    Raises
    ------
    ValueError
        If invalid parameter values are given or if the parameters conflict.
    """
    if ureg is Ellipsis: ureg = pimms.units
    # If obj is a quantity, we handle things differently.
    if is_quant(obj):
        q = obj
        obj = q.m
        if ureg is None: ureg = q._REGISTRY
        if quant is None: quant = True
    else:
        q = None
        if ureg is None: ureg = pimms.units
        if quant is None: quant = False
    # Translate obj depending on whether it's a pytorch array / scipy sparse matrix.
    # We need to think about whether the output array is being requested in sparse
    # format. If so, we handle the conversion differently.
    obj_is_spsparse = scipy_issparse(obj)
    obj_is_tensor = not obj_is_spsparse and torch is not None and torch.is_tensor(obj)
    obj_is_sparse = obj_is_spsparse or (obj_is_tensor and obj.is_sparse)
    newarr = False # True means we own the memory of arr; False means we don't.
    if sparse is not False and (sparse is not None or obj_is_sparse):
        if sparse is None or sparse is True:
            sparse = (('csr' if obj.layout == torch.sparse_csr else 'coo') if obj_is_tensor else
                      type(obj).__name__[:3]                               if obj_is_sparse else
                      'coo')
        sparse = strnorm(sparse.strip(), case=True, unicode=False)
        mtype = (sps.bsr_matrix if sparse == 'bsr' else
                 sps.coo_matrix if sparse == 'coo' else
                 sps.csc_matrix if sparse == 'csc' else
                 sps.csr_matrix if sparse == 'csr' else
                 sps.dia_matrix if sparse == 'dia' else
                 sps.dok_matrix if sparse == 'dok' else
                 sps.lil_matrix if sparse == 'lil' else
                 None)
        if mtype is None:
            raise ValueError(f"unrecognized scipy sparse matrix name: {sparse}")
        if obj_is_sparse:
            # We're creating a scipy sparse output from a sparse output.
            if obj_is_tensor:
                # We're creating a scipy sparse output from a sparse tensor.
                arr = obj.coalesce()
                if obj is not arr: newarr = True
                ii = arr.indices().numpy().detach()
                uu = arr.values().numpy().detach()
                vv = np.array(uu, dtype=dtype, order=order, copy=copy)
                if uu is not vv: newarr = True
                arr = mtype((vv, tuple(ii)), shape=arr.shape)
            else:
                # We're creating a scipy sparse output from another scipy sparse matrix.
                (rr,cc,uu) = sps.find(obj)
                vv = np.array(uu, dtyp=dtype, order=order, copy=copy)
                if mtype is type(obj) and uu is vv:
                    arr = obj
                else:
                    arr = mtype((vv, (rr,cc)), shape=obj.shape)
                    if uu is not vv: newarr = True
        else:
            # We're creating a scipy sparse matrix from a dense matrix.
            if obj_is_tensor: arr = obj.detach().numpy()
            else: arr = obj
            # Make sure our dtype matches.
            arr = np.asarray(arr, dtype=dtype, order=order)
            # We just call the appropriate constructor.
            arr = mtype(arr)
            newarr = True
        # We mark sparse as True so that below we know that the output is sparse.
        sparse = True
    else:
        # We are creating a dense array output.
        if obj_is_sparse:
            # We are creating a dense array output from a sparse input.
            if obj_is_tensor:
                # We are creating a dense array output from a sparse tensor input.
                arr = obj.todense().detach().numpy()
            else:
                # We are creating a dense array output from a scipy sparse matrix input.
                arr = obj.todense()
            # In both of these cases, a copy has already been made.
            arr = np.asarray(arr, dtype=dtype, order=order)
            newarr = True
        else:
            # We are creating a dense array output from a dense input.
            if obj_is_tensor:
                # We are creating a dense array output from a dense tensor input.
                arr = obj.detach().numpy()
            else:
                arr = obj
            # Whether we call array() or asarray() depends on the copy parameter.
            tmp = np.array(arr, dtype=dtype, order=order, copy=copy)
            newarr = tmp is not arr
            arr = tmp
        # We mark sparse as False so that below we know that the output is dense.
        sparse = False
    # If a read-only array is requested, we either return the object itself (if it is already a
    # read-only array), or we make a copy and make it read-only.
    if readonly is not None:
        if readonly is True:
            if sparse: raise ValueError("scipy sparse matrices cannot be made read-only")
            if arr.flags['WRITEABLE']:
                if not newarr: arr = np.array(arr)
                arr.flags['WRITEABLE'] = False
        elif readonly is False:
            if not sparse and not arr.flags['WRITEABLE']:
                arr = np.array(arr)
        else:
            raise ValueError("invalid parameter value for readonly: {readonly}")
    # Next, we switch on whether we are being asked to return a quantity or not.
    if quant is True:
        if q is None:
            if units is Ellipsis: units = None
            q = ureg.Quantity(arr, units)
        else:
            if units is Ellipsis: units = q.u
            if ureg is not q._REGISTRY or obj is not arr:
                q = ureg.Quantity(arr, q.u)
            return q.to(units)
    elif quant is False:
        # Don't return a quantity, whatever the input argument.
        if units is Ellipsis:
            # We return the current array/magnitude whatever its units.
            return arr
        elif q is None:
            # We just pretend this was already in the given units (i.e., ignore units).
            return arr
        else:
            if obj is not arr: q = ureg.Quantity(arr, q.u)
            # We convert to the given units and return that.
            return q.m_as(units)
    else:
        raise ValueError(f"invalid value for quant: {quant}")

# PyTorch Tensors ##################################################################################
# If PyTorch isn't imported, that's fine, we just write our methods to generate errors. We want
# these errors to explain the problem, so we create our own error type, then have a wrapper for the
# functions that follow that automatically raise the error when torch isn't found.
class TorchNotFound(Exception):
    """Exception raised when PyTorch is requested but is not installed."""
    def __str__(self):
        return ("PyTorch not found.\n\n"
                "The Pimms Library does not require PyTorch, but it must\n"
                "be installed for certain operations to work.\n\n"
                "See https://pytorch.org/get-started/locally/ for help installing PyTorch.")
    @staticmethod
    def raise_self(*args, **kw):
        """Raises a `TorchNotFound` error."""
        raise TorchNotFound()
try:
    import torch
    from torch import Tensor
    @docwrap(indent=8)
    def checktorch(f):
        """Decorator, ensures that PyTorch functions throw an error when torch
        isn't found.
        
        A function that is wrapped with the `@checktorch` decorator will always
        throw a descriptive error message when PyTorch isn't found on the system
        rather than raising a complex exception. Any `pimms` function that uses
        the `torch` library should use this decorator.

        The `torch` library was found on this system, so `checktorch(f)` always
        returns `f`.
        """
        return f
    def alttorch(f_alt):
        """Decorator that runs an alternate function when PyTorch isn't found.
        
        A function `f` that is wrapped with the `@alttorch(f_alt)` decorator
        will always run `f_alt` instead of `f` when called if PyTorch is not
        found on the system and will always run `f` when `PyTorch` is found.

        The `torch` library was found on this system, so `alttorch(f)(f_alt)`
        always returns `f`.
        """
        from functools import wraps
        return (lambda f: f)
except (ModuleNotFoundError, ImportError) as e:
    torch = None
    Tensor = None
    def checktorch(f):
        """Decorator, ensures that PyTorch functions throw an error when torch
        isn't found.
        
        A function that is wrapped with the `@checktorch` decorator will always
        throw a descriptive error message when PyTorch isn't found on the system
        rather than raising a complex exception. Any `pimms` function that uses
        the `torch` library should use this decorator.

        The `torch` library was not found on this system, so `checktorch(f)`
        always returns a function with the same docstring as `f` but which
        raises a `TorchNotFound` exception.
        """
        from functools import wraps
        return wraps(f)(TorchNotFound.raise_self)
    def alttorch(f_alt):
        """Decorator that runs an alternate function when PyTorch isn't found.
        
        A function `f` that is wrapped with the `@alttorch(f_alt)` decorator
        will always run `f_alt` instead of `f` when called if PyTorch is not
        found on the system and will always run `f` when `PyTorch` is found.

        The `torch` library was not found on this system, so
        `alttorch(f)(f_alt)` always returns `f_alt`, or rather a version of
        `f_alt` wrapped to `f`.
        """
        from functools import wraps
        return (lambda f: wraps(f)(f_alt))
# At this point, either torch has been imported or it hasn't, but either way, we can use @checktorch
# to make sure that errors are thrown when torch isn't present. Otherwise, we can just write the
# functions assuming that torch is imported.
@docwrap
@alttorch(lambda dt: False)
def is_torchdtype(dt):
    """Returns `True` for a PyTroch `dtype` object and `False` otherwise.
    
    `is_torchdtype(obj)` returns `True` if the given object `obj` is an instance
    of the `torch.dtype` class.
    
    Parameters
    ----------
    obj : object
        The object whose quality as a PyTorch `dtype` object is to be assessed.
    
    Returns
    -------
    boolean
        `True` if `obj` is a valid `torch.dtype`, otherwise `False`.
    """
    return isinstance(dt, torch.dtype)
@docwrap
def like_torchdtype(dt):
    """Returns `True` for any object that can be converted into a `torch.dtype`.
    
    `like_torchdtype(obj)` returns `True` if the given object `obj` is an
    instance of the `torch.dtype` class, is a string that names a `torch.dtype`
    object, or is a `numpy.dtype` object that is compatible with PyTorch. Note
    that `None` is equivalent to `torch`'s default dtype.
    
    Parameters
    ----------
    obj : object
        The object whose quality as a PyTorch `dtype` object is to be assessed.
    
    Returns
    -------
    boolean
        `True` if `obj` can be converted into a valid `torch.dtype`, otherwise
        `False`.
    """
    if is_torchdtype(dt):
        return True
    elif is_numpydtype(dt):
        try: return None is not torch.from_numpy(np.array([], dtype=dt))
        except TypeError: return False
    elif is_str(dt):
        try: return is_torchdtype(getattr(torch, dt))
        except AttributeError: return False
    elif dt is None:
        return True
    else:
        try: return None is not torch.from_numpy(np.array([], dtype=npdt))
        except Exception: return False
@docwrap
@checktorch
def to_torchdtype(dt):
    """Returns a `torch.dtype` object equivalent to the given argument `dt`.

    `to_torchdtype(obj)` attempts to coerce the given `obj` into a `torch.dtype`
    object. If `obj` is already a `torch.dtype` object, then `obj` itself is
    returned. If the object cannot be converted into a `torch.dtype` object,
    then an error is raised.

    The following kinds of objects can be converted into a `torch.dtype` (see
    also `like_numpydtype()`):
     * `torch.dtype` objects;
     * `numpy.dtype` objects with compatible (numeric) types;
     * strings that name `torch.dtype` objects; or
     * any object that can be passed to `numpy.dtype()`, such as `numpy.int32`,
       that also creates a compatible (numeric) type.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy `dtype` object is to be assessed.

    Returns
    -------
    numpy.dtype
        The `numpy.dtype` object that is equivalent to the argument `dt`.

    Raises
    ------
    TypeError
        If the given argument `dt` cannot be converted into a `numpy.dtype`
        object.
    """
    if   is_numpydtype(dt): return dt
    elif is_torchdtype(dt): return torch.tensor([], dtype=dt).numpy().dtype
    else: return np.dtype(dt)
def _is_never_tensor(obj,
                     dtype=None, shape=None, ndim=None,
                     device=None, requires_grad=None,
                     sparse=None, quant=None, units=Ellipsis):
    return False
@docwrap
@alttorch(_is_never_tensor)
def is_tensor(obj,
              dtype=None, shape=None, ndim=None,
              device=None, requires_grad=None,
              sparse=None, quant=None, units=Ellipsis, ureg=None):
    """Returns `True` if an object is a `torch.tensor` object, else `False`.

    `is_tensor(obj)` returns `True` if the given object `obj` is an instance of
    the `torch.Tensor` class or is a `pint.Quantity` object whose magnitude is
    an instance of `torch.Tensor`. Additional constraints may be placed on the
    object via the optional argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a PyTorch tensor object is to be assessed.
    dtype : dtype-like or None, optional
        The PyTorch `dtype` or dtype-like object that is required to match that
        of the `obj` in order to be considered a valid tensor. The `obj.dtype`
        matches the given `dtype` parameter if either `dtype` is `None` (the
        default) or if `obj.dtype` is equal to the PyTorch equivalent ot
        `dtype`. Alternately, `dtype` can be a tuple, in which case, `obj` is
        considered valid if its dtype is any of the dtypes in `dtype`.
    ndim : int or tuple or ints or None, optional
        The number of dimensions that the object must have in order to be
        considered a valid tensor. If `None`, then any number of dimensions is
        acceptable (this is the default). If this is an integer, then the number
        of dimensions must be exactly that integer. If this is a list or tuple
        of integers, then the dimensionality must be one of these numbers.
    shape : int or tuple of ints or None, optional
        If the `shape` parameter is not `None`, then the given `obj` must have a shape
        shape that matches the parameter value `sh`. The value `sh` must be a
        tuple that is equal to the `obj`'s shape tuple with the following
        additional rules: a `-1` value in the `sh` tuple will match any value in
        the `obj`'s shape tuple, and a single `Ellipsis` may appear in `sh`,
        which matches any number of values in the `obj`'s shape tuple. The
        default value of `None` indicates that no restriction should be applied
        to the `obj`'s shape.
    device : device-name or None, optional
        If `None`, then a tensor with any `device` field is considered valid;
        otherwise, the `device` parameter must equal `obj.device` for `obj` to
        be considered a valid tensor. The default value is `None`.
    requires_grad : boolean or None, optional
        If `None`, then a tensor with any `requires_grad` field is considered
        valid; otherwise, the `requires_grad` parameter must equal
        `obj.requires_grad` for `obj` to be considered a valid tensor. The
        default value is `None`.
    sparse : boolean or False, optional
        If the `sparse` parameter is `None`, then no requirements are placed on
        the sparsity of `obj` for it to be considered a valid tensor. If
        `sparse` is `True` or `False`, then `obj` must either be sparse or not
        be sparse, respectively, for `obj` to be considered valid. If `sparse`
        is a string, then it must be either `'coo'` or `'csr'`, indicating the
        required sparse array type.
    quant : boolean, optional
        Whether `Quantity` objects should be considered valid tensors or not.  If
        `quant=True` then `obj` is considered a valid array only when `obj` is a
        quantity object with a `torch` tensor as the magnitude. If `False`, then
        `obj` must be a `torch` tensor itself and not a `Quantity` to be
        considered valid. If `None` (the default), then either quantities or
        `torch` tensors are considered valid.
    units : unit-like or Ellipsis, optional
        A unit with which the object obj's units must be compatible in order
        for `obj` to be considered a valid tensor. An `obj` that is not a
        quantity is considered to have dimensionless units. If `units=Ellipsis`
        (the default), then the object's units are ignored.
    ureg : UnitRegistry or None or Ellipsis, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `pimms.units` is used. If `ureg` is `None` (the
        default), then the registry of `obj` is used if `obj` is a quantity, and
        `pimms.units` is used if not.

    Returns
    -------
    boolean
        `True` if `obj` is a valid PyTorch tensor, otherwise `False`.
    """
    if ureg is Ellipsis: from pimms import units as ureg
    # If this is a quantity, just extract the magnitude.
    if is_quant(obj):
        if quant is False: return False
        if ureg is None: ureg = obj._REGISTRY
        u = obj.u
        obj = obj.m
    else:
        if quant is True: return False
        if ureg is None: from pimms import units as ureg
        u = None
    # Also here: is this a torch tensor or not?
    if not torch.is_tensor(obj): return False
    # Do we match the varioous torch field requirements?
    if device is not None:
        if obj.device != device: return False
    if requires_grad is not None:
        if obj.requires_grad != requires_grad: return False
    # Do we match the sparsity requirement?
    if sparse is True:
        if not obj.is_sparse: return False
    elif sparse is False:
        if obj.is_sparse: return False
    elif streq(sparse, 'coo', case=False, unicode=False, strip=True):
        if obj.layout != torch.sparse_coo: return False
    elif streq(sparse, 'csr', case=False, unicode=False, strip=True):
        if obj.layout != torch.sparse_csr: return False
    elif sparse is not None:
        raise ValueErroor(f"invalid sparse parameter: {sparse}")
    if units is not Ellipsis and not ureg.is_compatible_with(u, units): return False
    # Check the match to the numeric collection last.
    if dtype is None and shape is None and ndim is None: return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, dtype)
def to_tensor(obj,
              dtype=None, device=None, requires_grad=None, copy=False,
              sparse=None, quant=None, ureg=None, units=Ellipsis):
    """Reinterprets `obj` as a PyTorch tensor or quantity with tensor magnitude.

    `pimms.to_tensor` is roughly equivalent to the `torch.as_tensor` function
    with a few exceptions:
      * `to_tensor(obj)` allows quantities for `obj` and, in such a case, will
        return a quantity whose magnitude has been reinterpreted as a tensor,
        though this behavior can be altered with the `quant` parameter;
      * `to_tensor(obj)` can convet a SciPy sparse matrix into a sparrse tensor.

    Parameters
    ----------
    obj : object
        The object that is to be reinterpreted as, or if necessary covnerted to,
        a PyTorch tensor object.
    dtype : data-type, optional
        The dtype that is passed to `torch.as_tensor(obj)`.
    device : device name or None, optional
        The `device` parameter that is passed to `torch.as_tensor(obj)`, `None`
        by default.
    requires_grad : boolean or None, optional
        Whether the returned tensor should require gradient calculations or not.
        If `None` (the default), then the objecct `obj` is not changed from its
        current gradient settings, if `obj` is a tensor, and `obj` is not made
        to track its gradient if it is converted into a tensor. If the 
        `requires_grad` parameter does not match the given tensor's
        `requires_grad` field, then a copy is always returned.
    copy : boolean, optional
        Whether to copy the data in `obj` or not. If `False`, then `obj` is only
        copied if doing so is required by the optional parameters. If `True`,
        then `obj` is always copied if possible.
    sparse : boolean or {'csr','coo'} or None, optional
        If `None`, then the sparsity of `obj` is the same as the sparsity of the
        tensor that is returned. Otherwise, the return value will always be
        either a spase tensor (`sparse=True`) or a dense tensor (`sparse=False`)
        based on the given value of `sparse`. The `sparse` parameter may also be
        set to `'coo'` or `'csr'` to return specific sparse layouts.
    quant : boolean or None, optional
        Whether the return value should be a `Quantity` object wrapping the
        array (`quant=True`) or the array itself (`quant=False`). If `quant` is
        `None` (the default) then the return value is a quantity if `obj` is a
        quantity and is not a quantity if `obj` is not a quantity.
    ureg : pint.UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `pimms.units` is used. If `ureg` is `None` (the
        default), then no specific coersion to a `UnitRegistry` is performed
        (i.e., the same quantity class is returned).
    units : unit-like or boolean or Ellipsis, optional
        The units that should be used in the return value. When the return value
        of this function is a `Quantity` (see the `quant` parameter), the
        returned quantity always has units matching the `units` parameter; if
        the provided `obj` is not a quantity, then its units are presumed to be
        those requested by `units`. When the return value of this function is
        not a `Quantity` object and is instead a PyTorch tensor object, then
        when `obj` is not a quantity the `units` parameter is ignored, and when
        `obj` is a quantity, its magnitude is returned after conversion into
        `units`. The default value of `units`, `Ellipsis`, indicates that, if
        `obj` is a quantity, its units should be used, and `units` should be
        considered dimensionless otherwise.

    Returns
    -------
    NumPy array or Quantity
        Either a PyTorch tensor equivalent to `obj` or a `Quantity` whose
        magnitude is a PyTorch tensor equivalent to `obj`.

    Raises
    ------
    ValueError
        If invalid parameter values are given or if the parameters conflict.
    """
    if ureg is Ellipsis: ureg = pimms.units
    dtype = to_torchdtype(dtype)
    # If obj is a quantity, we handle things differently.
    if is_quant(obj):
        q = obj
        obj = q.m
        if ureg is None: ureg = q._REGISTRY
        if quant is None: quant = True
    else:
        q = None
        if ureg is None: ureg = pimms.units
        if quant is None: quant = False
    # Translate obj depending on whether it's a pytorch tensor already or a scipy sparse matrix.
    if torch.is_tensor(obj):
        if requires_grad is None: requires_grad = obj.requires_grad
        if device is None: device = obj.device
        if copy or requires_grad != obj.requires_grad:
            arr = torch.tensor(obj, dtype=dtype, device=device, requires_grad=requires_grad)
        else:
            arr = torch.as_tensor(obj, dtype=dtype, device=device)
        arr = obj
    elif scipy_issparse(obj):
        if requires_grad is None: requires_grad = False
        (rows, cols, vals) = sps.find(obj)
        # Process these into a PyTorch COO matrix.
        ii = torch.tensor([rows, cols], dtype=torch.long, device=device)
        arr = torch.sparse_coo_tensor(ii, vals, obj.shape,
                                      dtype=dtype, device=device, requires_grad=requires_grad)
        # Convert to a CSR tensor if we were given a CSR matrix.
        if isinstance(obj, sps.csr_matrix): arr = arr.to_sparse_csr()
    elif copy or requires_grad is True:
        arr = torch.tensor(arr, dtype=dtype, device=devce, requires_grad=requires_grad)
    else:
        arr = torch.as_tensor(arr, dtype=dtype, device=device)
    # If there is an instruction regarding the output's sparsity, handle that now.
    if sparse is True:
        # arr must be sparse (COO by default); make sure it is.
        if not arr.is_sparse: arr = arr.to_sparse()
    elif sparse is False:
        # arr must not be a sparse array; make sure it isn't.
        if arr.is_sparse: arr = arr.to_dense()
    elif streq(sparse, 'csr', case=False, unicode=False, strip=True):
        if arr.layout is not torch.sparse_csr:
            arr = arr.to_sparse_csr()
    elif streq(sparse, 'coo', case=False, unicode=False, strip=True):
        if not arr.is_sparse: arr = arr.to_sparse()
        if arr.layout is not torch.sparse_coo:
            arr = arr.coalesce()
            arr = torch.sparse_coo_tensor(arr.indices(), arr.vales(), arr.shape,
                                          dtype=dtype, device=device, requires_grad=requires_grad)
    elif sparse is not None:
        raise ValueError(f"invalid value for parameter sparse: {sparse}")
    # Next, we switch on whether we are being asked to return a quantity or not.
    if quant is True:
        if q is None:
            if units is Ellipsis: units = None
            q = ureg.Quantity(arr, units)
        else:
            if units is Ellipsis: units = q.u
            if ureg is not q._REGISTRY or obj is not arr:
                q = ureg.Quantity(arr, q.u)
            return q.to(units)
    elif quant is False:
        # Don't return a quantity, whatever the input argument.
        if units is Ellipsis:
            # We return the current array/magnitude whatever its units.
            return arr
        elif q is None:
            # We just pretend this was already in the given units (i.e., ignore units).
            return arr
        else:
            if obj is not arr: q = ureg.Quantity(arr, q.u)
            # We convert to the given units and return that.
            return q.m_as(units)
    else:
        raise ValueError(f"invalid value for quant: {quant}")

# General Numeric Collection Functions #############################################################
@docwrap
def is_numeric(obj,
               dtype=None, shape=None, ndim=None,
               sparse=None, quant=None, units=Ellipsis, ureg=None):
    """Returns `True` if an object is a numeric type and `False` otherwise.

    `is_numeric(obj)` returns `True` if the given object `obj` is an instance of
    the `torch.Tensor` class, the `numpy.ndarray` class, one one of the
    `scipy.sparse` matrix classes, or is a `pint.Quantity` object whose
    magnitude is an instance of one of these types. Additional constraints may
    be placed on the object via the optional argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a numeric object is to be assessed.
    dtype : dtype-like or None, optional
        The NumPy or PyTorch `dtype` or dtype-like object that is required to
        match that of the `obj` in order to be considered valid. The `obj.dtype`
        matches the given `dtype` parameter if either `dtype` is `None` (the
        default) or if `obj.dtype` is equivalent to `dtype`. Alternately,
        `dtype` can be a tuple, in which case, `obj` is considered valid if its
        dtype is any of the dtypes in `dtype`.
    ndim : int or tuple or ints or None, optional
        The number of dimensions that the object must have in order to be
        considered valid. If `None`, then any number of dimensions is acceptable
        (this is the default). If this is an integer, then the number of
        dimensions must be exactly that integer. If this is a list or tuple of
        integers, then the dimensionality must be one of these numbers.
    shape : int or tuple of ints or None, optional
        If the `shape` parameter is not `None`, then the given `obj` must have a shape
        shape that matches the parameter value `sh`. The value `sh` must be a
        tuple that is equal to the `obj`'s shape tuple with the following
        additional rules: a `-1` value in the `sh` tuple will match any value in
        the `obj`'s shape tuple, and a single `Ellipsis` may appear in `sh`,
        which matches any number of values in the `obj`'s shape tuple. The
        default value of `None` indicates that no restriction should be applied
        to the `obj`'s shape.
    sparse : boolean or False, optional
        If the `sparse` parameter is `None`, then no requirements are placed on
        the sparsity of `obj` for it to be considered valid. If `sparse` is
        `True` or `False`, then `obj` must either be sparse or not be sparse,
        respectively, for `obj` to be considered valid. If `sparse` is a string,
        then it must be a valid sparse matrix type that matches the type of
        `obj` for `obj` to be considered valid.
    quant : boolean, optional
        Whether `Quantity` objects should be considered valid or not.  If
        `quant=True` then `obj` is considered a valid numerical object only when
        `obj` is a quantity object with a valid numerical object as the
        magnitude. If `False`, then `obj` must be a numerical object itself and
        not a `Quantity` to be considered valid. If `None` (the default), then
        either quantities or numerical objects are considered valid.
    units : unit-like or Ellipsis, optional
        A unit with which the object obj's units must be compatible in order for
        `obj` to be considered a valid numerical objbect. An `obj` that is not a
        quantity is considered to have dimensionless units. If `units=Ellipsis`
        (the default), then the object's units are ignored.
    ureg : UnitRegistry or None or Ellipsis, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `pimms.units` is used. If `ureg` is `None` (the
        default), then the registry of `obj` is used if `obj` is a quantity, and
        `pimms.units` is used if not.

    Returns
    -------
    boolean
        `True` if `obj` is a valid numerical object, otherwise `False`.
    """
    if torch is not None and torch.is_tensor(obj):
        return is_tensor(obj,
                         dtype=dtype, shape=shape, ndim=ndim,
                         sparse=sparse, quant=quant, units=units, ureg=ureg)
    else:
        return is_array(obj,
                        dtype=dtype, shape=shape, ndim=ndim,
                        sparse=sparse, quant=quant, units=units, ureg=ureg)
def to_numeric(obj,
               dtype=None, copy=False,
               sparse=None, quant=None, ureg=None, units=Ellipsis):
    """Reinterprets `obj` as a numeric type or quantity with such a magnitude.

    `pimms.to_numeric` is roughly equivalent to the `torch.as_tensor` or
    `numpy.asarray` function with a few exceptions:
      * `to_numeric(obj)` allows quantities for `obj` and, in such a case, will
        return a quantity whose magnitude has been reinterpreted as a numeric,
        object, though this behavior can be altered with the `quant` parameter;
      * `to_numeric(obj)` correctly handles SciPy sparse matrices, NumPy arrays,
        and PyTorch tensors.

    If the object `obj` passed to `pimms.to_numeric(obj)` is a PyTorch tensor,
    then a PyTorch tensor or a quantity with a PyTorch tensor magnitude is
    returned. Otherwise, a NumPy array, SciPy sparse matrix, or quantity with a
    magnitude matching one of these types is returned.

    Parameters
    ----------
    obj : object
        The object that is to be reinterpreted as, or if necessary covnerted to,
        a numeric object.
    dtype : data-type, optional
        The dtype that is passed to `torch.as_tensor(obj)` or `np.asarray(obj)`.
    copy : boolean, optional
        Whether to copy the data in `obj` or not. If `False`, then `obj` is only
        copied if doing so is required by the optional parameters. If `True`,
        then `obj` is always copied if possible.
    sparse : boolean or {'csr','coo'} or None, optional
        If `None`, then the sparsity of `obj` is the same as the sparsity of the
        object that is returned. Otherwise, the return value will always be
        either a spase object (`sparse=True`) or a dense object (`sparse=False`)
        based on the given value of `sparse`. The `sparse` parameter may also be
        set to `'coo'`, `'csr'`, or other sparse matrix names to return specific
        sparse layouts.
    quant : boolean or None, optional
        Whether the return value should be a `Quantity` object wrapping the
        object (`quant=True`) or the object itself (`quant=False`). If `quant`
        is `None` (the default) then the return value is a quantity if `obj` is
        a quantity and is not a quantity if `obj` is not a quantity.
    ureg : pint.UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `pimms.units` is used. If `ureg` is `None` (the
        default), then no specific coersion to a `UnitRegistry` is performed
        (i.e., the same quantity class is returned).
    units : unit-like or boolean or Ellipsis, optional
        The units that should be used in the return value. When the return value
        of this function is a `Quantity` (see the `quant` parameter), the
        returned quantity always has units matching the `units` parameter; if
        the provided `obj` is not a quantity, then its units are presumed to be
        those requested by `units`. When the return value of this function is
        not a `Quantity` object and is instead a numeric object, then
        when `obj` is not a quantity the `units` parameter is ignored, and when
        `obj` is a quantity, its magnitude is returned after conversion into
        `units`. The default value of `units`, `Ellipsis`, indicates that, if
        `obj` is a quantity, its units should be used, and `units` should be
        considered dimensionless otherwise.

    Returns
    -------
    NumPy array or PyTorch tensor or Quantity
        Either a NumPy array or PyTorch tensor equivalent to `obj` or a 
       `Quantity` whose magnitude is such an object.

    Raises
    ------
    ValueError
        If invalid parameter values are given or if the parameters conflict.

    """
    if torch is not None and torch.is_tensor(obj):
        return to_tensor(obj,
                         dtype=dtype, shape=shape, ndim=ndim,
                         sparse=sparse, quant=quant, units=units, ureg=ureg)
    else:
        return to_array(obj,
                        dtype=dtype, shape=shape, ndim=ndim,
                        sparse=sparse, quant=quant, units=units, ureg=ureg)


# Sparse Matrices and Dense Collections#############################################################
from scipy.sparse import issparse as scipy_issparse
@docwrap
def is_sparse(obj,
              dtype=None, shape=None, ndim=None,
              quant=None, ureg=None, units=Ellipsis):
    """Returns `True` if an object is a sparse SciPy matrix or PyTorch tensor.

    `is_sparse(obj)` returns `True` if the given object `obj` is an instance of
    one of the SciPy sprase matrix classes, is a sparse PyTorch tensor, or is a
    quantity whose magnintude is one of theese. Additional constraints may be
    placed on the object via the optional argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a sparse numerical object is to be assessed.
    %(pimms.types._core.is_numeric.dtype)s
    %(pimms.types._core.is_numeric.ndim)s
    %(pimms.types._core.is_numeric.shape)s
    %(pimms.types._core.is_numeric.quant)s
    %(pimms.types._core.is_numeric.ureg)s
    %(pimms.types._core.is_numeric.units)s

    Returns
    -------
    boolean
        `True` if `obj` is a valid sparse numerical object, otherwise `False`.
    """
    return is_numeric(obj, sparse=True,
                      dtype=dtype, shape=shape, ndim=ndim,
                      quant=quant, ureg=ureg, units=units)
@docwrap
def to_sparse(obj,
              dtype=None, shape=None, ndim=None,
              quant=None, ureg=None, units=Ellipsis):
    """Returns a sparse version of the numerical object `obj`.

    `to_sparse(obj)` returns `obj` if it is already a PyTorch sparse tensor or a
    SciPy sparse matrix or a quantity whose magnitude is one of these.
    Otherwise, it converts `obj` into a sparse representation and returns
    this. Additional requirements on the output format of the return value can
    be added using the optional parameters.

    Parameters
    ----------
    obj : object
        The object that is to be converted into a sparse representation.
    %(pimms.types._core.to_numeric.dtype)s
    %(pimms.types._core.to_numeric.ndim)s
    %(pimms.types._core.to_numeric.shape)s
    %(pimms.types._core.to_numeric.quant)s
    %(pimms.types._core.to_numeric.ureg)s
    %(pimms.types._core.to_numeric.units)s

    Returns
    -------
    sparse tensor or sparse matrix or quantity with a sparse magnitude
        A sparse version of the argument `obj`.
    """
    return to_numeric(obj, sparse=True,
                      dtype=dtype, shape=shape, ndim=ndim,
                      quant=quant, ureg=ureg, units=units)
@docwrap
def is_dense(obj,
             dtype=None, shape=None, ndim=None,
             quant=None, ureg=None, units=Ellipsis):
    """Returns `True` if an object is a dense NumPy array or PyTorch tensor.

    `is_dense(obj)` returns `True` if the given object `obj` is an instance of
    one of the NumPy `ndarray` classes, is a dense PyTorch tensor, or is a
    quantity whose magnintude is one of theese. Additional constraints may be
    placed on the object via the optional argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a dense numerical object is to be assessed.
    %(pimms.types._core.is_numeric.dtype)s
    %(pimms.types._core.is_numeric.ndim)s
    %(pimms.types._core.is_numeric.shape)s
    %(pimms.types._core.is_numeric.quant)s
    %(pimms.types._core.is_numeric.ureg)s
    %(pimms.types._core.is_numeric.units)s

    Returns
    -------
    boolean
        `True` if `obj` is a valid dense numerical object, otherwise `False`.
    """
    return is_numeric(obj, sparse=False,
                      dtype=dtype, shape=shape, ndim=ndim,
                      requires_grad=requires_grad, device=device,
                      quant=quant, ureg=ureg, units=units)
@docwrap
def to_dense(obj,
             dtype=None, shape=None, ndim=None,
             quant=None, ureg=None, units=Ellipsis):
    """Returns a dense version of the numerical object `obj`.

    `to_dense(obj)` returns `obj` if it is already a PyTorch dense tensor or a
    NumPy `ndarray` or a quantity whose magnitude is one of these.  Otherwise,
    it converts `obj` into a dense representation and returns this. Additional
    requirements on the output format of the return value can be added using the
    optional parameters.

    Parameters
    ----------
    obj : object
        The object that is to be converted into a dense representation.
    %(pimms.types._core.to_numeric.dtype)s
    %(pimms.types._core.to_numeric.ndim)s
    %(pimms.types._core.to_numeric.shape)s
    %(pimms.types._core.to_numeric.quant)s
    %(pimms.types._core.to_numeric.ureg)s
    %(pimms.types._core.to_numeric.units)s

    Returns
    -------
    dense tensor or dense ndarray or quantity with a dense magnitude
        A dense version of the argument `obj`.
    """
    return to_numeric(obj, sparse=False,
                      dtype=dtype, shape=shape, ndim=ndim,
                      quant=quant, ureg=ureg, units=units)

# Strings ##########################################################################################
@docwrap
def is_str(obj):
    """Returns `True` if an object is a string and `False` otherwise.

    `is_str(obj)` returns `True` if the given object `obj` is an instance
    of the `str` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a string object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is a string, otherwise `False`.
    """
    return isinstance(obj, str)
from unicodedata import normalize as unicodedata_normalize
@docwrap
def strnorm(s, case=False, unicode=True):
    """Normalizes a string using the `unicodedata` package.

    `strnorm(s)` returns a version of `s` that has been unicode-normalized using
    the `unicodedata.normalize(s)` function. Case-normalization can also be
    requested via the `case` parameter.

    Parameters
    ----------
    s : object
        The string to be normalized.
    case : boolean, optional
        Whether to perform case-normalization (`case=True`) or not
        (`case=False`, the default). If two strings are case-normalized, then an
        equality comparison will reveal whether the original (unnormalized
        strings) were equal up to the case of the characters. Case normalization
        is performed using the `s.casefold()` method.
    unicode : boolean, optional
        Whether to perform unicode normalization via the `unicodedata.normalize`
        function. The default behavior (`unicode=True`) is to perform
        normalization, but this can be disabled with `unicode=False`.

    Returns
    -------
    str
       A normalized versoin of `s`.
    """
    if unicode is True: unicode = 'NFD'
    if unicode:
        s = unicodedata_normalize(unicode, s)
        if case:
            s = s.casefold()
            s = unicodedata_normalize(unicode, s)
    elif case:
        s = s.casefold()
    return s
@docwrap
def strcmp(a, b, case=True, unicode=None, strip=False, split=False):
    """Determines if the given objects are equal strings and compares them.

    `strcmp(a, b)` returns `None` if either `a` or `b` is not a string;
    otherwise, it returns `-1`, `0`, or `1` if `a` is less than, equal to, or
    greater than `b`, respectively, subject to the constraints of the
    parameters.

    Parameters
    ----------
    a : object
        The first argument.
    b : object
        The second argument.
    case : boolean, optional
        Whether to perform case-sensitive (`case=True`, the default) or
        case-insensitive (`case=False`) string comparison.
    unicode : boolean or None, optional
        Whether to run unicode normalization on `a` and `b` prior to
        comparison. By default, this is `None`, which is interpreted as a `True`
        value when `case` is `False` and a `False` value when `case` is `True`
        (i.e., unicode normalization is performed when case-insensitive
        comparison is being performed but not when standard string comparison is
        being performed. Unicode normalization is always performed both before
        and after casefolding. Unicode normalization is performed using the
        `unicodedata` package's `normalize(unicode, string)` function. If this
        argument is a string, it is instead passed to the `normalize` function
        as the first argument.
    strip : boolean, optional
        If set to `True`, then `a.strip()` and `b.strip()` are used in place of
        `a` and `b`. If set to `False` (the default), then no stripping is
        performed. If a non-boolean value is given, then it is passed as an
        argument to the `strip()` method.
    split : boolean, optional
        If set to `True`, then `a.split()` and `b.split()` are used in place of
        `a` and `b`, where the lists that result from `a.split()` and
        `b.split()` are compared for their sorted ordering of strings. If set to
        `False` (the default), then no splitting is performed. If a non-boolean
        value is given, then it is passed as an argument to the `split()`
        method.

    Returns
    -------
    boolean
        `None` if either `a` is not a string or `b` is not a string; otherwise,
        `-1` if `a` is lexicographically less than `b`, `0` if `a == b`, and `1`
        if `a` is lexicographically greater than `b`, subject to the constraints
        of the optional parameters.
    """
    if not is_str(a) or not is_str(b): return None
    # We do case normalization when case comparison is *not* requested
    casenorm = not bool(case)
    # When unicode is None, we do its normalization only when case normalization is being done.
    if unicode is None: unicode = casenorm
    # We now perform normalization if it is required.
    if unicode or casenorm:
        a = strnorm(a, case=casenorm, unicode=unicode)
        b = strnorm(b, case=casenorm, unicode=unicode)
    # If we requested stripping, do that now.
    if strip is not False:
        if strip is True:
            a = a.strip()
            b = b.strip()
        else:
            a = a.strip(strip)
            b = b.strip(strip)
    # If straightforward non-split comparison is requested, we can just return the comparison
    if split is False:
        return (-1 if a < b else 1 if a > b else 0)
    # Otherwise, we split then do a list comparison.
    if split is True:
        a = a.split()
        b = b.split()
    else:
        a = a.split(split)
        b = b.split(split)
    # We do this efficiently, by first comparing elements in order, then lenghts.
    for (aa,bb) in zip(a,b):
        if   aa < bb: return -1
        elif aa > bb: return 1
    na = len(a)
    nb = len(b)
    if   na < nb: return -1
    elif na > nb: return 1
    return 0
@docwrap
def streq(a, b, case=True, unicode=None, strip=False, split=False):
    """Determines if the given objects are equal strings or not.

    `streq(a, b)` returns `True` if `a` and `b` are both strings and are equal
    to each other, subject to the constraints of the parameters.

    Parameters
    ----------
    %(pimms.types._core.strcmp.parameters)s

    Returns
    -------
    boolean
        `True` if `a` and `b` are both strings and if `a` is equal to `b`,
        subject to the constraints of the optional parameters.
    """
    cmpval = strcmp(a, b, case=case, unicode=unicode, strip=strip, split=split)
    return cmpval == 0

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
