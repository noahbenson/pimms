# -*- coding: utf-8 -*-
################################################################################
# pimms/types/_core.py
#
# Core implementation of the utility classes for the various types that are
# managed by pimms.
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

# Dependencies #################################################################
import pint, os, inspect
import numpy as np
import scipy.sparse as sps

from ..doc import docwrap

# Units ########################################################################
# Units are fundamentally treated as part of the pimms type-system. Pimms
# functios that deal with an object's type typically take an option `unit` that
# can be used to change the function's behavior depending on the units attached
# to an object.
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
def is_unit(q, ureg=None):
    """Returns `True` if `q` is a `pint` unit and `False` otherwise.

    `is_unit(q)` returns `True` if `q` is a `pint` unit and `False` otherwise.

    Parameters
    ----------
    q : object
        The object whose quality as a `pint` unit is to be assessed.
    ureg : UnitRegistry or None, optional
        The `pint` `UnitRegistry` object that the given unit object must belong
        to. If `None` (the default), then any unit registry is allowed. If
        `Ellipsis`, then the `pimms.units` registry is used. Otherwise, this
        must be a specific `UnitRegistry` object.

    Returns
    -------
    boolean
        `True` if `q` is a `pint` unit and `False` otherwise.

    Raises
    ------
    TypeError
        If the `ureg` parameter is not a `UnitRegistry`, `Ellipsis`, or `None`.
    """
    if ureg is None:
        return isinstance(q, Unit)
    elif ureg is Ellipsis:
        from pimms import units
        return isinstance(q, units.Unit)
    elif is_ureg(ureg):
        return isinstance(q, ureg.Unit)
    else:
        raise TypeError("parameter ureg must be a UnitRegistry")
@docwrap
def is_quant(obj, unit=Ellipsis, ureg=None):
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
        tuple of such units/unit-names, or `None`. If `Ellipsis` (the default),
        then the object must be a `Quantity` object, but it doesn't matter what
        the unit of the object is. Otherwise, the object must have a unit
        equivalent to the unit or to one of the units given (`unit` may be a
        tuple of possible units). The `UnitRegistry` objects for the units given
        via this parameter are ignored; only the `ureg` parameter influences the
        `UnitRegistry` requirements. Note that the value `None` for a unit type
        indicates a scalar without a unit (i.e., an object that is not a
        quantity), and so, while `None` is a valid value, this function will
        always return `False` when it is passed.
    ureg : pint.UnitRegistry or Ellipsis or None, optional
        The `pint` `UnitRegistry` object to use for units. If `Ellipsis`, then
        `pimms.units` is used. If `ureg` is `None` (the default), then a
        specific unit registry is not checked.

    Returns
    -------
    boolean
        `True` if `obj` is a `pint` quantity whose unit is compatible with the
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
            from pimms import units
            ureg = units
        elif not is_ureg(ureg):
            raise TypeError("parameter ureg must be a UnitRegistry")
        if not isinstance(obj, ureg.Quantity):
            return False
    return (True  if unit is Ellipsis else
            False if unit is None     else
            obj.is_compatible_with(unit))

# Numerical Collection Suport ##################################################
# Numerical collections include numpy arrays and torch tensors. These objects
# are handled similarly due to their overall functional similarity, and certain
# support functions are used for both.
def _numcoll_match(numcoll_shape, numcoll_dtype, ndim, shape, dtype):
    """Checks that the actual numcoll shape and the actual numcol dtype match
    the requirements of the ndim, shape, and dtype parameters.
    """
    # Parse the shape int front and back requirements and whether middle values
    # are allowed.
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
        # We might have finished with just that; otherwise, note the ellipsis
        # and move on.
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
    # If there are extras in the middle and we don't allow them, we fail the
    # match.
    if not sh_mid and nsuf + npre != ndim: return False
    # See if we match the dtype.
    if dtype is not None:
        if is_numpydtype(numcoll_dtype):
            if is_seq(dtype) or is_set(dtype):
                dtype = [to_numpydtype(dt) for dt in dtype]
            else:
                # If we have been given a torch dtype, we convert it, but
                # otherwise we let np.issubdtype do the converstion so that
                # users can pass in things like np.integer meaningfully.
                if is_torchdtype(dtype): dtype = to_numpydtype(dtype)
                if not np.issubdtype(numcoll_dtype, dtype):
                    return False
                dtype = (numcoll_dtype,)
        elif is_torchdtype(numcoll_dtype):
            if is_seq(dtype) or is_set(dtype):
                dtype = [to_torchdtype(dt) for dt in dtype]
            else: dtype = [to_torchdtype(dtype)]
        if numcoll_dtype not in dtype: return False
    # We match everything!
    return True

# Numpy Arrays #################################################################
# For testing whether numpy arrays or pytorch tensors have the appropriate
# dimensionality, shape, and dtype, we use some helper functions.
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
             sparse=None, quant=None, unit=Ellipsis, ureg=None):
    """Returns `True` if an object is a `numpy.ndarray` object, else `False`.

    `is_array(obj)` returns `True` if the given object `obj` is an instance of
    the `numpy.ndarray` class or is a `scipy.sparse` matrix or if `obj` is a
    `pint.Quantity` object whose magnitude is one of these. Additional
    constraints may be placed on the object via the optional argments.

    Note that to `pimms`, both `numpy.ndarray` arrays and `scipy.sparse`
    matrices are considered "arrays". This behavior can be changed with the
    `sparse` parameter.

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
        dtype is any of the dtypes in `dtype`. Note that in the case of a tuple,
        the dtype of `obj` must appear exactly in the tuple rather than be a
        subtype of one of the objects in the tuple.
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
    unit : unit-like or Ellipsis or None, optional
        A unit with which the object obj's unit must be compatible in order for
        `obj` to be considered a valid array. An `obj` that is not a quantity is
        considered to have a unit of `None`, which is not the same as being a
        quantity with a dimensionless unit. In other words, `is_array(array,
        quant=None)` will return `True` for a numpy array while `is_array(arary,
        quant='dimensionless')` will return `False`. If `unit=Ellipsis` (the
        default), then the object's unit is ignored.
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
    elif quant is True:
        return False
    else:
        if ureg is None: from pimms import units as ureg
        u = None
    # At this point we want to check if this is a valid numpy array or scipy
    # sparse matrix; however how we handle the answer to this question depends
    # on the sparse parameter.
    if sparse is True:
        if not scipy__is_sparse(obj): return False
    elif sparse is False:
        if not isinstance(obj, ndarray): return False
    elif sparse is None:
        if not (isinstance(obj, ndarray) or scipy__is_sparse(obj)): return False
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
            if scipy__is_sparse(obj) or obj.flags['WRITEABLE']: return False
        elif readonly is False:
            if not scipy__is_sparse(obj) and not obj.flags['WRITEABLE']:
                return False
        else:
            raise ValueError(f"invalid parameter readonly: {readonly}")
    # Next, check compatibility of the units.
    if unit is None:
        # We are required to not be a quantity.
        if u is not None: return False
    elif unit is not Ellipsis:
        from ._quantity import alike_units
        if not is_tuple(unit): unit = (unit,)
        if not any(alike_units(u, uu) for uu in unit):
            return False
    # Check the match to the numeric collection last.
    if dtype is None and shape is None and ndim is None: return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, dtype)
def to_array(obj,
             dtype=None, order=None, copy=False, sparse=None, readonly=None,
             quant=None, ureg=None, unit=Ellipsis):
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
        `readonly=True`, then the return value is always a read-only array; if
        `obj` is not already read-only, then a copy of `obj` is always returned
        in this case. If `readonly=False`, then the return-value is never
        read-only. Note that `scipy.sparse` arrays do not support read-only
        mode, and thus a `ValueError` is raised if a sparse matrix is requested
        in read-only format.
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
    unit : unit-like or boolean or Ellipsis, optional
        The unit that should be used in the return value. When the return value
        of this function is a `Quantity` (see the `quant` parameter), the
        returned quantity always has a unit matching the `unit` parameter; if
        the provided `obj` is not a quantity, then its unit is presumed to be
        that requested by `unit`. When the return value of this function is not
        a `Quantity` object and is instead is a NumPy array object, then when
        `obj` is not a quantity the `unit` parameter is ignored, and when `obj`
        is a quantity, its magnitude is returned after conversion into
        `unit`. The default value of `unit`, `Ellipsis`, indicates that, if
        `obj` is a quantity, its unit should be used, and `unit` should be
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
    if ureg is Ellipsis: from pimms import units as ureg
    # If obj is a quantity, we handle things differently.
    if is_quant(obj):
        q = obj
        obj = q.m
        if ureg is None: ureg = q._REGISTRY
        if quant is None: quant = True
    else:
        q = None
        if ureg is None: from pimms import units as ureg
        if quant is None: quant = False
    # Translate obj depending on whether it's a pytorch array / scipy sparse
    # matrix.  We need to think about whether the output array is being
    # requested in sparse format. If so, we handle the conversion differently.
    obj_is_spsparse = scipy__is_sparse(obj)
    obj_is_tensor = not obj_is_spsparse and torch__is_tensor(obj)
    obj_is_sparse = obj_is_spsparse or (obj_is_tensor and obj.is_sparse)
    newarr = False # True means we own the memory of arr; False means we don't.
    if sparse is not False and (sparse is not None or obj_is_sparse):
        if sparse is None or sparse is True:
            if obj_is_tensor:
                sparse = ('csr' if obj.layout == torch.sparse_csr else 'coo')
            elif obj_is_sparse:
                sparse = type(obj).__name__[:3]
            else:
                sparse = 'coo'
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
                # We're creating a scipy sparse output from another scipy sparse
                # matrix.
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
        # We mark sparse as True so that below we know that the output is
        # sparse.
        sparse = True
    else:
        # We are creating a dense array output.
        if obj_is_sparse:
            # We are creating a dense array output from a sparse input.
            if obj_is_tensor:
                # We are creating a dense array output from a sparse tensor
                # input.
                arr = obj.todense().detach().numpy()
            else:
                # We are creating a dense array output from a scipy sparse
                # matrix input.
                arr = obj.todense()
            # In both of these cases, a copy has already been made.
            arr = np.asarray(arr, dtype=dtype, order=order)
            newarr = True
        else:
            # We are creating a dense array output from a dense input.
            if obj_is_tensor:
                # We are creating a dense array output from a dense tensor
                # input.
                arr = obj.detach().numpy()
            else:
                arr = obj
            # Whether we call array() or asarray() depends on the copy
            # parameter.
            tmp = np.array(arr, dtype=dtype, order=order, copy=copy)
            newarr = tmp is not arr
            arr = tmp
        # We mark sparse as False so that below we know that the output is
        # dense.
        sparse = False
    # If a read-only array is requested, we either return the object itself (if
    # it is already a read-only array), or we make a copy and make it read-only.
    if readonly is not None:
        if readonly is True:
            if sparse:
                raise ValueError("scipy sparse matrices cannot be read-only")
            if arr.flags['WRITEABLE']:
                if not newarr: arr = np.array(arr)
                arr.flags['WRITEABLE'] = False
        elif readonly is False:
            if not sparse and not arr.flags['WRITEABLE']:
                arr = np.array(arr)
        else:
            raise ValueError(f"bad parameter value for readonly: {readonly}")
    # Next, we switch on whether we are being asked to return a quantity or not.
    if quant is True:
        if unit is None:
            raise ValueError("to_array: cannot make a quantity (quant=True)"
                             " without a unit (unit=None)")
        if q is None:
            if unit is Ellipsis: unit = None
            return ureg.Quantity(arr, unit)
        else:
            if unit is Ellipsis: unit = q.u
            if ureg is not q._REGISTRY or obj is not arr:
                q = ureg.Quantity(arr, q.u)
            return q.to(unit)
    elif quant is False:
        # Don't return a quantity, whatever the input argument.
        if unit is Ellipsis:
            # We return the current array/magnitude whatever its unit.
            return arr
        elif q is None:
            # We just pretend this was already in the given unit (i.e., ignore
            # unit).
            return arr
        elif unit is None:
            raise ValueError("cannot extract unit None from quantity; to get"
                             " the native unit, use unit=Ellipsis")
        else:
            if obj is not arr: q = ureg.Quantity(arr, q.u)
            # We convert to the given unit and return that.
            return q.m_as(unit)
    else:
        raise ValueError(f"invalid value for quant: {quant}")

# PyTorch Tensors ##############################################################
# If PyTorch isn't imported, that's fine, we just write our methods to generate
# errors. We want these errors to explain the problem, so we create our own
# error type, then have a wrapper for the functions that follow that
# automatically raise the error when torch isn't found.
class TorchNotFound(Exception):
    """Exception raised when PyTorch is requested but is not installed."""
    def __str__(self):
        return ("PyTorch not found.\n\n"
                "The Pimms Library does not require PyTorch, but it must\n"
                "be installed for certain operations to work.\n\n"
                "See https://pytorch.org/get-started/locally/ for help\n"
                "installing PyTorch.")
    @staticmethod
    def raise_self(*args, **kw):
        """Raises a `TorchNotFound` error."""
        raise TorchNotFound()
try:
    import torch
    from torch import Tensor
    from torch import is_tensor as torch__is_tensor
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
    def torch__is_tensor(obj):
        return False
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
# At this point, either torch has been imported or it hasn't, but either way, we
# can use @checktorch to make sure that errors are thrown when torch isn't
# present. Otherwise, we can just write the functions assuming that torch is
# imported.
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
                     sparse=None, quant=None, unit=Ellipsis):
    return False
@docwrap
@alttorch(_is_never_tensor)
def is_tensor(obj,
              dtype=None, shape=None, ndim=None,
              device=None, requires_grad=None,
              sparse=None, quant=None, unit=Ellipsis, ureg=None):
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
        If the `shape` parameter is not `None`, then the given `obj` must have a
        shape shape that matches the parameter value `sh`. The value `sh` must
        be a tuple that is equal to the `obj`'s shape tuple with the following
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
        Whether `Quantity` objects should be considered valid tensors or not.
        If `quant=True` then `obj` is considered a valid array only when `obj`
        is a quantity object with a `torch` tensor as the magnitude. If `False`,
        then `obj` must be a `torch` tensor itself and not a `Quantity` to be
        considered valid. If `None` (the default), then either quantities or
        `torch` tensors are considered valid.
    unit : unit-like or Ellipsis, optional
        A unit with which the object obj's unit must be compatible in order
        for `obj` to be considered a valid tensor. An `obj` that is not a
        quantity is considered to have a unit of `None`. If `unit=Ellipsis`
        (the default), then the object's unit is ignored.
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
    # Next, check compatibility of the units.
    if unit is None:
        # We are required to not be a quantity.
        if u is not None: return False
    elif unit is not Ellipsis:
        from ._quantity import alike_units
        if not is_tuple(unit): unit = (unit,)
        if not any(alike_units(u, uu) for uu in unit):
            return False
    # Check the match to the numeric collection last.
    if dtype is None and shape is None and ndim is None: return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, dtype)
def to_tensor(obj,
              dtype=None, device=None, requires_grad=None, copy=False,
              sparse=None, quant=None, ureg=None, unit=Ellipsis):
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
    unit : unit-like or boolean or Ellipsis, optional
        The unit that should be used in the return value. When the return value
        of this function is a `Quantity` (see the `quant` parameter), the
        returned quantity always has units matching the `unit` parameter; if
        the provided `obj` is not a quantity, then its unit is presumed to be
        that requested by `unit`. When the return value of this function is
        not a `Quantity` object and is instead a PyTorch tensor object, then
        when `obj` is not a quantity the `unit` parameter is ignored, and when
        `obj` is a quantity, its magnitude is returned after conversion into
        `unit`. The default value of `unit`, `Ellipsis`, indicates that, if
        `obj` is a quantity, its unit should be used, and `unit` should be
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
    if ureg is Ellipsis: from pimms import units as ureg
    dtype = to_torchdtype(dtype)
    # If obj is a quantity, we handle things differently.
    if is_quant(obj):
        q = obj
        obj = q.m
        if ureg is None: ureg = q._REGISTRY
        if quant is None: quant = True
    else:
        q = None
        if ureg is None: from pimms import units as ureg
        if quant is None: quant = False
    # Translate obj depending on whether it's a pytorch tensor already or a
    # scipy sparse matrix.
    if torch.is_tensor(obj):
        if requires_grad is None: requires_grad = obj.requires_grad
        if device is None: device = obj.device
        if copy or requires_grad != obj.requires_grad:
            arr = torch.tensor(obj, dtype=dtype, device=device,
                               requires_grad=requires_grad)
        else:
            arr = torch.as_tensor(obj, dtype=dtype, device=device)
        arr = obj
    elif scipy__is_sparse(obj):
        if requires_grad is None: requires_grad = False
        (rows, cols, vals) = sps.find(obj)
        # Process these into a PyTorch COO matrix.
        ii = torch.tensor([rows, cols], dtype=torch.long, device=device)
        arr = torch.sparse_coo_tensor(ii, vals, obj.shape,
                                      dtype=dtype, device=device,
                                      requires_grad=requires_grad)
        # Convert to a CSR tensor if we were given a CSR matrix.
        if isinstance(obj, sps.csr_matrix): arr = arr.to_sparse_csr()
    elif copy or requires_grad is True:
        arr = torch.tensor(arr, dtype=dtype, device=devce,
                           requires_grad=requires_grad)
    else:
        arr = torch.as_tensor(arr, dtype=dtype, device=device)
    # If there is an instruction regarding the output's sparsity, handle that
    # now.
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
                                          dtype=dtype, device=device,
                                          requires_grad=requires_grad)
    elif sparse is not None:
        raise ValueError(f"invalid value for parameter sparse: {sparse}")
    # Next, we switch on whether we are being asked to return a quantity or not.
    if quant is True:
        if unit is None:
            raise ValueError("to_array: cannot make a quantity (quant=True)"
                             " without a unit (unit=None)")
        if q is None:
            if unit is Ellipsis: unit = None
            q = ureg.Quantity(arr, unit)
        else:
            if unit is Ellipsis: unit = q.u
            if ureg is not q._REGISTRY or obj is not arr:
                q = ureg.Quantity(arr, q.u)
            return q.to(unit)
    elif quant is False:
        # Don't return a quantity, whatever the input argument.
        if unit is Ellipsis:
            # We return the current array/magnitude whatever its unit.
            return arr
        elif q is None:
            # We just pretend this was already in the given unit (i.e., ignore
            # unit).
            return arr
        elif unit is None:
            raise ValueError("cannot extract unit None from quantity; to get"
                             " the native unit, use unit=Ellipsis")
        else:
            if obj is not arr: q = ureg.Quantity(arr, q.u)
            # We convert to the given unit and return that.
            return q.m_as(unit)
    else:
        raise ValueError(f"invalid value for quant: {quant}")

# General Numeric Collection Functions #########################################
@docwrap
def is_numeric(obj,
               dtype=None, shape=None, ndim=None,
               sparse=None, quant=None, unit=Ellipsis, ureg=None):
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
        If the `shape` parameter is not `None`, then the given `obj` must have a
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
    unit : unit-like or Ellipsis, optional
        A unit with which the object obj's unit must be compatible in order for
        `obj` to be considered a valid numerical object. An `obj` that is not a
        quantity is considered to have dimensionless units. If `unit=Ellipsis`
        (the default), then the object's unit is ignored.
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
    if torch__is_tensor(obj):
        return is_tensor(obj,
                         dtype=dtype, shape=shape, ndim=ndim,
                         sparse=sparse, quant=quant, unit=unit, ureg=ureg)
    else:
        return is_array(obj,
                        dtype=dtype, shape=shape, ndim=ndim,
                        sparse=sparse, quant=quant, unit=unit, ureg=ureg)
@docwrap
def to_numeric(obj,
               dtype=None, copy=False,
               sparse=None, quant=None, ureg=None, unit=Ellipsis):
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
    unit : unit-like or boolean or Ellipsis, optional
        The unit that should be used in the return value. When the return value
        of this function is a `Quantity` (see the `quant` parameter), the
        returned quantity always has a unit matching the `unit` parameter; if
        the provided `obj` is not a quantity, then its unit is presumed to be
        those requested by `unit`. When the return value of this function is
        not a `Quantity` object and is instead a numeric object, then
        when `obj` is not a quantity the `unit` parameter is ignored, and when
        `obj` is a quantity, its magnitude is returned after conversion into
        `unit`. The default value of `unit`, `Ellipsis`, indicates that, if
        `obj` is a quantity, its unit should be used, and `unit` should be
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
    if torch__is_tensor(obj):
        return to_tensor(obj,
                         dtype=dtype, shape=shape, ndim=ndim,
                         sparse=sparse, quant=quant, unit=unit, ureg=ureg)
    else:
        return to_array(obj,
                        dtype=dtype, shape=shape, ndim=ndim,
                        sparse=sparse, quant=quant, unit=unit, ureg=ureg)


# Sparse Matrices and Dense Collections#########################################
from scipy.sparse import issparse as scipy__is_sparse
@docwrap
def is_sparse(obj,
              dtype=None, shape=None, ndim=None,
              quant=None, ureg=None, unit=Ellipsis):
    """Returns `True` if an object is a sparse SciPy matrix or PyTorch tensor.

    `is_sparse(obj)` returns `True` if the given object `obj` is an instance of
    one of the SciPy sprase matrix classes, is a sparse PyTorch tensor, or is a
    quantity whose magnintude is one of theese. Additional constraints may be
    placed on the object via the optional argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a sparse numerical object is to be assessed.
    %(pimms.types._core.is_numeric.parameters.dtype)s
    %(pimms.types._core.is_numeric.parameters.ndim)s
    %(pimms.types._core.is_numeric.parameters.shape)s
    %(pimms.types._core.is_numeric.parameters.quant)s
    %(pimms.types._core.is_numeric.parameters.ureg)s
    %(pimms.types._core.is_numeric.parameters.unit)s

    Returns
    -------
    boolean
        `True` if `obj` is a valid sparse numerical object, otherwise `False`.
    """
    return is_numeric(obj, sparse=True,
                      dtype=dtype, shape=shape, ndim=ndim,
                      quant=quant, ureg=ureg, unit=unit)
@docwrap
def to_sparse(obj,
              dtype=None, shape=None, ndim=None,
              quant=None, ureg=None, unit=Ellipsis):
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
    %(pimms.types._core.to_numeric.parameters.dtype)s
    %(pimms.types._core.is_numeric.parameters.ndim)s
    %(pimms.types._core.is_numeric.parameters.shape)s
    %(pimms.types._core.to_numeric.parameters.quant)s
    %(pimms.types._core.to_numeric.parameters.ureg)s
    %(pimms.types._core.to_numeric.parameters.unit)s

    Returns
    -------
    sparse tensor or sparse matrix or quantity with a sparse magnitude
        A sparse version of the argument `obj`.
    """
    return to_numeric(obj, sparse=True,
                      dtype=dtype, shape=shape, ndim=ndim,
                      quant=quant, ureg=ureg, unit=unit)
@docwrap
def is_dense(obj,
             dtype=None, shape=None, ndim=None,
             quant=None, ureg=None, unit=Ellipsis):
    """Returns `True` if an object is a dense NumPy array or PyTorch tensor.

    `is_dense(obj)` returns `True` if the given object `obj` is an instance of
    one of the NumPy `ndarray` classes, is a dense PyTorch tensor, or is a
    quantity whose magnintude is one of theese. Additional constraints may be
    placed on the object via the optional argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a dense numerical object is to be assessed.
    %(pimms.types._core.is_numeric.parameters.dtype)s
    %(pimms.types._core.is_numeric.parameters.ndim)s
    %(pimms.types._core.is_numeric.parameters.shape)s
    %(pimms.types._core.is_numeric.parameters.quant)s
    %(pimms.types._core.is_numeric.parameters.ureg)s
    %(pimms.types._core.is_numeric.parameters.unit)s

    Returns
    -------
    boolean
        `True` if `obj` is a valid dense numerical object, otherwise `False`.
    """
    return is_numeric(obj, sparse=False,
                      dtype=dtype, shape=shape, ndim=ndim,
                      requires_grad=requires_grad, device=device,
                      quant=quant, ureg=ureg, unit=unit)
@docwrap
def to_dense(obj,
             dtype=None, shape=None, ndim=None,
             quant=None, ureg=None, unit=Ellipsis):
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
    %(pimms.types._core.to_numeric.parameters.dtype)s
    %(pimms.types._core.is_numeric.parameters.ndim)s
    %(pimms.types._core.is_numeric.parameters.shape)s
    %(pimms.types._core.to_numeric.parameters.quant)s
    %(pimms.types._core.to_numeric.parameters.ureg)s
    %(pimms.types._core.to_numeric.parameters.unit)s

    Returns
    -------
    dense tensor or dense ndarray or quantity with a dense magnitude
        A dense version of the argument `obj`.
    """
    return to_numeric(obj, sparse=False,
                      dtype=dtype, shape=shape, ndim=ndim,
                      quant=quant, ureg=ureg, unit=unit)

# Strings ######################################################################
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
def _strbinop_prp(a, b, case=True, unicode=None, strip=False):
    if not is_str(a) or not is_str(b): return None
    # We do case normalization when case comparison is *not* requested
    casenorm = not bool(case)
    # When unicode is None, we do its normalization only when case normalization
    # is being done.
    if unicode is None: unicode = casenorm
    # We now perform normalization if it is required.
    if unicode or casenorm:
        a = strnorm(a, case=casenorm, unicode=unicode)
        b = strnorm(b, case=casenorm, unicode=unicode)
    # If we requested stripping, do that now.
    if strip is True:
        a = a.strip()
        b = b.strip()
    elif strip is not False:
        a = a.strip(strip)
        b = b.strip(strip)
    return (a,b)
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
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None: return None
    else: (a, b) = prep
    # If straightforward non-split comparison is requested, we can just return
    # the comparison
    if split is False:
        return (-1 if a < b else 1 if a > b else 0)
    # Otherwise, we split then do a list comparison.
    if split is True:
        a = a.split()
        b = b.split()
    else:
        a = a.split(split)
        b = b.split(split)
    # We do this efficiently, by first comparing elements in order, then
    # lenghts.
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
@docwrap
def strends(a, b, case=True, unicode=None, strip=False):
    """Determines if the string `a` ends with the string `b` or not.

    `strends(a, b)` returns `True` if `a` and `b` are both strings and if `a`
    ends with `b`, subject to the constraints of the parameters.

    Parameters
    ----------
    %(pimms.types._core.strcmp.parameters.case)s
    %(pimms.types._core.strcmp.parameters.unicode)s
    %(pimms.types._core.strcmp.parameters.strip)s

    Returns
    -------
    boolean
        `True` if `a` and `b` are both strings and if `a` ends with `b`,
        subject to the constraints of the optional parameters.
    """
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None: return None
    else: (a, b) = prep
    # Check the ending
    return a.endswith(b)
@docwrap
def strstarts(a, b, case=True, unicode=None, strip=False):
    """Determines if the string `a` starts with the string `b` or not.

    `strstarts(a, b)` returns `True` if `a` and `b` are both strings
    and if `a` starts with `b`, subject to the constraints of the
    parameters.

    Parameters
    ----------
    %(pimms.types._core.strcmp.parameters.case)s
    %(pimms.types._core.strcmp.parameters.unicode)s
    %(pimms.types._core.strcmp.parameters.strip)s

    Returns
    -------
    boolean
        `True` if `a` and `b` are both strings and if `a` starts with `b`,
        subject to the constraints of the optional parameters.
    """
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None: return None
    else: (a, b) = prep
    # Check the beginning.
    return a.startswith(b)
@docwrap
def strissym(s):
    """Determines if the given string is a valid symbol (identifier).

    `strissym(s)` returns `True` if `s` is both a string and a valid identifier.
    Otherwise, it returns `False`.

    See also: `striskey`, `strisvar`
    """
    return is_str(s) and s.isidentifier()
@docwrap
def striskey(s):
    """Determines if the given string is a valid keyword.

    `strissym(s)` returns `True` if `s` is both a string and a valid keyword
    (such as `'if'` or `'while'`). Otherwise, it returns `False`.

    See also: `strissym`, `strisvar`
    """
    from keyword import iskeyword
    return is_str(s) and iskeyword(s)
@docwrap
def strisvar(s):
    """Determines if the given string is a valid variable name.

    `strissym(s)` returns `True` if `s` is both a string and a valid name
    (i.e., a symbol but not a keyword). Otherwise, it returns `False`.

    See also: `strissym`, `striskey`
    """
    from keyword import iskeyword
    return strissym(s) and not iskeyword(s)


# Builtin Python Abstract Types ################################################
from collections.abc import Callable
@docwrap
def is_callable(obj):
    """Returns `True` if an object is a callable object like a function.

    `is_callable(obj)` returns `True` if the given object `obj` is an instance
    of the `collections.abc.Callable` type.

    Note that this function is included in `pimms` for historical
    reasons, but it should not generally be used, as the `callable()`
    function is a built-in Python function that does the same thing.

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
from types import LambdaType
@docwrap
def is_lambda(obj):
    """Returns `True` if an object is a lambda function, otherwise `False`.

    `is_lambda(obj)` returns `True` if the given object `obj` is an instance
    of the `types.LambdaType` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `LambdaType` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `LambdaType`, otherwise `False`.

    """
    return isinstance(obj, LambdaType)
from collections.abc import Sized
@docwrap
def is_sized(obj):
    """Returns `True` if an object implements `len()`, otherwise `False`.

    `is_sized(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.abc.Sized` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Sized` object is to be assessed.

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

    `is_mseq(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.abc.MutableSequence` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `MutableSequence` object is to be
        assessed.

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

    `is_mset(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.abc.MutableSet` type.

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

    `is_mmap(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.abc.MutableMapping` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `MutableMapping` object is to be
        assessed.

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

# Builtin Python Concrete Types ################################################
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

    `is_fset(obj)` is an alias for `is_frozenset(obj)`.

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
is_fset = is_frozenset
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
    """Returns `True` if an object is a `defaultdict` object.

    `is_ddict(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.defaultdict` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `defaultdict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `defaultdict`, otherwise `False`.

    """
    return isinstance(obj, defaultdict)
from frozendict import frozendict
@docwrap
def is_frozendict(obj):
    """Returns `True` if an object is a `frozendict` object.

    `is_frozendict(obj)` returns `True` if the given object `obj` is an instance
    of the `frozendict.frozendict` type.

    `is_fdict(obj)` is an alias for `is_frozendict(obj)`.

    Parameters
    ----------
    obj : object
        The object whose quality as a `frozendict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `frozendict`, otherwise `False`.
    """
    return isinstance(obj, frozendict)
is_fdict = is_frozendict
from frozendict import FrozenOrderedDict
@docwrap
def is_fodict(obj):
    """Returns `True` if an object is a `FrozenOrderedDict` object.

    `is_fodict(obj)` returns `True` if the given object `obj` is an instance of
    the `frozendict.FrozenOrderedDict` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `FrozenOrderedDict` object is to be
        assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `FrozenOrderedDict`, otherwise
        `False`.
    """
    return isinstance(obj, FrozenOrderedDict)
def hashsafe(obj):
    """Returns `hash(obj)` if `obj` is hashable, otherwise returns `None`.

    See also `is_hashable`; note that a fairly reliable test of immutability
    versus mutability for a Python object is whether it is hashable.
    
    Parameters
    ----------
    obj : object
        The object to be hashed.

    Returns
    -------
    int
        If the object is hashable, returns the hashcode.
    None
        If the object is not hashable, returns `None`.
    """
    try:              return hash(obj)
    except TypeError: return None
def is_hashable(obj):
    """Returns `True` if `obj` is hashable and `False` otherwise.

    `is_hashable(obj)` is equivalent to `hashsafe(obj) is not None`.

    Note that in Python, all builtin immutable types are hashable while all
    builtin mutable types are not.
    """
    return hashsafe(obj) is not None
def is_frozen(obj):
    """Returns `True` if an object is a `frozendict`, `frozenset`, or `tuple`.

    `is_frozen(obj)` returns `True` if the given object `obj` is an instance
    of the `frozendict.frozendict`, `frozenset`, or `tuple` types, all of which
    are "frozen" (immutable).

    In addition to being one of the above types, an object is considered frozen
    if it is a `numpy` array whose `'WRITEABLE'` flag has been set to `False`.

    Parameters
    ----------
    obj : object
        The object whose quality as an frozen object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is frozen, otherwise `False`.    
    """
    if   isinstance(obj, is_frozen.frozen_types): return True
    elif isinstance(obj, ndarray): return not obj.flags['WRITEABLE']
    else: return False
is_frozen.frozen_types = (tuple, frozenset, frozendict)
def is_thawed(obj):
    """Returns `True` if an object is a `dict`, `set`, or `list`.

    `is_thawed(obj)` returns `True` if the given object `obj` is an instance of
    the `dict`, `set`, or `list` types, all of which are "thawed" (mutable)
    instances of other frozen types.

    In addition to being one of the above types, an object is considered thawed
    if it is a `numpy` array whose `'WRITEABLE'` flag has been set to `True`.

    Parameters
    ----------
    obj : object
        The object whose quality as an thawed object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is thawed, otherwise `False`.
    """
    if   isinstance(obj, is_thawed.thawed_types): return True
    elif isinstance(obj, ndarray): return obj.flags['WRITEABLE']
    else: return False
is_thawed.thawed_types = (list, set, dict)
def frozenarray(arr, copy=None, subok=False):
    """Returns a copy of the given NumPy array that is read-only.

    If the argument `arr` is already a NumPy `ndarray` with its `'WRITEABLE'`
    flag set to `False`, then `arr` is returned; otherwise, a copy is made, its
    write-flag is set to `False`, and it is returned.

    If `arr` is not a NumPy array, it is first converted into one.

    Parameters
    ----------
    arr : array-like
        The NumPy array to freeze or an object to convert into a frozen NumPy
        array.
    copy : boolean or None, optional
        If `True` or `False`, then forces the array to be copied or not copied;
        if `None`, then a copy is made if the object is writeable and no copy
        is made otherwise.
    subok : boolean, optional
        If True, then sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (defaults to False).

    Returns
    -------
    numpy.ndarray
        A read-only copy of `arr` (or `arr` itself if it is already read-only).

    Raises
    ------
    TypeError
        If `arr` is not a NumPy array.
    """
    if not isinstance(arr, ndarray):
        if copy is None: copy = True
        arr = np.array(arr, copy=copy, subok=subok)
        copy = False
    rw = arr.flags['WRITEABLE']
    if copy is None: copy = rw
    if copy: arr = np.copy(arr, subok=subok)
    if rw: arr.setflags(write=False)
    return arr
def freeze(obj):
    """Converts certain collections into frozen (immutable) versions.

    This function converts `list`s into `tuple`s, `set`s into `frozenset`s, and
    `dict`s into `frozendict`s. If given a NumPy array whose writeable flag is
    set, it returns a duplicate copy that is read-only. If the given object is
    already one of these frozen types, it is returned as-it. Otherwise, an error
    is raised.

    Parameters
    ----------
    obj : object
        The object to be frozen.

    Returns
    -------
    object
        A frozen version of `obj` if `obj` is not already frozen, and `obj`
        if it is already frozen.

    Raises
    ------
    TypeError
        If the type of the object is not recognized.
    """
    if is_frozen(obj):
        return obj
    for (base, conv_fn) in freeze.freeze_types.items():
        if isinstance(obj, base):
            return conv_fn(obj)
    raise TypeError(f"cannot freeze object of type {type(obj)}")
freeze.freeze_types = {list:        tuple,
                       set:         frozenset,
                       ndarray:     frozenarray,
                       OrderedDict: FrozenOrderedDict,
                       dict:        frozendict}
def thaw(obj, copy=False):
    """Converts certain frozen collections into thawed (mutable) versions.

    This function converts `tuple`s into `list`s, `frozenset`s into `set`s, and
    `frozendict`s into `dict`s. If given a NumPy array whose writeable flag is
    unset, it returns a duplicate copy that is writeable. If the given object is
    already one of these thawed types, it is returned. Otherwise, an error is
    raised.

    Parameters
    ----------
    obj : object
        The object to be thawed.

    Returns
    -------
    object : object
        A thawed copy of `obj`. Note that if `obj` is already a thawed object,
        a duplicate is returned if and only if the `copy` option is `True`.
    copy : boolean, optional
        Whether to return a copy of `object` if `object` is already a thawed
        type. The default is `False`, indicating that the object should be
        returned as-is if it is thawed.

    Raises
    ------
    TypeError
        If the type of the object is not recognized.
    """
    if is_thawed(obj):
        return obj.copy() if copy else obj
    for (base, conv_fn) in that.thawed_types.items():
        if isinstance(obj, base):
            return conv_fn(obj)
    raise TypeError(f"cannot thaw object of type {type(obj)}")
thaw.thawed_types = {tuple:             list,
                     frozenset:         set,
                     ndarray:           np.array,
                     FrozenOrderedDict: OrderedDict,
                     frozendict:        dict}
