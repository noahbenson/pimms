# -*- coding: utf-8 -*-
################################################################################
# pimms/util/_numeric.py
#
# General numeric type utilities for pimms.
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
from ..doc     import docwrap
from ._core    import (is_tuple, is_list, is_seq, is_set,
                       is_str, streq, strnorm)
from functools import partial
import pint
import numpy as np
import scipy as sp, scipy.sparse as sps

# Numpy and SciPy Configuration ################################################
_scipy_version = tuple(map(int, sp.version.short_version.split('.')))
_numpy_version = tuple(map(int, np.version.short_version.split('.')))

# PyTorch Configuration ########################################################
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
# Get the torch version setup.
if torch is None:
    _torch_version = (0, 0, 0)
else:
    _torch_version = torch.__version__.split('.')
    try:
        _torch_version = (int(_torch_version[0]),
                          int(_torch_version[1]),
                          int(_torch_version[2]))
    except Exception:
        _torch_version = (int(_torch_version[0]),
                          int(_torch_version[1]),
                          _torch_version[2])

# Numerical Types ##############################################################
from numpy import ndarray
def _is_numtype_wouttorch(numtype, dtype, obj):
    if isinstance(obj, numtype):
        return True
    elif isinstance(obj, ndarray):
        return any(np.issubdtype(obj.dtype, dt) for dt in dtype)
    else:
        return False
def _is_numtype_withtorch(numtype, dtype, obj):
    if isinstance(obj, numtype):
        return True
    elif isinstance(obj, ndarray):
        return any(np.issubdtype(obj.dtype, dt) for dt in dtype)
    elif torch__is_tensor(obj):
        obj_dtype = obj.numpy().dtype
        return any(np.issubdtype(obj_dtype, dt) for dt in dtype)
    else:
        return False
from numbers import Number
_number_npdtype = (np.number, np.bool_)
def _is_number_wouttorch(obj):
    return is_numtype_wouttorch(Number, _number_npdtype, obj)
@alttorch(_is_numtype_wouttorch)
@docwrap
def is_number(obj):
    """Returns `True` if an object is a Python number, otherwise `False`.

    `is_number(obj)` returns `True` if the given object `obj` is an instance of
    the `numbers.Number` type or if it is an instance of a numeric NumPy array
    or PyTorch tensor.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Number` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Number`, otherwise `False`.
    """
    return _is_numtype_withtorch(Number, _number_npdtype, obj)
from numbers import Integral
_integer_npdtype = (np.integer, np.bool_)
def _is_integer_wouttorch(obj):
    return is_numtype_wouttorch(Integral, _integer_npdtype, obj)
@alttorch(_is_integer_wouttorch)
@docwrap
def is_integer(obj):
    """Returns `True` if an object is a Python number, otherwise `False`.

    `is_integer(obj)` returns `True` if the given object `obj` is an instance of
    the `numbers.Integral` type or if it is an instance of a numeric NumPy array
    or PyTorch tensor whose dtype is an integer type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Integral` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Integral`, otherwise `False`.
    """
    return _is_numtype_withtorch(Integral, _integer_npdtype, obj)
from numbers import Real
_real_npdtype = (np.floating, np.integer, np.bool_)
def _is_real_wouttorch(obj):
    return is_numtype_wouttorch(Real, _real_npdtype, obj)
@alttorch(_is_real_wouttorch)
@docwrap
def is_real(obj):
    """Returns `True` if an object is a Python number, otherwise `False`.

    `is_real(obj)` returns `True` if the given object `obj` is an instance of
    the `numbers.Real` type or of a real-valued NumPy array or PyTorch tensor.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Real` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Real`, otherwise `False`.
    """
    return _is_numtype_withtorch(Real, _real_npdtype, obj)
from numbers import Complex
_complex_npdtype = (np.number, np.bool_)
def _is_complex_wouttorch(obj):
    return is_numtype_wouttorch(Complex, _complex_npdtype, obj)
@alttorch(_is_complex_wouttorch)
@docwrap
def is_complex(obj):
    """Returns `True` if an object is a complex number, otherwise `False`.

    `is_complex(obj)` returns `True` if the given object `obj` is an instance of
    the `numbers.Complex` type or an instance of a complex-valued NumPy array or
    PyTorch tensor.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Complex` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Complex`, otherwise `False`.
    """
    return _is_numtype_withtorch(Complex, _complex_npdtype, obj)

# Numerical Collection Suport ##################################################
# Numerical collections include numpy arrays and torch tensors. These objects
# are handled similarly due to their overall functional similarity, and certain
# support functions are used for both.
def _numcoll_match(numcoll_shape, numcoll_dtype, ndim, shape, numel, dtype):
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
    # See if we match in terms of numel, ndim, and shape
    sh = numcoll_shape
    if ndim is not None and len(sh) not in ndim:
        return False
    if numel is not None:
        n = np.prod(sh)
        if is_tuple(numel):
            if n not in numel:
                return False
        elif n != numel:
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
_sparse_types = {'bsr': sps.bsr_matrix,
                 'coo': sps.coo_matrix,
                 'csc': sps.csc_matrix,
                 'csr': sps.csr_matrix,
                 'dia': sps.dia_matrix,
                 'dok': sps.dok_matrix,
                 'lil': sps.lil_matrix}
@docwrap
def is_array(obj,
             dtype=None, shape=None, ndim=None, numel=None, frozen=None,
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
    numel : int or tuple of ints or None, optional
        If the `numel` parameter is not `None`, then the given `obj` must have
        the same number of elements as given by `numel`. If `numel` is a tuple,
        then the number of elements in `obj` must be in the `numel` tuple. The
        number of elements is the product of its shape.
    frozen : boolean or None, optional
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
    if isinstance(obj, pint.Quantity):
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
        mtype = _sparse_types.get(sparse, None)
        if mtype is None:
            raise ValueErroor(f"invalid sparse matrix type: {sparse}")
        if not isinstance(obj, mtype): return False
    else:
        raise ValueErroor(f"invalid sparse parameter: {sparse}")
    # Check that the object is read-only
    if frozen is None:
        pass
    elif frozen is True:
        if scipy__is_sparse(obj):
            if obj.data.flags['WRITEABLE']: return False
        else:
            if obj.flags['WRITEABLE']: return False
    elif frozen is False:
        if scipy__is_sparse(obj):
            if not obj.data.flags['WRITEABLE']: return False
        else:
            if not obj.flags['WRITEABLE']: return False
    else:
        raise ValueError(f"invalid parameter frozen: {frozen}")
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
    if dtype is None and shape is None and ndim is None and numel is None:
        return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, numel, dtype)
def to_array(obj,
             dtype=None, order=None, copy=False, sparse=None, frozen=None,
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
    frozen : boolean or None, optional
        Whether the return value should be read-only or not. If `None`, then no
        changes are made to the return value; if a new array is allocated in the
        `to_array()` function call, then it is returned in a writeable form. If
        `frozen=True`, then the return value is always a read-only array; if
        `obj` is not already read-only, then a copy of `obj` is always returned
        in this case. If `frozen=False`, then the return-value is never
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
    if isinstance(obj, pint.Quantity):
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
        mtype = _sparse_types.get(sparse, None)
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
                vv = np.array(uu, dtype=dtype, order=order, copy=copy)
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
    if frozen is not None:
        if frozen is True:
            if sparse:
                if arr.data.flags['WRITEABLE']:
                    if not newarr: arr = arr.copy()
                    arr.data.setflags(write=False)
            elif arr.flags['WRITEABLE']:
                if not newarr: arr = np.array(arr)
                arr.flags['WRITEABLE'] = False
        elif frozen is False:
            if sparse:
                if not arr.data.flags['WRITEABLE']:
                    if not newarr: arr = arr.copy()
                    arr.data.setflags(write=True)
            elif not arr.flags['WRITEABLE']:
                arr = np.array(arr)
        else:
            raise ValueError(f"bad parameter value for frozen: {frozen}")
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
        try: return None is not torch.from_numpy(np.array([], dtype=dt))
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
    if is_torchdtype(dt):
        return dt
    else:
        return torch.as_tensor(np.array([], dtype=dt)).dtype
def _is_never_tensor(obj,
                     dtype=None, shape=None, ndim=None,
                     device=None, requires_grad=None,
                     sparse=None, quant=None, unit=Ellipsis):
    return False
@docwrap
@alttorch(_is_never_tensor)
def is_tensor(obj,
              dtype=None, shape=None, ndim=None, numel=None,
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
    numel : int or tuple of ints or None, optional
        If the `numel` parameter is not `None`, then the given `obj` must have
        the same number of elements as given by `numel`. If `numel` is a tuple,
        then the number of elements in `obj` must be in the `numel` tuple. The
        number of elements is the product of its shape.
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
    if isinstance(obj, pint.Quantity):
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
    if dtype is None and shape is None and ndim is None and numel is None:
        return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, numel, dtype)
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
    if dtype is not None: dtype = to_torchdtype(dtype)
    # If obj is a quantity, we handle things differently.
    if isinstance(obj, pint.Quantity):
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
        if device is None:        device = obj.device
        if dtype is None:         dtype = obj.dtype
        needs_copy = device != obj.device or dtype != obj.dtype
        prefs_copy = copy or requires_grad != obj.requires_grad
        if copy is False:
            if needs_copy:
                if device == obj.device: msg = "dtype change"
                elif dtype == obj.dtype: msg = "device change"
                else:                    msg = "device and dtype change"
                raise ValueError("copy=False requested,"
                                 f" but copy required by {msg}")
            else: 
                arr = obj
        elif needs_copy:
            arr = obj.to(dtype=dtype, device=device)
        elif copy:
            arr = obj.detach().clone()
        else:
            arr = obj
        if arr.requires_grad != requires_grad:
            arr = arr.requires_grad_(requires_grad)
    elif scipy__is_sparse(obj):
        if requires_grad is None: requires_grad = False
        if dtype is None: dtype = obj.dtype
        (rows, cols, vals) = sps.find(obj)
        # Process these into a PyTorch COO matrix.
        ii = torch.tensor([rows, cols], dtype=torch.long, device=device)
        arr = torch.sparse_coo_tensor(ii, vals, obj.shape,
                                      dtype=dtype, device=device,
                                      requires_grad=requires_grad)
        # Convert to a CSR tensor if we were given a CSR matrix.
        if isinstance(obj, sps.csr_matrix): arr = arr.to_sparse_csr()
    elif (copy or requires_grad is True or 
          (isinstance(obj, np.ndarray) and not obj.flags['WRITEABLE'])):
        arr = torch.tensor(obj, dtype=dtype, device=devce,
                           requires_grad=requires_grad)
        dtype = arr.dtype
    else:
        arr = torch.as_tensor(obj, dtype=dtype, device=device)
        dtype = arr.dtype
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

# General Numeric Collection Functions #########################################
@docwrap
def is_numeric(obj,
               dtype=None, shape=None, ndim=None, numel=None,
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
    numel : int or tuple of ints or None, optional
        If the `numel` parameter is not `None`, then the given `obj` must have
        the same number of elements as given by `numel`. If `numel` is a tuple,
        then the number of elements in `obj` must be in the `numel` tuple. The
        number of elements is the product of its shape.
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
                         dtype=dtype, shape=shape, ndim=ndim, numel=numel,
                         sparse=sparse, quant=quant, unit=unit, ureg=ureg)
    else:
        return is_array(obj,
                        dtype=dtype, shape=shape, ndim=ndim, numel=numel,
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
                         dtype=dtype, sparse=sparse,
                         quant=quant, unit=unit, ureg=ureg)
    else:
        return to_array(obj,
                        dtype=dtype, sparse=sparse,
                        quant=quant, unit=unit, ureg=ureg)


# Sparse Matrices and Dense Collections#########################################
from scipy.sparse import issparse as scipy__is_sparse
@docwrap
def is_sparse(obj,
              dtype=None, shape=None, ndim=None, numel=None,
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
    %(pimms.util._numeric.is_numeric.parameters.dtype)s
    %(pimms.util._numeric.is_numeric.parameters.ndim)s
    %(pimms.util._numeric.is_numeric.parameters.shape)s
    %(pimms.util._numeric.is_numeric.parameters.numel)s
    %(pimms.util._numeric.is_numeric.parameters.quant)s
    %(pimms.util._numeric.is_numeric.parameters.ureg)s
    %(pimms.util._numeric.is_numeric.parameters.unit)s
    sparsetype : 'matrix' or 'array' or None, optional
        The kind of sparse array to accept: either `'matrix'` for the scipy
        sparse matrix types (e.g., `scipy.sparse.csr_matrix`) or `'array'` for
        the sparse array types (e.g., `scipy.sparse.csr_array`). If the value is
        `None` (the default) then either is accepted.

    Returns
    -------
    boolean
        `True` if `obj` is a valid sparse numerical object, otherwise `False`.
    """
    return is_numeric(obj, sparse=True,
                      dtype=dtype, shape=shape, ndim=ndim, numel=numel,
                      quant=quant, ureg=ureg, unit=unit)
@docwrap
def to_sparse(obj,
              dtype=None, quant=None, ureg=None, unit=Ellipsis):
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
    %(pimms.util._numeric.to_numeric.parameters.dtype)s
    %(pimms.util._numeric.to_numeric.parameters.quant)s
    %(pimms.util._numeric.to_numeric.parameters.ureg)s
    %(pimms.util._numeric.to_numeric.parameters.unit)s

    Returns
    -------
    sparse tensor or sparse matrix or quantity with a sparse magnitude
        A sparse version of the argument `obj`.
    """
    return to_numeric(obj, sparse=True,
                      dtype=dtype, quant=quant,
                      ureg=ureg, unit=unit)
@docwrap
def is_dense(obj,
             dtype=None, shape=None, ndim=None, numel=None,
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
    %(pimms.util._numeric.is_numeric.parameters.dtype)s
    %(pimms.util._numeric.is_numeric.parameters.ndim)s
    %(pimms.util._numeric.is_numeric.parameters.shape)s
    %(pimms.util._numeric.is_numeric.parameters.numel)s
    %(pimms.util._numeric.is_numeric.parameters.quant)s
    %(pimms.util._numeric.is_numeric.parameters.ureg)s
    %(pimms.util._numeric.is_numeric.parameters.unit)s

    Returns
    -------
    boolean
        `True` if `obj` is a valid dense numerical object, otherwise `False`.
    """
    return is_numeric(obj, sparse=False,
                      dtype=dtype, shape=shape, ndim=ndim, numel=numel,
                      quant=quant, ureg=ureg, unit=unit)
@docwrap
def to_dense(obj,
             dtype=None, quant=None, ureg=None, unit=Ellipsis):
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
    %(pimms.util._numeric.to_numeric.parameters.dtype)s
    %(pimms.util._numeric.to_numeric.parameters.quant)s
    %(pimms.util._numeric.to_numeric.parameters.ureg)s
    %(pimms.util._numeric.to_numeric.parameters.unit)s

    Returns
    -------
    dense tensor or dense ndarray or quantity with a dense magnitude
        A dense version of the argument `obj`.
    """
    return to_numeric(obj, sparse=False,
                      dtype=dtype, quant=quant, ureg=ureg, unit=unit)

# Scalar Utilities #############################################################
def is_scalar(obj):
    """Determines whether the argument is a scalar number or not.

    `is_scalar(x)` returns `True` if `x` is a scalar number and false otherwise.

    See also: `like_scalar`
    """
    # Fundamentally, scalar objects are just numbers.
    return isinstance(obj, Number)
def _like_scalar_wouttorch(obj):
    if isinstance(obj, Number):
        return True
    if not isinstance(obj, np.ndarray):
        try:
            obj = np.asarray(obj)
        except TypeError:
            return False
    return (obj.size == 1 and 
            any(np.issubdtype(obj.dtype, tt) for tt in _number_npdtype))
@alttorch(_like_scalar_wouttorch)
def like_scalar(obj):
    """Determines whether the argument holds a scalar value or not.

    `like_scalar(x)` returns `True` if `x` is already a scalar number, if `x` is
    a single-element numpy array or tensor, or if `x` is a sequence or set that
    has only one numerical element; otherwise, it returns `False`.

    If `like_scalar(x)` returns `True`, then `to_scalar(x)` will always return a
    valid Python number (i.e., an object of type `numbers.Number`).

    See also: `is_scalar`, `to_scalar`
    """
    if isinstance(obj, Number):
        return True
    if torch__is_tensor(obj):
        return torch.numel(obj) == 1
    if not isinstance(obj, np.ndarray):
        try:
            obj = np.asarray(obj)
        except TypeError:
            return False
    return (obj.size == 1 and 
            any(np.issubdtype(obj.dtype, tt) for tt in _number_npdtype))
def _to_scalar_wouttorch(obj):
    if isinstance(obj, Number):
        return obj
    elif isinstance(obj, np.ndarray):
        if (obj.size == 1 and
            any(np.issubdtype(obj.dtype, tt) for tt in _number_npdtype)):
            return obj.item()
    else:
        try:
            u = np.asarray(obj)
            return _to_scalar_wouttorch(u)
        except TypeError:
            pass
    raise TypeError(f"given object is not scalar-like: {obj}")
def to_scalar(obj):
    """Converts the argument into a simple Python number.

    `to_scalar(x)` returns a simple Python number representation of `x` (in
    other words, `x` will be a subtype of Python's `numbers.Number` type). Any
    number, any NumPy array with only one element, and any PyTorch tensor with
    only one element can be converted into a scalar.

    Parameters
    ----------
    obj : object
        The object that is to be converted into a scalar.

    Returns
    -------
    number
        A scalar number that is an object whose class is a subtype of
        `numbers.Number`.

    Raises
    ------
    TypeError
        If the argument is not like a scalar object.
    """
    if isinstance(obj, Number):
        return obj
    elif isinstance(obj, np.ndarray):
        if (obj.size == 1 and 
            any(np.issubdtype(obj.dtype, tt) for tt in _number_npdtype)):
            return obj.item()
    elif torch__is_tensor(obj):
        if torch.numel(obj) == 1:
            return obj.item()
    else:
        try:
            u = np.asarray(obj)
            return to_scalar(u)
        except TypeError:
            pass
    raise TypeError(f"given object is not scalar-like: {obj}")
