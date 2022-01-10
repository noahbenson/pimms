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
            if is_sequence(dtype) or is_set(dtype): dtype = [to_numpydtype(dt) for dt in dtype]
            else: dtype = [to_numpydtype(dtype)]
        elif is_torchdtype(numcoll_dtype):
            if is_sequence(dtype) or is_set(dtype): dtype = [to_torchdtype(dt) for dt in dtype]
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
    elif is_torchdtype(dt): return torch.tensor([], dtype=dt).numpy().dtype
    else: return np.dtype(dt)
from numpy import ndarray
@docwrap
def is_numpyarray(obj, dtype=None, shape=None, ndim=None):
    """Returns `True` if an object is a `numpy.ndarray` object, else `False`.

    `is_numpyarray(obj)` returns `True` if the given object `obj` is an instance
    of the `numpy.ndarray`. Additional constraints may be placed on the object
    via the optional argments.

    Note that the `is_numpyarray()` function is intended as a simple test of
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
        dtype is any of the dtypes in `dtype`.
    ndim : int or tuple or ints or None, optional
        The number of dimensions that the object must have in order to be
        considered a valid numpy array. If `None`, then any number of dimensions
        is acceptable (this is the default). If this is an integer, then the
        number of dimensions must be exactly that integer. If this is a list or
        tuple of integers, then the dimensionality must be one of these numbers.
    shape : int or tuple of ints or None, optional
        If the `shape` parameter is not `None`, then the given `obj` must have a shape
        shape that matches the parameter value `sh`. The value `sh` must be a
        tuple that is equal to the `obj`'s shape tuple with the following
        additional rules: a `-1` value in the `sh` tuple will match any value in
        the `obj`'s shape tuple, and a single `Ellipsis` may appear in `sh`,
        which matches any number of values in the `obj`'s shape tuple. The
        default value of `None` indicates that no restriction should be applied
        to the `obj`'s shape.

    Returns
    -------
    boolean
        `True` if `obj` is a valid numpy array, otherwise `False`.
    """
    # First, let's do the easy things
    if not isinstance(obj, ndarray): return False
    if dtype is None and shape is None and ndim is None: return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, dtype)

# PyTorch Tensors ##################################################################################
# If PyTorch isn't imported, that's fine, we just write our methods to generate errors.
try:
    import torch
    @docwrap(indent=8)
    def is_torchtype(dt):
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
    @docwrap(indent=8)
    def like_torchdtype(dt):
        """Returns `True` for any object that can be converted into a `torch.dtype`.
        
        `like_torchdtype(obj)` returns `True` if the given object `obj` is an
        instance of the `torch.dtype` class, is a string that names a
        `torch.dtype` object, or is a `numpy.dtype` object that is compatible
        with PyTorch. Note that `None` is equivalent to `torch`'s default dtype.
        
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
    #TODO #here -- finish the below pytorch functions
    @docwrap(indent=8)
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
    @docwrap(indent=8)
    def is_torchtensor(obj, dtype=None, shape=None, ndim=None):
        """Returns `True` if an object is a `torch.tensor` object, else `False`.
    
        `is_torchtensor(obj)` returns `True` if the given object `obj` is an
        instance of the `numpy.ndarray`. Additional constraints may be placed on the
        object via the optional argments.
    
        Note that the `is_torchtensor()` function is intended as a simple test of
        whether an object is a PyTorch tennsor; it does not recognize `Quantity`
        objects; for that, see `is_tensor` and `is_numeric`.
    
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
            If the `shape` parameter is not `None`, then the given `obj` must have a shape
            shape that matches the parameter value `sh`. The value `sh` must be a
            tuple that is equal to the `obj`'s shape tuple with the following
            additional rules: a `-1` value in the `sh` tuple will match any value in
            the `obj`'s shape tuple, and a single `Ellipsis` may appear in `sh`,
            which matches any number of values in the `obj`'s shape tuple. The
            default value of `None` indicates that no restriction should be applied
            to the `obj`'s shape.
    
        Returns
        -------
        boolean
            `True` if `obj` is a valid numpy array, otherwise `False`.
        """
        # First, let's do the easy things
        if not torch.is_tensor(obj): return False
        if dtype is None and shape is None and ndim is None: return True
        return _numcoll_match(obj.shape, obj.dtype, ndim, shape, dtype)
except ModuleNotFoundError:
    Tensor = None
    torch_is_tensor = None
    toorch_dtype = None
    def is_torchtensor(obj, dtype=None, shape=None, ndim=None):
        """Returns `False` (`torch` was not found)."""
        return False
    


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
