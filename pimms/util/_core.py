# -*- coding: utf-8 -*-
################################################################################
# pimms/util/_core.py
#
# Core implementation of the utility classes for the various utilities that are
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
def _strbinop_prep(a, b, case=True, unicode=None, strip=False):
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
    boolean or None
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
    %(pimms.util._core.strcmp.parameters)s

    Returns
    -------
    boolean or None
        `True` if `a` and `b` are both strings and if `a` is equal to `b`,
        subject to the constraints of the optional parameters. If either `a` or
        `b` is not a string, then `None` is returned.
    """
    cmpval = strcmp(a, b, case=case, unicode=unicode, strip=strip, split=split)
    return None if cmpval is None else (cmpval == 0)
@docwrap
def strends(a, b, case=True, unicode=None, strip=False):
    """Determines if the string `a` ends with the string `b` or not.

    `strends(a, b)` returns `True` if `a` and `b` are both strings and if `a`
    ends with `b`, subject to the constraints of the parameters.

    Parameters
    ----------
    %(pimms.util._core.strcmp.parameters.case)s
    %(pimms.util._core.strcmp.parameters.unicode)s
    %(pimms.util._core.strcmp.parameters.strip)s

    Returns
    -------
    boolean or None
        `True` if `a` and `b` are both strings and if `a` ends with `b`,
        subject to the constraints of the optional parameters. If either `a` or
        `b` is not a string, then `None` is returned.
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
    %(pimms.util._core.strcmp.parameters.case)s
    %(pimms.util._core.strcmp.parameters.unicode)s
    %(pimms.util._core.strcmp.parameters.strip)s

    Returns
    -------
    boolean or None
        `True` if `a` and `b` are both strings and if `a` starts with `b`,
        subject to the constraints of the optional parameters. If either `a` or
        `b` is not a string, then `None` is returned.
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
    Otherwise, it returns `False` if `s` is a string and `None` if not.

    See also: `striskey`, `strisvar`
    """
    return s.isidentifier() if is_str(s) else None
@docwrap
def striskey(s):
    """Determines if the given string is a valid keyword.

    `strissym(s)` returns `True` if `s` is both a string and a valid keyword
    (such as `'if'` or `'while'`). Otherwise, it returns `False` if `s` is a
    string and `None` if not.

    See also: `strissym`, `strisvar`
    """
    from keyword import iskeyword
    return iskeyword(s) if is_str(s) else None
@docwrap
def strisvar(s):
    """Determines if the given string is a valid variable name.

    `strissym(s)` returns `True` if `s` is both a string and a valid name (i.e.,
    a symbol but not a keyword). Otherwise, it returns `False` if `s` is a
    string and `None` if not.

    See also: `strissym`, `striskey`
    """
    from keyword import iskeyword
    return (None if not is_str(s) else
            False if iskeyword(s) else
            s.isidentifier())


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
def is_mutseq(obj):
    """Returns `True` if an object is a mutable sequence, otherwise `False`.

    `is_mutseq(obj)` returns `True` if the given object `obj` is an instance of
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
def is_mutset(obj):
    """Returns `True` if an object is a mutable set, otherwise `False`.

    `is_mutset(obj)` returns `True` if the given object `obj` is an instance of
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
def is_mutmap(obj):
    """Returns `True` if an object is a mutable mapping, otherwise `False`.

    `is_mutmap(obj)` returns `True` if the given object `obj` is an instance of
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
    of the `collections.abc.Hashable` type. This differs from the `can_hash`
    function, which checks whehter calling `hash` on an object raises an
    exception.

    See also: `can_hash`

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
@docwrap
def hashsafe(obj):
    """Returns `hash(obj)` if `obj` is hashable, otherwise returns `None`.

    See also: `can_hash`, `is_hashable` (note that a fairly reliable test of
    immutability versus mutability for a Python object is whether it is
    hashable).
        
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
@docwrap
def can_hash(obj):
    """Returns `True` if `obj` is safe to hash and `False` otherwise.

    `can_hash(obj)` is equivalent to `hashsafe(obj) is not None`. This differs
    from `is_hashable(obj)` in that `is_hashable` only checks whether `obj` is
    an instance of `Hashable`.

    Note that in Python, all builtin immutable types are hashable while all
    builtin mutable types are not.

    See also: `is_hashable`, `hashsafe`
    """
    return hashsafe(obj) is not None
@docwrap
def itersafe(obj):
    """Returns an iterator or None if the given object is not iterable.

    `itersafe(obj)` is equivalent to `iter(obj)` with the exception that, if
    `obj` is not iterable, it returns `None` instead of raising an exception.

    Parameters
    ----------
    obj : object
        The object to be iterated.

    Returns
    -------
    iterator or None
        If `obj` is iterable, returns `iter(obj)`; otherwise, returns `None`.
    """
    try:              return iter(obj)
    except TypeError: return None
@docwrap
def can_iter(obj):
    """Returns `True` if `obj` is safe to iterate and `False` otherwise.

    `can_iter(obj)` is equivalent to `itersafe(obj) is not None`. This differs
    from `is_iterable(obj)` in that `is_iterable` only checks whether `obj` is
    an instance of `Iterable`.

    Note that in Python, all builtin immutable types are hashable while all
    builtin mutable types are not.

    See also: `is_iterable`, `itersafe`
    """
    return itersafe(obj) is not None
@docwrap
def is_frozen(obj):
    """Returns `True` if an object is a `frozendict`, `frozenset`, or `tuple`.

    `is_frozen(obj)` returns `True` if the given object `obj` is an instance
    of the `frozendict.frozendict`, `frozenset`, or `tuple` types, all of which
    are "frozen" (immutable). If `obj` is a thawed type (`list`, `dict`, or
    `set`), then `False` is returned. Otherwise, `None` is returned.

    In addition to being one of the above types, an object is considered frozen
    if it is a `numpy` array whose `'WRITEABLE'` flag has been set to `False`.

    Parameters
    ----------
    obj : object
        The object whose quality as an frozen object is to be assessed.

    Returns
    -------
    boolean or None
        `True` if `obj` is frozen, `False` if `obj` is thawed, otherwise `None`.
    """
    if   isinstance(obj, is_frozen.frozen_types): return True
    elif isinstance(obj, is_thawed.thawed_types): return False
    elif isinstance(obj, np.ndarray): return not obj.flags['WRITEABLE']
    elif sps.issparse(obj): return not obj.data.flags['WRITEABLE']
    else: return None
is_frozen.frozen_types = (tuple, frozenset, frozendict, FrozenOrderedDict)
@docwrap
def is_thawed(obj):
    """Returns `True` if an object is a `dict`, `set`, or `list`.

    `is_thawed(obj)` returns `True` if the given object `obj` is an instance of
    the `dict`, `set`, or `list` types, all of which are "thawed" (mutable)
    instances of other frozen types. If `obj` is a frozen type (`tuple`,
    `frozendict`, and `frozenset`), then `False` is returned. Otherwise, `None`
    is returned.

    In addition to being one of the above types, an object is considered thawed
    if it is a `numpy` array whose `'WRITEABLE'` flag has been set to `True`.

    Parameters
    ----------
    obj : object
        The object whose quality as an thawed object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is thawed, `False` is `obj` is frozen, and `None`
        otherwise.
    """
    # We check the frozen types first because frozendict objects inherit from
    # dict, so if we don't do it in this order, they come out as thawed.
    if   isinstance(obj, is_frozen.frozen_types): return False
    elif isinstance(obj, is_thawed.thawed_types): return True
    elif isinstance(obj, np.ndarray): return obj.flags['WRITEABLE']
    elif sps.issparse(obj): return obj.data.flags['WRITEABLE']
    else: return None
is_thawed.thawed_types = (list, set, dict, OrderedDict)
@docwrap
def to_frozenarray(arr, copy=None, subok=False):
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
    if sps.issparse(arr):
        if not (arr.data.flags['WRITEABLE'] or copy): return arr
        arr = arr.copy()
        arr.data.flags['WRITEABLE'] = False
    elif not isinstance(arr, np.ndarray):
        arr = np.array(arr, copy=(True if copy is None else copy), subok=subok)
        copy = False
    rw = arr.flags['WRITEABLE']
    if copy is None: copy = rw
    if copy: arr = np.copy(arr, subok=subok)
    if rw: arr.setflags(write=False)
    return arr
to_farray = to_frozenarray
@docwrap
def frozenarray(obj, *args, **kwargs):
    """Equivalent to `numpy.array` but returns read-only arrays.

    `frozenarray(obj)` is equivalent to `numpy.array(obj)` with the exception
    that the returned object is always a frozen array (i.e., an array with the
    `'WRITEABLE'` flag set to `False`). If `copy=False` is requested, but the
    passed array is writeable, then an error is raised; if you wish to convert
    an array to a read-only array without copying, you must use the
    `to_frozenarray` function.

    See Also
    --------
    `numpy.array`
        Create an array that is not frozen by default.
    `to_frozenarray`
        Convert an argument to a frozen array (allows `copy=False`).
    """
    arr = np.array(obj, *args, **kwargs)
    if arr is obj and arr.flags['WRITEABLE']:
        arr = arr.copy()
    arr.setflags(write=False)
    return arr
farray = frozenarray
@docwrap
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
                       np.ndarray:  to_frozenarray,
                       OrderedDict: FrozenOrderedDict,
                       dict:        frozendict}
@docwrap
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
    for (base, conv_fn) in thaw.thawed_types.items():
        if isinstance(obj, base):
            return conv_fn(obj)
    raise TypeError(f"cannot thaw object of type {type(obj)}")
thaw.thawed_types = {tuple:             list,
                     frozenset:         set,
                     np.ndarray:        np.array,
                     FrozenOrderedDict: OrderedDict,
                     frozendict:        dict}
