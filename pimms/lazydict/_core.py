# -*- coding: utf-8 -*-
################################################################################
# pimms/lazydict/_core.py
#
# Declaration of the pimms lazy dictionary type.
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
import inspect, types, sys, os, warnings, weakref
import collections.abc as colls_abc
import numpy as np
import frozendict as fd

from functools import (reduce, partial)

from ..doc import docwrap
from ..util import (is_str, is_fdict, is_map, is_mutmap, is_set, is_lambda)

# #delay #######################################################################
class DelayError(RuntimeError):
    """A runtime error that occurs while evaluating a delay function.

    See also: `delay`.
    """
    def __init__(self, partial):
        self.partial = partial
    def __str__(self):
        (fn, args, kwargs) = self.partial
        fnname = getattr(fn, '__name__', '<anonymous>')
        errmsg = ('delay raised error during call to '
                  f'{fnname}{tuple(args)}')
        if len(kwargs) > 0:
            opts = ', '.join([f'{k}={v}' for (k,v) in kwargs.items()])
            errmsg = f'{errmsg[:-1]}, {opts})'
        return errmsg
    def __repr__(self):
        return str(self)
class delay:
    """A delayed computation that can be either weak or strong.

    `delay(fn)` constructs a `delay` object, which is essentially a
    fully-specified partial object (i.e., a function with a fully-formed
    argument lists) that caches the return value of the respective partial
    function after it is called once and remembers that value from then on.

    Once a `delay` has been constucted, its value can be retrieved by calling
    it: `d = delay(fn)` followed by `d()`.

    Once a `delay` object's value has been calculated, it loses track of its
    function and arguments so that the memory can be recovered.

    Parameters
    ----------
    func : callable
        The function whose value is to be cached when requested.
    *args
        The arguments to be passed to the given function `fn`.
    **kw
        The keyword arguments to be passed to the given function `fn`.
    """
    __slots__ = ('_partial', '_status')
    def __new__(cls, func, *args, **kw):
        # Make sure func is callable.
        if not callable(func):
            raise TypeError("delay functions must be callable")
        self = object.__new__(cls)
        # We set _partial to None once we've run the delay (so that all the args
        # can be garbage-collected). Until then, _results is the exception that
        # traces back to here.
        args = tuple(args)
        partial = (func, args, kw)
        object.__setattr__(self, '_partial', partial)
        # Here we create the exception that we will raise if an error occurs
        # during the function's execution.
        try:
            raise DelayError(partial)
        except DelayError as e:
            object.__setattr__(self, '_status', e)
        return self
    def __call__(self):
        # If _partial is None, then we know that the _result contains the result
        # value; otherwise, we need to calculate it first.
        if self._partial is not None:
            # We need to run the calculation and save its result.
            (fn, args, kw) = self._partial
            try:
                val = fn(*args, **kw)
            except Exception:
                # This raises the exception that was saved during __init__,
                # which will trace back to the origin of the delayed function.
                raise self._status
            # Save the results.
            object.__setattr__(self, '_status', val)
            # Forget the partial info so that it can be garbage collected.
            object.__setattr__(self, '_partial', None)
        # Finally, return the value
        return self._status
    def __repr__(self):
        s = 'cached' if self.is_cached() else 'delayed'
        return f"delay(<{id(self)}>, {s})"
    def __str__(self):
        return f"delay(<{id(self)}>)"
    def is_cached(self):
        """Returns `True` if the delay value is cached and `False` otherwise.

        Returns
        -------
        boolean
            `True` if the given delay object has cached its value and `False`
            otherwise.
        """
        return self._partial is None
@docwrap
def is_delay(obj):
    """Determines if an object is a `delay` or not.

    `delay` objects are specific to the `pimms` library; however `dask`'s
    `Delayed` type is also supported by `pimms`: see `is_delayed`,
    `is_dask_delayed`, and `undelay`.

    Parameters
    ----------
    obj : object
        The object whose quality as a `delay` is to be determined.

    Returns
    -------
    boolean
        `True` if `obj` is a `delay` object and `False` if not.
    """
    return isinstance(obj, delay)
@docwrap
def is_dask_delayed(obj):
    """Determines if an object is a `dask.delay.Delayed` object or not.

    `delay` objects are specific to the `pimms` library; however `dask`'s
    `Delayed` type is also supported by `pimms`: see `is_delay`,
    `is_delayed`, and `undelay`.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Delayed` object is to be determined.

    Returns
    -------
    boolean
        `True` if `obj` is a `Delayed` object and `False` if not.
    """
    try:
        from dask.delayed import Delayed
        return isinstance(obj, Delayed)
    except (ModuleNotFoundError, ImportError):
        return False
@docwrap
def is_delayed(obj):
    """Determines if an object is a `delay` or `dask.delayed.Delayed` object.

    `delay` objects are specific to the `pimms` library; however `dask`'s
    `Delayed` type is also supported by `pimms`: see `is_delay`,
    `is_dask_delayed`, and `undelay`.

    Parameters
    ----------
    obj : object
        The object whose quality as a `delay` or `Delayed` object is to be
        determined.

    Returns
    -------
    boolean
        `True` if `obj` is a `Delayed` object or a `delay` object and `False` if
        not.
    """
    return is_delay(obj) or is_dask_delayed(obj)
@docwrap
def undelay(obj, dask=True):
    """Either returns the argument or the delayed value if it is a `delay`.

    `undelay(obj)` is equivalent to `obj() if is_delay(obj) else obj` unless
    `obj` is a `dask.delayed.Delayed` object, in which case `obj,compute()` is
    used instead of `obj()`.

    Parameters
    ----------
    obj : object
        The object that should be converted from a `delay`.
    dask : boolean
        If `True`, then treats `dask.Delayed` objects as `delay` objects and
        undelays them as well. If `False`, then undelays only `delay` objects.

    Returns
    -------
    object
        Either `obj`, if `obj` is not a `delay` instance, or `obj()` if it is a
        `Delay` object.
    """
    if   isinstance(obj, delay):        return obj()
    elif dask and is_dask_delayed(obj): return obj.compute()
    else:                               return obj

# #frozendict ##################################################################
# The pimms frozendict is a thin wrapper around frozendict.frozendict. The
# entire purpose of this wrapper is to improve the string conversion methods.
class frozendict(fd.frozendict):
    """A persistent dict type based on `frozendict.frozendict`.

    The `pimms.frozendict` type is a thin wrapper around the type
    `frozendict.frozendict`. The purpose of this wrapper is to improve the
    conversion of `frozendict`s into strings.

    `pimms.fdict` is an alias for `pimms.frozendict`.
    """
    _empty = None
    @classmethod
    def _prefix(self):
        return '<'
    @classmethod
    def _suffix(self):
        return '>'
    def __repr__(self):
        s = dict.__repr__(self)
        return self._prefix() + s + self._suffix()
    def __new__(cls, *args, **kwargs):
        if len(kwargs) == 0:
            if len(args) == 0:
                return cls._empty
            elif len(args) == 1:
                arg0 = args[0]
                if type(arg0) is cls:
                    return arg0
                elif len(arg0) == 0:
                    return cls._empty
        return fd.frozendict.__new__(cls, *args, **kwargs)
frozendict._empty = fd.frozendict.__new__(frozendict)
fdict = frozendict
# Now that we've made this frozendict type, we want to update the freeze
# function's version of a frozen dict to this one.
from ..util import freeze
freeze.freeze_types[dict] = frozendict

# #lazydict ####################################################################
class lazydict_keys(colls_abc.KeysView, colls_abc.Set):
    __slots__ = ('_lazydict_keys', '_select_lazy', '_filter', '_count')
    @staticmethod
    def _make_filter(cls, ld, sel_lazy):
        if   sel_lazy is False: return ld.is_eager
        elif sel_lazy is True:  return ld.is_lazy
        else:                   return None
    @staticmethod
    def _counter(x):
        return len(list(iter(x)))
    @staticmethod
    def _make_count(x):
        return delay(lazydict_keys._counter, x)
    def __new__(cls, ld, lazy=None):
        if not isinstance(ld, lazydict):
            raise ValueError("can only make lazydict_keys object from lazydict")
        self = object.__new__(cls)
        object.__setattr__(self, '_lazydict_keys', fdict.keys(ld))
        lazy   = None if lazy   is None else bool(lazy)
        filt   = lazydict_keys._make_filter(cls, ld, lazy)
        count  = lazydict_keys._make_count(self)
        object.__setattr__(self, '_select_lazy', lazy)
        object.__setattr__(self, '_filter', filt)
        object.__setattr__(self, '_count', count)
        return self
    def __init__(self, ld, lazy=None):
        colls_abc.KeysView.__init__(self, ld)
        colls_abc.Set.__init__(self)
    def __iter__(self):
        return filter(self._filter, iter(self._lazydict_keys))
    def __reversed__(self):
        return filter(self._filter, reversed(self._lazydict_keys))
    def __contains__(self, k):
        return (k in self._lazydict_keys and self._filter(k))
    def __len__(self):
        return self._count()
class lazydict_items(colls_abc.ItemsView, colls_abc.Set):
    __slots__ = ('_lazydict', '_lazydict_items',
                 '_select_lazy', '_select_raw',
                 '_filter', '_count')
    @staticmethod
    def _make_filter(cls, ld, sel_lazy):
        if   sel_lazy is False: return lambda kv: ld.is_eager(kv[0])
        elif sel_lazy is True:  return lambda kv: ld.is_lazy(kv[0])
        else:                   return None
    def _undelay(self, kv):
        if isinstance(kv[1], delay):
            k = kv[0]
            return (k, self._lazydict[k])
        else:
            return kv
    def __new__(cls, ld, lazy=None, raw=False):
        if not isinstance(ld, lazydict):
            raise ValueError("can only make lazydict_items object from"
                             " a lazydict")
        self = object.__new__(cls)
        object.__setattr__(self, '_lazydict', ld)
        object.__setattr__(self, '_lazydict_items', fdict.items(ld))
        lazy   = None if lazy   is None else bool(lazy)
        filt   = lazydict_items._make_filter(cls, ld, lazy)
        count  = lazydict_keys._make_count(self)
        object.__setattr__(self, '_select_lazy', lazy)
        object.__setattr__(self, '_select_raw', bool(raw))
        object.__setattr__(self, '_filter', filt)
        object.__setattr__(self, '_count', count)
        return self
    def __init__(self, ld, lazy=None, raw=None):
        colls_abc.ItemsView.__init__(self, ld)
        colls_abc.Set.__init__(self)
    def __iter__(self):
        it = filter(self._filter, iter(self._lazydict_items))
        if self._select_raw:
            return it
        else:
            return map(self._undelay, it)
    def __reversed__(self):
        return filter(self._filter, reversed(self._lazydict_items))
    def __contains__(self, k):
        return (k in self._lazydict and self._filter(k))
    def __len__(self):
        return self._count()
class lazydict_values(colls_abc.ValuesView, colls_abc.Sequence):
    __slots__ = ('_lazydict', '_lazydict_items',
                 '_select_lazy', '_select_cached', '_select_raw',
                 '_filter', '_count')
    @staticmethod
    def _take_value(kv):
        return kv[1]
    def _undelay(self, kv):
        v = kv[1]
        if isinstance(v, delay):
            return self._lazydict[kv[0]]
        else:
            return v
    def __new__(cls, ld, lazy=None, raw=False):
        if not isinstance(ld, lazydict):
            raise ValueError("can only make lazydict_values object from"
                             " lazydict")
        self = object.__new__(cls)
        object.__setattr__(self, '_lazydict', ld)
        object.__setattr__(self, '_lazydict_items', fdict.items(ld))
        lazy   = None if lazy   is None else bool(lazy)
        filt   = lazydict_items._make_filter(cls, ld, lazy)
        count  = lazydict_keys._make_count(self)
        object.__setattr__(self, '_select_lazy', lazy)
        object.__setattr__(self, '_select_raw', bool(raw))
        object.__setattr__(self, '_filter', filt)
        object.__setattr__(self, '_count', count)
        return self
    def __init__(self, ld, lazy=None, raw=None):
        colls_abc.ValuesView.__init__(self, ld)
        colls_abc.Sequence.__init__(self)
    def __getitem__(self, k):
        raise TypeError("lazydict_values is not subscriptable")
    def __iter__(self):
        it = filter(self._filter, iter(self._lazydict_items))
        if self._select_raw:
            return map(lazydict_values._take_value, it)
        else:
            return map(self._undelay, it)
    def __reversed__(self):
        it = filter(self._filter, reversed(list(self._lazydict_items)))
        if self._select_raw:
            return map(lazydict_values._take_value, it)
        else:
            return map(self._undelay, it)
    def __contains__(self, v):
        return (k in self._lazydict and self._filter(k))
    def __len__(self):
        return self._count()
class lazydict(frozendict):
    """A frozen dictionary types whose values can be lazily computed.

    `lazydict` is an immutable dictionary that is identical to the `frozendict`
    type except that when a key is assigned a value that is a delay object, that
    delay object's stored valuee is returned instead of the delay itself.

    `ldict` is an alias for `lazydict`.
    """
    _empty = None
    @classmethod
    def _prefix(self):
        return '<:'
    @classmethod
    def _suffix(self):
        return ':>'
    def __new__(cls, *args, **kw):
        if len(kw) == 0:
            if len(args) == 1:
                if isinstance(args[0], lazydict):
                    return args[0]
                elif len(args[0]) == 0:
                    return cls._empty
            elif len(args) == 0:
                return cls._empty
        # If the first positional argument is a lazydict, we preserve its lazy
        # values by turning it into an iterator of rawitems.
        if len(args) > 0 and isinstance(args[0], lazydict):
            args = (args[0].rawitems(),) + args[1:]
        return frozendict.__new__(cls, *args, **kw)
    def __repr__(self):
        s = [f'{repr(k)}: {"<lazy>" if self.is_lazy(k) else repr(self[k])}'
             for k in self.keys()]
        s = ', '.join(s)
        return self._prefix() + '{' + s + '}' + self._suffix()
    def __str__(self):
        s = [f'{repr(k)}: {"<lazy>" if self.is_lazy(k) else repr(self[k])}'
             for k in self.keys()]
        s = ', '.join(s)
        return self._prefix() + '{' + s + '}' + self._suffix()
    def __hash__(self):
        # We have to undelay everything for this to work.
        for (k,v) in self.items(): pass
        # Now hash ourselves.
        return fd.frozendict.__hash__(self)
    def __eq__(self, other):
        return (len(self) == len(other) and
                self.keys() == other.keys() and
                all(v == other[k] for (k,v) in self.items()))
    def __getitem__(self, k):
        d = frozendict.__getitem__(self, k)
        if isinstance(d, delay):
            d = d()
            dict.__setitem__(self, k, d)
        return d
    def get(self, k, default=None):
        d = frozendict.get(self, k, default=default)
        if isinstance(d, delay):
            d = d()
            dict.__setitem__(self, k, d)
        return d
    def raw(self):
        """Returns a frozendict copy of the lazydict without undelaying.

        The raw version of a lazydict is the underlying frozendict, which may
        contain `delay` objects. In the lazydict, these delay objects are
        automatically undelayed when accessed.
        """
        return frozendict(self.rawitems())
    def __iter__(self):
        # We have to define __iter__ or low-level C-API routines will assume
        # that accessing the values directly (and thus obtaining the delay
        # objects sometimes) is working as intended.
        return frozendict.__iter__(self)
    def keys(self, lazy=None):
        return lazydict_keys(self, lazy=lazy)
    def items(self, lazy=None):
        return lazydict_items(self, lazy=lazy)
    def rawitems(self, lazy=None):
        """Iterates over the raw items in the frozendict.

        Raw items include delay objects rather than the values encapsulaetd by
        the delay.
        """
        return lazydict_items(self, lazy=lazy, raw=True)
    def values(self, lazy=None):
        return lazydict_values(self, lazy=lazy)
    def rawvalues(self, lazy=None):
        """Iterates over the raw values in the frozendict.

        Raw values include delay objects rather than the values encapsulaetd by
        the delay.
        """
        return lazydict_values(self, lazy=lazy, raw=True)
    def set(self, key, val):
        d = lazydict(self.rawitems())
        dict.__setitem__(d, key, val)
        return d
    def setdefault(self, key, default=None):
        if key in self: return self
        else:           return self.set(key, default)
    def delete(self, key):
        if key not in self: raise KeyError(key)
        return lazydict({(k,v) for (k,v) in self.rawitems() if k != key})
    def getdelay(self, k):
        """Returns the `Delay` object for a key if the key is mapped to one.

        `ld.getdelay(k)` returns `None` if the key `k` is not lazy; otherwise,
        returns the function that calculates the value for `k`. If a value has
        already been cached, then `None` is returned.

        Parameters
        ----------
        k : object
            The key whose delay is being looked up.

        Returns
        -------
        Delay or None
            If `k` is in the lazydict and is mapped to a `Delay` object, then
            that `Delay` object is returned. Otherwise, `None` is returned.
        """
        v = frozendict.__getitem__(self, k)
        return v if is_delay(v) else None
    def is_lazy(self, k):
        """Determines if the given key is associated with a lazy value.

        `ld.is_lazy(k)` returns `True` if the given key `k` is in the given lazy
        map and is currently a lazy key/value. If `ld` does not contain `k` then
        an error is raised. If `k` is not lazy then `False` is returned. Note
        that if `k` has been calculated, it is no longer considered lazy.

        Parameters
        ----------
        k : object
            The key whose laziness is to be assessed.

        Returns
        -------
        boolean
            If `k` is in the lazydict and is mapped to a lazy value, then `True`
            is returned; if `k` is mapped to a non-lazy value, then `False` is
            returned.

        Raises
        ------
        KeyError
            If the key `k` is not in the lazydict.
        """
        v = frozendict.__getitem__(self, k)
        return is_delay(v)
    def is_eager(self, k):
        """Determines if the given key is associated with a non-lazy value.

        `ld.is_eager(k)` returns `True` if the given key `k` is in the given
        lazydict but is not a lazy key/value. If `ld` does not contain `k` then
        an error is raised. If `k` is lazy then `False` is returned.

        Parameters
        ----------
        k : object
            The key whose eagerness is to be assessed.

        Returns
        -------
        boolean
            If `k` is in the lazydict and is mapped to a non-lazy value, then
            `True` is returned; if `k` is mapped to a lazy value, then `False`
            is returned.

        Raises
        ------
        KeyError
            If the key `k` is not in the lazydict.
        """
        v = frozendict.__getitem__(self, k)
        return not is_delay(v)
    # In order to support pickling, we define the following; note that these
    # methods pickle the data by casting it to a plain dict, which has the
    # (intended) effect of dereferencing all the delays.
    def __getstate__(self):
        return dict(self)
lazydict._empty = fd.frozendict.__new__(lazydict)
colls_abc.Mapping.register(lazydict)
colls_abc.Hashable.register(lazydict)
ldict = lazydict
def is_lazydict(m):
    """Returns `True` if the given object is a lazydict, otherwise `False`.

    `is_lazydict(m)` returns `True` if `m` is an instance of `lazydict` and
    `False` otherwise.

    `is_ldict` is an alias for `is_lazydict`.

    Parameters
    ----------
    m : object
        The object whose quaality as a lazy map is to be determined.

    Returns
    -------
    boolean
        `True` if `m` is a lazy map and `False` otherwise.
    """
    return isinstance(m, lazydict)
is_ldict = is_lazydict
def _lazyvalmap_undelay(f, v, *args, **kw):
    return f(undelay(v), *args, **kw)
def lazyvalmap(f, d, *args, **kwargs):
    """Returns a dict object whose values are transformed by a function.

    `lazyvalmap(f, d)` returns a dict whose keys are the same as those of the
    given dict object and whose values, for each key `k` are `f(d[k])`. All
    values are created lazily.

    `lazyvalmap(f, d, *args, **kw)` additionally passes the given arguments to
    the function `f`, such that in the resulting map, each key `k` is mapped to
    `f(d[k], *args, **kw)`.

    This function always returns a lazydict object.
    """
    if is_ldict(d):
        d = {k: delay(_lazyvalmap_undelay, v, *args, **kwargs)
             for (k,v) in d.rawitems()}
    else:
        d = {k: delay(f, v, *args, **kwargs)
             for (k,v) in d.items()}
    return ldict(d)
def valmap(f, d, *args, **kwargs):
    """Returns a dict object whose values are transformed by a function.

    `valmap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(d[k])`.

    `valmap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(d[k], *args, **kw)`.

    Unlike `lazyvalmap`, this function returns either a dict, a frozendict, or a
    lazydict depending on the input argument `d`. If `d` is a lazydict, then a
    lazydict is returned; if `d` is a frozendict, a frozendict is returned, and
    otherwise, a dict is returnd.
    """
    if is_ldict(d):
        return lazyvalmap(f, d, *args, **kwargs)
    dd = {k: f(v, *args, **kwargs) for (k,v) in d.items()}
    return fdict(dd) if is_fdict(d) else dd
def lazykeymap(f, d, *args, **kwargs):
    """Returns a lazydict object whose values are a function of a dict's keys.

    `keymap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(k)`.

    `keymap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(k, *args, **kw)`.

    This function always returns a lazydict whose values have not yet been
    cached.
    """
    return ldict({k: delay(f, k, *args, **kwargs) for k in d.keys()})
def keymap(f, d, *args, **kwargs):
    """Returns a dict object whose values are a function of a dict's keys.

    `keymap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(k)`.

    `keymap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(k, *args, **kw)`.

    This function returns either a dict, a frozendict, or a lazydict depending
    on the input argument `d`. If `d` is a lazydict, then a lazydict is
    returned; if `d` is a frozendict, a frozendict is returned, and otherwise, a
    dict is returnd.
    """
    dd = {k: f(k, *args, **kwargs) for k in d.keys()}
    return ldict(dd) if is_ldict(d) else fdict(dd) if is_fdict(d) else dd
def _lazyitemmap_undelay(f, k, v, *args, **kw):
    return f(k, undelay(v), *args, **kw)
def lazyitemmap(f, d, *args, **kwargs):
    """Returns a lazydict object whose values are a function of a dict's items.

    `itemmap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(k, d[k])`.

    `itemmap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(k, *args, **kw)`.

    Unlike `lazyitemmap`, this function returns either a dict, a frozendict, or
    a lazydict depending on the input argument `d`. If `d` is a lazydict, then a
    lazydict is returned; if `d` is a frozendict, a frozendict is returned, and
    otherwise, a dict is returnd.
    """
    if is_ldict(d):
        d = {k: delay(_lazyvalmap_undelay, k, v, *args, **kwargs)
             for (k,v) in d.rawitems()}
    else:
        d = {k: delay(f, k, v, *args, **kwargs)
             for (k,v) in d.items()}
    return ldict(d)
def itemmap(f, d, *args, **kwargs):
    """Returns a dict object whose values are a function of a dict's items.

    `itemmap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(k, d[k])`.

    `itemmap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(k, *args, **kw)`.

    Unlike `lazyitemap`, this function returns either a dict, a frozendict, or a
    lazydict depending on the input argument `d`. If `d` is a lazydict, then a
    lazydict is returned; if `d` is a frozendict, a frozendict is returned, and
    otherwise, a dict is returnd.
    """
    if is_ldict(d):
        return lazyitemmap(f, d, *args, **kwargs)
    dd = {k: f(k, v, *args, **kwargs) for (k,v) in d.items()}
    return ldict(dd) if is_ldict(dd) else fdict(dd) if is_fdict(d) else dd
def dictmap(f, keys, *args, **kw):
    """Returns a dict with the given keys and the values `map(f, keys)`.

    `dictmap(f, keys)` returns a dict object whose keys are the elements of
    `iter(keys)` and whose values are the elements of `map(f, keys)`.

    `dictmap(f, keys, *args, **kw)` returns a dict object whose keys are the
    elements of `iter(keys)` and whose values are the elements of
    `[f(k, *args, **kw) for k in iter(keys)]`.
    """
    return {k: f(k, *args, **kw) for k in keys}
def frozendictmap(f, keys, *args, **kw):
    """Returns a frozendict with the given keys and the values `map(f, keys)`.

    `frozendictmap(f, keys)` returns a dict object whose keys are the elements
    of `iter(keys)` and whose values are the elements of `map(f, keys)`.

    `frozendictmap(f, keys, *args, **kw)` returns a dict object whose keys are
    the elements of `iter(keys)` and whose values are the elements of
    `[f(k, *args, **kw) for k in iter(keys)]`.
    """
    return fdict({k: f(k, *args, **kw) for k in keys})
fdictmap = frozendictmap
def lazydictmap(f, keys, *args, **kw):
    """Returns a lazydict with the given keys and the values `map(f, keys)`.

    `lazydictmap(f, keys)` returns a dict object whose keys are the elements of
    `iter(keys)` and whose values are the elements of `map(f, keys)`. All values
    are uncached and lazy.

    `lazydictmap(f, keys, *args, **kw)` returns a dict object whose keys are the
    elements of `iter(keys)` and whose values are the elements of
    `[f(k, *args, **kw) for k in iter(keys)]`.
    """
    return ldict({k: delay(f, k, *args, **kw) for k in keys})
ldictmap = lazydictmap
def merge(*args, **kw):
    '''Merges dict-like objects left-to-right. See also `rmerge`.

    `merge(...)` collapses all arguments, which must be python `Mapping` objects
    of some kind, into a single mapping from left-to-right. The mapping that is
    returned depends on the inputs: if any of the input mappings are lazydict
    objects, then a lazydict is returned (and the laziness of arguments is
    respected); otherwise, a frozendict object is retuend.

    Note that optional keyword arguments may be passed; these are considered the
    right-most dictionary argument.
    '''
    if len(args) == 0:
        return frozendict(kw)
    # Make the initial dictionary.
    res = args[0]
    lazy = is_lazydict(res)
    res = dict(res.rawitems() if lazy else res)
    for d in args[1:]:
        if is_lazydict(d):
            lazy = True
            res.update(d.rawitems())
        else:
            res.update(d)
    res.update(kw)
    return lazydict(res) if lazy else frozendict(res)
def rmerge(*args, **kw):
    '''Merges dict-like objects right-to-left. See also `merge`.

    `rmerge(...)` collapses all arguments, which must be python `Mapping`
    objects of some kind, into a single mapping from right-to-left. The mapping
    that is returned depends on the inputs: if any of the input mappings are
    lazydict objects, then a lazydict is returned (and the laziness of arguments
    is respected); otherwise, a frozendict object is retuend.

    Note that optional keyword arguments may be passed; these are considered the
    right-most dictionary argument.
    '''
    if len(args) == 0:
        return frozendict(kw)
    # Make the initial dictionary.
    res = dict(kw)
    lazy = False
    for d in reversed(args):
        if is_lazydict(d):
            lazy = True
            res.update(d.rawitems())
        else:
            res.update(d)
    res.update(kw)
    return lazydict(res) if lazy else frozendict(res)
def assoc(d, *args, **kw):
    """Returns a copy of the given dictionary with additional key-value pairs.

    `assoc(d, key, val)` returns a copy of the dictionary `d` with the given
    key-value pair associated in the new copy. The return value is always the
    same type as the argument `d`.

    `assoc(d, key1, val1, key2, val2 ...)` associates all the given keys to the
    given values in the returned copy.

    `assoc(d, key1=val1, key2=val2 ...)` uses the keyword arguments as the
    arguments that are to be associated.
    """
    if len(args) % 2 != 0:
        raise ValueError("assoc requires matched key-value arguments")
    ks = args[0::2]
    vs = args[1::2]
    if is_mutmap(d):
        # This is a mutable mapping, so we copy it.
        d = d.copy()
        for (k,v) in zip(ks,vs):
            d[k] = v
        for (k,v) in kw.items():
            d[k] = v
    elif is_fdict(d):
        for (k,v) in zip(ks,vs):
            d = d.set(k, v)
        for (k,v) in kw.items():
            d = d.set(k, v)
    else:
        raise TypeError("can only assoc to MutableMapping and frozendict types")
    return d
def dissoc(d, *args):
    """Returns a copy of the given dictionary with certain keys removed.

    `dissoc(d, key)` returns a copy of the dictionary `d` with the given
    key disssociated in the new copy. The return value is always the
    same type as the argument `d`.

    `dissoc(d, key1, key2 ...)` dissociates all the given keys from their values
    in the returned copy.
    """
    if is_mutmap(d):
        # This is a mutable mapping, so we copy it.
        d = d.copy()
        for k in args:
            if k in d:
                del d[k]
    elif is_fdict(d):
        for k in args:
            d = d.delete(k)
    else:
        raise TypeError("can only dissoc from MutableMapping and frozendict"
                        " types")
    return d
def _lambdadict_call(data, fn):
    spec = inspect.getfullargspec(fn)
    dflts = spec.defaults or fdict()
    args = []
    kwargs = {}
    pos = True
    for k in spec.args:
        if k in data:
            v = undelay(data[k])
            if pos: args.append(v)
            else:   kwargs[k] = v
        else:
            pos = False
            if k in dflts: kwargs[k] = dflts[k]
    for k in spec.kwonlyargs:
        if k in data:
            kwargs[k] = undelay(data[k])
        else:
            if k in dflts: kwargs[k] = dflts[k]
    return fn(*args, **kwargs)
def lambdadict(*args, **kwargs):
    """Builds and returns a `lazydict` with lambda functions calculated lazily.

    `lambdadict(args...)` is equivalent to `lazydict(args...)` except that any
    lambda function in the values provided by the merged arguments is delayed as
    a function whose inputs come from the lambda-function variable names in
    the same resulting lazydict.

    WARNING: This function will gladly return a lazydict that encapsulates an
    infinite loop if you are not careful. For example, the following lambdadict
    will infinite loop when either key is requested:
    `ld = lambdadict(a=lambda b:b, b=lambda a:a)`.
    
    Examples
    --------
    >>> d = lambdadict(a=1, b=2, c=lambda a,b: a + b); d
    <:{'a': 1, 'b': 2, 'c': <lazy>}:>
    >>> d['c']
    3
    >>> d
    <:{'a': 1, 'b': 2, 'c': 3}:>

    """
    d = merge(*args, **kwargs)
    finals = {} # We use a mutable hack here.
    for (k,v) in (d.rawitems() if is_ldict(d) else d.items()):
        if isinstance(v, types.LambdaType):
            finals[k] = delay(_lambdadict_call, finals, v)
        else:
            finals[k] = v
    return ldict(finals)
