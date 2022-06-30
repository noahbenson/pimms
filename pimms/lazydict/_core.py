# -*- coding: utf-8 -*-
################################################################################
# pimms/lazydict/_core.py
#
# Declaration of the pimms lazy dictionary type.
#
# @author Noah C. Benson

# Dependencies #################################################################
import inspect, types, sys, os, warnings, weakref
import collections.abc as colls_abc
import numpy as np

from functools import (reduce, partial)
from frozendict import frozendict, frozendict as fdict

from ..doc import docwrap
from ..types import (is_str, is_fdict, is_map, is_mmap, is_set)

# #Delay #######################################################################
class Delay:
    """A delayed computation that can be either weak or strong.

    `Delay` objects are essentially fully-specified partial objects (i.e., they
    are functions with full argument lists) that cache the return value of their
    respective partial function after it is called once and remember that value
    from then on.

    Once a `Delay` has been constucted, its value can be retrieved by calling
    it: `delay = Delay(fn)` followed by `delay()`.

    Delays can be using either weak or strong references. When made using a weak
    reference, the function and argument list must be maintained
    indefinitely. When made using a strong reference, the function and argument
    list are discarded as soon as the value has been calculated. To create a
    weak delay object, the `weak_delay()` function should be used instead of the
    `delay()` function. The `Delay` constructor should usually not be called
    directly.

    Parameters
    ----------
    weak : boolean
        Whether to use prefer weak reference to the resulting object or not.
    func : callable
        The function whose value is to be cached when requested.
    *args
        The arguments to be passed to the given function `fn`.
    **kw
        The keyword arguments to be passed to the given function `fn`.
    """
    __slots__ = ('_partial', '_results')
    def __init__(self, weak, func, /, *args, **kw):
        # Make sure func is callable.
        if not callable(func):
            raise TypeError("Delay functions must be callable")
        # We want to use a persistent map for the keywords.
        kw = fdict(kw)
        # We want to setup our memoized value as well.
        object.__setattr__(self, '_partial', (func, args, kw))
        object.__setattr__(self, '_results', bool(weak))
    def __call__(self):
        # If call_data is None, then we know that the _result contains the raw
        # (not weakref'ed) value.
        if self._partial is None:
            return self._results
        # Otherwise, we have either not calculated the result yet, or, we have
        # cached it as a weakref. In the latter case, the result must be a
        # weakref.ref object, and in the former case it must be a boolean.
        res = self._results
        if not (res is True or res is False):
            # This must be a weak reference, so we first try to dereference it.
            val = res()
            # If val is not None, then this is the correct value; otherwise, it
            # indicates that the weak reference has expired and we need to
            # re-cache it.
            if val is not None:
                return val
        # At this point, we either haven't yet cached anything or we have an
        # expired weakref. Start by calculating the value we should save.
        (fn, args, kw) = self._partial
        val = fn(*args, **kw)
        # If res is False, that means that we prefer a weak reference;
        # otherwise, we prefer a strong reference. We try to make a weak ref
        # first because if it fails we can always use a strong ref.
        if res is True:
            try:
                val = weakref.ref(val)
                object.__setattr__('_results', val)
                # Since we have a weak ref, we don't forget the partial data.
                return val
            except TypeError:
                pass
        # If we reach here, then we prefer a strong reference or we weren't able
        # to create a weak reference due to the type of val.
        object.__setattr__(self, '_results', val)
        # With strong references, we forget the partial information.
        object.__setattr__(self, '_partial', None)
        # Finally, return the value
        return val
    def __repr__(self):
        s = 'cached' if self.is_cached() else 'delayed'
        return f"<Delay at {id(self)}: {s}>"
    def __str__(self):
        return f"<Delay at {id(self)}>"
    def is_cached(self):
        """Returns `True` if the delay value is cached and `False` otherwise.

        Returns
        -------
        boolean
            `True` if the given delay object has cached its value and `False`
            otherwise.
        """
        if self._partial is None: return True
        r = self._results
        return not (r is True or r is False)
    def is_weak(self):
        """Determines whether the delay object uses a weak reference or not.

        Returns
        -------
        boolean or None
            `delay_obj.is_weak()` returns `True` if `delay_obj` currently holds
            a weakly-referenced cached value, `False` if it holds a strong
            reference to a cached object, and `None` if it is not currently
            cached.
        """
        # For a strong reference, the _partial is always None.
        if self._partial is None: return False
        # Otherwise, if _results is True or False, it's not cached.
        r = self._results
        if r is True or r is False: return None
        # And the only other possibility is that it is weakly cached.
        return True
    def prefers_weak(self):
        """Returns the delay object's preference for a weak reference.

        A `Delay` that is already cached will always return `None` from this
        method.

        Returns
        -------
        boolean or None
            `delay_obj.prefers_weak()` returns `True` if `delay_obj` prefers to
            cache its return values weakly, `False` if it prefers to cache its
            values strongly, and `None` if it already contains a cached value.
        """
        # If the partial is empty, we're already cached.
        if self._partial is None: return None
        # Similarly if _results isn't either True or False, it's already cached.
        r = self._results
        if not (r is True or r is False): return None
        # Otherwise, the _results stores the weakness preference.
        return True
@docwrap
def delay(func, /, *args, **kw):
    """Returns a delayed computation.

    `delay(fn)` constructs a `Delay` object, which is essentially a
    fully-specified partial object (i.e., a function with a fully-formed
    argument lists) that caches the return value of the respective partial
    function after it is called once and remembers that value from then on.

    Once a `Delay` has been constucted, its value can be retrieved by calling
    it: `d = delay(fn)` followed by `d()`.

    Delays can be using either weak or strong references. When made using a weak
    reference, the function and argument list must be maintained
    indefinitely. When made using a strong reference, the function and argument
    list are discarded as soon as the value has been calculated, but the
    calculated value cannot be garbage collected. To create a weak delay object,
    the `weak_delay()` or `wdelay()` functions instead of the `delay()`
    function.

    Parameters
    ----------
    func : callable
        The function whose value is to be cached when requested.
    args
        The arguments to be passed to the given function `fn`.
    kw
        The keyword arguments to be passed to the given function `fn`.

    Returns
    -------
    Delay
        A `Delay` object for the given function.
    """
    return Delay(False, func, *args, **kw)
@docwrap
def weak_delay(func, /, *args, **kw):
    """Returns a delayed computation that holds its memoized value weakly.

    `weak_delay(fn)` and, equivalently, `wdelay(fn)` both construct a `Delay`
    object, which is essentially a fully-specified partial object (i.e., a
    function with a fully-formed argument lists) that caches the return value of
    the respective partial function after it is called once and remembers that
    value from then on. Unlike the `delay(fn)` function, the `Delay` object
    returned from a `weak_delay(fn)` function will use a weak reference to the
    cached value if possible.

    Once a `Delay` has been constucted, its value can be retrieved by calling
    it: `d = delay(fn)` followed by `d()`.

    Delays can be using either weak or strong references. When made using a weak
    reference, the function and argument list must be maintained indefinitely in
    case the value needs to be recalculated later. When made using a strong
    reference, the function and argument list are discarded as soon as the value
    has been calculated, but the calculated value cannot be garbage collected
    until the `Delay` object itself is garbage collected. To create a delay
    whose reference is not weak, the `delay()` function should be used instead
    of the `weak_delay()` or `wdelay()` functions.

    Parameters
    ----------
    func : callable
        The function whose value is to be cached when requested.
    args
        The arguments to be passed to the given function `fn`.
    kw
        The keyword arguments to be passed to the given function `fn`.

    Returns
    -------
    Delay
        A `Delay` object for the given function.
    """
    return Delay(True, func, *args, **kw)
wdelay = weak_delay
@docwrap
def is_delay(obj):
    """Determines if an object is a `Delay` or not.

    `Delay` objects are specific to the `pimms` library; however `dask`'s
    `Delayed` type is also supported by `pimms`: see `is_delayed`,
    `is_dask_delayed`, and `undelay`.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Delay` is to be determined.

    Returns
    -------
    boolean
        `True` if `obj` is a `Delay` object and `False` if not.
    """
    return isinstance(obj, Delay)
@docwrap
def is_dask_delayed(obj):
    """Determines if an object is a `dask.delay.Delayed` object or not.

    `Delay` objects are specific to the `pimms` library; however `dask`'s
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
    """Determines if an object is a `Delay` or `dask.delayed.Delayed` object.

    `Delay` objects are specific to the `pimms` library; however `dask`'s
    `Delayed` type is also supported by `pimms`: see `is_delay`,
    `is_dask_delayed`, and `undelay`.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Delayed` object is to be determined.

    Returns
    -------
    boolean
        `True` if `obj` is a `Delayed` object and `False` if not.
    """
    return is_delay(obj) or is_dask_delayed(obj)
@docwrap
def undelay(obj, dask=True):
    """Either returns the argument or the delayed value if it is a `Delay`.

    `undelay(obj)` is equivalent to `obj() if is_delay(obj) else obj` unless
    `obj` is a `dask.delayed.Delayed` object, in which case `obj,compute()` is
    used instead of `obj()`.

    Parameters
    ----------
    obj : object
        The object that should be converted from a `Delay`.
    dask : boolean
        If `True`, then treats `dask.Delayed` objects as `Delay` objects and
        undelays them as well. If `False`, then undelays only `Delay` objects.

    Returns
    -------
    object
        Either `obj`, if `obj` is not a `Delay` instance, or `obj()` if it is a
        `Delay` object.
    """
    if   isinstance(obj, Delay):        return obj()
    elif dask and is_dask_delayed(obj): return obj.compute()
    else:                               return obj

# #lazydict ####################################################################
class lazydict_keys(colls_abc.KeysView, colls_abc.Set):
    __slots__ = ('mapping', '_lazydict_keys', '_select_lazy', '_select_cached',
                 '_filter', '_count')
    @staticmethod
    def _make_filter(cls, ld, sel_lazy, sel_cached):
        if   sel_lazy   is False: return lambda k: ld.is_eager(k)
        elif sel_cached is True:  return lambda k: ld.is_cached(k)
        elif sel_cached is False: return lambda k: ld.is_uncached(k)
        else: return None
    @staticmethod
    def _counter(x):
        return len(list(iter(x)))
    @staticmethod
    def _make_count(x):
        return delay(lazydict_keys._counter, x)
    def __new__(cls, ld, lazy=None, cached=None):
        if not isinstance(ld, lazydict):
            raise ValueError("can only make lazydict_keys object from lazydict")
        self = object.__new__(cls)
        object.__setattr__(self, 'mapping', ld)
        object.__setattr__(self, '_lazydict_keys', fdict.keys(ld))
        lazy   = None if lazy   is None else bool(lazy)
        cached = None if cached is None else bool(cached)
        filt   = lazydict_keys._make_filter(cls, ld, lazy, cached)
        count  = lazydict_keys._make_count(self)
        object.__setattr__(self, '_select_lazy',   lazy)
        object.__setattr__(self, '_select_cached', cached)
        object.__setattr__(self, '_filter', filt)
        object.__setattr__(self, '_count', count)
        return self
    def __init__(self, ld, lazy=None, cached=None, raw=None):
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
    __slots__ = ('_lazydict_items',
                 '_select_lazy', '_select_cached', '_select_raw',
                 '_filter', '_count')
    @staticmethod
    def _make_filter(cls, ld, sel_lazy, sel_cached):
        if   sel_lazy   is False: return lambda kv: ld.is_eager(kv[0])
        elif sel_cached is True:  return lambda kv: ld.is_cached(kv[0])
        elif sel_cached is False: return lambda kv: ld.is_uncached(kv[0])
        else: return None
    @staticmethod
    def _undelay(kv):
        return (kv[0], undelay(kv[1], dask=False))
    def __new__(cls, ld, lazy=None, cached=None, raw=False):
        if not isinstance(ld, lazydict):
            raise ValueError("can only make lazydict_items object from"
                             " lazydict")
        self = object.__new__(cls)
        object.__setattr__(self, '_lazydict_items', fdict.items(ld))
        lazy   = None if lazy   is None else bool(lazy)
        cached = None if cached is None else bool(cached)
        filt   = lazydict_items._make_filter(cls, ld, lazy, cached)
        count  = lazydict_keys._make_count(self)
        object.__setattr__(self, '_select_lazy',   lazy)
        object.__setattr__(self, '_select_cached', cached)
        object.__setattr__(self, '_select_raw',    bool(raw))
        object.__setattr__(self, '_filter', filt)
        object.__setattr__(self, '_count', count)
        return self
    def __init__(self, ld, lazy=None, cached=None, raw=None):
        colls_abc.ItemsView.__init__(self, ld)
        colls_abc.Set.__init__(self)
    def __iter__(self):
        it = filter(self._filter, iter(self._lazydict_items))
        if self._select_raw:
            return it
        else:
            return map(lazydict_items._undelay, it)
    def __reversed__(self):
        return filter(self._filter, reversed(self._lazydict_items))
    def __contains__(self, k):
        return (k in self._lazydict_items and self._filter(k))
    def __len__(self):
        return self._count()
class lazydict_values(colls_abc.ValuesView, colls_abc.Sequence):
    __slots__ = ('_lazydict_items',
                 '_select_lazy', '_select_cached', '_select_raw',
                 '_filter', '_count')
    @staticmethod
    def _take_value(kv):
        return kv[1]
    def __new__(cls, ld, lazy=None, cached=None, raw=False):
        if not isinstance(ld, lazydict):
            raise ValueError("can only make lazydict_values object from"
                             " lazydict")
        self = object.__new__(cls)
        object.__setattr__(self, '_lazydict_items', fdict.items(ld))
        lazy   = None if lazy   is None else bool(lazy)
        cached = None if cached is None else bool(cached)
        filt   = lazydict_items._make_filter(cls, ld, lazy, cached)
        count  = lazydict_keys._make_count(self)
        object.__setattr__(self, '_select_lazy',   lazy)
        object.__setattr__(self, '_select_cached', cached)
        object.__setattr__(self, '_select_raw',    bool(raw))
        object.__setattr__(self, '_filter', filt)
        object.__setattr__(self, '_count', count)
        return self
    def __init__(self, ld, lazy=None, cached=None, raw=None):
        colls_abc.ValuesView.__init__(self, ld)
        colls_abc.Sequence.__init__(self)
    def __getitem__(self, k):
        raise TypeError("lazydict_values is not subscriptable")
    def __iter__(self):
        it = map(lazydict_values._take_value,
                 filter(self._filter, iter(self._lazydict_items)))
        if self._select_raw:
            return it
        else:
            return map(undelay, it)
    def __reversed__(self):
        return map(lazydict_values._take_value,
                   filter(self._filter, iter(self._lazydict_items)))
    def __contains__(self, v):
        return (k in self._lazydict_items and self._filter(k))
    def __len__(self):
        return self._count()
class lazydict(frozendict):
    """A frozen dictionary types whose values can be lazily computed.

    `lazydict` is an immutable dictionary that is identical to the `frozendict`
    type except that when a key is assigned a value that is a delay object, that
    delay object's stored valuee is returned instead of the delay itself.

    `ldict` is an alias for `lazydict`.
    """
    def __repr__(self):
        s = ['%s: %s' % (repr(k),
                         '<lazy>' if self.is_uncached(k) else repr(self[k]))
             for k in self.keys()]
        s = ', '.join(s)
        return '<:{' + s + '}:>'
    def __str__(self):
        s = ['%s: %s' % (str(k),
                         '<lazy>' if self.is_uncached(k) else str(self[k]))
             for k in self.keys()]
        s = ', '.join(s)
        return '<:{' + s + '}:>'
    def __getitem__(self, k):
        return undelay(frozendict.__getitem__(self, k), dask=False)
    def get(self, k, default=None):
        return undelay(frozendict.get(self, k, default=default), dask=False)
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
    def keys(self, lazy=None, cached=None):
        return lazydict_keys(self, lazy=lazy, cached=cached)
    def items(self, lazy=None, cached=None):
        return lazydict_items(self, lazy=lazy, cached=cached)
    def rawitems(self, lazy=None, cached=None):
        """Iterates over the raw items in the frozendict.

        Raw items include delay objects rather than the values encapsulaetd by
        the delay.
        """
        return lazydict_items(self, lazy=lazy, cached=cached, raw=True)
    def values(self, lazy=None, cached=None):
        return lazydict_values(self, lazy=lazy, cached=cached)
    def rawvalues(self, lazy=None, cached=None):
        """Iterates over the raw values in the frozendict.

        Raw values include delay objects rather than the values encapsulaetd by
        the delay.
        """
        return lazydict_values(self, lazy=lazy, cached=cached, raw=True)
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
    def is_uncached(self, k):
        """Determines if a key is associated with an uncached lazy value or not.

        `ld.is_uncached(k)` returns `True` if the given key `k` is lazy and
        unmemoized in the given lazy map, `ld`. If `ld` does not contain `k`
        then an error is raised, and if `k` is not lazy, then `False` is
        returned.

        Parameters
        ----------
        k : object
            The key whose laziness is to be assessed.

        Returns
        -------
        boolean or None
            If `k` is in the lazydict and is mapped to an uncached lazy value,
            then `True` is returned; if `k` is mapped to a cached lazy value,
            then `False` is returned; if `k` is mapped to a value that is not
            lazy, then `None` is returned.

        Raises
        ------
        KeyError
            If the key `k` is not in the lazydict.
        """
        v = frozendict.__getitem__(self, k)
        if is_delay(v):
            return not v.is_cached()
        else:
            return None
    def is_cached(self, k):
        """Determines if the given key is associated with a cached lazy value.

        `ld.is_cached(k)` returns `True` if the given key `k` is lazy and
        memoized/cached in the given lazyd, `ld`. If `ld` does not
        contain `k` then an error is raised. If `k` is lazy but not yet cached,
        then `False` is returned. If `k` is in the map but is not a lazy key,
        then `None` is returned.

        Parameters
        ----------
        k : object
            The key whose cache-state is to be assessed.

        Returns
        -------
        boolean or None
            If `k` is in the lazydict and is mapped to a cached lazy value,
            then `True` is returned; if `k` is mapped to an uncached lazy value,
            then `False` is returned; if `k` is mapped to a value that is not
            lazy, then `None` is returned.

        Raises
        ------
        KeyError
            If the key `k` is not in the lazydict.
        """
        v = frozendict.__getitem__(self, k)
        if is_delay(v):
            return v.is_cached()
        else:
            return None
    def is_lazy(self, k):
        """Determines if the given key is associated with a lazy value.

        `ld.is_lazy(k)` returns `True` if the given key `k` is in the given
        lazy map and is/was a lazy key/value. If `ld` does not contain `k`
        then an error is raised. If `k` is not lazy then `False` is
        returned. Note that `k` may be cached permanently, but in this case it
        is still considered lazy.

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

    `lazyvalmap(f, d)` yields a dict whose keys are the same as those of the
    given dict object and whose values, for each key `k` are `f(d[k])`. All
    values are created lazily.

    `lazyvalmap(f, d, *args, **kw)` additionally passes the given arguments to
    the function `f`, such that in the resulting map, each key `k` is mapped to
    `f(d[k], *args, **kw)`.

    This function always returns a lazydict object.
    """
    if is_ldict(d):
        d = {k: delay(_lazyvalmap_undelay, v, *args, **kw)
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
        return lazyvalmap(f, d, *args, **kw)
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
    return ldict(dd) if is_ldict(dd) else fdict(dd) if is_fdict(d) else dd
def _lazyitemmap_undelay(f, k, v, *args, **kw):
    return f(k, undelay(v), *args, **kw)
def lazyitemmap(f, d, *args, **kwargs):
    """Returns a lazydict object whose values are a function of a dict's items.

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
        d = {k: delay(_lazyvalmap_undelay, k, v, *args, **kw)
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
    res = dict(res)
    for d in args[1:]:
        if is_lazydict(d):
            islazy = True
            res.update(d.rawitems())
        else:
            res.update(d)
    res.update(kw)
    return lazydict(res) if islazy else frozendict(res)
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
    islazy = False
    for d in reversed(args):
        if is_lazydict(d):
            islazy = True
            res.update(d.rawitems())
        else:
            res.update(d)
    res.update(kw)
    return lazydict(res) if islazy else frozendict(res)
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
    if is_mmap(d):
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
    if is_mmap(d):
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
