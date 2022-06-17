# -*- coding: utf-8 -*-
################################################################################
# pimms/lazydict/_core.py
#
# Declaration of the pimms lazy dictionary type.
#
# @author Noah C. Benson

# Dependencies #################################################################
import inspect, types, sys, os, warnings, weakref
import collections as colls
import numpy as np

from functools import (reduce, partial)
from frozendict import frozendict

from ..doc import docwrap
from ..types import (is_str, is_fdict, is_map)

# #Delay #######################################################################
@docwrap
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
    def __init__(self, weak, func, /, *args, **kw):
        # Make sure func is callable.
        if not callable(func):
            raise TypeError("Delay functions must be callable")
        # We want to use a persistent map for the keywords.
        kw = fdict(kw)
        # We want to setup our memoized value as well.
        object.__setattr__(self, '_partial', (func, args, kw))
        object.__setattr__(self, '_results', boolean(weak))
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
        return (r is True or r is False)
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
def delay(f, /, *args, **kw):
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
    *args
        The arguments to be passed to the given function `fn`.
    **kw
        The keyword arguments to be passed to the given function `fn`.

    Returns
    -------
    Delay
        A `Delay` object for the given function.
    """
    return Delay(False, f, *args, **kw)
def weak_delay(f, /, *args, **kw):
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
    *args
        The arguments to be passed to the given function `fn`.
    **kw
        The keyword arguments to be passed to the given function `fn`.

    Returns
    -------
    Delay
        A `Delay` object for the given function.
    """
    return Delay(True, f, *args, **kw)
wdelay = weak_delay
def is_delay(obj):
    """Determines if an object is a `Delay` or not.

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
def undelay(obj):
    """Either returns the argument or the delayed value if it is a `Delay`.

    `undelay(obj)` is equivalent to `obj() if is_delay(obj) else obj`.

    Parameters
    ----------
    obj : object
        The object that should be converted from a `Delay`.

    Returns
    -------
    object
        Either `obj`, if `obj` is not a `Delay` instance, or `obj()` if it is a
        `Delay` object.
    """
    return obj() if isinstance(obj, Delay) else obj
    
# #lazydict ####################################################################
@docwrap
class lazydict(frozendict):
    """A frozen dictionary types whose values can be lazily computed.

    `lazydict` is an immutable dictionary that is identical to the `frozendict`
    type except that when a key is assigned a value that is a delay object, that
    delay object's stored valuee is returned instead of the delay itself.

    `ldict` is an alias for `lazydict`.
    """
    def __repr__(self):
        s = ['%s: %s' % (repr(k),
                         '<lazy>' if self.is_lazy(k) else repr(self[k]))
             for k in self.keys()]
        s = ', '.join(s)
        return '<:{' + s + '}:>'
    def __str__(self):
        s = ['%s: %s' % (str(k), '<lazy>' if self.is_lazy(k) else str(self[k]))
             for k in self.keys()]
        s = ', '.join(s)
        return '<:{' + s + '}:>'
    def __getitem__(self, k):
        return undelay(frozendict.__getitem__(self, k))
    def get(self, k, default=None):
        return undelay(frozendict.get(self, k, default=default))
    def item(self, *args, **kw):
        return undelay(frozendict.item(self, *args, **kw))
    def value(self, *args, **kw):
        return undelay(frozendict.value(self, *args, **kw))
    def iterrawitems(self):
        """Iterates over the raw items in the frozendict.

        Raw items include delay objects rather than the values encapsulaetd by
        the delay.
        """
        for k in self.keys():
            v = frozendict.__getitem__(self, k)
            yield (k,v)
    def iterrawvalues(self):
        """Iterates over the raw values in the frozendict.

        Raw values include delay objects rather than the values encapsulaetd by
        the delay.
        """
        for k in self.keys():
            v = frozendict.__getitem__(self, k)
            yield v
    def iterlazy(self):
        """Returns an interator over only the lazy keys in the dictionary.

        `ld.iterlazy()` returns an iterator over the lazy keys only
        (memoized/cached lazy keys are not included in the iteration).

        Returns
        -------
        iter : iterator
            An iterator of the lazy keys in the mapping.
        """
        for (k,v) in self.iterrawitems():
            if isinstance(v, Delay):
                yield k
    def itercached(self):
        """Returns an interator over only the cached lazy keys in the lazydict.

        `ld.itercached()` returns an iterator over the memoized/cached keys
        only (lazy keys and keys that aren't associated with delays are not
        included in the iteration).

        Returns
        -------
        iter : iterator
            An iterator of the cached keys in the mapping.
        """
        for (k,v) in self.iterrawitems():
            if isinstance(v, Delay) and v.is_cached():
                yield k
    def iteruncached(self):
        """Returns an interator over the uncached lazy keys in the lazydict.

        `ld.iteruncached()` returns an iterator over the only the keys in
        `ld` thare are not cached (eager keys that aren't associated with
        delays are not included in the iteration).

        Returns
        -------
        iter : iterator
            An iterator of the uncached keys in the lazydict.
        """
        for (k,v) in self.iterrawitems():
            if isinstance(v, Delay) and not v.is_cached():
                yield k
    def itereager(self):
        """Returns an interator over only the eager keys in the lazydict.

        `ld.itereager()` returns an iterator over the eaeger keys that were
        not (ever) attached to a delayed/lazy calculation. Neither pending lazy
        values nor cached lazy values are included in the iteration.

        Returns
        -------
        iter : iterator
            An iterator of the eager (non-delayed) keys in the lazydict.
        """
        for (k,v) in self.iterrawitems():
            if not isinstance(v, Delay):
                yield k
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
        return v if isinstance(v, Delay) else None
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
        v = frozendict..__getitem__(self, k)
        if isinstance(v, Delay):
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
        if isinstance(v, Delay):
            return None
        else:
            return v.is_cached()
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
        return isinstance(v, Delay)
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
        return not isinstance(v, Delay)
colls.abc.Mapping.register(lazydict)
colls.abc.Hashable.register(lazydict)
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
    #from .table import is_itable
    #return isinstance(m, LazyMap) or is_itable(m)
    #TODO: if/when itable rewrite is done, should it be a lazy map?
    return isinstance(m, lazydict)
is_ldict = is_lazydict
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
    #TODO: make this compatible with the rewrite of tables?
    #from .table import (is_itable, ITable)
    if len(args) == 0:
        return frozendict(kw)
    # Make the initial dictionary.
    res = dict(args[0])
    islazy = False
    for d in args[1:]:
        if is_lazydict(d):
            islazy = True
            res.update(d.iterrawitems())
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
    #TODO: make this compatible with the rewrite of tables?
    #from .table import (is_itable, ITable)
    if len(args) == 0:
        return frozendict(kw)
    # Make the initial dictionary.
    res = dict(kw)
    islazy = False
    for d in reversed(args):
        if is_lazydict(d):
            islazy = True
            res.update(d.iterrawitems())
        else:
            res.update(d)
    res.update(kw)
    return lazydict(res) if islazy else frozendict(res)
