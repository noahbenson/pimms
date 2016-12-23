####################################################################################################
# pimms/util.py
# Utility classes for functional programming with pimms!
# By Noah C. Benson

import copy, inspect, types, sys, six, pint
import pyrsistent as ps

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

def is_quantity(q):
    '''
    is_quantity(q) yields True if q is a pint quantity and False otherwise.
    '''
    cls = type(q)
    return cls.__module__ == 'pint.unit' and cls.__name__ == 'Quantity'
def is_unit(q):
    '''
    is_unit(q) yields True if q is a pint unit and False otherwise.
    '''
    cls = type(q)
    return cls.__module__ == 'pint.unit' and cls.__name__ == 'Unit'
def is_map(arg):
    '''
    is_map(x) yields True if x implements Python's builtin Mapping class.
    '''
    return isinstance(arg, colls.Mapping)
def is_pmap(arg):
    '''
    is_pmap(x) yields True if x is a persistent map object and False otherwise.
    '''
    return isinstance(arg, ps.PMap)

class LazyPMap(ps.PMap):
    '''
    LazyPMap is an immutable map that is identical to pyrsistent's PMap, but that treats functions
    of 0 arguments, when values, as lazy values, and memoizes them as it goes.
    '''
    def __init__(self, *args, **kwargs):
        self._memoized = ps.m()
    def _examine_val(self, val):
        'should only be called internally'
        if not isinstance(val, types.FunctionType): return val
        vid = id(val)
        if vid in self._memoized:
            return self._memoized[vid]
        elif ([], None, None, None) != inspect.getargspec(val):
            return val
        else:
            val = val()
            object.__setattr__(self, '_memoized', self._memoized.set(vid, val))
            return val
    def __getitem__(self, k):
        return self._examine_val(ps.PMap.__getitem__(self, k))
    def iterkeys(self):
        for (k,_) in ps.PMap.iteritems(self):
            yield k
    def iteritems(self):
        for (k,v) in ps.PMap.iteritems(self):
            yield (k, self._examine_val(v))
    def iterlazy(self):
        '''
        lmap.iterlazy() yields an iterator over the lazy keys only (memoized lazy keys are not
        considered lazy).
        '''
        for k in self.iterkeys():
            if self.is_lazy(k):
                yield k
    def itermemoized(self):
        '''
        lmap.itermemoized() yields an iterator over the memoized keys only (neihter unmemoized lazy
        keys nor normal keys are considered memoized).
        '''
        for k in self.iterkeys():
            if self.is_memoized(k):
                yield k
    def iternormal(self):
        '''
        lmap.iternormal() yields an iterator over the normal unlazy keys only (memoized lazy keys
        are not considered normal).
        '''
        for k in self.iterkeys():
            if self.is_normal(k):
                yield k
    class _LazyEvolver(ps.PMap._Evolver):
        def persistent(self):
            if self.is_dirty():
                lmap = LazyPMap(self._size, self._buckets_evolver.persistent())
                mems = self._original_pmap._memoized
                for (k,v) in ps.PMap.iteritems(self._original_pmap):
                    if not isinstance(v, types.FunctionType): continue
                    vid = id(v)
                    if vid not in mems or ps.PMap.__getitem__(lmap, k) is v: continue
                    mems = mems.discard(vid)
                object.__setattr__(lmap, '_memoized', mems)
                self._original_pmap = lmap
            return self._original_pmap
    def evolver(self):
        return self._LazyEvolver(self)
    def update_with(self, update_fn, *maps):
        evolver = self.evolver()
        for map in maps:
            if isinstance(map, LazyPMap):
                for key in map.iterkeys():
                    value = ps.PMap.__getitem__(map, key)
                    evolver.set(key, update_fn(evolver[key], value) if key in evolver else value)
            else:
                for key, value in map.items():
                    evolver.set(key, update_fn(evolver[key], value) if key in evolver else value)
        return evolver.persistent()
    def is_lazy(self, k):
        '''
        lmap.is_lazy(k) yields True if the given k is lazy and unmemoized in the given lazy map,
        lmap, otherwise False.
        '''
        v = ps.PMap.__getitem__(self, k)
        if not isinstance(v, types.FunctionType) or \
           id(v) in self._memoized or \
           ([], None, None, None) != inspect.getargspec(v):
            return False
        else:
            return True
    def is_memoized(self, k):
        '''
        lmap.is_memoized(k) yields True if k is a key in the given lazy map lmap that is both lazy
        and already memoized.
        '''
        v = ps.PMap.__getitem__(self, k)
        if not isinstance(v, types.FunctionType):
            return False
        else:
            return id(v) not in self._memoized
    def is_normal(self, k):
        '''
        lmap.is_normal(k) yields True if k is a key in the given lazy map lmap that is neither lazy
        nor a formerly-lazy memoized key.
        '''
        v = ps.PMap.__getitem__(self, k)
        if not isinstance(v, types.FunctionType) or ([],None,None,None) != inspect.getargspec(v):
            return True
        else:
            return False
colls.Mapping.register(LazyPMap)
colls.Hashable.register(LazyPMap)
def _lazy_turbo_mapping(initial, pre_size):
    '''
    _lazy_turbo_mapping is a blatant copy of the pyrsistent._pmap._turbo_mapping function, except
    it works for lazy maps; this seems like the only way to fully overload PMap.
    '''
    size = pre_size or (2 * len(initial)) or 8
    buckets = size * [None]
    if not isinstance(initial, colls.Mapping): initial = dict(initial)
    for k, v in six.iteritems(initial):
        h = hash(k)
        index = h % size
        bucket = buckets[index]
        if bucket: bucket.append((k, v))
        else:      buckets[index] = [(k, v)]
    return LazyPMap(len(initial), ps.pvector().extend(buckets))
_EMPTY_LMAP = _lazy_turbo_mapping({}, 0)
def lazy_map(initial={}, pre_size=0):
    '''
    lazy_map is a blatant copy of the pyrsistent.pmap function, and is used to create lazy maps.
    '''
    if not initial:
        return _EMPTY_LMAP
    return _lazy_turbo_mapping(initial, pre_size)
def is_lazy_map(m):
    '''
    is_lazy_map(m) yields True if m is an instance if LazyPMap and False otherwise.
    '''
    return isinstance(m, LazyPMap)

def _choose_last(k, vs):
    '_choose_last(k, vs) yields vs[-1][k].'
    return vs[-1][k]
def merge(*args, **kwargs):
    '''
    merge(...) lazily collapses all arguments, which must be python Mapping objects of some kind,
      into a single mapping from left-to-right. The mapping that is returned is a lazy persistent
      object that does not request the value of a key from any of the maps provided until they are
      requested of it; in this fashion it preserves the laziness of immutable map objects that are
      passed to it. Arguments may be mappings or lists/tuples of mappings.

    The following options are accepted:
    * choose (default None) specifies a function that chooses from which map, of those maps given
      to merge, the value should be drawn when keys overlap. The function is always passed two
      arguments: the key for which the conflict occurs and a list of maps containing that key; it
      should return the value to which the key should be mapped. The default uses the first map.
    '''
    args = tuple(arg for arg0 in args for arg in ([arg0] if is_map(arg0) else arg0))
    if not all(is_map(arg) for arg in args):
        raise ValueError('marge requires Mapping collections')
    all_keys = reduce(lambda r,s: r|s, [set(m.iterkeys()) for m in args])
    choose_fn = None
    if 'choose' in kwargs: choose_fn = kwargs['choose']
    kwargs = set(kwargs.iterkeys()) - set(['choose'])
    if len(kwargs) != 0:
        raise ValueError('Unidentified options given to merge: %s' (list(kwargs),))
    if choose_fn is None: choose_fn = _choose_last
    def _make_lambda(k, args):
        return lambda:choose_fn(k, args)
    return lazy_map({k:_make_lambda(k, [m for m in args if k in m]) for k in all_keys})
