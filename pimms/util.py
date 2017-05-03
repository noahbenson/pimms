####################################################################################################
# pimms/util.py
# Utility classes for functional programming with pimms!
# By Noah C. Benson

import copy, inspect, types, sys, six, pint, os
import numpy as np, pyrsistent as ps
try:    import cStringIO as strio
except: import StringIO  as strio
try:    import cPickle   as pickle
except: import pickle

units = pint.UnitRegistry()
units.define('pixel = [image_length] = px')

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

def is_quantity(q):
    '''
    is_quantity(q) yields True if q is a pint quantity and False otherwise.
    '''
    cls = type(q)
    return cls.__module__.startswith('pint.') and cls.__name__ == 'Quantity'
def is_unit(q):
    '''
    is_unit(q) yields True if q is a pint unit and False otherwise.
    '''
    cls = type(q)
    return cls.__module__ == 'pint.unit' and cls.__name__ == 'Unit'
def quant(val, unit):
    '''
    quant(value, unit) returns a quantity with the given unit; if value is not currently a quantity,
      then value * unit is returned; if value is a quantity, then it is coerced into the given unit;
      this may raise an error if the units are not compatible.
    '''
    return val.to(unit) if is_quantity(val) else units.Quantity(val, unit)
def mag(val, unit=Ellipsis):
    '''
    mag(value) returns the magnitide of the given value; if value is not a quantity, then value is
      returned; if value is a quantity, then its magnitude is returned. If the option unit is given
      then, if the val is quantity, it is cast to the given unit before being the magnitude is
      returned, otherwise it is returned alone
    '''
    return (val   if not is_quantity(val) else 
            val.m if unit is Ellipsis     else
            val.to(unit).m)
def like_units(a, b):
    '''
    like_units(a,b) yields True if a and b can be cast to each other in terms of units and False
      otherwise. Non-united units are considered dimensionless units.
    '''
    a = quant(a, 'dimensionless') if not is_quantity(a) else a
    b = quant(b, 'dimensionless') if not is_quantity(b) else b
    if a.u == b.u: return True
    try:
        c = a.to(b.u)
        return True
    except:
        return False
def qhashform(o):
    '''
    qhashform(o) yields a version of o, if possible, that yields a hash that can be reproduced
      across instances. This correctly handles quantities and numpy arrays, among other things.
    '''
    if is_quantity(o): return ('__#quant', qhashform(mag(o)), str(o.u))
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.number):
        return ('__#ndarray', o.tobytes())
    elif isinstance(o, (set, frozenset)): return ('__#set', tuple([qhashform(x) for x in o]))
    elif is_map(o): return ps.pmap({qhashform(k):qhashform(v) for (k,v) in o.iteritems()})
    elif hasattr(o, '__iter__'): return tuple([qhashform(u) for u in o])
    else: return o
def qhash(o):
    '''
    qhash(o) is a hash function that operates like hash(o) but attempts to, where possible, hash
      quantities in a useful way. It also correctly handles numpy arrays and various other normally
      mutable and/or unhashable objects.
    '''
    return hash(qhashform(o))
io_formats = colls.OrderedDict(
    [('numpy', {
        'match': lambda o:   isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.number),
        'write': lambda s,o: np.save(s, o),
        'read':  lambda s:   np.load(s)}),
     ('pickle', {
        'match': lambda o:   True,
        'write': lambda s,o: pickle.dump(o, s),
        'read':  lambda s:   pickle.load(s)})])
def _check_io_format(obj, fmt):
    try:    return io_formats[fmt]['match'](obj)
    except: return False
def _save_stream_format(stream, obj, fmt):
    fdat = io_formats[fmt]
    fdat['write'](stream, obj)
def _load_stream_format(stream, fmt):
    fdat = io_formats[fmt]
    return fdat['read'](stream)
def _save_stream(stream, obj):
    for (fmt,fdat) in io_formats.iteritems():
        if not _check_io_format(obj, fmt): continue
        s = strio.StringIO()
        try:
            _save_stream_format(s, obj, fmt)
            pickle.dump(fmt, stream)
            stream.write(s.getvalue())
        except:  continue
        else:    return stream
        finally: s.close()
    raise ValueError('unsavable object: did not match any exporters')
def _load_stream(stream):
    try:    fmt = pickle.load(stream)
    except: raise ValueError('could not unpickle format; probably not a pimms save file')
    if not isinstance(fmt, basestring):
        raise ValueError('file format object is not a string; probably not a pimms save file')
    if fmt not in io_formats:
        raise ValueError('file has unrecognized format \'%s\'' % fmt)
    return _load_stream_format(stream, fmt)
def save(filename, obj, overwrite=False, create_directories=False):
    '''
    pimms.save(filename, obj) attempts to pickle the given object obj in the filename (or stream,
      if given). An error is raised when this cannot be accomplished; the first argument is always
      returned; though if the argument is a filename, it may be a differet string that refers to
      the same file.

    The save/load protocol uses pickle for all saving/loading except when the object is a numpy
    object, in which case it is written using obj.tofile(). The save function writes meta-data
    into the file so cannot simply be unpickled, but must be loaded using the pimms.load()
    function. Fundamentally, however, if an object can be picled, it can be saved/loaded.

    Options:
      * overwrite (False) The optional parameter overwrite indicates whether an error should be
        raised before opening the file if the file already exists.
      * create_directories (False) The optional parameter create_directories indicates whether the
        function should attempt to create the directories in which the filename exists if they do
        not already exist.
    '''
    if isinstance(filename, basestring):
        filename = os.path.expanduser(filename)
        if not overwrite and os.path.exists(filename):
            raise ValueError('save would overwrite file %s' % filename)
        if create_directories:
            dname = os.path.dirname(os.path.realpath(filename))
            if not os.path.isdir(dname): os.makedirs(dname)
        with open(filename, 'w') as f:
            _save_stream(f, obj)
    else:
        _save_stream(filename, obj)
    return filename
def load(filename, ureg='pimms'):
    '''
    pimms.load(filename) loads a pimms-formatted save-file from the given filename, which may
      optionaly be a string. By default, this function forces all quantities (via the pint
      module) to be loaded using the pimms.units unit registry; the option ureg can change
      this.

    If the filename is not a correctly formatted pimms save-file, an error is raised.

    Options:
      * ureg ('pimms') specifies the unit-registry to use for ping module units that are loaded
        from the files; 'pimms' is equivalent to pimms.units. None is equivalent to using the
        pint._APP_REGISTRY unit registry.
    '''
    if ureg is not None:
        if ureg == 'pimms': ureg = units
        orig_app_ureg = pint._APP_REGISTRY
        orig_dfl_ureg = pint._DEFAULT_REGISTRY
        pint._APP_REGISTRY = ureg
        pint._DEFAULT_REGISTRY = ureg
        try:    return load(filename, ureg=None)
        except: raise
        finally:
            pint._APP_REGISTRY     = orig_app_ureg
            pint._DEFAULT_REGISTRY = orig_dfl_ureg
    if isinstance(filename, basestring):
        filename = os.path.expanduser(filename)
        with open(filename, 'r') as f:
            return _load_stream(f)
    else:
        return _load_stream(filename)

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
