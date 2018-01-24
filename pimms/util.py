####################################################################################################
# pimms/util.py
# Utility classes for functional programming with pimms!
# By Noah C. Benson

import inspect, types, sys, six, pint, os, numbers
import numpy as np, pyrsistent as ps
from functools import reduce
try:    import cStringIO as strio
except: import StringIO  as strio
try:    import cPickle   as pickle
except: import pickle

units = pint.UnitRegistry()
units.define('pixel = [image_length] = px')

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls


def is_unit(q):
    '''
    is_unit(q) yields True if q is a pint unit or a string that names a pint unit and False
      otherwise.
    '''
    if isinstance(q, six.string_types):
        return hasattr(units, q)
    else:
        cls = type(q)
        return cls.__module__.startswith('pint.') and cls.__name__ == 'Unit'
def is_quantity(q):
    '''
    is_quantity(q) yields True if q is a pint quantity or a tuple (scalar, unit) and False
      otherwise.
    '''
    if isinstance(q, tuple):
        return len(q) == 2 and is_unit(q[1])
    else:
        cls = type(q)
        return cls.__module__.startswith('pint.') and cls.__name__ == 'Quantity'
def unit(u):
    '''
    unit(u) yields the pimms-library unit object for the given unit object u (which may be from a
      separate pint.UnitRegistry instance).
    unit(uname) yields the unit object for the given unit name uname.
    unit(None) yields None.
    unit(q) yields the unit of the given quantity q.
    '''
    if u is None:    return None
    elif is_unit(u): return getattr(units, str(u))
    elif is_quantity(u):
        if isinstance(u, tuple): return getattr(units, str(u[1]))
        else: return getattr(units, str(u.u))
    else:
        raise ValueError('unrecotnized unit argument')
def mag(val, u=Ellipsis):
    '''
    mag(scalar) yields scalar for the given scalar (numeric value without a unit).
    mag(quant) yields the scalar magnitude of the quantity quant.
    mag(scalar, u=u) yields scalar.
    mag(quant, u=u) yields the given magnitude of the given quant after being translated into the
      given unit u.

    The quant arguments in the above list may be replaced with (scalar, unit) as in (10, 'degrees')
    or (14, 'mm'). Note that mag always translates all lists and tuples into numpy ndarrays.
    '''
    if not is_quantity(val): return val
    if isinstance(val, tuple):
        val = units.Quantity(val[0], val[1])
    return val.m if u is Ellipsis else val.to(unit(u)).m
def imm_array(q):
    '''
    imm_array(val) yields a version of val that is wrapped in numpy's ndarray class. If val is a
      read-only numpy array, then it is returned as-is; if it is not, then it is cast copied to a
      numpy array, duplicated, the read-only flag is set on the new object, and then it is returned.
    '''
    if is_quantity(q):
        m = mag(q)
        mm = imm_array(m)
        return q if mm is m else units.Quantity(mm, unit(q))
    elif not isinstance(q, np.ndarray) or q.flags['WRITEABLE']:
        q = np.array(q)
        q.setflags(write=False)
    return q
def quant(val, u=Ellipsis):
    '''
    quant(scalar) yields a dimensionless quantity with the magnitude given by scalar.
    quant(q) yields q for any quantity q; if q is not part of the pimms units registry, then a
      version of q registered to pimms.units is yielded. Note that q is a quantity if
      pimms.is_quantity(q) is True.
    quant(x, u) yields the given scalar or quantity x converted to the given unit u; if x is a
      scalar or a dimensionless quantity, then the unit u is given to the new quantity with no
      conversion; otherwise, x must be a quantity whose unit can be converted into the unit u.
    '''
    if is_quantity(val):
        if isinstance(val, tuple) or val._REGISTRY is not units:
            val = units.Quantity(mag(val), unit(val))
        return val if u is Ellipsis or u is None else val.to(unit(u))
    else:
        return units.Quantity(val, units.dimensionless if u is Ellipsis or u is None else unit(u))
def iquant(val, u=Ellipsis):
    '''
    iquant(...) is equivalent to quant(...) except that the magnitude of the return value is always
      a read-only numpy array object.
    '''
    if u is not Ellipsis and u is not None: u = unit(u)
    if is_quantity(val):
        uu = unit(val)
        if u is Ellipsis or u == uu:
            # no conversion necessary; might be able to reuse old array
            m  = mag(val)
            mm = imm_array(m)
            if m is not mm or isinstance(val, tuple) or val._REGISTRY is not units:
                val = units.Quantity(mm, uu)
            return val
        else:
            # we convert to another type first, then make an imm array
            if isinstance(val, tuple) or val._REGISTRY is not units:
                val = units.Quantity(mag(val), uu)
            v = val.to(u)
            return units.Quantity(imm_array(v.m), v.u)
    else:
        return units.Quantity(imm_array(val), units.dimensionless if u is Ellipsis else unit(u))
def like_units(a, b):
    '''
    like_units(a,b) yields True if a and b can be cast to each other in terms of units and False
      otherwise. Non-united units are considered dimensionless units.
    '''
    a = quant(0.0, a) if is_unit(a) else a if is_quantity(a) else quant(a, units.dimensionless)
    b = quant(0.0, b) if is_unit(b) else b if is_quantity(b) else quant(b, units.dimensionless)
    if a == b: return True
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
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.dtype(np.number).type):
        return ('__#ndarray', o.tobytes())
    elif isinstance(o, (set, frozenset)): return ('__#set', tuple([qhashform(x) for x in o]))
    elif is_map(o): return ps.pmap({qhashform(k):qhashform(v) for (k,v) in six.iteritems(o)})
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
        'match': lambda o:   (isinstance(o, np.ndarray) and
                              np.issubdtype(o.dtype, np.dtype(np.number).type)),
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
    for (fmt,fdat) in six.iteritems(io_formats):
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
    if not isinstance(fmt, six.string_types):
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
    if isinstance(filename, six.string_types):
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
    if isinstance(filename, six.string_types):
        filename = os.path.expanduser(filename)
        with open(filename, 'r') as f:
            return _load_stream(f)
    else:
        return _load_stream(filename)

def is_str(arg):
    '''
    is_str(x) yields True if x is a string object and False otherwise.
    '''
    return isinstance(arg, six.string_types)
def is_class(arg):
    '''
    is_class(x) yields True if x is a class object and False otherwise.
    '''
    return isinstance(arg, six.class_types)

# type translations:
_numpy_type_names = {'int':     np.integer,
                     'float':   np.floating,
                     'inexact': np.inexact,
                     'complex': np.complexfloating,
                     'real':    (np.integer, np.floating),
                     'number':  np.number,
                     'bool':    np.bool_,
                     'any':     np.generic}
def numpy_type(type_id):
    '''
    numpy_type(type) yields a valid numpy type for the given type argument. The type argument may be
      a numpy type such as numpy.bool, or it may be a string that labels a numpy type.

    Valid numpy type strings include:
      * 'int'
      * 'float'
      * 'real'
      * 'complex'
      * 'number'
      * 'bool'
      * 'any'
    '''
    if is_str(type_id):                                    return _numpy_type_names[type_id.lower()]
    elif np.issubdtype(type_id, np.generic):               return type_id
    elif type_id in six.integer_types:                     return _numpy_type_names['int']
    elif type_id is float:                                 return _numpy_type_names['float']
    elif type_id is numbers.Real:                          return _numpy_type_names['real']
    elif type_id is complex or type_id is numbers.Complex: return _numpy_type_names['complex']
    elif type_id is bool:                                  return _numpy_type_names['bool']
    elif type_id is None:                                  return _numpy_type_names['any']
    else: raise ValueError('Could not deduce numpy type for %s' % type_id)

def is_nparray(u, dtype=None, dims=None):
    '''
    is_nparray(u) yields True if u is an instance of a numpy array and False otherwise.
    is_nparray(u, dtype) yields True if is_array(u) and the dtype of u is a sub-dtype of the given
      dtype.
    is_nparray(u, dtype, dims) yields True if is_array(u, dtype) and the number of dimensions is
      equal to dims (note that dtype may be set to None for no dtype requirement).
    
    Note that either dims or dtype may be None to indicate no requirement; additionally, either may
    be a tuple to indicate that the dtype or dims may be any of the given values.

    See also: is_npscalar, is_npvector, is_npmatrix, is_array, is_scalar, is_vector, is_matrix
    '''
    if   is_quantity(u):                return is_array(mag(u), dtype=dtype, dims=dims)
    elif not isinstance(u, np.ndarray): return False
    # it's an array... check dtype
    if dtype is not None:
        if is_str(dtype): dtype = numpy_type(dtype)
        if isinstance(dtype, types.TupleType):
            if not any(np.issubdtype(u.dtype, d) for d in dtype):
                return False
        elif not np.issubdtype(u.dtype, np.dtype(dtype).type):
            return False
    # okay, the dtype is fine; check the dims
    if dims is None: return True
    if isinstance(dims, types.TupleType):
        return len(u.shape) in dims
    else:
        return len(u.shape) == dims
def is_npscalar(u, dtype=None):
    '''
    is_npscalar(u) yields True if u is an instance of a numpy array with 0 shape dimensions.
    is_npscalar(u, dtype) additionally requires that u have a dtype that is a sub-dtype of the given
      dtype.

    See also: is_nparray, is_npvector, is_npmatrix, is_array, is_scalar, is_vector, is_matrix
    '''
    return is_nparray(u, dtype=dtype, dims=0)
def is_npvector(u, dtype=None):
    '''
    is_npvector(u) yields True if u is an instance of a numpy array with one shape dimension.
    is_npvector(u, dtype) additionally requires that u have a dtype that is a sub-dtype of the given
      dtype.

    See also: is_nparray, is_npscalar, is_npmatrix, is_array, is_scalar, is_vector, is_matrix
    '''
    return is_nparray(u, dtype=dtype, dims=1)
def is_npmatrix(u, dtype=None):
    '''
    is_npmatrix(u) yields True if u is an instance of a numpy array with two shape dimensions.
    is_npmatrix(u, dtype) additionally requires that u have a dtype that is a sub-dtype of the given
      dtype.

    See also: is_nparray, is_npscalar, is_npvector, is_array, is_scalar, is_vector, is_matrix
    '''
    return is_array(u, dtype=dtype, dims=2)
def is_npgeneric(u, dtype=None):
    '''
    is_npgeneric(u) yields True if u is a numpy generic type object.
    is_npgeneric(u, dtype) yields True if u is a numpy generic type object and u's type is a subdtype
      of the given dtype.
    '''
    if is_str(dtype): dtype = numpy_type(dtype)
    return np.issubdtype(type(u), np.generic if dtype is None else np.dtype(dtype).type)

def is_array(u, dtype=None, dims=None):
    '''
    is_array(u) is equivalent to is_nparray(np.asarray(u)), meaning is_array(u) will always yield
      True, but is_array(u, dtype, dims) may not.

    See also: is_nparray, is_npscalar, is_npvector, is_npmatrix, is_scalar, is_vector, is_matrix
    '''
    if is_quantity(u): return is_array(mag(u), dtype=dtype, dims=dims)
    else:              return is_nparray(np.asarray(u), dtype=dtype, dims=dims)
def is_scalar(u, dtype=None):
    '''
    is_scalar(u) is equivalent to is_npscalar(np.asarray(u)).
    is_scalar(u, dtype) is equivalent to is_npscalar(np.asarray(u), dtype).

    See also: is_nparray, is_npscalar, is_npvector, is_npmatrix, is_array, is_vector, is_matrix
    '''
    return is_array(u, dtype=dtype, dims=0)
def is_vector(u, dtype=None):
    '''
    is_vector(u) is equivalent to is_npvector(np.asarray(u)).
    is_vector(u, dtype) is equivalent to is_npvector(np.asarray(u), dtype).

    See also: is_nparray, is_npscalar, is_npvector, is_npmatrix, is_array, is_scalar, is_matrix
    '''
    return is_array(u, dtype=dtype, dims=1)
def is_matrix(u, dtype=None):
    '''
    is_matrix(u) is equivalent to is_npmatrix(np.asarray(u)).
    is_matrix(u, dtype) is equivalent to is_npmatrix(np.asarray(u), dtype).

    See also: is_nparray, is_npscalar, is_npvector, is_npmatrix, is_array, is_scalar, is_vector
    '''
    return is_array(u, dtype=dtype, dims=2)
    
def is_int(arg):
    '''
    is_int(x) yields True if x is an integer object and False otherwise; integer objects include the
      standard Python integer types as well as numpy single integer arrays (i.e., where
      x.shape == ()) and quantities with integer magnitudes.
    '''
    return (is_int(mag(arg)) if is_quantity(arg)                   else
            True             if isinstance(arg, six.integer_types) else
            is_npscalar(arg, np.int) or is_npgeneric(arg, np.int))
def is_float(arg):
    '''
    is_float(x) yields True if x is an floating-point object and False otherwise; floating-point
      objects include the standard Python float types as well as numpy single floating-point arrays
      (i.e., where x.shape == ()) and quantities with floating-point magnitudes.

    Note that is_float(x) will yield False if x is an integer or a complex number; to check for real
    (integer or floating-point) values, use is_real(); to check for inexact (floating-point or
    complex) values, use is_inexact().
    '''
    return (is_float(mag(arg)) if is_quantity(arg)       else
            True               if isinstance(arg, float) else
            is_npscalar(arg, np.floating) or is_npgeneric(arg, np.floating))
def is_inexact(arg):
    '''
    is_inexact(x) yields True if x is a number represented by floating-point data (i.e., either a
      non-integer real number or a complex number) and False otherwise.

    Note that is_float(x) will yield False if x is an integer but True if x is a complex number; to
    check for real (integer or floating-point) values, use is_real(); to check for real
    floating-point values only, use is_float().
    '''
    return (is_inexact(mag(arg)) if is_quantity(arg)                          else
            True                 if isinstance(arg, (float, numbers.Complex)) else
            is_npscalar(u, np.inexact))
def is_real(arg):
    '''
    is_real(x) yields True if x is a non-complex numeric object and False otherwise.

    Note that is_real(i) will yield True for an integer i; to check for floating-point
    representations of numbers, use is_float().
    '''
    return (is_real(mag(arg)) if is_quantity(arg)              else
            True              if isinstance(arg, numbers.Real) else
            is_npscalar(arg,(np.integer,np.floating)) or is_npgeneric(arg,(np.integer,np.floating)))
def is_complex(arg):
    '''
    is_complex(x) yields True if x is a complex numeric object and False otherwise.
    '''
    return (is_complex(mag(arg)) if is_quantity(arg)                 else
            True                 if isinstance(arg, numbers.Complex) else
            is_npscalar(arg, np.complexfloating) or is_npgeneric(arg, np.complexfloating))
def is_number(arg):
    '''
    is_number(x) yields True if x is a numeric object and False otherwise.
    '''
    return (is_number(mag(arg)) if is_quantity(arg)              else
            True                if isinstance(arg, numbers.Real) else
            is_npscalar(arg, np.number) or is_npgeneric(arg, np.number))
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
            return id(v) in self._memoized
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
    all_keys = reduce(lambda r,s: r|s, [set(six.iterkeys(m)) for m in args])
    choose_fn = None
    if 'choose' in kwargs: choose_fn = kwargs['choose']
    kwargs = set(six.iterkeys(kwargs)) - set(['choose'])
    if len(kwargs) != 0:
        raise ValueError('Unidentified options given to merge: %s' (list(kwargs),))
    if choose_fn is None: choose_fn = _choose_last
    def _make_lambda(k, args):
        return lambda:choose_fn(k, args)
    return lazy_map({k:_make_lambda(k, [m for m in args if k in m]) for k in all_keys})
