####################################################################################################
# pimms/util.py
# Utility classes for functional programming with pimms!
# By Noah C. Benson

import inspect, types, sys, six, pint, os, numbers, warnings, base64
import collections as colls, numpy as np, pyrsistent as ps
import scipy.sparse as sps
from functools import (reduce, partial)
if six.PY2:
    try:              from cStringIO import StringIO as BytesIO
    except Exception: from StringIO  import StringIO as BytesIO
else:
    from io import BytesIO
try:              from six.moves import cPickle as pickle
except Exception: import pickle

# Python 2/3 compatibility
try: collsABC = colls.abc
except AttributeError: collsABC = colls

if six.PY2: tuple_type = types.TupleType
else:       tuple_type = tuple
if six.PY2: list_type = types.ListType
else:       list_type = list

# Setup pint / units:
units = pint.UnitRegistry()

# We want to disable the awful pint warning fo numpy if it's present:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    units.Quantity([])

# Make sure there's a pixel unit
if not hasattr(units, 'pixels'):
    units.define('pixel = [image_length] = px')

if six.PY2:
    def getargspec_py27like(f):
        '''
    getargspec_py27like(f) yields the results of calling inspect.getargspec(f) in Python 2.7, or the
      equivalent in Python 3.
    '''
        return inspect.getargspec(f)
else:
    def getargspec_py27like(f):
        '''
    getargspec_py27like(f) yields the results of calling inspect.getargspec(f) in Python 2.7, or the
      equivalent in Python 3.
    '''
        return inspect.getfullargspec(f)[:4]

def is_unit(q):
    '''
    is_unit(q) yields True if q is a pint unit or a string that names a pint unit and False
      otherwise.
    '''
    if isinstance(q, six.string_types):
        try: return hasattr(units, q)
        except Exception: return False
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
    except Exception:
        return False
def qhashform(o, consistent=False):
    '''
    qhashform(o) yields a version of o, if possible, that yields a hash that can be reproduced
      across instances. This correctly handles quantities and numpy arrays, among other things.
    '''
    c = consistent
    if is_quantity(o): return ('__#quant', qhashform(mag(o), consistent=c), str(o.u))
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.dtype(np.number).type):
        if consistent:
            return ('__#ndarray', base64.b64encode(o.tobytes()).decode('utf-8'))
        else:
            return ('__#ndarray', o.tobytes())
    elif is_set(o):
        return ('__#set', tuple(sorted([qhashform(x, consistent=c) for x in o])))
    elif is_map(o):
        if consistent:
            oo = sorted([qhashform(u, True) for u in six.iteritems(o)],
                        key=lambda kv:kv[0])
            return ('__#dict', tuple(oo))
        else:
            return ps.pmap({qhashform(k, consistent=c): qhashform(v, consistent=c)
                            for (k,v) in six.iteritems(o)})
    elif is_str(o): return o
    elif hasattr(o, '__iter__'): return tuple([qhashform(u, consistent=c) for u in o])
    else: return o
def qhash(o):
    '''
    qhash(o) is a hash function that operates like hash(o) but attempts to, where possible, hash
      quantities in a useful way. It also correctly handles numpy arrays and various other normally
      mutable and/or unhashable objects.
    '''
    o = qhashform(o, consistent)
    return hash(o)
def digest(o):
    '''
    digest(o) is like qhash(o) but it produces a consistent hex digest string. This function will
      only work with objects whose consistent qhashform is JSONable. I.e., for an input object o,
      json.dumps(to_jsonable(qhashform(o, consistent=True))) must succeed.
    '''
    
    # we have to make the has ourselves; this is a pain and will fail for objects we don't 
    # know about (i.e., things not handles by qform and not convertable to JSON)
    import json, hashlib
    try: jsstr = json.dumps(to_jsonable(qhashform(o, consistent=True)))
    except Exception: jsstr = None
    if jsstr is None:
        raise ValueError('could not convert object into JSON (consistent hashing requested)')
    return hashlib.sha256(jsstr.encode('utf-8')).hexdigest()
    
_pickle_load_options = {} if six.PY2 else {'encoding': 'latin1'}
io_formats = colls.OrderedDict(
    [('numpy', {
        'match': lambda o:   (isinstance(o, np.ndarray) and
                              np.issubdtype(o.dtype, np.dtype(np.number).type)),
        'write': lambda s,o: np.save(s, o),
        'read':  lambda s:   np.load(s)}),
     ('pickle', {
        'match': lambda o:   True,
        'write': lambda s,o: pickle.dump(o, s),
        'read':  lambda s:   pickle.load(s, **_pickle_load_options)})])
def _check_io_format(obj, fmt):
    try:              return io_formats[fmt]['match'](obj)
    except Exception: return False
def _save_stream_format(stream, obj, fmt):
    fdat = io_formats[fmt]
    fdat['write'](stream, obj)
def _load_stream_format(stream, fmt):
    fdat = io_formats[fmt]
    return fdat['read'](stream)
def _save_stream(stream, obj):
    for (fmt,fdat) in six.iteritems(io_formats):
        if not _check_io_format(obj, fmt): continue
        s = BytesIO()
        try:
            _save_stream_format(s, obj, fmt)
            pickle.dump(fmt, stream)
            stream.write(s.getvalue())
        except Exception:  continue
        else:              return stream
        finally:           s.close()
    raise ValueError('unsavable object: did not match any exporters')
def _load_stream(stream):
    try:              fmt = pickle.load(stream, **_pickle_load_options)
    except Exception: raise ValueError('could not unpickle format; probably not a pimms save file')
    if not isinstance(fmt, six.string_types):
        raise ValueError('file format object is not a string; probably not a pimms save file')
    if fmt not in io_formats:
        raise ValueError('file has unrecognized format \'%s\'' % fmt)
    return _load_stream_format(stream, fmt)
def save(filename, obj, overwrite=False, create_directories=False, create_mode=0o755):
    '''
    pimms.save(filename, obj) attempts to pickle the given object obj in the filename (or stream,
      if given). An error is raised when this cannot be accomplished; the first argument is always
      returned; though if the argument is a filename, it may be a differet string that refers to
      the same file.

    The save/load protocol uses pickle for all saving/loading except when the object is a numpy
    object, in which case it is written using obj.tofile(). The save function writes meta-data into
    the file so it cannot simply be unpickled, but must be loaded using the pimms.load() function.
    Fundamentally, however, if an object can be pickled, it can be saved/loaded by pimms.

    Options:
      * overwrite (False) The optional parameter overwrite indicates whether an error should be
        raised before opening the file if the file already exists.
      * create_directories (False) The optional parameter create_directories indicates whether the
        function should attempt to create the directories in which the filename exists if they do
        not already exist.
      * create_mode (0o755) The optional parameter create_mode specifies the mode given to
        directories that are created by the save function (if any).
    '''
    if isinstance(filename, six.string_types):
        filename = os.path.expanduser(filename)
        if not overwrite and os.path.exists(filename):
            raise ValueError('save would overwrite file %s' % filename)
        if create_directories:
            dname = os.path.dirname(os.path.realpath(filename))
            if not os.path.isdir(dname): os.makedirs(dname, mode=create_mode)
        with open(filename, 'wb') as f:
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
        try:              return load(filename, ureg=None)
        except Exception: raise
        finally:
            pint._APP_REGISTRY     = orig_app_ureg
            pint._DEFAULT_REGISTRY = orig_dfl_ureg
    if isinstance(filename, six.string_types):
        filename = os.path.expanduser(filename)
        with open(filename, 'rb') as f:
            return _load_stream(f)
    else:
        return _load_stream(filename)

def _load_json(filename, to='auto'):
    '''
    _load_json(filename) yields the object represented by the json file or stream object filename.
    
    The optional argument to may be set to None to indicate that the JSON data should be returned
    verbatim rather than parsed by neuropythy's denormalize system.
    '''
    import json
    if is_str(filename):
        try:
            with gzip.open(filename, 'rt') as fl: return json.load(fl)
        except Exception: pass
        with open(filename, 'rt') as fl: return json.load(fl)
    else: return json.load(filename)
def to_jsonable(obj):
    '''
    to_jsonable(obj) attempts to convert the given obj into an object that can be encoded into
      JSON using the json package's dumps() function. Essentially, this converts persistent
      datatypes into the typical python datatypes (e.g., pyrsistent.pmap -> dict,
      pyrsistent.pvec -> list). 
    '''
    if   is_map(obj): return dict({k: to_jsonable(v) for (k,v) in six.iteritems(obj)})
    elif is_seq(obj): return [to_jsonable(u) for u in obj]
    else: return obj
def cache_filename(cache_path, params, suffix='.pp', create_mode=0o755):
    '''
    cache_filename(cache_path, params, ext) yields a path p such that p is a file in the given
      directory cache_path that is associated with the given set of params such that a repeated
      call to find_cache_file with the same arguments will return the same file for as long as
      the cache_path is not cleared. The returned file has the given extention.
    cache_filename(cache_path, params) uses the extention '.pp' (for pimms-pickle).
    '''
    import json
    if not os.path.isdir(cache_path): raise ValueError('cache_path not found: %s' % (cache_path,))
    if suffix is None: suffix = ''
    # for this to work we need to be able to hash h...
    try: h = digest(params)
    except Exception: h = None
    if h is None: raise ValueError('could not digest given parameters')
    hform = qhashform(params, True)
    # ...we need to be able to convert params into json...
    try: j = json.dumps(to_jsonable(hform))
    except Exception: j = None
    if j is None: raise ValueError('could not convert parameters into json')
    # okay, the has determines the cache subdirectory:
    hstr = h
    # see if this directory exists...
    cache_path = os.path.join(cache_path, hstr)
    if   not os.path.exists(cache_path): os.makedirs(cache_path, mode=create_mode)
    elif not os.path.isdir(cache_path):  raise ValueError('params cache to a file, not a dir')
    # okay, try to find the file:
    k = 1
    while True:
        code = '%06x' % k
        pfile = os.path.join(cache_path, 'key-' + code + '.json')
        if os.path.isfile(pfile):
            try:
                with open(pfile, 'r') as fl: s = fl.read()
                if s == j: break
            except Exception: pass
        else:
            # write out this pfile then break
            with open(pfile, 'w') as fl: fl.write(j)
            break
        k += 1
    # okay, return the filename that goes with this code
    return os.path.join(cache_path, code + suffix)

def cache_lmap(cache_path, params, lmap, create_mode=0o755, key_translator=None):
    '''
    cache_lmap(cache_path, params, lmap) yields a lazy-map whose keys and values are identical to
      the given map lmap except that the returned map attempts to load the lazy values in lmap
      from cache prior to calculating them, and saves the lmap values out to cache if they must be
      calculated.

    Note that for any (key,val) pair in lmap, if either val cannot be saved via pimms.save() or if
    key is not a valid filename, then that pair will not be cached. The optional argument
    key_translator may be given (either a map or a function) that converts lmap's keys into
    appropriate strings.

    This function catches and ignores exceptions, so it is safe to wrap a map using cache_lmap, even
    if the params are not hashable or the map's values cannot be saved via pimms.save; any such
    values are simply returned without caching.
    '''
    if not is_map(lmap): raise ValueError('cache_map given non-map object of type %s'%(type(lmap),))
    if not is_lazy_map(lmap): lmap = lazy_map(lmap)
    params = persist(params)
    ktr_key = '__pimms_cache_lmap_key_translation__'
    # if ktr is given, make sure it will work:
    if ktr_key in params: key_translator = params[ktr_key]
    if key_translator is not None:
        if pimms.is_map(key_translator):
            ktr = {k:key_translator.get(k,k) for k in six.iterkeys(lmap)}
        else: ktr = {k:key_translator(k) for k in six.iterkeys(lmap)}
        params = pimms.assoc(params, ktr_key, ps.pmap(ktr))
    else: ktr = ps.m()
    # okay, create the layered lmap:
    def _cache(k0):
        if not lmap.is_lazy(k0): return lmap[k0]
        if _cache.cdir is None:
            try:
                # find a directory for the lmap items:
                _cache.cdir = cache_filename(cache_path, params, '', create_mode=create_mode)
                # make the dir if necessary
                if not os.path.isdir(cdir): os.makedirs(cdir, mode=create_mode)
            except Exception: _cache.cdir = Ellipsis
        k = ktr.get(k0,k0)
        if _cache.cdir is Ellipsis: flnm = None
        else:
            try:
                flnm = os.path.join(_cache.cdir, k + '.pp')
                if os.path.isfile(flnm): return load(flnm)
            except Exception: flnm = None
        v = lmap[k0]
        if flnm is None: return v
        try: save(flnm, v)
        except Exception: pass
        return v
    _cache.cdir = None
    return lazy_map({k: curry(_cache, k) for k in six.iterkeys(lmap)})
def cache_fn(cache_path, params, fn, create_mode=0o755):
    '''
    cache_fn(cache_path, params, fn) yields a function that is equivalent to the given function fn
      except that the returned function first attempts to load the result of calling fn() from
      cache and returns that value if found; if not, fn() is called and the result is cached, if
      possible.

    This function catches and ignores exceptions, so it is always safe to wrap a function using
    cache_fn: even if the caching fails, the returned function will mimic fn.
    '''
    def _fn():
        if _fn._fn is None: return _fn._result
        # first, get the cache filename:
        try:
            # find a directory for the lmap items:
            cfl = cache_filename(cache_path, params, create_mode=create_mode)
            # load it if possible
            if os.path.isfile(cfl): return load(cfl)
        except Exception: cfl = Ellipsis
        x = _fn._fn()
        _fn._result = x
        _fn._fn = None # allow garbage collection
        if cfl is Ellipsis: return x
        try: save(cfl, x)
        except Exception: pass
        return x
    _fn._fn = fn
    _fn._result = None
    return _fn                

def is_str(arg):
    '''
    is_str(x) yields True if x is a string object or a 0-dim numpy array of a string and yields
      False otherwise. 
    '''
    return (isinstance(arg, six.string_types) or
            is_npscalar(arg, 'string') or
            is_npvalue(arg, 'string'))
def is_class(arg):
    '''
    is_class(x) yields True if x is a class object and False otherwise.
    '''
    return isinstance(arg, six.class_types)

# handy type translations:
_numpy_type_names = {'bool':    (np.bool_,),
                     'int':     (np.integer, np.bool_),
                     'integer': (np.integer, np.bool_),
                     'float':   (np.floating, np.integer, np.bool_),
                     'real':    (np.floating, np.integer, np.bool_),
                     'complex': (np.number,),
                     'number':  (np.number,),
                     'string':  ((np.bytes_ if six.PY2 else np.unicode_),),
                     'unicode': (np.unicode_,),
                     'bytes':   (np.bytes_,),
                     'chars':   (np.character,),
                     'object':  (np.object_,),
                     'any':     (np.generic,),}
if not six.PY2: unicode = str
def numpy_type(type_id):
    '''
    numpy_type(type) yields a tuple of valid numpy types that can represent the type specified in
      the given type argument. The type argument may be a numpy type such as numpy.signedinteger, in
      which case a tuple containing only it is returned. Alternately, it may be a string that labels
      a numpy type or a builtin type that should be translated to a numpy type (see below).

    Note that numpy_types() does note intend to reproduce the numpy type hierarchy! If you want to
    perform comparisons on the numpy hierarchy, use numpy's functions. Rather, numpy_type()
    represents types in a mathematical hierarchy, so, for example, numpy_type('real') yields the
    tuple (numpy.floating, numpy.integer, numpy.bool_) because all of these are valid real numbers.

    Valid numpy type strings include:
      * 'bool'
      * 'int' / 'integer' (integers and booleans)
      * 'float' / 'real' (integers, booleans, or floating-point)
      * 'complex' / 'number' (integers, booleans, reals, or complex numbers)
      * 'string' ('unicode' in Python 3, 'bytes' in Python 2)
      * 'unicode'
      * 'bytes'
      * 'chars' (numpy.character)
      * 'object'
      * 'any'
    Valid builtin types include:
      * int (also long in Python 2), like 'int'
      * float, like 'float'
      * complex, like 'complex'
      * str, like 'string'
      * bytes, like 'bytes',
      * unicode, like 'unicode'
      * object, like 'object'
      * None, like 'any'
    '''
    if is_str(type_id):
        return _numpy_type_names[type_id.lower()]
    elif isinstance(type_id, (list, tuple)):
        return tuple([dt for s in type_id for dt in numpy_type(s)])
    elif type_id is None:
        return _numpy_type_names['any']
    elif type_id in six.integer_types:       return _numpy_type_names['int']
    elif type_id is float:                   return _numpy_type_names['float']
    elif type_id is numbers.Real:            return _numpy_type_names['real']
    elif type_id is complex:                 return _numpy_type_names['complex']
    elif type_id is numbers.Complex:         return _numpy_type_names['complex']
    elif type_id is bool:                    return _numpy_type_names['bool']
    elif type_id is str:                     return _numpy_type_names['string']
    elif type_id is unicode:                 return _numpy_type_names['unicode']
    elif type_id is bytes:                   return _numpy_type_names['bytes']
    elif type_id is object:                  return _numpy_type_names['object']
    elif np.issubdtype(type_id, np.generic): return (type_id,)
    else: raise ValueError('Could not deduce numpy type for %s' % type_id)
_numpy_best_type_names = {'bool':    np.bool_,
                          'int':     np.integer,
                          'integer': np.integer,
                          'float':   np.floating,
                          'real':    np.floating,
                          'complex': np.complex_,
                          'number':  np.complex_,
                          'string':  (np.bytes_ if six.PY2 else np.unicode_),
                          'unicode': np.unicode_,
                          'bytes':   np.bytes_,
                          'chars':   np.character,
                          'object':  np.object_,
                          'any':     np.object_}
def numpy_best_type(type_id):
    '''
    numpy_best_type(type) yields a single valid numpy type that can best represent the type
      specified in the given type argument. The type argument may be a numpy type such as 
      numpy.signedinteger, in which case it is returned. Alternately, it may be a string
      that labels a numpy type or a builtin type that should be translated to a numpy type
      (see below).

    Valid numpy type strings include:
      * 'bool'
      * 'int' / 'integer' (integers and booleans)
      * 'float' / 'real' (integers, booleans, or floating-point)
      * 'complex' / 'number' (integers, booleans, reals, or complex numbers)
      * 'string' ('unicode' in Python 3, 'bytes' in Python 2)
      * 'unicode'
      * 'bytes'
      * 'chars' (numpy.character)
      * 'object'
      * 'any'
    Valid builtin types include:
      * int (also long in Python 2), like 'int'
      * float, like 'float'
      * complex, like 'complex'
      * str, like 'string'
      * bytes, like 'bytes',
      * unicode, like 'unicode'
      * object, like 'object'
      * None, like 'any'
    '''
    if is_str(type_id):
        return _numpy_best_type_names[type_id.lower()]
    elif isinstance(type_id, (list, tuple)):
        return tuple([dt for s in type_id for dt in numpy_type(s)])
    elif type_id is None:
        return _numpy_best_type_names['any']
    elif type_id in six.integer_types:       return _numpy_best_type_names['int']
    elif type_id is float:                   return _numpy_best_type_names['float']
    elif type_id is numbers.Real:            return _numpy_best_type_names['real']
    elif type_id is complex:                 return _numpy_best_type_names['complex']
    elif type_id is numbers.Complex:         return _numpy_best_type_names['complex']
    elif type_id is bool:                    return _numpy_best_type_names['bool']
    elif type_id is str:                     return _numpy_best_type_names['string']
    elif type_id is unicode:                 return _numpy_best_type_names['unicode']
    elif type_id is bytes:                   return _numpy_best_type_names['bytes']
    elif type_id is object:                  return _numpy_best_type_names['object']
    elif np.issubdtype(type_id, np.generic): return type_id
    else: raise ValueError('Could not deduce numpy best-type for %s' % type_id)

def is_nparray(u, dtype=None, dims=None):
    '''
    is_nparray(u) yields True if u is an instance of a numpy array and False otherwise.
    is_nparray(u, dtype) yields True if is_nparray(u) and the dtype of u is a sub-dtype of the given
      dtype.
    is_nparray(u, dtype, dims) yields True if is_nparray(u, dtype) and the number of dimensions is
      equal to dims (note that dtype may be set to None for no dtype requirement).
    
    Notes:
      * Either dims or dtype may be None to indicate no requirement; additionally, either may be a
        tuple to indicate that the dtype or dims may be any of the given values.
      * If u is a quantity, then this is equivalent to querying mag(u).

    See also: is_npscalar, is_npvector, is_npmatrix, is_array, is_scalar, is_vector, is_matrix
    '''
    if   is_quantity(u):                return is_nparray(mag(u), dtype=dtype, dims=dims)
    elif not isinstance(u, np.ndarray): return False
    # it's an array... check dtype
    if dtype is not None:
        if (not any(np.issubdtype(u.dtype, d) for d in numpy_type(dtype)) and
            not any(np.issubdtype(u.dtype, np.dtype(d)) for d in numpy_type(dtype))):
            return False
    # okay, the dtype is fine; check the dims
    if dims is None: return True
    if isinstance(dims, (tuple,list)): return len(u.shape) in dims
    else:                              return len(u.shape) == dims
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
    return is_nparray(u, dtype=dtype, dims=2)
def is_npvalue(u, dtype):
    '''
    is_npvalue(u, dtype) yields True if u is a member of the given dtype according to numpy. The
      dtype may be specified as a string (see numpy_type) or a type. Note that dtype may be None,
      'any', or np.generic, but this will always return True if so.

    Note that is_npvalue(1, 'int') will yield True, while is_npvalue(np.array(1), 'int'),
    is_npvalue(np.array([1]), 'int'), and is_npvalue([1], 'int') will yield False (because lists
    and numpy arrays of ints aren't ints).

    See also is_nparray, is_npscalar, is_scalar.
    '''
    if is_quantity(u): return is_npvalue(mag(u), dtype=dtype)
    return any(np.issubdtype(type(u), np.generic if d is None else d) for d in numpy_type(dtype))

def is_array(u, dtype=None, dims=None):
    '''
    is_array(u) is equivalent to is_nparray(np.asarray(u)), meaning is_array(u) will always yield
      True.
    is_array(u, dtype) yields True if np.asarray(u) is of the given dtype, which is looked up using
      numpy_type. If dtype is None, then no dtype requirement is applied.
    is_array(u, dtype, dims) yields True if np.asarray(u) has the given dtype has the given number
      of dimensions.

    As in is_nparray(), dtype and dims may be tuples to indicate that any of the listed values are
    acceptable.

    See also: is_nparray, is_npscalar, is_npvector, is_npmatrix, is_scalar, is_vector, is_matrix
    '''
    if   is_map(u): return False # sometimes turns into a named-column array
    elif is_quantity(u): return is_array(mag(u), dtype=dtype, dims=dims)
    elif sps.issparse(u): return is_nparray(u[[],[]].toarray(), dtype=dtype, dims=dims)
    elif is_nparray(u): return is_nparray(u, dtype=dtype, dims=dims)
    else:
        npts = numpy_best_type(dtype)
        if not isinstance(npts, (list_type,tuple_type)): npts = [npts]
        for npt in npts:
            try:              u = np.asarray(u, dtype=npt)
            except Exception: continue
            if is_nparray(u, dtype=dtype, dims=dims):
                return True
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
def is_tuple(arg):
    '''
    is_tuple(arg) yields True if arg is a tuple and False otherwise.
    '''
    return isinstance(arg, tuple_type)
def is_list(arg):
    '''
    is_list(arg) yields True if arg is a list and False otherwise.
    '''
    return isinstance(arg, list_type)
def is_set(arg):
    '''
    is_set(arg) yields True if arg is a set or frozenset and False otherwise.
    '''
    return isinstance(arg, collsABC.Set)
def curry(f, *args0, **kwargs0):
    '''
    curry(f, ...) yields a function equivalent to f with all following arguments and keyword
      arguments passed. This is much like the partial function, but yields a function instead of
      a partial object and thus is suitable for use with pimms lazy maps.
    '''
    def curried_f(*args, **kwargs): 
        return f(*(args0 + args), **merge(kwargs0, kwargs))
    return curried_f
def is_seq(arg):
    '''
    is_seq(arg) yields True if arg is a sequential collection otherwise False; i.e., it must be a
      list, tuple, persistent vector, persistent list, or numpy array.
    
    Note that strings are not considered sequences.
    '''
    return isinstance(arg, (list_type, tuple_type, ps.PVector, ps.PList)) or is_nparray(arg)
def is_pseq(arg):
    '''
    is_pseq(arg) yields True if arg is a persistent vector, a persistent list, a tuple, or a numpy
      array with its writeable flag off.
    '''
    if is_nparray(arg): return arg.flags['WRITEABLE'] == False
    return isinstance(arg, (tuple_type, ps.PVector, ps.PList))
    
def is_int(arg):
    '''
    is_int(x) yields True if x is an integer object and False otherwise; integer objects include the
      standard Python integer types as well as numpy single integer arrays (i.e., where
      x.shape == ()) and quantities with integer magnitudes.
    '''
    return (is_int(mag(arg)) if is_quantity(arg)                   else
            True             if isinstance(arg, six.integer_types) else
            is_npscalar(arg, 'int') or is_npvalue(arg, 'int'))
def is_float(arg):
    '''
    is_float(x) yields True if x is a non-complex numeric object and False otherwise. It is an alias
      for is_real(x).

    Note that is_float(i) will yield True for an integer or bool i; to check for floating-point
    representations of numbers, use is_array(x, numpy.floating) or similar.
    '''
    return (is_float(mag(arg)) if is_quantity(arg)       else
            True               if isinstance(arg, float) else
            is_npscalar(arg, 'real') or is_npvalue(arg, 'real'))
def is_real(arg):
    '''
    is_real(x) yields True if x is a non-complex numeric object and False otherwise.

    Note that is_real(i) will yield True for an integer or bool i; to check for floating-point
    representations of numbers, use is_array(x, numpy.floating) or similar.
    '''
    return (is_real(mag(arg)) if is_quantity(arg)       else
            True              if isinstance(arg, float) else
            is_npscalar(arg, 'real') or is_npvalue(arg, 'real'))
def is_inexact(arg):
    '''
    is_inexact(x) yields True if x is a number represented by floating-point data (i.e., either a
      non-integer real number or a complex number) and False otherwise.
    '''
    return (is_inexact(mag(arg)) if is_quantity(arg) else
            is_npscalar(u, np.inexact) or is_npvalue(arg, np.inexact))
def is_complex(arg):
    '''
    is_complex(x) yields True if x is a complex numeric object and False otherwise. Note that this
      includes anything representable as as a complex number such as an integer or a boolean value.
      In effect, this makes this function an alias for is_number(arg).
    '''
    return (is_complex(mag(arg)) if is_quantity(arg)                 else
            True                 if isinstance(arg, numbers.Complex) else
            is_npscalar(arg, 'complex') or is_npvalue(arg, 'complex'))
def is_number(arg):
    '''
    is_number(x) yields True if x is a numeric object and False otherwise.
    '''
    return (is_number(mag(arg)) if is_quantity(arg)              else
            is_npscalar(arg, 'number') or is_npvalue(arg, 'number'))
def is_map(arg):
    '''
    is_map(x) yields True if x implements Python's builtin Mapping class.
    '''
    return isinstance(arg, collsABC.Mapping)

class LazyPMap(ps.PMap):
    '''
    LazyPMap is an immutable map that is identical to pyrsistent's PMap, but that treats functions
    of 0 arguments, when values, as lazy values, and memoizes them as it goes.
    '''
    def __init__(self, *args, **kwargs):
        self._memoized = ps.m()
    def __repr__(self):
        s = ', '.join(['%s: %s' % (repr(k), '<lazy>' if self.is_lazy(k) else self[k])
                       for k in self.iterkeys()])
        return 'lmap({' + s + '})'
    def _examine_val(self, k, val):
        'should only be called internally'
        if not isinstance(val, (types.FunctionType, partial)): return val
        vid = id(val)
        if vid in self._memoized:
            return self._memoized[vid]
        elif [] != getargspec_py27like(val)[0]:
            return val
        else:
            val = val()
            object.__setattr__(self, '_memoized', self._memoized.set(vid, val))
            return val
    def __getitem__(self, k):
        return self._examine_val(k, ps.PMap.__getitem__(self, k))
    def iterkeys(self):
        for (k,_) in ps.PMap.iteritems(self):
            yield k
    def iteritems(self):
        for (k,v) in ps.PMap.iteritems(self):
            yield (k, self._examine_val(k, v))
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
                    if not isinstance(v, (types.FunctionType, partial)): continue
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
        if not isinstance(v, (types.FunctionType, partial)) or \
           id(v) in self._memoized or \
           [] != getargspec_py27like(v)[0]:
            return False
        else:
            return True
    def lazyfn(self, k):
        '''
        lmap.lazyfn(k) yields None if the key k is not lazy; otherwise, yields the function that
        calculates the value for k. If a value has already been cached, then None is returned.
        '''
        v = ps.PMap.__getitem__(self, k)
        if not isinstance(v, (types.FunctionType, partial)) or \
           id(v) in self._memoized or \
           [] != getargspec_py27like(v)[0]:
            return None
        else:
            return v
    def is_memoized(self, k):
        '''
        lmap.is_memoized(k) yields True if k is a key in the given lazy map lmap that is both lazy
        and already memoized.
        '''
        v = ps.PMap.__getitem__(self, k)
        if not isinstance(v, (types.FunctionType, partial)):
            return False
        else:
            return id(v) in self._memoized
    def is_normal(self, k):
        '''
        lmap.is_normal(k) yields True if k is a key in the given lazy map lmap that is neither lazy
        nor a formerly-lazy memoized key.
        '''
        v = ps.PMap.__getitem__(self, k)
        if not isinstance(v, (types.FunctionType, partial)) or [] != getargspec_py27like(v)[0]:
            return True
        else:
            return False
collsABC.Mapping.register(LazyPMap)
collsABC.Hashable.register(LazyPMap)
def _lazy_turbo_mapping(initial, pre_size):
    '''
    _lazy_turbo_mapping is a blatant copy of the pyrsistent._pmap._turbo_mapping function, except
    it works for lazy maps; this seems like the only way to fully overload PMap.
    '''
    size = pre_size or (2 * len(initial)) or 8
    buckets = size * [None]
    if not isinstance(initial, collsABC.Mapping): initial = dict(initial)
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
    if is_lazy_map(initial): return initial
    if not initial: return _EMPTY_LMAP
    return _lazy_turbo_mapping(initial, pre_size)
def lmap(initial={}, pre_size=0):
    '''
    lmap is an alias for lazy_map.
    '''
    return lazy_map(initial, pre_size)
def is_lazy_map(m):
    '''
    is_lazy_map(m) yields True if m is an instance if LazyPMap and False otherwise. Note that this
      will yield True for a pimms itable object as well as a pimms lazy map because itables respect
      laziness. To check if an object is a pimms lazy map specifically, use
      isinstance(m, pimms.LazyPMap).
    '''
    from .table import is_itable
    return isinstance(m, LazyPMap) or is_itable(m)
def is_lmap(m):
    '''
    is_lmap(m) is identical to is_lazy_map(m).
    '''
    return is_lazy_map(m)
def is_pmap(arg):
    '''
    is_pmap(x) yields True if x is a persistent map object and False otherwise. Note that this will
      yield True for any of the following types: a pyrsistent.PMap (persistent mapping object), a
      pimms lazy map, a pimms itable object. If you want to check specifically if an object is a
      pyrsistent PMap object, use isinstance(arg, pyrsistent.PMap).
    '''
    return isinstance(arg, ps.PMap) or is_lazy_map(arg)

def lazy_value_map(f, m, *args, **kwargs):
    '''
    lazy_value_map(f, mapping) yields a lazy map whose keys are the same as those of the given dict
      or mapping object and whose values, for each key k are f(mapping[k]).
    lazy_value_map(f, mapping, *args, **kw) additionally passes the given arguments to the function
      f, so in the resulting map, each key k is mapped to f(mapping[k], *args, **kw).

    If a dict object that is not persistent is passed to lazy_value_map, then a persistent copy of
    it is made for use with the lazy-map; accordingly, it's not necessary to worry about the 
    persistence of the map you pass to this function. It is, however, important to worry about the
    persistence of the values in the map you pass. If these values are mutated in-place, then the
    lazy map returned from this function could change as well.

    If the given mapping object is an ITable object, then an ITable object is returned.
    '''
    if not is_map(m): raise ValueError('Non-mapping object passed to lazy_value_map')
    if not is_lazy_map(m) and not is_pmap(m): m = ps.pmap(m)
    def curry_fn(k): return lambda:f(m[k], *args, **kwargs)
    m0 = {k:curry_fn(k) for k in six.iterkeys(m)}
    from .table import (is_itable, itable)
    return itable(m0) if is_itable(m) else lazy_map(m0)
def value_map(f, m, *args, **kwargs):
    '''
    value_map(f, mapping) yields a persistent map whose keys are the same as those of the given dict
      or mapping object and whose values, for each key k are f(mapping[k]).
    value_map(f, mapping, *args, **kw) additionally passes the given arguments to the function
      f, so in the resulting map, each key k is mapped to f(mapping[k], *args, **kw).

    Unlike lazy_value_map, this function yields either a persistent or a lazy map depending on the
    input argument mapping. If mapping is a lazy map, then a lazy map is returned; otherwise, a
    persistent non-lazy map is returned.
    '''
    if is_lazy_map(m): return lazy_value_map(f, m, *args, **kwargs)
    else:              return ps.pmap({k:f(v, *args, **kwargs) for (k,v) in six.iteritems(m)})
def key_map(f, m, *args, **kwargs):
    '''
    key_map(f, m) is equivalent to {f(k):v for (k,v) in m.items()} except that it returns a
      persistent mapping object instead of a dict. Additionally, it respects the laziness of maps
      and does not evaluate the values of a lazy map that has been passed to it.
    key_map(f, m, *args, **kwargs) uses f(k, *args, **kwargs) instead of f(k).
    '''
    if is_lazy_map(m):
        from .table import (is_itable, itable)
        def _curry_getval(k): return lambda:m[k]
        m0 = {f(k, *args, **kwargs):_curry_getval(k) for k in six.iterkeys(m)}
        return itable(m0) if is_itable(m) else lazy_map(m0)
    else:
        return ps.pmap({f(k, *args, **kwargs):v for (k,v) in six.iteritems(m)})
def flatten_maps(*args, **kwargs):
    '''
    flatten_maps(*args, **kwags) yields a tuple of the maps in the given arguments; this flattens
      over lists and iterables so long as all elements eventually yield True to is_map(el). The
      optional keyword arguments passed make up the final map.

    This funtion does not evaluate any values of any of the maps and thus implicitly respects the
    laziness of the provided maps.
    '''
    def _recur(arg, work):
        if is_map(arg): work.append(arg)
        elif is_str(arg) or not hasattr(arg, '__iter__'):
            raise ValueError('Non-map given to flatten_maps')
        else:
            for a in arg: _recur(a, work)
    res = []
    for arg in args: _recur(arg, res)
    if len(kwargs) > 0: res.append(kwargs)
    return tuple(res)
def collect(*args, **kwargs):
    '''
    collect(m1, m2, ...) yields a persistent map whose keys are the union of all keys in the given
      maps m1, m2, etc. and whose values are tuples containing each of the given maps (in provided
      order) that contain the given key. This function never evaluates the values in the maps so it
      implicitly supports laziness.

    The collect function fist passes its arguments to flatten_maps, so it is fine to pass lists or
    nested lists of maps to this function; all will be collected.
    '''
    args = flatten_maps(args, **kwargs)
    if len(args) == 0: return ps.m()
    m = {}
    for arg in args:
        for k in six.iterkeys(arg):
            if k in m: m[k].append(arg)
            else: m[k] = [arg]
    return ps.pmap({k:tuple(v) for (k,v) in six.iteritems(m)})
def _choose_first(k, vs):
    '_choose_first(k, vs) yields vs[0][k].'
    return vs[0][k]
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

    If all of the arguments passed to merge are pimms itables with the same row_count, then an
    itable object is returned instead of a lazy map.

    The following options are accepted:
    * choose (default None) specifies a function that chooses from which map, of those maps given
      to merge, the value should be drawn when keys overlap. The function is always passed two
      arguments: the key for which the conflict occurs and a list of maps containing that key; it
      should return the value to which the key should be mapped. The default uses the first map.
    '''
    from .table import (is_itable, ITable)
    # figure out the choose-fn
    choose_fn = None
    if 'choose' in kwargs:
        choose_fn = kwargs['choose']
    if len(kwargs) > 1 or (len(kwargs) > 0 and 'choose' not in kwargs):
        raise ValueError('Unidentified options given to merge: %s' % (kwargs.keys(),))
    # collect the maps...
    maps = flatten_maps(*args)
    if len(maps) == 0: return ps.m()
    elif len(maps) == 1: return maps[0]
    coll = collect(maps)
    if choose_fn is None: choose_fn = _choose_last
    def curry_choice(k, args): return lambda:choose_fn(k, args)
    resmap = lazy_map({k:curry_choice(k, v) for (k,v) in six.iteritems(coll)})
    # if they're all itables of the same size, return an itable
    if is_itable(maps[0]):
        n = maps[0].row_count
        if all(is_itable(m) and m.row_count == n for m in maps):
            return ITable(resmap, n)
    # otherwise return the lazy map
    return resmap
def lmerge(*args, **kwargs):
    '''
    lmerge(...) is equivalent to merge(...) except for two things: (1) any keyword arguments passed
      to lmerge are bundled into a map that is appended to the argument list to merge, and (2) the
      choose function for the merge always takes the left-most argument. Essentially, this means
      that the map returned from lmerge always chooses the first value it finds in the argument list
      for a particular key.

    See also merge, rmerge.
    '''
    from pimms import (is_itable, itable)
    if len(args) == 0: return ps.pmap(kwargs)
    lm = any(is_lmap(m) for m in args)
    if lm: it = any(is_itable(m) for m in args)
    else:  it = False
    m = args[0]
    if is_lmap(m): res = {k:(m.lazyfn(k) if m.is_lazy(k) else m[k]) for k in six.iterkeys(m)}
    else: res = dict(m)
    ms = args[1:] if len(kwargs) == 0 else (args[1:] + (kwargs,))
    for m in ms:
        if is_lmap(m):
            for k in six.iterkeys(m):
                if k not in res:
                    res[k] = m.lazyfn(k) if m.is_lazy(k) else m[k]
        else:
            for (k,v) in six.iteritems(m):
                if k not in res:
                    res[k] = v
    return itable(res) if it else lmap(res) if lm else ps.pmap(res)
def rmerge(*args, **kwargs):
    '''
    rmerge(...) is equivalent to merge(...) except for two things: (1) any keyword arguments passed
      to lmerge are bundled into a map that is appended to the argument list to merge, and (2) the
      choose function for the merge always takes the right-most argument. Essentially, this means
      that the map returned from rmerge always chooses the last value it finds in the argument list
      for a particular key.

    Note that rmerge is the 'traditional' form of the merge operation as found in languages like
    clojure; merge(a,b) overwrites the values of a with any found in b, and merge(a, k=v) is
    roughly equivalent to assoc(a, k=v) (though merge and assoc respect the types of their maps
    slightly differently).

    See also merge, lmerge.
    '''
    from pimms import (is_itable, itable)
    if len(args) == 0: return ps.pmap(kwargs)
    lm = any(is_lmap(m) for m in args)
    if lm: it = any(is_itable(m) for m in args)
    else:  it = False
    m = args[0]
    if is_lmap(m): res = {k:(m.lazyfn(k) if m.is_lazy(k) else m[k]) for k in six.iterkeys(m)}
    else: res = dict(m)
    ms = args[1:] if len(kwargs) == 0 else (args[1:] + (kwargs,))
    for m in ms:
        if is_lmap(m):
            for k in six.iterkeys(m):
                res[k] = m.lazyfn(k) if m.is_lazy(k) else m[k]
        else:
            for (k,v) in six.iteritems(m):
                res[k] = v
    return itable(res) if it else lmap(res) if lm else ps.pmap(res)
def is_persistent(arg):
    '''
    is_persistent(x) yields True if x is a persistent object and False if not.

    Note that this persistence can only be checked by the pimms library, so immutable/persistent
    structures not known to pimms or defined in terms of pimms's immutables library cannot be
    evaluated correctly.

    Additionally, is_persistent(x) checks only x; if x is a tuple of mutable objects, it is still
    considered a persistent object.
    '''
    from .immutable import (is_imm, imm_is_persistent)
    if is_imm(arg): return imm_is_persistent(arg)
    elif isinstance(arg, (np.generic, np.ndarray)): return not arg.flags.writeable
    elif is_quantity(arg) and isinstance(mag(arg), (np.generic, np.ndarray)):
        return not mag(arg).flags.writable
    elif is_str(arg): return True
    elif is_number(arg): return True
    elif is_pmap(arg): return True
    elif isinstance(arg, frozenset): return True
    elif isinstance(arg, (ps.PVector, ps.PSet, ps.PList, ps.PRecord)): return True
    else: return False
def persist(arg, depth=Ellipsis, on_mutable=None):
    '''
    persist(x) yields a persistent version of x if possible, or yields x itself.

    The transformations performed by persist(x) are as follows:
      * If x is an immutable object, yields x.persist()
      * If x is a set, yield a frozenset of of persist(u) for all u in x.
      * If x is a numpy array, yield imm_array(x).
      * If x is a map, yields a persistent version of x with all keys and values replaced with their
        persist()'ed form; note that this respects laziness and itables.
      * If x is a list/tuple type, yields a tuple of persist()'ed contents.
      * Otherwise, if the type of x is not recognized, yields x.

    The depth to which persist() searches the argument's elements is controlled by the depth option;
    the default behavior is to persist objects down to the point that a persistent object is found,
    at which its elements are not checked for persistence.

    Note that persist() is not guaranteed to recognize a particular object; it is intended as a
    utility function for basic functional-style and immutable data code in Python. In particular,
    it is usefl for pimms's immutable objects; dicts and mappings; pimms's lazy-maps and itables;
    pyrsistent's sets, vectors, and maps; sets and frozensets (which are both communted to
    frozensets); and anything implementing __iter__ (which are commuted to tuples). Objects that are
    not numbers or strings are considered potentially-mutable and will trigger the on_mutable case.

    The optional arguments may be passed to persist:
      * depth (default: Ellipsis) specifies the depth to which toe persist() function should search
        when persisting objects. The given argument is considered depth 0, so persist(arg, 0) will
        persist only arg and not its contents, if it is a collection. If None is given, then goes to
        any depth; if Ellipsis is given, then searches until a persistent object is found, but does
        not attempt to persist the elements of already-persistent containers (this is the default).
      * on_mutable (default: None) specifies what to do when a non-persistable object is encountered
        in the search. If None, then the object is left; if 'error', then an error is raised;
        otherwise, this must be a function that is passed the object--the return value of this
        function is the replacement used in the object returned from persist().
    '''
    from .immutable import (is_imm, imm_copy)
    # Parse the on_mutable argument
    if on_mutable is None: on_mutable = lambda x:x
    elif on_mutable == 'error':
        def _raise(x):
            raise ValueError('non-persistable: %s' % x)
        on_mutable = _raise
    if depth in (None, Ellipsis): depth_next = depth
    elif depth < 0: return arg
    else: depth_next = depth - 1
    precur = lambda x:persist(x, depth=depth_next, on_mutable=on_mutable)
    # See if we have an easy type to handle
    if is_imm(arg): return imm_copy(arg)
    if is_quantity(arg):
        (m,u) = (mag(arg), unit(arg))
        mm = precur(m)
        if mm is m: return arg
        else: return quant(mm, u)
    elif isinstance(arg, np.ndarray): return imm_array(arg)
    elif isinstance(arg, np.generic):
        x = type(arg)(arg)
        x.setflags(write=False)
        return x
    elif is_str(arg) or is_number(arg): return arg
    elif isinstance(arg, ps.PVector):
        if depth is Ellipsis or depth == 0: return arg
        for (k,v0) in zip(range(len(arg)), arg):
            v = precur(v0)
            if v0 is not v: arg = arg.set(k,v)
        return arg
    elif isinstance(arg, ps.PSet):
        if depth is Ellipsis or depth == 0: return arg
        for v0 in arg:
            v = precur(v0)
            if v0 is not v: arg = arg.discard(v0).add(v)
        return arg
    elif is_pmap(arg):
        if depth is Ellipsis or depth == 0: return arg
        return key_map(precur, value_map(precur, arg))
    elif is_map(arg):
        if not is_pmap(arg): arg = ps.pmap(arg)
        if depth == 0: return arg
        return key_map(precur, value_map(precur, arg))
    elif isinstance(arg, frozenset):
        if depth is Ellipsis or depth == 0: return frozenset(arg)
        a = [x for x in arg]
        q = [precur(x) for x in a]
        if all(ai is qi for (ai,qi) in zip(a,q)): return arg
        return frozenset(q)
    elif isinstance(arg, set):
        if depth == 0: return frozenset(arg)
        a = [x for x in arg]
        q = [precur(x) for x in a]
        if isinstance(arg, frozenset) and all(ai is qi for (ai,qi) in zip(a,q)): return arg
        return frozenset(q)
    elif hasattr(arg, '__iter__'):
        if depth == 0 or (depth is Ellipsis and isinstance(arg, tuple)): return tuple(arg)
        q = tuple(precur(x) for x in arg)
        if isinstance(arg, tuple) and all(ai is qi for (ai,qi) in zip(arg,q)): return arg
        else: return q
    elif isinstance(arg, (types.FunctionType, partial)):
        return arg
    else: return on_mutable(arg)
def dissoc(m, *args, **kw):
    '''
    dissoc(m, k1, k2...) yields a map equivalent to m without the arguments k1, k2, etc. If m is a
      mutable pythin dictionary, this is equivalent to using del m[k] for k in [k1, k2...] then
      returning m itself. If m is a persistent map, this is equivalent to calling m = m.discard(k)
      for all the keys then returning m.

    If m is not a map but is instead a list or persistent vector, this operates as if the
    list/vector is a map whose keys are integers. If an item not at the end of the vector is deleted
    then it is replaced with None instead of pop'ed.

    The optional argument error (default: False) specifies whether an error should be raised when
    one of the keys is not found; this applies to mutable or immutable arguments.
    '''
    if   len(kw) == 0: err = False
    elif len(kw) == 1: err = kw['error']
    else:              raise ValueError('unrecognized optional arguments')
    if is_pmap(m):
        if err:
            for k in args: m = m.remove(k)
        else:
            for k in args: m = m.discard(k)
        return m
    elif is_map(m):
        if err:
            for k in args: del m[k]
        else:
            for k in args:
                if k in m: del m[k]
        return m
    elif isinstance(m, tuple_type):
        return m if len(args) == 0 else tuple(dissoc(list(m), *args, error=err))
    elif isinstance(m, list_type):
        if len(args) == 0: return m
        n = len(m)
        if err:
            for k in args:
                if not is_int(k) or k < 0 or k >= n: raise KeyError(k)
        else: args = [k for k in args if is_int(k)]
        args = sorted(list(set(k if k >= 0 else n + k for k in args)))
        if err:
            for k in args:
                if k < 0 or k >= n: raise KeyError(k)
        else: args = [k for k in args if k < n and k >= 0]
        n -= 1 # just for eliminating the == n-1 in the next line
        while len(args) > 0 and args[-1] == n:
            m.pop()
            args.pop()
            n -= 1
        for k in args: m[k] = None
        return m
    elif is_nparray(m):
        args = list(args)
        if len(args) == 0: return m
        return np.asarray(dissoc(m.tolist(), *args, error=err))
    else: raise ValueError('Cannot dissoc from given type: %s' % type(m))
def assoc(m, *args, **kw):
    '''
    assoc(m, k1, v1, k2, v2...) yields a map equivalent to m without the arguments k1, k2, etc.
      associated with the values v1, v2, etc. If m is a mutable python dictionary, this is
      equivalent to using m[k] = v for all the given key-value pairs then returning m itself. If m
      is a persistent map, this is equivalent to calling m = m.set(k, v) for all the key-value pairs
      then returning m.

    Keys given in the arguments list are always associated in order, then keys in the keyword
    arguments are associated, meaning that the keyword arguments have precedence.

    Note that in addition to key-value pairs in the ordinary arguments list, assoc() also uses the
    key-value pairs given as named/keyword arguments.
    
    If m is not a map but is instead a list or persistent vector, this operates as if the
    list/vector is a map whose keys are integers. If an item is added beyond the extent of the list
    then it is extended by adding None to the list. If negative indices are given, they are always
    interpreted relative to the end of the initial input list m and not in terms of the list as it
    has grown at the point that the key-value pair is processed.
    '''
    if len(args) % 2 != 0: raise ValueError('assoc arguments must be given as key-value pairs')
    args = (u for it in [zip(args[::2],args[1::2]), six.iteritems(kw)] for u in it)
    if is_pmap(m):
        for (k,v) in args: m = m.set(k, v)
        return m
    elif is_map(m):
        for (k,v) in args: m[k] = v
        return m
    elif isinstance(m, tuple_type):
        args = list(args)
        return m if len(args) == 0 else tuple(assoc(list(m), *args, **kw))
    elif isinstance(m, list_type):
        n0 = len(m)
        n = n0
        for (k,v) in args:
            if not is_int(k): TypeError('Keys for list args must be integers')
            if k < -n: KeyError(k)
            if k < 0: k = n0 + k
            if k >= n: (n,m) = (k+1, m + [None]*(k - n + 1))
            m[k] = v
        return m
    elif is_nparray(m):
        args = list(args)
        if len(args) == 0: return m
        return np.asarray(assoc(m.tolist(), *[x for u in args for x in u]))
    else: raise ValueError('Cannot assoc given type: %s' % type(m))
    
                    
            
