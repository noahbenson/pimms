####################################################################################################
# pimms/util.py
# Utility classes for functional programming with pimms!
# By Noah C. Benson

import copy, inspect, types, sys
from pysistence import make_dict
from pysistence.persistent_dict import PDict

# Python3 compatibility check:
if sys.version_info[0] == 3:
    from collections import abc as colls
else:
    import collections as colls

def is_map(arg):
    '''
    is_map(x) yields True if x implements Python's builtin Mapping class.
    '''
    return isinstance(arg, colls.Mapping)
def is_pdict(arg):
    '''
    is_pdict(x) yields True if x is a persistent dictionary object and False otherwise.
    '''
    return isinstance(arg, PDict)

class MergedMap(colls.Mapping):
    '''
    A MergedMap object should be created using the merge function; it represents a set of maps that
    have been merged in some fashion into a single map, lazily.
    '''
    def __setattr__(self, key, val):
        raise TypeError('MergedMap obects are immutable')
    def __delattr__(self, key, val):
        raise TypeError('MergedMap obects are immutable')
    def __init__(self, maps, choose_fn):
        object.__setattr__(self, 'maps', maps)
        object.__setattr__(self, 'choose_function', choose_fn)
        object.__setattr__(self, '_group_cache', None)
        object.__setattr__(self, '_final_cache', make_dict())
    def _reify_group_cache(self):
        '''
        _reify_group_cache should only be called internally by a MergedMap object.
        '''
        if self._group_cache is None:
            dd = {}
            for m in self.maps:
                for k in m.iterkeys():
                    if k in dd: dd[k].append(m)
                    else:       dd[k] = [m]
            object.__setattr__(self, '_group_cache', make_dict(dd))
    def __len__(self):
        if self._group_cache is None: self._reify_group_cache()
        return len(self._group_cache)
    def __iter__(self):
        if self._group_cache is None: self._reify_group_cache()
        return self._group_cache.iterkeys()
    def __getitem__(self, k):
        if self._group_cache is None: self._reify_group_cache()
        if k in self._final_cache:
            return self._final_cache[k]
        if k not in self._group_cache:
            raise ValueError('Key \'%s\' not found in merged-map' % k)
        else:
            val = self.choose_function(k, self._group_cache[k])
            object.__setattr__(self, '_final_cache', self._final_cache.using(**{k:val}))
            return val
    def __contains__(self, k):
        if self._group_cache is None: self._reify_group_cache()
        return k in self._group_cache
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
      arguments: the key for which the conflict occurs and a tuple of maps containing that key; it
      should return the value to which the key should be mapped. The default uses the first map.
    '''
    args = tuple(arg for arg0 in args for arg in ([arg0] if is_map(arg0) else arg0))
    if not all(is_map(arg) for arg in args):
        raise ValueError('marge requires Mapping collections')
    choose_fn = None
    if 'choose' in kwargs:
        choose_fn = kwargs['choose']
        del kwargs['choose']
    if len(kwargs) > 0:
        raise ValueError('Unidentified options given to merge: %s' % (kwargs.keys(),))
    if choose_fn is None: choose_fn = _choose_last
    return MergedMap(args, choose_fn)
