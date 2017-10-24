####################################################################################################
# pimms/calculation.py
# Decorator and class definition for functional calculations.
# By Noah C. Benson

import copy, inspect, types, os, sys, re, warnings, pickle, pint, six
import pyrsistent as ps, numpy as np
from functools import reduce
from .util  import (merge, is_pmap, is_map, is_lazy_map, is_quantity, quant, mag, units, qhash,
                    save, load)
from .table import (itable, is_itable)

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

####################################################################################################
# The Calc, Plan, and IMap classes
class Calc(object):
    '''
    The Calc class encapsulates data regarding the calculation of a single set of data from a
    separate set of input data. The input parameters are referred to as afferent values and the
    output variables are referred to as efferent values.
    '''
    # This is a helper function used in __init__
    @staticmethod
    def _parse_doc(f, affs, effs):
        # pull out the documentation, make it into a map of parameter data for both afferent and
        # efferent values
        res = {'afferent': {}, 'efferent': {}}
        try:
            s = f.__doc__
            assert(isinstance(s, six.string_types))
        except:
            return res
        lines = s.split('\n')
        marks = [re.search('^\s*@\s*([a-zA-Z_]\w*)\s+(.*)$', ln) for ln in lines]
        marks = [None if m is None else (m.group(1), m.group(2)) for m in marks]
        ltype = ['@' if m is not None else 'x' if l == '' else 'c' for (l,m) in zip(lines, marks)]
        dat = [(t,m,l) for (t,m,l) in zip(ltype, marks, lines)]
        while len(dat) > 0:
            pdoc0 = next((i for (i,r) in enumerate(dat) if r[0] == '@'), None)
            if pdoc0 == None: return res
            # okay, there's at least one comment
            pdoc1 = next((i+pdoc0+1 for (i,r) in enumerate(dat[(pdoc0+1):]) if r[0] != 'c'),
                         len(dat))
            # pop the lines
            lns = dat[pdoc0:pdoc1]
            dat = dat[pdoc1:]
            pname = lns[0][1][0]
            txt = pname + ': ' + lns[0][1][1]
            for (_,_,l) in lns[1:]: txt += '\n' + l
            # add this to the appropriate document
            if pname in affs:   (dest, dnm) = (affs, 'afferent')
            elif pname in effs: (dest, dnm) = (effs, 'efferent')
            else:
                warnings.warn('document for unrecognized value %s' % pname)
                continue
            if pname in res[dnm]: res[dnm][pname] += '\n\n' + txt
            else:                 res[dnm][pname] = txt
        # That's it; we can return what's been collected
        return res
    def __init__(self, affs, f, effs, dflts, lazy=True, meta_data={}, cache=False, memoize=True):
        ff_set = set(affs)
        ff_set = ff_set.intersection(effs)
        if ff_set: raise ValueError('calc functions may not overwrite their parameters')
        object.__setattr__(self, 'afferents', affs)
        object.__setattr__(self, 'efferents', effs)
        object.__setattr__(self, 'function',  f)
        object.__setattr__(self, 'defaults',  dflts)
        object.__setattr__(self, 'lazy',      lazy)
        object.__setattr__(self, 'meta_data', ps.pmap(meta_data))
        object.__setattr__(self, 'cache', cache)
        object.__setattr__(self, 'memoize', memoize)
        object.__setattr__(self, 'name', f.__module__ + '.' + f.__name__)
        pdoc = Calc._parse_doc(f, affs, effs)
        pdoc = {k:ps.pmap(v) for (k,v) in six.iteritems(pdoc)}
        object.__setattr__(self, 'afferent_docs', pdoc['afferent'])
        object.__setattr__(self, 'efferent_docs', pdoc['efferent'])
    def __call__(self, *args, **kwargs):
        opts = merge(self.defaults, args, kwargs)
        args = []
        for name in self.afferents:
            if name not in opts:
                raise ValueError('required parameter %s not given to %s' % (name, self.name))
            args.append(opts[name])
        result = self.function(*args)
        if is_map(result):
            if len(result) != len(self.efferents) or not all(e in result for e in self.efferents):
                raise ValueError('keys returned by %s did not match calc declaration' % self.name)
        elif isinstance(result, types.TupleType):
            result = {k:v for (k,v) in zip(self.efferents, result)}
        elif len(self.efferents) == 1:
            result = {self.efferents[0]: result}
        elif not self.efferents and not result:
            result = {}
        else:
            raise ValueError(
                'Illegal return value from function call (%s): did not match efferents' % self.name)
        return result
    def __setattr__(self, k, v):
        raise TypeError('Calc objects are immutable')
    def __delattr__(self, k):
        raise TypeError('Calc objects are immutable')
    def set_meta(self, meta_data):
        '''
        node.set_meta(meta) yields a calculation node identical to the given node except that its
        meta_data attribute has been set to the given dictionary meta. If meta is not persistent,
        it is cast to a persistent dictionary first.
        '''
        if not (isinstance(meta_data, ps.PMap) or isinstance(meta_data, IMap)):
            meta_data = ps.pmap(meta_data)
        new_cnode = copy.copy(self)
        object.__setattr__(new_cnode, 'meta_data', meta_data)
        return new_cnode
    def set_defaults(self, *args, **kwargs):
        '''
        node.set_defaults(a=b...) yields a new calculation node identical to the given node except
        with the default values matching the given key-value pairs. Arguments are collapsed left-to
        right with later arguments overwriting earlier arguments.
        '''
        args = merge(self.defaults, args, kwargs)
        new_cnode = copy.copy(self)
        object.__setattr__(new_cnode, 'defaults', ps.pmap(args))
        return new_cnode
    def discard_defaults(self, *args):
        '''
        node.discard_defaults(a, b...) yields a new calculation node identical to the given node
        except that the default values for the given afferent parameters named by the arguments a,
        b, etc. have been removed. In the new node that is returned, these parameters will be
        required.
        '''
        rms = set(arg for aa in args for arg in ([aa] if isinstance(aa, six.string_types) else aa))
        new_defaults = ps.pmap({k:v for (k,v) in six.iteritems(args) if k not in rms})
        new_cnode = copy.copy(self)
        object.__setattr__(new_cnode, 'defaults', new_defaults)
        return new_cnode
    def remove_defaults(self, *args):
        '''
        node.remove_defaults(a, b...) is identical to node.discard_defaults(a, b...) except that
        it raises a KeyError if any of the given arguments are not already defaults.
        '''
        rms = set(arg for aa in args for arg in ([aa] if isinstance(aa, six.string_types) else aa))
        for arg in rms:
            if arg not in self.defaults:
                raise KeyError('{0}'.format(arg))
        return self.discard_defaults(*args)
    def tr(self, *args, **kwargs):
        '''
        calc_fn.tr(...) yields a copy of calc_fn in which the afferent and efferent values of the
        function have been translated. The translation is found from merging the list of 0 or more
        dictionary arguments given left-to-right followed by the keyword arguments. 
        '''
        d = merge(args, kwargs)
        # make a copy
        translation = copy.copy(self)
        object.__setattr__(translation, 'afferents',
                           tuple(d[af] if af in d else af for af in self.afferents))
        object.__setattr__(translation, 'efferents',
                           tuple(d[ef] if ef in d else ef for ef in self.efferents))
        object.__setattr__(translation, 'value_docs',
                           ps.pmap({kk:ps.pmap({(d[k] if k in d else k):v
                                                for (k,v) in six.iteritems(vv)})
                                    for (kk,vv) in six.iteritems(self.value_docs)}))
        fn = self.function
        def _tr_fn_wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            if isinstance(res, colls.Mapping):
                return {(d[k] if k in d else k):v for (k,v) in six.iteritems(res)}
            else:
                return res
        object.__setattr__(translation, 'function', _tr_fn_wrapper)
        return translation
class Plan(object):
    '''
    The Plan class encapsulates individual functions that require parameters as inputs and
    produce outputs in the form of named values. Plan objects can be called as functions with
    a dictionary and/or a list of keyword parameters specifying their parameters; they always return
    a dictionary of the values they calculate, even if they calculate only a single value.

    The Plan class should not be overloaded and should be instantiated using the @calculates
    decorator only; it should not be overloaded directly.
    '''
    # Restricted things:
    def __setattr__(self, k, v):
        raise TypeError('Plan objects are immutable')
    def __detattr__(self, k):
        raise TypeError('Plan objects are immutable')
    # Constructor
    def __init__(self, *args, **kwargs):
        '''
        Plan(nodes...) yields a new Plan object made out of the given keyword list of Calc
        objects, which may be preceded by any number of dictionaries (or plans) whose values are
        calc objects. For each element of the nodes list, if the object is not a Calc, an attempt is
        made to coerce the object to a Calc; if this does not work, an error is raised. Arguments
        are merged left-to-right.
        '''
        nodes = ps.pmap({name: (node if isinstance(node, Calc) else calc(node))
                         for (name,node) in six.iteritems(merge(args, kwargs))})
        if not all(isinstance(n, Calc) for n in six.itervalues(nodes)):
            raise ValueError('All arguments given to Plan must be @calc functions')
        object.__setattr__(self, 'nodes', nodes)
        # let's get the default-value arguments and make sure they all match!
        defaults = {}
        for node in six.itervalues(nodes):
            for (k,v) in six.iteritems(node.defaults):
                if k in defaults:
                    if defaults[k] != v:
                        raise ValueError(
                            'conflicting default values found for \'%s\': %s and %s' % (
                                k, v, defaults[k]))
                else:
                    defaults[k] = v
        # no error yet, so we seem to have agreeing default values at least; note them
        defaults = ps.pmap(defaults)
        object.__setattr__(self, 'defaults', defaults)
        # okay, make the dependency graph
        deps = {}
        affs = set()
        effs = {}
        for node in six.itervalues(self.nodes):
            affs = affs.union(node.afferents)
            effs.update({eff:node for eff in node.efferents})
            for eff in node.efferents:
                if eff in deps:
                    deps[eff].union(node.afferents)
                else:
                    deps[eff] = set(node.afferents)
        for aff in affs:
            if aff not in deps:
                deps[aff] = set([])
        # okay, now find its transitive closure...
        deps0 = ps.pmap({k:tuple(v) for (k,v) in six.iteritems(deps)})
        changed = True
        while changed:
            changed = False
            for (k,v) in six.iteritems(deps):
                if k in v: raise ValueError('self-loop detected in dependency graph (%s)' % k)
                new_set = v.union([depdep for dep in v for depdep in deps[dep] if depdep not in v])
                if new_set != v:
                    changed = True
                    deps[k] = new_set
        deps = ps.pmap({k:tuple(v) for (k,v) in six.iteritems(deps)})
        # alright, deps is now the transitive closure; those afferents that have no dependencies are
        # the calculation's afferent parameters and the efferents are the output values
        affs = tuple(aff for aff in affs if len(deps[aff]) == 0)
        # last, we need to know the afferent dependencies
        aff_deps = ps.pmap(
            {nd.name: tuple(set([aff
                                 for nd_aff in nd.afferents
                                 for aff in ([nd_aff] if nd_aff in affs else deps[nd_aff])
                                 if aff in affs]))
             for nd in nodes.values()})
        object.__setattr__(self, 'afferents', affs)
        object.__setattr__(self, 'efferents', ps.pmap(effs))
        object.__setattr__(self, 'dependencies', deps)
        object.__setattr__(self, 'dependency_edges', deps0)
        object.__setattr__(self, 'afferent_dependencies', aff_deps)
        # we also want to reverse the dependencies so that when an afferent value is edited, we can
        # invalidate the relevant efferent values
        dpts = {}
        for (k,v) in six.iteritems(deps):
            for u in v:
                if u in dpts: dpts[u].add(k)
                else:         dpts[u] = set([k])
        dpts = ps.pmap({k:tuple(v) for (k,v) in six.iteritems(dpts)})
        object.__setattr__(self, 'dependants', dpts)
        # finally, we need to make sure that we know when to call the non-lazy functions; this we
        # do by walking through nodes and picking out all the fully afferent dependencies of each
        reqs = {}
        zero_reqs = []
        for node in six.itervalues(nodes):
            if node.lazy:            continue               # we only care about non-lazy nodes...
            elif not node.afferents: zero_reqs.append(node) # this node depends on nothing...
            else:                                           # otherwise, find the afferent deps...
                # we make a set of all upstream afferent values then intersect it with the
                # calculation's afferent values 
                node_affs = set(node.afferents)
                for aff in node.afferents:
                    node_affs = node_affs.union(deps[aff])
                node_affs &= set(self.afferents)
                # okay, add this node to the relevant afferent requirements lists
                for aff in node_affs:
                    if aff in reqs: reqs[aff].append(node)
                    else:           reqs[aff] = [node]
        for aff in self.afferents:
            if aff not in reqs:
                reqs[aff] = []
        reqs = ps.pmap({k:tuple(v) for (k,v) in six.iteritems(reqs)})
        zero_reqs = tuple(zero_reqs)
        object.__setattr__(self, 'proactive_dependants', reqs)
        object.__setattr__(self, 'initializers', zero_reqs)
        # Okay, now that we have everything else organized, lets build the documentation for the
        # afferent and efferent data
        # Afferent docs first:
        adocs = {}
        for aff in affs:
            txt = aff
            if aff in defaults: txt += ' (default: %s)' % defaults[aff]
            for (nnm, node) in six.iteritems(nodes):
                if aff not in node.afferent_docs: continue
                txt += ('\n\n(%s) ' % nnm) + node.afferent_docs[aff]
            adocs[aff] = txt
        # Then efferents:
        edocs = {}
        for eff in effs:
            txt = eff
            for (nnm, node) in six.iteritems(nodes):
                if eff not in node.efferent_docs: continue
                txt += ('\n\n(%s) ' % nnm) + node.efferent_docs[eff]
            edocs[eff] = txt
        object.__setattr__(self, 'afferent_docs', ps.pmap(adocs))
        object.__setattr__(self, 'efferent_docs', ps.pmap(edocs))
        # Make a value for memoized data:
        object.__setattr__(self, '_memoized_data', {})
        # That's it; we should be constructed now!
    def __call__(self, *args, **kwargs):
        '''
        cplan(args...) runs the given calculation plan cplan on the given args and returns a
        dictionary of the yielded values.
        '''
        return IMap(self, merge(self.defaults, args, kwargs))
    def _check(self, calc_dict, changes=None):
        '''
        cplan._check(calc_dict) should be called only by the calc_dict object itself.
        The check method makes sure that all of the proactive methods on the calc_dict are run; if
        the optional keyword argument changes is given, then checks are only run for the list of
        changes given.
        '''
        if changes is None:
            changes = self.afferents
            # run the zero-reqs
            for req in self.initializers:
                calc_dict._run_node(req)
        else:
            # invalidate these data if needed
            dpts = set(dpt for aff in changes for dpt in self.dependants[aff])
            for dpt in dpts: del calc_dict[dpt]
        # now, make sure that proactive dependants get run
        proactives = set(node for change in changes for node in self.proactive_dependants[change])
        for node in proactives:
            calc_dict._run_node(node)
        # That's it!
    def set(self, **kwargs):
        '''
        cplan.set(a=b...) yields a new caclulation plan identical to cplan except such that the
        calculation pieces specified by the arguments have been replaced with the given
        calculations instead.
        '''
        return Plan(reduce(lambda m,(k,v): m.set(k,v), six.iteritems(kwargs), self.nodes))
    def discard(self, *args):
        '''
        cplan.discard(...) yields a new calculation plan identical to cplan except without any of
        the calculation steps listed in the arguments.
        '''
        return Plan(reduce(lambda m,k: m.discard(k), args, self.nodes))
    def remove(self, *args):
        '''
        cplan.remove(...) yields a new calculation plan identical to cplan except without any of
        the calculation steps listed in the arguments. An exception is raised if any keys are not
        found in the calc-plan.
        '''
        return Plan(reduce(lambda m,k: m.remove(k), args, self.nodes))
    def set_defaults(self, *args, **kwargs):
        '''
        cplan.set_defaults(a=b...) yields a new caclulation plan identical to cplan except such
        that the calculation default values specified by the arguments have been replaced with the
        given values instead. E.g., cplan.set_defaults(a=1) would return a new plan with a default
        value for a=1.
        '''
        d = merge(args, kwargs)
        # make a copy of this object's nodes with translations...
        nodes = ps.pmap({k:v.set_defaults(d) for (k,v) in six.iteritems(self.nodes)})
        # make a new plan with that!
        return Plan(nodes)
    def discard_defaults(self, *args):
        '''
        cplan.discard_defaults(a, b...) yields a new caclulation plan identical to cplan except 
        without default values for any of the given parameter names.
        '''
        # make a copy of this object's nodes with translations...
        nodes = ps.pmap({k:v.discard_defaults(*args) for (k,v) in six.iteritems(self.nodes)})
        # make a new plan with that!
        return Plan(nodes)        
    def remove_defaults(self, *args):
        '''
        cplan.remove_defaults(a, b...) yields a new caclulation plan identical to cplan except 
        without default values for any of the given parameter names. An exception is raised if any
        default value given is not found in cplan.
        '''
        for arg in args:
            if arg not in self.defaults:
                raise KeyError('{0}'.format(arg))
        return self.discard_defaults(*args)
    def tr(self, *args, **kwargs):
        '''
        p.tr(...) yields a copy of plan p in which the afferent and efferent values of all of the
        calc functions contained in the plan have been translated. The translation is found from
        merging the list of 0 or more dictionary arguments given left-to-right followed by the
        keyword arguments.
        '''
        d = merge(args, kwargs)
        # make a copy of this object's nodes with translations...
        nodes = ps.pmap({k:v.tr(d) for (k,v) in six.iteritems(self.nodes)})
        # make a new plan with that!
        return Plan(nodes)
    def forget(self, node=None, cache_directory=None):
        '''
        plan.forget() clears the in-memory memoized cache for the plan and returns the cache dict
          prior to clearing.
        '''
        #plan.forget(node) clears only the cache for the given node (which may be a node name) and
        #  all nodes that depend on it; plan.forget(None) is equivalent to plan.forget()
        #plan.forget(node, cache_directory) additionally deletes all directories in the given cache
        #  directory that correspond to the nodes downstream of node. The node may be None to delete
        #  all relevant cache directories.
        m = self._memoized_data
        self._memoized_data = {}
        return m
class IMap(colls.Mapping):
    '''
    The IMap class instantiates a lazy immutable mapping from both parameters and calculated
    value names to their values. IMap objects should only be created in two ways:
     (1) by passing a calculation plan a complete set of parameters, or,
     (2) by calling the using method on an existing IMap to update the parameters.
    '''
    def __init__(self, plan, afferents):
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'afferents', ps.pmap(afferents))
        object.__setattr__(self, 'efferents', ps.m())
        # We need to run the checks from the calculation
        plan._check(self)
        # otherwise, we're set!
    # IMap does not allow certain things:
    def __setattr__(self, key, val):
        raise TypeError('IMap objects are immutable')
    def __delattr__(self, key):
        raise TypeError('IMap objects are immutable')
    # The Mapping functions that must be overloaded:
    def __len__(self):
        return len(self.afferents) + len(self.plan.efferents)
    def __iter__(self):
        for k in six.iterkeys(self.afferents):
            yield k
        for k in six.iterkeys(self.plan.efferents):
            yield k
    def __getitem__(self, k):
        if k in self.afferents:
            return self.afferents[k]
        elif k not in self.plan.efferents:
            raise ValueError('Key \'%s\' not found in calc-dictionary' % k)
        else:
            # using get instead of 'if k in efferents: return ...' avoids the race-condition that
            # would otherwise cause problems when someone deletes an item from the cache using
            # delitem below
            effval = self.efferents.get(k, self)
            if effval is self:
                self._run_node(self.plan.efferents[k])
                return self.efferents[k]
            else:
                return effval
    get = colls.Mapping.get
    # We want the representation to look something like a dictionary
    def __repr__(self):
        affstr = ', '.join([repr(k) + ': ' + repr(v) for (k,v) in six.iteritems(self.afferents)])
        effstr = ', '.join([repr(k) + ': <lazy>' for k in six.iterkeys(self.plan.efferents)])
        if len(self.afferents) == 0:
            return 'imap({' + effstr + '})'
        elif len(self.plan.efferents) == 0:
            return 'imap({' + affstr + '})'
        else:
            return 'imap({' + affstr + ', ' + effstr + '})'
    # There are a few other methods we want to be careful of also:
    def __contains__(self, k):
        return (k in self.afferents) or (k in self.plan.efferents)
    def __delitem__(self, k):
        '''
        IMap's __delitem__ method allows one to clear the cached value of an efferent.
        '''
        if k in self.afferents:
            raise TypeError('Cannot delete a parameter (%s) from a IMap' % k)
        elif k in self.efferents:
            object.__setattr__(self, 'efferents', self.efferents.discard(k))
        elif k not in self.plan.efferents:
            raise TypeError('IMap object has no item named \'%s\'' % k)
        # else we don't worry about it; not yet calculated.
    @staticmethod
    def _cache(cpath, arg):
        '''
        IMap._cache(cpath, arg) is an internally-called method that saves the dict of arguments to
          cache files in the given cpath directory.
        '''
        if not os.path.isdir(cpath): os.makedirs(cpath)
        for (k,v) in six.iteritems(arg):
            save(os.path.join(cpath, k + '.pp'), v, create_directories=True, overwrite=True)
        return True
    def _uncache(self, cpath, node, ureg):
        '''
        calc._uncache(cpath, uret) is an internally called function that handles loading of cached
          data from disk given the afferent parameters.
        '''
        # load the results, one at a time; all errors can just be allowed to raise upward since
        # this is always called from in a try block (in __call__)
        result = {}
        for eff in node.efferents:
            result[eff] = load(os.path.join(cpath, eff) + '.pp', ureg=ureg)
        return result
    #indent = ' -' #dbg
    def _run_node(self, node):
        '''
        calc_dict._run_node(node) calculates the results of the given calculation node in the
        calc_dict's calculation plan and caches the results in the calc_dict. This should only
        be called by calc_dict itself internally.
        '''
        #print IMap.indent, ('Node: %s' % node.name) #dbg
        #IMap.indent = '  ' + IMap.indent #dbg
        #
        # We need to pause here and handle caching, if needed.
        res = None
        if ('memoize' in self.afferents and self.afferents['memoize']) and node.memoize:
            memdat = self.plan._memoized_data
            try:
                h = qhash({k:self.afferents[k] for k in self.plan.afferent_dependencies[node.name]})
                ho = (node.name, h)
                if ho in memdat:
                    # memoization success; no need to memoize the result after processing it
                    res = memdat[ho]
                    h = None
                    ho = None
                    #print IMap.indent, 'retrieved' #dbg
                else:
                    cpath = self.afferents['cache_directory'] \
                            if node.cache and 'cache_directory' in self.afferents else \
                            None
                    if cpath is not None:
                        ureg = self.afferents['unit_registry'] \
                               if 'unit_registry' in self.afferents else \
                               'pimms'
                        cpath = os.path.join(cpath, node.name, ('0' + str(-h)) if h < 0 else str(h))
                        try:
                            res = self._uncache(cpath, node, ureg)
                            cpath = None
                            #print IMap.indent, 'loaded cache' #dbg
                        except: pass
            except:
                # memoization failure, must run the node normally (don't memoize/cache)
                h = None
                ho = None
                res = None
        else:
            h = None
            ho = None
        # ensure we have a result
        if res is None: res = node(self)
        # process the result:
        effs = reduce(lambda m,(k,v): m.set(k,v), six.iteritems(res), self.efferents)
        object.__setattr__(self, 'efferents', effs)
        # Handle the caching if needed:
        if h is not None:
            memdat[ho] = res
            #if cpath is None: print IMap.indent, 'hashed' #dbg
            if cpath is not None:
                try:
                    self._cache(cpath, res)
                    #print IMap.indent, 'saved cache' #dbg
                except: pass
        #elif node.memoize: print IMap.indent, 'simple-calc' #dbg
        #IMap.indent = IMap.indent[2:] #dbg
    # Also, since we're a persistent object, we should follow the conventions for copying
    def set(self, *args, **kwargs):
        '''
        d.set(...) yields a copy of the IMap object d; the ... may be replaced with either
        nothing (in which case d is returned) or a list of 0 or more dictionaries followed by a lsit
        of zero or more keyword arguments. These dictionaries and keywords arguments are merged
        left-to-right; the result may contain only afferent parameters of d and replaces the values
        of d in the newly returned calc dictionary.
        '''
        args = merge(args, kwargs)
        if len(args) == 0: return self
        affs = self.afferents
        pln = self.plan
        # make sure these are all valid parameters
        if any(k not in affs for k in six.iterkeys(args)):
            raise TypeError(
                'The given key \'%s\' is not an afferent parameter of IMap object')
        # okay, we can make the change...
        new_calc_dict = copy.copy(self)
        new_affs = reduce(lambda m,(k,v): m.set(k,v), six.iteritems(args), affs)
        object.__setattr__(new_calc_dict, 'afferents', new_affs)
        # we need to run checks and delete any cache that has been invalidated.
        # The calculation's check method does this; it raises an exception if there is an error
        pln._check(new_calc_dict, changes=args)
        return new_calc_dict
    def tr(self, *args, **kwargs):
        '''
        imap.tr(...) yields a copy of the immutable map imap in which both the plan and the keys of
        the new map have been translated according to the translation given in the arguments list.
        The translation is found from merging the list of 0 or more dictionary arguments given
        left-to-right followed by the keyword arguments.
        '''
        d = merge_args(args, kwargs)
        # make a copy of the plan first:
        new_plan = self.plan.tr(d)
        # now use that plan to re-initialize ourself
        return new_plan(**{(d[k] if k in d else k):v for (k,v) in six.iteritems(self.afferents)})

####################################################################################################
# Identification functions for these types
def is_calc(arg):
    '''
    is_calc(x) yields True if x is a function that was decorated with an @calc directive and False
    otherwise.
    '''
    return isinstance(arg, Calc)
def is_plan(arg):
    '''
    is_plan(x) yields True if x is a calculation plan made with the plan function and False
    otherwise.
    '''
    return isinstance(arg, Plan)
def is_imap(arg):
    '''
    is_imap(x) yields True if x is an IMap object and False otherwise.
    '''
    return isinstance(arg, IMap)
    
####################################################################################################
# Creation function for Calc, Plan, and IMap objects
def calc(*args, **kwargs):
    '''
    @calc is a decorator that indicates that the function that follows is a calculation component;
      calculation components can be used to form Plan objects (see Plan). In this
      case, the return value is given the same name as the function.
    @calc(names...) accepts a string or list/tuple of strings that name the output values of the
      calc function. In this case, the function must return either a tuple of thes values in the
      order or a dictionary in which the keys are the same as the given names. In this case, the
      optional value lazy=False may be passed after the names to indicate that the calculation
      should be run immediately when parameters are set/changed in a calculation rather than lazily
      when requested.
    @calc(None) is a special instance in which the lazy argument is ignored, no efferent values are
      expected, and the calculation is always run when the afferent parameters are updated.
    '''
    # parse out the only accepted keywords args:
    opt_names = ['lazy', 'meta', 'cache', 'memoize']
    opt_dflts = [True,   {},     False,   True]
    (lazy, meta, cache, mem) = [kwargs.get(nm,df) for (nm,df) in zip(opt_names, opt_dflts)]
    if len(kwargs) != len([k for k in opt_names if k in kwargs]):
        raise ValueError('calc accepts only the options lazy, meta, and cache')
    if len(args) == 1 and not isinstance(args[0], six.string_types):
        if isinstance(args[0], types.FunctionType):
            f = args[0]
            effs = (f.__name__,)
            (affs, varargs, kwargs, dflts) = inspect.getargspec(f)
            if varargs or kwargs: raise ValueError('@calc functions may not accept variadic args')
            affs = tuple(affs)
            dflts = ps.pmap({} if dflts is None else
                            {k:v for (k,v) in zip(affs[-len(dflts):], dflts)})
            kwargs = {} if not kwargs else kwargs
            return Calc(affs, f, effs, dflts, **kwargs)
        elif args[0] is None:
            effs = ()
            def _calc_req(f):
                (affs, varargs, kwargs, dflts) = inspect.getargspec(f)
                if varargs or kwargs: raise ValueError('@calc functions only accept simple params')
                affs = tuple(affs)
                dflts = ps.pmap({} if dflts is None else
                                {k:v for (k,v) in zip(affs[-len(dflts):], dflts)})
                return Calc(affs, f, effs, dflts,
                            lazy=False, meta_data=meta, cache=cache, memoize=mem)
            return _calc_req
        else:
            raise ValueError('calc only accepts strings, None, or no argument')
    elif len(args) < 1:
        raise ValueError('calc should be used as a function decorator')
    elif not all(isinstance(arg, six.string_types) for arg in args):
        raise ValueError('@calc(...) requires that all arguments be keyword strings')
    else:
        effs = tuple(args)
        kwargs0 = kwargs
        def _calc(f):
            (affs, varargs, kwargs, dflts) = inspect.getargspec(f)
            if varargs or kwargs:
                raise ValueError('@calc functions may only accept simple parameters')
            affs = tuple(affs)
            dflts = ps.pmap({} if dflts is None else
                            {k:v for (k,v) in zip(affs[-len(dflts):], dflts)})
            if kwargs is None: kwargs = {}
            return Calc(affs, f, effs, dflts, **kwargs0)
        return _calc
def plan(*args, **kwargs):
    '''
    plan(name1=calcs1, name2=calc2...) yields a new calculation plan (object of type
      Plan) that is itself a constructor for the calculation dictionary that is implied by
      the given calc functionss given. The names that are given are used as identifiers for
      updating the calc plan (using the without and using methods).
    plan(arg1, arg2..., name1=calc1, name2=calc2...) additionally initializes the dictionary
      of calculations and names with the calculation plans or dictionaries given as arguments. These
      are collapsed left-to-right.
    plan(imap) yields the plan object for the given IMap imap.
    '''
    if len(args) == 1 and len(kwargs) == 0 and is_imap(args[0]):
        return args[0].plan
    adict = merge(tuple(arg.nodes if isinstance(arg, Plan) else arg for arg in args),
                  kwargs)
    return Plan(**adict)
def imap(p, *args, **kwargs):
    '''
    imap(p, args...) yields an immutable map object made from the plan object p and the given
    arguments, which may be any number of mappings followed by any number of keyword arguments,
    all of which are merged left-to-right then interpreted as the parameters of the given plan p.
    '''
    p = p if is_plan(p) else plan(p)
    params = merge(args, kwargs)
    return p(params)

####################################################################################################
# Functions for updating the immutable types.
def calc_tr(calc_fn, *args, **kwargs):
    '''
    calc_tr(calc_fn, ...) yields a copy of calc_fn in which the afferent and efferent values of the
      function have been translated. The translation is found from merging the list of 0 or more
      dictionary arguments given left-to-right followed by the keyword arguments. If the calc_fn
      that is given is not a @calc function explicitly, calc_tr will attempt to coerce it to one.
    '''
    if not is_calc(calc_fn):
        calc_fn = calc(calc_fn)
    return calc_fn.tr(*args, **kwargs)
def plan_tr(p, *args, **kwargs):
    '''
    plan_tr(p, ...) yields a copy of plan p in which the afferent and efferent values of its
      functions have been translated. The translation is found from merging the list of 0 or more
      dictionary arguments given left-to-right followed by the keyword arguments. If the plan that
      is given is not a plan object explicitly, calc_tr will attempt to coerce it to one.
    '''
    if not is_plan(p):
        p = plan(p)
    return p.tr(*args, **kwargs)
def imap_tr(imap, *args, **kwargs):
    '''
    imap_tr(m, ...) yields a copy of the immutable map m in which the keywords have been translated
      according to the given arguments. Arguments may be any number of dictionaries followed by any
      number of keyword arguments, all of which are merged left-to-right then used as the
      translation.
    '''
    if not is_imap(imap):
        raise TypeError('IMap object required of imap_tr')
    return imap.tr(*args, **kwargs)
