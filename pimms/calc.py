####################################################################################################
# pimms/calc.py
# Decorator and class definition for functional calculations.
# By Noah C. Benson

import copy, inspect, types
from pysistence import make_dict

# Python3 compatibility check:
if sys.version_info[0] == 3:
    from collections import abc as colls
else:
    import collections as colls

def merge_args(args, kwargs):
    '''
    merge_args(args, kwargs) yields a single dictionary that collapses all the values in
    the dictionaries in the given list of args and the given keyword-arguments dict. This function
    is intended to be used as follows:
    def some_function(*args, **kwargs):
      merged_args = merge_args(args, kwargs)
    '''
    if len(args) == 0:
        return copy.copy(kwargs)
    else:
        dd = copy.copy(args[0])
        for arg in args[1:]:
            dd.update(arg)
        dd.update(kwargs)
        return dd

class CalcNode(object):
    '''
    The CalcNode class encapsulates data regarding the calculation of a single set of data from a
    separate set of input data. The input parameters are referred to as afferent values and the
    output variables are referred to as efferent values.
    '''

    def __init__(self, affs, f, effs, lazy=True):
        ff_set = set(affs)
        ff_set.intersection(effs)
        if ff_set: raise ValueError('calc functions may not overwrite their parameters')
        object.__setattr__(self, 'afferent_values', affs)
        object.__setattr__(self, 'efferent_values', effs)
        object.__setattr__(self, 'function', f)
        object.__setattr__(self, 'lazy', lazy)

    def __call__(self, *args, **kwargs):
        arg_dict = merge_args(args, kwargs)
        result = self.function(*[arg_dict[name] for name in self.afferent_values])
        if isinstance(result, types.DictType):
            return result
        elif isinstance(result, types.TupleType):
            return {k:v for (k,v) in zip(self.efferent_values, result)}
        elif len(self.efferent_values) == 1:
            return {self.efferent_values[0]: result}
        else:
            raise ValueError('Illegal return value from function call: did not match efferents')

    def __setattr__(self, k, v):
        raise TypeError('CalcNode objects are immutable')

def calc(*args, lazy=True):
    '''
    @calc is a decorator that indicates that the function that follows is a calculation component;
      calculation components can be used to form Calculation objects (see Calculation). In this
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
    if len(args) == 1:
        if isinstance(args[0], types.FuncType):
            f = args[0]
            effs = (f.__name__,)
            (affs, varargs, kwargs, dflts) = inspect.getargspec(f)
            if varargs or kwargs or dflts:
                raise ValueError('@calc functions may only accept simple parameters')
            return CalcNode(affs, f, effs, lazy=lazy)
        elif args[0] is None:
            effs = ()
            def _calc_req(f):
                (affs, varargs, kwargs, dflts) = inspect.getargspec(f)
                if varargs or kwargs or dflts:
                    raise ValueError('@calc functions may only accept simple parameters')
                return CalcNode(affs, f, effs, lazy=False)
    elif len(args) < 1:
        raise ValueError('calc should be used as a function decorator')
    elif not all(isinstance(arg, basestring) for arg in args):
        raise ValueError('@calc(...) requires that all arguments be keyword strings')
    else:
        effs = tuple(args)
        def _calc(f):
            (affs, varargs, kwargs, dflts) = inspect.getargspec(f)
            if varargs or kwargs or dflts:
                raise ValueError('@calc functions may only accept simple parameters')
            return CalcNode(affs, f, effs, lazy=lazy)
        return _calc

class CalcDict(colls.Mapping):
    '''
    The CalcDict class instantiates a lazy immutable mapping from both parameters and calculated
    value names to their values. CalcDict objects should only be created in two ways:
     (1) by passing a calculation plan a complete set of parameters, or,
     (2) by calling the using method on an existing CalcDict to update the parameters.
    '''
    def __init__(self, calc, afferents):
        object.__setattr__(self, 'calculation', calc)
        object.__setattr__(self, 'afferents', make_dict(afferents))
        object.__setattr__(self, 'efferents', make_dict({}))
        # We need to run the checks from the calculation
        calc.check(self)
        # otherwise, we're set!

    # CalcDict does not allow certain things:
    def __setattr__(self, key, val):
        raise TypeError('CalcDict objects are immutable')

    # The Mapping functions that must be overloaded:
    def __len__(self):
        return len(self.afferents) + len(self.calculation.efferents)
    def __iter__(self):
        for k in self.afferents.iterkeys():
            yield k
        for k in self.calculation.efferents.iterkeys():
            yield k
    def __getitem__(self, k):
        if k in self.afferents:
            return self.afferents[k]
        elif k in self.efferents:
            return self.efferents[k]
        elif k in self.calculations.efferents:
            node = self.calculations.efferents[k]
            self._run_node(node)
            return self.efferents[k]
        else:
            raise ValueError('Key \'%s\' not found in calc-dictionary' % k)

    # There are a few other methods we want to be careful of also:
    def __contains__(self, k):
        return (k in self.afferents) or (k in self.calculation.efferents)
    def __delitem__(self, k):
        '''
        CalcDict's __delitem__ method allows one to clear the cached value of an efferent.
        '''
        if k in self.afferents:
            raise TypeError('Cannot delete a parameter (%s) from a CalcDict' % k)
        elif k in self.efferents:
            object.__setattr__(self, 'efferents', self.efferents.without(k))
        elif k not in self.calculation.efferents:
            raise TypeError('CalcDict object has no item named \'%s\'' % k)
        # else we don't worry about it; not yet calculated.

    def _run_node(self, node):
        '''
        calc_dict._run_node(node) calculates the results of the given calculation node in the
        calc_dict's calculation plan and caches the results in the calc_dict. This should only
        be called by calc_dict itself internally.
        '''
        res = node(self)
        effs = self.efferents.using(**res)
        object.__setattr__(self, 'efferents', effs)
        
    # Also, since we're a persistent object, we should follow the conventions for copying
    def using(self, *args, **kwargs):
        '''
        d.using(...) yields a copy of the CalcDict object d; the ... may be replaced with either
        nothing (in which case d is returned) or a list of 0 or more dictionaries followed by a lsit
        of zero or more keyword arguments. These dictionaries and keywords arguments are merged
        left-to-right; the result may contain only afferent parameters of d and replaces the values
        of d in the newly returned calc dictionary.
        '''
        args = merge_args(args, kwargs)
        if len(args) == 0: return self
        affs = self.afferents
        calc = self.calculation
        # make sure these are all valid parameters
        if any(k not in affs for k in args.iterkeys()):
            raise TypeError('The given key \'%s\' is not an afferent parameter of CalcDict object')
        # okay, we can make the change...
        new_calc_dict = copy.copy(self)
        new_affs = affs.using(**args)
        object.__setattr__(new_calc_dict, 'afferents', new_affs)
        # we need to run checks and delete any cache that has been invalidated.
        # The calculation's check method does this; it raises an exception if there is an error
        calc.check(new_calc_dict, changes=args)
        return new_calc_dict
    def without(self, *args):
        '''
        The without method of CalcDict is implemented only so that an intelligent error is raised
        when a user mistakenly believes that a CalcDict is the same as a PDict object.
        '''
        raise TypeError('CalcDict objects, unlike PDict objects, do not support the without method')
    
class Calculation(object):
    '''
    The Calculation class encapsulates individual functions that require parameters as inputs and
    produce outputs in the form of named values. Calculation objects can be called as functions with
    a dictionary and/or a list of keyword parameters specifying their parameters; they always return
    a dictionary of the values they calculate, even if they calculate only a single value.

    The Calculation class should not be overloaded and should be instantiated using the @calculates
    decorator only; it should not be overloaded directly.
    '''

    def __setattr__(self, k, v):
        raise TypeError('Calculation objects are immutable')
    def __detattr__(self, k):
        raise TypeError('Calculation objects are immutable')
    
    def __init__(self, *nodes):
        '''
        Calculation(nodes) yields a new Calculation object made out of the given list of CalcNode
        objects. For each element of the nodes list, if the object is not a CalcNode, an attempt is
        made to coerce the object to a CalcNode; if this does not work, an error is raised.
        '''
        object.__setattr__(self, 'nodes',
                           tuple(node if isinstance(node, CalcNode) else calc(node)
                                 for nn in nodes
                                 for node in (nn if hasattr(nn, '__iter__') else (nn,))))
        if not all(isinstance(n, CalcNode) for n in self.nodes):
            raise ValueError('All arguments given to Calculation must be @calc functions')
        # make the dependency graph
        deps = {}
        affs = set()
        effs = {}
        for node in self.nodes:
            affs |= node.afferents
            effs.update({eff:node for eff in node.efferents})
            for eff in node.efferents:
                if eff in deps:
                    deps[eff].union(node.afferents)
                else:
                    deps[eff] = set(node.afferents)
        # okay, now find its transitive closure...
        changed = True
        while changed:
            changed = False
            for (k,v) in deps.iteritems():
                if k in v: raise ValueError('self-loop detected in dependency graph')
                new_set = v | set(depdep for dep in v for depdepd in deps[dep] if depdep not in v)
                if new_set != v:
                    changed = True
                    deps[k] = new_set
        deps = make_dict({k:tuple(v) for (k,v) in deps.iteritems()})
        # alright, deps is now the transitive closure; those afferents that have no dependencies are
        # the calculation's afferent parameters and the efferents are the output values
        object.__setattr__(self, 'afferents', tuple(aff for aff in affs if len(deps[aff]) == 0))
        object.__setattr__(self, 'efferents', make_dict(effs))
        object.__setattr__(self, 'dependencies', deps)
        # we also want to reverse the dependencies so that when an afferent value is edited, we can
        # invalidate the relevant efferent values
        dpts = {}
        for (k,v) in deps.iteritems():
            for u in v:
                if u in dpts: dpts[u].add(k)
                else:         dpts[u] = set([k])
        dpts = make_dict({k:tuple(v) for (k,v) in dpts.iteritems()})
        object.__setattr__(self, 'dependants', dpts)
        # finally, we need to make sure that we know when to call the non-lazy functions; this we
        # do by walking through nodes and picking out all the fully afferent dependencies of each
        reqs = {}
        zero_reqs = []
        for node in nodes:
            if node.lazy:            continue               # we only care about non-lazy nodes...
            elif not node.afferents: zero_reqs.append(node) # this node depends on nothing...
            else:                                           # otherwise, find the afferent deps...
                # we make a set of all upstream afferent values then intersect it with the
                # calculation's afferent values 
                node_affs = set(node.afferents)
                for aff in node.afferents:
                    node_affs |= set(deps[aff])
                node_affs &= set(self.afferents)
                # okay, add this node to the relevant afferent requirements lists
                for aff in node_affs:
                    if aff in reqs: reqs[aff].append(node)
                    else:           reqs[aff] = [node]
        reqs = make_dict({k:tuple(v) for (k,v) in reqs.iteritems()})
        zero_reqs = tuple(zero_reqs)
        object.__setattr__(self, 'proactive_dependants', reqs)
        object.__setattr__(self, 'initializers', zero_reqs)
        # That's it; we should be constructed now!
        
    def __call__(self, *args, **kwargs):
        '''
        calcul(args...) runs the given calculation calcul on the given args and returns a
        dictionary of the yielded values.
        '''
        return CalcDict(self, merge_args(args, kwargs))

    def check(self, calc_dict, changes=None):
        '''
        calc.check(calc_dict) should be called only by the calc_dict object itself.
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

def calc_plan(*args):
    '''
    calc_plan(calcs...) yields a new calculation plan (object of type Calculation) that is itself
    a constructor for the calculation dictionary that is implied by the list of calculation
    functions (calcs...) given.
    '''
    return Calculation(*args)
