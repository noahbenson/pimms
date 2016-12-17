####################################################################################################
# pimms/immutable.py
# Simple class decorator for immutable lazily-loading classes.
# By Noah C. Benson

import copy, inspect, types

# An immutable has three important values in the _neuropythy_immutable_data_ attribute of its class:
# (1) params
#     A hash from param-name to a tuple:
#     (default, transform_fn, arg_lists, check_fns, deps)
#     * default is None if required or the 1-element list [value] where value is the default-value
#     * transform_fn is None if no transform function is provided, otherwise a transformation of
#       the provided value to a possibly-new value that 'interprets' the value
#     * arg_lists is a list of argument-name-tuples for the check_fns
#     * check_fns is a list of functions that should be called with the attributes matching those
#       named in arg_lists in order to check the new value of the particular parameter
#     * deps is a list of the names of values in the immutable that depend on this param
# (2) values
#     A hash from value-name to a tuple:
#     (arg_list, calc_fn, deps)
#     * arg_list is the list of attribute names on which this value depends, and which should be
#       passed to calc_fn as arguments
#     * calc_fn is the function that actually calculates the value given the attributes in arg_list
#     * deps is the list of dependants on this particular value
# (3) checks
#     A hash of tuples, each of which describes one check function:
#     (arg_list, fn)
# (4) consts
#     A hash of tuples, each of which describes one const-value (value with no inputs)
#     (arg_lists, check_fns)
#     Where the arg_lists and check_fns are as in params.

def is_imm(obj):
    '''
    is_imm(obj) yields True if obj is an instance of an immutable class and False otherwise.
    '''
    return hasattr(type(obj), '_neuropythy_immutable_data_')
def is_imm_type(cls):
    '''
    is_imm_type(cls) yields True if cls is an immutable class and False otherwise.
    '''
    return hasattr(cls, '_neuropythy_immutable_data_')
def _imm_is_init(imm):
    dd = object.__getattribute__(imm, '__dict__')
    return '_neuropythy_immutable_is_init' in dd
def _imm_is_trans(imm):
    dd = object.__getattribute__(imm, '__dict__')
    return '_neuropythy_immutable_is_trans' in dd
def _imm_is_persist(imm):
    dd = object.__getattribute__(imm, '__dict__')
    return '_neuropythy_immutable_is_init' not in dd and '_neuropythy_immutable_is_trans' not in dd
def _imm_param_data(imm):
    return type(imm)._neuropythy_immutable_data_['params']
def _imm_value_data(imm):
    return type(imm)._neuropythy_immutable_data_['values']
def _imm_check_data(imm):
    return type(imm)._neuropythy_immutable_data_['checks']
def _imm_const_data(imm):
    return type(imm)._neuropythy_immutable_data_['consts']
def _imm_clear(imm):
    dd = object.__getattribute__(imm, '__dict__')
    for val in _imm_value_data(imm).iterkeys():
        if val in dd:
            del dd[val]
    return imm
def _imm_check(imm, names=Ellipsis):
    # runs all the checks on all the names
    all_checks = set([])
    params = _imm_param_data(imm)
    consts = _imm_const_data(imm)
    names = params.keys() + consts.keys() if names is Ellipsis             else \
            [names]                       if isinstance(names, basestring) else \
            names
    for name in names:
        if name in params:
            (_, _, arg_lists, check_fns, _) = params[name]
        elif name in consts:
            (arg_lists, check_fns) = consts[name]
        else:
            raise ValueError('Attempt to check non-existent param named \'%s\'' % name)
        for (arg_list, check_fn) in zip(arg_lists, check_fns):
            all_checks.add((tuple(arg_list), check_fn))
    # Run the checks; if anything fails, we let the exception rise
    for (arg_list, check_fn) in zip(arg_lists, check_fns):
        if not check_fn(*[getattr(imm, arg) for arg in arg_list]):
            raise RuntimeError('Failed parameter-check on values % of func %s' % (args, check_fn))
    # All checks passed!
    return imm
def _imm_default_init(self, *args, **kwargs):
    '''
    An immutable's defalt initialization function is to accept any number of dictionaries followed
    by any number of keyword args and to turn them all into the parameters of the immutable that is
    being created.
    '''
    for (k,v) in {k:v for dct in (args + (kwargs,)) for (k,v) in dct}.iteritems():
        setattr(self, k, v)
def _imm_init_getattribute(self, name):
    '''
    During the initial transient state, getattribute works on params; as soon as a non-param is
    requested, all checks are forced and the getattr switches to standard transient form.
    '''
    values = _imm_value_data(self)
    params = _imm_param_data(self)
    if name in values:
        _imm_init_to_trans(self)
        return getattr(self, name)
    elif name in params:
        dd = object.__getattribute__(self, '__dict__')
        if name in dd: return dd[name]
        else: raise RuntimeError('Required immutable parameter %s requested before set' % name)
    else:
        # if they request a required param before it's set, raise an exception; that's fine
        return object.__getattribute__(self, name)
def _imm_getattribute(self, name):
    '''
    An immutable's getattribute calculates lazy values when not yet cached in the object then adds
    them as attributes.
    '''
    if _imm_is_init(self):
        return _imm_init_getattribute(self, name)
    else:
        dd = object.__getattribute__(self, '__dict__')
        if name == '__dict__': return dd
        curval = dd.get(name, dd)
        if curval is not dd: return dd[name]
        values = _imm_value_data(self)
        if name not in values:
            return object.__getattribute__(self, name)
        (args, memfn, _) = values[name]
        value = memfn(*[getattr(self, arg) for arg in args])
        dd[name] = value
        # if this is a const, it may have checks to run
        if name in _imm_const_data(self):
            # #TODO
            # Note that there's a race condition that eventually needs to be handled here:
            # If dd[name] is set then a check fails, there may have been something that read the
            # improper value in the meantime
            try:
                _imm_check(self, [name])
            except:
                del dd[name]
                raise
        # if those pass, then we're fine
        return value
def _imm_init_setattr(self, name, value):
    '''
    An immutable's initial setattr allows only param's to be set and does not run checks on the new
    parameters until a full parameter-set has been specified, at which point it runs all checks and
    switches over to a normal setattr and getattr method.
    '''
    params = _imm_param_data(self)
    if name in params:
        tx_fn = params[name][1]
        value = value if tx_fn is None else tx_fn(value)
        # Set the value
        object.__getattribute__(self, '__dict__')[name] = value
        # No checks are run, as we're in initialization mode...
    else:
        raise TypeError(
            'Attempt to change non-parameter \'%s\' of initializing immutable' % name)
def _imm_trans_setattr(self, name, value):
    '''
    An immutable's transient setattr allows params to be set, and runs checks as they are.
    '''
    params = _imm_param_data(self)
    dd = object.__getattribute__(self, '__dict__')
    if name in params:
        (_, tx_fn, arg_lists, check_fns, deps) = params[name]
        value = value if tx_fn is None else tx_fn(value)
        old_deps = {}
        orig_value = dd[name]
        # clear the dependencies before we run the checks; save them in case the checks fail and we
        # go back to how things were...
        for dep in deps:
            if dep in dd:
                old_deps[dep] = dd[dep]
                del dd[dep]
        try:
            dd[name] = value
            for (args, check_fn) in zip(arg_lists, check_fns):
                if not check_fn(*[getattr(self, arg) for arg in args]):
                    raise RuntimeError(
                        ('Changing value of immutable attribute \'%s\'' +
                         ' caused validation failure: %s') % (name, (args, check_fn)))
            # if all the checks ran, we don't return the old deps; they are now invalid
            old_deps = None
        finally:
            if old_deps:
                # in this case, something didn't check-out, so we return the old deps and let the
                # exception ride; we also return the original value of the edited param
                for (dep,val) in old_deps.iteritems():
                    dd[dep] = val
                dd[name] = orig_value
    else:
        raise TypeError(
            'Attempt to change non-parameter member \'%s\' of transient immutable' % name)
def _imm_setattr(self, name, value):
    '''
    A persistent immutable's setattr simply does not allow attributes to be set.
    '''
    if _imm_is_persist(self):
        raise TypeError('Attempt to change parameter \'%s\' of non-transient immutable' % name)
    elif _imm_is_trans(self):
        return _imm_trans_setattr(self, name, value)
    else:
        return _imm_init_setattr(self, name, value)
def _imm_trans_delattr(self, name):
    '''
    A transient immutable's delattr allows the object's value-caches to be invalidated; a var that
    is deleted returns to its default-value in a transient immutable, otherwise raises an exception.
    '''
    (params, values) = (_imm_param_data(self), _imm_value_data(self))
    if name in params:
        dflt = params[name][0]
        if dflt is None:
            raise TypeError(
                'Attempt to reset required parameter \'%s\' of immutable' % name)
        setattr(self, name, dflt[0])
    elif name in values:
        dd = object.__getattribute__(self, '__dict__')
        if name in dd:
            del dd[name]
            if name in _imm_const_data(self): _imm_check(imm, [name])
    else:
        raise TypeError('Cannot delete non-value non-param attribute \'%s\' from immutable' % name)
def _imm_delattr(self, name):
    '''
    A persistent immutable's delattr allows the object's value-caches to be invalidated, otherwise
    raises an exception.
    '''
    if _imm_is_persist(self):
        values = _imm_value_data(self)
        if name in values:
            dd = object.__getattribute__(self, '__dict__')
            if name in dd:
                del dd[name]
                if name in _imm_const_data(self): _imm_check(imm, [name])
        else:
            raise TypeError('Attempt to reset parameter \'%s\' of non-transient immutable' % name)
    else:
        return _imm_trans_delattr(self, name)
def _imm_dir(self):
    '''
    An immutable object's dir function should list not only its attributes, but also its un-cached
    lazy values.
    '''
    dir0 = object.__dir__(self)
    values = _imm_value_data(self)
    for val in values.iterkeys():
        if val not in dir0:
            dir0.append(val)
    return dir0
def _imm_repr(self):
    '''
    The default representation function for an immutable object.
    '''
    return (type(self).__name__
            + ('(' if _imm_is_persist(self) else '*(')
            + ', '.join([k + '=' + str(v) for (k,v) in imm_params(self).iteritems()])
            + ')')
def _imm_new(cls, *args, **kwargs):
    '''
    All immutable new classes use a hack to make sure the post-init cleanup occurs.
    '''
    imm = object.__new__(cls, *args, **kwargs)
    # Note that right now imm has a normal setattr method;
    # Give any parameter that has one a default value
    params = cls._neuropythy_immutable_data_['params']
    for (p,dat) in params.iteritems():
        dat = dat[0]
        if dat: object.__setattr__(imm, p, dat[0])
    # Clear any values; they are not allowed yet
    _imm_clear(imm)
    # Note that we are initializing...
    dd = object.__getattribute__(imm, '__dict__')
    dd['_neuropythy_immutable_is_init'] = True
    # That should do it!
    return imm
def _imm__copy__(self):
    '''
    The default immutable copy operation yields a new instance of the immutable with an identical
    dictionary.
    '''
    if _imm_is_init(self): raise RuntimeError('Cannot copy an initializing immutable')
    dup = _imm_new(type(self))
    sd = object.__getattribute__(self, '__dict__')
    dd = object.__getattribute__(dup,  '__dict__')
    del dd['_neuropythy_immutable_is_init']
    for (k,v) in sd.iteritems():
        dd[k] = v
    return dup
    
def _imm_init_to_trans(imm):
    # changes state from initializing to transient; runs checks along the way
    # if all params are not yet set, this is an error
    params = _imm_param_data(imm)
    dd = object.__getattribute__(imm, '__dict__')
    if not _imm_is_init(imm):
        raise RuntimeError(
            'Attempted to change non-initializing immutable from initializing to transient')
    if not all(p in dd for p in params.iterkeys()):
        raise RuntimeError('Not all parameters were set prior to accessing values')
    # Okay, we can run the checks now; we need to remove init status, though...
    del dd['_neuropythy_immutable_is_init']
    dd['_neuropythy_immutable_is_trans'] = True
    _imm_check(imm)
    # Those passed, so we can actually change the methods now
    return imm
def _imm_trans_to_persist(imm):
    # changes state from transient to persistent
    params = _imm_param_data(imm)
    if _imm_is_init(imm):
        _imm_init_to_trans(imm)
    elif _imm_is_persist(imm):
        raise RuntimeError('Attempted to change persistent immutable from transient to persistent')
    # There are no checks to pass at this point, just gotta change the methods:
    #imm.__dict__['__setattr__'] = _imm_setattr
    #imm.__dict__['__delattr__'] = _imm_delattr
    dd = object.__getattribute__(imm, '__dict__')
    del dd['_neuropythy_immutable_is_trans']
    return imm

def imm_transient(imm):
    '''
    imm_transient(imm) yields a duplicate of the given immutable imm that is transient.
    '''
    if not is_imm(imm):
        raise ValueError('Non-immutable given to imm_transient')
    # make a duplicate immutable that is in the transient state
    dup = copy.copy(imm)
    if _imm_is_init(imm):
        # this is an initial-state immutable...
        _imm_init_to_trans(dup)
    elif _imm_is_persist(imm):
        # it's persistent; re-transient-ize this new one
        dd = object.__getattribute__(dup, '__dict__')
        dd['_neuropythy_immutable_is_trans'] = True
    return dup
def imm_persist(imm):
    '''
    imm_persist(imm) turns imm from a transient into a persistent immutable and returns imm. If imm
    is already persistent, then it is simply returned.
    '''
    if not is_imm(imm):
        raise ValueError('imm_persist given non-immutable')
    if not _imm_is_persist(imm):
        _imm_trans_to_persist(imm)
    return imm
def imm_copy(imm, **kwargs):
    '''
    imm_copy(imm, a=b, c=d...) yields a persisent copy of the immutable object imm that differs from
    imm only in that the parameters a, c, etc. have been changed to have the values b, d, etc.
    '''
    if not is_imm(imm):
        raise ValueError('Non-immutable given to imm_copy')
    dup = copy.copy(imm)
    dd = object.__getattribute__(dup, '__dict__')
    if _imm_is_persist(dup):
        # we update values directly then recompute checks and invalidate cache
        all_checks = set([])
        all_deps = set([])
        params = _imm_param_data(imm)
        for (p,v) in kwargs.iteritems():
            if p not in params:
                raise ValueError('attempt to set non-parameter \'%s\' in imm_copy()' % p)
            (_, tx_fn, arg_lists, check_fns, deps) = params[p]
            for (arg_list, check_fn) in zip(arg_lists, check_fns):
                all_checks.add((tuple(arg_list), check_fn))
            all_deps |= set(deps)
            dd[p] = v if tx_fn is None else tx_fn(v)
        # now invalidate the deps
        for dep in all_deps:
            if dep in dd:
                del dd[dep]
        # now run the tests
        for (arg_list, check_fn) in all_checks:
            if not check_fn(*[getattr(dup, arg) for arg in arg_list]):
                raise ValueError(
                    'Requirement \'%s%s\' failed when copying immutable' % (check_fn, arg_list))
    elif _imm_is_trans(dup):
        # we set values then want it persisted...
        for (p,v) in kwargs.iteritems(): setattr(dup, p, v)
        _imm_trans_to_persist(dup)
    else:
        # this is an initial-state immutable...
        for (p,v) in kwargs.iteritems(): setattr(dup, p, v)
        _imm_init_to_trans(dup)
        _imm_trans_to_persist(dup)
    return dup
def imm_params(imm):
    '''
    imm_params(imm) yields a dictionary of the parameters of the immutable object imm.
    '''
    return {p: getattr(imm, p) for p in _imm_param_data(imm).iterkeys()}
def imm_values(imm):
    '''
    imm_values(imm) yields a dictionary of the values of the immutable object imm. Note that this
    forces all of the values to be reified, so only use it if you want to force execution of all
    lazy values.
    '''
    return {p: getattr(imm, p) for p in _imm_value_data(imm).iterkeys()}
def imm_dict(imm):
    '''
    imm_dict(imm) yields a persistent dictionary of the params and values of the immutable
    object im. Note that this forces all of the values to be reified, so only use it if you want to
    force execution of all lazy values.
    '''
    immd = dict(**imm_params(imm))
    for (k,v) in imm_values.iteritems():
        immd[k] = v
    return immd
def imm_is_persistent(imm):
    '''
    imm_is_persistent(imm) yields True if imm is a persistent immutable object, otherwise False.
    '''
    return is_imm(imm) and _imm_is_persist(imm)
def imm_is_transient(imm):
    '''
    imm_is_transient(imm) yields True if imm is a transient immutable object, otherwise False.
    '''
    return is_imm(imm) and not _imm_is_persist(imm)

def value(f):
    '''
    The @value decorator, usable in an immutable class (see immutable), specifies that the following
    function is actually a calculator for a lazy value. The function parameters are the attributes
    of the object that are part of the calculation.
    '''
    (args, varargs, kwargs, dflts) = inspect.getargspec(f)
    if varargs is not None or kwargs is not None or dflts:
        raise ValueError('Values may not accept variable, variadic keyword, or default arguments')
    f._neuropythy_immutable_data_ = {}
    f._neuropythy_immutable_data_['is_value'] = True
    f._neuropythy_immutable_data_['inputs'] = args
    f._neuropythy_immutable_data_['name'] = f.__name__
    f = staticmethod(f)
    return f
def param(f):
    '''
    The @param decorator, usable in an immutable class (see immutable), specifies that the following
    function is actually a transformation on an input parameter; the parameter is required, and is 
    set to the value returned by the function decorated by the parameter; i.e., if you decorate the
    function abc with @param, then imm.abc = x will result in imm's abc attribute being set to the
    value of type(imm).abc(x).
    '''
    (args, varargs, kwargs, dflts) = inspect.getargspec(f)
    if varargs is not None or kwargs is not None or dflts:
        raise ValueError('Params may not accept variable, variadic keyword, or default arguments')
    if len(args) != 1:
        raise ValueError('Parameter transformation functions must take exactly one argument')
    f._neuropythy_immutable_data_ = {}
    f._neuropythy_immutable_data_['is_param'] = True
    f._neuropythy_immutable_data_['name'] = f.__name__
    f = staticmethod(f)
    return f
def option(default_value):
    '''
    The @option(x) decorator, usable in an immutable class (see immutable), is identical to the
    @param decorator except that the parameter is not required and instead takes on the default
    value x when the immutable is created.
    '''
    def _option(f):
        (args, varargs, kwargs, dflts) = inspect.getargspec(f)
        if varargs is not None or kwargs is not None or dflts:
            raise ValueError(
                'Options may not accept variable, variadic keyword, or default arguments')
        if len(args) != 1:
            raise ValueError('Parameter transformation functions must take exactly one argument')
        f._neuropythy_immutable_data_ = {}
        f._neuropythy_immutable_data_['is_param'] = True
        f._neuropythy_immutable_data_['default_value'] = default_value
        f._neuropythy_immutable_data_['name'] = f.__name__
        f = staticmethod(f)
        return f
    return _option
def require(f):
    '''
    The @require decorator, usable in an immutable class (see immutable), specifies that the
    following function is actually a validation check on the immutable class. These functions
    will appear as static members of the class and get called automatically when the relevant
    data change. Daughter classes can overload requirements to change them, or may add new
    requirements with different function names.
    '''
    (args, varargs, kwargs, dflts) = inspect.getargspec(f)
    if varargs is not None or kwargs is not None or dflts:
        raise ValueError(
            'Requirements may not accept variable, variadic keyword, or default arguments')
    f._neuropythy_immutable_data_ = {}
    f._neuropythy_immutable_data_['is_check'] = True
    f._neuropythy_immutable_data_['inputs'] = args
    f._neuropythy_immutable_data_['name'] = f.__name__
    f = staticmethod(f)
    return f

def _imm_trans_clos(edges):
    closure = {k: set([]) for e in edges for k in e}
    for (src,dst) in edges:   closure[src].add(dst)
    for (src,dst) in closure.iteritems():
        if len(dst) == 0:
            closure[src] = []
    running = True
    while running:
        running = False
        for (src,dsts) in closure.iteritems():
            if isinstance(dsts, list): continue
            new_dsts = dsts.copy()
            for dst in dsts:
                further_dsts = closure[dst]
                if isinstance(further_dsts, list):
                    new_dsts |= set(further_dsts)
                else:
                    new_dsts |= further_dsts
                    running = True
            closure[src] = list(dsts) if new_dsts == dsts else new_dsts
    return set([(src,dst) for (src,dsts) in closure.iteritems() for dst in dsts])
def _imm_resolve_deps(cls):
    '''
    _imm_resolve_deps(imm_class) resolves the dependencies of the given immutable class imm_class
    and edits the immutable metadata appropriately.
    '''
    dat = cls._neuropythy_immutable_data_
    params = dat['params']
    values = dat['values']
    consts = dat['consts']
    checks = dat['checks']
    members = params.keys() + values.keys()
    mem_ids = {k:i for (i,k) in enumerate(members)}
    # make sure that every input that's not already a value or param becomes a param:
    all_inputs = [v[0] for v in values.itervalues()] + [c[0] for c in checks.itervalues()]
    all_inputs = set([i for inp in all_inputs for i in inp])
    extra_inputs = [i for i in all_inputs if i not in mem_ids]
    for i in extra_inputs:
        params[i] = (None, None, [], [], [])
        mem_ids[i] = len(members)
        members.append(i)
    # create a graph of the dependencies:
    dep_edges = set([])
    for (v,(inputs,_,_)) in values.iteritems():
        for i in inputs:
            dep_edges.add((mem_ids[v], mem_ids[i]))
    # get the transitive closure...
    deps = _imm_trans_clos(dep_edges)
    # we can put all the param and value deps into their appropriate places now
    for (dependant, dependency) in deps:
        if dependency is dependant:
            raise RuntimeError('circular dependency in immutable: value \'%s\'' % dependant)
        (mdpcy, mdpdt) = (members[dependency], members[dependant])
        if mdpcy in params:
            params[mdpcy][4].append(mdpdt)
        elif mdpcy in values:
            values[mdpcy][2].append(mdpdt)
    # last major task is to setup the checks
    deps2params = {v: set([]) for v in values.iterkeys()}
    for (p,pd) in params.iteritems():
        for v in pd[4]:
            deps2params[v].add(p)
    deps2consts = {v: set([]) for v in values.iterkeys()}
    for c in consts.iterkeys():
        deps = values[c][2]
        for v in deps:
            deps2consts[v].add(c)
    for (c,(arg_list,check_fn)) in checks.iteritems():
        param_list = set([])
        const_list = set([])
        for a in arg_list:
            if a in params: param_list.add(a)
            elif a in values:
                if a in consts: const_list.add(a)
                else:
                    param_list |= deps2params[a]
                    const_list |= deps2consts[a]
            else:
                raise RuntimeError('requirement %s requested non-member: %s' % (c, a))
        for p in param_list:
            params[p][2].append(arg_list)
            params[p][3].append(check_fn)
        for c in const_list:
            consts[p][0].append(arg_list)
            consts[p][1].append(check_fn)
    # That's it; all data should be built at this point
    return cls
def _imm_merge_class(cls, parent):
    '''
    _imm_merge_class(imm_class, parent) updates the given immutable class imm_class to have the
    appropriate attributes of its given parent class. The parents should be passed through this
    function in method-resolution order.
    '''
    # If this is not an immutable parent, ignore it
    if not hasattr(parent, '_neuropythy_immutable_data_'): return cls
    # otherwise, let's look at the data
    cdat = cls._neuropythy_immutable_data_
    pdat = parent._neuropythy_immutable_data_
    # for params, values, and checks, we add them to cls only if they do not already exist in cls
    cparams = cdat['params']
    for (param, (dflt, tx_fn, arg_lists, check_fns, deps)) in pdat['params'].iteritems():
        if param not in cparams:
            cparams[param] = (dflt, tx_fn, [], [], [])
    cvalues = cdat['values']
    cconsts = cdat['consts']
    for (value, (arg_list, calc_fn, deps)) in pdat['values'].iteritems():
        if value not in cvalues:
            cvalues[value] = (arg_list, calc_fn, [])
            if len(arg_list) == 0:
                cconsts[value] = ([], [])
    cchecks = cdat['checks']
    for (check, (arg_list, check_fn)) in pdat['checks'].iteritems():
        if check not in cchecks:
            cchecks[check] = (arg_list, check_fn)
    # That's it for now
    return cls
def _annotate_imm(cls):
    '''
    _annotate_imm(cls) crawls the class members and adds the _neuropythy_immutable_data_ attribute
    (a dict) with the following items to the class's attributes:
      * 'params'
      * 'values'
      * 'checks'
    '''
    dat = {}
    cls._neuropythy_immutable_data_ = dat
    # First, crawl this class's static members to see what has been added...
    mems = inspect.getmembers(cls)
    immmems = [m for m in mems
               if isinstance(m[1], types.FunctionType)
               if hasattr(m[1], '_neuropythy_immutable_data_')]
    dat['params'] = {}
    dat['values'] = {}
    dat['checks'] = {}
    dat['consts'] = {}
    for (fname,f) in immmems:
        if 'is_param' in f._neuropythy_immutable_data_:
            dat['params'][f.__name__] = (
                None if 'default_value' not in dat else (dat['default_value'],),
                f, [], [], [])
        elif 'is_value' in f._neuropythy_immutable_data_:
            inputs = f._neuropythy_immutable_data_['inputs']
            dat['values'][f.__name__] = (inputs, f, [])
            if len(inputs) == 0:
                dat['consts'][f.__name__] = ([], [])
        elif 'is_check' in f._neuropythy_immutable_data_:
            dat['checks'][f.__name__] = (f._neuropythy_immutable_data_['inputs'], f)
    # That grabs all the relevant immutable data; we now have two tasks:
    # (1) merge with parent immutable classes to generate final value/param/check lists
    # --- to do this, we start by getting the class hierarchy
    mro = [c for c in inspect.getmro(cls) if c is not cls and is_imm_type(c)]
    # --- now walk through, merging each
    for parent in mro: _imm_merge_class(cls, parent)
    # (2) resolve dependencies
    _imm_resolve_deps(cls)
    return cls

def immutable(cls):
    '''
    The @immutable decorator makes an abstract type out of the decorated class that overloads
    __new__ to create interesting behavior consistent with immutable data types. The following
    decorators may be used inside of the decorated class to define immutable behavior:
      * @value indicates that the following function is really a value that should be calculated
        and stored as a value of its arguments. The arguments should not start with self and 
        should instead be named after other values from which it is calculated. If there are no
        arguments, then the returned value is a constant. Note that self is not an argument to this
        function.
      * @param indicates that the following function is really a variable that should be checked
        by the decorated function. Params are settable as long as the immutable object is transient.
        The check function decorated by @param() is actually a transformation function that is
        called every time the parameter gets changed; the actual value given to which the param is
        set is the value returned by this function. The function may raise exceptions to flag
        errors. Note that self is not an argument to this function. All parameters are required for
        an instantiated object; this means that all parameters must either be provided as values or
        options of implementing classes or must be assigned in the constructor.
      * @option(x) indicates that the following function is really an optional value; the syntax and
        behavior of @option is identical to @param except that @option(x) indicates that, if not
        provided, the parameter should take the value x, while @param indicates that an exception
        should be raised.
      * @require indicates that the following function is a requirement that should be run on the
        given arguments (which should name params/options/values of the class). Note that self is
        an argument to the function. If the function yields a truthy value, then the requirement is
        considered to be met; if it raises an exception or yields a non-trithy value (like None or
        []), then the requirement is not met and the object is considered invalid.
    In immutable objects, the functions defined by @require decorators are not instantiated; they
    may, however, be overloaded and called back to the parent class.
    '''
    # Annotate the class!
    cls = _annotate_imm(cls)
    # The attributes we want to make sure get set:
    auto_members = (('__getattribute__', _imm_getattribute),
                    ('__setattr__',      _imm_setattr),
                    ('__delattr__',      _imm_delattr),
                    ('__copy__',         _imm__copy__),
                    ('__deepcopy__',     _imm__copy__))
    for (name, fn) in auto_members: setattr(cls, name, types.MethodType(fn, None, cls))
    # __new__ is special...
    @staticmethod
    def _custom_new(c, *args, **kwargs):
        return _imm_new(c, *args, **kwargs)
    setattr(cls, '__new__', _custom_new)
    # and the attributes we set only if they haven't been specified
    optl_members = (('is_persistent',    _imm_is_persist),
                    ('is_transient',     _imm_is_trans),
                    ('persist',          imm_persist),
                    ('transient',        imm_transient),
                    ('copy',             imm_copy),
                    ('params',           imm_params),
                    ('values',           imm_values),
                    ('todict',           imm_dict))
    for (name, fn) in optl_members:
        if not hasattr(cls, name):
            setattr(cls, name, types.MethodType(fn, None, cls))
    # and the attributes we set if they're not overloaded from object
    initfn = _imm_default_init if not hasattr(cls, '__init__') else cls.__init__
    def _imm_init_wrapper(imm, *args, **kwargs):
        # call the init normally...
        initfn(imm, *args, **kwargs)
        # If we're still initializing after running the constructor, we need to switch to transient
        if _imm_is_init(imm): _imm_init_to_trans(imm)
        # Okay, all checks passed!
    setattr(cls, '__init__', types.MethodType(_imm_init_wrapper, None, cls))
    dflt_members = (('__dir__',          _imm_is_persist),
                    ('__repr__',         _imm_repr))
    for (name, fn) in dflt_members:
        if not hasattr(cls, name) or getattr(cls, name) == getattr(object, name):
            setattr(cls, name, types.MethodType(fn, None, cls))
    # Done!
    return cls
    
    
