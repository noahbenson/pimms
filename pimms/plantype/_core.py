# -*- coding: utf-8 -*-
################################################################################
# pimms/plantype/_core.py
#
# Simple meta-class for lazily-calculating plantype classes.
#
# @author Noah C. Benson

# Dependencies #################################################################
import copy, types, inspect
from collections import (defaultdict, namedtuple)

from ..types import (is_str)
from ..lazydict import (ldict, fdict, is_ldict, is_fdict, assoc)
from ..calculation import (calc, plan, plandict, is_calc)

# #plantype ####################################################################
class plantype(type):
    """A metaclass that allows one to create lazy types from calculation plans.

    The `plantype` metaclass handles classes with the base-class `planobject`.
    In general, one should create a plan-object by inheriting from `planobject`,
    not by providing the `plantype` metaclass, but passing `plantype` has the
    same effect (all classes created with metaclass `plantype` will inherit from
    `planobject`).

    See `planobject` for more information.
    """
    #TODO: write the documentation.
    @staticmethod
    def _planobject__init__(self, *args, **kwargs):
        # This method acts as a default initializer for subtypes.
        r = self.__planobject_init__(*args, **kwargs)
        # Postprocess the argument.
        plan = type(self).plan
        params = dict(plan.defaults, **self.__plandict__)
        if params.keys() != plan.inputs:
            raise ValueError(f"bad parameterization of plantype {type(self)};"
                             f" expected {tuple(plan.inputs)} but found"
                             f" {tuple(params.keys())}")
        pd = plan(params)
        object.__setattr__(self, '__plandict__', pd)
    def __new__(cls, name, bases, attrs, **kwargs):
        sup = super(plantype, cls)
        # Before we go too far, let's extract the valid args from kwargs.
        initplan = kwargs.pop('plan', None)
        if len(kwargs) > 0:
            ks = tuple(kwargs.keys())
            raise ValueError(f"unsupported type options: {ks}")
        # We want to go over the attributes and make a few changes:
        # (1) We want to make the basic updates.
        for (k,v) in [('__getattr__', planobject.__getattr__),
                      ('__setattr__', planobject.__setattr__),
                      ('__new__',     planobject.__new__),
                      ('__dir__',     planobject.__dir__)]:
            if k in attrs:
                raise ValueError(f"plantype classes may not define {k}")
            else:
                attrs[k] = v
        # (2) We want to save the init function and update it to our version.
        init = attrs.get('__init__', planobject.__init__)
        attrs['__planobject_init__'] = init
        attrs['__init__'] = _planobject__init__
        # (3) Go through the bases: see if there are planobject bases already,
        #     and if not, add planobject in. As we go, collect calculations.
        calcs = {k:v for (k,v) in attrs.items() if is_calc(v)}
        found_planobj = False
        for b in bases:
            if not issubclass(b, planobject): continue
            found_planobj = True
            for (k,v) in inspect.getmembers(b):
                if k not in calcs and is_calc(v):
                    calcs[k] = v
        if not found_planobj:
            bases.append(planobject)
        # (4) We now want to create the plan from these calculations and make
        #     sure it's part of the class.
        if initplan is None:
            attrs['plan'] = plan(calcs)
        else:
            attrs['plan'] = plan(initplan, **calcs)
        # (5) Return the type with all the updated attributes.
        return sup.__new__(cls, name, bases, attrs)
    def __str__(self):
        return "plantype"
    def __repr__(self):
        return "plantype"

# #planobject ##################################################################
class planobject(metaclass=plantype):
    """Base class for objects that are based on lazy calculation plans.

    `planobject` is the base-class for all objects that use `pimms` `plan`
    objects as their base type. Objects that inherit from `planobject` (which
    uses metaclass `plantype`) are defined in the same way that calculation
    plans are defined. Any attributes of the class (including those inherited
    from base classes) that are calculations (see `pimms.calc` and `pimms.plan`)
    are turned into a plan. The inputs of the plan are the required parameters
    for the class, and the outputs of the plan become the attributes of the
    class, which are resolved lazily like in `plandict`s.

    The `__init__` function of a `planobject` is special. During the `__init__`
    function only, the parameters of a `planobject` function can be set using
    the usual `setattr` interface. All `planobject`s are immutable once they
    have been initialized, however. At the end of the `__init__` function, the
    object must have all of its parameters set, otherwise an error is raised. If
    no `__init__` function is defined, then the `planobject` default init
    function calls `merge` on its arguments and keywords; the resulting dict
    must be a dictionary of the class's parameters.

    `planobject` types must not overload the following methods, as they are used
    by the `planobject` / `plantype` system. These are:
      * `__new__`
      * `__setattr__`
      * `__getattr__ `
      * `__dir__`
    """
    __slots__ = ('__plandict__')
    def __setattr__(self, k, v):
        if type(self.__plandict__) is dict:
            plan = type(self).plan
            if k not in plan.inputs:
                raise ValueError("can only set planobject inputs during init")
            self.__plandict__[k] = v
        else:
            raise TypeError(f"type {type(self)} is immutable")
    def __getattr__(self, k):
        pd = self.__paramdict__
        r = pd.get(r, pd)
        if r is pd:
            return object.__getattr__(self, k)
        else:
            return r
    def __dir__(self):
        l = object.__dir__(self)
        for k in self.__plandict__.keys():
            l.append(k)
        return l
    def __new__(cls, *args, **kwargs):
        # Start by creating the object itself and setting up its slots.
        obj = object.__new__(cls)
        object.__setattr__(obj, '__plandict__', dict())
        # Once the __init__ function is done running, the plandict will be
        # cleaned up (this is guaranteed by the plantype meta-class).
        return obj
    def __init__(self, *args, **kwargs):
        for (k,v) in merge(*args, **kwargs).items():
            setattr(self, k, v)
    def __str__(self):
        p = self.__plandict__.plan
        s = ", ".join([f"{k}={v}" for (k,v) in p.items()])
        return f"{type(self)}({s})"
    def __repr__(self):
        return str(self)
    def __eq__(self, other):
        if type(self) is not type(other): return False
        return self.__plandict__.inputs == other.__plandict__.inputs
    def __ne__(self, other):
        if type(self) is not type(other): return True
        return self.__plandict__.inputs != other.__plandict__.inputs
    def __hash__(self):
        return hash((type(self), self.__plandict__.inputs))
    def copy(self, **kwargs):
        """Creates a copy of the plabobject with optional parameter updates.
        """
        pd = plandict(self.__plandict__, **kwargs)
        obj = object.__new__(cls)
        object.__setattr__(obj, '__plandict__', pd)
        return obj

# Utilities ####################################################################
def is_planobject(obj):
    '''Determines if an object is an instance of a `pimms` `plantype` object.
    
    `is_planobject(obj)` returns `True` if `obj` is an instance of a `pimms`
    `plantype` class and `False` otherwise.

    See also: `plantype`, `is_plantype`
    '''
    return isinstance(obj, planobject)
def is_plantype(obj):
    '''Determines if an object is a `pimms` `plantype`.
    
    `is_plantype(obj)` returns `True` if `obj` is a `pimms` `plantype` class and
    `False` otherwise. Note that this works for the type but not instances of
    the type, for which you should use `is_planobject`.

    See also: `is_planobject`, `plantype`
    '''
    return isinstance(obj, plantype)
