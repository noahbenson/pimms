# -*- coding: utf-8 -*-
################################################################################
# pimms/plantype/_core.py
#
# Simple meta-class for lazily-calculating plantype classes.
#
# Copyright 2022 Noah C. Benson
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Dependencies #################################################################
import copy, types, inspect
from collections import (defaultdict, namedtuple)

from ..types import (is_str, is_fdict)
from ..lazydict import (ldict, fdict, is_ldict, assoc)
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
    __slots__ = ()
    class planobject_base:
        """The base-class for the `pimms.planobject` class.

        `plantype.planobject_base` is a simple class that implements the basic
        features of the `planobject` class. The separation for certain methods
        from the `planobject` type itself is required due to details of how the
        `planobject` class, which has the `plantype` meta-class, gets
        initialized while the `plantype.__new__` method depends on methods in
        the `planobject` class (which hasn't been initialized/defined at the
        time that the `planobject.__new__` method is called. This class
        shouldn't be used directly and shouldn't be inherited. Use the
        `planobject` class instead.
        """
        __slots__ = ('__plandict__',)
        def __getattr__(self, k):
            pd = self.__paramdict__
            r = pd.get(r, pd)
            if r is pd:
                return object.__getattr__(self, k)
            else:
                return r
        def __setattr__(self, k, v):
            if type(self.__plandict__) is dict:
                plan = type(self).plan
                if k not in plan.inputs:
                    raise ValueError("planobject inputs become immutable after"
                                     " the __init__ method returns")
                self.__plandict__[k] = v
            else:
                raise TypeError(f"type {type(self)} is immutable")
        def __new__(cls, *args, **kwargs):
            # Start by creating the object itself and setting up its slots.
            obj = object.__new__(cls)
            object.__setattr__(obj, '__plandict__', dict())
            # Once the __init__ function is done running, the plandict will be
            # cleaned up (this is guaranteed by the plantype meta-class).
            return obj
        def __dir__(self):
            l = object.__dir__(self)
            for k in self.__plandict__.keys():
                l.append(k)
            return l
        def __init__(self, *args, **kwargs):
            for (k,v) in merge(*args, **kwargs).items():
                setattr(self, k, v)
        def _init_wrapper(self, *args, **kwargs):
            """Manages the initialization (`__init__`) for `planobject` types.

            The `planobject` type, and any types that inherit from it, is
            generally immutable; however, When a `planobject` is first created,
            it is allowed to set its inputs, as if they were mutable attributes,
            during the `__init__()` method. In order to facilitate this, a
            reorganization of the initialization code for each `planobject`
            subclass is performed when that class is defined. The class's true
            `__init__` method is stored in the method `__planobject_init__`,
            while the `plantype.planobject_base._init_wrapper` method is stored
            in the type's `__init__` method. This method calls the type's
            `__planobject_init__` method then makes the initialized object
            immutable.

            This method should not be called directly by the user.
            """
            # If the plandict is not a mutable dictionary, we have already been
            # initialized.
            if is_fdict(self.__plandict__):
                raise RuntimeError("_init_wrapper method called on an already-"
                                   "initialized planobject")
            # This method is the real initializer for the class (what the
            # class's actual code wrote as the __init__ method).
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
        for (k,v) in [('__getattr__', plantype.planobject_base.__getattr__),
                      ('__setattr__', plantype.planobject_base.__setattr__),
                      ('__new__',     plantype.planobject_base.__new__),
                      ('__dir__',     plantype.planobject_base.__dir__)]:
            if k in attrs:
                raise ValueError(f"plantype classes may not define {k}")
            else:
                attrs[k] = v
        # (2) We want to save the init function and update it to our version.
        init = attrs.get('__init__', plantype.planobject_base.__init__)
        attrs['__planobject_init__'] = init
        attrs['__init__'] = plantype.planobject_base._init_wrapper
        # (3) Go through the bases: see if there are planobject bases already,
        #     and if not, add planobject in. As we go, collect calculations.
        calcs = {k:v for (k,v) in attrs.items() if is_calc(v)}
        found_planobj = False
        for b in bases:
            if not issubclass(b, plantype.planobject_base): continue
            found_planobj = True
            for (k,v) in inspect.getmembers(b):
                if k not in calcs and is_calc(v):
                    calcs[k] = v
        if not found_planobj:
            bases.append(plantype.planobject_base)
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
class planobject(plantype.planobject_base, metaclass=plantype):
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
    __slots__ = ()
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
