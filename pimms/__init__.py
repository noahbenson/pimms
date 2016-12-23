####################################################################################################
# pimms/__init__.py
#
# This source-code file is part of the pimms library.
#
# The pimms library is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If
# not, see <http://www.gnu.org/licenses/>.

'''
The pimms library is an immutable data-structure toolkit. It works primarily by decorators applied
to classes and their members to declare how an immutable data-structure's members are related.
Taken together, these decorators form a DSL-like system for declaring immutable data-structures with
full inheritance support.

An immutable data-structure is simply a class that has been modified by the @immutable decorator.
Inside an immutable class, a few things can be declared normally while others must be declared via
the special immutable syntax.
Things that can be delcared normally in an immutable class:
  * Instance methods of the class (def some_method(self, arg): ...)
  * Static methods of the class (using @staticmethod)
  * Static members of the class (some_class_member = 10)
Things that cannot be declared normally in an immutable class:
  * All member variables of a class instance (usually assigned in __init__)
  * __new__, __setattr__, __getattribute__, __delattr__, __copy__, and __deepcopy__ cannot be
    overloaded in immutable classes; doing so will result in undefined behavior
  * Immutable classes should generally only inherit from object or other immutable classes;
    inheritance with other non-immutable classes is fine in theory, especially if only methods are
    added to the class, but member access from immutable objects to non-immutable class members is
    beyond the scope of this library.

Immutable instance member variables, which are usually simply assigned in the class's __init__
method, must be declared specially in immutable classes. All immutable instance members fall into
one of two categories: parameters and values. Parameters are values that must be assigned by the
end of the object's __init__ function, in order for the object to be valid (an exception is raised
if these are not filled). Options are a special kind of parameter that also have default values in
case no assignment is given. Values, unlike parameters, can never be assigned directly; instead,
they are lazily and automatically calculated by a user-provided function of zero or more other
members of the class. Values may depend on either parameters or other values as long as there is
not a circular dependency graph implicit in the declarations.

All such instance member declarations are made using the @param, @option, @value, and @require
decorators, documented briefly here. In all four cases, @param, @option(<default value>), @value,
and @require, the decorator should precede a static function definition.
  * @param declares two things: first, that the name of the static function that follows it is an
    instance member and required parameter of the class and, second, that the static function
    itself, which must take exactly one argument, should be considered a translation function on
    any value assigned to the object; the return value of the function is the value actually
    assigned to the object before any checks are run.
  * @option(<default value>) is identical to @param except that it declares that the given default
    value should be used if no value is assigned to the object in the __init__ method. This value,
    if it is used, is not passed through the translation function that follows.
  * @value declares three things: first, that the name of the static function that follows it is an
    instance member and (lazy) value of the class; second, that the arguments to that static
    function, which must be named exactly after other instance members of the class, are instance on
    members on which this value depends (thus this value will be reset when those members change);
    and third, that the return value of that static function, when given the appropriate member
    values, should be the value assigned to the instance member when requested.
  * @require declares three things: first, that the name of the following static function is the
    identifier for a particular requirement check on the instance members of any object of this
    class; second, that the parameters of that static function, which must match exactly the names
    of other instance members of the class, are the instance members that this requirement checks;
    and third, that the static function's return value will be True if and only if the check is
    passed. The requirement function may throw its own exception or return False, in which case a
    generic exception is raised.

All four decorator types may be overloaded in immutable child classes. Overloading works much as it
does with normal methods; only the youngest child-class's method is required. This can be used to
overload requirements, but new requirements can be placed to act as additional constraints; i.e.,
the youngest class's requirement is always run for all requirements in an object's entire class
hierarchy when a relevant instance members is updated. Overloading may also be used to change an
instance member's type in the child class, such as from a value to a parameter or vice versa. The
child class must, of course, be free of circular dependencies despite these rearrangements.

Note that a required parameter may be implied by the other instance member declarations; this is not
an error and instead inserts the parameter into the class automatically with no translation
function. This occurs when either a value or a requirement declares the parameter as an argument
but no instance member with the parameter's name appears elsewhere in the class or its ancestor
classes. This can be used to create a sort of abstract immutable base class, in that a class may
declare a requirement of some parameter that is not otherwise defined; in order to instantiate the
class, that parameter must be given, either in a child class's __init__ method or as a value or an
explicit (required or optional) parameter.

When an immutable object is constructed, it begins its life in a special 'init' state; this state is
unique in that no requirement checks are run until either the __init__ method returns or a value is
requested of the object; at that point, all non-optional parameters must be specified or an
exception is raised. If all parameters were set, then all requirement checks are run and the
object's state shifts to 'transient'. An immutable object imm can be tested for transience by using
the method imm.is_transient(). A transient object allows its parameters (but never its values) to be
set using normal setattr (imm.x = y) syntax. Requirement checks that are related to a parameter are
changed every time that parameter is set, after the translation function for that parameter is used.
An immutable object remains in the transient state until it is persisted via the imm.persist()
method. Once an object is persistent, it cannot be updated via setattr mechanisms and should be
considered permanent. A new transient duplicate of the object may be created using the
imm.transient() method (this may also be used while the object is still transient). To update the
values of an immutable object, the imm.copy(param1=val1, param2=val2, ...) method should be used.
This method returns a persistent duplicate of the immutable object imm with the given parameters
updated to the given values; these values are always passed through the translation functions
and all relevant chacks are run prior to the return of the copy function. The copy function may
be called on transient or persistent immutables, but the return value is always peresistent.

The additional utility functions are provided as part of the pimms package:
  * is_imm(x) yields True if x is an object that is an instance of an immutable class and False
    otherwise.
  * is_imm_type(x) yields True if x is a class that is immutable and False otherwise.
  * imm_copy(imm, ...) is identical to imm.copy(...) for an immutable object imm.
  * imm_persist(imm) is identical to imm.persist() for a transient immutable object imm.
  * imm_transient(imm) is identical to imm.transient() for an immutable object imm.
  * imm_values(imm_t) yields a list of the values of the immutable class imm_t.
  * imm_params(imm_t) yields a list of the parameters of the immutable class imm_t.
  * imm_dict(imm) is identical to imm.asdict() for an immutable object imm.
  * imm_is_persistent(imm) is identical to imm.is_persistent() for an immutable object imm.
  * imm_is_transient(imm) is identical to imm.is_transient() for an immutable object imm.
'''

from .util        import (lazy_map, is_lazy_map, LazyPMap, is_map, is_pmap, merge,
                          is_quantity, is_unit)
from .immutable   import (immutable, require, value, param, option, is_imm, is_imm_type, imm_copy,
                          imm_persist, imm_transient, imm_params, imm_values, imm_dict,
                          imm_is_persistent, imm_is_transient)
from .calculation import (calc,    plan,    imap,
                          Calc,    Plan,    IMap,
                          is_calc, is_plan, is_imap)
from .table       import (itable, is_itable, ITable)


def reload_pimms():
    '''
    reload_pimms() reloads the entire pimms module and returns it.
    '''
    import sys
    reload(sys.modules['pimms.util'])
    reload(sys.modules['pimms.table'])
    reload(sys.modules['pimms.immutable'])
    reload(sys.modules['pimms.calculation'])
    reload(sys.modules['pimms'])
    return sys.modules['pimms']

__version__ = '0.1.3'
description = 'Lazy immutable library for Python built on top of pyrsistent'
