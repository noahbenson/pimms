# pimms ############################################################################################
Pimms, a Python immutable data structures library.

## Author ##########################################################################################
Noah C. Benson &lt;<nben@nyu.edu>&gt;

## Installation ####################################################################################

The pimms library is available on [PyPI](https://pypi.python.org/pypi/pimms) and can be
installed via pip:

```bash
pip install pimms
```

The dependencies (below) should be installed auotmatically. Alternately, you can check out this
github repository and run setuptools:

```bash
# Clone the repository
git clone https://github.com/noahbenson/pimms
# Enter the repo directory
cd pimms
# Install the library
python setup.py install

```

### Current Version ################################################################################
The current stable version of pimms is 0.1.9.

### Dependencies ###################################################################################

The pimms library currently depends on a small number of libraries, all installable via pip.
* [pyrsistent](https://github.com/tobgu/pyrsistent); pimms is intended as a complement to
  pyrsistent and heavily uses pyrsistent throughout. The lazy map interface in pimms, in fact, is
  largely a duplicate of pyrsistent's PMap code with additional hooks to handle lazy values.
* [numpy](http://www.numpy.org/); the numpy library is required for pimms's ITable structures.
* [six](https://github.com/benjaminp/six); a library for helping write code compatible with both
  Python 2 and 3 versions (note: pimms has not been tested at all with Python 3).
* [pint](https://github.com/hgrecco/pint); a units library broadly compatible with numpy; pimms uses
  this library and manages its own unit registry (`pimms.units`).

### Tests ##########################################################################################

Tests have been placed in the `pimms.test` package and can be run via the command 
`python -m unittest pimms.test`.


## Documentation ###################################################################################

The pimms library is an immutable data-structure and lazy calculation toolkit. It can be broken down
into four components, each of which is described below.

### Immutable Classes ##############################################################################
The pimms package allows one to construct a kind of immutable class. This mechanism works primarily
by decorators applied to classes and their members to declare how an immutable data-structure's
members are related. Taken together, these decorators form a DSL-like system for declaring immutable
data-structures with most inheritance support.

An immutable data-structure is simply a class that has been modified by the @immutable decorator.
Inside an immutable class, a few things can be declared normally while others must be declared via
the special immutable syntax.

Things that can be delcared normally in an immutable class:
* Instance methods of the class (`def some_method(self, arg): ...`)
* Static methods of the class (`using @staticmethod`)
* Static members of the class (`some_class_member = 10`)

Things that cannot be declared normally in an immutable class:
* All member variables of a class instance (usually assigned in `__init__`).
* `__new__`, `__setattr__`, `__getattribute__`, `__delattr__`, `__copy__`, and `__deepcopy__`
  cannot be overloaded in immutable classes; doing so will result in undefined behavior.
* Immutable classes should generally only inherit from object or other immutable classes;
  inheritance with other non-immutable classes is fine in theory, especially if only methods are
  added to the class, but member access from immutable objects to non-immutable class members is
  beyond the scope of this library.

Immutable instance member variables, which are usually simply assigned in the class's `__init__`
method, must be declared specially in immutable classes. All immutable instance members fall into
one of two categories: parameters and values. Parameters are values that must be assigned by the
end of the object's `__init__` function, in order for the object to be valid (an exception is raised
if these are not filled). Options are a special kind of parameter that also have default values in
case no assignment is given. Values, unlike parameters, can never be assigned directly; instead,
they are lazily and automatically calculated by a user-provided function of zero or more other
members of the class. Values may depend on either parameters or other values as long as there is
not a circular dependency graph implicit in the declarations.

All such instance member declarations are made using the `@param`, `@option`, `@value`, and
`@require` decorators, documented briefly here. In all four cases, `@param`,
`@option(default_value)`, `@value`, and `@require`, the decorator should precede a static function
definition.
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
class, that parameter must be given, either in a child class's `__init__` method or as a value or an
explicit (required or optional) parameter.

When an immutable object is constructed, it begins its life in a special 'init' state; this state is
unique in that no requirement checks are run until either the `__init__` method returns or a value
is requested of the object; at that point, all non-optional parameters must be specified or an
exception is raised. If all parameters were set, then all requirement checks are run and the
object's state shifts to 'transient'. An immutable object imm can be tested for transience by using
the method `imm.is_transient()`. A transient object allows its parameters (but never its values) to
be set using normal setattr (`imm.x = y`) syntax. Requirement checks that are related to a parameter
are changed every time that parameter is set, after the translation function for that parameter is
used. An immutable object remains in the transient state until it is persisted via the 
`imm.persist()` method. Once an object is persistent, it cannot be updated via setattr mechanisms
and should be considered permanent. A new transient duplicate of the object may be created using the
`imm.transient()` method (this may also be used while the object is still transient). To update the
values of an immutable object, the `imm.copy(param1=val1, param2=val2, ...)` method should be used.
This method returns a persistent duplicate of the immutable object imm with the given parameters
updated to the given values; these values are always passed through the translation functions
and all relevant chacks are run prior to the return of the copy function. The copy function may
be called on transient or persistent immutables, but the return value is always peresistent.

The additional utility functions are provided as part of the pimms package:
* `isimm(x)` yields True if x is an object that is an instance of an immutable class and False
  otherwise.
* `isimmtype(x)` yields True if x is a class that is immutable and False otherwise.
* `imm_copy(imm, ...)` is identical to `imm.copy(...)` for an immutable object imm.
* `imm_persist(imm)` is identical to `imm.persist()` for a transient immutable object imm.
* `imm_transient(imm)` is identical to `imm.transient()` for an immutable object imm.
* `imm_values(imm_t)` yields a list of the values of the immutable class imm_t.
* `imm_params(imm_t)` yields a list of the parameters of the immutable class imm_t.
* `imm_dict(imm)` is identical to `imm.asdict()` for an immutable object imm.
* `imm_is_persistent(imm)` is identical to `imm.is_persistent()` for an immutable object imm.
* `imm_is_transient(imm)` is identical to `imm.is_transient()` for an immutable object imm.

### Lazy Calculations ##############################################################################
The pimms lazy calculation system allows you to declare a set of independent calculation units, each
of which requiers a specific set of input parameters and generates a specific set of output values,
then to put them together into a calculation plan. A calculation plan can then be given the set of
input parameters not provided by the ouputs of other calculation units in order to obtain a lazy map
of all the output variables; values get calculated only the first time they are requested from this
output map.

A calculation unit is declared by the decorator `@calc(...)` where the ... should be a list of
output value names. The decorated function must yield either a tuple of these outputs in order or a
dict-like object whose keys correspond to these output values; if there is only one output and it is
neither a dict nor a tuple, then it can be returned alone. The input parameters for the calculation
unit are the parameters of the function (default values are allowed, and any plan that includes the
calculation unit will inherit this default, if appropriate). Other functionality details can be
found via `help(pimms.calc)`.

Once a set of calculation units has been defined, they may be grouped together in a calculation
plan. To do this, they are passed to the function `plan()`. The `plan()` function accepts any number
of dict-like objects followed by any number of keyword arguments, all of which are merged into a
single dict from left to right (i.e., later dictionaries overwrite earlier dictionaries, and keyword
arguments overwrite all dictionaries). This final dict contains key-value pairs where the keys are
arbitrary (string) names of each calculation step and the values are the calculation units for that
step. To change one of the calculation units, the method `set` can be used (see also,
`pimms.Plan.discard`, `pimms.Plan.remove`, `pimms.Plan.discard_defaults`, and
`pimms.Plan.remove_defaults`).

Plans can be queried in a number of ways; primarily, a plan keeps track of the afferent (input)
parameters that it requires as well as the efferent (output) values it produces in sum; if one of
the calculation units provides an efferent value that is an afferent input of another calculation
unit in the plan, then this value would be an efferent value of the plan, but not an afferent
parameter. These data can be accessed via `p.afferents` and `p.efferents`; the `p.defaults` member
additionally gives a dict of the default values for any afferent parameter that has one.
Documentation about each parameter (if the author of the calculation unit included documentation
in the unit's doc-string) can also be found in the `p.afferent_docs` and `p.efferent_docs` dicts.

To invoke a plan, it must be caled using all of its afferent parameters; those parameters with
default values may be provided but do not need to be. When a plan `p` is called, it accepts any
number of dict-like objects followed by any number of keyword arguments as input; these are merged
left-to-right and used as a dictionary of the afferent parameters. If all afferent parameters are
provided, then the plan returns an immutable mapping (a persistent dict-like object) immediately.
The return value is immediate because no calculation is actually performed until it is required;
the keys of the map are the names of both the afferent parameters and efferent values, all of which
can be obtained via a normal getitem lookup.

#### Example

An example of a lazy calculation plan is shown here (see also `pimms.test.tri_calc`):

```python
# Usage example: calculating the area of a triangle
import pimms

# We make a lazy calculation plan to calculate the area of a triangle; first it calculates the base
# and height, then the area.

# First calc unit: calculate the base and the height
@pimms.calc('base', 'height')
def calc_triangle_dims(a, b, c):
    '''
    calc_triangle_dims computes the base/width (x-span) and height (y-span) of the triangle a-b-c.
    
    Afferent parameters:
     @ a Must be the (x,y) coordinate of point a in triangle a-b-c.
     @ b Must be the (x,y) coordinate of point b in triangle a-b-c.
     @ c Must be the (x,y) coordinate of point c in triangle a-b-c.

    Efferent values:
     @ base Will be the base, or width, of the triangle a-b-c.
     @ height Will be the height of the triangle a-b-c.
    '''
    print 'Calculating base...'
    xs = [a[0], b[0], c[0]]
    xmin = min(xs)
    xmax = max(xs)
    print 'Calculating height...'
    ys = [a[1], b[1], c[1]]
    ymin = min(ys)
    ymax = max(ys)
    return (xmax - xmin, ymax - ymin)

# Second calc unit: calculate the area
@pimms.calc('area')
def calc_triangle_area(base, height):
    '''
    calc_triangle_are computes the area of a triangle with a given base and height.
    
    Efferent values:
     @ area Will be the area of the triangle with the given base and height.
    '''
    print 'Calculating area...'
    return {'area': base * height * 0.5}

# Make a calculation plan out of these:
area_plan = pimms.plan(tri_dims=calc_triangle_dims, tri_area=calc_triangle_area)

# What afferent parameters does it require?
area_plan.afferents
# >> ('a', 'c', 'b')

# Do any of these have defaults?
area_plan.defaults
# >> pmap({})

# What about the efferent values?
area_plan.efferents
# >> pmap({'height': <pimms.calculation.Calc object at 0x106ee9910>,
# >>       'base': <pimms.calculation.Calc object at 0x106ee9910>,
# >>       'area': <pimms.calculation.Calc object at 0x106ee9990>})

# Documentation on parameter b?
print area_plan.afferent_docs['b']
# >> (tri_dims) b: Must be the (x,y) coordinate of point b in triangle a-b-c.

# Go ahead and execute the plan; note that the latter c overwrites the former
data = area_plan({'a':(0,0), 'b':(0,1), 'c':(1,0)}, c=(2,1))
data.keys()
# >> ['b', 'a', 'c', 'base', 'area', 'height']

# What's the base?
data['base']
# >> Calculating base...
# >> Calculating height...
# >> 2

# What's the height? Note that this doesn't get recalculated...
data['height']
# >> 1

# What's the area?
data['area']
# >> Calculating area...
# >> 1.0

# How about we update the c value? We can't change data directly because it's
# immutable, but we can re-bind a copy of it with new parameters...
new_data = data.set(c=(1,0))

# The old data remains unchanged, but new_data has the new value of c:
(data['c'], new_data['c'])
# >> ((2, 1), (1, 0))

# What's the area of the new triange?
new_data['area']
# >> Calculating base...
# >> Calculating height...
# >> Calculating area...
# >> 0.5
```

### Immutable Tables ###############################################################################
An immutable table is a read-only data structure consisting of a matrix of named columns. The
`pimms.itable` function is used to construct such a table by passing it dictionaries and/or keyword
arguments, all values for which are the same length. Immutable tables support lazy operations in
that if one of the values given to it is a lambda function requiring no arguments, then that lambda
function is stored and called when the column is first needed; the function's output must be an
appropriately-sized column of values. The columns of an itable themselves are stored as numpy arrays
and the itable toolkit supports unit types for the columns via the library
(pint)[https://github.com/hgrecco/pint]. In fact, pimms keeps track of its own pint unit registry in
`pimms.units` (which may be changed if you manage your units separately).

The immutable table class (pimms.ITable) is decorated with pimms's own immutable toolkit, so its
members follow that paradigm. Overall the class is fairly simple, but some of its features are documented here:
 * `tab.data` gives the dict-like object of columns of table;
 * `tab.column_names` gives a tuple of the names of the columns in the table;
 * `tab.columns` gives a tuple of the columns themselves in the same order as `column_names`;
 * `tab.row_count` gives the number of rows in the table;
 * `tab.rows` gives a tuple of maps, one per row, each of which contains a key for each column in
   the table.

### Utilities ######################################################################################
The pimms library includes a number of utility functions, all of which are documented in Python
via normal doc-strings. They are listed here:
 * `pimms.lazy_map`
 * `pimms.is_lazy_map`
 * `pimms.is_map`
 * `pimms.is_pmap`
 * `pimms.merge`
 * `pimms.is_quantity`
 * `pimms.is_unit`
 * `pimms.quant`
 * `pimms.mag`
 * `pimms.like_units`
 * `pimms.units`
 * `pimms.save`
 * `pimms.load`

## License #########################################################################################

This README file is part of the pimms library.

The pimms library is free software: you can redistribute it and/or Modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not,
see <http://www.gnu.org/licenses/>.
