# -*- coding: utf-8 -*-
################################################################################
# pimms/test/util/test_core.py
#
# Tests of the core utilities module in pimms: i.e., tests for the code in the
# pimms.util._core module.
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
from unittest import TestCase

class TestUtilCore(TestCase):
    """Tests the pimms.util._core module."""

    # Pint Utilities ###########################################################
    def test_is_ureg(self):
        from pimms import (units, is_ureg)
        from pint import UnitRegistry
        # pimms.units is a registry.
        self.assertTrue(is_ureg(units))
        # So is any new UnitsRegistry we create.
        self.assertTrue(is_ureg(UnitRegistry()))
        # Other objects are not.
        self.assertFalse(is_ureg(None))
    def test_is_unit(self):
        from pimms import (units, is_unit)
        from pint import UnitRegistry
        # We will use an alternate unit registry in some tests.
        alt_units = UnitRegistry()
        # Units from any unit registry are allowed by default.
        self.assertTrue(is_unit(units.mm))
        self.assertTrue(is_unit(units.gram))
        self.assertTrue(is_unit(alt_units.mm))
        self.assertTrue(is_unit(alt_units.gram))
        # Things that aren't units are never units.
        self.assertFalse(is_unit('mm'))
        self.assertFalse(is_unit(10.0))
        self.assertFalse(is_unit(None))
        self.assertFalse(is_unit(10.0 * units.mm))
        # If the ureg parameter is ..., then only units from the pimms units
        # registry are allowed.
        self.assertTrue(is_unit(units.mm, ureg=...))
        self.assertFalse(is_unit(alt_units.mm, ureg=...))
        # Alternately, the ureg parameter may be a specific unit registry.
        self.assertFalse(is_unit(units.mm, ureg=alt_units))
        self.assertTrue(is_unit(alt_units.mm, ureg=alt_units))
    def test_is_quant(self):
        from pimms import (units, is_quant)
        from pint import UnitRegistry
        # We will use an alternate unit registry in some tests.
        alt_units = UnitRegistry()
        # By default, it does not matter what registry a quantity comes from;
        # it is considered a quantity.
        q = 10.0 * units.mm
        alt_q = 10.0 * alt_units.mm
        self.assertTrue(is_quant(q))
        self.assertTrue(is_quant(alt_q))
        # Other objects aren't quantities.
        self.assertFalse(is_quant(10.0))
        self.assertFalse(is_quant(units.mm))
        self.assertFalse(is_quant(None))
        # We can require that a quantity meet a certain kind of unit type.
        self.assertTrue(is_quant(q, unit=units.mm))
        self.assertTrue(is_quant(q, unit='inches'))
        self.assertFalse(is_quant(q, unit=units.grams))
        self.assertFalse(is_quant(q, unit='seconds'))
        # The ureg parameter changes whether any unit registry is allowed (the
        # default, or ureg=None), only pimms.units is allowed (ureg=Ellipsis),
        # or a specific unit registry is allowed.
        self.assertTrue(is_quant(q, ureg=...))
        self.assertFalse(is_quant(alt_q, ureg=...))
        self.assertFalse(is_quant(q, ureg=alt_units))
        self.assertTrue(is_quant(alt_q, ureg=alt_units))
    
    # String Functions #########################################################
    def test_is_str(self):
        from pimms import is_str
        # is_str is just a wrapper for isinstance(obj, str).
        self.assertTrue(is_str('abc'))
        self.assertTrue(is_str(''))
        self.assertFalse(is_str(100))
        self.assertFalse(is_str(None))
    def test_strnorm(self):
        from pimms import strnorm
        # There are a lot of string encoding details that should probably be
        # tested carefully here, but for now, we're mostly concerned that the
        # most basic strings get normalized properly.
        self.assertEqual('abc', strnorm('abc'))
        self.assertEqual('abc', strnorm('aBc', case=True))
    def test_strcmp(self):
        from pimms import strcmp
        # strcmp is, at its simplest, just a string-comparison function.
        self.assertEqual(0,  strcmp('abc', 'abc'))
        self.assertEqual(-1, strcmp('abc', 'bca'))
        self.assertEqual(1,  strcmp('bca', 'abc'))
        # There are a few bells and whistles for strcmp, thought. First, the
        # case option lets you decide whether to ignore case (via strnorm).
        self.assertEqual(-1, strcmp('ABC', 'abc'))
        self.assertEqual(0,  strcmp('ABC', 'abc', case=False))
        # The strip option lets one ignore whitespace on either side of the
        # arguments.
        self.assertEqual(-1, strcmp(' abc', 'abc  '))
        self.assertEqual(0,  strcmp(' abc', 'abc  ', strip=True))
        # The split argument lets you split on whitespace then compare the
        # individual split parts (i.e., this option should make all strings that
        # are identical up to the amount of whitespace should be equal).
        self.assertEqual(1, strcmp('abc def ghi', ' abc  def ghi '))
        self.assertEqual(0, strcmp('abc def ghi', ' abc  def ghi ', split=True))
        # If one of the arguments isn't a string, strcmp returns None.
        self.assertIsNone(strcmp(None, 10))
    def test_streq(self):
        from pimms import streq
        # streq is just a string equality predicate function.
        self.assertTrue(streq('abc', 'abc'))
        self.assertFalse(streq('abc', 'def'))
        # The case option can tell it to ignore case.
        self.assertFalse(streq('ABC', 'abc'))
        self.assertTrue(streq('ABC', 'abc', case=False))
        # The strip option can be used to ignore trailing/leading whitespace.
        self.assertFalse(streq(' abc', 'abc  '))
        self.assertTrue(streq(' abc', 'abc  ', strip=True))
        # The split argument lets you split on whitespace then compare the
        # individual split parts (i.e., this option should make all strings that
        # are identical up to the amount of whitespace should be equal).
        self.assertFalse(streq('abc def ghi', ' abc  def ghi '))
        self.assertTrue(streq('abc def ghi', ' abc  def ghi ', split=True))
        # Nonstring arguments return None.
        self.assertIsNone(streq(None, 'abc'))
    def test_strends(self):
        from pimms import strends
        # strends is just a string equality predicate function.
        self.assertTrue(strends('abcdef', 'def'))
        self.assertFalse(strends('abcdef', 'bcd'))
        # The case option can tell it to ignore case.
        self.assertFalse(strends('ABCDEF', 'def'))
        self.assertTrue(strends('ABCDEF', 'def', case=False))
        # The strip option can be used to ignore trailing/leading whitespace.
        self.assertFalse(strends(' abcdef ', 'def  '))
        self.assertTrue(strends(' abcdef ', 'def  ', strip=True))
        # Nonstring arguments return None.
        self.assertIsNone(strends(None, 'abc'))
    def test_strstarts(self):
        from pimms import strstarts
        # strstarts is just a string equality predicate function.
        self.assertTrue(strstarts('abcdef', 'abc'))
        self.assertFalse(strstarts('abcdef', 'bcd'))
        # The case option can tell it to ignore case.
        self.assertFalse(strstarts('ABCDEF', 'abc'))
        self.assertTrue(strstarts('ABCDEF', 'abc', case=False))
        # The strip option can be used to ignore trailing/leading whitespace.
        self.assertFalse(strstarts(' abcdef ', '  abc'))
        self.assertTrue(strstarts(' abcdef ', '  abc', strip=True))
        # Nonstring arguments return None.
        self.assertIsNone(strstarts(None, 'abc'))
    def test_strissym(self):
        from pimms import strissym
        # strissym tests whether a string is both a string and a valid Python
        # symbol.
        self.assertTrue(strissym('abc'))
        self.assertTrue(strissym('def123'))
        self.assertTrue(strissym('_10xyz'))
        self.assertFalse(strissym('abc def'))
        self.assertFalse(strissym(' abcdef '))
        self.assertFalse(strissym('a-b'))
        self.assertFalse(strissym('10'))
        # Keywords are allowed.
        self.assertTrue(strissym('for'))
        self.assertTrue(strissym('and'))
        # Non-strings return Nonee.
        self.assertFalse(strissym(None))
        self.assertFalse(strissym(10))
    def test_striskey(self):
        from pimms import striskey
        # striskey tests whether a string is (1) a string, (2) a valid Python
        # symbol, and (3) an existing Python keyword.
        self.assertTrue(striskey('for'))
        self.assertTrue(striskey('and'))
        self.assertTrue(striskey('None'))
        self.assertFalse(striskey('abc'))
        self.assertFalse(striskey('def123'))
        self.assertFalse(striskey('_10xyz'))
        self.assertFalse(striskey('abc def'))
        self.assertFalse(striskey(' abcdef '))
        self.assertFalse(striskey('a-b'))
        self.assertFalse(striskey('10'))
        # Non-strings return None.
        self.assertIsNone(striskey(None))
        self.assertIsNone(striskey(10))
    def test_strisvar(self):
        from pimms import strisvar
        # strisvar tests whether a string is (1) a string, (2) a valid Python
        # symbol, and (3) an *not* existing Python keyword.
        self.assertFalse(strisvar('for'))
        self.assertFalse(strisvar('and'))
        self.assertFalse(strisvar('None'))
        self.assertTrue(strisvar('abc'))
        self.assertTrue(strisvar('def123'))
        self.assertTrue(strisvar('_10xyz'))
        self.assertFalse(strisvar('abc def'))
        self.assertFalse(strisvar(' abcdef '))
        self.assertFalse(strisvar('a-b'))
        self.assertFalse(strisvar('10'))
        # Non-strings return Nonee.
        self.assertIsNone(strisvar(None))
        self.assertIsNone(strisvar(10))

    # Other Utilities ##########################################################
    def test_hashsafe(self):
        from pimms import hashsafe
        # hashsafe returns hash(x) if x is hashable and None otherwise.
        self.assertIsNone(hashsafe({}))
        self.assertIsNone(hashsafe([1, 2, 3]))
        self.assertIsNone(hashsafe(set(['a', 'b'])))
        self.assertEqual(hash(10), hashsafe(10))
        self.assertEqual(hash('abc'), hashsafe('abc'))
        self.assertEqual(hash((1, 10, 100)), hashsafe((1, 10, 100)))
    def test_can_hash(self):
        from pimms import can_hash
        # can_hash(x) returns True if hash(x) will successfully return a hash
        # and returns False if such a call would raise an error.
        self.assertTrue(can_hash(10))
        self.assertTrue(can_hash('abc'))
        self.assertTrue(can_hash((1, 10, 100)))
        self.assertFalse(can_hash({}))
        self.assertFalse(can_hash([1, 2, 3]))
        self.assertFalse(can_hash(set(['a', 'b'])))
    def test_itersafe(self):
        from pimms import itersafe
        # itersafe returns iter(x) if x is iterable and None otherwise.
        self.assertIsNone(itersafe(10))
        self.assertIsNone(itersafe(lambda x:x))
        self.assertEqual(list(itersafe([1, 2, 3])), [1, 2, 3])
    def test_can_iter(self):
        from pimms import can_iter
        # can_iter(x) returns True if iter(x) will successfully return an
        # iterator and returns False if such a call would raise an error.
        self.assertTrue(can_iter('abc'))
        self.assertTrue(can_iter([]))
        self.assertTrue(can_iter((1, 10, 100)))
        self.assertFalse(can_iter(10))
        self.assertFalse(can_iter(lambda x:x))

    # Freeze/Thaw Utilities ####################################################
    def test_is_frozen(self):
        from pimms import (is_frozen, fdict, ldict)
        from frozendict import frozendict
        import torch, numpy as np
        # is_frozen returns True for any frozen object: tuple, frozenset, or
        # frozendict, as well as their downstream types.
        self.assertTrue(is_frozen((1, 2, 3)))
        self.assertTrue(is_frozen(fdict(a=1, b=2, c=3)))
        self.assertTrue(is_frozen(frozenset([1, 2, 3])))
        self.assertTrue(is_frozen(ldict(a=1, b=2, c=3)))
        # The thawed types are not frozen: list, set, dict.
        self.assertFalse(is_frozen([1, 2, 3]))
        self.assertFalse(is_frozen(dict(a=1, b=2, c=3)))
        self.assertFalse(is_frozen(set([1, 2, 3])))
        # NumPy arrays are frozen only if they are not writeable.
        x = np.linspace(0, 1, 25)
        self.assertFalse(is_frozen(x))
        x.setflags(write=False)
        self.assertTrue(is_frozen(x))
        # Objects that aren't any of these types result in None.
        self.assertIsNone(is_frozen('abc'))
        self.assertIsNone(is_frozen(10))
        self.assertIsNone(is_frozen(range(10)))
    def test_is_thawed(self):
        from pimms import (is_thawed, fdict, ldict)
        from frozendict import frozendict
        import numpy as np
        # is_thawed returns True for the thawed types: list, set, dict.
        self.assertTrue(is_thawed([1, 2, 3]))
        self.assertTrue(is_thawed(dict(a=1, b=2, c=3)))
        self.assertTrue(is_thawed(set([1, 2, 3])))
        # is_thawed returns False for any frozen object: tuple, frozenset, or
        # frozendict, as well as their downstream types.
        self.assertFalse(is_thawed((1, 2, 3)))
        self.assertFalse(is_thawed(fdict(a=1, b=2, c=3)))
        self.assertFalse(is_thawed(frozenset([1, 2, 3])))
        self.assertFalse(is_thawed(ldict(a=1, b=2, c=3)))
        # NumPy arrays are thawed only if they are not writeable.
        x = np.linspace(0, 1, 25)
        self.assertTrue(is_thawed(x))
        x.setflags(write=False)
        self.assertFalse(is_thawed(x))
        # Objects that aren't any of these types result in None.
        self.assertIsNone(is_thawed('abc'))
        self.assertIsNone(is_thawed(10))
        self.assertIsNone(is_thawed(range(10)))        
    def test_to_frozenarray(self):
        from pimms.util import to_frozenarray
        import numpy as np
        # frozenarray converts a read-write numpy array into a frozen one.
        x = np.linspace(0, 1, 25)
        y = to_frozenarray(x)
        self.assertTrue(np.array_equal(x, y))
        self.assertIsNot(x, y)
        self.assertTrue(x.flags['WRITEABLE'])
        self.assertFalse(y.flags['WRITEABLE'])
        # If a frozenarray of an already frozen array is requested, the array is
        # returned as-is.
        self.assertIs(y, to_frozenarray(y))
        # However, one can override this with the copy argument.
        self.assertIsNot(y, to_frozenarray(y, copy=True))
        # Typically a copy is made of the original array if it is not already
        # frozen, but one can request that it not be copied.
        z = to_frozenarray(x, copy=False)
        self.assertIs(z, x)
        self.assertFalse(x.flags['WRITEABLE'])
    def test_freeze(self):
        from pimms import freeze
        from frozendict import frozendict
        import numpy as np
        # Freeze lets you convert from a thawed type to a frozen type.
        self.assertIsInstance(freeze([]), tuple)
        self.assertIsInstance(freeze(set([])), frozenset)
        self.assertIsInstance(freeze({}), frozendict)
        self.assertIsInstance(freeze(np.array([])), np.ndarray)
        # The return values remain equal if not the same type.
        self.assertEqual(tuple([1,2,3]), freeze([1,2,3]))
        self.assertEqual(set([1,2,3]), freeze(set([1,2,3])))
        self.assertEqual(dict(a=1,b=2), freeze(dict(a=1,b=2)))
        # If a frozen type is given it is returned as-is.
        t = (1,2,3)
        s = frozenset(t)
        d = frozendict(a=1, b=2)
        self.assertIs(t, freeze(t))
        self.assertIs(s, freeze(s))
        self.assertIs(d, freeze(d))
        # If the type isn't frozen or thawed, then an error is raised.
        with self.assertRaises(TypeError): freeze(None)
        with self.assertRaises(TypeError): freeze(10)
        with self.assertRaises(TypeError): freeze('abc')
        # Numpy arrays are frozen like with the to_frozenarray function.
        x = np.linspace(0, 1, 25)
        y = freeze(x)
        self.assertTrue(np.array_equal(x, y))
        self.assertIsNot(x, y)
        self.assertTrue(x.flags['WRITEABLE'])
        self.assertFalse(y.flags['WRITEABLE'])
    def test_thaw(self):
        from pimms import thaw
        from frozendict import frozendict
        import numpy as np
        # Thaw lets you convert from a frozen type to a thawed type.
        self.assertIsInstance(thaw(()), list)
        self.assertIsInstance(thaw(frozenset([])), set)
        self.assertIsInstance(thaw(frozendict()), dict)
        # The return values remain equal if not the same type.
        self.assertEqual([1,2,3], thaw((1,2,3)))
        self.assertEqual(set([1,2,3]), thaw(frozenset([1,2,3])))
        self.assertEqual(dict(a=1,b=2), thaw(frozendict(a=1,b=2)))
        # If a thawed type is given, it is returned as-is.
        t = [1,2,3]
        s = set(t)
        d = dict(a=1, b=2)
        self.assertIs(t, thaw(t))
        self.assertIs(s, thaw(s))
        self.assertIs(d, thaw(d))
        # This can be changed using the copy option.
        self.assertIsNot(t, thaw(t, copy=True))
        self.assertIsNot(s, thaw(s, copy=True))
        self.assertIsNot(d, thaw(d, copy=True))
        # If the type isn't frozen or thawed, then an error is raised.
        with self.assertRaises(TypeError): thaw(None)
        with self.assertRaises(TypeError): thaw(10)
        with self.assertRaises(TypeError): thaw('abc')
        # Numpy arrays are thawed via copying.
        x = np.linspace(0, 1, 25)
        x.setflags(write=False)
        y = thaw(x)
        self.assertTrue(np.array_equal(x, y))
        self.assertIsNot(x, y)
        self.assertFalse(x.flags['WRITEABLE'])
        self.assertTrue(y.flags['WRITEABLE'])
