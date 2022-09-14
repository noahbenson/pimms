# -*- coding: utf-8 -*-
################################################################################
# pimms/test/types/test_quantity.py
#
# Tests of the quantity module in pimms: i.e., tests for the code in the
# pimms.types._quantity module.
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

class TestTypesQuantity(TestCase):
    """Tests the pimms.types._quantity module."""
    def test_default_ureg(self):
        from pimms import default_ureg
        from pint import UnitRegistry
        # You can set the pimms default units registry (pimms.units) temporarily
        # in an execution context using the default_ureg function:
        ureg = UnitRegistry()
        with default_ureg(ureg):
            from pimms import units
            self.assertIs(units, ureg)
        # This only affects the code inside the with-block.
        from pimms import units
        self.assertIsNot(units, ureg)
    def test_like_unit(self):
        from pimms import (like_unit, units)
        # like_unit returns True when its argument is like a unit. This can be,
        # for one, objects that already are units.
        self.assertTrue(like_unit(units.mm))
        self.assertTrue(like_unit(units.count))
        # Otherwise, only strings may be unit-like.
        self.assertFalse(like_unit(None))
        self.assertFalse(like_unit(10))
        self.assertFalse(like_unit([]))
        # Strings that name units are unit-like.
        self.assertTrue(like_unit('mm'))
        self.assertTrue(like_unit('count'))
    def test_unit(self):
        from pimms import (unit, units)
        from pint.errors import UndefinedUnitError
        # unit converts its argument into a unit. Units themselves are returned
        # as-is.
        u = units.mm
        self.assertIs(u, unit(u))
        u = units.count
        self.assertIs(u, unit(u))
        # Strings that name units can be converted into units.
        self.assertEqual(units.mm, unit('mm'))
        self.assertEqual(units.count, unit('count'))
        # unit(None) returns the dimensionless unit (which is the unit same as
        # count).
        self.assertEqual(units.dimensionless, unit(None))
        # If the argument isn't a valid unit, then an error is raised.
        with self.assertRaises(ValueError): unit(10)
        with self.assertRaises(ValueError): unit([])
        with self.assertRaises(UndefinedUnitError): unit('fdjsklfajdk')
    def test_alike_units(self):
        from pimms import (alike_units, units)
        # alike_units tells us if two units are of the same unit category, like
        # meters and feet both being lengths.
        self.assertTrue(alike_units(units.mm, units.feet))
        self.assertTrue(alike_units(units.seconds, units.days))
        self.assertTrue(alike_units(units.rad, units.degree))
        self.assertFalse(alike_units(units.rad, units.feet))
        self.assertFalse(alike_units(units.days, units.mm))
        self.assertFalse(alike_units(units.mm, units.degree))
    def test_quant(self):
        from pimms import (default_ureg, units, quant)
        from pint import (UnitRegistry, Quantity)
        import torch, numpy as np
        # The quant function lets you create pint quantities; by default these
        # are registered in the pimms.units UnitRegistry.
        circ = np.linspace(0, 1, 25)
        self.assertIsInstance(quant(10, 'mm'), Quantity)
        self.assertIsInstance(quant([10, 30, 40], 'days'), Quantity)
        self.assertIsInstance(quant(circ, 'turns'), Quantity)
        self.assertEqual(quant(10, 'mm').m, 10)
        self.assertTrue(np.array_equal(quant(circ, 'turns').m, circ))
        # Iterables are upgraded to numpy arrays when applicable.
        self.assertIsInstance(quant([10, 30, 40], 'days').m, np.ndarray)
        self.assertTrue(np.array_equal(quant([10, 30, 40], 'days').m,
                                       [10, 30, 40]))
        # Units are registered in the pimms.units registry by default.
        self.assertEqual(quant(10, 'mm').u, units.mm)
        self.assertEqual(quant([10, 30, 40], 'days').u, units.days)
        self.assertEqual(quant(circ, 'turns').u, units.turns)
        # Tensors also work as quantities.
        t = torch.linspace(0,1,5)
        tq = quant(t, 'mm')
        self.assertIsInstance(tq.m, torch.Tensor)
        self.assertIs(tq.m, t)
        # Changing the default unit registry changes how these are registered.
        ureg = UnitRegistry()
        with default_ureg(ureg):
            q = quant(10, 'mm')
        self.assertIsInstance(q, ureg.Quantity)
        self.assertFalse(isinstance(q, units.Quantity))
        # This can also be done with the ureg option.
        q = quant(10, 'mm', ureg=ureg)
        self.assertIsInstance(q, ureg.Quantity)
        self.assertFalse(isinstance(q, units.Quantity))
    def test_mag(self):
        from pimms import (mag, units, quant)
        import numpy as np
        from pint.errors import DimensionalityError
        # The mag function extracts the magnitude from a quantity.
        m = np.linspace(0, 1, 5)
        q = quant(m, 'second')
        self.assertIs(m, mag(q))
        self.assertEqual(10, mag(10 * units.mm))
        # mag can extract in the quantity's native units if none are given, or
        # in another unit, if requested.
        self.assertTrue(np.array_equal(m * 1000, mag(q, 'ms')))
        # If the value passed to mag is not a quantity, it is returned as-is,
        # and is assumed to be in the correct unit.
        self.assertIs(m, mag(m))
        # The same is true, even if a unit is passed: non-quantities are always
        # assumed to be in the correct units.
        self.assertIs(m, mag(m, 'ms'))
        self.assertIs(m, mag(m, 'days'))
        self.assertIs(m, mag(m, 'feet'))
        # If unit=None, then the argument must not be a quantity.
        self.assertIs(m, mag(m, None))
        with self.assertRaises(ValueError): mag(q, None)
        # If the unit doesn't match, a DimensionalityError is raised.
        with self.assertRaises(DimensionalityError): mag(q, 'miles')
    def test_promote(self):
        from pimms import (promote, units)
        import torch, numpy as np
        # promote converts all arguments into quantities.
        a = np.linspace(0, 1, 5)
        b = np.arange(10)
        (qa, qb) = promote(a, b)
        self.assertIsInstance(qa, units.Quantity)
        self.assertIsInstance(qb, units.Quantity)
        self.assertIs(qa.m, a)
        self.assertIs(qb.m, b)
        # If one of the arguments is a tensor, all results will be tensors.
        c = torch.linspace(0, 1, 5)
        (qa, qb, qc) = promote(a, b, c)
        self.assertIsInstance(qa, units.Quantity)
        self.assertIsInstance(qb, units.Quantity)
        self.assertIsInstance(qc, units.Quantity)
        self.assertIsInstance(qa.m, torch.Tensor)
        self.assertIsInstance(qb.m, torch.Tensor)
        self.assertIsInstance(qc.m, torch.Tensor)
        self.assertTrue(np.array_equal(qa.numpy(), a))
        self.assertTrue(np.array_equal(qb.numpy(), b))
        self.assertIs(qc.m, c)
