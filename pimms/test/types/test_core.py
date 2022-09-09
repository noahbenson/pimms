# -*- coding: utf-8 -*-
################################################################################
# pimms/test/types/test_core.py
#
# Tests of the core types module in pimms: i.e., tests for the code in the
# pimms.types._core module.
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

class TestTypesCore(TestCase):
    """Tests the pimms.types._core module."""
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
    def test_is_numpydtype(self):
        from pimms.types import is_numpydtype
        import torch, numpy as np
        # is_numpydtype returns true for dtypes and dtypes alone.
        self.assertTrue(is_numpydtype(np.dtype('int')))
        self.assertTrue(is_numpydtype(np.dtype(float)))
        self.assertTrue(is_numpydtype(np.dtype(np.bool_)))
        self.assertFalse(is_numpydtype('int'))
        self.assertFalse(is_numpydtype(float))
        self.assertFalse(is_numpydtype(np.bool_))
        self.assertFalse(is_numpydtype(torch.float))
    def test_like_numpydtype(self):
        from pimms.types import like_numpydtype
        import torch, numpy as np
        # Anything that can be converted into a numpy dtype object is considered
        # to be like a numpy dtype.
        self.assertTrue(like_numpydtype('int'))
        self.assertTrue(like_numpydtype(float))
        self.assertTrue(like_numpydtype(np.bool_))
        self.assertFalse(like_numpydtype('abc'))
        self.assertFalse(like_numpydtype(10))
        self.assertFalse(like_numpydtype(...))
        # Note that None can be converted to a numpy dtype (float64).
        self.assertTrue(like_numpydtype(None))
        # numpy dtypes themselves are like numpy dtypes, as are torch dtypes.
        self.assertTrue(like_numpydtype(np.dtype(int)))
        self.assertTrue(like_numpydtype(torch.float))
    def test_to_numpydtype(self):
        from pimms.types import to_numpydtype
        import torch, numpy as np
        # Converting a numpy dtype into a dtype results in the identical dtype.
        dt = np.dtype(int)
        self.assertIs(dt, to_numpydtype(dt))
        # Torch dtypes can be converted into a numpy dtype.
        self.assertEqual(np.dtype(np.float64), to_numpydtype(torch.float64))
        self.assertEqual(np.dtype('int32'), to_numpydtype(torch.int32))
        # Ordinary tags can be converted into dtypes as well.
        self.assertEqual(np.dtype(np.float64), to_numpydtype(np.float64))
        self.assertEqual(np.dtype('int32'), to_numpydtype('int32'))
    
