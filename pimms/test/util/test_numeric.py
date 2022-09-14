# -*- coding: utf-8 -*-
################################################################################
# pimms/test/util/test_numeric.py
#
# Tests of the numeric module in pimms: i.e., tests for the code in the
# pimms.util._numeric module.
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

class TestUtilNumeric(TestCase):
    """Tests the pimms.util._numeric module."""
    def test_is_number(self):
        from pimms import is_number
        import torch, numpy as np
        # is_number returns True for numbers and False for non-numbers.
        self.assertTrue(is_number(0))
        self.assertTrue(is_number(5))
        self.assertTrue(is_number(10.0))
        self.assertTrue(is_number(-2.0 + 9.0j))
        self.assertFalse(is_number('abc'))
        self.assertFalse(is_number('10'))
        self.assertFalse(is_number(None))
        # Scalar arrays and tensors (but not 1-element arrays and tensors) are
        # also considered numbers.
        self.assertTrue(is_number(np.array(5)))
        self.assertTrue(is_number(np.array(10.0 + 2.0j)))
        self.assertTrue(is_number(torch.tensor(5)))
        self.assertTrue(is_number(torch.tensor(10.0 + 2.0j)))
        self.assertFalse(is_number(torch.tensor([1,2,3])))
        self.assertFalse(is_number(np.array([1])))
    def test_is_integer(self):
        from pimms import is_integer
        import torch, numpy as np
        # is_integer returns True for integers and False for non-integers.
        self.assertTrue(is_integer(0))
        self.assertTrue(is_integer(5))
        self.assertFalse(is_integer(10.0))
        self.assertFalse(is_integer(-2.0 + 9.0j))
        self.assertFalse(is_integer('abc'))
        self.assertFalse(is_integer('10'))
        self.assertFalse(is_integer(None))
        # Scalar arrays and tensors (but not 1-element arrays and tensors) are
        # also considered integers.
        self.assertTrue(is_integer(np.array(5)))
        self.assertFalse(is_integer(np.array(10.0 + 2.0j)))
        self.assertTrue(is_integer(torch.tensor(5)))
        self.assertFalse(is_integer(torch.tensor(10.0 + 2.0j)))
        self.assertFalse(is_integer(torch.tensor([1,2,3])))
        self.assertFalse(is_integer(np.array([1])))
    def test_is_real(self):
        from pimms import is_real
        import torch, numpy as np
        # is_real returns True for reals and False for non-reals.
        self.assertTrue(is_real(0))
        self.assertTrue(is_real(5))
        self.assertTrue(is_real(10.0))
        self.assertFalse(is_real(-2.0 + 9.0j))
        self.assertFalse(is_real('abc'))
        self.assertFalse(is_real('10'))
        self.assertFalse(is_real(None))
        # Scalar arrays and tensors (but not 1-element arrays and tensors) are
        # also considered integers.
        self.assertTrue(is_real(np.array(5)))
        self.assertFalse(is_real(np.array(10.0 + 2.0j)))
        self.assertTrue(is_real(torch.tensor(5)))
        self.assertFalse(is_real(torch.tensor(10.0 + 2.0j)))
        self.assertFalse(is_real(torch.tensor([1,2,3])))
        self.assertFalse(is_real(np.array([1])))
    def test_is_complex(self):
        from pimms import is_complex
        import torch, numpy as np
        # is_complex returns True for complexs and False for non-complexs.
        self.assertTrue(is_complex(0))
        self.assertTrue(is_complex(5))
        self.assertTrue(is_complex(10.0))
        self.assertTrue(is_complex(-2.0 + 9.0j))
        self.assertFalse(is_complex('abc'))
        self.assertFalse(is_complex('10'))
        self.assertFalse(is_complex(None))
        # Scalar arrays and tensors (but not 1-element arrays and tensors) are
        # also considered integers.
        self.assertTrue(is_complex(np.array(5)))
        self.assertTrue(is_complex(np.array(10.0 + 2.0j)))
        self.assertTrue(is_complex(torch.tensor(5)))
        self.assertTrue(is_complex(torch.tensor(10.0 + 2.0j)))
        self.assertFalse(is_complex(torch.tensor([1,2,3])))
        self.assertFalse(is_complex(np.array([1])))
    
