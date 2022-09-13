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
    # NumPy Utilities ##########################################################
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
    def test_is_array(self):
        from pimms import (is_array, quant)
        from numpy import (array, linspace, dot)
        from scipy.sparse import csr_matrix
        import torch, numpy as np
        # By default, is_array() returns True for numpy arrays, scipy sparse
        # matrices, and quantities of these.
        arr = linspace(0, 1, 25)
        mtx = dot(linspace(0, 1, 10)[:,None], linspace(1, 2, 10)[None,:])
        sp_mtx = csr_matrix(([1.0, 0.5, 0.5, 0.2, 0.1],
                             ([0, 0, 4, 5, 9], [4, 9, 4, 1, 8])),
                            shape=(10, 10), dtype=float)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        self.assertTrue(is_array(arr))
        self.assertTrue(is_array(mtx))
        self.assertTrue(is_array(sp_mtx))
        self.assertTrue(is_array(q_arr))
        self.assertTrue(is_array(q_mtx))
        self.assertTrue(is_array(q_sp_mtx))
        # Things like lists, numbers, and torch tensors are not arrays.
        self.assertFalse(is_array('abc'))
        self.assertFalse(is_array(10))
        self.assertFalse(is_array([12.0, 0.5, 3.2]))
        self.assertFalse(is_array(torch.tensor([1.0, 2.0, 3.0])))
        self.assertFalse(is_array(quant(torch.tensor([1.0, 2.0, 3.0]), 'mm')))
        # We can use the dtype argument to restrict what we consider an array by
        # its dtype. The dtype of the is_array argument must be a sub-dtype of
        # the dtype parameter.
        self.assertTrue(is_array(arr, dtype=np.number))
        self.assertTrue(is_array(arr, dtype=arr.dtype))
        self.assertFalse(is_array(arr, dtype=np.str_))
        # If a tuple is passed for the dtype, the dtype must match one of the
        # tuple's members exactly.
        self.assertTrue(is_array(mtx, dtype=(mtx.dtype,)))
        self.assertTrue(is_array(mtx, dtype=(mtx.dtype,np.dtype(int),np.str_)))
        self.assertFalse(is_array(mtx, dtype=(np.dtype(int),np.str_)))
        self.assertFalse(is_array(np.array([], dtype=np.int32),
                                  dtype=(np.int64,)))
        # torch dtypes can be interpreted into numpy dtypes.
        self.assertTrue(is_array(mtx, dtype=torch.as_tensor(mtx).dtype))
        # We can use the ndim argument to restrict the number of dimensions that
        # an array can have in order to be considered a matching array.
        # Typically, this is just the number of dimensions.
        self.assertTrue(is_array(arr, ndim=1))
        self.assertTrue(is_array(mtx, ndim=2))
        self.assertFalse(is_array(arr, ndim=2))
        self.assertFalse(is_array(mtx, ndim=1))
        # Alternately, a tuple may be given, in which case any of the dimension
        # counts in the tuple are accepted.
        self.assertTrue(is_array(mtx, ndim=(1,2)))
        self.assertTrue(is_array(arr, ndim=(1,2)))
        self.assertFalse(is_array(mtx, ndim=(1,3)))
        self.assertFalse(is_array(arr, ndim=(0,2)))
        # Scalar arrays have 0 dimensions.
        self.assertTrue(is_array(array(0), ndim=0))
        # The shape option is a more specific version of the ndim parameter. It
        # lets you specify what kind of shape is required of the array. The most
        # straightforward usage is to require a specific shape.
        self.assertTrue(is_array(arr, shape=(25,)))
        self.assertTrue(is_array(mtx, shape=(10,10)))
        self.assertFalse(is_array(arr, shape=(25,25)))
        self.assertFalse(is_array(mtx, shape=(10,)))
        # A -1 value that appears in the shape option represents any size along
        # that dimension (a wildcard). Any number of -1s can be included.
        self.assertTrue(is_array(arr, shape=(-1,)))
        self.assertTrue(is_array(mtx, shape=(-1,10)))
        self.assertTrue(is_array(mtx, shape=(10,-1)))
        self.assertTrue(is_array(mtx, shape=(-1,-1)))
        self.assertFalse(is_array(mtx, shape=(1,-1)))
        # No more than 1 ellipsis may be included in the shape to indicate that
        # any number of dimensions, with any sizes, can appear in place of the
        # ellipsis.
        self.assertTrue(is_array(arr, shape=(...,25)))
        self.assertTrue(is_array(arr, shape=(25,...)))
        self.assertFalse(is_array(arr, shape=(25,...,25)))
        self.assertTrue(is_array(mtx, shape=(...,10)))
        self.assertTrue(is_array(mtx, shape=(10,...)))
        self.assertTrue(is_array(mtx, shape=(10,...,10)))
        self.assertTrue(is_array(mtx, shape=(10,10,...)))
        self.assertTrue(is_array(mtx, shape=(...,10,10)))
        self.assertFalse(is_array(mtx, shape=(10,...,10,10)))
        self.assertFalse(is_array(mtx, shape=(10,10,...,10)))
        self.assertTrue(is_array(np.zeros((1,2,3,4,5)), shape=(1,...,4,5)))
        # The readonly option can be used to test whether an array is readonly
        # or not. This is judged by the array's 'WRITEABLE' flag.
        self.assertFalse(is_array(arr, readonly=True))
        self.assertTrue(is_array(arr, readonly=False))
        self.assertFalse(is_array(mtx, readonly=True))
        self.assertTrue(is_array(mtx, readonly=False))
        # If we change the flags of these arrays, they become readonly.
        arr.setflags(write=False)
        mtx.setflags(write=False)
        self.assertTrue(is_array(arr, readonly=True))
        self.assertFalse(is_array(arr, readonly=False))
        self.assertTrue(is_array(mtx, readonly=True))
        self.assertFalse(is_array(mtx, readonly=False))
        # The sparse option can test whether an object is a sparse matrix or
        # not. By default sparse is None, meaning that it doesn't matter whether
        # an object is sparse, but sometimes you want to check for strict
        # numpy arrays only.
        self.assertTrue(is_array(arr, sparse=False))
        self.assertTrue(is_array(mtx, sparse=False))
        self.assertFalse(is_array(sp_mtx, sparse=False))
        self.assertFalse(is_array(arr, sparse=True))
        self.assertFalse(is_array(mtx, sparse=True))
        self.assertTrue(is_array(sp_mtx, sparse=True))
        # You can also require a kind of sparse matrix.
        self.assertTrue(is_array(sp_mtx, sparse='csr'))
        self.assertFalse(is_array(sp_mtx, sparse='csc'))
        # The quant option can be used to control whether the object must or
        # must not be a quantity.
        self.assertTrue(is_array(arr, quant=False))
        self.assertTrue(is_array(mtx, quant=False))
        self.assertFalse(is_array(arr, quant=True))
        self.assertFalse(is_array(mtx, quant=True))
        self.assertTrue(is_array(q_arr, quant=True))
        self.assertTrue(is_array(q_mtx, quant=True))
        self.assertFalse(is_array(q_arr, quant=False))
        self.assertFalse(is_array(q_mtx, quant=False))
        # The units option can be used to require that either an object have
        # no units (or is not a quantity) or that it have specific units.
        self.assertTrue(is_array(arr, unit=None))
        self.assertTrue(is_array(mtx, unit=None))
        self.assertFalse(is_array(arr, unit='mm'))
        self.assertFalse(is_array(mtx, unit='s'))
        self.assertFalse(is_array(q_arr, unit=None))
        self.assertFalse(is_array(q_mtx, unit=None))
        self.assertTrue(is_array(q_arr, unit='mm'))
        self.assertTrue(is_array(q_mtx, unit='s'))
        self.assertFalse(is_array(q_arr, unit='s'))
        self.assertFalse(is_array(q_mtx, unit='mm'))
    def test_to_array(self):
        from pimms import (to_array, quant, is_quant, units)
        from numpy import (array, linspace, dot)
        from scipy.sparse import (csr_matrix, issparse)
        import torch, numpy as np
        # We'll use a few objects throughout our tests, which we setup now.
        arr = linspace(0, 1, 25)
        mtx = dot(linspace(0, 1, 10)[:,None], linspace(0, 2, 10)[None,:])
        sp_mtx = csr_matrix(([1.0, 0.5, 0.5, 0.2, 0.1],
                             ([0, 0, 4, 5, 9], [4, 9, 4, 1, 8])),
                            shape=(10, 10), dtype=float)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        # For an object that is already a numpy array, any call that doesn't
        # request a copy and that doesn't change its parameters will return the
        # identical object.
        self.assertIs(arr, to_array(arr))
        self.assertIs(arr, to_array(arr, sparse=False, readonly=False))
        self.assertIs(arr, to_array(arr, quant=False))
        # If we change the parameters of the returned array, we will get
        # different (but typically equal) objects back.
        self.assertIsNot(arr, to_array(arr, readonly=True))
        self.assertTrue(np.array_equal(arr, to_array(arr, readonly=True)))
        # We can also request that a copy be made like with np.array.
        self.assertIsNot(arr, to_array(arr, copy=True))
        self.assertTrue(np.array_equal(arr, to_array(arr, copy=True)))
        # The sparse flag can be used to convert to/from a sparse array.
        self.assertIsInstance(to_array(sp_mtx, sparse=False), np.ndarray)
        self.assertTrue(np.array_equal(to_array(sp_mtx, sparse=False),
                                       sp_mtx.todense()))
        self.assertTrue(issparse(to_array(mtx, sparse=True)))
        self.assertTrue(np.array_equal(to_array(mtx, sparse=True).todense(),
                                       mtx))
        # The readonly flag ensures that the return value does or does not have
        # the writeable flag set.
        self.assertFalse(to_array(mtx, readonly=True).flags['WRITEABLE'])
        self.assertTrue(np.array_equal(to_array(mtx, readonly=True), mtx))
        self.assertIsNot(to_array(mtx, readonly=True), mtx)
        # The quant argument can be used to enforce the return of quantities or
        # non-quantities.
        self.assertIsInstance(to_array(arr, quant=True), units.Quantity)
        # The unit parameter can be used to specify what unit to use.
        self.assertTrue(np.array_equal(q_arr.m,
                                       to_array(arr, quant=True, unit='mm').m))
        # If no unit is provided, then dimensionless units are assumed (1
        # dimensionless is equivalent to 1 count, 1 turn, and a few others).
        self.assertEqual(to_array(arr, quant=True).u, units.dimensionless)
        # We can also use unit to extract a specific unit from a quantity.
        self.assertEqual(1000, to_array(1 * units.meter, unit='mm').m)
        # However, a non-quantity is always assumed to already have the units
        # requested, so converting it to a particular unit (but not converting
        # it to a quantity) results in the same object.
        self.assertIs(to_array(arr, unit='mm'), arr)
        # An error is raised if you try to request no units for a quantity.
        with self.assertRaises(ValueError):
            to_array(arr, quant=True, unit=None)
    # PyTorch Utilities ########################################################
    def test_is_torchdtype(self):
        from pimms.types import is_torchdtype
        import torch, numpy as np
        # is_torchdtype returns true for torch's dtypes and its dtypes alone.
        self.assertTrue(is_torchdtype(torch.int))
        self.assertTrue(is_torchdtype(torch.float))
        self.assertTrue(is_torchdtype(torch.bool))
        self.assertFalse(is_torchdtype('int'))
        self.assertFalse(is_torchdtype(float))
        self.assertFalse(is_torchdtype(np.bool_))
    def test_like_torchdtype(self):
        from pimms.types import like_torchdtype
        import torch, numpy as np
        # Anything that can be converted into a torch dtype object is considered
        # to be like a torch dtype.
        self.assertTrue(like_torchdtype('int'))
        self.assertTrue(like_torchdtype(float))
        self.assertTrue(like_torchdtype(np.bool_))
        self.assertFalse(like_torchdtype('abc'))
        self.assertFalse(like_torchdtype(10))
        self.assertFalse(like_torchdtype(...))
        # Note that None can be converted to a torch dtype (float64).
        self.assertTrue(like_torchdtype(None))
        # torch dtypes themselves are like torch dtypes.
        self.assertTrue(like_torchdtype(torch.float))
    def test_to_torchdtype(self):
        from pimms.types import to_torchdtype
        import torch, numpy as np
        # Converting a numpy dtype into a dtype results in the identical dtype.
        dt = torch.int
        self.assertIs(dt, to_torchdtype(dt))
        # Numpy dtypes can be converted into a torch dtype.
        self.assertEqual(torch.float64, to_torchdtype(np.dtype('float64')))
        self.assertEqual(torch.int32, to_torchdtype(np.int32))
    def test_is_tensor(self):
        from pimms import (is_tensor, quant)
        from scipy.sparse import csr_matrix
        import torch, numpy as np
        # By default, is_tensor() returns True for PyTorch tensors and
        # quantities whose magnitudes are PyTorch tensors.
        arr = torch.linspace(0, 1, 25)
        mtx = torch.mm(torch.linspace(0, 1, 10)[:,None],
                       torch.linspace(1, 2, 10)[None,:])
        sp_mtx = torch.sparse_coo_tensor(torch.tensor([[0, 0, 4, 5, 9],
                                                       [4, 9, 4, 1, 8]]),
                                         torch.tensor([1, 0.5, 0.5, 0.2, 0.1]),
                                         (10, 10),
                                         dtype=float)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        self.assertTrue(is_tensor(arr))
        self.assertTrue(is_tensor(mtx))
        self.assertTrue(is_tensor(sp_mtx))
        self.assertTrue(is_tensor(q_arr))
        self.assertTrue(is_tensor(q_mtx))
        self.assertTrue(is_tensor(q_sp_mtx))
        # Things like lists, numbers, and numpy arrays are not tensors.
        self.assertFalse(is_tensor('abc'))
        self.assertFalse(is_tensor(10))
        self.assertFalse(is_tensor([12.0, 0.5, 3.2]))
        self.assertFalse(is_tensor(np.array([1.0, 2.0, 3.0])))
        self.assertFalse(is_tensor(quant(np.array([1.0, 2.0, 3.0]), 'mm')))
        # We can use the dtype argument to restrict what we consider an array by
        # its dtype. The dtype of the is_array argument must be a sub-dtype of
        # the dtype parameter.
        self.assertTrue(is_tensor(arr, dtype=arr.dtype))
        self.assertFalse(is_tensor(arr, dtype=torch.int))
        # If a tuple is passed for the dtype, the dtype must match one of the
        # tuple's elements.
        self.assertTrue(is_tensor(mtx, dtype=(mtx.dtype,)))
        self.assertTrue(is_tensor(mtx, dtype=(mtx.dtype, torch.int)))
        self.assertFalse(is_tensor(mtx, dtype=(torch.int, torch.bool)))
        self.assertFalse(is_tensor(torch.tensor([], dtype=torch.int32),
                                   dtype=torch.int64))
        # torch dtypes can be interpreted into PyTorch dtypes.
        self.assertTrue(is_tensor(mtx, dtype=mtx.numpy().dtype))
        # We can use the ndim argument to restrict the number of dimensions that
        # an array can have in order to be considered a matching tensor.
        # Typically, this is just the number of dimensions.
        self.assertTrue(is_tensor(arr, ndim=1))
        self.assertTrue(is_tensor(mtx, ndim=2))
        self.assertFalse(is_tensor(arr, ndim=2))
        self.assertFalse(is_tensor(mtx, ndim=1))
        # Alternately, a tuple may be given, in which case any of the dimension
        # counts in the tuple are accepted.
        self.assertTrue(is_tensor(mtx, ndim=(1,2)))
        self.assertTrue(is_tensor(arr, ndim=(1,2)))
        self.assertFalse(is_tensor(mtx, ndim=(1,3)))
        self.assertFalse(is_tensor(arr, ndim=(0,2)))
        # Scalar tensors have 0 dimensions.
        self.assertTrue(is_tensor(torch.tensor(0), ndim=0))
        # The shape option is a more specific version of the ndim parameter. It
        # lets you specify what kind of shape is required of the tensor. The
        # most straightforward usage is to require a specific shape.
        self.assertTrue(is_tensor(arr, shape=(25,)))
        self.assertTrue(is_tensor(mtx, shape=(10,10)))
        self.assertFalse(is_tensor(arr, shape=(25,25)))
        self.assertFalse(is_tensor(mtx, shape=(10,)))
        # A -1 value that appears in the shape option represents any size along
        # that dimension (a wildcard). Any number of -1s can be included.
        self.assertTrue(is_tensor(arr, shape=(-1,)))
        self.assertTrue(is_tensor(mtx, shape=(-1,10)))
        self.assertTrue(is_tensor(mtx, shape=(10,-1)))
        self.assertTrue(is_tensor(mtx, shape=(-1,-1)))
        self.assertFalse(is_tensor(mtx, shape=(1,-1)))
        # No more than 1 ellipsis may be included in the shape to indicate that
        # any number of dimensions, with any sizes, can appear in place of the
        # ellipsis.
        self.assertTrue(is_tensor(arr, shape=(...,25)))
        self.assertTrue(is_tensor(arr, shape=(25,...)))
        self.assertFalse(is_tensor(arr, shape=(25,...,25)))
        self.assertTrue(is_tensor(mtx, shape=(...,10)))
        self.assertTrue(is_tensor(mtx, shape=(10,...)))
        self.assertTrue(is_tensor(mtx, shape=(10,...,10)))
        self.assertTrue(is_tensor(mtx, shape=(10,10,...)))
        self.assertTrue(is_tensor(mtx, shape=(...,10,10)))
        self.assertFalse(is_tensor(mtx, shape=(10,...,10,10)))
        self.assertFalse(is_tensor(mtx, shape=(10,10,...,10)))
        self.assertTrue(is_tensor(torch.zeros((1,2,3,4,5)), shape=(1,...,4,5)))
        # The sparse option can test whether an object is a sparse tensor or
        # not. By default sparse is None, meaning that it doesn't matter whether
        # an object is sparse, but sometimes you want to check for strict
        # sparsity requirements.
        self.assertTrue(is_tensor(arr, sparse=False))
        self.assertTrue(is_tensor(mtx, sparse=False))
        self.assertFalse(is_tensor(sp_mtx, sparse=False))
        self.assertFalse(is_tensor(arr, sparse=True))
        self.assertFalse(is_tensor(mtx, sparse=True))
        self.assertTrue(is_tensor(sp_mtx, sparse=True))
        # The quant option can be used to control whether the object must or
        # must not be a quantity.
        self.assertTrue(is_tensor(arr, quant=False))
        self.assertTrue(is_tensor(mtx, quant=False))
        self.assertFalse(is_tensor(arr, quant=True))
        self.assertFalse(is_tensor(mtx, quant=True))
        self.assertTrue(is_tensor(q_arr, quant=True))
        self.assertTrue(is_tensor(q_mtx, quant=True))
        self.assertFalse(is_tensor(q_arr, quant=False))
        self.assertFalse(is_tensor(q_mtx, quant=False))
        # The units option can be used to require that either an object have
        # no units (or is not a quantity) or that it have specific units.
        self.assertTrue(is_tensor(arr, unit=None))
        self.assertTrue(is_tensor(mtx, unit=None))
        self.assertFalse(is_tensor(arr, unit='mm'))
        self.assertFalse(is_tensor(mtx, unit='s'))
        self.assertFalse(is_tensor(q_arr, unit=None))
        self.assertFalse(is_tensor(q_mtx, unit=None))
        self.assertTrue(is_tensor(q_arr, unit='mm'))
        self.assertTrue(is_tensor(q_mtx, unit='s'))
        self.assertFalse(is_tensor(q_arr, unit='s'))
        self.assertFalse(is_tensor(q_mtx, unit='mm'))
    def test_to_tensor(self):
        from pimms import (to_tensor, quant, is_quant, units)
        import torch, numpy as np
        # We'll use a few objects throughout our tests, which we setup now.
        arr = torch.linspace(0, 1, 25)
        mtx = torch.mm(torch.linspace(0, 1, 10)[:,None],
                       torch.linspace(0, 2, 10)[None,:])
        sp_mtx = torch.sparse_coo_tensor(torch.tensor([[0, 0, 4, 5, 9],
                                                       [4, 9, 4, 1, 8]]),
                                         torch.tensor([1, 0.5, 0.5, 0.2, 0.1]),
                                         (10,10),
                                         dtype=float)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        # For an object that is already a numpy array, any call that doesn't
        # request a copy and that doesn't change its parameters will return the
        # identical object.
        self.assertIs(arr, to_tensor(arr))
        self.assertIs(arr, to_tensor(arr, sparse=False))
        self.assertIs(arr, to_tensor(arr, quant=False))
        # If we change the parameters of the returned array, we will get
        # different (but typically equal) objects back.
        self.assertTrue(torch.equal(arr, to_tensor(arr, requires_grad=True)))
        # We can also request that a copy be made like with np.array.
        self.assertIsNot(arr, to_tensor(arr, copy=True))
        self.assertTrue(torch.equal(arr, to_tensor(arr, copy=True)))
        # The sparse flag can be used to convert to/from a sparse array.
        self.assertIsInstance(to_tensor(sp_mtx, sparse=False), torch.Tensor)
        self.assertTrue(torch.equal(to_tensor(sp_mtx, sparse=False),
                                    sp_mtx.to_dense()))
        self.assertTrue(to_tensor(mtx, sparse=True).is_sparse)
        self.assertTrue(torch.equal(to_tensor(mtx, sparse=True).to_dense(),
                                    mtx))
        # The quant argument can be used to enforce the return of quantities or
        # non-quantities.
        self.assertIsInstance(to_tensor(arr, quant=True), units.Quantity)
        # The unit parameter can be used to specify what unit to use.
        self.assertTrue(torch.equal(q_arr.m,
                                    to_tensor(arr, quant=True, unit='mm').m))
        # If no unit is provided, then dimensionless units are assumed (1
        # dimensionless is equivalent to 1 count, 1 turn, and a few others).
        self.assertEqual(to_tensor(arr, quant=True).u, units.dimensionless)
        # We can also use unit to extract a specific unit from a quantity.
        self.assertEqual(1000, to_tensor(1 * units.meter, unit='mm').m)
        # However, a non-quantity is always assumed to already have the units
        # requested, so converting it to a particular unit (but not converting
        # it to a quantity) results in the same object.
        self.assertIs(to_tensor(arr, unit='mm'), arr)
        # An error is raised if you try to request no units for a quantity.
        with self.assertRaises(ValueError):
            to_tensor(arr, quant=True, unit=None)
    def test_is_numeric(self):
        from pimms import is_numeric
        import torch, numpy as np
        from scipy.sparse import csr_matrix
        # The is_numeric function is just a wrapper around is_array and
        # is_tensor that calls one or the other depending on whether the object
        # requested is a tensor or not. I.e., it passes all arguments through
        # and merely switches on the type.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        a = sp_a.todense()
        t = sp_t.to_dense()
        self.assertTrue(is_numeric(a))
        self.assertTrue(is_numeric(t))
        self.assertTrue(is_numeric(sp_a))
        self.assertTrue(is_numeric(sp_t))
        self.assertFalse(is_numeric('abc'))
        self.assertFalse(is_numeric([1,2,3]))
    def test_to_numeric(self):
        from pimms import to_numeric
        import torch, numpy as np
        from scipy.sparse import csr_matrix
        # The is_numeric function is just a wrapper around to_array and
        # to_tensor that calls one or the other depending on whether the object
        # requested is a tensor or not. I.e., it passes all arguments through
        # and merely switches on the type.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        a = np.array(sp_a.todense())
        t = sp_t.to_dense()
        self.assertIs(a, to_numeric(a))
        self.assertIs(t, to_numeric(t))
        self.assertIs(sp_a, to_numeric(sp_a))
        self.assertIs(sp_t, to_numeric(sp_t))
        self.assertIsInstance(to_numeric([1,2,3]), np.ndarray)
    def test_is_sparse(self):
        from pimms import is_sparse
        import torch, numpy as np
        from scipy.sparse import csr_matrix
        # is_sparse returns True for any sparse array and False for anything
        # other than a sparse array.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        self.assertTrue(is_sparse(sp_a))
        self.assertTrue(is_sparse(sp_t))
        self.assertFalse(is_sparse(sp_a.todense()))
        self.assertFalse(is_sparse(sp_t.to_dense()))
    def test_to_sparse(self):
        from pimms import to_sparse
        import torch, numpy as np
        from scipy.sparse import issparse
        # to_sparse supports the arguments of to_array and to_tensor (because it
        # simply calls through to these functions), but it always returns a
        # sparse object.
        m = np.array([[1.0, 0, 0, 0], [0, 0, 0, 0],
                      [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
        t = torch.tensor(m)
        self.assertTrue(issparse(to_sparse(m)))
        self.assertTrue(to_sparse(t).is_sparse)
    def test_is_dense(self):
        from pimms import is_dense
        import torch, numpy as np
        from scipy.sparse import csr_matrix
        # is_dense returns True for any dense array and False for anything
        # other than a dense array.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        self.assertTrue(is_dense(sp_a.todense()))
        self.assertTrue(is_dense(sp_t.to_dense()))
        self.assertFalse(is_dense(sp_a))
        self.assertFalse(is_dense(sp_t))
    def test_to_dense(self):
        from pimms import to_dense
        import torch, numpy as np
        from scipy.sparse import (issparse, csr_matrix)
        # to_dense supports the arguments of to_array and to_tensor (because it
        # simply calls through to these functions), but it always returns a
        # dense object.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        self.assertFalse(issparse(to_dense(sp_a)))
        self.assertFalse(to_dense(sp_t).is_sparse)
    
        