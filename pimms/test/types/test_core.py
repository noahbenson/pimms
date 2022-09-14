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
        # The frozen option can be used to test whether an array is frozen
        # or not. This is judged by the array's 'WRITEABLE' flag.
        self.assertFalse(is_array(arr, frozen=True))
        self.assertTrue(is_array(arr, frozen=False))
        self.assertFalse(is_array(mtx, frozen=True))
        self.assertTrue(is_array(mtx, frozen=False))
        # If we change the flags of these arrays, they become frozen.
        arr.setflags(write=False)
        mtx.setflags(write=False)
        self.assertTrue(is_array(arr, frozen=True))
        self.assertFalse(is_array(arr, frozen=False))
        self.assertTrue(is_array(mtx, frozen=True))
        self.assertFalse(is_array(mtx, frozen=False))
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
        self.assertIs(arr, to_array(arr, sparse=False, frozen=False))
        self.assertIs(arr, to_array(arr, quant=False))
        # If we change the parameters of the returned array, we will get
        # different (but typically equal) objects back.
        self.assertIsNot(arr, to_array(arr, frozen=True))
        self.assertTrue(np.array_equal(arr, to_array(arr, frozen=True)))
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
        # The frozen flag ensures that the return value does or does not have
        # the writeable flag set.
        self.assertFalse(to_array(mtx, frozen=True).flags['WRITEABLE'])
        self.assertTrue(np.array_equal(to_array(mtx, frozen=True), mtx))
        self.assertIsNot(to_array(mtx, frozen=True), mtx)
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

    # PyTorch and Numpy Helper Functions #######################################
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
        from pimms.types import to_frozenarray
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
