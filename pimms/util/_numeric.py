# -*- coding: utf-8 -*-
################################################################################
# pimms/util/_numeric.py
#
# General numeric type utilities for pimms.
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
from ..doc import docwrap
from ._core import (alttorch, checktorch, scipy__is_sparse,
                    torch__is_tensor, torch)
from numpy import ndarray

# Numerical Types ##############################################################
from numbers import Number
def _is_number_notorch(obj):
    if isinstance(obj, Number):
        return True
    elif isinstance(obj, ndarray) and obj.shape == ():
        return isinstance(obj.item(), Number)
    else:
        return False
@alttorch(_is_number_notorch)
@docwrap
def is_number(obj):
    """Returns `True` if an object is a Python number, otherwise `False`.

    `is_number(obj)` returns `True` if the given object `obj` is an instance of
    the `numbers.Number` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Number` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Number`, otherwise `False`.
    """
    if isinstance(obj, Number):
        return True
    elif isinstance(obj, ndarray) and obj.shape == ():
        return isinstance(obj.item(), Number)
    elif torch__is_tensor(obj) and obj.shape == ():
        return isinstance(obj.item(), Number)
    else:
        return False
from numbers import Integral
def _is_integer_notorch(obj):
    if isinstance(obj, Integral):
        return True
    elif isinstance(obj, ndarray) and obj.shape == ():
        return isinstance(obj.item(), Integral)
    else:
        return False
@docwrap
def is_integer(obj):
    """Returns `True` if an object is a Python number, otherwise `False`.

    `is_integer(obj)` returns `True` if the given object `obj` is an instance of
    the `numbers.Integral` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Integral` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Integral`, otherwise `False`.
    """
    if isinstance(obj, Integral):
        return True
    elif isinstance(obj, ndarray) and obj.shape == ():
        return isinstance(obj.item(), Integral)
    elif torch__is_tensor(obj) and obj.shape == ():
        return isinstance(obj.item(), Integral)
    else:
        return False
from numbers import Real
def _is_real_notorch(obj):
    if isinstance(obj, Real):
        return True
    elif isinstance(obj, ndarray) and obj.shape == ():
        return isinstance(obj.item(), Real)
    else:
        return False
@docwrap
def is_real(obj):
    """Returns `True` if an object is a Python number, otherwise `False`.

    `is_real(obj)` returns `True` if the given object `obj` is an instance of
    the `numbers.Real` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Real` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Real`, otherwise `False`.
    """
    if isinstance(obj, Real):
        return True
    elif isinstance(obj, ndarray) and obj.shape == ():
        return isinstance(obj.item(), Real)
    elif torch__is_tensor(obj) and obj.shape == ():
        return isinstance(obj.item(), Real)
    else:
        return False
from numbers import Complex
def _is_complex_notorch(obj):
    if isinstance(obj, Complex):
        return True
    elif isinstance(obj, ndarray) and obj.shape == ():
        return isinstance(obj.item(), Complex)
    else:
        return False
@docwrap
def is_complex(obj):
    """Returns `True` if an object is a complex number, otherwise `False`.

    `is_complex(obj)` returns `True` if the given object `obj` is an instance of
    the `numbers.Complex` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Complex` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Complex`, otherwise `False`.
    """
    if isinstance(obj, Complex):
        return True
    elif isinstance(obj, ndarray) and obj.shape == ():
        return isinstance(obj.item(), Complex)
    elif torch__is_tensor(obj) and obj.shape == ():
        return isinstance(obj.item(), Complex)
    else:
        return False
