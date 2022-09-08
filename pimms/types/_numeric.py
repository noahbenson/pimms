# -*- coding: utf-8 -*-
################################################################################
# pimms/types/_numeric.py
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

# Numerical Types ##############################################################
from numbers import Number
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
    return isinstance(obj, Number)
from numbers import Integral
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
    return isinstance(obj, Integral)
from numbers import Real
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
    return isinstance(obj, Real)
from numbers import Complex
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
    return isinstance(obj, Complex)
