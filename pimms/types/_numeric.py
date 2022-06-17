# -*- coding: utf-8 -*-
################################################################################
# pimms/types/_numeric.py
#
# General numeric type utilities for pimms.
#
# @author Noah C. Benson

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
