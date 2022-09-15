# -*- coding: utf-8 -*-
################################################################################
# pimms/util/__init__.py
#
# General utilities managed by pimms.
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

"""Utilites managed by pimms.

The `pimms.util` module contains two kinds of type definitions: basic Python
definitions and Pimms hierarchy definitions. The former represent the basic
Python data types collected in a single namespace while the latter represent
the type hierarchy as Pimms interprets it.
"""

from ._core import (
    is_str,
    strnorm,
    strcmp,
    streq,
    strstarts,
    strends,
    strissym,
    striskey,
    strisvar,
    is_ureg,
    is_unit,
    is_quant,
    is_callable,
    is_lambda,
    is_sized,
    is_container,
    is_iterable,
    is_iterator,
    is_reversible,
    is_coll,
    is_seq,
    is_mutseq,
    is_bytestr,
    is_set,
    is_mutset,
    is_map,
    is_mutmap,
    is_hashable,
    is_tuple,
    is_list,
    is_pyset,
    is_frozenset,
    is_fset,
    is_dict,
    is_odict,
    is_ddict,
    frozendict,
    is_frozendict,
    is_fdict,
    is_fodict,
    hashsafe,
    can_hash,
    itersafe,
    can_iter,
    is_frozen,
    is_thawed,
    to_frozenarray,
    to_farray,
    frozenarray,
    farray,
    freeze,
    thaw,
)

from ._numeric import (
    is_number,
    is_integer,
    is_real,
    is_complex,
    is_array,
    to_array,
    is_tensor,
    to_tensor,
    is_numeric,
    to_numeric,
    is_sparse,
    to_sparse,
    is_dense,
    to_dense,
    is_numpydtype,
    like_numpydtype,
    to_numpydtype,
    is_torchdtype,
    like_torchdtype,
    to_torchdtype,
)

from ._quantity import (
    default_ureg,
    like_unit,
    alike_units,
    unit,
    quant,
    mag,
    promote,
)

__all__ = (
    "is_array",
    "to_array",
    "is_tensor",
    "to_tensor",
    "is_numeric",
    "to_numeric",
    "is_sparse",
    "to_sparse",
    "is_dense",
    "to_dense",
    "is_str",
    "strnorm",
    "strcmp",
    "streq",
    "strstarts",
    "strends",
    "strissym",
    "striskey",
    "strisvar",
    "is_ureg",
    "is_unit",
    "is_quant",
    #"is_numpydtype",
    #"like_numpydtype",
    #"to_numpydtype",
    #"is_torchdtype",
    #"like_torchdtype",
    #"to_torchdtype",
    #"is_callable",
    "is_lambda",
    "is_sized",
    "is_container",
    "is_iterable",
    "is_iterator",
    "is_reversible",
    "is_coll",
    "is_seq",
    "is_mutseq",
    "is_bytestr",
    "is_set",
    "is_mutset",
    "is_map",
    "is_mutmap",
    "is_hashable",
    "is_tuple",
    "is_list",
    "is_pyset",
    "is_frozenset",
    "is_fset",
    "is_dict",
    "is_odict",
    "is_ddict",
    "frozendict",
    "is_frozendict",
    "is_fdict",
    "is_fodict",
    "hashsafe",
    "is_hashable",
    "can_hash",
    "itersafe",
    "can_iter",
    "is_frozen",
    "is_thawed",
    "to_frozenarray",
    "to_farray",
    "frozenarray",
    "farray",
    "freeze",
    "thaw",
    "is_number",
    "is_integer",
    "is_real",
    "is_complex",
    "default_ureg",
    "like_unit",
    "alike_units",
    "unit",
    "quant",
    "mag",
    "promote",
)
