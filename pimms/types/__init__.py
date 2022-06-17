# -*- coding: utf-8 -*-
################################################################################
# pimms/types/__init__.py
# Utility classes for the various types that are managed by pimms.
# By Noah C. Benson

"""Type declarations for objects used in Pimms.

The `pimms.types` module contains two kinds of type definitions: basic Python
definitions and Pimms hierarchy definitions. The former represent the basic
Python data types collected in a single namespace while the latter represent
the type hierarchy as Pimms interprets it.
"""

from ._core import (
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
    is_str,
    strnorm,
    strcmp,
    streq,
    strstarts,
    strends,
    is_ureg,
    is_unit,
    is_quant,
    is_callable,
    void_callable,
    is_sized,
    is_container,
    is_iterable,
    is_iterator,
    is_reversible,
    is_coll,
    is_seq,
    is_mseq,
    is_bytestr,
    is_set,
    is_mset,
    is_map,
    is_mmap,
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
    is_hashable,
    is_frozen,
    is_thawed,
    frozenarray,
    freeze,
    thaw,
)

from ._numeric import (
    is_number,
    is_integer,
    is_real,
    is_complex,
    is_array,
    is_tensor,
    is_numeric,
    is_scalar,
    is_vector,
    is_matrix,
)

from ._quant import (
    default_units,
    like_unit,
    alike_units,
    unit,
    quant,
    mag,
    to_quants,
    promote,
)
