# -*- coding: utf-8 -*-
####################################################################################################
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
    is_ndarray,
    is_str,
    is_ureg,
    is_unit,
    is_quant,
    is_callable,
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
    is_dict,
    is_odict,
    is_ddict,
    is_pvector,
    is_plist,
    is_pset,
    is_pmap,
)

from ._quant import (
    Quantity,
    UnitRegistry,
    units,
    default_units,
    like_unit,
    to_unit,
    unit,
    alike_units,
    like_quant,
    to_quant,
    quant,
    mag,
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
