# -*- coding: utf-8 -*-
################################################################################
# pimms/calculation/__init__.py
#
# Definition of the calc/plan machinery of pimms.
#
# By Noah C. Benson

from ._core import (to_pathcache, to_lrucache,
                    calc, is_calc,
                    plan, is_plan,
                    plandict, is_plandict)
