####################################################################################################
# pimms/test/lazy_complex.py
#
# This source-code file is part of the pimms library.
#
# The pimms library is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If
# not, see <http://www.gnu.org/licenses/>.

import pimms
import math

@pimms.immutable
class LazyComplex(object):

    def __init__(self, re, im=None):
        if isinstance(re, tuple) and im is None:
            (re, im) = re
        self.re = re
        self.im = im
    
    @pimms.param
    def re(r):
        return float(r)
    @pimms.option(0.0)
    def im(r):
        return float(r) if r is not None else 0.0

    @pimms.value
    def abs(re, im):
        return math.sqrt(re*re + im*im)
    @pimms.value
    def arg(re, im):
        return math.atan2(im, re)

    def __repr__(self):
        return '(%.3f exp(%.3f i))' % (self.abs, self.arg)
