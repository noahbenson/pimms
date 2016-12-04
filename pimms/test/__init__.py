####################################################################################################
# pimms/test/__init__.py
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

'''
The test_pimms test package contains tests for the pimms library as well as examples of the library
usage.
'''

import unittest
import math
import pimms

class TestPimms(unittest.TestCase):
    '''
    The TestPimms class defines all the tests for the pimms library.
    '''

    def test_lazy_complex(self):
        '''
        test_lazy_complex makes sure that the example in pimms.test.lazy_complex works.
        '''
        from .lazy_complex import LazyComplex

        z1 = LazyComplex(0)
        z2 = LazyComplex((3.0, 4.0))
        z3 = LazyComplex(0, -1)

        # Test the names:
        self.assertEqual(z1.abs, 0)
        self.assertEqual(z1.arg, 0)
        self.assertEqual(z2.abs, 5.0)
        self.assertLess(abs(z2.arg - 0.9272952180016122), 1e-9)
        self.assertEqual(z3.abs, 1)
        self.assertLess(abs(z3.arg - (-0.5 * math.pi)), 1e-9)

if __name__ == '__main__':
    unittest.main()


