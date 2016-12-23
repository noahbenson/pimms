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

    def test_normal_calc(self):
        from .normal_calc import (normal_distribution, pdf)
        
        # instantiate a normal distribution object
        std_norm_dist = normal_distribution(mean=0.0,
                                            standard_deviation=1.0)
        #>> Checking standard_deviation...
        self.assertEqual(std_norm_dist['mean'], 0)
        self.assertEqual(std_norm_dist['standard_deviation'], 1)

        self.assertEqual(
            set(std_norm_dist.keys()),
            set(['mean', 'standard_deviation', 'variance', 'ci95', 'ci99',
                 'inner_normal_distribution_constant',
                 'outer_normal_distribution_constant']))
        
        self.assertEqual(std_norm_dist['variance'], 1)
        #>> Calculating variance...
        
        # this gets cached after being calculated above
        self.assertTrue('variance' in std_norm_dist.efferents)

        self.assertLess(abs(pdf(std_norm_dist, 1.0) - 0.241971), 0.001)
        #>> Calculating outer constant...
        #>> Calculating inner constant...

        # Make a new normal_distribution object similar to the old
        new_norm_dist_1 = std_norm_dist.set(mean=10.0)
        # the standard_deviation check doesn't get rerun because none of
        # it's parameters have changed
        self.assertTrue('variance' in new_norm_dist_1.efferents)
        self.assertEqual(new_norm_dist_1['variance'], 1)
        
        self.assertLess(abs(pdf(new_norm_dist_1, 9.0) - 0.241971), 0.001)

        # Here, the calculations get rerun because the standard_deviation
        # changes
        new_norm_dist_2 = std_norm_dist.set(standard_deviation=2.0)
        #>> Checking standard_deviation...
        self.assertFalse('variance' in new_norm_dist_2.efferents)
        self.assertEqual(new_norm_dist_2['variance'], 4)
        #>> Calculating variance...

        
if __name__ == '__main__':
    unittest.main()
