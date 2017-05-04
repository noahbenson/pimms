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
The pimms.test test package contains tests for the pimms library as well as examples of the library
usage.
'''

import unittest, math, sys, pimms
#from ..immutable import TestPimmsImmutables

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

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

    _test_map_af = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    @staticmethod
    def _make_test_counter_map():
        count = []
        def _counter(arg):
            count.append(arg)
            return arg
        def _make_counter(arg):
            return lambda:_counter(arg)
        return (count, {k: _make_counter(v) for (k,v) in TestPimms._test_map_af.iteritems()})

    def _assertEquivMaps(self, a, b):
        self.assertTrue(isinstance(a, colls.Mapping))
        self.assertTrue(isinstance(b, colls.Mapping))
        self.assertEqual(len(a), len(b))
        for (f,s) in [(a,b), (b,a)]:
            for (k,v) in f.iteritems():
                self.assertTrue(k in s)
                self.assertEqual(v, s[k])
    
    def test_lazy_basics(self):
        # First, test that a lazy map works as a normal map:
        lmap = pimms.lazy_map(TestPimms._test_map_af)
        self._assertEquivMaps(lmap, TestPimms._test_map_af)
        # Now a lazy map
        (c, lmap) = self._make_test_counter_map()
        lmap = pimms.lazy_map(lmap)
        print lmap
        self._assertEquivMaps(lmap, TestPimms._test_map_af)
        self.assertEqual(set(c), set(TestPimms._test_map_af.values()))
        
if __name__ == '__main__':
    unittest.main()
