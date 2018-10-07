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

import unittest, math, sys, six, pimms
import numpy as np
import pyrsistent as pyr

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

class TestPimms(unittest.TestCase):
    '''
    The TestPimms class defines all the tests for the pimms library.
    '''

    def test_predicates(self):
        '''
        test_predicates ensures that the various pimms functions of the form is_<type>(obj) are
        working properly.
        '''
        # some arbitrary values to use in testing
        qr = pimms.quant([1.5, 3.3, 9.5, 10.4, 6.2, 0.1], 'mm')
        qi = pimms.quant([1, 3, 9, 10, 6, 0], 'seconds')
        mr = np.random.rand(10,3)
        vi = np.random.randint(0, 5000, size=12)
        sr = np.array(10.0)
        si = np.array(2)*pimms.units.deg
        l = [1, 2.0, 'abc']
        lm = [[1,1], [2.0,2.0], [4,7.7]]
        u = u'a unicode string of stuff'
        b = b'a byte string of stuff'
        f0 = lambda:np.linspace(0,100,117)
        f1 = lambda x:x**2 + 1
        d = {'a': 12, 'b':None, 'c':f0, 'd':f1,
             'e':lambda:'some string', 'f':lambda:None}
        pm = pyr.pmap(d)
        lm = pimms.lazy_map(d)
        # a function for testing predicates
        def tpred(p, tvals, fvals):
            for s in tvals: self.assertTrue(p(s))
            for s in fvals: self.assertFalse(p(s))
        # Map types
        tpred(pimms.is_lazy_map, [lm], [qr,qi,mr,vi,sr,si,l,u,b,f0,f1,d,pm])
        tpred(pimms.is_map, [lm,d,pm], [qr,qi,mr,vi,sr,si,l,u,b,f0,f1])
        tpred(pimms.is_pmap, [lm,pm], [qr,qi,mr,vi,sr,si,l,u,b,f0,f1,d])
        # Numpy types require a little attention due to their optional arguments and the
        # complexities of the type relationships
        tpred(pimms.is_nparray, [qr,qi,mr,vi,sr], [l,lm,u,b,f0,f1,d,pm,lm])
        self.assertTrue(pimms.is_nparray(qr, 'real'))
        self.assertTrue(pimms.is_nparray(qr, 'any', 1))
        self.assertFalse(pimms.is_nparray(qr, 'any', 2))
        self.assertFalse(pimms.is_nparray(qi, 'string'))
        self.assertTrue(pimms.is_nparray(qi, ('real','int'), (1,3)))
        self.assertFalse(pimms.is_nparray(qi, ('real','int'), 2))
        self.assertFalse(pimms.is_nparray(qr, ('string','bool','bytes'), (2,3)))
        self.assertTrue(pimms.is_nparray(mr, None, 2))
        self.assertFalse(pimms.is_nparray(mr, None, 1))
        self.assertFalse(pimms.is_nparray(vi, 'int', 2))
        tpred(pimms.is_npscalar, [sr], [qr,qi,mr,vi,l,lm,u,b,f0,f1,d,pm,lm])
        self.assertTrue(pimms.is_npscalar(np.array(12.0), 'real'))
        self.assertFalse(pimms.is_npscalar(np.array(12.0), 'string'))
        self.assertTrue(pimms.is_npscalar(np.array(12.0), ('real','complex')))
        tpred(pimms.is_npmatrix, [mr, pimms.quant(mr, 'm/s')],
              [sr,si,qr,qi,vi,l,lm,u,b,f0,f1,d,pm,lm])
        self.assertTrue(pimms.is_npmatrix(mr, ('int','real','string')))
        self.assertTrue(pimms.is_npmatrix(mr, 'number'))
        self.assertFalse(pimms.is_npmatrix(mr, ('bool','string')))
        tpred(pimms.is_npvector, [qr,qi,vi,vi*pimms.units.mol,qr,qi],
              [sr,si,mr,l,lm,u,b,f0,f1,d,pm,lm])
        self.assertTrue(pimms.is_npvector(vi, 'real'))
        self.assertTrue(pimms.is_npvector(qi, 'int'))
        self.assertFalse(pimms.is_npvector(qr, ('bool','string')))
        self.assertTrue(pimms.is_npvalue('abc', 'string'))
        self.assertTrue(pimms.is_npvalue(u'abc', ('unicode','real')))
        self.assertFalse(pimms.is_npvalue(np.array(5.6), ('unicode','real')))
        self.assertFalse(pimms.is_npvalue(np.array(5.6), ('unicode','bool')))
        self.assertFalse(pimms.is_npvalue(np.array([5.6]), ('unicode','real')))
        # Also the un-nump'ified versions
        tpred(pimms.is_array, [qr,qi,vi,sr,si,mr,qr,qi,l,lm,u,b,f0,f1,d,pm,lm], [])
        self.assertTrue(pimms.is_array(qr, 'real'))
        self.assertTrue(pimms.is_array(qr, 'any', 1))
        self.assertTrue(pimms.is_array(qr, 'any', 1))
        self.assertFalse(pimms.is_array(qi, 'string'))
        self.assertTrue(pimms.is_array(qi, ('real','int'), (1,3)))
        self.assertFalse(pimms.is_array(qi, ('real','int'), 2))
        self.assertFalse(pimms.is_array(qr, ('string','bool','bytes'), (2,3)))
        self.assertTrue(pimms.is_array(mr, None, 2))
        self.assertFalse(pimms.is_array(mr, None, 1))
        self.assertFalse(pimms.is_array(vi, 'int', 2))
        self.assertFalse(pimms.is_array(l, 'number', 1))
        self.assertTrue(pimms.is_array(lm, 'any', (1,2)))
        tpred(pimms.is_scalar, [u,b,f0,f1,d,sr,si], [qr,qi,vi,mr,l,pm,lm])
        tpred(pimms.is_int, [vi[0],si,1,10], [u,b,f0,f1,d,pm,lm,sr,qr,mr])
        tpred(pimms.is_real, [vi[0],si,1,10,sr], [4j,u,b,f0,f1,d,pm,lm,mr,qr])
        tpred(pimms.is_complex, [vi[0],si,1,10,sr,4j], [u,b,f0,f1,d,pm,lm,mr,qr])
        tpred(pimms.is_number, [vi[0],si,1,10,sr], [u,b,f0,f1,d,pm,lm,mr,qr])
        tpred(pimms.is_str, ['abc'], [vi,si,1,10,sr,qr,f0,f1,d,pm,lm,mr])
        tpred(pimms.is_class, [str,int], [vi,si,1,10,sr,qr,u,b,f0,f1,d,pm,lm,mr])
        tpred(pimms.is_quantity, [qr,qi,si], [vi,10,sr,u,b,f0,f1,d,pm,lm,mr])
        tpred(pimms.is_unit,
              ('seconds', 's', 'mm', 'deg', pimms.units.seconds, pimms.units.s, pimms.units.mm,
               pimms.units.deg),
              (1, 10.0, np.asarray([10]), None, 'nonunitstring', qr))
        self.assertTrue(pimms.is_nparray(mr, np.inexact))
        self.assertFalse(pimms.is_nparray(vi, np.inexact))
    def test_units(self):
        '''
        test_units ensures that the various pimms functions related to pint integration work
        correctly; these functions include pimms.unit, .mag, .quant, .is_quantity, etc.
        '''
        # make a few pieces of data with types
        x = np.asarray([1.0, 2.0, 3.0, 4.0]) * pimms.units.mm
        y = pimms.quant([2, 4, 6, 8], 'sec')
        for u in [x,y]: self.assertTrue(pimms.is_quantity(u))
        for u in ('abc', 123, 9.0, []): self.assertFalse(pimms.is_quantity(u))
        for u in [x,y]: self.assertFalse(pimms.is_quantity(pimms.mag(u)))
        self.assertTrue(pimms.like_units(pimms.unit(x), pimms.unit('yards')))
        self.assertTrue(pimms.like_units(pimms.unit(y), pimms.unit('minutes')))
        self.assertFalse(pimms.like_units(pimms.unit(y), pimms.unit('mm')))
        z = x / y
        self.assertTrue(pimms.is_vector(x, 'real'))
        self.assertTrue(pimms.is_vector(y, 'real'))
        self.assertFalse(pimms.is_vector(x, 'int'))
        self.assertTrue(pimms.is_vector(y, 'int'))
        self.assertTrue(pimms.is_vector(y, 'float'))
        self.assertTrue(pimms.is_vector(z, 'real'))
        
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
        self.assertTrue(z1.is_lazy_complex)
        
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
        return (count, {k: _make_counter(v) for (k,v) in six.iteritems(TestPimms._test_map_af)})

    def _assertEquivMaps(self, a, b):
        self.assertTrue(isinstance(a, colls.Mapping))
        self.assertTrue(isinstance(b, colls.Mapping))
        self.assertEqual(len(a), len(b))
        for (f,s) in [(a,b), (b,a)]:
            for (k,v) in six.iteritems(f):
                self.assertTrue(k in s)
                self.assertEqual(v, s[k])
    
    def test_lazy_basics(self):
        # First, test that a lazy map works as a normal map:
        lmap = pimms.lazy_map(TestPimms._test_map_af)
        self._assertEquivMaps(lmap, TestPimms._test_map_af)
        # Now a lazy map
        (c, lmap) = self._make_test_counter_map()
        lmap = pimms.lazy_map(lmap)
        self._assertEquivMaps(lmap, TestPimms._test_map_af)
        self.assertEqual(set(c), set(TestPimms._test_map_af.values()))

    ################################################################################################
    # Test the pimms utilities
    def test_lazy_map(self):
        # We setup a map-testing function that takes a few args and asserts everything is as it
        # should be:
        def _test_lm(lm, m, lazy_ks, norm_ks):
            self.assertTrue(isinstance(lm, pimms.LazyPMap))
            self.assertEqual(len(lm), len(m))
            for k in six.iterkeys(m): self.assertTrue(k in lm)
            # the normal keys should just be normal:
            for k in norm_ks:
                self.assertEqual(m[k], lm[k])
                self.assertTrue(lm.is_normal(k))
                self.assertFalse(lm.is_lazy(k))
                self.assertFalse(lm.is_memoized(k))
            # these tests should not memoize any keys:
            for k in six.iterkeys(lm):
                self.assertTrue(k in m and k in lm)
            for k in lm.iternormal():
                self.assertTrue(k in m and k in norm_ks)
                self.assertEqual(lm[k], m[k])
            self.assertEqual(len(set(lm.itermemoized())), 0)
            self.assertEqual(len(set(lm.iterlazy())), len(lazy_ks))
            # now check the lazy keys (should still all be lazy):
            for k in lazy_ks:
                self.assertFalse(lm.is_normal(k))
                self.assertTrue(lm.is_lazy(k))
                self.assertFalse(lm.is_memoized(k))
                self.assertEqual(m[k], lm[k]) # here the memoization should happen
                self.assertTrue(lm.is_memoized(k))
                self.assertFalse(lm.is_lazy(k))
                self.assertFalse(lm.is_normal(k))
            # everything should be memoized now:
            self.assertEqual(len(set(lm.itermemoized())), len(lazy_ks))
            self.assertEqual(len(set(lm.iterlazy())), 0)
            # it should be an error to delete anything or set anything
            for k in (lm.keys() + ['random_key1', 'random_key2']):
                with self.assertRaises(TypeError): lm[k] = 10
                with self.assertRaises(TypeError): del lm[k]
        # Okay, we can test a few lazy maps now:
        # (1) lazy maps with normal values are just normal maps:
        m = {k:v for (v,k) in enumerate(map(chr, range(ord('a'), ord('z')+1)))}
        lm = pimms.lazy_map(m)
        _test_lm(lm, m, [], m.keys())
        # (2) lazy maps should be build-able...
        norm_ks = set(m.keys())
        lazy_ks = set([])
        d_ord = ord('a') - ord('A')
        def _make_lazy_lambda(v):
            def _fn():
                _make_lazy_lambda.counter += 1
                return v
            return _fn
        _make_lazy_lambda.counter = 0
        for (v,k) in enumerate(map(chr, range(ord('A'), ord('Z') + 1))):
            m[k] = v
            lm = lm.set(k, _make_lazy_lambda(v))
            lazy_ks.add(k)
            if ord(k) % 2 == 0:
                kl = chr(ord(k) + d_ord)
                lm = lm.remove(kl)
                del m[kl]
                norm_ks.remove(kl)
        _test_lm(lm, m, lazy_ks, norm_ks)
        self.assertEqual(_make_lazy_lambda.counter, len(lazy_ks))
        remem = [v for (k,v) in six.iteritems(lm)]
        self.assertEqual(_make_lazy_lambda.counter, len(lazy_ks))

    def test_itable(self):
        '''
        test_itable() tests pimms itable objects and makes sure they work correctly.
        '''
        class nloc:
            lazy_loads = 0
        def _load_lazy():
            nloc.lazy_loads += 1
            return pimms.quant(np.random.rand(10), 'sec')
        dat = pimms.lazy_map({'a': [1,2,3,4,5,6,7,8,9,10],
                              'b': pimms.quant(np.random.rand(10), 'mm'),
                              'c': ['abc','def','ghi','jkl','mno','pqr','stu','vwx','yz!','!!!'],
                              'd': _load_lazy})
        tbl = pimms.itable(dat)
        # make sure the data is the right size
        for k in tbl.keys():
            self.assertTrue(tbl[k].shape == (10,))
        self.assertTrue(tbl.row_count == 10)
        self.assertTrue(len(tbl.rows) == 10)
        self.assertTrue(len(tbl.column_names) == 4)
        self.assertTrue(len(tbl.columns) == 4)
        # Check a few of the entries
        for (i,ki) in zip(np.random.randint(0, tbl.row_count, 50), np.random.randint(0, 4, 50)):
            ki = tbl.column_names[ki]
            self.assertTrue(tbl.rows[i][ki] == tbl[ki][i])
        self.assertTrue(nloc.lazy_loads == 1)
        # see if we can discard stuff
        self.assertTrue('a' in tbl)
        self.assertFalse('a' in tbl.discard('a'))
        self.assertFalse('b' in tbl.discard('b'))
        self.assertFalse('c' in tbl.discard('b').discard('c'))
        self.assertTrue('c' in tbl.discard('b').discard('a'))

    def test_persist(self):
        '''
        test_persist() tests pimms persist() function.
        '''
        from .lazy_complex import LazyComplex

        z = LazyComplex((1.0, 2.0))
        self.assertFalse(z.is_persistent())
        self.assertTrue(z.is_transient())
        z.persist()
        self.assertTrue(z.is_persistent())
        self.assertFalse(z.is_transient())
        
        z = LazyComplex((1.0, 2.0))
        self.assertFalse(z.is_persistent())
        self.assertTrue(z.is_transient())
        zp = pimms.persist(z)
        self.assertTrue(zp.is_persistent())
        self.assertFalse(zp.is_transient())
        self.assertFalse(z.is_persistent())
        self.assertTrue(z.is_transient())

        m0 = {'a': [1,2,3], 'b': (2,3,4),
              'c': {'d':'abc', 'e':set(['def','ghi']), 'f':frozenset([10,11,12])},
              'z': z, 'zp': zp,
              'q': (1,2,[3,4]),
              't': pimms.itable({'c1': range(10), 'c2': range(1,11), 'c3': range(2,12)})}
        m = pimms.persist(m0)
        self.assertIs(m['b'], m0['b'])
        self.assertIsNot(m['a'], m0['a'])
        self.assertTrue(all(ai == bi for (ai,bi) in zip(m['a'], m0['a'])))
        self.assertTrue(pimms.is_pmap(m['c']))
        self.assertIs(m['c']['d'], m0['c']['d'])
        self.assertTrue(isinstance(m['c']['e'], frozenset))
        self.assertTrue(isinstance(m['c']['f'], frozenset))
        self.assertTrue(all(ai == bi for (ai,bi) in zip(m['c']['f'], m0['c']['f'])))
        self.assertTrue(m['z'].is_persistent())
        self.assertIs(m['zp'], m0['zp'])
        self.assertIs(m['q'], m0['q'])
        self.assertIs(m['q'][2], m0['q'][2])
        self.assertTrue(pimms.is_itable(m['t']))
        self.assertTrue(m['t'].is_persistent())
        m = pimms.persist(m0, depth=1)
        self.assertIs(m['b'], m0['b'])
        self.assertIsNot(m['a'], m0['a'])
        self.assertTrue(all(ai == bi for (ai,bi) in zip(m['a'], m0['a'])))
        self.assertTrue(pimms.is_pmap(m['c']))
        self.assertIs(m['c']['d'], m0['c']['d'])
        self.assertTrue(isinstance(m['c']['e'], set))
        self.assertTrue(isinstance(m['c']['f'], frozenset))
        self.assertTrue(all(ai == bi for (ai,bi) in zip(m['c']['f'], m0['c']['f'])))
        self.assertTrue(m['z'].is_persistent())
        self.assertIs(m['zp'], m0['zp'])
        self.assertIs(m['q'], m0['q'])
        self.assertIs(m['q'][2], m0['q'][2])
        self.assertTrue(pimms.is_itable(m['t']))
        self.assertTrue(m['t'].is_persistent())

    def test_cmdline(self):
        '''
        test_cmdline() ensures that the command-line functions such as argv_parse() are working
          correctly and work correctly with calculation plans.
        '''
        from .normal_calc import normal_distribution

        # command-line parsing works with all kinds of string literals:
        schema = [('a', 'arg-a', 'a',  0),
                  ('b', 'arg-b', 'b', ''),
                  ('c', 'arg-c', 'c', {}),
                  ('d', 'arg-d', 'd', []),
                  ('e', 'arg-e', 'e', True),
                  ('f', 'arg-f', 'f', False),
                  ('g', 'arg-g', 'g', Ellipsis),
                  ('h', 'arg-h', 'h', 'test')]
        a = pimms.argv_parse(schema,
                             ['-a1.5', '--arg-b=string arg', '-c', '{10:"str"}', '-d[1,2,3]',
                              '--arg-e', '-g...'])
        (leftover, a) = a
        self.assertEqual(a['a'], 1.5)
        self.assertEqual(a['b'], 'string arg')
        self.assertEqual(a['c'], {10:'str'})
        self.assertEqual(a['d'], [1,2,3])
        self.assertEqual(a['e'], False)
        self.assertEqual(a['f'], False)
        self.assertEqual(a['g'], Ellipsis)
        self.assertEqual(a['h'], 'test')

        # test use with plans:
        dflts = {'mean':0.0, 'standard_deviation':1.0}
        a = pimms.argv_parse(normal_distribution,
                             ['-m4.5', '--standard-deviation=6.3', 'leftover'],
                             defaults=dflts)
        self.assertTrue(pimms.is_imap(a))
        self.assertEqual(a['argv'], ('leftover',))
        self.assertEqual(a['argv_parsed']['mean'], 4.5)
        self.assertEqual(a['mean'], 4.5)
        self.assertEqual(a['standard_deviation'], 6.3)
        a = pimms.argv_parse(normal_distribution,
                             ['--standard-deviation=6.3', 'leftover'],
                             defaults=dflts)
        self.assertTrue(pimms.is_imap(a))
        self.assertEqual(a['argv'], ('leftover',))
        self.assertEqual(a['argv_parsed']['mean'], 0.0)
        self.assertEqual(a['mean'], 0.0)
        self.assertEqual(a['standard_deviation'], 6.3)

if __name__ == '__main__':
    unittest.main()
