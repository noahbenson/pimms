# -*- coding: utf-8 -*-
################################################################################
# pimms/test/lazydict/test_core.py
#
# Tests of the core lazydict module in pimms: i.e., tests for the code in the
# pimms.lazydict._core module.
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

# Dependencies #################################################################
from unittest import TestCase

class TestLazyDictCore(TestCase):
    """Tests the pimms.lazydict._core module."""

    # Delays ###################################################################
    def test_delay(self):
        from pimms.lazydict import (delay, DelayError)
        # Delays encapsulate calculations that haven't been run yet--i.e., lazy
        # computations.
        def run():
            run.run_count += 1
            return run.run_count
        run.run_count = 0
        # We can make a delay with the delay constructor.
        d = delay(run)
        # Initially, the delay is not cached.
        self.assertFalse(d.is_cached())
        # To get its value, we can call it.
        self.assertEqual(1, d())
        # Running the computation, incremented the run_count.
        self.assertEqual(1, run.run_count)
        # However, running it again does not--the delay caches the result.
        self.assertEqual(1, d())
        self.assertEqual(1, run.run_count)
        # We can see that the delay is cached now also.
        self.assertTrue(d.is_cached())
        # If a delay raises an error, then it propagates up as a delay error.
        def run_err():
            raise RuntimeError()
        d_err = delay(run_err)
        with self.assertRaises(DelayError): d_err()
        # The delay constructor can accept any number of arguments, which are
        # stored like with a partial function.
        def run_add(a, b, gain=1):
            return (a + b) * gain
        d = delay(run_add, 10, 20, gain=0.5)
        self.assertEqual(15, d())
    def test_is_delay(self):
        from pimms.lazydict import (delay, is_delay)
        # is_delay returns True for any delay object and False for anything
        # else.
        d = delay(lambda:0)
        self.assertTrue(is_delay(d))
        self.assertFalse(is_delay(10))
        self.assertFalse(is_delay('abc'))
        self.assertFalse(is_delay(lambda:0))
    def test_undelay(self):
        from pimms.lazydict import (delay, undelay)
        # undelay is equivalent to calling a delay.
        d = delay(lambda:0)
        self.assertEqual(0, undelay(d))

    # lazydict #################################################################
    def test_lazydict(self):
        from pimms.lazydict import (lazydict, ldict, delay, frozendict)
        # The lazydict type and ldict are aliases.
        self.assertIs(ldict, lazydict)
        # The lazydict type is just like a frozen dictionary except that if any
        # of its values is a delay, the delay is dereferenced when that value's
        # key is requested.
        ld = ldict(a=1, b=2, c=delay(lambda:3))
        self.assertIn('a', ld)
        self.assertIn('b', ld)
        self.assertIn('c', ld)
        self.assertEqual(3, len(ld))
        # lazydicts are instances of frozendict
        self.assertIsInstance(ld, lazydict)
        self.assertIsInstance(ld, frozendict)
        # Weirdly, whether a frozendict is a dict is dependent on frozendict and
        # whether the C or Python source version is installed.
        #self.assertIsInstance(ld, dict)
        # The keys can be tested for their status as lazy or eager.
        self.assertTrue(ld.is_eager('a'))
        self.assertTrue(ld.is_eager('b'))
        self.assertFalse(ld.is_eager('c'))
        self.assertFalse(ld.is_lazy('a'))
        self.assertFalse(ld.is_lazy('b'))
        self.assertTrue(ld.is_lazy('c'))
        # The delay itself can be retrieved using the getdelay method.
        self.assertIsInstance(ld.getdelay('c'), delay)
        # If the delay for an eager key is requested, the return value is None.
        self.assertIs(ld.getdelay('a'), None)
        # When the values are requested, the delays are undelayed automatically.
        self.assertEqual(1, ld['a'])
        self.assertEqual(2, ld['b'])
        self.assertEqual(3, ld['c'])
        # After the value has been undelayed, we can see that the key is no
        # longer lazy.
        self.assertTrue(ld.is_eager('c'))
        self.assertFalse(ld.is_lazy('c'))
    def test_is_lazydict(self):
        from pimms.lazydict import (is_lazydict, is_ldict, ldict, fdict)
        # is_lazydict and is_ldict are aliases.
        self.assertIs(is_lazydict, is_ldict)
        # is_ldict tells you when the argument is a lazydict.
        self.assertTrue(is_lazydict(ldict()))
        self.assertFalse(is_lazydict(fdict()))
        self.assertFalse(is_lazydict(dict()))
    def test_lazyvalmap(self):
        from pimms.lazydict import (lazyvalmap, ldict, fdict)
        # lazyvalmap can update the keys to a dictionary like the map function.
        # lazyvalmap(f, d) is similar to {k:f(v) for (k,v) in d.items()}.
        d = dict(a=1, b=2, c=3)
        d_tx = lazyvalmap(lambda v: v**2, d)
        # lazyvalmap always returns a lazydict.
        self.assertIsInstance(d_tx, ldict)
        # The values will initially be uncached.
        self.assertTrue(d_tx.is_lazy('a'))
        self.assertTrue(d_tx.is_lazy('b'))
        self.assertTrue(d_tx.is_lazy('c'))
        # They will have the values requested by our lambda function, above.
        self.assertEqual(1, d_tx['a'])
        self.assertEqual(4, d_tx['b'])
        self.assertEqual(9, d_tx['c'])
    def test_valmap(self):
        from pimms.lazydict import (valmap, ldict, fdict)
        # valmap can update the keys to a dictionary like the map function.
        # valmap(f, d) is similar to {k:f(v) for (k,v) in d.items()}.
        d = dict(a=1, b=2, c=3)
        d_tx = valmap(lambda v: v**2, d)
        # valmap always returns a dict, frozendict, or lazydict, depending on
        # the type of the dictionary given.
        self.assertIsInstance(d_tx, dict)
        self.assertIsInstance(valmap(lambda v: v**2, fdict(d)), fdict)
        self.assertIsInstance(valmap(lambda v: v**2, ldict(d)), ldict)
        # The values will be those requested by our lambda function, above.
        self.assertEqual(1, d_tx['a'])
        self.assertEqual(4, d_tx['b'])
        self.assertEqual(9, d_tx['c'])
    def test_lazykeymap(self):
        from pimms.lazydict import (lazykeymap, ldict, fdict)
        # lazykeymap can update the keys to a dictionary like the map function.
        # lazykeymap(f, d) is similar to {k:f(k) for (k,v) in d.items()}.
        d = dict(a=1, b=2, c=3)
        d_tx = lazykeymap(lambda k: f'<{k}>', d)
        # lazykeymap always returns a lazydict.
        self.assertIsInstance(d_tx, ldict)
        # The values will initially be uncached.
        self.assertTrue(d_tx.is_lazy('a'))
        self.assertTrue(d_tx.is_lazy('b'))
        self.assertTrue(d_tx.is_lazy('c'))
        # They will have the values requested by our lambda function, above.
        self.assertEqual('<a>', d_tx['a'])
        self.assertEqual('<b>', d_tx['b'])
        self.assertEqual('<c>', d_tx['c'])
    def test_keymap(self):
        from pimms.lazydict import (keymap, ldict, fdict)
        # keymap can update the keys to a dictionary like the map function.
        # keymap(f, d) is similar to {k:f(v) for (k,v) in d.items()}.
        d = dict(a=1, b=2, c=3)
        d_tx = keymap(lambda k: f'<{k}>', d)
        # keymap always returns a dict, frozendict, or lazydict, depending on
        # the type of the dictionary given.
        self.assertIsInstance(d_tx, dict)
        self.assertIsInstance(keymap(lambda k: f'<{k}>', fdict(d)), fdict)
        self.assertIsInstance(keymap(lambda k: f'<{k}>', ldict(d)), ldict)
        # The values will be those requested by our lambda function, above.
        self.assertEqual('<a>', d_tx['a'])
        self.assertEqual('<b>', d_tx['b'])
        self.assertEqual('<c>', d_tx['c'])
    def test_lazyitemmap(self):
        from pimms.lazydict import (lazyitemmap, ldict, fdict)
        # lazyitemmap can update the keys to a dictionary like the map function.
        # lazyitemmap(f, d) is similar to {k:f(k,v) for (k,v) in d.items()}.
        d = dict(a=1, b=2, c=3)
        tx_fn = lambda k,v: f'{k}: {v**2}'
        d_tx = lazyitemmap(tx_fn, d)
        # lazyitemmap always returns a lazydict.
        self.assertIsInstance(d_tx, ldict)
        # The values will initially be uncached.
        self.assertTrue(d_tx.is_lazy('a'))
        self.assertTrue(d_tx.is_lazy('b'))
        self.assertTrue(d_tx.is_lazy('c'))
        # They will have the values requested by our lambda function, above.
        self.assertEqual('a: 1', d_tx['a'])
        self.assertEqual('b: 4', d_tx['b'])
        self.assertEqual('c: 9', d_tx['c'])
    def test_itemmap(self):
        from pimms.lazydict import (itemmap, ldict, fdict)
        # itemmap can update the keys to a dictionary like the map function.
        # itemmap(f, d) is similar to {k:f(v) for (k,v) in d.items()}.
        d = dict(a=1, b=2, c=3)
        tx_fn = lambda k,v: f'{k}: {v**2}'
        d_tx = itemmap(tx_fn, d)
        # itemmap always returns a dict, frozendict, or lazydict, depending on
        # the type of the dictionary given.
        self.assertIsInstance(d_tx, dict)
        self.assertIsInstance(itemmap(tx_fn, fdict(d)), fdict)
        self.assertIsInstance(itemmap(tx_fn, ldict(d)), ldict)
        # The values will be those requested by our lambda function, above.
        self.assertEqual('a: 1', d_tx['a'])
        self.assertEqual('b: 4', d_tx['b'])
        self.assertEqual('c: 9', d_tx['c'])
    def test_dictmap(self):
        from pimms.lazydict import (dictmap, ldict, fdict)
        # dictmap creates a dict from a function and keys.
        fn = lambda k: f'<{k}>'
        d = dictmap(fn, ['a', 'b', 'c'])
        # The new dictionary has the keys requested from the second argument.
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertIn('c', d)
        # And these keys map to fn(k), each.
        self.assertEqual('<a>', d['a'])
        self.assertEqual('<b>', d['b'])
        self.assertEqual('<c>', d['c'])
        # The return type of a dictmap is a dict.
        self.assertIs(type(d), dict)
    def test_frozendictmap(self):
        from pimms.lazydict import (frozendictmap, fdictmap, ldict, fdict)
        # frozendictmap and fdictmap are aliases.
        self.assertIs(frozendictmap, fdictmap)
        # frozendictmap creates a dict from a function and keys.
        fn = lambda k: f'<{k}>'
        d = frozendictmap(fn, ['a', 'b', 'c'])
        # The new dictionary has the keys requested from the second argument.
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertIn('c', d)
        # And these keys map to fn(k), each.
        self.assertEqual('<a>', d['a'])
        self.assertEqual('<b>', d['b'])
        self.assertEqual('<c>', d['c'])
        # The return type of a frozendictmap is a frozendict.
        self.assertIs(type(d), fdict)
    def test_lazydictmap(self):
        from pimms.lazydict import (lazydictmap, ldictmap, ldict, fdict)
        # lazydictmap and ldictmap are aliases.
        self.assertIs(lazydictmap, ldictmap)
        # lazydictmap creates a dict from a function and keys.
        fn = lambda k: f'<{k}>'
        d = lazydictmap(fn, ['a', 'b', 'c'])
        # The new dictionary has the keys requested from the second argument.
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertIn('c', d)
        # The return type of a lazydictmap is a lazydict.
        self.assertIs(type(d), ldict)
        # The keys are initially lazy.
        self.assertTrue(d.is_lazy('a'))
        self.assertTrue(d.is_lazy('b'))
        self.assertTrue(d.is_lazy('c'))
        # And these keys map to fn(k), each.
        self.assertEqual('<a>', d['a'])
        self.assertEqual('<b>', d['b'])
        self.assertEqual('<c>', d['c'])
    def test_merge(self):
        from pimms.lazydict import (merge, ldict, fdict, delay)
        # merge joins dictionaries with keys to the right in the argument list
        # overwriting those to the left.
        d = merge(dict(a=1, b=2, c=3),
                  dict(b=10, c=20, d=30),
                  dict(c=100, d=200, e=300))
        self.assertEqual(len(d), 5)
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertIn('c', d)
        self.assertIn('d', d)
        self.assertIn('e', d)
        self.assertEqual(1, d['a'])
        self.assertEqual(10, d['b'])
        self.assertEqual(100, d['c'])
        self.assertEqual(200, d['d'])
        self.assertEqual(300, d['e'])
        # merge returns a frozendict if none of the arguments are lazy.
        self.assertIsInstance(d, fdict)
        # If the arguments are lazy, then the merge respects the laziness of the
        # key whose value is used.
        d = merge(dict(a=1, b=2, c=3),
                  ldict(b=delay(lambda:10), c=20, d=30),
                  dict(c=100, d=200, e=300))
        self.assertIsInstance(d, ldict)
        self.assertTrue(d.is_lazy('b'))
        self.assertEqual(len(d), 5)
        self.assertEqual(10, d['b'])
    def test_rmerge(self):
        from pimms.lazydict import (rmerge, ldict, fdict, delay)
        # rmerge joins dictionaries with keys to the left in the argument list
        # overwriting those to the right.
        d = rmerge(dict(a=1, b=2, c=3),
                   dict(b=10, c=20, d=30),
                   dict(c=100, d=200, e=300))
        self.assertEqual(len(d), 5)
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertIn('c', d)
        self.assertIn('d', d)
        self.assertIn('e', d)
        self.assertEqual(1, d['a'])
        self.assertEqual(2, d['b'])
        self.assertEqual(3, d['c'])
        self.assertEqual(30, d['d'])
        self.assertEqual(300, d['e'])
        # rmerge returns a frozendict if none of the arguments are lazy.
        self.assertIsInstance(d, fdict)
        # If the arguments are lazy, then the rmerge respects the laziness of
        # the key whose value is used.
        d = rmerge(dict(a=1, b=2, c=3),
                  ldict(b=10, c=20, d=delay(lambda:30)),
                  dict(c=100, d=200, e=300))
        self.assertIsInstance(d, ldict)
        self.assertTrue(d.is_lazy('d'))
        self.assertEqual(len(d), 5)
        self.assertEqual(30, d['d'])
    def test_assoc(self):
        from pimms.lazydict import (assoc, ldict, fdict)
        # assoc updates a dictionary with a new key-value (or many) and returns
        # a copy (it never changes the argument).
        d = dict(a=1, b=2, c=3)
        d1 = assoc(d, a=10)
        d2 = assoc(d1, 'd', 4)
        self.assertIsNot(d, d1)
        self.assertIsNot(d1, d2)
        self.assertEqual(1, d['a'])
        self.assertEqual(3, len(d))
        self.assertEqual(10, d1['a'])
        self.assertEqual(4, d2['d'])
        self.assertEqual(3, len(d1))
        self.assertEqual(4, len(d2))
        # The type of dictionary is preserved.
        d = fdict(a=1, b=2, c=3)
        d1 = assoc(d, 'a', 10, 'd', 4)
        self.assertEqual(len(d1), 4)
        self.assertEqual(10, d1['a'])
        self.assertEqual(4, d1['d'])
        self.assertIsInstance(d1, fdict)
    def test_dissoc(self):
        from pimms.lazydict import (dissoc, ldict, fdict)
        # assoc updates a dictionary to no longer have a key (or many keys) and
        # returns a copy (it never changes the argument).
        d = dict(a=1, b=2, c=3)
        d1 = dissoc(d, 'a')
        d2 = dissoc(d1, 'b')
        self.assertIsNot(d, d1)
        self.assertIsNot(d1, d2)
        self.assertEqual(1, d['a'])
        self.assertEqual(3, len(d))
        self.assertEqual(2, len(d1))
        self.assertNotIn('a', d1)
        self.assertNotIn('b', d2)
        self.assertEqual(1, len(d2))
        # The type of dictionary is preserved.
        d = fdict(a=1, b=2, c=3)
        d1 = dissoc(d, 'a', 'b')
        self.assertEqual(len(d1), 1)
        self.assertIsInstance(d1, fdict)
    def test_lambdadict(self):
        from pimms import (lambdadict, ldict)
        # A lambdadict is a dictionary whose lambda functions all have arguments
        # whose names are keys in the lambdadict itself, and when they are
        # requested, they lazily calculate their values.
        d = lambdadict(data=[0.1, 0.3, 0.5, 0.5, 0.9],
                       sum=lambda data: sum(data),
                       len=lambda data: len(data),
                       mean=lambda sum, len: sum / len)
        self.assertEqual(len(d), 4)
        self.assertIn('data', d)
        self.assertIn('sum', d)
        self.assertIn('len', d)
        self.assertIn('mean', d)
        self.assertEqual(d['sum'], 2.3)
        self.assertEqual(d['len'], 5)
        self.assertEqual(d['mean'], 2.3/5)
        # Lambdadicts are lazydicts.
        self.assertIsInstance(d, ldict)
