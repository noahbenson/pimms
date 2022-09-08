# -*- coding: utf-8 -*-
################################################################################
# pimms/test/doc/test_core.py
#
# Tests of the core documentation system in pimms: i.e., tests for the code in
# the pimms.doc._core module.
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

class TestDocCore(TestCase):
    """Tests the pimms.doc._core module.

    The only public functions in the module are the `make_docproc` function and
    the `docwrap` decorator.
    """
    def test_make_docproc(self):
        from pimms.doc._core import (make_docproc, docproc)
        from docrep import DocstringProcessor
        # make_docproc() takes no arguments.
        new_docproc = make_docproc()
        # It makes a new DocstringProcessor.
        self.assertIsInstance(new_docproc, DocstringProcessor)
        # That DocstringProcessor isn't the same as the original processor.
        self.assertIsNot(docproc, new_docproc)
    def test_docwrap(self):
        from pimms.doc._core import (docwrap, make_docproc)
        # For this test we will use a custom docproc.
        dp = make_docproc()
        # First, make sure we can duplicate parameter documentation.
        @docwrap('fn1', proc=dp)
        def fn1(a, b, c=None):
            """Documentation test function 1.

            This function tests the documentation formatter `@docwrap` of the
            `pimms` library.

            Parameters
            ----------
            a : object
                The first parameter to the function.
            b : str
                The second parameter to the function.
            c : str or None, optional
                The first optional parameter to the function. The default is
                `None`. Must be a string or `None`.

            Returns
            -------
            tuple
                A tuple of `(a, b, c)`.
            """
            return (a,b,c)
        @docwrap('fn2', proc=dp)        
        def fn2(a, b, c=None, d=None):
            """Documentation test function 1.

            This function tests the documentation formatter `@docwrap` of the
            `pimms` library.

            Parameters
            ----------
            %(fn1.parameters.a)s
            %(fn1.parameters.b)s
            %(fn1.parameters.c)s
            d : int or None, optional
                The second optional parameter to the function. The default is
                `None`. Must be an integer or `None`.

            Returns
            -------
            tuple
                A tuple of `(a, b, c, d)`.
            """
            return (a,b,c,d)
        # Make sure the appropriate text made it into the fn2 documentation.
        for s in ["a : object",
                  "The first parameter to the function.",
                  "b : str",
                  "The second parameter to the function.",
                  "c : str or None, optional",
                  "The first optional parameter to the function.",
                  "d : int or None, optional",
                  "The second optional parameter to the function."]:
            self.assertIn(s, fn2.__doc__)
    

