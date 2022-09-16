# -*- coding: utf-8 -*-
################################################################################
# pimms/test/workflow/test_core.py
#
# Tests of the core workflow module in pimms: i.e., tests for the code in the
# pimms.workflow._core module.
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

class TestWorkflowCore(TestCase):
    """Tests the pimms.workflow._core module."""
    def test_calc(self):
        from pimms.workflow import calc
        # The calc decorator creates calculation objects.
        @calc
        def result(input_1, input_2=None):
            """Calculation for a result from input_1 and input_2.

            Returns a single value, `'result'`, which is a list whose length is
            `input_1` and whose elements are all `input_2`.
            
            Inputs
            ------
            input_1 : int
                The number of elements to include in the result.
            input_2 : object
                The object to put in the list.

            Outputs
            -------
            result : list
                A list of `input_1` occurrences of `input_2`.
            """
            return ([input_2] * input_1,)
        self.assertIsInstance(result, calc)
        # Calculation objects have a number of members that keep track of the
        # meta-data of the calculation.
        # Firs is the name of the calculation--this is the name of the function.
        self.assertEqual(result.name, 'pimms.test.workflow.test_core.result')
        # The inputs of the calculation are a set of the inputs of the function.
        self.assertEqual(result.inputs, set(['input_1', 'input_2']))
        # The default values of the inputs are stored in the defaults member.
        self.assertEqual(result.defaults, {'input_2': None})
        # The outputs are a tuple of the output names. For a calc without
        # explicitly listed outputs has only one output, its name.
        self.assertEqual(result.outputs, ('result',))
        # The input documentation is stored in the input_docs member.
        self.assertIn('input_1', result.input_docs)
        self.assertIn('input_2', result.input_docs)
        self.assertEqual(len(result.input_docs), 2)
        self.assertIn('The number of elements to include in the result.',
                      result.input_docs['input_1'])
        self.assertIn('The object to put in the list.',
                      result.input_docs['input_2'])
        # The output documentation is stored in the output_docs member.
        self.assertIn('result', result.output_docs)
        self.assertIn('A list of `input_1` occurrences of `input_2`.',
                      result.output_docs['result'])
        self.assertEqual(len(result.output_docs), 1)
        # The calculation can be called using its normal signature.
        self.assertEqual(result(1), {'result': [None]})
        self.assertEqual(result(2, 0), {'result': [0, 0]})
        # It can also be called using the mapcall method.
        self.assertEqual(result.mapcall(dict(input_1=1)),
                         {'result': [None]})
        self.assertEqual(result.mapcall(dict(input_1=2, input_2=0)),
                         {'result': [0,0]})
        
