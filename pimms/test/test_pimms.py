####################################################################################################
# tests/test_pimms.py
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
from .context import pimms

class TestPimms(unittest.TestCase):
    '''
    The TestPimms class defines all the tests for the pimms library.
    '''

    def test_basics(self):
        '''
        test_basics makes sure that the basic syntax of immutables works.
        '''
        # Declare a Class
        @pimms.immutable
        class TestImm(object):
            @pimms.param
            def first_name(val):
                return val.lower().capitalize()
            @pimms.param
            def last_name(val):
                return val.lower().capitalize()
            @pimms.option(None)
            def title(val):
                return val
            @pimms.value
            def full_name(title, first_name, last_name):
                return ('' if title is None else (title + ' ')) + ' '.join([first_name, last_name])
            @pimms.require
            def full_name_long_enough(full_name):
                return len(full_name) > 5
            @pimms.require
            def last_name_long_enough(last_name):
                return len(last_name) > 3

            def __init__(self, first, last, title=None):
                self.first_name = first
                self.last_name = last
                self.title = title
        # instantiate that class
        imm1 = TestImm('Nurgle', 'Turgles', title='Mr.')
        imm2 = TestImm('Mogget', 'Bogglish', title='Cpl.')
        imm3 = TestImm('Tobias', 'Van Pelt')

        # Test the names:
        self.assertEqual(imm1.full_name, 'Mr. Nurgle Turgles')
        self.assertEqual(imm2.full_name, 'Cpl. Mogget Bogglish')
        self.assertEqual(imm3.full_name, 'Tobias Van pelt')

if __name__ == '__main__':
    unittest.main()

