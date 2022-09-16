# -*- coding: utf-8 -*-
################################################################################
# pimms/doc/_core.py
#
# Implementation of the documentation tools used by pimms.
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
from re import compile as _re_compile
from functools import wraps as _wraps
from docrep import DocstringProcessor as _DocstringProcessor

# The Document Processor #######################################################
def make_docproc():
    """Creates and returns a document preprocessor."""
    docproc = _DocstringProcessor()
    # We need to add a few features to the docproc's members so that we can
    # process the Inputs and Outputs sections when present.
    docproc.param_like_sections = \
        docproc.param_like_sections + ['Inputs','Outputs']
    docproc.patterns['Inputs'] = _re_compile(
        docproc.patterns['Parameters'].pattern
            .replace('Parameters', 'Inputs')
            .replace('----------', '------'))
    docproc.patterns['Outputs'] = _re_compile(
        docproc.patterns['Parameters'].pattern
            .replace('Parameters', 'Outputs')
            .replace('----------', '-------'))
    return docproc

docproc = make_docproc()
"""The `docrep.DocstringProcessor` object used by `pimms`.

This object is used to process all of the doc-strings in the `pimms` library;
it should be used only with the `pimms.docwrap` decorator, which can safely be
applied anywhere in a sequence of decorators and which correctly applies the
`wraps` decorator to its argument. Function documentation is always processed
using the `sections=('Parameters', 'Returns', 'Raises', 'Examples', 'Inputs',
'Outputs')` parameter and the `with_indent(4)` decorator. The base-name for the
function `f` is `f.__module__ + '.' + f.__name__`.
"""

def _docwrap(f, fnname, indent=None, proc=docproc):
    # If no ident number was provided, deduce it.
    if indent is None:
        if not hasattr(f, '__doc__') or f.__doc__ is None:
            # Doesn't matter, no documentation.
            indent = 0
        else:
            lines = f.__doc__.split('\n')
            # We always skip the first line (the one that starts with """).
            lines = lines[1:]
            # Strip the lines.
            striplines = [s.lstrip() for s in lines]
            # Pick out the ones with text in them and calculate the indentation.
            indents = [len(ws_s) - len(s)
                       for (ws_s,s) in zip(lines, striplines)
                       if len(s) > 0]
            # The minimum is the one we want.
            indent = min(indents) if len(indents) > 0 else 0
    ff = f
    ff = proc.with_indent(indent)(ff)
    fd = proc.get_sections(base=fnname, sections=proc.param_like_sections)
    ff = fd(ff)
    ff = _wraps(f)(ff)
    # Post-process the documentation sections.
    for section in ['parameters', 'other_parameters', 'inputs', 'outputs']:
        k = fnname + '.' + section
        v = proc.params.get(k, '')
        if len(v) == 0: continue
        for ln in v.split('\n'):
            # Skip lines that start with whitespace.
            if ln[0].strip() == '': continue
            pname = ln.split(':')[0].strip()
            proc.keep_params(k, pname)
    return ff
def docwrap(f=None, indent=None, proc=docproc):
    """Applies standard doc-string processing to the decorated function.

    The `pimms.docwrap` decorator applies a standard set of pre-processing to
    the docstring of the function that follows it. This processing amounts to
    using the `docrep` module's `DocstringProcessor` as a filter on the
    documentation of the function. The function's documentation is always placed
    in the base-name equal to its fully-qualified namespace name.

    When called as `@docwrap(name)` for a string `name`, the documentation for
    the decorated function is instead placed under the base-name `name`.
    """
    # If we've been given a string, then we've been called as @docwrap(name)
    # instead of @docwrap.
    if f is None:
        return lambda fn: _docwrap(fn, fn.__module__ + '.' + fn.__name__,
                                   indent=indent, proc=proc)
    if isinstance(f, str):
        return lambda fn: _docwrap(fn, f, indent=indent, proc=proc)
    else:
        return _docwrap(f, f.__module__ + '.' + f.__name__,
                        indent=indent, proc=proc)
