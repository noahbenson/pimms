# -*- coding: utf-8 -*-
####################################################################################################
# pimms/doc/_core.py
# Implementation of the documentation tools used by pimms.
# By Noah C. Benson

# Dependencies #####################################################################################
from re import compile as _re_compile
from functools import wraps as _wraps
from docrep import DocstringProcessor as _DocstringProcessor

# The Document Processor ###########################################################################
docproc = _DocstringProcessor()
"""The `docrep.DocstringProcessor` object used by `pimms`.

This object is used to process all of the doc-strings in the `pimms` library;
it should be used only with the `pimms.docwrap` decorator, which can safely be
applied anywhere in a sequence of decorators and which correctly applies the
`wraps` decorator to its argument. Function documentation is always processed
using the `sections=('Parameters', 'Returns', 'Raises', 'Examples', 'Inputs',
'Outputs')` parameter and the `with_indent(4)` decorator. The base-name for the
function `f` is `f.__module__ + '.' + f.__name__`.
"""
# We need to add a few features to the docproc's members so that we can process the Inputs and
# Outputs sections when present.
docproc.param_like_sections = docproc.param_like_sections + ['Inputs', 'Outputs']
docproc.patterns['Inputs'] = _re_compile(
    docproc.patterns['Parameters'].pattern
       .replace('Parameters', 'Inputs')
       .replace('----------', '------'))
docproc.patterns['Outputs'] = _re_compile(
    docproc.patterns['Parameters'].pattern
       .replace('Parameters', 'Outputs')
       .replace('----------', '-------'))
def _docwrap(f, fnname, indent=4):
    ff = f
    ff = docproc.with_indent(indent)(ff)
    ff = docproc.get_sections(base=fnname, sections=docproc.param_like_sections)(ff)
    ff = _wraps(f)(ff)
    # Post-process the documentation sections.
    for section in ['parameters', 'other_parameters', 'inputs', 'outputs']:
        k = fnname + '.' + section
        v = docproc.params.get(k, '')
        if len(v) == 0: continue
        for ln in v.split('\n'):
            # Skip lines that start with whitespace.
            if ln[0].strip() == '': continue
            pname = ln.split(':')[0].strip()
            docproc.keep_param(k, pname)        
    return ff
def docwrap(f=None, indent=4):
    """Applies standard doc-string processing to the decorated function.

    The `pimms.docwrap` decorator applies a standard set of pre-processing to
    the docstring of the function that follows it. This processing amounts to
    using the `docrep` module's `DocstringProcessor` as a filter on the
    documentation of the function. The function's documentation is always placed
    in the base-name equal to its fully-qualified namespace name.

    When called as `@docwrap(name)` for a string `name`, the documentation for
    the decorated function is instead placed under the base-name `name`.
    """
    # If we've been given a string, then we've been called as @docwrap(name) instead of @docwrap.
    if f is None:
        return lambda fn: _docwrap(fn, fn.__module__ + '.' + fn.__name__), indent=indent)
    if isinstance(f, str):
        return lambda fn: _docwrap(fn, f, indent=indent)
    else:
        return _docwrap(f, f.__module__ + '.' + f.__name__, indent=indent)
