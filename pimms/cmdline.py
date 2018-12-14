####################################################################################################
# pimms/cmdline.py
# Immutable Class and functions for parsing command-lines and in particular for parsing a set of
# command-line arguments for use with a pimms calc-plan.
# By Noah C. Benson

import copy, types, os, sys, re, warnings, textwrap, six
import pyrsistent   as     pyr
import numpy        as     np
import collections  as     colls
from   functools    import reduce
from   .immutable   import (immutable, param, value)
from   .calculation import (Plan, is_plan)
from   .util        import (is_map, is_pmap, is_str, is_vector, merge)

@immutable
class CommandLineParser(object):
    '''
    CommandLineParser(instructions) yields a command line parser object, which acts as a function,
      that is capable of parsing command line options into a dictionary of opts and a list of args,
      as defined by the list of instructions.

    The instructions given to a command-line parser should be a list of lists (or tuples), 
    each of which should have three or four entries: [character, word, entry, default] where the
    default is optional. The character and word are the -x and --xxx versions of the argument; the
    entry is the name of the entry in the opts dictionary that is yielded for that command line
    option, and the default is the value inserted into the dictionary if the command line option is
    not found; if no default value is given, then the entry will appear in the dictionary only if it
    appears in the command line arguments.

    If the default value given is either True or False, then the option is understood to be a flag;
    i.e., the option does not take an argument (and single letter flags can appear together, such
    as -it instead of -i -t), and the appearance of the flag toggles the default to the opposite
    value.

    Unless the optional argument value_parse=False is provided, all values are interpreted via the
    ast.literal_eval() function; if this fails or raises an exception then the value is left as a
    string.

    The following optional arguments may be provided to the constructor:
      * value_parse (default: True) specifies whether the values are interpreted via the
        ast.literal_eval() function. This may be set to False to leave the values as strings or it
        may be set to a function that takes one argument and performs the parsing itself; such a
        function f must obey the syntax `parsed_val = f(string_val)`. The value_parse function is
        only called on arguments that have string values included. Note that by default the
        value_parse function interprets the string '...' as Ellipsis in addition to the typical
        ast.literal_eval() behavior.
      * filters (default: None) optionally specifies a dictionary of filter functions, each of which
        is passed the parsed value of the associated argument. Each filter function f must obey the
        syntax `final_value = f(parsed_value)`. The keys of this dictionary must be the entry names
        of the arguments.
    
    Example:
      parser = CommandLineParser(
        [('a', 'flag-a', 'aval', False),
         ('b', 'flag-b', 'bval', True),
         ('c', 'flag-c', 'cval', True),
         ('d', 'flag-d', 'dval', False),
         ('e', 'opt-e',  'eval', None),
         ('f', 'opt-f',  'fval', None),
         ('g', 'opt-g',  'gval'),
         ('h', 'opt-h',  'hval')])
      cmd_line = ['-ab', 'arg1', '--flag-d', '-etestE', '--opt-f=123', '-htestH', 'arg2']
      parser(cmd_line) == parser(*cmd_line)  # identical calls
      parser(cmd_line)
      # ==> (['arg1', 'arg2'],
      # ==>  {'a':True, 'b':False, 'c':True, 'd':True, 'e':'testE', 'f':'123', 'h':'testH'})
    '''

    @staticmethod
    def parse_literal(s):
        '''
        CommandLineParser.parse_literal(s) is equivalent to ast.literal_eval(s) except that (1) it
          additionally interprets '...' to be Ellipsis and (2) when s cannot be parsed as a literal
          it instead returns the string s.
        '''
        from ast import literal_eval
        try:    return literal_eval(s)
        except: return Ellipsis if s.strip() == '...' else s
    
    def __init__(self, instructions, value_parser=True, filters=None):
        'See help(CommandLineParser).'
        wflags = {}
        cflags = {}
        wargs = {}
        cargs = {}
        defaults = {}
        for row in instructions:
            if not hasattr(row, '__iter__') or len(row) < 3 or len(row) > 4 or \
               any(x is not None and not is_str(x) for x in row[:3]):
                raise ValueError('Invalid instruction row: %s ' % row)
            (c, w, var, dflt) = row if len(row) == 4 else (list(row) + [None])
            defaults[var] = dflt
            if dflt is True or dflt is False:
                if c is not None: cflags[c] = var
                if w is not None: wflags[w] = var
            else:
                if c is not None: cargs[c] = var
                if w is not None: wargs[w] = var
        self.default_values = pyr.pmap(defaults)
        self.flag_words = pyr.pmap(wflags)
        self.flag_characters = pyr.pmap(cflags)
        self.option_words = pyr.pmap(wargs)
        self.option_characters = pyr.pmap(cargs)
        self.filters = filters
        self.value_parser = value_parser
    @param
    def filters(fs):
        '''
        parser.filters is a dictionary of filter functions for the various entries in the given
        parser's command line arguments. See also help(CommandLineParser).
        '''
        if not fs: return pyr.m()
        if not is_map(fs): raise ValueError('filters must be a mapping of entries to functions.')
        return pyr.pmap(fs)
    @param
    def value_parser(vp):
        '''
        parser.value_parser is a parse function that accepts a string and yields an interpretation
        of that string as a Python object (which may be the same string or a new value). If this
        value is set to True of Ellipsis, then the default ast.literal_eval()-like behavior is used.
        See also help(CommandLineParser).
        '''
        if vp in [Ellipsis, True]: return CommandLineParser.parse_literal
        elif not vp: return lambda x:x
        else: return vp
    @param
    def default_values(dv):
        '''
        clp.default_values yields the persistent map of default values for the given command-line
          parser clp.
        '''
        if is_pmap(dv): return dv
        elif is_map(dv): return pyr.pmap(dv)
        else: raise ValueError('default_value must be a mapping')
    @param
    def flag_words(u):
        '''
        clp.flag_words yields the persistent map of optional flag words recognized by the given
          command-line parser clp.
        '''
        if is_pmap(u): return u
        elif is_map(u): return pyr.pmap(u)
        else: raise ValueError('flag_words must be a mapping')
    @param
    def flag_characters(u):
        '''
        clp.flag_characters yields the persistent map of the flag characters recognized by the given
          command-line parser clp.
        '''
        if is_pmap(u): return u
        elif is_map(u): return pyr.pmap(u)
        else: raise ValueError('flag_characters must be a mapping')
    @param
    def option_words(u):
        '''
        clp.option_words yields the persistent map of optional words recognized by the given
          command-line parser clp.
        '''
        if is_pmap(u): return u
        elif is_map(u): return pyr.pmap(u)
        else: raise ValueError('option_words must be a mapping')
    @param
    def option_characters(u):
        '''
        clp.option_characters yields the persistent map of optional characters recognized by the
          given command-line parser clp.
        '''
        if is_pmap(u): return u
        elif is_map(u): return pyr.pmap(u)
        else: raise ValueError('option_characters must be a mapping')
        
    def __call__(self, *args):
        if len(args) > 0 and not is_str(args[0]) and is_vector(args[0]):
            args = list(args)
            return self.__call__(*(list(args[0]) + args[1:]))
        parse_state = None
        more_opts = True
        remaining_args = []
        opts = dict(self.default_values)
        wflags = self.flag_words
        cflags = self.flag_characters
        wargs  = self.option_words
        cargs  = self.option_characters
        dflts  = self.default_values
        for arg in [aa for arg in args for aa in (arg if is_vector(arg) else [arg])]:
            larg = arg.lower()
            if parse_state is not None:
                opts[parse_state] = arg
                parse_state = None
            else:
                if arg == '': pass
                elif more_opts and arg[0] == '-':
                    if len(arg) == 1:
                        remaining_args.append(arg)
                    elif arg[1] == '-':
                        trimmed = arg[2:]
                        if trimmed == '':     more_opts = False
                        if trimmed in wflags: opts[wflags[trimmed]] = not dflts[wflags[trimmed]]
                        else:
                            parts = trimmed.split('=')
                            if len(parts) == 1:
                                if trimmed not in wargs:
                                    #raise ValueError('Unrecognized flag/option: %s' % trimmed)
                                    remaining_args.append(arg)
                                else:
                                    # the next argument specifies this one
                                    parse_state = wargs[trimmed]
                            else:
                                k = parts[0]
                                if k not in wargs:
                                    #raise ValueError('Unrecognized option: %s' % k)
                                    remaining_args.append(arg)
                                else:
                                    opts[wargs[k]] = trimmed[(len(k) + 1):]
                    else:
                        trimmed = arg[1:]
                        for (k,c) in enumerate(trimmed):
                            if c in cflags: opts[cflags[c]] = not dflts[cflags[c]]
                            elif c in cargs:
                                remainder = trimmed[(k+1):]
                                if len(remainder) > 0: opts[cargs[c]] = remainder
                                else:
                                    # next argument...
                                    parse_state = cargs[c]
                                break
                else:
                    remaining_args.append(arg)
        if parse_state is not None:
            raise ValueError('Ran out of arguments while awaiting value for %s' % parse_state)
        # that's done; all args are parsed; now we can do the value parsing and the filter functions
        for (k,v) in six.iteritems(opts):
            u = v
            # parse the string, if needed:
            if is_str(v): u = self.value_parser(v)
            # filter if needed:
            if k in self.filters: u = self.filters[k](u)
            # update in options if need-be
            if v is not u: opts[k] = u
        return (remaining_args, opts)

def to_argv_schema(data, arg_names=None, arg_abbrevs=None, filters=None, defaults=None):
    '''
    to_argv_schema(instructions) yields a valid tuple of CommandLineParser instructions for the
      given instructions tuple; by itself, this will only return the instructions as they are, but
      optional arguments (below) will override the values in the instructions if provided.
    to_argv_schema(plan) yields a valid tuple of CommandLineParser instructions for the given plan
      object.

    These schema returned by this function will parse a command-line list (sys.argv) for parameters
    either listed in the instructions (see help(CommandLineParser)) or the afferent parameters of
    the plan. Generally, this should be called by the argv_parse() function and not directly.

    If a plan is given as the first argument, then the following rules are used to determine how
    arguments are parsed:
      * An argument that begins with -- (e.g., --quux-factor=10) is checked for a matching
        plan parameter; the argument name "quux-factor" will match either a parameter called
        "quux-factor" or "quux_factor" (dashes in command-line arguments are auto-translated
        into underscores).
      * If "quux_factor" is a parameter to the plan and is the only parameter that starts with a
        'q', then -q10 or -q 10 are both equivalent to --quux-factor=10. If other parameters
        also start with a 'q' then neither "quux_factor" or the other parameter(s) will be
        matched with the -q flag unless it is specified explicitly via the arg_abbrevs option.
      * Argument values are parsed using Python's ast.literal_eval(); if this raises an
        exception then the value is left as a string.
      * If an argument or flag is provided without an argument (e.g. "--quuxztize" or "-q") then
        it is interpreted as a boolean flag and is given the value True.
      * Arguments that come after the flag "--" are never processed.

    The following options may be given:
      * arg_names (default: None) may be a dictionary that specifies explicity command-line
        argument names for the plan parameters; plan parameters should be keys and the argument
        names should be values. Any parameter not listed in this option will be interpreted
        according to the above rules. If a parameter is mapped to None then it will not be
        filled from the command-line arguments.
      * arg_abbrevs (default:None) may be a dictionary that is handled identically to that of
        arg_names except that its values must be single letters, which are used for the
        abbreviated flag names.
      * defaults (default: None) may specify the default values for the plan parameters; this
        dictionary overrides the default values of the plan itself.
    '''
    if is_plan(data):
        # First we must convert it to a valid instruction list
        (plan, data) = (data, {})
        # we go through the afferent parameters...
        for aff in plan.afferents:
            # these are provided by the parsing mechanism and shouldn't be processed
            if aff in ['argv', 'argv_parsed', 'stdout', 'stderr', 'stdin']: continue
            # we ignore defaults for now
            data[aff] = (None, aff.replace('_', '-'), aff)
        # and let's try to guess at abbreviation names
        entries = sorted(data.keys())
        n = len(entries)
        for (ii,entry) in enumerate(entries):
            if ii > 0   and entry[0] == entries[ii-1][0]: continue
            if ii < n-1 and entry[0] == entries[ii+1][0]: continue
            r = data[entry]
            data[entry] = (entry[0], r[1], entry, r[3]) if len(r) == 4 else (entry[0], r[1], entry)
        # now go through and fix defaults...
        for (entry,dflt) in six.iteritems(plan.defaults):
            if entry not in data: continue
            r = data[entry]
            data[entry] = (r[0], r[1], r[2], dflt)
    elif arg_names is None and arg_abbrevs is None and defaults is None:
        # return the same object if there are no changes to a schema
        return data
    else:
        data = {r[2]:r for r in data}
    # Now we go through and make updates based on the optional arguments
    if arg_names is None: arg_names = {}
    for (entry,arg_name) in six.iteritems(arg_names):
        if entry not in data: continue
        r = data[entry]
        data[entry] = (r[0], arg_name, entry) if len(r) == 3 else (r[0], arg_name, entry, r[3])
    if arg_abbrevs is None: arg_abbrevs = {}
    for (entry,arg_abbrev) in six.iteritems(arg_abbrevs):
        if entry not in data: continue
        r = data[entry]
        data[entry] = (arg_abbrev, r[1], entry) if len(r) == 3 else (arg_abbrev, r[1], entry, r[3])
    if defaults is None: defaults = {}
    for (entry,dflt) in six.iteritems(defaults):
        if entry not in data: continue
        r = data[entry]
        data[entry] = (r[0], r[1], entry, dflt)
    # return the list-ified version of this
    return [tuple(row) for row in six.itervalues(data)]

def argv_parser(instructs,
                arg_names=None, arg_abbrevs=None, defaults=None,
                value_parser=True, filters=None):
    '''
    argv_parser(instructions) is equivalent to CommandLineParser(instructions).
    argv_parser(plan) is equivalent to CommandLineParser(plan_to_argv_instructions(plan)).

    See also help(CommandLineParser) and help(to_argv_schema); all optional arguments accepted by
    these functions are passed along by argv_parser.

    See also help(argv_parse), which is the recommended interface for parsing command-line
    arguments.
    '''
    schema = to_argv_schema(instructs,
                            arg_names=arg_names, arg_abbrevs=arg_abbrevs, defaults=defaults)
    return CommandLineParser(schema, value_parser=value_parser, filters=filters)

def argv_parse(schema, argv, init=None,
               arg_names=None, arg_abbrevs=None, value_parser=True, defaults=None, filters=None):
    '''
    argv_parse(schema, argv) yields the tuple (unparsed_argv, params) where unparsed_argv is a list
      subset of argv that contains only those command line arguments that were not understood by
      the given argument schema and params is a dictionary of parameters as parsed by the given 
      schema. It is equivalent to argv_parser(schema)(argv). See also help(CommandLineParser) for
      information about the instructions format.
    argv_parse(plan, argv) yields a pimms IMap object whose parameters have been initialized from
      the arguments in argv using the given pimms calculation plan as a template for the argument
      schema; see help(to_argv_schema) for information about the way plans are interpreted as argv
      schemas. The plan is initialized with the additional parameters 'argv' and 'argv_parsed'. The
      'argv' parameter contains the command-line arguments in argv that were not interpreted by the
      command-line parser; the 'argv_parsed' parameter contains the parsed command-line parameters.
      To avoid the plan-specific behavior and instead only parse the arguments from a plan, use
      argv_parse(to_argv_schema(plan), argv).

    The following options may be given:
      * init (default: None) specifies that the given dictionary should be merged into either the
        resulting options dictionary (if schema is a schema and not a plan) or into the parameters
        initially provided to the plan (if schema is a plan).
      * arg_names (default: None) may be a dictionary that specifies explicity command-line
        argument names for the plan parameters; plan parameters should be keys and the argument
        names should be values. Any parameter not listed in this option will be interpreted
        according to the above rules. If a parameter is mapped to None then it will not be
        filled from the command-line arguments.
      * arg_abbrevs (default:None) may be a dictionary that is handled identically to that of
        arg_names except that its values must be single letters, which are used for the
        abbreviated flag names.
      * defaults (default: None) may specify the default values for the plan parameters; this
        dictionary overrides the default values of the plan itself.
      * value_parse (default: True) specifies whether the values are interpreted via the
        ast.literal_eval() function. This may be set to False to leave the values as strings or it
        may be set to a function that takes one argument and performs the parsing itself; such a
        function f must obey the syntax `parsed_val = f(string_val)`. The value_parse function is
        only called on arguments that have string values included. Note that by default the
        value_parse function interprets the string '...' as Ellipsis in addition to the typical
        ast.literal_eval() behavior.
      * filters (default: None) optionally specifies a dictionary of filter functions, each of which
        is passed the parsed value of the associated argument. Each filter function f must obey the
        syntax `final_value = f(parsed_value)`. The keys of this dictionary must be the entry names
        of the arguments. Note that filter functions are called on provided default values but the
        value_parse function is not called on these.
    '''
    parser = argv_parser(schema,
                         arg_names=arg_names, arg_abbrevs=arg_abbrevs, defaults=defaults,
                         value_parser=value_parser, filters=filters)
    res = parser(argv)
    if is_plan(schema):
        init = {} if init is None else init
        init = pyr.pmap(init) if not is_pmap(init) else init
        return schema({'argv': tuple(res[0]), 'argv_parsed': pyr.pmap(res[1])}, res[1], init)
    else:
        return res if init is None else (res[0], dict(merge(res[1], init)))

class WorkLog(object):
    '''
    A WorkLog object is a simple print formatter. Given a column width, allows the user to print
    bullet points with customized indentation. Useful for printing progress in long computations
    or workflows.
    '''

    def __init__(self, columns=80, bullet='  * ', stdout=Ellipsis, stderr=Ellipsis):
        if stdout is Ellipsis: stdout = sys.stdout
        if stderr is Ellipsis: stderr = sys.stderr
        self.stdout = stdout
        self.stderr = stderr
        self.columns = columns
        self.bullet = bullet
        self.blanks = ''.join([' ' for x in bullet])
        self.inner_worklog = None
        
    def indent(self, n=1):
        '''
        worklog.indent() yields a duplicate worklog that is indented relative to worklog.
        worklog.indent(x) indents x times.
        '''
        return WorkLog(columns=self.columns,
                       bullet=(self.blanks + self.bullet),
                       stdout=self.stdout, stderr=self.stderr)

    def _write(self, fl, *args):
        if fl is None: return None
        n = self.columns - len(self.blanks)
        bksep = '\n' + self.blanks
        busep = '\n' + self.bullet
        pasep = '\n\n' + self.blanks
        s = self.bullet + busep.join(
            [(bksep.join(textwrap.wrap(arg.strip(), n)) if is_str(arg) else
              pasep.join([bksep.join(textwrap.wrap(s.strip(), n)) for s in arg]))
             for arg in args])
        r = fl.write(s + '\n')
        fl.flush()
        return r
    
    def __call__(self, *args):
        '''
        worlog(a, b, c...) prints each argument as a separate bullet point at the worklog's current
          indentation level.
        
        Each argument may be a string (which is word-wrapped) or a list of strings, each of which is
        word-wrapped and printed as a separate paragraph.
        '''
        return self._write(self.stdout, *args)
    def warn(self, *args):
        '''
        worlog.warn(a, b, c...) prints each argument as a separate bullet point at the worklog's
          current indentation level to the worklog's standard-error.
        
        Each argument may be a string (which is word-wrapped) or a list of strings, each of which is
        word-wrapped and printed as a separate paragraph.
        '''
        return self._write(self.stderr, *args)
    
def worklog(columns=None, bullet='  * ', stdout=Ellipsis, stderr=Ellipsis, verbose=False):
    '''
    worklog() yields a worklog object using sys.stdout and sys.stderr as the outputs.
    worklog(n) yields a worklog that formats output to n columns.

    The following options may be give:
      * bullet (default: '  * ') the bullet-text to print for each bullet.
      * stdout (default: Ellipsis) the file to use in place of stdout; if None, no stdout is
        printed; if Ellipsis, then is set to sys.stdout if verbose is True and None otherwise.
      * stderr (default: Ellipsis) the file to use in place of stderr; if None, no stderr is
        printed; if Ellipsis, then is set to sys.stderr.
      * verbose (default: False) specifies whether to use verbose output. This only has an effect
        if the stdout option is Ellipsis.
    '''
    if columns is None:
        try: columns = int(os.environ['COLUMNS'])
        except: columns = 80
    if stdout is Ellipsis:
        if verbose: stdout = sys.stdout
        else: stdout = None
    return WorkLog(columns=columns, bullet=bullet, stdout=stdout, stderr=stderr)
