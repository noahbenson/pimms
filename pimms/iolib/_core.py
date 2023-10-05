# -*- coding: utf-8 -*-
################################################################################
# pimms/iolib/_core.py

"""Input and output operations via the load() and save() functions."""


# Dependencies #################################################################

import os, sys, io
from collections  import namedtuple

from pcollections import pdict, plist, pset, ldict, lazy

from ..doc        import docwrap
from ..util       import is_str, is_amap, is_aseq, is_real
from ..pathlib    import path, is_path, like_path


# Save #########################################################################

class Save:
    """Saves a Python object to a stream or path then returns the stream/path.

    `pimms.save(path, object, format)` saves the given `object` to the given
    `path` using the named file `format` and returns the path on success or
    rasies an error on failure. If the format can be deduced from the path
    suffix, then it may be omitted.

    `pimms.save(stream, object, format)` writes the given `object` to the given
    stream object using the named `format` and returns the stream. 
    
    In fact, `pimms.save` is an object of the `pimms.pathlib.Save` type that
    primarily behaves like a function. The `pimms.save.register` and
    `pimms.save.unregister` methods can be used to add understood formats.

    Parameters
    ----------
    dest : path-like or stream
        The output destination to which the object is to be saved. This may be
        either a path-like object or a writeable `IOBase` stream.
    obj : object
        Any object that can be saved in the given format.
    format : str or None, optional
        If provided, `format` must be a string that names a format that has been
        previously registered with `save` using the `save.register` method. If
        `format` is not provided or is `None` (the default), then an attempt is
        made to deduce the format using the suffix of the `dest` argument,
        assuming that `dest` is a path and not a stream. If `dest` is a stream
        or if the suffix does not indicate a specific format, then a
        `ValueError` is raised.
    **kwargs
        Any additional parameters are passed to the registered export function.

    Returns
    -------
    path-like or stream
        The destination stream or path. If the `dest` argument is a path name,
        then a path object is returned instead of the path name.

    Raises
    ------
    TypeError
        If the destination is not a stream or path-like object or if the format
        is not a string.
    ValueError
        If the format is not recognized or if it cannot be deduced from the
        destination object.
    """
    # The Format class, an inset class for managing the various stream formats.
    class Format:
        """Manages the details of a stream format for use with `pimms.save`.

        `Format` objects are typically created with the `save.register` method.
        This class should be handled internally but should not be instantiated
        outside of the `Save` class.

        `Format` objects can be treated as save functions: the `__call__` method
        on a `Format` object accepts a stream or path-like object representing
        the destination to which the format should be written, an object to save
        in the format, and any optional arguments understood by the type.
        """
        #__slots__ = ("name", "function", "suffixes", "mode")
        def __init__(self, name, function, *suffixes, mode='b'):
            self.name = name
            self.function = function
            self.suffixes = []
            for suff in suffixes:
                if is_str(suff):
                    suff = (suff,)
                elif all(is_str, suff):
                    suff = tuple(suff)
                else:
                    raise ValueError(f"invalid path suffix: {suff}")
                self.suffixes.append(suff)
            if mode != 'b' and mode != 't':
                raise ValueError("format mode must be 't' or 'b'")
            self.mode = mode
            # We also want to change the documentation.
            self.__doc__ = self.function.__doc__
        def __call__(self, dest, obj, **kwargs):
            # If dest isn't a stream, we need to open it as a path.
            if isinstance(dest, io.IOBase):
                self.function(dest, obj, **kwargs)
            else:
                dest = path(dest)
                with dest.open('w' + self.mode) as stream:
                    self.function(stream, obj, **kwargs)
            return dest
    # Details of the Save class itself.
    __slots__ = ("formats", "_format_by_suffix")
    def __init__(self, template=None):
        self.formats = {}
        self._format_by_suffix = {}
        if template is not None:
            if isinstance(template, Save):
                formats = template.formats
            elif is_amap(template):
                formats = templates
            else:
                raise TypeError("templates must be a Save object or a mapping")
            for (k,format) in formats.items():
                if not isinstance(format, Save.Format):
                    raise TypeError(f"template contains non-format named {k}")
                self.register(format)
    def __call__(self, dest, obj, format=None, /, **kwargs):
        if isinstance(dest, io.IOBase):
            dest_path = None
            dest_stream = dest
        else:
            dest_path = path(dest)
            dest_stream = None
        if format is None:
            if dest_path is None:
                raise ValueError("format cannot be None when dest is a stream")
            # We can walk through the formats and attempt to deduce the correct
            # format from the ending of the dest_str.
            suff = tuple(dest_path.suffixes)
            while suff:
                format = self._format_by_suffix.get(suff)
                if format:
                    break
                suff = suff[1:]
            else:
                raise ValueError(f"format cannot be deduced from path: {dest}")
        # At this point we have a format name or we've raised an error.
        if is_str(format):
            fmt = self.formats.get(format)
            if fmt is None:
                raise ValueError(f"format not recognized: {format}")
            else:
                format = fmt
        elif not isinstance(format, Save.Format):
            raise TypeError("format arg must be a format name or Format object")
        # Now we know the format and thus have a format function; if we were
        # given a path instead of a stream, we need to open the path for the
        # formatting function.
        if dest_stream:
            format.function(dest_stream, obj, **kwargs)
            return dest_stream
        else:
            with dest_path.open('w' + format.mode) as stream:
                format.function(stream, obj, **kwargs)
            return dest_path
        # That concludes the save function!
    def register(self, name, /, *suffixes, mode='b'):
        """Registers a format type with the `pimms.save` interface.

        The `pimms.save.register` method is intended to be used as a decorator:
        `@pimms.save.register(format_name, suffix1, suffix2...)` ensures that
        the function that follows it is registered as a format managed by the
        `pimms.save` system. The function that is decorated should be written to
        accept a stream object (though the decorator will ensure that it works
        when called with a path-like object or a stream). The function's
        signature must always match `f(output_stream, object_to_save, **opts)`
        where the `**opts` must be any set of named parameters. The return value
        of the function is ignored, but it should raise an error if the object
        cannot be saved.

        Parameters
        ----------
        name : str
            The name of the format.
        *suffixes : strings or tuples of strings, optional
            Any number of suffixes that this format uses. Suffixes must be
            unique, and an error is raised if a requested suffix is already
            registered to another format. Each suffix may be a string such as
            `.tgz` or a tuple such as `('.tar', '.gz')`.
        mode : 't' or 'b', optional
            Whether any stream that is opened in order to save the file should
            be opened in text mode (`'t'`) or binary mode (`'b'`). The default
            is `'b'`.
        """
        # There are actually two uses: (1) that described in the help above and
        # (2) save.register(format) where format is already a Format object. In
        # the latter case we aren't a decorator.
        if isinstance(name, Save.Format):
            # We're in case 2, so we register this format specifically.
            format = name
            if format.name in self.formats:
                raise RuntimeError(f"format {format.name} already registered")
            for suff in format.suffixes:
                ex = self._format_by_suffix.get(suff)
                if ex is not None:
                    raise RuntimeError(
                        f"suffix {suff} already mapped to format {ex.name}")
            # We pass the criteria; go ahead and add this format.
            self.formats[format.name] = format
            for suff in format.suffixes:
                self._format_by_suffix[suff] = format
            # Return the format itself.
            return format
        else:
            # We're in case 1, so we return a decorator for the function.
            def _save_register_dec(f):
                if isinstance(f, Save.Format):
                    f = f.function
                format = Save.Format(name, f, *suffixes, mode=mode)
                return self.register(format)
            return _save_register_dec
        # That's it for the register function.
    def unregister(self, name, *, error_on_missing=False):
        """Unregisters the format with the given name from the save system.

        The format with the given name is unregistered from the `pimms.save`
        system, and the `Format` object that is removed is returned. If the
        format is not found, then `None` is returned, but no error is raised by
        default.

        Parameters
        ----------
        name : str
            The name of the format to unregister.
        error_on_missing : boolean, optional
            Whether to throw a `RuntimeError` if the named format is not found
            in the save manager. By default this is `False`.

        Returns
        -------
        Format or None
            The format object that is unregistered or `None` if the given name
            was not a registered format.

        Raises
        ------
        RuntimeError
            If the given name does not map to a format in the save manager and
            the `error_on_missing` option is `True`.
        """
        format = self.formats.get(name)
        if format is None:
            if error_on_missing:
                raise RuntimeError(f"format {name} not found in save system")
            else:
                return None
        # Actually remove things:
        for suff in format.suffixes:
            del self._format_by_suffix[suff]
        del self.formats[name]
        return format
    def copy(self):
        """Returns a copy of the given save manager.

        `pimms.save.copy()` can be used to return a copy of the save manager,
        for instances where a single save manager is not ideal.
        """
        return Save(self)

# We can go ahead and declare a single global save object for the pimms library.
# We will later use this object to register various basic format types.
save = Save()

# Register a few basic file formats.
@save.register('str', mode='t')
def save_str(stream, obj, append_nl=False):
    """Saves `str(obj)` to the given stream."""
    stream.write(str(obj))
    if append_nl:
        stream.write('\n')
@save.register('bytes', mode='t')
def save_bytes(stream, obj):
    """Saves `bytes(obj)` to the given stream."""
    stream.write(bytes(obj))
@save.register('repr', mode='t')
def save_repr(stream, obj, append_nl=False):
    """Saves `repr(obj)` to the given stream."""
    stream.write(repr(obj))
    if append_nl:
        stream.write('\n')
@save.register('text', '.txt', '.text', mode='t')
def save_text(stream, lines, append_nls=False):
    """Saves a blob of text or a sequence of lines to a stream or path.

    Parameters
    ----------
    stream : stream or path-like
        The stream or path to which to write the lines.
    lines : string or sequence of strings
        The text that should be written to the file or stream.
    append_nls : boolean, optional
        Whether to append newlines to the end of each line. If `False` (the
        default), then the lines are written as they are; otherwise, a newline
        character is appended after each line in the `lines` argument.
    """
    if is_str(lines):
        lines = (lines,)
    if not is_aseq(lines):
        raise TypeError("lines must be text or a sequence of text objects")
    if append_nls:
        for ln in lines:
            stream.write(ln)
            stream.write('\n')
    else:
        for ln in lines:
            stream.write(ln)
@save.register('pickle', '.pickle', '.pkl', '.pcl', mode='b')
def save_pickle(stream, obj, protocol=None, **kwargs):
    """Saves a pickled object to a destination path or stream.

    All keyword options are forwarded to the `pickle.dump` function. The
    `protocol` keyword option is provided as the `protocol` positional argument
    to the `pickle.dump` function.
    """
    import pickle
    pickle.dump(obj, stream, protocol, **kwargs)
def json_default(obj):
    """Converts an object to a json-formattable object or raises TypeError.
    """
    if is_str(obj) or is_real(obj) or obj is None:
        return obj
    elif is_amap(obj):
        return dict(obj)
    elif is_aseq(obj):
        return list(obj)
    else:
        raise TypeError(type(obj))
@save.register('json', '.json', mode='t')
def save_json(stream, obj, /, default=json_default, **kwargs):
    """Saves an object as a JSON string or raises a TypeError if not possible.

    All keywords are passed along to the `json.dump` function. The `default`
    option uses the `pimms.iolib.json_default` function, which is different than
    the default used by `json.dump`, but all other options are unaltered.
    """
    import json
    json.dump(obj, stream, default=default, **kwargs)
def yaml_prepare(obj):
    """Returns a version of the argument that can be JSON/YAML serialized.
    """
    if is_str(obj) or is_real(obj) or obj is None:
        return obj
    elif is_amap(obj):
        r = {}
        for (k,v) in obj.items():
            if not is_str(k):
                raise TypeError("JSON/YAML dict keys must be strings")
            r[k] = yaml_prepare(v)
        return r
    elif is_aseq(obj):
        return [yaml_prepare(u) for u in obj]
    else:
        raise TypeError(type(obj))
@save.register('yaml', '.yaml', mode='t')
def save_yaml(stream, obj, /, **kwargs):
    """Saves an object as a YAML string or raises a TypeError if not possible.

    All keywords are passed along to the `yaml.Dumper` object that is used for
    dumping `obj`. Note that unlike with the `yaml.dump` function, object are
    only serialized if they are JSON/YAML serializable; classes that are not
    mappings, sequences, numbers, strings, or booleans cannot be serialized
    using this function and instead raise TypeErrors.
    """
    import yaml
    yaml.dump(yaml_prepare(obj), stream, **kwargs)
@save.register('csv', '.csv', mode='t')
def save_csv(stream, obj, /, index=False, **kwargs):
    """Saves a pandas DataFrame to a CSV file.

    All options are passed along to `pandas.DataFrame.to_csv()`. The option
    `index` has the default value of `False` in this function.
    """
    import pandas
    obj = pandas.DataFrame(obj)
    obj.to_csv(stream, index=index, **kwargs)
@save.register('tsv', '.tsv', mode='t')
def save_tsv(stream, /, sep="\t", index=False, **kwargs):
    """Saves a pandas DataFrame to a TSV file.

    All options are passed along to `pandas.DataFrame.to_csv()`. The option
    `index` has the default value of `False` in this function, and the option
    `sep` has the default value `"\t"`.
    """
    import pandas
    obj = pandas.DataFrame(obj)
    obj.to_csv(stream, sep=sep, index=index, **kwargs)


# Load #########################################################################

class Load:
    """Loads a Python object from a stream or path then returns the object.

    `pimms.load(path, format)` loads a Python object from the given `path` using
    the named file `format` and returns the loaded object or rasies an error on
    failure. If the format can be deduced from the path suffix, then it may be
    omitted.

    `pimms.load(stream, format)` reads a Python object from the given stream
    object using the named `format` and returns the object. The format argument
    cannot be omitted when the first argument is a stream.
    
    In fact, `pimms.load` is an object of the `pimms.pathlib.Load` type that
    primarily behaves like a function. The `pimms.load.register` and
    `pimms.load.unregister` methods can be used to add understood formats.

    Parameters
    ----------
    source : path-like or stream
        The input source from which the object is to be loaded. This may be
        either a path-like object or a readable `IOBase` stream.
    format : str or None, optional
        If provided, `format` must be a string that names a format that has been
        previously registered with `load` using the `load.register` method. If
        `format` is not provided or is `None` (the default), then an attempt is
        made to deduce the format using the suffix of the `source` argument,
        assuming that `source` is a path and not a stream. If `source` is a
        stream or if the suffix does not indicate a specific format, then a
        `ValueError` is raised.
    **kwargs
        Any additional parameters are passed to the registered export function.

    Returns
    -------
    object
        The object that was loaded from the given stream or path.

    Raises
    ------
    TypeError
        If the input source is not a stream or path-like object or if the format
        is not a string.
    ValueError
        If the format is not recognized or if it cannot be deduced from the
        source.
    """
    # The Format class, an inset class for managing the various stream formats.
    class Format:
        """Manages the details of a stream format for use with `pimms.load`.

        `Format` objects are typically created with the `load.register` method.
        This class should be handled internally but should not be instantiated
        outside of the `Load` class.

        `Format` objects can be treated as load functions: the `__call__` method
        on a `Format` object accepts a stream or path-like object representing
        the source from which the format should be read and any optional
        arguments understood by the type. It returns the read object.
        """
        #__slots__ = ("name", "function", "suffixes", "mode")
        def __init__(self, name, function, *suffixes, mode='b'):
            self.name = name
            self.function = function
            self.suffixes = []
            for suff in suffixes:
                if is_str(suff):
                    suff = (suff,)
                elif all(is_str, suff):
                    suff = tuple(suff)
                else:
                    raise ValueError(f"invalid path suffix: {suff}")
                self.suffixes.append(suff)
            if mode != 'b' and mode != 't':
                raise ValueError("format mode must be 't' or 'b'")
            self.mode = mode
            # We also want to change the documentation.
            self.__doc__ = self.function.__doc__
        def __call__(self, source, **kwargs):
            # If dest isn't a stream, we need to open it as a path.
            if isinstance(source, io.IOBase):
                return self.function(source, **kwargs)
            else:
                source = path(source)
                with dest.open('r' + self.mode) as stream:
                    return self.function(stream, **kwargs)
    # Details of the Save class itself.
    __slots__ = ("formats", "_format_by_suffix")
    def __init__(self, template=None):
        self.formats = {}
        self._format_by_suffix = {}
        if template is not None:
            if isinstance(template, Save):
                formats = template.formats
            elif is_amap(template):
                formats = templates
            else:
                raise TypeError("templates must be a Load object or a mapping")
            for (k,format) in formats.items():
                if not isinstance(format, Save.Format):
                    raise TypeError(f"template contains non-format named {k}")
                self.register(format)
    def __call__(self, src, format=None, /, **kwargs):
        if isinstance(src, io.IOBase):
            src_path = None
            src_stream = src
        else:
            src_path = path(src)
            src_stream = None
        if format is None:
            if src_path is None:
                raise ValueError("format cannot be None when src is a stream")
            # We can walk through the formats and attempt to deduce the correct
            # format from the ending of the src_str.
            suff = tuple(src_path.suffixes)
            while suff:
                format = self._format_by_suffix.get(suff)
                if format:
                    break
                suff = suff[1:]
            else:
                raise ValueError(f"format cannot be deduced from path: {src}")
        # At this point we have a format name or we've raised an error.
        if is_str(format):
            fmt = self.formats.get(format)
            if fmt is None:
                raise ValueError(f"format not recognized: {format}")
            else:
                format = fmt
        elif not isinstance(format, Load.Format):
            raise TypeError("format arg must be a format name or Format object")
        # Now we know the format and thus have a format function; if we were
        # given a path instead of a stream, we need to open the path for the
        # formatting function.
        if src_stream:
            return format.function(src_stream, **kwargs)
        else:
            with src_path.open('r' + format.mode) as stream:
                return format.function(stream, **kwargs)
        # That concludes the load function!
    def register(self, name, /, *suffixes, mode='b'):
        """Registers a format type with the `pimms.load` interface.

        The `pimms.load.register` method is intended to be used as a decorator:
        `@pimms.load.register(format_name, suffix1, suffix2...)` ensures that
        the function that follows it is registered as a format managed by the
        `pimms.load` system. The function that is decorated should be written to
        accept a stream object (though the decorator will ensure that it works
        when called with a path-like object or a stream). The function's
        signature must always match `f(input_stream, **opts)` where the `**opts`
        must be any set of named parameters. The return value of the function is
        ignored, but it should raise an error if an object cannot be loaded.

        Parameters
        ----------
        name : str
            The name of the format.
        *suffixes : strings or tuples of strings, optional
            Any number of suffixes that this format uses. Suffixes must be
            unique, and an error is raised if a requested suffix is already
            registered to another format. Each suffix may be a string such as
            `.tgz` or a tuple such as `('.tar', '.gz')`.
        mode : 't' or 'b', optional
            Whether any stream that is opened in order to save the file should
            be opened in text mode (`'t'`) or binary mode (`'b'`). The default
            is `'b'`.
        """
        # There are actually two uses: (1) that described in the help above and
        # (2) load.register(format) where format is already a Format object. In
        # the latter case we aren't a decorator.
        if isinstance(name, Load.Format):
            # We're in case 2, so we register this format specifically.
            format = name
            if format.name in self.formats:
                raise RuntimeError(f"format {format.name} already registered")
            for suff in format.suffixes:
                ex = self._format_by_suffix.get(suff)
                if ex is not None:
                    raise RuntimeError(
                        f"suffix {suff} already mapped to format {ex.name}")
            # We pass the criteria; go ahead and add this format.
            self.formats[format.name] = format
            for suff in format.suffixes:
                self._format_by_suffix[suff] = format
            # Return the format itself.
            return format
        else:
            # We're in case 1, so we return a decorator for the function.
            def _load_register_dec(f):
                if isinstance(f, Load.Format):
                    f = f.function
                format = Load.Format(name, f, *suffixes, mode=mode)
                return self.register(format)
            return _load_register_dec
        # That's it for the register function.
    def unregister(self, name, *, error_on_missing=False):
        """Unregisters the format with the given name from the load system.

        The format with the given name is unregistered from the `pimms.load`
        system, and the `Format` object that is removed is returned. If the
        format is not found, then `None` is returned, but no error is raised by
        default.

        Parameters
        ----------
        name : str
            The name of the format to unregister.
        error_on_missing : boolean, optional
            Whether to throw a `RuntimeError` if the named format is not found
            in the load manager. By default this is `False`.

        Returns
        -------
        Format or None
            The format object that is unregistered or `None` if the given name
            was not a registered format.

        Raises
        ------
        RuntimeError
            If the given name does not map to a format in the load manager and
            the `error_on_missing` option is `True`.
        """
        format = self.formats.get(name)
        if format is None:
            if error_on_missing:
                raise RuntimeError(f"format {name} not found in load system")
            else:
                return None
        # Actually remove things:
        for suff in format.suffixes:
            del self._format_by_suffix[suff]
        del self.formats[name]
        return format
    def copy(self):
        """Returns a copy of the given load manager.

        `pimms.load.copy()` can be used to return a copy of the load manager,
        for instances where a single load manager is not ideal.
        """
        return Load(self)

# We can go ahead and declare a single global save object for the pimms library.
# We will later use this object to register various basic format types.
load = Load()

# Register a few basic file formats.
@load.register('str', mode='t')
def load_str(stream, strip_nl=False, size=-1):
    """Loads a string from the given source."""
    s = stream.read(size)
    if strip_nl:
        s = s.rstrip('\n')
    return s
@load.register('bytes', mode='b')
def load_bytes(stream, size=-1):
    """Loads a string from the given source."""
    return stream.read(size)
@load.register('repr', mode='t')
def load_repr(stream, size=-1):
    """Loads an object using the `ast.literal_eval` function."""
    from ast import literal_eval
    return literal_eval(stream.read(size))
@load.register('text', '.txt', '.text', mode='t')
def load_text(stream, strip_nls=True, size=-1):
    """Loads a blob of text or a sequence of lines from a stream or path.

    Parameters
    ----------
    stream : stream or path-like
        The stream or path from which to read the lines.
    strip_nls : boolean, optional
        Whether to strip newlines from the end of each line. If `False`, then
        the lines are returned as they are; if `True` (the default), any newline
        character is stripped from each line in the `lines` argument.
    """
    if strip_nls:
        return stream.read(size).splitlines()
    else:
        return stream.readlines(size)
@load.register('pickle', '.pickle', '.pkl', '.pcl', mode='b')
def load_pickle(stream, **kwargs):
    """Loads a pickled object from a path or stream and returns the object.

    All keyword options are forwarded to the `pickle.load` function.
    """
    import pickle
    return pickle.load(stream, **kwargs)
@load.register('json', '.json', mode='t')
def load_json(stream, /, **kwargs):
    """Loads an object from a JSON stream or path and returns the object.

    All keywords are passed along to the `json.load` function.
    """
    import json
    return json.load(stream, **kwargs)
@load.register('yaml', '.yaml', '.yml', mode='t')
def load_yaml(stream, /, safe=True):
    """Loads an object from a YAML stream or path and returns the object.

    The optional argument `safe` may be set to `False` to use unsafe YAML
    loading via the `yaml.load()` function.
    """
    import yaml
    if safe:
        return yaml.safe_load(stream)
    else:
        return yaml.load(stream)
@load.register('csv', '.csv', mode='t')
def load_csv(stream, /, sep=',', **kwargs):
    """Loads a pandas DataFrame from a CSV file.

    All options are passed along to `pandas.read_csv()`.
    """
    import pandas
    return pandas.read_csv(stream, sep=sep, **kwargs)
@load.register('tsv', '.tsv', mode='t')
def load_tsv(stream, /, sep="\t", **kwargs):
    """Loads a pandas DataFrame from a TSV file.

    All options are passed along to `pandas.read_csv()`.
    """
    import pandas
    return pandas.read_csv(stream, sep=sep, **kwargs)
