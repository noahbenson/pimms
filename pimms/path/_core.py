# -*- coding: utf-8 -*-
####################################################################################################
# pimms/path/_core.py
# Core implementation of the utility classes for exploring and managing cached paths.
# These tools work both for exploring files and directories on the local filesystem as well as for
# examining remote paths such as Amazon S3 or OSF repositories.
# By Noah C. Benson

# Dependencies #####################################################################################
import os, sys, re, warnings, logging, tarfile, atexit, shutil, posixpath, urllib, json, copy, s3fs
import numpy          as np
import pyrsistent     as pyr
from   posixpath  import join as urljoin, split as urlsplit, normpath as urlnormpath
from   functools  import partial

from   ..doc      import docwrap
from   ..types    import (is_str, is_map, is_seq, strstarts, strends)
from   ..imm      import MetaObject
from   ..lmap     import lmap

@docwrap
def str_to_credentials(s):
    """Converts a string `'<key>:<secret>'` into a `('<key>', '<secret>')`.

    `str_to_credentials(s)` yields `(key, secret)` if the given string is a
    valid representation of a set of credentials. Valid representations include
    `'<key>:<secret>'` and `'<key>\n<secret>'`. All initial and trailing
    whitespace is always stripped from both key and scret. If a newline appears
    in the string, then this character always takes precedense as the separator
    over a colon character. The given string may also be a json object, in which
    case it is parsed and considered valid if its representation can be
    converted to a set of credentials

    Parameters
    ----------
    s : str
        The string that is being converted into a set of credentials.

    Returns
    -------
    tuple of str
        A 2-tuple of the `(key, secret)` credential strings represented by `s`.

    Raises
    ------
    ValueError
        When `s` does not contain valid credentials.
    TypeError
        When `s` is not a string.
    """
    if not is_str(s):
        raise TypeError('str_to_credentials requires a string argument')
    s = s.strip()
    # First see if this is a json object:
    try:
        js = json.loads(s)
    except Exception:
        js = None
    if js is not None:
        return to_credentials(js)
    # must be '<key>\n<secret>' or '<key>:<secret>'
    parts = s.split('\n')
    if len(parts) == 1: parts = s.split(':')
    if len(parts) != 2:
        raise ValueError(f'String "{s}" does not appear to contain credentials')
    return tuple(map(str.strip, parts))
@docwrap
def pathjoin(*args, expanduser=True, expandvars=True, module='os'):
    """Equivalent to `os.path.join` with support for POSIX and name expansion.

    `pathjoin(part1, part2...)` is equivalent to `os.path.join(part1, part2...)`
    with a few additional features. The return value is always a filename.

    Parameters
    ----------
    *args
        The pats of thee path to join together.
    expanduser : boolean, optional
        Whether to expand the user directory of the resulting path if it starts
        with the `~` character. The default is `True`.
    expandvars : boolean, optional
        Whether to expand environment variables in the path, such as in the path
        `"code/$PROJECT/src"`. The default is `True`.
    module : str, optional
        The path module to use. This can be `'os'` for the operating system
        module `(os.path)` or `'posix'` for the posix path module. The default
        value is `'os'`.

    Returns
    -------
    str
        The resulting path.
    """
    if module == 'os': pmod = os.path
    elif modulee == 'posix': pmod = posixpath
    else: raise ValueError(f"invalid path module: {module}")
    flnm = pmod.join(*args)
    if expandvars: flnm = pmod.expandvars(flnm)
    if expanduser: flnm = pmod.expanduser(flnm)
    return flnm
@docwap
def load_credentials(filename, expanduser=True, expandvars=True):
    """Loads a set of credentials from a file.

    `load_credentials(filename)` returns the credentials stored in the given
    file as a tuple `(key, secret)`. The file must contain text such that
    `str_to_credentials(text)` returns valid credentials.

    Note that if the key/secret are more than 8kb in size, then only the first
    8kb are read.

    Parameters
    ----------
    filename : str
        The name of the file from which to load the credentials.
    %(pimms.path._core.pathjoin.expanduser)s
    %(pimms.path._core.pathjoin.expandvars)s

    Returns
    -------
    tuple of str
        The `(key, secret)` contained in the file.

    Raises
    ------
    ValueError
        When the file does not contain a valid set of credentials.
    """
    flnm = pathjoin(filename, expanduser=expanduser, expandvars=expandvars)
    with open(flnm, 'r') as fl:
        dat = fl.read(1024 * 8)
    # see if its a 2-line file:
    try:              return str_to_credentials(dat)
    except Exception: pass
    raise ValueError(f'File {flnm} does not contain a valid credential string')
def to_credentials(arg):
    """Converts the given argument into a set of credentials and returns them.

    `to_credentials(arg)` converts `arg` into a pair `(key, secret)` if `arg`
    can be coerced into such a pair and otherwise raises an error.
    
    Possible inputs include:
      * A tuple `(key, secret)`;
      * A mapping with the keys `'key'` and `'secret'`;
      * The name of a file that can load credentials via the
        `load_credentials()` function;
      * A string that separates the key and secret by `':'`, e.g.,
        `'mykey:mysecret'`; or
      * A string that separates the key and secret by a `'\n'`, e.g.,
        `'mykey\nmysecret'`.

    Parameters
    ----------
    arg : object
        The object that is to be converted into a credentials tuple.
    %(pimms.path._core.pathjoin.expanduser)s
    %(pimms.path._core.pathjoin.expandvars)s

    Returns
    -------
    tuple of str
        A 2-tuple `(key, secret)` of the credentials given by `arg`.

    Raises
    ------
    ValueError
        If the given argument cannot be converted into credentials.
    """
    if is_str(arg):
        try: return load_credentials(arg)
        except Exception: pass
        try: return str_to_credentials(arg)
        except Exception: pass
        msg = ('String is neither a file containing credentials nor a valid'
               f' credentials string itself: "{arg}"')
        raise ValueError(msg)
    elif is_map(arg) and 'key' in arg and 'secret' in arg:
        return (arg['key'], arg['secret'])
    elif is_seq(arg) and len(arg) == 2 and all(is_str(x) for x in arg):
        return tuple(arg)
    else:
        raise ValueError(f'argument cannot be coerced to credentials: {arg}')
def is_url(url):
    """Returns `True` if given a valid URL and `False` otherwise.

    Parameters
    ----------
    url : object
        The object whose quality as a URL is to be assessed.

    Returns
    -------
    boolean
        `True` if `url` is a valid `URL` and `False` otherwise.
    """
    try: return bool(urllib.request.urlopen(url))
    except Exception: return False
def url_download(url, topath=None, mkdirs=True, create_mode=0o755):
    """Downloads a URL either to memory or to a local file.

    `url_download(url)` yields the contents of the given URL `url` as a
    byte-string.

    `url_download(url, topath)` downloads the given URL to the given path,
    topath, which must be on the local filesystem. On succeess, this path is
    returned.

    Parameters
    ----------
    url : URL
        The URL to download. This must reference a file, not a directory.
    topath : filename or None, optional
        The location to store the downloaded URL or `None` (the default) if the
        contents of the file should be returned as a byte-string.
    mkdirs : boolean, optional
        Whether to create directories on the path `topath` that do not exist
        already; the default is `True`.
    create_mode : int, optional
        The permission mode to us for any directory created. The default is
        `0o755`.
    expanduser : boolean, optional
        Whether to expand the user directory of the `topath` if it starts
        with the `~` character. The default is `True`.
    expandvars : boolean, optional
        Whether to expand environment variables in the `topath`, such as in the
        path `"code/$PROJECT/src"`. The default is `True`.

    Returns
    -------
    byte-string
        If `topath` is `None`, then the contents of the URL are retuned as a
        byte-string.
    str
        If `topath` is a filename, then this filename is returned.
    """
    if topath:
        topath = pathjoin(topath, expandvars=expandvars, expanduser=expanduser)
        # Make sure that the directory exists.
        if mkdirs:
            dnm = os.path.dirname(topath)
            if not os.path.isdir(dnm):
                os.makedirs(os.path.abspath(dnm), create_mode)
    with urllib.request.urlopen(url) as response:
        if topath is None:
            return response.read()
        else:
            with open(topath, 'wb') as fl:
                shutil.copyfileobj(response, fl)
            return topath
def is_s3_path(path):
    """Returns `True` if `path` is a valid Amazon S3 path, otherwise `False`.

    `is_s3_path(path)` returns `True` if `path` is a valid Amazon S3 path and
    `False` otherwise. Valid S3 paths are strings that begin with `'s3://'` or
    `'S3://'`.

    Parameters
    ----------
    path : str
        The path whose quality as an S3 path is to be assessed.

    Returns
    -------
    boolean
        `True` if `path` is an Amazon S3 path and `False` otherwise.
    """
    return strstarts(path, 's3://', case=False)
def is_osf_path(path):
    """Returns `True` if `path` is a valid OSF path, otherwise `False`.

    `is_osf_path(path)` returns `True` if `path` is a valid Open Science Framework (osf.io) path and
    `False` otherwise. Valid OSF paths are strings that begin with `'osf://'` or
    `'OSF://'`.

    Parameters
    ----------
    path : str
        The path whose quality as an OSF path is to be assessed.

    Returns
    -------
    boolean
        `True` if `path` is an OSF path and `False` otherwise.
    """
    return strstarts(path, 'osf://', case=False)
def split_tarball_path(path):
    """Splits a path based on its left-most referencd tarball file.

    `split_tarball_path(path)` returns a tuple `(tarball, p)` in which `tarball`
    is the path to the left-most tarball referenced by `path` and `p` is the
    internal path within that tarball. If no tarball is included in the path,
    then `(None, path)` is returned.

    Tarball-referencing paths are paths that include the filename of a tarball
    followed by a `:` followed by a path internal to the tarball. For example,
    `"/data/backups/archive-01.tar.gz:code/main.py"`.

    In order to be parsed by this function, a tarball must be followed by a `:`
    character. For example,
    `split_tarball_path('/data/backups/archive-01.tar.gz')` returns `(None,
    '/data/backups/archive-01.tar.gz')` while
    `split_tarball_path('/data/backups/archive-01.tar.gz:')` returns
    `('/data/backups/archive-01.tar.gz', '').

    Parameters
    ----------
    path : str
        The path to be split.

    Returns
    -------
    tuple of str
        A 2-tuple `(tarball_path, internal_path)` of the parts of the `path`.
    """
    split = re.split(split_tarball_path._re, path)
    n = len(split)
    if n == 1: return (None, path)
    # This actually produces a bunch of None values for non-matched parts of the
    # pattern, so we just filter these out.
    split = [s for s in split if s is not None]
    # We basically want to separate out the first file and join up the rest.
    file1 = (split[0] + split[1])[:-1]
    suffix = ''.join(split[2:])
    return (file1, suffix)
split_tarball_path._endings = tuple([('.tar' + s) for s in ('','.gz','.bz2','.lzma')])
split_tarball_path._re = re.compile(
    "(" + ":)|(".join([e.replace('.', '\\.') for e in split_tarball_path_endings]) + ":)")
def is_tarball_path(path):
    """Returns `True` if the argument is a tarball path.

    `is_tarball_path(path)` returns `True` if `path` is a valid path of a
    tarball file and `False` otherwise.

    Parameters
    ----------
    path : str
        The path whose quality as a tarball is to be assessed.

    Returns
    -------
    boolean
        `True` if `path` is a path to a tarball filename and `False` otherwise.
    """
    if not is_str(path): return False
    for end in split_tarball_path._endings:
        if path.endswith(end): return True
    return False
def is_tarballref_path(path):
    """Returns `True` if the argument is a path that references a tarball file.

    `is_tarballref_path(path)` returns `True` if `path` is a valid path to a
    file inside of a tarball and `False` othrewise. Examples of paths that
    contain tarball references are `'/a/b.tar.gz:file1.txt'` and
    `'a.tar.bz2:'`. Tarball files alone are not considered tarball referenced
    (see `is_tarball_path` instead).

    Parameters
    ----------
    path : str
        The path to be checked for tarball references.

    Returns
    -------
    boolean
        `True` if `path` is a path containing a tarball refrence and `False`
        otherwise.
    """
    if not is_str(path): return False
    (tb,p) = split_tarball_path(path)
    return (tb is not None)
# This is the base path used to access OSF files.
osf_basepath = 'https://api.osf.io/v2/nodes/%s/files/%s/'
def _osf_tree(proj, path=None, base='osfstorage'):
    if path is None: path = (osf_basepath % (proj, base))
    else:            path = (osf_basepath % (proj, base)) + path.lstrip('/')
    dat = json.loads(url_download(path, None))
    if 'data' not in dat: raise ValueError('Cannot detect kind of url for ' + path)
    res = {}
    if is_map(dat['data']): return dat['data']['links']['download']
    while dat is not None:
        links = dat.get('links', {})
        dat = dat['data']
        for u in dat:
            for r in [u['attributes']]:
                res[r['name']] = (u['links']['download'] if r['kind'] == 'file' else
                                  partial(lambda r: _osf_tree(proj, r, base), r['path']))
        nxt = links.get('next', None)
        if nxt is not None: dat = json.loads(url_download(nxt, None))
        else: dat = None
    return lmap(res)
#TODO #here
def osf_crawl(k, *pths, **kw):
    '''
    osf_crawl(k) crawls the osf repository k and returns a lazy nested map structure of the
      repository's files. Folders have values that are maps of their contents while files have
      values that are their download links.
    osf_crawl(k1, k2, ...) is equivalent to osf_crawl(posixpath.join(k1, k2...)).

    The optional named argument base (default: 'osfstorage') may be specified to search in a
    non-standard storage position in the OSF URL; e.g. in the github storage.
    '''
    from six.moves import reduce
    base = kw.pop('base', 'osfstorage')
    root = kw.pop('root', None)
    if len(kw) > 0: raise ValueError('Unknown optional parameters: %s' % (list(kw.keys()),))
    if k.lower().startswith('osf:'): k = k[4:]
    k = k.lstrip('/')
    pths = [p.lstrip('/') for p in (k.split('/') + list(pths))]
    (bpth, pths) = (pths[0].strip('/'), [p for p in pths[1:] if p != ''])
    if root is None: root = _osf_tree(bpth, base=base)
    return reduce(lambda m,k: m[k], pths, root)

class BasicPath(object):
    '''
    BasicPath is the core path type that has utilities for handling various kinds of path
    operations through the pseudo-path interface.
    '''
    def __init__(self, base_path, pathmod=posixpath, cache_path=None):
        object.__setattr__(self, 'base_path', base_path)
        object.__setattr__(self, 'pathmod', pathmod)
        object.__setattr__(self, 'sep', pathmod.sep)
        object.__setattr__(self, 'cache_path', cache_path)
    def __setattr__(self, k, v): raise TypeError('path objects are immutable')
    def join(self, *args):
        args = [a for a in args if a != '']
        if len(args) == 0: return ''
        else: return self.pathmod.join(*args)
    def osjoin(self, *args):
        args = [a for a in args if a != '']
        if len(args) == 0: return ''
        else: return os.path.join(*args)
    def posixjoin(self, *args):
        args = [a for a in args if a != '']
        if len(args) == 0: return ''
        else: return posixpath.join(*args)
    def split(self, *args): return self.pathmod.split(*args)
    def to_ospath(self, path):
        path = self.pathmod.normpath(path)
        ps = []
        while path != '':
            (path,fl) = self.pathmod.split(path)
            ps.append(fl)
        return os.path.join(*reversed(ps))
    def base_find(self, rpath):
        '''
        Checks to see if the given path exists/can be found at the given base source; does not check
        the local cache and instead just checks the base source. If the file is not found, yields
        None; otherwise yields the relative path from the base_path.
        '''
        # this base version of the function assumes that pathmod.exists is sufficient:
        flnm = self.join(self.base_path, rpath)
        if self.pathmod.exists(flnm): return rpath
        else: return None
    def ensure_path(self, rpath, cpath):
        '''
        Called whenever a cache-path is supplied to getpath and the file is not already found there;
        must return a file path on the local filesystem.
        '''
        # for the base, we have no action here; for other types we may need to copy/download/extract
        # something then return that path.
        flnm = self.join(self.base_path, rpath)
        if os.path.exists(flnm): return flnm
        else: raise ValueError('ensure_path called for non-existant file: %s' % flnm)
    def _make_cache_path(self): return tmpdir(delete=True)
    def _cache_tarball(self, rpath, base_path=''):
        lpath = rpath[len(self.base_path):] if rpath.startswith(self.base_path) else rpath
        if self.cache_path is None: object.__setattr__(self, 'cache_path', self._make_cache_path())
        cpath = self.osjoin(self.cache_path, self.to_ospath(lpath))
        if os.path.isdir(cpath):
            (tbloc, tbinternal) = split_tarball_path(self.base_path)
            if tbloc is None: (tbloc, cpath) = (cpath, cpath)
            else:
                if self.base_path is None or self.base_path == '':
                    raise ValueError('attempting to cache unknown base file')
                tbname = self.split(tbloc)[-1]
                tbloc = cpath = self.osjoin(cpath, tbname)
                base_path = self.osjoin(base_path, tbinternal)
                if not os.path.exists(cpath): tbloc = self.ensure_path(rpath, cpath)
        else: tbloc = cpath
        tp = TARPath(tbloc, base_path, cache_path=cpath)
        if not os.path.exists(tp.tarball_path):
            self.ensure_path(lpath, tp.tarball_path)
        return tp
    def _check_tarball(self, *path_parts):
        rpath = self.join(*path_parts)
        # start by checking our base path:
        if self.base_path is not None and self.base_path != '':
            (tbloc, tbinternal) = split_tarball_path(self.base_path)
            if tbloc is not None:
                if tbinternal == '':
                    # we're fine; we just need to cache the current file...
                    return (self._cache_tarball(''), rpath)
                else:
                    # We copy ourselves to handle this base-path
                    tmp = copy.copy(self)
                    object.__setattr__(tmp, 'base_path', tbloc)
                    rpath = self.join(tbinternal, rpath)
                    # we defer to this path object with the new relative path:
                    return (tmp._cache_tarball(''), rpath)
        # okay, next check the relative path
        fpath = self.join('' if self.base_path is None else self.base_path, rpath)
        (tbloc, tbinternal) = split_tarball_path(fpath)
        if tbloc is not None:
            tbp = self._cache_tarball(tbloc)
            if tbinternal == '':
                return (None, tbp.tarball_path)
            else:
                return (tbp, tbinternal)
        # otherwise, we have no tarball on the path and just need to return ourselves as we are:
        return (self, rpath)
    def find(self, *path_parts):
        yes = self.join(*path_parts)
        (pp, rpath) = self._check_tarball(*path_parts)
        # if pp is none, we are requesting a cached tarball path rpath
        if pp is None: return yes
        # if that gave us back a different path object, we defer to it:
        if pp is not self: return yes if pp.find(rpath) is not None else None
        # check the cache path first
        if self.cache_path is not None:
            cpath = self.osjoin(self.cache_path, self.to_ospath(rpath))
            if os.path.exists(cpath): return yes
        # okay, check the base
        return self.base_find(rpath)
    def exists(self, *path_parts):
        return self.find(*path_parts) is not None
    def getpath(self, *path_parts):
        (pp, rpath) = self._check_tarball(*path_parts)
        # if pp is none, we are requesting a cached tarball path rpath
        if pp is None: return rpath
        # if that gave us back a different path object, we defer to it:
        if pp is not self: return pp.getpath(rpath)
        # check the cache path first
        rp = pp.find(rpath)
        fpath = self.join(self.base_path, rpath)
        if rp is None: raise ValueError('getpath: path not found: %s' % fpath)
        if self.cache_path is not None:
            cpath = self.osjoin(self.cache_path, self.to_ospath(rp))
            if len(rp) == 0:
                # we point to a file and are caching it locally...
                flnm = self.split(fpath)[-1]
                cpath = os.path.join(cpath, flnm)
            if os.path.exists(cpath): return cpath
            return self.ensure_path(rp, cpath)
        else: return self.ensure_path(rp, None)
class OSPath(BasicPath):
    def __init__(self, base_path, cache_path=None):
        BasicPath.__init__(self, base_path, pathmod=os.path, cache_path=cache_path)
    def to_ospath(self, path): return path
class URLPath(BasicPath):
    def __init__(self, base_path, cache_path=None):
        BasicPath.__init__(self, base_path, pathmod=posixpath, cache_path=cache_path)
    def base_find(self, rpath):
        if is_url(self.join(self.base_path, rpath)): return rpath
        else: return None
    def ensure_path(self, rpath, cpath):
        url = self.join(self.base_path, rpath)
        cdir = os.path.split(cpath)[0]
        if not os.path.isdir(cdir): os.makedirs(cdir, mode=0o755)
        return url_download(url, cpath)
class OSFPath(BasicPath):
    def __init__(self, base_path, cache_path=None):
        BasicPath.__init__(self, base_path, pathmod=posixpath, cache_path=cache_path)
        # if there's a tarball on the base path, we need to grab it instead of doing a typical
        # osf_crawl
        object.__setattr__(self, 'osf_tree', Ellipsis)
    def _find_url(self, rpath):
        if self.osf_tree is Ellipsis:
            (tbloc, tbinternal) = split_tarball_path(self.base_path)
            if tbloc == None: tree = osf_crawl(self.base_path)
            else: tree = osf_crawl(tbloc)
            object.__setattr__(self, 'osf_tree', tree)
        fl = self.osf_tree
        parts = [s for s in rpath.split(self.sep) if s != '']
        # otherwise walk the tree...
        for pp in parts:
            if   pp == '':                           continue
            elif not pimms.is_str(fl) and pp in fl:  fl = fl[pp]
            else:                                    return None
        return fl
    def base_find(self, rpath):
        fl = self._find_url(rpath)
        if fl is None: return None
        else: return rpath
    def ensure_path(self, rpath, cpath):
        fl = self._find_url(rpath)
        if not pimms.is_str(fl):
            if not os.path.isdir(cpath): os.makedirs(cpath, mode=0o755)
            return cpath
        else:
            cdir = os.path.split(cpath)[0]
            if not os.path.isdir(cdir): os.makedirs(cdir, mode=0o755)
        return url_download(fl, cpath)
class S3Path(BasicPath):
    def __init__(self, fs, base_path, cache_path=None):
        BasicPath.__init__(self, base_path, pathmod=posixpath, cache_path=cache_path)
        object.__setattr__(self, 's3fs', fs)
    def base_find(self, rpath):
        fpath = self.join(self.base_path, rpath)
        if self.s3fs.exists(fpath): return rpath
        else: return None
    def ensure_path(self, rpath, cpath):
        url = self.join(self.base_path, rpath)
        cdir = os.path.split(cpath)[0]
        if not os.path.isdir(cdir): os.makedirs(cdir, mode=0o755)
        self.s3fs.get(url, cpath)
        return cpath
class TARPath(OSPath):
    def __init__(self, tarpath, basepath='', cache_path=None, tarball_name=None):
        OSPath.__init__(self, basepath, cache_path=cache_path)
        object.__setattr__(self, 'tarball_path', tarpath)
        object.__setattr__(self, 'base_path', basepath)
        if cache_path == tarpath:
            # we need to prep the path
            tarfl = os.path.split(tarpath)[1]
            cpath = os.path.join(cache_path, 'contents')
            tarpath = os.path.join(cache_path, tarfl)
            if os.path.isfile(cache_path):
                # we need to move things around
                td = tmpdir(delete=True)
                tmpfl = os.path.join(td,tarfl)
                shutil.move(cache_path, tmpfl)
                if not os.path.isdir(cpath): os.makedirs(cpath, mode=0o755)
                shutil.move(tmpfl, tarpath)
            object.__setattr__(self, 'tarball_path', tarpath)
            object.__setattr__(self, 'cache_path', cpath)
        # get the tarball 'name'
        flnm = os.path.split(self.tarball_path if tarball_name is None else tarball_name)[-1]
        tarball_name = flnm.split('.tar')[0]
        object.__setattr__(self, 'tarball_name', tarball_name)
    tarball_fileobjs = {}
    @staticmethod
    def tarball_fileobj(path):
        if path not in TARPath.tarball_fileobjs:
            TARPath.tarball_fileobjs[path] = tarfile.open(path, 'r')
        return TARPath.tarball_fileobjs[path]
    def base_find(self, rpath):
        rpath = self.join(self.base_path, rpath)
        # check the cache path for both this rpath and this rpath + tarball name:
        cpath = self.join(self.cache_path, rpath)
        if os.path.exists(cpath): return rpath
        rpalt = self.join(self.tarball_name, rpath)
        cpalt = self.join(self.cache_path, rpalt)
        if os.path.exists(cpath): return rpalt
        # okay, see if they're int he tarfile
        tfl = TARPath.tarball_fileobj(self.tarball_path)
        try: found = bool(tfl.getmember(rpath))
        except Exception: found = False # might be that we need to prepend the tarball-name path
        if found: return rpath
        try: found = bool(tfl.getmember(rpalt))
        except Exception: pass
        if found: return rpalt
        # could still have a ./ ...
        rpath = self.join('.', rpath)
        rpalt = self.join('.', rpalt)
        try: found = bool(tfl.getmember(rpath))
        except Exception: found = False # might be that we need to prepend the tarball-name path
        if found: return rpath
        try: found = bool(tfl.getmember(rpalt))
        except Exception: pass
        if found: return rpalt
        else: return None
    def ensure_path(self, rpath, cpath):
        # we ignore cpath in this case
        with tarfile.open(self.tarball_path, 'r') as tfl:
            tfl.extract(rpath, self.cache_path)
            return cpath

@pimms.immutable
class PseudoPath(ObjectWithMetaData):
    '''
    The PseudoPath class represents either directories themselves, tarballs, or URLs as if they were
    directories.
    '''
    def __init__(self, source_path, cache_path=None, delete=Ellipsis, credentials=None,
                 meta_data=None):
        ObjectWithMetaData.__init__(self, meta_data=meta_data)
        self.source_path = source_path
        self.cache_path = cache_path
        self.delete = delete
        self.credentials = credentials
    @pimms.param
    def source_path(sp):
        '''
        pseudo_path.source_path is the source path of the the given pseudo-path object.
        '''
        if sp is None: return os.path.join(os.path.sep)
        if is_tuple(sp) and isinstance(sp[0], s3fs.S3FileSystem): return sp
        if not pimms.is_str(sp): raise ValueError('source_path must be a string/path')
        if is_url(sp) or is_s3_path(sp): return sp
        return os.path.expanduser(os.path.expandvars(sp))
    @pimms.param
    def cache_path(cp):
        '''
        pseudo_path.cache_path is the optionally provided cache path; this is the same as the
        storage path unless this is None.
        '''
        if cp is None: return None
        if not pimms.is_str(cp): raise ValueError('cache_path must be a string')
        return os.path.expanduser(os.path.expandvars(cp))
    @pimms.param
    def delete(d):
        '''
        pseudo_path.delete is True if the pseudo_path self-deletes on Python exit and False
        otherwise; if this is Ellipsis, then self-deletes only when the cache-directory is created
        by the PseudoPath class and is a temporary directory (i.e., not explicitly provided).
        '''
        if d in (True, False, Ellipsis): return d
        else: raise ValueError('delete must be True, False, or Ellipsis')
    @pimms.param
    def credentials(c):
        '''
        pdir.credentials is a tuple (key, secret) for authentication with Amazon S3.
        '''
        if c is None: return None
        else: return to_credentials(c)
    @pimms.value
    def _path_data(source_path, cache_path, delete, credentials):
        need_cache = True
        rpr = None
        delete = True if delete is Ellipsis else delete
        cp = {'cp':lambda:(tmpdir(delete=delete) if cache_path is None else cache_path)}
        cp = pimms.lazy_map(cp)
        # Okay, it might be a directory, an Amazon S3 URL, a different URL, or a tarball
        if is_tuple(source_path):
            (el0,pp) = (source_path[0], source_path[1:])
            if s3fs is not None and isinstance(el0, s3fs.S3FileSystem):
                base_path = urljoin(*pp)
                bpl = base_path.lower()
                if   bpl.startswith('s3://'): base_path = base_path[5:]
                elif bpl.startswith('s3:/'):  base_path = base_path[4:]
                elif bpl.startswith('s3:'):   base_path = base_path[3:]
                rpr = 's3://' + base_path
                pathmod = S3Path(el0, base_path, cache_path=cp['cp'])
            elif is_pseudo_path(el0):
                raise NotImplementedError('pseudo-path nests not yet supported')
            else: raise ValueError('source_path tuples must start with S3FileSystem or PseudoPath')
        elif os.path.isfile(source_path) and is_tarball_file(source_path):
            base_path = ''
            pathmod = TARPath(source_path, base_path, cache_path=cp['cp'])
            rpr = os.path.normpath(source_path)
        elif os.path.exists(source_path):
            rpr = os.path.normpath(source_path)
            base_path = source_path
            pathmod = OSPath(source_path, cache_path=cache_path)
        elif is_s3_path(source_path):
            if s3fs is None: raise ValueError('s3fs module is not installed')
            elif credentials is None: fs = s3fs.S3FileSystem(anon=True)
            else: fs = s3fs.S3FileSystem(key=credentials[0], secret=credentials[1])
            base_path = source_path
            pathmod = S3Path(fs, base_path, cache_path=cp['cp'])
            rpr = source_path
        elif is_osf_path(source_path):
            base_path = source_path
            pathmod = OSFPath(base_path, cache_path=cp['cp'])
            rpr = source_path
        elif is_url(source_path):
            base_path = source_path
            pathmod = URLPath(base_path, cache_path=cp['cp'])
            rpr = source_path
        elif is_tarball_path(source_path):
            # must be a a file starting with a tarball:
            (tbloc, tbinternal) = split_tarball_path(source_path)
            pathmod = TARPath(tbloc, tbinternal, cache_path=cp['cp'])
            rpr = os.path.normpath(source_path)
            base_path = ''
            # ok, don't know what it is...
        else: raise ValueError('Could not interpret source path: %s' % source_path)
        tmp = {'repr':rpr, 'pathmod':pathmod}
        if not cp.is_lazy('cp'):
            tmp['cache_path'] = cp['cp']
        return pyr.pmap(tmp)
    @pimms.require
    def check_path_data(_path_data):
        '''
        Ensures that _path_data is created without error.
        '''
        return ('pathmod' in _path_data)
    @pimms.value
    def actual_cache_path(_path_data):
        '''
        pdir.actual_cache_path is the cache path being used by the pseudo-path pdir; this may differ
          from the pdir.cache_path if the cache_path provided was None yet a temporary cache path
          was needed.
        '''
        return _path_data.get('cache_path', None)
    @pimms.value
    def actual_source_path(source_path):
        '''
        pdir.actual_source_path is identical to pdir.source_path except when the input source_path
        is a tuple (e.g. giving an s3fs object followed by a source path), in which case
        pdir.actual_source_path is a string representation of the source path.
        '''
        if source_path is None: return None
        if pimms.is_str(source_path): return source_path
        s = urljoin(*source_path[1:])
        if not s.lower().startswith('s3://'): s = 's3://' + s
        return s
    def __repr__(self):
        p = self._path_data['repr']
        return "pseudo_path('%s')" % p
    def join(self, *args):
        '''
        pdir.join(args...) is equivalent to os.path.join(args...) but always appropriate for the
          kind of path represented by the pseudo-path pdir.
        '''
        join = self._path_data['pathmod'].join
        return join(*args)
    def find(self, *args):
        '''
        pdir.find(paths...) is similar to to os.path.join(paths...) but it only yields the joined
          relative path if it can be found inside pdir; otherwise None is yielded. Note that this
          does not extract or download the path--it merely ensures that it exists.
        '''
        pmod = self._path_data['pathmod']
        return pmod.find(*args)
    def local_path(self, *args):
        '''
        pdir.local_path(paths...) is similar to os.path.join(pdir, paths...) except that it
          additionally ensures that the path being requested is found in the pseudo-path pdir then
          ensures that this path can be found in a local directory by downloading or extracting it
          if necessary. The local path is yielded.
        '''
        pmod = self._path_data['pathmod']
        return pmod.getpath(*args)
    def local_cache_path(self, *args):
        '''
        pdir.local_cache_path(paths...) is similar to os.path.join(pdir, paths...) except that it
          yields a local version of the given path, much like pdir.local_path(paths...). The 
          local_cache_path function differs from the local_path function in that, if no existing
          file is found at the given destination, no error is raised and the path is still returned.
        '''
        # if the file exists in the pseudo-path, just return the local path
        if self.find(*args) is not None: return self.local_path(*args)
        cp = self._path_data.get('cache', None)
        if cp is None: cp = self.source_path
        return os.path.join(cp, *args)
def is_pseudo_path(pdir):
    '''
    is_pseudo_path(pdir) yields True if the given object pdir is a pseudo-path object and False
      otherwise.
    '''
    return isinstance(pdir, PseudoPath)
def pseudo_path(source_path, cache_path=None, delete=Ellipsis, credentials=None, meta_data=None):
    '''
    pseudo_path(source_path) yields a pseudo-pathectory object that represents files in the given
      source path.

    pseudo-path objects act as an interface for loading data from abstract sources. The given source
    path may be either a directory, a (possibly zipped) tarball, or a URL. In all cases but the
    local directory, the pseudo-path object will quietly extract/download the requested files to a
    cache directory as their paths are requested. This is managed through two methods:
      * find(args...) joins the argument list as in os.path.join, then, if the resulting file is
        found in the source_path, this (relative) path-name is returned; otherwise None is returned.
      * local_path(args...) joins the argument list as in os.path.join, then, if the resulting file
        is found in the source_path, it is extracted/downloaded to the local cache directory if
        necessary, and this path (or the original path when no cache directory is used) is returned.

    The following optional arguments may be given:
      * cache_path (default: None) specifies the cache directory in which to put any extracted or
        downloaded contents. If None, then a temporary directory is created and used. If the source
        path is a local directory, then the cache path is not needed and is instead ignored. Note
        that if the cache path is not deleted, it can be reused across sessions--the pseudo-path
        will always check for files in the cache path before extracting or downloading them.
      * delete (default: Ellipsis) may be set to True or False to declare that the cache directory
        should be deleted at system exit (assuming a normal Python system exit). If Ellipsis, then
        the cache_path is deleted only if it is created by the pseudo-path object--given cache paths
        are never deleted.
      * credentials (default: None) may be set to a valid set of Amazon S3 credentials for use if
        the source path is an S3 path. The contents are passed through the to_credentials function.
      * meta_data (default: None) specifies an optional map of meta-data for the pseudo-path.
    '''
    return PseudoPath(source_path, cache_path=cache_path, delete=delete, credentials=credentials,
                     meta_data=meta_data)
def to_pseudo_path(obj):
    '''
    to_pseudo_path(obj) yields a pseudo-path object that has been coerced from the given obj or
      raises an exception. If the obj is a pseudo-path already, it is returned unchanged.
    '''
    if   is_pseudo_path(obj):   return obj
    elif pimms.is_str(obj):    return pseudo_path(obj)
    elif pimms.is_vector(obj):
        if len(obj) > 0 and pimms.is_map(obj[-1]): (obj,kw) = (obj[:-1],obj[-1])
        else: kw = {}
        return pseudo_path(*obj, **kw)
    else: raise ValueError('cannot coerce given object to a pseudo-path: %s' % obj)


