# -*- coding: utf-8 -*-
################################################################################
# pimms/pathlib/__init__.py

"""Core implementation of the classes for exploring and managing cached paths.

These tools work both for exploring files and directories on the local
filesystem as well as for examining remote paths such as Amazon S3 or OSF
repositories. The `pimms` library builds on the `pathlib` library along with the
`cloudpath` library to provide straightforward interface using the `PimmsPath`
type.
"""

from ._osf  import (
    OSFClient,
    OSFPath,
)
from ._core import (
    pathstr,
    filepath,
    s3path,
    gspath,
    azpath,
    osfpath,
    path,
    is_filepath,
    is_s3path,
    is_azpath,
    is_gspath,
    is_osfpath,
    is_path,
    like_filepath,
    like_s3path,
    like_azpath,
    like_gspath,
    like_osfpath,
    like_path,
    # Symbols that aren't exported via __all__:
    PathTypeRecord,
    pathtypes,
    pathtype,
)
__all__ = [
    "OSFClient",
    "OSFPath",
    "pathstr",
    "filepath",
    "s3path",
    "gspath",
    "azpath",
    "osfpath",
    "path",
    "is_filepath",
    "is_s3path",
    "is_azpath",
    "is_gspath",
    "is_osfpath",
    "is_path",
    "like_filepath",
    "like_s3path",
    "like_azpath",
    "like_gspath",
    "like_osfpath",
    "like_path",
]
