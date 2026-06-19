'''
socutils.lib -- bundled C/C++ shared libraries and their ctypes loaders.

All libraries are built by ``lib/CMakeLists.txt`` (a thin pyscf/lib-style
top-level CMake) and emitted flat into this directory:

  libx2camf_c.so   X2CAMF spin-orbit integrals   (exposed as ``import x2camf``)
  libccsdt_clib.so spinor CCSDT residual kernels  (socutils.cc)
  libzquatev.so    quaternion eigensolver         (Kramers-restricted SCF)

``load_library`` mirrors ``pyscf.lib.load_library``: it loads a library by name
from this directory.  A ``SOCUTILS_<NAME>_LIBRARY`` environment variable (e.g.
``SOCUTILS_ZQUATEV_LIBRARY``) overrides the search with an explicit path.
'''

import os
import ctypes


def library_dir():
    '''Absolute path of this directory (where the compiled .so files live).'''
    return os.path.dirname(os.path.abspath(__file__))


def load_library(libname, env_var=None):
    '''Load a ctypes CDLL by bare name (e.g. ``'libzquatev'``) from socutils/lib.

    If ``env_var`` is given and set in the environment, that path is tried
    first.  Returns the loaded CDLL, or raises OSError if it cannot be found.
    '''
    candidates = []
    if env_var:
        override = os.environ.get(env_var)
        if override:
            candidates.append(override)
    here = library_dir()
    for ext in ('.so', '.dylib', '.dll'):
        candidates.append(os.path.join(here, libname + ext))
    last_err = None
    for path in candidates:
        if os.path.sep in path and not os.path.exists(path):
            continue
        try:
            return ctypes.CDLL(path)
        except OSError as err:
            last_err = err
    raise OSError('Cannot load %s from %s. Build it with `make` (or '
                  'lib/build.sh). Last error: %s' % (libname, here, last_err))
