import os as _os
import sys as _sys

# Expose the bundled x2camf interface (a dual-backend dispatcher: the pure-C
# reimplementation, default, plus the upstream pybind11 reference) as the
# top-level ``x2camf`` package.
#
# The bundled package is the default even if an external ``x2camf`` is
# installed: the Python layer of the external package places the atomic
# matrices by position and is wrong for basis sets whose primitives are not
# ordered as in gto.mole.uncontracted_basis (segmented contractions, added
# functions, shells not sorted by l).
#
# Control with the environment variable SOCUTILS_X2CAMF:
#   bundled   (default) always use the in-tree dispatcher
#   auto      bundled only if no external x2camf is installed (old default)
#   external  never touch sys.path (use whatever x2camf is installed)
def _setup_x2camf():
    bundled = _os.path.join(_os.path.dirname(__file__), 'x2camf_c')
    if not _os.path.isdir(bundled):
        return
    mode = _os.environ.get('SOCUTILS_X2CAMF', 'bundled').strip().lower()
    if mode == 'external':
        return
    if mode == 'auto':
        import importlib.util
        if importlib.util.find_spec('x2camf') is not None:
            return  # an external x2camf is installed; leave it in charge
    if bundled not in _sys.path:
        _sys.path.insert(0, bundled)
    loaded = _sys.modules.get('x2camf')
    if loaded is not None and not str(getattr(loaded, '__file__', '')).startswith(bundled):
        import warnings
        warnings.warn('x2camf was imported from %s before socutils; the bundled x2camf in %s '
                      'is not used. Import socutils first.' % (loaded.__file__, bundled))


_setup_x2camf()
