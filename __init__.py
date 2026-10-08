import os as _os
import sys as _sys

# Optionally expose the bundled x2camf interface (a dual-backend dispatcher:
# the pure-C reimplementation plus the upstream pybind11 reference) as the
# top-level ``x2camf`` package.
#
# Migration-safe default: if an ``x2camf`` package is already importable
# (e.g. the upstream Warlocat pybind build that collaborators already use),
# it is left in charge and nothing about an existing workflow changes after
# pulling -- no rebuild required.  Only when no external ``x2camf`` is found
# is the in-tree dispatcher put on the path.
#
# Control with the environment variable SOCUTILS_X2CAMF:
#   auto      (default) bundled only if no external x2camf is installed
#   bundled   always use the in-tree dual-backend dispatcher
#   external  never touch sys.path (use whatever x2camf is installed)
def _setup_x2camf():
    bundled = _os.path.join(_os.path.dirname(__file__), 'x2camf_c')
    if not _os.path.isdir(bundled):
        return
    mode = _os.environ.get('SOCUTILS_X2CAMF', 'auto').strip().lower()
    if mode == 'external':
        return
    if mode == 'auto':
        import importlib.util
        if importlib.util.find_spec('x2camf') is not None:
            return  # an external x2camf is installed; leave it in charge
    if bundled not in _sys.path:
        _sys.path.insert(0, bundled)


_setup_x2camf()


# Register the j-adapted basis sets of socutils/gto/basis with PySCF (basis='<name>') and make
# scalar SCF objects warn when given such a set.  SOCUTILS_BASIS=off skips this.
def _setup_basis():
    if _os.environ.get('SOCUTILS_BASIS', 'on').strip().lower() == 'off':
        return
    try:
        from .gto import basis as _basis
        _basis.register()
    except ImportError:
        pass


_setup_basis()
