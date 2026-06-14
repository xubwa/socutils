'''
Backend selection for the x2camf interface.

Two backends provide the identical amfi / atm_integrals / pcc_K API:

  'c'   -- pure-C reimplementation (libx2camf_c.so, loaded via ctypes)
  'cpp' -- original pybind11 extension (libx2camf, the C++ reference)

The default is 'c'.  Override globally with the environment variable
X2CAMF_BACKEND=c|cpp, or at runtime with set_backend(); the latter is
handy for A/B comparison in a single process:

    from x2camf import backend
    backend.set_backend('cpp')   # use the C++ reference
    ...
    backend.set_backend('c')     # back to the pure-C implementation

You can also scope a choice with a context manager:

    with backend.using('cpp'):
        ref = x2camf.amfi(x2cobj, ...)
'''

import contextlib
import os

_VALID = ("c", "cpp")
_current = None


def _default():
    env = os.environ.get("X2CAMF_BACKEND")
    if env is not None:
        env = env.strip().lower()
        if env not in _VALID:
            raise ValueError(
                "X2CAMF_BACKEND must be one of %r, got %r" % (_VALID, env))
        return env
    # Auto: prefer the pure-C library when it is built and loadable,
    # otherwise fall back to the upstream pybind11 reference so an
    # unbuilt checkout still works for users who have the C++ module.
    try:
        from x2camf import _c_backend  # importing loads libx2camf_c.so
        assert _c_backend is not None
        return "c"
    except Exception:
        return "cpp"


def get_backend():
    global _current
    if _current is None:
        _current = _default()
    return _current


def set_backend(name):
    global _current
    name = str(name).strip().lower()
    if name not in _VALID:
        raise ValueError("backend must be one of %r, got %r" % (_VALID, name))
    _current = name
    return _current


def load_backend(name=None):
    '''Return the backend module ('c' or 'cpp').'''
    name = get_backend() if name is None else name
    if name == "c":
        from x2camf import _c_backend
        return _c_backend
    elif name == "cpp":
        from x2camf import _cpp_backend
        return _cpp_backend
    raise ValueError("unknown backend %r" % name)


@contextlib.contextmanager
def using(name):
    prev = get_backend()
    set_backend(name)
    try:
        yield
    finally:
        set_backend(prev)
