'''ctypes loader for the CCSDT C backend (libccsdt_clib).

Provides signed antisymmetric pack/unpack of t2 (i<j, a<b) and t3 (i<j<k,
a<b<c) in C (avoids the 36-copy fullasym blowup of the pure-numpy path).
``HAVE_CLIB`` is False if the shared library has not been built; callers then
fall back to the numpy reference in zccsdt.

Build: cc/clib/build.sh  (or set $SOCUTILS_CCSDT_CLIB to the .so path).
'''
import ctypes
import os

import numpy as np


def _load():
    env = os.environ.get('SOCUTILS_CCSDT_CLIB')
    here = os.path.dirname(os.path.abspath(__file__))
    cands = ([env] if env else []) + [
        os.path.join(here, 'clib', 'libccsdt_clib.so'),
        os.path.join(here, 'clib', 'libccsdt_clib.dylib'),
    ]
    for p in cands:
        if p and os.path.exists(p):
            try:
                return ctypes.CDLL(p)
            except OSError:
                pass
    return None


_lib = _load()
HAVE_CLIB = _lib is not None
_cp = ctypes.POINTER(ctypes.c_double)

if HAVE_CLIB:
    for _fn in ('ccsdt_unpack_t3', 'ccsdt_pack_t3',
                'ccsdt_unpack_t2', 'ccsdt_pack_t2'):
        _f = getattr(_lib, _fn)
        _f.restype = None
        _f.argtypes = [_cp, _cp, ctypes.c_int, ctypes.c_int]


def _ptr(a):
    return a.ctypes.data_as(_cp)


def unpack_t3(packed, nocc, nvir):
    full = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.complex128)
    pk = np.ascontiguousarray(packed, dtype=np.complex128)
    _lib.ccsdt_unpack_t3(_ptr(full), _ptr(pk), nocc, nvir)
    return full


def pack_t3(full, nocc, nvir):
    n_otri = nocc * (nocc - 1) * (nocc - 2) // 6
    n_vtri = nvir * (nvir - 1) * (nvir - 2) // 6
    packed = np.zeros(n_otri * n_vtri, dtype=np.complex128)
    f = np.ascontiguousarray(full, dtype=np.complex128)
    _lib.ccsdt_pack_t3(_ptr(f), _ptr(packed), nocc, nvir)
    return packed


def unpack_t2(packed, nocc, nvir):
    full = np.zeros((nocc, nocc, nvir, nvir), dtype=np.complex128)
    pk = np.ascontiguousarray(packed, dtype=np.complex128)
    _lib.ccsdt_unpack_t2(_ptr(full), _ptr(pk), nocc, nvir)
    return full


def pack_t2(full, nocc, nvir):
    n_opair = nocc * (nocc - 1) // 2
    n_vpair = nvir * (nvir - 1) // 2
    packed = np.zeros(n_opair * n_vpair, dtype=np.complex128)
    f = np.ascontiguousarray(full, dtype=np.complex128)
    _lib.ccsdt_pack_t2(_ptr(f), _ptr(packed), nocc, nvir)
    return packed
