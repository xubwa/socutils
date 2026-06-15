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
                'ccsdt_unpack_t2', 'ccsdt_pack_t2',
                'ccsdt_fullasym', 'ccsdt_Pabc', 'ccsdt_Pc_ab',
                'ccsdt_Pijk', 'ccsdt_Pk_ij', 'ccsdt_Pijk_Pabc'):
        _f = getattr(_lib, _fn)
        _f.restype = None
        _f.argtypes = [_cp, _cp, ctypes.c_int, ctypes.c_int]

# the permutation (anti)symmetrizers are exposed only when the library is
# present; zccsdt falls back to numpy otherwise.
HAVE_ASYM = HAVE_CLIB and hasattr(_lib, 'ccsdt_fullasym')

# antisymmetrize-into-packed kernels: (Rp, X, scale, op, nocc, nvir)
HAVE_ASYM_PACK = HAVE_CLIB and hasattr(_lib, 'ccsdt_full_to_pack')
if HAVE_ASYM_PACK:
    for _fn in ('ccsdt_pvir_to_pack', 'ccsdt_pocc_to_pack', 'ccsdt_full_to_pack'):
        _f = getattr(_lib, _fn)
        _f.restype = None
        _f.argtypes = [_cp, _cp, ctypes.c_double, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int]
    _lib.ccsdt_ring_to_pack.restype = None
    _lib.ccsdt_ring_to_pack.argtypes = [_cp, _cp, ctypes.c_double,
                                        ctypes.c_int, ctypes.c_int]
    _lib.ccsdt_drive_to_pack.restype = None
    _lib.ccsdt_drive_to_pack.argtypes = [_cp, _cp, ctypes.c_double,
                                         ctypes.c_int, ctypes.c_int]
    for _fn in ('ccsdt_pp_to_pack', 'ccsdt_hh_to_pack'):
        _f = getattr(_lib, _fn)
        _f.restype = None
        _f.argtypes = [_cp, _cp, ctypes.c_double, ctypes.c_int, ctypes.c_int]


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


# ---- T3 residual permutation (anti)symmetrizers ----

def _asym(cfn, t):
    nocc, nvir = t.shape[0], t.shape[3]
    inp = np.ascontiguousarray(t, dtype=np.complex128)
    out = np.empty_like(inp)
    cfn(_ptr(out), _ptr(inp), nocc, nvir)
    return out


def fullasym(t):
    return _asym(_lib.ccsdt_fullasym, t)


def Pabc(t):
    return _asym(_lib.ccsdt_Pabc, t)


def Pc_ab(t):
    return _asym(_lib.ccsdt_Pc_ab, t)


def Pijk(t):
    return _asym(_lib.ccsdt_Pijk, t)


def Pk_ij(t):
    return _asym(_lib.ccsdt_Pk_ij, t)


def Pijk_Pabc(t):
    return _asym(_lib.ccsdt_Pijk_Pabc, t)


# ---- antisymmetrize-into-packed accumulation ----

def pvir_to_pack(Rp, X, scale, op, nocc, nvir):
    '''Rp[notri,nvtri] += scale * P_vir(X), X occ-restricted (notri,nv,nv,nv).
    op=0 P(a/bc), op=1 P(c/ab).'''
    Xc = np.ascontiguousarray(X, dtype=np.complex128)
    _lib.ccsdt_pvir_to_pack(_ptr(Rp), _ptr(Xc), float(scale), int(op),
                            nocc, nvir)


def pocc_to_pack(Rp, X, scale, op, nocc, nvir):
    '''Rp += scale * P_occ(X), X vir-restricted (no,no,no,nvtri).
    op=0 P(i/jk), op=1 P(k/ij).'''
    Xc = np.ascontiguousarray(X, dtype=np.complex128)
    _lib.ccsdt_pocc_to_pack(_ptr(Rp), _ptr(Xc), float(scale), int(op),
                            nocc, nvir)


def full_to_pack(Rp, X, scale, op, nocc, nvir):
    '''Rp += scale * OP(X), X full (no,no,no,nv,nv,nv).
    op=0 fullasym, op=1 P(i/jk)P(a/bc).'''
    Xc = np.ascontiguousarray(X, dtype=np.complex128)
    _lib.ccsdt_full_to_pack(_ptr(Rp), _ptr(Xc), float(scale), int(op),
                            nocc, nvir)


def ring_to_pack(Rp, Xr, scale, nocc, nvir):
    '''Rp += scale * P(i/jk)P(a/bc)(X), X reduced to (no,nv,opair,vpair)
    = (i, a, [j<k], [b<c]); never forms the full O(o^3 v^3) tensor.'''
    Xc = np.ascontiguousarray(Xr, dtype=np.complex128)
    _lib.ccsdt_ring_to_pack(_ptr(Rp), _ptr(Xc), float(scale), nocc, nvir)


def drive_to_pack(Rp, Xr, scale, nocc, nvir):
    '''Rp += scale * fullasym(X), X reduced to canonical (no,opair,vpair,nv)
    = (free_occ, [<], [<], free_vir); never forms the full O(o^3 v^3) tensor.'''
    Xc = np.ascontiguousarray(Xr, dtype=np.complex128)
    _lib.ccsdt_drive_to_pack(_ptr(Rp), _ptr(Xc), float(scale), nocc, nvir)


def pp_to_pack(Rp, Xpp, scale, nocc, nvir):
    '''Rp += scale * P(c/ab)(X), X = (notri, vpair, nv) = (i<j<k, [a<b], c).'''
    Xc = np.ascontiguousarray(Xpp, dtype=np.complex128)
    _lib.ccsdt_pp_to_pack(_ptr(Rp), _ptr(Xc), float(scale), nocc, nvir)


def hh_to_pack(Rp, Xhh, scale, nocc, nvir):
    '''Rp += scale * P(k/ij)(X), X = (opair, no, nvtri) = ([i<j], k, a<b<c).'''
    Xc = np.ascontiguousarray(Xhh, dtype=np.complex128)
    _lib.ccsdt_hh_to_pack(_ptr(Rp), _ptr(Xc), float(scale), nocc, nvir)
