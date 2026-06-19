'''
ctypes binding to libzquatev -- Toru Shiozaki's quaternionic eigensolver,
bundled into socutils.

This is a drop-in replacement for the external ``zquatev`` Python package
(github.com/xubwa/zquatev, a pybind11 wrapper): it exposes ``eigh``,
``geigh``, ``check_kramers_structure`` and ``solve_KR_FCSCE`` with the same
signatures, so downstream code works unchanged.  The numerical solver
(``csrc/``) is unmodified BSD-2-Clause zquatev; only the C ABI shim
(``capi/zquatev_capi.cc``) and this loader are new.

The shared library is located in the following order:
  1. the path in the environment variable SOCUTILS_ZQUATEV_LIBRARY
  2. libzquatev.so / .dylib / .dll in the socutils/lib directory (one level up:
     all bundled libraries are built flat into lib/, pyscf-style)
  3. the system library search path (ctypes.util.find_library)
'''

import ctypes
import ctypes.util
import os
import sys

import numpy


def _candidate_paths():
    env = os.environ.get("SOCUTILS_ZQUATEV_LIBRARY")
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    libdir = os.path.dirname(here)   # socutils/lib -- the flat output directory
    if sys.platform.startswith("win"):
        names = ["zquatev.dll", "libzquatev.dll"]
    elif sys.platform == "darwin":
        names = ["libzquatev.dylib", "libzquatev.so"]
    else:
        names = ["libzquatev.so"]
    for name in names:
        yield os.path.join(libdir, name)
        yield os.path.join(here, name)   # legacy in-place location
    found = ctypes.util.find_library("zquatev")
    if found:
        yield found


def _load_library():
    tried = []
    for path in _candidate_paths():
        if os.path.sep in path and not os.path.exists(path):
            tried.append(path)
            continue
        try:
            return ctypes.CDLL(path)
        except OSError:
            tried.append(path)
    raise ImportError(
        "Cannot load libzquatev shared library. Tried: %s. "
        "Build it with cmake (see lib/zquatev/README.md or run build.sh) "
        "and/or set the SOCUTILS_ZQUATEV_LIBRARY environment variable."
        % ", ".join(tried))


_lib = _load_library()

_lib.zquatev_eigh.restype = ctypes.c_int
_lib.zquatev_eigh.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # complex128 buffer, in/out
    ctypes.POINTER(ctypes.c_double),  # eigenvalues, out
]


def eigh(mat, iop=0):
    '''HC=CE for a Hermitian quaternionic matrix (A, -B*; B, A*).'''
    n = mat.shape[0] // 2
    fmat = numpy.asarray(mat).T.copy()          # column-major layout of mat
    fmat = numpy.ascontiguousarray(fmat, dtype=numpy.complex128)
    eigs = numpy.zeros(2 * n, dtype=numpy.float64)
    info = _lib.zquatev_eigh(
        ctypes.c_int(2 * n),
        fmat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        eigs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if info:
        raise RuntimeError('zquatev failed (info=%d)' % info)
    fmat = fmat.T.copy()                         # back to C order
    e = eigs.copy()
    v = fmat.copy()
    if iop == 1:
        e[0::2] = eigs[:n]
        e[1::2] = eigs[n:]
        v[:, 0::2] = fmat[:, :n]
        v[:, 1::2] = fmat[:, n:]
    return e, v


def geigh(tfock, tova, debug=False):
    '''FC=SCE: generalized eigenproblem in a Kramers symmetry-adapted basis.'''
    e, v = eigh(tova)
    if debug:
        print('eig(S)=', e)
    vcanon = v / numpy.sqrt(e)
    tfockmo = vcanon.conj().T.dot(tfock.dot(vcanon))
    e, v = eigh(tfockmo, iop=1)
    if debug:
        print('eig(Fmo)=', e)
    v = vcanon.dot(v)
    return e, v


def check_kramers_structure(mat, thresh=1.e-8):
    print('check_kramers_structure for matrix = [A,B;C,D]')
    n = mat.shape[0] // 2
    a = mat[:n, :n]
    b = mat[:n, n:]
    c = mat[n:, :n]
    d = mat[n:, n:]
    diff1 = numpy.linalg.norm(a - a.T.conj())
    diff2 = numpy.linalg.norm(b + b.T)
    diff3 = numpy.linalg.norm(c + b.conj())
    diff4 = numpy.linalg.norm(d - a.conj())
    print('A-Ah=', diff1)
    print('B+Bt=', diff2)
    print('C+B*=', diff3)
    print('D-A*=', diff4)
    assert (diff1 < thresh and diff2 < thresh and
            diff3 < thresh and diff4 < thresh)
    return 0


def solve_KR_FCSCE(mol, fock, ova, debug=False):
    trmaps = mol.time_reversal_map()
    idxA = numpy.where(trmaps > 0)[0]
    idxB = trmaps[idxA] - 1
    if fock.shape[0] == trmaps.size:
        idx2 = numpy.hstack((idxA, idxB))
    else:
        n = trmaps.size
        idx2 = numpy.hstack((idxA, idxA + n, idxB, idxB + n))

    tova = ova[numpy.ix_(idx2, idx2)]
    tfock = fock[numpy.ix_(idx2, idx2)]
    if debug:
        labels = mol.spinor_labels()
        print(labels)
        for i in range(n):
            print(i, labels[i], trmaps[i])
            print('idxA=', idxA)
            print('idxB=', idxB)
            print('idx2=', idx2)
            check_kramers_structure(tova)
            check_kramers_structure(tfock)
    e, v = geigh(tfock, tova, debug)
    kmo_coeff = numpy.zeros_like(v)
    kmo_coeff[idx2, :] = v
    return e, kmo_coeff
