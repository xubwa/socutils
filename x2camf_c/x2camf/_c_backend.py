'''
Pure-C backend: ctypes binding to libx2camf_c (the C reimplementation).

This module is a drop-in replacement for the pybind11 extension module
``x2camf.libx2camf``: it exposes ``amfi``, ``atm_integrals`` and ``pcc_K``
with the same signatures and return values (a list of numpy matrices), so
downstream code such as socutils works unchanged.

The shared library is located in the following order:
  1. the path in the environment variable X2CAMF_C_LIBRARY
  2. libx2camf_c.so / .dylib / .dll in the socutils/lib directory (all bundled
     libraries are built flat into lib/, pyscf-style)
  3. libx2camf_c.* next to this file (legacy in-place location)
  4. the system library search path (ctypes.util.find_library)
'''

import ctypes
import ctypes.util
import os
import sys

import numpy

SPEED_OF_LIGHT_DEFAULT = 137.0359991

_N_ATM_INTEGRALS = 13

_ERROR_MESSAGES = {
    1: "invalid argument (atom number must be in [1,118], nbas > 0, "
       "shell entries >= 0)",
    2: "PCC is not implemented for the PT scheme",
    3: "internal error in libx2camf_c (C++ exception)",
    4: "internal dimension check failed in libx2camf_c",
}


def _candidate_paths():
    env = os.environ.get("X2CAMF_C_LIBRARY")
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    # socutils/lib -- the flat output directory (this package lives at
    # socutils/x2camf_c/x2camf/, so lib/ is two levels up).
    libdir = os.path.abspath(os.path.join(here, os.pardir, os.pardir, "lib"))
    if sys.platform.startswith("win"):
        names = ["x2camf_c.dll", "libx2camf_c.dll"]
    elif sys.platform == "darwin":
        names = ["libx2camf_c.dylib", "libx2camf_c.so"]
    else:
        names = ["libx2camf_c.so"]
    for name in names:
        yield os.path.join(libdir, name)
        yield os.path.join(here, name)   # legacy in-place location
    found = ctypes.util.find_library("x2camf_c")
    if found:
        yield found


def _load_library():
    tried = []
    for path in _candidate_paths():
        if os.path.sep in path and not os.path.exists(path):
            tried.append(path)
            continue
        try:
            lib = ctypes.CDLL(path)
            sys.stdout.write(
                "[socutils] x2camf: BUNDLED pure-C backend loaded (%s)\n"
                % os.path.abspath(path))
            return lib
        except OSError:
            tried.append(path)
    raise ImportError(
        "Cannot load libx2camf_c shared library. Tried: %s. "
        "Build it with cmake (or pip install .) and/or set the "
        "X2CAMF_C_LIBRARY environment variable." % ", ".join(tried))


_lib = _load_library()

_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_double_p = ctypes.POINTER(ctypes.c_double)

_lib.x2camf_n2c.restype = ctypes.c_int
_lib.x2camf_n2c.argtypes = [ctypes.c_int, _c_int_p]

_lib.x2camf_amfi.restype = ctypes.c_int
_lib.x2camf_amfi.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _c_int_p, _c_double_p, ctypes.c_double, _c_double_p]

_lib.x2camf_pcc_k.restype = ctypes.c_int
_lib.x2camf_pcc_k.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _c_int_p, _c_double_p, ctypes.c_double, _c_double_p]

_lib.x2camf_atm_integrals.restype = ctypes.c_int
_lib.x2camf_atm_integrals.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _c_int_p, _c_double_p, ctypes.c_double, ctypes.c_int,
    ctypes.POINTER(_c_double_p)]


_c_int_out = ctypes.POINTER(ctypes.c_int)
_HAS_ATOM_SCF = hasattr(_lib, "x2camf_atom_scf")   # absent in older builds
if _HAS_ATOM_SCF:
    _lib.x2camf_atom_scf.restype = ctypes.c_int
    _lib.x2camf_atom_scf.argtypes = (
        [ctypes.c_int] * 4 + [_c_int_p, _c_double_p, ctypes.c_double]
        + [ctypes.c_int] * 6 + [ctypes.c_char_p, ctypes.c_double, ctypes.c_int]
        + [ctypes.c_int, ctypes.c_double, ctypes.c_double]
        + [_c_double_p, _c_int_out] + [_c_double_p] * 7)


def _check(ierr, name):
    if ierr != 0:
        msg = _ERROR_MESSAGES.get(ierr, "unknown error code %d" % ierr)
        raise RuntimeError("%s failed: %s" % (name, msg))


def _prepare_basis(shell, exp_a):
    shell = numpy.ascontiguousarray(
        numpy.asarray(shell).reshape(-1), dtype=numpy.intc)
    exp_a = numpy.ascontiguousarray(
        numpy.asarray(exp_a).reshape(-1), dtype=numpy.float64)
    if shell.size != exp_a.size:
        raise ValueError("shell and exp_a must have the same length")
    n2c = int(numpy.sum(4 * shell + 2))
    return shell, exp_a, n2c


def _as_int_p(arr):
    return arr.ctypes.data_as(_c_int_p)


def _as_double_p(arr):
    return arr.ctypes.data_as(_c_double_p)


def amfi(input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
         speed_of_light=SPEED_OF_LIGHT_DEFAULT):
    '''Compute the AMFI matrix. Returns [amfi_matrix].'''
    shell, exp_a, n2c = _prepare_basis(shell, exp_a)
    int4c = (int(input_string) >> 6) & 1
    dim = 2 * n2c if int4c else n2c
    out = numpy.zeros((dim, dim))
    ierr = _lib.x2camf_amfi(
        int(input_string), int(atom_number), int(nshell), int(nbas),
        int(printLevel), _as_int_p(shell), _as_double_p(exp_a),
        float(speed_of_light), _as_double_p(out))
    _check(ierr, "x2camf_amfi")
    return [out]


def pcc_K(input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
          speed_of_light=SPEED_OF_LIGHT_DEFAULT):
    '''Compute the exchange-only 2e-PCC matrix. Returns [pcc_matrix].'''
    shell, exp_a, n2c = _prepare_basis(shell, exp_a)
    int4c = (int(input_string) >> 6) & 1
    dim = 2 * n2c if int4c else n2c
    out = numpy.zeros((dim, dim))
    ierr = _lib.x2camf_pcc_k(
        int(input_string), int(atom_number), int(nshell), int(nbas),
        int(printLevel), _as_int_p(shell), _as_double_p(exp_a),
        float(speed_of_light), _as_double_p(out))
    _check(ierr, "x2camf_pcc_k")
    return [out]


# (name, is_four_component) in the order returned by the C library,
# matching the pybind11 atm_integrals return value.
ATM_INTEGRALS_LAYOUT = (
    ("atm_X", False), ("atm_R", False),
    ("h1e_4c", True), ("fock_4c", True), ("fock_2c", False),
    ("fock_4c_2e", True), ("fock_2c_2e", False),
    ("fock_4c_K", True), ("fock_2c_K", False),
    ("so_4c", True), ("so_2c", False),
    ("den_4c", True), ("den_2c", False),
)


def atm_integrals(input_string, atom_number, nshell, nbas, printLevel,
                  shell, exp_a, speed_of_light=SPEED_OF_LIGHT_DEFAULT,
                  spin_free=False):
    '''Compute all the atomic integrals.

    Returns the same list of 13 matrices as the pybind11 interface:
    [atm_X, atm_R, h1e_4c, fock_4c, fock_2c, fock_4c_2e, fock_2c_2e,
     fock_4c_K, fock_2c_K, so_4c, so_2c, den_4c, den_2c]
    '''
    shell, exp_a, n2c = _prepare_basis(shell, exp_a)
    outs = []
    for _, four_c in ATM_INTEGRALS_LAYOUT:
        dim = 2 * n2c if four_c else n2c
        outs.append(numpy.zeros((dim, dim)))
    out_ptrs = (_c_double_p * _N_ATM_INTEGRALS)(
        *[_as_double_p(mat) for mat in outs])
    ierr = _lib.x2camf_atm_integrals(
        int(input_string), int(atom_number), int(nshell), int(nbas),
        int(printLevel), _as_int_p(shell), _as_double_p(exp_a),
        float(speed_of_light), int(bool(spin_free)), out_ptrs)
    _check(ierr, "x2camf_atm_integrals")
    return outs


def atom_scf(atom_number, nshell, nbas, printLevel, shell, exp_a,
             speed_of_light=SPEED_OF_LIGHT_DEFAULT, scf_type=0,
             spin_free=False, gaunt=False, gauge=False, gau_nuc=False,
             aoc=False, occ_file="", conv=1e-9, max_iter=100,
             nuc_model=0, nuc_p1=0.0, nuc_p2=0.0):
    '''One atomic SCF: energy and, per kappa block, spinor energies,
    coefficients, occupations, overlap, Fock, h1e and density matrices.

    scf_type 0: 4c DHF (DC, +Gaunt/gauge, or spin-free); 1: X2C-1e HF;
    2: X2C-1e HF + X2CAMF spin-orbit integrals of a 4c DHF.
    nuc_model 0: point (or the built-in Gaussian nucleus with gau_nuc),
    1 Fermi (c, a), 2 Gaussian (exponent), 3 uniform sphere (radius); bohr.

    Returns {'e_scf', 'converged', 'blocks'}; each block is a dict with
    l, two_j, exp, e, c, occ, ovlp, fock, h1e, den (see x2camf_c.h).
    '''
    if not _HAS_ATOM_SCF:
        raise RuntimeError("this libx2camf_c has no x2camf_atom_scf: "
                           "rebuild the socutils libraries (make)")
    shell, exp_a, _ = _prepare_basis(shell, exp_a)
    if numpy.any(numpy.diff(shell) < 0):
        raise ValueError("shell must be sorted by angular momentum")
    twoC = int(scf_type) != 0
    # kappa blocks: (l, two_j, primitives), s1/2, p1/2, p3/2, d3/2, ...
    layout = []
    for l in range(int(nshell)):
        n = int(numpy.count_nonzero(shell == l))
        for two_j in ([1] if l == 0 else [2 * l - 1, 2 * l + 1]):
            layout.append((l, two_j, n))
    dims = [n if twoC else 2 * n for _, _, n in layout]
    n1 = sum(n for _, _, n in layout)
    eps, occ = numpy.zeros(n1), numpy.zeros(n1)
    coeff = numpy.zeros(sum(d * n for d, (_, _, n) in zip(dims, layout)))
    mats = [numpy.zeros(sum(d * d for d in dims)) for _ in range(4)]
    e_scf = ctypes.c_double(0.0)
    converged = ctypes.c_int(0)
    ierr = _lib.x2camf_atom_scf(
        int(atom_number), int(nshell), int(nbas), int(printLevel),
        _as_int_p(shell), _as_double_p(exp_a), float(speed_of_light),
        int(scf_type), int(bool(spin_free)), int(bool(gaunt)),
        int(bool(gauge)), int(bool(gau_nuc)), int(bool(aoc)),
        os.fsencode(occ_file or ""), float(conv), int(max_iter),
        int(nuc_model), float(nuc_p1), float(nuc_p2),
        ctypes.byref(e_scf), ctypes.byref(converged), _as_double_p(eps),
        _as_double_p(occ), _as_double_p(coeff),
        *[_as_double_p(m) for m in mats])
    _check(ierr, "x2camf_atom_scf")
    blocks, i1, ic, im = [], 0, 0, 0
    for (l, two_j, n), d in zip(layout, dims):
        blocks.append(dict(
            l=l, two_j=two_j, exp=exp_a[shell == l].copy(),
            e=eps[i1:i1 + n].copy(), occ=occ[i1:i1 + n].copy(),
            c=coeff[ic:ic + d * n].reshape(d, n).copy(),
            ovlp=mats[0][im:im + d * d].reshape(d, d).copy(),
            fock=mats[1][im:im + d * d].reshape(d, d).copy(),
            h1e=mats[2][im:im + d * d].reshape(d, d).copy(),
            den=mats[3][im:im + d * d].reshape(d, d).copy()))
        i1 += n
        ic += d * n
        im += d * d
    return dict(e_scf=e_scf.value, converged=bool(converged.value),
                blocks=blocks)
