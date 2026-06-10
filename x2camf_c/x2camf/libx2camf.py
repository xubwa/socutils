'''
ctypes binding to libx2camf_c, the plain-C interface of the X2CAMF code.

This module is a drop-in replacement for the pybind11 extension module
``x2camf.libx2camf``: it exposes ``amfi``, ``atm_integrals`` and ``pcc_K``
with the same signatures and return values (a list of numpy matrices), so
downstream code such as socutils works unchanged.

The shared library is located in the following order:
  1. the path in the environment variable X2CAMF_C_LIBRARY
  2. libx2camf_c.so / .dylib / .dll next to this file
  3. the system library search path (ctypes.util.find_library)
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
    if sys.platform.startswith("win"):
        names = ["x2camf_c.dll", "libx2camf_c.dll"]
    elif sys.platform == "darwin":
        names = ["libx2camf_c.dylib", "libx2camf_c.so"]
    else:
        names = ["libx2camf_c.so"]
    for name in names:
        yield os.path.join(here, name)
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
            return ctypes.CDLL(path)
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
