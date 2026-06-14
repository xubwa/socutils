'''
C++ backend: thin wrapper around the original pybind11 extension module
``libx2camf`` (built from the upstream X2CAMF C++ sources).

This exists so socutils can switch between the pure-C backend and the
reference C++ backend for A/B comparison. The pybind module is located in
the following order:
  1. X2CAMF_CPP_LIBRARY  -- full path to the libx2camf*.so file
  2. X2CAMF_CPP_PATH     -- a directory to prepend to sys.path
  3. a plain ``import libx2camf`` (module already on PYTHONPATH)
'''

import importlib
import importlib.util
import os
import sys

import numpy

SPEED_OF_LIGHT_DEFAULT = 137.0359991

# (name, is_four_component) in the order returned by the C++ atm_integrals;
# identical to the pure-C backend layout.
ATM_INTEGRALS_LAYOUT = (
    ("atm_X", False), ("atm_R", False),
    ("h1e_4c", True), ("fock_4c", True), ("fock_2c", False),
    ("fock_4c_2e", True), ("fock_2c_2e", False),
    ("fock_4c_K", True), ("fock_2c_K", False),
    ("so_4c", True), ("so_2c", False),
    ("den_4c", True), ("den_2c", False),
)

_cpp = None


def _load():
    global _cpp
    if _cpp is not None:
        return _cpp
    libpath = os.environ.get("X2CAMF_CPP_LIBRARY")
    if libpath:
        spec = importlib.util.spec_from_file_location("libx2camf", libpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cpp = mod
        return _cpp
    extra = os.environ.get("X2CAMF_CPP_PATH")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        _cpp = importlib.import_module("libx2camf")
    except ImportError as e:
        raise ImportError(
            "x2camf C++ backend requested but the pybind11 module "
            "'libx2camf' could not be loaded. Build the upstream x2camf and "
            "set X2CAMF_CPP_LIBRARY=/path/to/libx2camf*.so (or "
            "X2CAMF_CPP_PATH=/path/to/build).") from e
    return _cpp


def _shaped(shell, exp_a):
    # the pybind interface reads shell(i,0)/exp_a(i,0), i.e. column matrices
    shell = numpy.ascontiguousarray(
        numpy.asarray(shell).reshape(-1, 1), dtype=numpy.int32)
    exp_a = numpy.ascontiguousarray(
        numpy.asarray(exp_a).reshape(-1, 1), dtype=numpy.float64)
    return shell, exp_a


def amfi(input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
         speed_of_light=SPEED_OF_LIGHT_DEFAULT):
    lib = _load()
    shell, exp_a = _shaped(shell, exp_a)
    return lib.amfi(int(input_string), int(atom_number), int(nshell),
                    int(nbas), int(printLevel), shell, exp_a,
                    speed_of_light=float(speed_of_light))


def pcc_K(input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
          speed_of_light=SPEED_OF_LIGHT_DEFAULT):
    lib = _load()
    shell, exp_a = _shaped(shell, exp_a)
    return lib.pcc_K(int(input_string), int(atom_number), int(nshell),
                     int(nbas), int(printLevel), shell, exp_a,
                     speed_of_light=float(speed_of_light))


def atm_integrals(input_string, atom_number, nshell, nbas, printLevel,
                  shell, exp_a, speed_of_light=SPEED_OF_LIGHT_DEFAULT,
                  spin_free=False):
    lib = _load()
    shell, exp_a = _shaped(shell, exp_a)
    return lib.atm_integrals(int(input_string), int(atom_number), int(nshell),
                             int(nbas), int(printLevel), shell, exp_a,
                             speed_of_light=float(speed_of_light),
                             spin_free=bool(spin_free))
