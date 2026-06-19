'''
Backend-dispatching facade for the X2CAMF low-level interface.

Existing code (socutils and the high-level x2camf.x2camf module) reaches
this module via ``from x2camf import libx2camf`` and calls
``libx2camf.amfi`` / ``atm_integrals`` / ``pcc_K``.  Each call is routed to
the currently selected backend (see x2camf.backend): the pure-C ctypes
implementation ('c', default) or the pybind11 C++ reference ('cpp').

Switch backends for comparison testing with:

    from x2camf import backend
    backend.set_backend('cpp')          # or 'c'
    # or scoped:
    with backend.using('cpp'):
        ...
'''

import os
import sys
import time

from x2camf import backend

SPEED_OF_LIGHT_DEFAULT = 137.0359991

# Per-call wall-clock timing, printed to stdout. On by default; silence with
# X2CAMF_TIMING=0 (also accepts false/off/no).
_TIMING = os.environ.get('X2CAMF_TIMING', '1').strip().lower() \
    not in ('0', 'false', 'off', 'no')


def _dispatch(method, atom_number, args, kwargs):
    '''Route a call to the selected backend, optionally timing it.'''
    fn = getattr(backend.load_backend(), method)
    if not _TIMING:
        return fn(*args, **kwargs)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    sys.stdout.write("[socutils] x2camf.%s  Z=%s  backend=%s  %.3f s\n"
                     % (method, atom_number, backend.get_backend(), dt))
    return out

# (name, is_four_component) for the 13 matrices returned by atm_integrals;
# identical across both backends.
ATM_INTEGRALS_LAYOUT = (
    ("atm_X", False), ("atm_R", False),
    ("h1e_4c", True), ("fock_4c", True), ("fock_2c", False),
    ("fock_4c_2e", True), ("fock_2c_2e", False),
    ("fock_4c_K", True), ("fock_2c_K", False),
    ("so_4c", True), ("so_2c", False),
    ("den_4c", True), ("den_2c", False),
)


def amfi(input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
         speed_of_light=SPEED_OF_LIGHT_DEFAULT):
    return _dispatch('amfi', atom_number,
                     (input_string, atom_number, nshell, nbas, printLevel,
                      shell, exp_a),
                     dict(speed_of_light=speed_of_light))


def pcc_K(input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
          speed_of_light=SPEED_OF_LIGHT_DEFAULT):
    return _dispatch('pcc_K', atom_number,
                     (input_string, atom_number, nshell, nbas, printLevel,
                      shell, exp_a),
                     dict(speed_of_light=speed_of_light))


def atm_integrals(input_string, atom_number, nshell, nbas, printLevel,
                  shell, exp_a, speed_of_light=SPEED_OF_LIGHT_DEFAULT,
                  spin_free=False):
    return _dispatch('atm_integrals', atom_number,
                     (input_string, atom_number, nshell, nbas, printLevel,
                      shell, exp_a),
                     dict(speed_of_light=speed_of_light, spin_free=spin_free))
