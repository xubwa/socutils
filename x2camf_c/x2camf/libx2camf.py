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

from x2camf import backend

SPEED_OF_LIGHT_DEFAULT = 137.0359991

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
    return backend.load_backend().amfi(
        input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
        speed_of_light=speed_of_light)


def pcc_K(input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
          speed_of_light=SPEED_OF_LIGHT_DEFAULT):
    return backend.load_backend().pcc_K(
        input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
        speed_of_light=speed_of_light)


def atm_integrals(input_string, atom_number, nshell, nbas, printLevel,
                  shell, exp_a, speed_of_light=SPEED_OF_LIGHT_DEFAULT,
                  spin_free=False):
    return backend.load_backend().atm_integrals(
        input_string, atom_number, nshell, nbas, printLevel, shell, exp_a,
        speed_of_light=speed_of_light, spin_free=spin_free)
