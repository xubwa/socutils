'''Validation of the pure-C implementation against the original C++ code.

Requires the original pybind11 module (built from the upstream x2camf
repository) importable as `libx2camf`, e.g.:

    PYTHONPATH=/path/to/x2camf/build python tests/test_against_cpp.py

Eigen and LAPACK produce slightly different floating-point round-off, and
the SCF amplifies it, so the comparison uses a tolerance rather than
bit-equality.
'''
import os
import sys

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libx2camf as cpp_lib  # noqa: E402  (pybind11 reference)
from x2camf import libx2camf as c_lib  # noqa: E402  (ctypes / pure C)

TOL = 1e-8


def basis_O():
    """Small O 6s3p uncontracted even-tempered basis (fast smoke test)."""
    shell = numpy.array([0]*6 + [1]*3)
    exp_a = numpy.array([2.0**(8-i) for i in range(6)]
                        + [2.0**(4-i) for i in range(3)])
    return 8, shell, exp_a


def basis_heavy(Z):
    """A basis that exercises d (and f) shells for a heavier atom."""
    shell = numpy.array([0]*8 + [1]*5 + [2]*3 + [3]*1)
    exp_a = numpy.array([2.0**(9-i) for i in range(8)]
                        + [2.0**(5-i) for i in range(5)]
                        + [2.0**(3-i) for i in range(3)]
                        + [1.0])
    return Z, shell, exp_a


def compare(label, Z, shell, exp_a, flavors, do_atm=True):
    nbas, nshell = shell.size, int(shell[-1] + 1)
    shell_m = shell.reshape(-1, 1).astype(numpy.int32)
    exp_m = exp_a.reshape(-1, 1)
    worst = 0.0
    print(f"\n=== {label} (Z={Z}, nbas={nbas}, lmax={shell[-1]}) ===")
    for name, flavor in flavors:
        a = cpp_lib.amfi(flavor, Z, nshell, nbas, 0, shell_m, exp_m)[0]
        b = c_lib.amfi(flavor, Z, nshell, nbas, 0, shell, exp_a)[0]
        diff = abs(a - b).max()
        worst = max(worst, diff)
        print(f"amfi[{name:15s}] max|diff| = {diff:.3e}")
        assert a.shape == b.shape and diff < TOL, name

    a = cpp_lib.pcc_K(3, Z, nshell, nbas, 0, shell_m, exp_m)[0]
    b = c_lib.pcc_K(3, Z, nshell, nbas, 0, shell, exp_a)[0]
    diff = abs(a - b).max()
    worst = max(worst, diff)
    print(f"pcc_K max|diff| = {diff:.3e}")
    assert diff < TOL

    if do_atm:
        for sf in (False, True):
            ra = cpp_lib.atm_integrals(3, Z, nshell, nbas, 0, shell_m, exp_m,
                                       spin_free=sf)
            rb = c_lib.atm_integrals(3, Z, nshell, nbas, 0, shell, exp_a,
                                     spin_free=sf)
            for (nm, _), ma, mb in zip(c_lib.ATM_INTEGRALS_LAYOUT, ra, rb):
                diff = abs(ma - mb).max()
                worst = max(worst, diff)
                assert ma.shape == mb.shape and diff < TOL, (nm, sf, diff)
        print(f"atm_integrals (both spin_free) OK")
    print(f"--- {label}: worst deviation {worst:.3e}")
    return worst


FLAVORS = [("coulomb", 0), ("gaunt", 1), ("gaunt+gauge", 3),
           ("gaunt+gauge 4c", 3 | (1 << 6)), ("aoc", 3 | (1 << 3)),
           ("pt", (1 << 4)), ("pcc", 3 | (1 << 5)),
           ("sd-gaunt", 1 | (1 << 7))]


def main():
    worst = 0.0
    # Level 1: fast O smoke test
    Z, shell, exp_a = basis_O()
    worst = max(worst, compare("O smoke test", Z, shell, exp_a, FLAVORS))

    # Level 2: heavier atom with d/f shells (Xe by default; override with
    # X2CAMF_TEST_Z, e.g. 60 for Nd to stress f electrons + AOC)
    Z = int(os.environ.get("X2CAMF_TEST_Z", "54"))
    Zh, shell, exp_a = basis_heavy(Z)
    worst = max(worst, compare(f"heavy atom Z={Z}", Zh, shell, exp_a,
                               FLAVORS))

    print(f"\nALL COMPARISONS PASSED (worst deviation {worst:.3e} < {TOL})")


if __name__ == '__main__':
    main()
