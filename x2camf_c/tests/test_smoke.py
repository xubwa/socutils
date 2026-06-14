'''Smoke test for the ctypes interface (no pyscf required).

Runs an O atom with a small uncontracted even-tempered basis and checks
shapes and basic numerics of all three entry points.
'''
import os
import sys

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from x2camf import libx2camf


def small_basis():
    # uncontracted even-tempered toy basis: 6s 3p
    shell = numpy.array([0]*6 + [1]*3)
    exp_a = numpy.array([2.0**(8-i) for i in range(6)]
                        + [2.0**(4-i) for i in range(3)])
    return shell, exp_a


def test_amfi():
    shell, exp_a = small_basis()
    nbas = shell.size
    nshell = shell[-1] + 1
    n2c = int(numpy.sum(4*shell + 2))

    flavor = (1 << 0) | (1 << 1)  # gaunt + gauge
    res = libx2camf.amfi(flavor, 8, nshell, nbas, 0, shell, exp_a)
    assert len(res) == 1
    amf = res[0]
    assert amf.shape == (n2c, n2c)
    assert numpy.all(numpy.isfinite(amf))
    assert numpy.linalg.norm(amf) > 1e-8
    # the AMFI matrix in the real spinor representation is symmetric
    assert abs(amf - amf.T).max() < 1e-10
    print("test_amfi passed, |amfi| =", numpy.linalg.norm(amf))


def test_amfi_4c():
    shell, exp_a = small_basis()
    nbas = shell.size
    nshell = shell[-1] + 1
    n2c = int(numpy.sum(4*shell + 2))

    flavor = (1 << 0) | (1 << 1) | (1 << 6)  # gaunt + gauge + int4c
    res = libx2camf.amfi(flavor, 8, nshell, nbas, 0, shell, exp_a)
    amf = res[0]
    assert amf.shape == (2*n2c, 2*n2c)
    assert numpy.all(numpy.isfinite(amf))
    print("test_amfi_4c passed, |amfi_4c| =", numpy.linalg.norm(amf))


def test_atm_integrals():
    shell, exp_a = small_basis()
    nbas = shell.size
    nshell = shell[-1] + 1
    n2c = int(numpy.sum(4*shell + 2))

    flavor = (1 << 0) | (1 << 1)  # gaunt + gauge
    res = libx2camf.atm_integrals(flavor, 8, nshell, nbas, 0, shell, exp_a)
    assert len(res) == 13
    for (name, four_c), mat in zip(libx2camf.ATM_INTEGRALS_LAYOUT, res):
        dim = 2*n2c if four_c else n2c
        assert mat.shape == (dim, dim), (name, mat.shape, dim)
        assert numpy.all(numpy.isfinite(mat)), name
    den_4c = res[11]
    # the trace of the 4c density matrix ~ number of electrons is not
    # directly Tr(D) in a non-orthogonal basis, so just check hermiticity
    fock_4c = res[3]
    assert abs(fock_4c - fock_4c.T).max() < 1e-8
    print("test_atm_integrals passed")


def test_pcc_k():
    shell, exp_a = small_basis()
    nbas = shell.size
    nshell = shell[-1] + 1
    n2c = int(numpy.sum(4*shell + 2))

    flavor = (1 << 0) | (1 << 1)
    res = libx2camf.pcc_K(flavor, 8, nshell, nbas, 0, shell, exp_a)
    assert res[0].shape == (n2c, n2c)
    assert numpy.all(numpy.isfinite(res[0]))
    print("test_pcc_k passed, |pcc_k| =", numpy.linalg.norm(res[0]))


def test_errors():
    shell, exp_a = small_basis()
    nbas = shell.size
    nshell = shell[-1] + 1
    try:
        libx2camf.amfi(0, 300, nshell, nbas, 0, shell, exp_a)
    except RuntimeError as e:
        print("test_errors passed:", e)
    else:
        raise AssertionError("expected RuntimeError for atom number 300")
    # pt + pcc must be rejected instead of calling exit(99)
    try:
        libx2camf.amfi((1 << 4) | (1 << 5), 8, nshell, nbas, 0, shell, exp_a)
    except RuntimeError as e:
        print("test_errors (pt+pcc) passed:", e)
    else:
        raise AssertionError("expected RuntimeError for pt+pcc")


if __name__ == '__main__':
    test_amfi()
    test_amfi_4c()
    test_atm_integrals()
    test_pcc_k()
    test_errors()
    print("All smoke tests passed.")
