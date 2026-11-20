'''
Tests for the effective 4c densities of the X2C one-electron Hamiltonian
(somf/x2c_grad.py: hfw1_adjoint).

hfw1_adjoint is the adjoint of get_hfw1: for a 2c density D,
tr(D get_hfw1(h4c1, S4c1)) = tr(Gh h4c1) + tr(GS S4c1) for every Hermitian
h4c1, S4c1.  That identity is what it guarantees, so it is checked for random
D, h4c1 and S4c1 against get_hfw1, the response of X and R per perturbation.
'''

import unittest
import numpy as np

from pyscf import gto, lib

from socutils.somf import x2c_grad


def _herm(rng, n, scale=1.):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (a + a.conj().T) * (.5 * scale)


def _x2c_blocks(mol):
    c = lib.param.LIGHT_SPEED
    s = mol.intor('int1e_ovlp_spinor')
    t = mol.intor('int1e_spsp_spinor') * .5
    v = mol.intor('int1e_nuc_spinor')
    w = mol.intor('int1e_spnucsp_spinor')
    return x2c_grad.x2c1e_hfw0_block(v, t, t, w * (.25 / c**2) - t, s, t * (.5 / c**2))


class KnownValues(unittest.TestCase):
    def test_adjoint(self):
        rng = np.random.default_rng(1)
        for atom in ('H 0 0 0; F 0 0 0.92', 'H 0 0 0; Br 0 0 1.41'):
            mol = gto.M(atom=atom, basis='unc-ccpvdz', verbose=0)
            a, e, x, st, r, l, h4c, m4c = _x2c_blocks(mol)
            n = x.shape[0]
            D = _herm(rng, n)
            Gh, GS = x2c_grad.hfw1_adjoint(a, x, st, m4c, h4c, e, r, l, D)
            self.assertAlmostEqual(abs(Gh - Gh.conj().T).max(), 0, 14)
            self.assertAlmostEqual(abs(GS - GS.conj().T).max(), 0, 14)
            for trial in range(3):
                h1 = _herm(rng, 2 * n)
                S1 = np.zeros((2 * n, 2 * n), dtype=complex)
                S1[:n, :n] = _herm(rng, n)
                S1[n:, n:] = _herm(rng, n, 1e-4)
                for S in (None, S1):
                    ref = np.trace(D @ x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h1, S))
                    val = np.trace(Gh @ h1) + (0 if S is None else np.trace(GS @ S))
                    self.assertLess(abs(val - ref), 1e-10 * abs(ref))

    def test_helpers(self):
        # get_hfw1_adjoint of the .x2camf() helpers (spinor and spin-orbital),
        # both settings of x_response, against their get_hfw1
        from socutils.scf import spinor_hf, ghf
        rng = np.random.default_rng(3)
        mol = gto.M(atom='H 0 0 0; F 0 0 0.92', basis='cc-pvdz', verbose=0)
        for mf in (spinor_hf.SCF(mol).x2camf(), ghf.GHF(mol).x2camf()):
            helper = mf.with_x2c
            helper.get_hcore(mol)
            n = helper.h4c.shape[0] // 2
            D = _herm(rng, n)
            for x_response in (True, False):
                Gh, GS = helper.get_hfw1_adjoint(D, x_response=x_response)
                for trial in range(2):
                    h1 = _herm(rng, 2 * n)
                    S1 = np.zeros((2 * n, 2 * n), dtype=complex)
                    S1[:n, :n] = _herm(rng, n)
                    S1[n:, n:] = _herm(rng, n, 1e-4)
                    for S in ((None, S1) if x_response else (None,)):
                        ref = np.trace(D @ helper.get_hfw1(h1, S, x_response=x_response))
                        val = np.trace(Gh @ h1) + (0 if S is None else np.trace(GS @ S))
                        self.assertLess(abs(val - ref), 1e-10 * abs(ref))


if __name__ == '__main__':
    print('Tests for the effective 4c densities of X2C')
    unittest.main()
