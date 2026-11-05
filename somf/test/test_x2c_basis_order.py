#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
X2CAMF/X2CMP for basis sets whose primitives are not in the order of the atomic
calculation (gto.mole.uncontracted_basis: per l, descending exponents).

The atomic matrices must be placed on the right functions of the uncontracted
molecule whatever the order of the input shells:

1. cc-pVDZ with the Br shells given in reverse order (d, p, s) gives the same
   X2CAMF and X2CMP energies as cc-pVDZ as given.
2. Segmented 6-31G and def2-SVP: X2CMP runs (it aborted in zquatev before, with
   a garbage, non-Hermitian hcore), its hcore is Hermitian and Kramers symmetric.
3. GHF (spin-orbital, via spinor2sph) equals the spinor result for 6-31G.
'''

import unittest
import numpy
from pyscf import gto
from pyscf.scf import dhf
from socutils.scf import spinor_hf, ghf
import x2camf   # after socutils, which puts the bundled x2camf first on sys.path


def _mol(basis):
    return gto.M(atom='H 0 0 0; Br 0 0 1.41', basis=basis, verbose=0)


def _energy(cls, mol, flavor):
    mf = getattr(cls(mol), flavor)()
    mf.verbose = 0
    mf.conv_tol = 1e-10
    return mf.kernel(), mf


class KnownValues(unittest.TestCase):
    def test_bundled_x2camf(self):
        self.assertIn('x2camf_c', x2camf.__file__)

    def test_shell_order_invariance(self):
        normal = _mol('cc-pvdz')
        rev = _mol({'Br': gto.load('cc-pvdz', 'Br')[::-1], 'H': 'cc-pvdz'})
        for flavor in ('x2camf', 'x2cmp'):
            e0 = _energy(spinor_hf.KRHF, normal, flavor)[0]
            e1 = _energy(spinor_hf.KRHF, rev, flavor)[0]
            self.assertAlmostEqual(e0, e1, 8, flavor)

    def test_segmented_x2cmp(self):
        mol = _mol('6-31g')
        e, mf = _energy(spinor_hf.KRHF, mol, 'x2cmp')
        self.assertTrue(mf.converged)
        h = mf.get_hcore()
        self.assertLess(abs(h - h.conj().T).max(), 1e-10)
        # loose: the unconverged atomic Dirac-HF of Br in 6-31G (see below) leaves ~1e-5 noise
        self.assertAlmostEqual(e, -2595.61943, 4)
        # Kramers symmetry is checked with def2-SVP: in 6-31G the atomic Dirac-HF of Br
        # stalls at a density change of ~1e-5 and the hcore is Kramers symmetric only to ~1e-5
        mol = _mol('def2-svp')
        e, mf = _energy(spinor_hf.KRHF, mol, 'x2cmp')
        h = mf.get_hcore()
        self.assertLess(abs(dhf.time_reversal_matrix(mol, h) - h).max(), 1e-8)
        self.assertAlmostEqual(e, -2595.683130, 5)

    def test_segmented_ghf(self):
        mol = _mol('6-31g')
        e_spinor = _energy(spinor_hf.KRHF, mol, 'x2camf')[0]
        e_ghf = _energy(ghf.GHF, mol, 'x2camf')[0]
        self.assertAlmostEqual(e_spinor, e_ghf, 9)
        self.assertAlmostEqual(e_spinor, -2594.783289, 5)


if __name__ == '__main__':
    print('Tests for X2CAMF/X2CMP with segmented or unsorted basis sets')
    unittest.main()
