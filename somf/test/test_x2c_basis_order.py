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
2. cc-pVDZ written as a segmented basis (one shell per contracted function,
   so primitives repeat across shells as in segmented sets): the same function
   space, hence the same X2CAMF and X2CMP energies as the general contraction
   (X2CMP aborted in zquatev before, with a garbage, non-Hermitian hcore); the
   hcore is Hermitian and Kramers symmetric.
3. GHF (spin-orbital, via spinor2sph) equals the spinor result.

cc-pVXZ throughout: in 6-31G the atomic Dirac-HF of Br does not converge.
'''

import unittest
import numpy
from pyscf import gto
from pyscf.scf import dhf
from socutils.scf import spinor_hf, ghf
import x2camf   # after socutils, which puts the bundled x2camf first on sys.path


def _segmented(basis):
    '''A generally contracted basis as a segmented one: one shell per
    contracted function.'''
    out = []
    for shell in basis:
        l, prims = shell[0], shell[1:]
        for i in range(1, len(prims[0])):
            out.append([l] + [[p[0], p[i]] for p in prims if p[i] != 0])
    return out


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

    def test_segmented(self):
        general = _mol('cc-pvdz')
        segmented = _mol({e: _segmented(gto.load('cc-pvdz', e)) for e in ('H', 'Br')})
        self.assertGreater(segmented.nbas, general.nbas)
        for flavor in ('x2camf', 'x2cmp'):
            e0 = _energy(spinor_hf.KRHF, general, flavor)[0]
            e1, mf = _energy(spinor_hf.KRHF, segmented, flavor)
            self.assertTrue(mf.converged)
            self.assertAlmostEqual(e0, e1, 8, flavor)
            h = mf.get_hcore()
            self.assertLess(abs(h - h.conj().T).max(), 1e-10)
            self.assertLess(abs(dhf.time_reversal_matrix(segmented, h) - h).max(), 1e-8)

    def test_ghf(self):
        mol = _mol({e: _segmented(gto.load('cc-pvdz', e)) for e in ('H', 'Br')})
        e_spinor = _energy(spinor_hf.KRHF, mol, 'x2camf')[0]
        e_ghf = _energy(ghf.GHF, mol, 'x2camf')[0]
        self.assertAlmostEqual(e_spinor, e_ghf, 9)


if __name__ == '__main__':
    print('Tests for X2CAMF/X2CMP with segmented or unsorted basis sets')
    unittest.main()
