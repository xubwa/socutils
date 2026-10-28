#!/usr/bin/env python
'''
socutils.grad.rhf_fast (exact 2e part from the order-1 unique-quartet kernel,
XC part without derivative matrices) against pyscf's RHF / RKS gradients,
which compute the same quantities.

    python grad/test/test_rhf_fast_grad.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf, dft
from socutils.grad import rhf_fast
from socutils.hessian import rhf_fast as hess_fast

WATER2 = '''
O  0.000000  0.000000  0.000000
H  0.756950  0.000000  0.585882
H -0.756950  0.000000  0.585882
O  0.000000  0.000000  3.000000
H  0.876934  0.133858  3.359583
H -0.280961 -0.841423  3.359583
'''


def _mf(mol, xc, df=None):
    if xc == 'hf':
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level = 2
    if df is not None:
        mf = mf.density_fit(auxbasis='def2-universal-jfit', only_dfj=(df == 'onx'))
    mf.conv_tol = 1e-12
    return mf.run()


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=WATER2, basis='cc-pvdz', verbose=0)

    def _check(self, xc, df=None, places=9):
        mf = _mf(self.mol, xc, df)
        ref = mf.nuc_grad_method().kernel()
        g = rhf_fast.Gradients(mf)
        de = g.kernel()
        self.assertIsNotNone(g._de_fast)          # the fast path ran
        self.assertAlmostEqual(abs(de - ref).max(), 0, places)

    def test_grad2e_energy(self):
        # mode 1 of the order-1 kernel: the 2e energies
        mf = _mf(self.mol, 'hf')
        dm = mf.make_rdm1()
        vj, vk = mf.get_jk(self.mol, dm)
        e = hess_fast.grad2e_jk_cart(self.mol, dm, [(1, 0), (0, 1)], mode=1)
        self.assertAlmostEqual(e[0], .5 * numpy.einsum('ij,ji', vj, dm), 9)
        self.assertAlmostEqual(e[1], .25 * numpy.einsum('ij,ji', vk, dm), 9)

    def test_md_j(self):
        # MD J engine: energy and gradient against the exact J and the Rys kernel;
        # the symmetric propane (coincident pair centres, Boys functions at t ~ 0)
        propane = '''C 0 0 0; C 1.249173 0.884052 0; C 2.498346 0 0
                     H -0.628924 0 0.890229; H -0.628924 0 -0.890229
                     H 1.249173 1.517 0.890229; H 1.249173 1.517 -0.890229
                     H 3.127270 0 0.890229; H 3.127270 0 -0.890229
                     H -0.512 -0.95 0; H 3.010 -0.95 0'''
        for mol in (self.mol, gto.M(atom=propane, basis='sto-3g', verbose=0)):
            mf = _mf(mol, 'hf')
            dm = mf.make_rdm1()
            g, e = hess_fast.md_j_grad(mol, dm)
            vj = mf.get_j(mol, dm)
            self.assertAlmostEqual(e, .5 * numpy.einsum('ij,ji', vj, dm), 9)
            ref = hess_fast.grad2e_jk_cart(mol, dm, [(1, 0)])[0]
            self.assertAlmostEqual(abs(g - ref).max(), 0, 9)

    def test_rhf(self):
        self._check('hf')

    def test_lda(self):
        self._check('lda,vwn')

    def test_pbe(self):
        self._check('pbe')

    def test_pbe0(self):
        self._check('pbe0')

    def test_tpss(self):
        self._check('tpss')

    def test_rij_pbe(self):
        self._check('pbe', 'rij')

    def test_rijonx_pbe0(self):
        self._check('pbe0', 'onx')


if __name__ == '__main__':
    unittest.main()
