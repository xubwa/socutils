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
from pyscf import lib, gto, scf, dft
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
        # tight cutoffs: the comparison with pyscf to 1e-9 (the defaults are looser)
        with lib.temporary_env(rhf_fast, GRAD2E_TOL=1e-13, GRAD2E_PRIM_TOL=1e-15):
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


    def test_cosx_fixed_grid_terms(self):
        '''the fixed-grid terms of the default RIJCOSX gradient, by the net force (zero for the exact
        derivative; it tracks the largest error): the K gradient on cosx.GRAD_GRID rather than the SCF's
        GRIDX2 (HF: 1.6e-5 against 2.7e-4 Eh/bohr from a grid-response reference), and the Becke cells'
        size adjustment by the electronegativity rule (LiF: 4.3e-5 against all pairs on the XC grid, net 4.0e-4)'''
        from pyscf.dft import radi
        from socutils.scf import cosx
        from socutils.dft import vxc_fast, grids as sgrids

        def net(atom, grad_grid, adjust):
            mol = gto.M(atom=atom, basis='def2-svp', verbose=0)
            with lib.temporary_env(cosx, GRAD_GRID=grad_grid):
                mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='def2-universal-jfit', only_dfj=True)
                mf.grids = sgrids.Grids(mol)
                if adjust is not None:
                    mf.grids.radii_adjust = adjust
                mf = cosx.apply(vxc_fast.apply(mf))
                mf.conv_tol = 1e-10
                mf.kernel()
                g = rhf_fast.Gradients(mf)
                g.kernel()
                return g.net_force
        if not cosx.available(gto.M(atom='H 0 0 0; F 0 0 .96', basis='def2-svp', verbose=0)):
            self.skipTest('no COSX kernel')
        self.assertEqual(sgrids.Grids.radii_adjust, sgrids.polar_pair_atomic_radii_adjust)
        hf = 'H 0 0 0; F 0 0 .96'
        self.assertLess(net(hf, cosx.GRAD_GRID, None), 3e-5)                    # 0.8e-5
        self.assertGreater(net(hf, None, None), 1.5e-4)                         # 2.7e-4: K gradient on GRIDX2
        lif = 'Li 0 0 0; F 0 0 1.62'
        self.assertLess(net(lif, cosx.GRAD_GRID, None), 1e-4)                   # 3.2e-5
        self.assertGreater(net(lif, cosx.GRAD_GRID, radi.treutler_atomic_radii_adjust), 2.5e-4)  # 4.0e-4

if __name__ == '__main__':
    unittest.main()
