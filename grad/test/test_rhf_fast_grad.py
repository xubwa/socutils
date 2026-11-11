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
        derivative; it tracks the largest error): the K gradient on cosx.GRAD_GRID_DENSE rather than the SCF's
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
        self.assertLess(net(hf, cosx.GRAD_GRID_DENSE, None), 3e-5)              # 0.8e-5
        self.assertLess(net(hf, 'auto', None), 6e-5)                            # the default, GRIDX3
        self.assertGreater(net(hf, None, None), 1.5e-4)                         # 2.7e-4: K gradient on GRIDX2
        lif = 'Li 0 0 0; F 0 0 1.62'
        self.assertLess(net(lif, cosx.GRAD_GRID_DENSE, None), 1e-4)             # 3.2e-5
        self.assertGreater(net(lif, cosx.GRAD_GRID_DENSE, radi.treutler_atomic_radii_adjust), 2.5e-4)  # 4.0e-4

    def test_cosx_grad_grid_by_accuracy(self):
        '''the K-gradient grid follows the accuracy asked for (cosx.GRAD_GRIDS): GRIDX3 for a plain gradient,
        the dense tier for a tight max-force threshold, GRIDX2 while the forces of an optimisation are large'''
        from socutils.scf import cosx
        from socutils.dft import vxc_fast, grids as sgrids
        mol = gto.M(atom='H 0 0 0; F 0 0 .96', basis='def2-svp', verbose=0)
        if not cosx.available(mol):
            self.skipTest('no COSX kernel')
        mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='def2-universal-jfit', only_dfj=True)
        mf.grids = sgrids.Grids(mol)
        mf = cosx.apply(vxc_fast.apply(mf))
        mf.conv_tol = 1e-10
        mf.kernel()
        self.assertEqual(cosx.grad_grid(mf)[1], 3)
        g = rhf_fast.Gradients(mf)
        g3 = g.kernel()
        self.assertEqual(mf._cosx['grad_key'], 3)
        self.assertAlmostEqual(mf._cosx['last_gmax'], abs(g3).max(), 12)
        # a tight threshold: the dense tier; its gradient differs from GRIDX3's at the 1e-5 level
        cosx.set_grad_accuracy(mf, conv=3e-5)
        self.assertIsInstance(cosx.grad_grid(mf)[1], str)
        gd = rhf_fast.Gradients(mf).kernel()
        self.assertLess(abs(gd - g3).max(), 6e-5)
        self.assertGreater(abs(gd - g3).max(), 1e-7)
        # a loose one, and the argument of a single call
        cosx.set_grad_accuracy(mf, conv=4.5e-3)
        self.assertEqual(cosx.grad_grid(mf)[1], 2)
        self.assertEqual(cosx.grad_grid(mf, accuracy=1e-5)[1], 3)
        # auto: the grid of the threshold near convergence, a coarser one while the last max force is large
        cosx.set_grad_accuracy(mf, conv=3e-4, auto=True)
        mf._cosx['last_gmax'] = 2e-2
        self.assertEqual(cosx.grad_grid(mf)[1], 2)
        mf._cosx['last_gmax'] = 5e-4
        self.assertEqual(cosx.grad_grid(mf)[1], 3)
        g2 = rhf_fast.Gradients(mf)
        g2.cosx_grad_accuracy = 3e-4
        g2.kernel()
        self.assertEqual(mf._cosx['grad_key'], 2)

    def test_size_adjustment_rules(self):
        '''the Becke cells' size adjustment on the XC grid, by the net force of the fixed-grid gradient (RIJ +
        exact K, so the XC grid alone): pairs with hydrogen only (the default) is right for Li-F but loses the
        adjustment the C-Br bond needs; the electronegativity rule (grids.polar_pair_atomic_radii_adjust)
        keeps both.  Net force / error against pyscf's grid-response gradient, 1e-5 Eh/bohr, all pairs / H
        pairs / polar rule: CH3Br 2.6 / 10.9 / 2.6 (error 2.2 / 10.6 / 2.2), LiF 50 / 7.2 / 7.2 (49 / 3.8 / 3.8)'''
        from pyscf.dft import radi
        from socutils.scf import md_j
        from socutils.dft import vxc_fast, grids as sgrids
        ch3br = '''C -0.05234690 -0.09282871 -0.01095316; Br 3.69644722 0.02334271 -0.00337690
                   H -0.74076981 -0.03988177 2.04294994; H -0.59904116 1.74909446 -1.03617690
                   H -0.55310689 -1.62853108 -1.07434140'''
        lif = 'Li -0.03606235 -0.05829594 -0.01354086; F 2.96231605 0.03834494 0.00121771'

        def net(atom, adjust):
            mol = gto.M(atom=atom, unit='bohr', basis='def2-svp', verbose=0)
            mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='def2-universal-jfit', only_dfj=True)
            mf.grids = sgrids.Grids(mol)
            mf.grids.radii_adjust = adjust
            mf = md_j.apply(vxc_fast.apply(mf))
            mf.conv_tol = 1e-10
            mf.kernel()
            g = rhf_fast.Gradients(mf)
            g.kernel()
            return g.net_force
        allp, hp, polar = (radi.treutler_atomic_radii_adjust, sgrids.h_pair_atomic_radii_adjust,
                           sgrids.polar_pair_atomic_radii_adjust)
        self.assertLess(net(ch3br, polar), 5e-5)
        self.assertGreater(net(ch3br, hp), 8e-5)
        self.assertAlmostEqual(net(ch3br, polar), net(ch3br, allp), 9)
        self.assertLess(net(lif, polar), 1.5e-4)
        self.assertGreater(net(lif, allp), 3e-4)
        self.assertAlmostEqual(net(lif, polar), net(lif, hp), 9)


if __name__ == '__main__':
    unittest.main()
