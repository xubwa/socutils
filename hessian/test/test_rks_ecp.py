#!/usr/bin/env python
'''
The fast RKS path with effective core potentials (def2-SVP / def2 ECP, PBE0):
every piece against pyscf's own code in the same model, on HI and AgH.

  * md_j.apply (RIJ J, exact K by jk_multi), vxc_fast, the direct RI-J engine:
    the SCF energy of pyscf's density_fit(only_dfj=True)
  * grad.rhf_fast: pyscf's gradient
  * hessian.rhf_fast: pyscf's Hessian (no density fitting; pyscf's DF Hessian
    has no only_dfj)
  * cosx: K against the exact K on a fine grid
  * cosx.gridx_args: the heavy-atom rule must see the nuclear charge, not the
    ECP valence charge mol.atom_charges() returns (Rb 9, Sr 10, Zr 12, Mo 14,
    Ru 16, Pd 18, Hf 12, W 14, Os 16, Pt 18, La 11 all fall under HEAVY_Z = 18)

    python hessian/test/test_rks_ecp.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, dft, lib
from socutils.hessian import rhf_fast
from socutils.grad import rhf_fast as grad_fast
from socutils.scf import cosx, md_j
from socutils.dft import vxc_fast

AUX = 'def2-universal-jfit'


def _mol(atom, heavy):
    return gto.M(atom=atom, basis='def2-svp', ecp={heavy: 'def2-svp'}, verbose=0)


def _rks(mol, rij=True):
    mf = dft.RKS(mol, xc='pbe0')
    if rij:
        mf = mf.density_fit(auxbasis=AUX, only_dfj=True)
    mf.grids.level = 3
    mf.conv_tol = 1e-11
    return mf


class KnownValues(unittest.TestCase):
    def _check(self, mol):
        self.assertTrue(mol.has_ecp())
        self.assertTrue(cosx.available(mol))
        mf0 = _rks(mol)
        e0 = mf0.kernel()
        g0 = mf0.nuc_grad_method().kernel()
        # SCF energy: md_j.apply, + vxc_fast, direct builds, the direct RI-J engine
        mf1 = md_j.apply(vxc_fast.apply(_rks(mol)))
        self.assertAlmostEqual(mf1.kernel(), e0, 9)
        with lib.temporary_env(md_j, INCORE=False, RIJ_DIRECT=True):
            mf2 = md_j.apply(_rks(mol))
            self.assertAlmostEqual(mf2.kernel(), e0, 9)
            self.assertIsNotNone((getattr(mf2, '_rij_cache', None) or {}).get('direct'))
            self.assertAlmostEqual(abs(grad_fast.Gradients(mf2).kernel() - g0).max(), 0, 6)
        # gradient
        self.assertAlmostEqual(abs(grad_fast.Gradients(mf1).kernel() - g0).max(), 0, 6)
        # Hessian without density fitting
        mfn = _rks(mol, rij=False)
        mfn.kernel()
        h = rhf_fast.RKSHessian(mfn)
        h.cpscf_grids = None
        h.sum_rule = False
        self.assertAlmostEqual(abs(h.kernel() - mfn.Hessian().kernel()).max(), 0, 7)
        # COSX K on a fine grid against the exact K
        dm = mf0.make_rdm1()
        vk = mf0.get_k(mol, dm)
        k = cosx.COSX(mol, cosx.make_grids(mol, intacc=5.0, angular_grid=5))
        self.assertLess(abs(k.get_k(dm) - vk).max(), 5e-5)
        self.assertLess(abs(k.energy(dm) - numpy.einsum('ij,ji', dm, vk)) / 16, 2e-6)

    def test_hi(self):
        self._check(_mol('I 0 0 0; H 0 0.05 1.62', 'I'))

    def test_agh(self):
        self._check(_mol('Ag 0 0 0; H 0 0.04 1.62', 'Ag'))

    @unittest.expectedFailure
    def test_heavy_rule_with_ecp(self):
        '''cosx.gridx_args tests mol.atom_charges() (the ECP valence charge): the Z > HEAVY_Z rule does not fire for
        Rb, Sr, Zr ... (remove the decorator once it tests the nuclear charge)'''
        for atom, heavy in (('Rb 0 0 0; F 0 0 2.27', 'Rb'), ('Zr 0 0 0; H 0 0 1.85', 'Zr'), ('Pt 0 0 0; H 0 0 1.53', 'Pt')):
            mol = gto.M(atom=atom, basis='def2-svp', ecp={heavy: 'def2-svp'}, spin=None, verbose=0)
            self.assertGreater(gto.charge(heavy), cosx.HEAVY_Z)
            for level in (0, 1, 2):
                self.assertEqual(cosx.gridx_args(mol, level)['angular_grid'], cosx.HEAVY_ANGULAR)

    def test_heavy_rule_all_electron(self):
        mol = gto.M(atom='K 0 0 0; H 0 0 2.24', basis='def2-svp', verbose=0)
        for level in (0, 1, 2):
            self.assertEqual(cosx.gridx_args(mol, level)['angular_grid'], cosx.HEAVY_ANGULAR)
        mol = gto.M(atom='Cl 0 0 0; H 0 0 1.28', basis='def2-svp', verbose=0)
        self.assertEqual(cosx.gridx_args(mol, 2)['angular_grid'], cosx.GRIDX[2]['angular_grid'])


if __name__ == '__main__':
    print('Fast RKS path with ECP atoms against pyscf')
    unittest.main()
