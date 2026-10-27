#!/usr/bin/env python
'''
socutils.hessian.rhf_fast (unique-quartet 2e terms, contracted XC Hessian,
first-order Fock matrices in one pass, multi-density CPHF J / K, MO-basis
XC response) against pyscf's RHF / RKS Hessians, which compute the same
quantities, and the C drivers against explicit integral contractions.

    python hessian/test/test_rhf_fast.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf, dft
from socutils.hessian import rhf_fast
from socutils.scf import jk_multi

WATER2 = '''
O  0.000000  0.000000  0.000000
H  0.756950  0.000000  0.585882
H -0.756950  0.000000  0.585882
O  0.000000  0.000000  3.000000
H  0.876934  0.133858  3.359583
H -0.280961 -0.841423  3.359583
'''


def _mf(mol, xc):
    if xc == 'hf':
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level = 2
    mf.conv_tol = 1e-12
    return mf.run()


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=WATER2, basis='6-31g*', verbose=0)

    def test_fock1_jk(self):
        mol = gto.M(atom='O 0 0 0; H .757 0 .586; H -.757 0 .586', basis='6-31g', verbose=0)
        nao = mol.nao
        dm = numpy.random.RandomState(1).randn(nao, nao)
        dm = dm + dm.T
        vj, vk = rhf_fast.fock1_jk(mol, dm)
        eri1 = mol.intor('int2e_ip1', comp=3)
        for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
            g = numpy.zeros((3, nao, nao, nao, nao))
            g[:, p0:p1] -= eri1[:, p0:p1]
            g = g + g.transpose(0, 2, 1, 3, 4)
            g = g + g.transpose(0, 3, 4, 1, 2)
            self.assertAlmostEqual(abs(vj[ia] - numpy.einsum('xabcd,cd->xab', g, dm)).max(), 0, 11)
            self.assertAlmostEqual(abs(vk[ia] - numpy.einsum('xabcd,bc->xad', g, dm)).max(), 0, 11)

    def test_hess2e_cart(self):
        # combined Cartesian kernel vs the libcint-based unique-quartet driver
        from socutils.hessian import ghf as ghf_hess
        for basis in ('cc-pvdz', 'cc-pvtz'):
            mol = gto.M(atom='O 0 0 0; H .757 .1 .586; H -.757 0 .586', basis=basis, verbose=0)
            mf = _mf(mol, 'hf')
            dm = mf.make_rdm1()
            e = rhf_fast.hess2e_jk_cart(mol, dm, [(1, 0), (0, 1)], mode=1)
            vj, vk = mf.get_jk(mol, dm)
            self.assertAlmostEqual(e[0], .5 * numpy.einsum('ij,ji', dm, vj), 9)
            self.assertAlmostEqual(e[1], .25 * numpy.einsum('ij,ji', dm, vk), 9)
            h = rhf_fast.hess2e_jk_cart(mol, dm, [(1, 0), (0, 1)])
            ref = ghf_hess.deriv2_energies_direct(
                mol, [(dm, 'ba', dm, 'dc', .5, 0), (dm, 'bc', dm, 'da', .25, 1)])
            self.assertAlmostEqual(abs(h - ref).max(), 0, 9)

    def test_fxc_batched(self):
        mol = self.mol
        rs = numpy.random.RandomState(2)
        for xc in ('lda,vwn', 'pbe'):
            mf = _mf(mol, xc)
            ni = mf._numint
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 0)
            dms = rs.randn(5, mol.nao, mol.nao) * 1e-2
            dms = dms + dms.transpose(0, 2, 1)
            ref = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms, 0, 1, rho0, vxc, fxc)
            v = rhf_fast._FxcBatched(ni)(mol, mf.grids, mf.xc, None, dms, 0, 1, rho0, vxc, fxc)
            self.assertAlmostEqual(abs(v - ref).max(), 0, 12)

    def test_fxc_batched_subsets(self):
        # a water chain: most grid blocks see a fraction of the AOs (block-local subsets)
        atom = '; '.join('O %g 0 0; H %g .757 .586; H %g -.757 .586' % (3. * i, 3. * i, 3. * i) for i in range(4))
        mol = gto.M(atom=atom, basis='6-31g', verbose=0)
        mf = _mf(mol, 'pbe')
        ni = mf._numint
        nt = mf.grids.non0tab
        w = mol.ao_loc_nr()[1:] - mol.ao_loc_nr()[:-1]
        self.assertLess(((nt > 0) * w).sum(axis=1).mean(), .7 * mol.nao)
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 0)
        dms = numpy.random.RandomState(3).randn(6, mol.nao, mol.nao) * 1e-2
        dms = dms + dms.transpose(0, 2, 1)
        ref = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms, 0, 1, rho0, vxc, fxc)
        v = rhf_fast._FxcBatched(ni)(mol, mf.grids, mf.xc, None, dms, 0, 1, rho0, vxc, fxc)
        self.assertAlmostEqual(abs(v - ref).max(), 0, 12)

    def test_xc_deriv1_batched(self):
        from pyscf.hessian import rks as rks_hess
        for xc in ('lda,vwn', 'pbe'):
            mf = _mf(self.mol, xc)
            h = rhf_fast.RKSHessian(mf)
            ref = rks_hess._get_vxc_deriv1(h, mf.mo_coeff, mf.mo_occ, 4000)
            v = rhf_fast.xc_deriv1_batched(h, mf.mo_coeff, mf.mo_occ, 4000)
            self.assertAlmostEqual(abs(v - ref).max(), 0, 11)

    def test_jk_multi(self):
        mol = self.mol
        rs = numpy.random.RandomState(0)
        d = rs.randn(5, mol.nao, mol.nao)
        rj, rk = scf.hf.get_jk(mol, d, hermi=0)
        vj, vk = jk_multi.get_jk(mol, d)
        self.assertAlmostEqual(abs(vj - rj).max(), 0, 10)
        self.assertAlmostEqual(abs(vk - rk).max(), 0, 10)
        d = d + d.transpose(0, 2, 1)
        rj, rk = scf.hf.get_jk(mol, d, hermi=1)
        vj, vk = jk_multi.get_jk(mol, d, hermi=1)
        self.assertAlmostEqual(abs(vj - rj).max(), 0, 10)
        self.assertAlmostEqual(abs(vk - rk).max(), 0, 10)

    def _check(self, mol, xc):
        mf = _mf(mol, xc)
        ref = mf.Hessian().kernel()
        cls = rhf_fast.RHFHessian if xc == 'hf' else rhf_fast.RKSHessian
        h = cls(mf).kernel()
        self.assertAlmostEqual(abs(h - ref).max(), 0, 9)

    def test_rhf(self):
        self._check(self.mol, 'hf')

    def test_lda(self):
        self._check(self.mol, 'lda,vwn')

    def test_pbe(self):
        self._check(self.mol, 'pbe')

    def test_pbe0(self):
        self._check(self.mol, 'pbe0')

    def test_tpss(self):
        self._check(self.mol, 'tpss')

    def test_mo_xc_response(self):
        # nao > 8 nocc: the MO-basis GGA response of gen_vind is active
        mol = gto.M(atom='O 0 0 0; H .757 0 .586; H -.757 0 .586', basis='cc-pvtz', verbose=0)
        self.assertGreater(mol.nao, 8 * mol.nelectron // 2)
        self._check(mol, 'b3lyp')


if __name__ == '__main__':
    unittest.main()
