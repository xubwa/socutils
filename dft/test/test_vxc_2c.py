#!/usr/bin/env python
'''The fused two-component Vxc (socutils.dft.vxc_2c) against the reference paths
(pyscf's collinear NumInt2C.nr_vxc, socutils.dft.noncollinear ncol / mcol).'''
import unittest
import numpy
from pyscf import gto, lib
from socutils.dft import dft, vxc_2c, noncollinear
from socutils.scf import spinor_hf


def setUpModule():
    global mol, dm_kr, dm_ku
    mol = gto.M(atom='Au 0 0 0; H 0 0 1.524', basis='x2c-svpall', verbose=0)
    mf = dft.RKS(mol, xc='pbe')
    dm = mf.get_init_guess()
    ni = mf._numint
    dm_kr = ni._dm_to_ghf(dm)                      # Kramers-even GHF density
    rng = numpy.random.default_rng(5)
    n2 = dm_kr.shape[0]
    x = rng.standard_normal((n2, n2)) + 1j * rng.standard_normal((n2, n2))
    s = mol.intor('int1e_ovlp')
    # a Hermitian, spin-polarised, non-collinear perturbation that decays like a density
    nao = mol.nao
    p = numpy.zeros((n2, n2), dtype=complex)
    p[:nao, :nao] = s @ (x[:nao, :nao] + x[:nao, :nao].conj().T) @ s * 2e-3
    p[nao:, nao:] = -s @ (x[nao:, nao:] + x[nao:, nao:].conj().T) @ s * 1e-3
    p[:nao, nao:] = s @ x[:nao, nao:] @ s * 2e-3
    p[nao:, :nao] = p[:nao, nao:].conj().T
    dm_ku = dm_kr + p


def tearDownModule():
    global mol, dm_kr, dm_ku
    del mol, dm_kr, dm_ku


def _both(ni, grids, xc, dm):
    vxc_2c.ENABLE = False
    try:
        ref = noncollinear.NumInt2C.nr_vxc(ni, mol, grids, xc, dm)     # the GHF-form entry
    finally:
        vxc_2c.ENABLE = True
    out = vxc_2c.nr_vxc(ni, mol, grids, xc, dm)
    return ref, out


class KnowValues(unittest.TestCase):
    def _check(self, collinear, xc, dm, tol=1e-11):
        mf = dft.RKS(mol, xc=xc)
        ni = mf._numint
        ni.collinear = collinear
        mf.grids.atom_grid = (60, 194)
        mf.grids.build(with_non0tab=True)
        ref, out = _both(ni, mf.grids, xc, dm)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out[0], ref[0], 9)
        self.assertAlmostEqual(out[1], ref[1], 10)
        err = abs(out[2] - ref[2]).max() / abs(ref[2]).max()
        self.assertLess(err, tol, (collinear, xc))

    def test_col_kramers_restricted(self):
        for xc in ('lda,vwn', 'pbe'):
            self._check('col', xc, dm_kr)

    def test_col_unrestricted(self):
        for xc in ('lda,vwn', 'pbe'):
            self._check('col', xc, dm_ku)

    def test_ncol(self):
        for xc in ('lda,vwn', 'pbe'):
            self._check('ncol', xc, dm_ku)
            self._check('ncol', xc, dm_kr)

    def test_mcol_lda(self):
        self._check('mcol', 'lda,vwn', dm_ku)

    def test_mcol_gga_not_fused(self):
        mf = dft.RKS(mol, xc='pbe')
        mf._numint.collinear = 'mcol'
        mf.grids.build(with_non0tab=True)
        self.assertIsNone(vxc_2c.nr_vxc(mf._numint, mol, mf.grids, 'pbe', dm_ku))

    def test_spinor_scf_unchanged(self):
        es = []
        for on in (False, True):
            vxc_2c.ENABLE = on
            mf = dft.RKS(mol, xc='pbe0').x2camf()
            mf.conv_tol = 1e-10
            es.append(mf.kernel())
        vxc_2c.ENABLE = True
        self.assertAlmostEqual(es[0], es[1], 9)


if __name__ == '__main__':
    unittest.main()
