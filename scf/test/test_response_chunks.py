#!/usr/bin/env python
'''The memory-chunked response J / K (spinor_jk COSX, RESPONSE_MEM_FRAC) and f_xc (fxc_2c.MEM_FRAC): identical to
the unchunked results.'''
import unittest
import numpy
from pyscf import gto
from socutils.dft import dft, fxc_2c, noncollinear
from socutils.scf import spinor_jk


def setUpModule():
    global mol, mf
    mol = gto.M(atom='Au 0 0 0; H 0 0 1.524', basis='x2c-svpall', verbose=0)
    mf = dft.RKS(mol, xc='pbe0').x2camf()
    mf.rijcosx = True
    mf.conv_tol = 1e-10
    mf.kernel()


def tearDownModule():
    global mol, mf
    del mol, mf


class KnowValues(unittest.TestCase):
    def test_cosx_chunks(self):
        nao = mol.nao
        rng = numpy.random.default_rng(3)
        d = (rng.standard_normal((5, 2 * nao, 2 * nao)) + 1j * rng.standard_normal((5, 2 * nao, 2 * nao))) * .05
        ref = spinor_jk.get_jk_ghf(mf, mol, d, hermi=0)[1]
        frac = spinor_jk.RESPONSE_MEM_FRAC
        spinor_jk.RESPONSE_MEM_FRAC = 1e-9          # one density per chunk
        try:
            out = spinor_jk.get_jk_ghf(mf, mol, d, hermi=0)[1]
        finally:
            spinor_jk.RESPONSE_MEM_FRAC = frac
        self.assertLess(abs(out - ref).max(), 1e-12 * abs(ref).max())

    def test_fxc_chunks(self):
        ni = mf._numint
        ni.collinear = 'ncol'
        mo = ni.U @ mf.mo_coeff
        rho0, vxc, fxc = noncollinear.NumInt2C.cache_xc_kernel(ni, mol, mf.grids, mf.xc, mo, mf.mo_occ, 1)
        n2 = mo.shape[0]
        rng = numpy.random.default_rng(4)
        d = (rng.standard_normal((5, n2, n2)) + 1j * rng.standard_normal((5, n2, n2))) * .05
        ref = fxc_2c.nr_fxc(ni, mol, mf.grids, mf.xc, None, d, 0, rho0, vxc, fxc, 4000)
        frac = fxc_2c.MEM_FRAC
        fxc_2c.MEM_FRAC = 1e-9
        try:
            out = fxc_2c.nr_fxc(ni, mol, mf.grids, mf.xc, None, d, 0, rho0, vxc, fxc, 4000)
        finally:
            fxc_2c.MEM_FRAC = frac
        self.assertLess(abs(out - ref).max(), 1e-12 * abs(ref).max())


if __name__ == '__main__':
    unittest.main()
