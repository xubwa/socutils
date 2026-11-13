#!/usr/bin/env python
'''The batched two-component f_xc of a closed-shell reference (socutils.dft.fxc_2c) against the
noncollinear path: random complex non-Hermitian GHF trial densities, LDA / GGA, ncol and mcol, and
TDA excitation energies.'''
import unittest
import numpy
from pyscf import gto
from socutils.dft import dft, fxc_2c, noncollinear
from socutils.tdscf import gks


def setUpModule():
    global mol, mfs
    mol = gto.M(atom='Au 0 0 0; H 0 0 1.524', basis='x2c-svpall', verbose=0)
    mfs = {}
    for xc in ('lda,vwn', 'pbe0'):
        mf = dft.RKS(mol, xc=xc).x2camf()
        mf.grids.atom_grid = (60, 194)
        mf.conv_tol = 1e-10
        mf.kernel()
        mfs[xc] = mf


def tearDownModule():
    global mol, mfs
    del mol, mfs


class KnowValues(unittest.TestCase):
    def _check(self, xc, coll):
        mf = mfs[xc]
        ni = mf._numint
        ni.collinear = coll
        ni.spin_samples = 50
        mo = ni.U @ mf.mo_coeff            # the GHF MOs (scalar AO x spin)
        rho0, vxc, fxc = noncollinear.NumInt2C.cache_xc_kernel(ni, mol, mf.grids, mf.xc, mo, mf.mo_occ, 1)
        n2 = mo.shape[0]
        rng = numpy.random.default_rng(2)
        dms = (rng.standard_normal((3, n2, n2)) + 1j * rng.standard_normal((3, n2, n2))) * .05
        fxc_2c.MODE = 'old'
        ref = noncollinear.NumInt2C.nr_fxc(ni, mol, mf.grids, mf.xc, None, dms, 0, 0, 0, rho0, vxc, fxc)
        fxc_2c.MODE = 'auto'
        out = fxc_2c.nr_fxc(ni, mol, mf.grids, mf.xc, None, dms, 0, rho0, vxc, fxc)
        self.assertIsNotNone(out)
        self.assertLess(abs(out - ref).max() / abs(ref).max(), 1e-11, (xc, coll))

    def test_ncol(self):
        for xc in ('lda,vwn', 'pbe0'):
            self._check(xc, 'ncol')

    def test_mcol(self):
        self._check('pbe0', 'mcol')

    def test_mo_form(self):
        from pyscf import lib
        for xc in ('lda,vwn', 'pbe0'):
            mf = mfs[xc]
            ni = mf._numint
            ni.collinear = 'ncol'
            mo = ni.U @ mf.mo_coeff
            occ = mf.mo_occ > 0
            orbo, orbv = mo[:, occ], mo[:, ~occ]
            rho0, vxc, fxc = noncollinear.NumInt2C.cache_xc_kernel(ni, mol, mf.grids, mf.xc, mo, mf.mo_occ, 1)
            xctype = ni._xc_type(mf.xc)
            iso = fxc_2c.isotropic_kernel(fxc, xctype)
            rng = numpy.random.default_rng(4)
            z = rng.standard_normal((3, orbo.shape[1], orbv.shape[1])) \
                + 1j * rng.standard_normal((3, orbo.shape[1], orbv.shape[1]))
            dms = lib.einsum('xov,pv,qo->xpq', z, orbv, orbo.conj())
            v = fxc_2c.nr_fxc(ni, mol, mf.grids, mf.xc, None, dms, 0, rho0, vxc, fxc)
            ref = lib.einsum('xpq,qo,pv->xov', v, orbo, orbv.conj())
            out = fxc_2c.mo_response(ni, mol, mf.grids, iso, orbo, orbv, z, xctype)
            self.assertLess(abs(out - ref).max() / abs(ref).max(), 1e-11, xc)

    def test_spin_samples_default(self):
        mf = dft.UKS(mol, xc='pbe')
        ni = mf._numint
        self.assertEqual(ni.spin_samples, noncollinear.SPIN_SAMPLES)
        self.assertEqual(noncollinear.SPIN_SAMPLES, 110)
        ni.spin_samples = 266               # the user's setting wins
        self.assertEqual(ni.spin_samples, 266)
        self.assertEqual(dft.UKS(mol, xc='pbe')._numint.spin_samples, 110)

    def test_tda(self):
        es = []
        for mode in ('old', 'ao', 'mo'):
            fxc_2c.MODE = mode
            td = gks.TDA(mfs['pbe0'])
            td.nstates = 4
            es.append(td.kernel()[0])
        fxc_2c.MODE = 'auto'
        self.assertLess(abs(es[0] - es[1]).max(), 1e-9)
        self.assertLess(abs(es[0] - es[2]).max(), 1e-9)


if __name__ == '__main__':
    unittest.main()
