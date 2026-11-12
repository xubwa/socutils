#!/usr/bin/env python
'''RIJ / COSX of two-component densities (socutils.scf.spinor_jk, mf.rijcosx):
COSX K of non-symmetric real densities against the exact K, and the spinor
X2CAMF SCF energies of the fast modes against the exact path.'''
import unittest
import numpy
from pyscf import gto, scf
from socutils.scf import cosx, spinor_hf, spinor_jk


def setUpModule():
    global mol, mol_hi
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvdz', verbose=0)
    mol_hi = gto.M(atom='H 0 0 0; I 0 0 1.609', basis='x2c-svpall', verbose=0)


def tearDownModule():
    global mol, mol_hi
    del mol, mol_hi


def _dens(n, nao, seed):
    rng = numpy.random.default_rng(seed)
    x = rng.standard_normal((n, nao, nao)) * .1
    s = x + x.transpose(0, 2, 1)
    a = x - x.transpose(0, 2, 1)
    g = rng.standard_normal((n, nao, nao)) * .1
    return s, a, g


@unittest.skipUnless(cosx.available(gto.M(atom='He', basis='sto3g', verbose=0)), 'COSX kernel not built')
class KnowValues(unittest.TestCase):
    def _check(self, m, nset):
        k = cosx.COSX(m, cosx.make_grids(m, **cosx.gridx_args(m, 2)))
        nao = m.nao
        for kind, d in zip('SAG', _dens(nset, nao, 7)):
            ref = scf.hf.get_jk(m, d, hermi=0, with_j=False)[1]
            vk = k.get_k(d, hermi=0)
            # random densities are a hard case (no decay with the distance): GRIDX2 ~1e-3
            err = abs(vk - ref).max() / abs(ref).max()
            self.assertLess(err, 1e-2, kind)
            sym = (vk + vk.transpose(0, 2, 1)) * .5
            asym = (vk - vk.transpose(0, 2, 1)) * .5
            if kind == 'S':
                self.assertLess(abs(asym).max() / abs(ref).max(), 1e-2)
            elif kind == 'A':
                self.assertLess(abs(sym).max() / abs(ref).max(), 1e-2)
            # the energy-only path: tr(X K[X]) = sum X^T o K (exchange_energy's convention)
            e = k.energy(d)
            e_ref = numpy.einsum('xji,xij->x', d, vk)
            self.assertAlmostEqual(abs(e - e_ref).max() / abs(e_ref).max(), 0, 4)

    def test_cosx_nonsym_single(self):
        self._check(mol, 1)

    def test_cosx_nonsym_multi(self):
        self._check(mol, 4)

    def test_cosx_nonsym_heavy(self):
        self._check(mol_hi, 4)

    def test_spinor_scf_modes(self):
        es = {}
        for m in ('', 'rij', 'cosx', 'rijcosx'):
            mf = spinor_hf.RHF(mol_hi).x2camf()
            mf.rijcosx = m
            mf.conv_tol = 1e-9
            es[m] = mf.kernel()
        # the AutoAux RIJ error of HI (scalar X2C RHF: -2.84e-4)
        self.assertLess(abs(es['rij'] - es['']), 5e-4)
        self.assertLess(abs(es['cosx'] - es['']), 2e-4)
        self.assertLess(abs(es['rijcosx'] - es['']), 6e-4)
        # the split adds up: RI and COSX errors nearly independent
        self.assertAlmostEqual(es['rijcosx'] - es[''], (es['rij'] - es['']) + (es['cosx'] - es['']), 5)

    def test_cosx_scf_density_pieces(self):
        # the real pieces of a Kramers-restricted density: K to ~1e-4, the exchange energy of the
        # final-grid path against the exact one
        mf = spinor_hf.RHF(mol_hi).x2camf()
        mf.kernel()
        dm = spinor_hf.spinor2sph(mol_hi, mf.make_rdm1())
        nao = mol_hi.nao
        vk_ref = spinor_jk.get_jk_ghf(mf, mol_hi, dm, with_j=False)[1]
        mf.rijcosx = 'cosx'
        vk = spinor_jk.get_jk_ghf(mf, mol_hi, dm, with_j=False)[1]
        self.assertLess(abs(vk - vk_ref).max() / abs(vk_ref).max(), 1e-4)
        e_ref = numpy.einsum('ij,ji', dm, vk_ref).real
        for lev in (2, 3):
            e = spinor_jk.exchange_energy(mf, mol_hi, dm, lev)
            self.assertLess(abs(e - e_ref), 5e-4)
        e2 = numpy.einsum('ij,ji', dm, vk).real
        self.assertAlmostEqual(spinor_jk.exchange_energy(mf, mol_hi, dm, 2), e2, 6)

    def test_heavy_radial_grid(self):
        # dyall-v2z Au: the plain GRIDX2 radial grid misses the tight s primitives (S_g singular); the
        # spinor COSX grid (COSX_HEAVY_INTACC on Au) represents the basis
        from pyscf import lib
        m = gto.M(atom='Au 0 0 0; H 0 0 1.524', basis='dyall-v2z', verbose=0)
        with lib.temporary_env(cosx, HEAVY_RADIAL_Z=200):        # the plain radial rule
            k = cosx.COSX(m, cosx.make_grids(m, **cosx.gridx_args(m, 2))).build()
        self.assertLess(spinor_jk.fit_quality(k)[0], 1e-6)
        for args in (spinor_jk.grid_args(m, 2), cosx.gridx_args(m, 2)):    # spinor rule; scalar default
            k = cosx.COSX(m, cosx.make_grids(m, **args)).build()
            lo, hi = spinor_jk.fit_quality(k)
            self.assertTrue(spinor_jk.COSX_FIT_RANGE[0] < lo and hi < spinor_jk.COSX_FIT_RANGE[1])

    def test_get_jk_ghf_fast_vs_exact(self):
        # Kramers-unrestricted (complex, not time-even) densities, hermi=1 and 0
        mf = spinor_hf.SCF(mol_hi)
        nao = mol_hi.nao
        rng = numpy.random.default_rng(3)
        x = rng.standard_normal((2 * nao, 2 * nao)) + 1j * rng.standard_normal((2 * nao, 2 * nao))
        dh = (x + x.conj().T) * .05
        for d, h in ((dh, 1), (x * .05, 0)):
            mf.rijcosx = ''
            vj0, vk0 = spinor_jk.get_jk_ghf(mf, mol_hi, d, hermi=h)
            mf.rijcosx = 'rijcosx'
            mf._spinor_cosx = None
            vj1, vk1 = spinor_jk.get_jk_ghf(mf, mol_hi, d, hermi=h)
            self.assertLess(abs(vj1 - vj0).max() / abs(vj0).max(), 1e-3)
            self.assertLess(abs(vk1 - vk0).max() / abs(vk0).max(), 5e-3)
            if h == 1:
                self.assertLess(abs(vk1 - vk1.conj().T).max(), 1e-12)


if __name__ == '__main__':
    unittest.main()
