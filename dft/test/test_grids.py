#!/usr/bin/env python
'''
Tests for socutils.dft.grids: the compact DFT grid (M3 / M4 radial mappings,
Krack-Koester radial counts, five-region pruning, weight cutoff) and the
translational sum rule for analytic Hessians.

Run: python dft/test/test_grids.py
'''

import unittest
import numpy
from pyscf import gto, dft
from socutils.dft import grids as sgrids


def setUpModule():
    global mol, mf_ref
    mol = gto.M(atom='''O 0 0 0; H 0.75695 0 0.585882; H -0.75695 0 0.585882''',
                basis='cc-pvdz', verbose=0)
    mf_ref = dft.RKS(mol, xc='PBE')
    mf_ref.grids.atom_grid = (150, 974)
    mf_ref.grids.prune = None
    mf_ref.conv_tol = 1e-11
    mf_ref.kernel()


def tearDownModule():
    global mol, mf_ref
    del mol, mf_ref


class KnownValues(unittest.TestCase):
    def test_radial_mappings(self):
        # int_0^inf 4 pi r^2 exp(-a r^2) dr = (pi/a)^1.5
        for f in (sgrids.treutler_m3, sgrids.treutler_m4):
            for a in (0.3, 1., 50.):
                r, dr = f(80, 8)
                val = numpy.dot(4 * numpy.pi * r**2 * dr, numpy.exp(-a * r**2))
                self.assertAlmostEqual(val / (numpy.pi / a)**1.5, 1, 8)
            self.assertTrue(numpy.all(numpy.diff(r) > 0))

    def test_m4_matches_pyscf_treutler(self):
        from pyscf.dft import radi
        r0, w0 = radi.treutler(40, 8)
        r1, w1 = sgrids.treutler_m4(40, 8)
        self.assertAlmostEqual(abs(r0 - r1).max(), 0, 12)
        self.assertAlmostEqual(abs(w0 - w1).max(), 0, 12)

    def test_radial_count(self):
        self.assertEqual(sgrids.krack_koster_nrad(1, 4.388), 31)
        self.assertEqual(sgrids.krack_koster_nrad(8, 4.388), 36)
        self.assertEqual(sgrids.krack_koster_nrad(17, 4.388), 41)

    def test_prune(self):
        r, dr = sgrids.treutler_m3(36, 8)
        angs = sgrids.region_prune(8, r, 4)
        self.assertEqual(angs.max(), 302)
        self.assertEqual(set(angs), set(sgrids.ANGULAR_TABLES[4]))

    def test_grid_energy(self):
        g = sgrids.Grids(mol).build()
        self.assertTrue(abs(g.weights[g.weights != 0]).min() > g.weight_cutoff)
        npts = (g.weights != 0).sum()
        g2 = dft.Grids(mol)
        g2.level = 2
        g2.build()
        self.assertLess(npts, 0.75 * g2.weights.size)
        mf = dft.RKS(mol, xc='PBE')
        mf.grids = g
        mf.conv_tol = 1e-11
        e = mf.kernel()
        self.assertLess(abs(e - mf_ref.e_tot), 2e-5)

    def test_heavy_radial(self):
        m = gto.M(atom='Cl 0 0 0; H 0 0 1.27', basis='sto-3g', verbose=0)
        g = sgrids.Grids(m)
        tab = g.gen_atomic_grids(m)
        # Cl: Mura-Knowles with b_heavy = 7.5 -> 48 shells; H: M3, 31 shells
        self.assertEqual(g._atom_setting(17, 'Cl')[0], 48)
        self.assertEqual(g._atom_setting(1, 'H')[0], 31)
        r = numpy.linalg.norm(tab['Cl'][0], axis=1)
        self.assertEqual(len(numpy.unique(numpy.round(r, 10))), 48)

    def test_sum_rule(self):
        h = numpy.random.RandomState(1).rand(3, 3, 3, 3)
        h = h + h.transpose(1, 0, 3, 2)
        hs = sgrids.translational_sum_rule(h)
        rs = hs.sum(axis=1)
        self.assertAlmostEqual(abs(rs + rs.transpose(0, 2, 1)).max(), 0, 12)
        self.assertAlmostEqual(abs(hs - hs.transpose(1, 0, 3, 2)).max(), 0, 12)
        self.assertAlmostEqual(abs(sgrids.translational_sum_rule(hs) - hs).max(), 0, 12)
        # off-diagonal blocks untouched
        self.assertAlmostEqual(abs(hs[0, 1] - h[0, 1]).max(), 0, 12)

    def test_hessian_frequencies(self):
        mf = dft.RKS(mol, xc='PBE')
        mf.grids = sgrids.hessian_grids(mol)
        mf.conv_tol = 1e-11
        mf.kernel()
        h = sgrids.translational_sum_rule(mf.Hessian().kernel())
        href = sgrids.translational_sum_rule(mf_ref.Hessian().kernel())
        from pyscf.hessian import thermo
        f = thermo.harmonic_analysis(mol, h)['freq_wavenumber']
        fref = thermo.harmonic_analysis(mol, href)['freq_wavenumber']
        self.assertLess(abs(f - fref).max(), 0.5)


if __name__ == '__main__':
    print('Full Tests for socutils.dft.grids')
    unittest.main()
