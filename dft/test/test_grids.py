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

    def test_alpha_scale(self):
        # element-specific pruning boundaries: K only, nothing else changes; the scaled atom's coarse tables end
        # earlier (more points), exactly as a global scale of alphas for that atom
        m = gto.M(atom='K 0 0 0; F 0 0 2.17', basis='sto-3g', verbose=0)
        g0 = sgrids.Grids(m)
        self.assertEqual(g0.alpha_scale, {})
        t0 = g0.gen_atomic_grids(m)
        g1 = sgrids.Grids(m)
        g1.alpha_scale = {'K': .5}
        t1 = g1.gen_atomic_grids(m)
        self.assertEqual(t1['F'][0].shape, t0['F'][0].shape)
        self.assertAlmostEqual(abs(t1['F'][0] - t0['F'][0]).max(), 0, 12)
        self.assertGreater(t1['K'][0].shape[0], t0['K'][0].shape[0])
        g2 = sgrids.Grids(m)
        g2.alphas = g0.alphas * .5
        t2 = g2.gen_atomic_grids(m)
        self.assertEqual(t1['K'][0].shape, t2['K'][0].shape)
        self.assertAlmostEqual(abs(t1['K'][1] - t2['K'][1]).max(), 0, 12)
        self.assertNotEqual(t2['F'][0].shape, t0['F'][0].shape)
        # the same radial shells: the number of points changes through the angular tables only
        r0 = numpy.unique(numpy.round(numpy.linalg.norm(t0['K'][0], axis=1), 9))
        r1 = numpy.unique(numpy.round(numpy.linalg.norm(t1['K'][0], axis=1), 9))
        self.assertAlmostEqual(abs(r0 - r1).max(), 0, 9)
        # setting it after a build resets the grid; the quadrature still integrates the density
        g0.build()
        n0 = g0.weights.size
        g0.alpha_scale = {'K': .5}
        self.assertIsNone(g0.coords)
        g0.build()
        self.assertGreater(g0.weights.size, n0)
        # atom labels: "K1" is K
        m2 = gto.M(atom='K1 0 0 0; F 0 0 2.17', basis={'K1': 'sto-3g', 'F': 'sto-3g'}, verbose=0)
        g3 = sgrids.Grids(m2)
        g3.alpha_scale = {'K': .5}
        self.assertEqual(g3.gen_atomic_grids(m2)['K1'][0].shape, t1['K'][0].shape)

    def test_polar_pair_adjust(self):
        # the electronegativity rule: C-Br keeps Treutler's adjustment (as for all pairs), Li-F and S-O lose it
        # (as the H-pair rule); the C partition takes the rule's pair table
        from pyscf.dft import radi

        def weights(m, adj):
            g = sgrids.Grids(m)
            return sgrids.get_partition_fast(m, g.gen_atomic_grids(m), adj)[1]

        m = gto.M(atom='C 0 0 0; Br 0 0 1.94; H 1.03 0 -.36; H -.51 .89 -.36; H -.51 -.89 -.36', basis='sto-3g', verbose=0)
        w = weights(m, sgrids.polar_pair_atomic_radii_adjust)
        self.assertAlmostEqual(abs(w - weights(m, radi.treutler_atomic_radii_adjust)).max(), 0, 12)
        self.assertGreater(abs(w - weights(m, sgrids.h_pair_atomic_radii_adjust)).max(), 1e-3)
        for atom in ('Li 0 0 0; F 0 0 1.57', 'S 0 0 0; O 0 1.24 .72; O 0 -1.24 .72'):
            m = gto.M(atom=atom, basis='sto-3g', verbose=0)
            w = weights(m, sgrids.polar_pair_atomic_radii_adjust)
            self.assertAlmostEqual(abs(w - weights(m, sgrids.h_pair_atomic_radii_adjust)).max(), 0, 12)
            self.assertGreater(abs(w - weights(m, radi.treutler_atomic_radii_adjust)).max(), 1e-3)
        f = sgrids.polar_pair_atomic_radii_adjust(m, radi.BRAGG_RADII)
        self.assertEqual(f(0, 1, .3), .3)

    def test_element_intacc(self):
        m = gto.M(atom='Li 0 0 0; F 0 0 1.57', basis='sto-3g', verbose=0)
        g = sgrids.Grids(m)
        # default: Li, Be, Na, Mg on intacc 5.0, where that exceeds the grid's intacc
        self.assertEqual(g.element_intacc, {'Li': 5.0, 'Be': 5.0, 'Na': 5.0, 'Mg': 5.0})
        self.assertEqual(g._atom_setting(3, 'Li')[0], sgrids.krack_koster_nrad(3, 5.0, g.radial_b))
        self.assertEqual(g._atom_setting(9, 'F')[0], sgrids.krack_koster_nrad(9, g.intacc, g.radial_b))
        g.intacc = 6.0
        self.assertEqual(g._atom_setting(3, 'Li')[0], sgrids.krack_koster_nrad(3, 6.0, g.radial_b))
        g.intacc = 4.388
        g.element_intacc = {}
        self.assertEqual(g._atom_setting(3, 'Li')[0], sgrids.krack_koster_nrad(3, g.intacc, g.radial_b))
        g.element_intacc = {'F': 5.5}
        self.assertEqual(g._atom_setting(9, 'F')[0], sgrids.krack_koster_nrad(9, 5.5, g.radial_b))
        # the COSX grids do not take the rule
        from socutils.scf import cosx
        self.assertEqual(cosx.make_grids(m).element_intacc, {})

    def test_mgga_radial_floor(self):
        # the SCAN family and the Minnesota meta-GGAs (also in mixtures) get the radial floor; TPSS / GGAs do not
        for xc in ('r2scan', 'scan', 'rscan', 'MGGA_X_R2SCAN,MGGA_C_R2SCAN', '.25*HF + .75*MGGA_X_R2SCAN, MGGA_C_R2SCAN',
                   'm06l', 'm062x', 'm06', 'mn15', 'mn15l', 'MGGA_X_REVM06_L,MGGA_C_REVM06_L', '.5*MGGA_X_M06_L + .5*GGA_X_PBE, GGA_C_PBE'):
            self.assertTrue(sgrids.needs_fine_radial(xc), xc)
        for xc in ('tpss', 'tpssh', 'revtpss', 'pbe', 'b3lyp', 'lda,vwn', 'pbe0', 'hf'):
            self.assertFalse(sgrids.needs_fine_radial(xc), xc)
        # the unregularised SCAN (also revSCAN, SCAN0, in mixtures) gets the finer floor; r2SCAN / rSCAN do not
        for xc in ('scan', 'revscan', 'scan0', 'MGGA_X_SCAN,MGGA_C_R2SCAN', '.5*MGGA_X_SCAN + .5*MGGA_X_M06_L, GGA_C_PBE'):
            self.assertEqual(sgrids.radial_floor(xc), sgrids.SCAN_RADIAL_INTACC, xc)
        for xc in ('r2scan', 'rscan', 'm06l', 'MGGA_X_R2SCAN,MGGA_C_R2SCAN'):
            self.assertEqual(sgrids.radial_floor(xc), sgrids.MGGA_RADIAL_INTACC, xc)
        gs = sgrids.Grids(mol)
        self.assertTrue(sgrids.adapt_to_xc(gs, 'scan', rebuild=False))
        self.assertEqual(gs.xc_intacc, sgrids.SCAN_RADIAL_INTACC)
        g = sgrids.Grids(mol).build()
        n0 = g.weights.size
        self.assertTrue(sgrids.adapt_to_xc(g, 'r2scan'))
        self.assertEqual(g.xc_intacc, sgrids.MGGA_RADIAL_INTACC)
        self.assertGreater(g.weights.size, n0)                  # rebuilt with more radial shells
        g6 = sgrids.Grids(mol)
        g6.intacc = sgrids.MGGA_RADIAL_INTACC
        self.assertEqual(g.weights.size, g6.build().weights.size)
        self.assertFalse(sgrids.adapt_to_xc(g, 'r2scan'))       # once
        self.assertTrue(sgrids.adapt_to_xc(g, 'pbe'))           # back to the defaults for another functional
        self.assertEqual(g.weights.size, n0)
        ge = sgrids.Grids(mol)
        ge.intacc = 5.0                                         # an explicit setting wins
        self.assertFalse(sgrids.adapt_to_xc(ge, 'r2scan'))
        # the SCF through vxc_fast and the CP-SCF auto grid of the Hessian
        from socutils.dft import vxc_fast
        from socutils.hessian import rhf_fast
        mf = dft.RKS(mol, xc='r2scan')
        mf.grids = sgrids.Grids(mol)
        vxc_fast.apply(mf)
        mf.kernel()
        self.assertEqual(mf.grids.xc_intacc, sgrids.MGGA_RADIAL_INTACC)
        self.assertEqual(mf.grids.weights.size, g6.weights.size)
        h = rhf_fast.RKSHessian(mf)
        h.kernel()
        self.assertEqual(h._cpscf_auto.intacc, sgrids.MGGA_RADIAL_INTACC)

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
