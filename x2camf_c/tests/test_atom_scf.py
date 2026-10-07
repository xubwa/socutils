'''
Tests of the atomic-SCF entry point x2camf_atom_scf and of the numerically
integrated nuclear models (no dependency beyond numpy).

    python x2camf_c/tests/test_atom_scf.py

Reference energy of Ne in the (10s5p) set below, Dirac-Coulomb, point nucleus:
C backend -128.6682062476, C++ X2CAMF -128.6682062477, PySCF DHF -128.6682062488.
See tools/basisopt/README.md for the validation against numerical Dirac-Fock.
'''
import os
import sys
import unittest

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from x2camf import _c_backend as cb  # noqa: E402

# a (10s5p) primitive set for Ne
NE_S = [24350.0, 3650.0, 829.6, 234.0, 75.61, 26.73, 9.927, 3.905, 1.102, 0.4764]
NE_P = [54.7, 12.43, 3.679, 1.143, 0.33]
C = 137.03599967994


def ne(**kw):
    shell = numpy.array([0] * len(NE_S) + [1] * len(NE_P))
    exps = numpy.array(NE_S + NE_P)
    kw.setdefault('speed_of_light', C)
    return cb.atom_scf(10, 2, len(shell), 0, shell, exps, **kw)


class KnownValues(unittest.TestCase):
    def test_dhf_energy(self):
        r = ne()
        self.assertTrue(r['converged'])
        self.assertAlmostEqual(r['e_scf'], -128.6682062476, 8)
        s, p1, p3 = r['blocks']
        self.assertEqual([(b['l'], b['two_j']) for b in r['blocks']], [(0, 1), (1, 1), (1, 3)])
        self.assertEqual(s['c'].shape, (2 * len(NE_S), len(NE_S)))
        # spin-orbit splitting of 2p and the electron count
        self.assertGreater(p3['e'][0] - p1['e'][0], 0.004)
        nel = sum((b['two_j'] + 1) * b['occ'].sum() for b in r['blocks'])
        self.assertAlmostEqual(nel, 10.0, 12)
        # the spinor energies are the eigenvalues of the returned Fock matrix
        for b in r['blocks']:
            n = len(b['exp'])
            resid = b['fock'] @ b['c'] - b['ovlp'] @ b['c'] * b['e']
            self.assertLess(abs(resid[:, :n]).max(), 1e-7)

    def test_energy_from_density(self):
        r = ne()
        e = sum(0.5 * (b['two_j'] + 1) * numpy.sum(b['den'] * (b['h1e'] + b['fock']).T)
                for b in r['blocks'])
        self.assertAlmostEqual(e, r['e_scf'], 9)

    def test_gaussian_nucleus_by_quadrature(self):
        # the built-in (analytic) Gaussian nucleus of Ne: A = 20
        rnuc = (0.836 * 20 ** (1 / 3) + 0.570) / 52917.7249
        xi = 1.5 / rnuc ** 2
        a = ne(gau_nuc=True)
        b = ne(nuc_model=2, nuc_p1=xi)
        self.assertAlmostEqual(a['e_scf'], b['e_scf'], 9)
        self.assertLess(abs(a['blocks'][0]['e'] - b['blocks'][0]['e']).max(), 1e-9)
        # a finite nucleus raises the energy; Fermi and sphere of the same rms
        # radius differ from the Gaussian one by much less
        pt = ne()
        self.assertGreater(a['e_scf'] - pt['e_scf'], 1e-5)
        sphere = ne(nuc_model=3, nuc_p1=(5 / 3) ** 0.5 * rnuc)
        self.assertLess(abs(sphere['e_scf'] - a['e_scf']), 1e-6)

    def test_x2c_and_two_step(self):
        sf = ne(scf_type=1, spin_free=True)
        so = ne(scf_type=2)
        self.assertEqual(sf['blocks'][0]['c'].shape, (len(NE_S), len(NE_S)))
        # spin-free: p1/2 and p3/2 are degenerate; with the X2CAMF integrals they split
        self.assertAlmostEqual(sf['blocks'][1]['e'][0], sf['blocks'][2]['e'][0], 9)
        self.assertGreater(so['blocks'][2]['e'][0] - so['blocks'][1]['e'][0], 0.003)

    def test_unconverged_is_nan(self):
        r = ne(max_iter=1)
        self.assertFalse(r['converged'])
        self.assertTrue(numpy.isnan(r['e_scf']))

    def test_unsorted_shells_rejected(self):
        shell = numpy.array([0, 1, 0])
        with self.assertRaises(ValueError):
            cb.atom_scf(10, 2, 3, 0, shell, numpy.array([10.0, 1.0, 1.0]))


if __name__ == '__main__':
    unittest.main()
