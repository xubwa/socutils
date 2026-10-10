'''
The selected-CI driver for spatial orbitals (socutils.hci.shci.SpatialSHCI,
program lib/zshci) against the FCI solvers of PySCF: energies, spin-traced
dm1 and dm2, a transition dm1, point-group labels, and a CASSCF calculation.

Needs the zshci executable (make in socutils/lib/zshci).  Run:
    python hci/test/test_spatial_shci.py
'''

import unittest

import numpy

from pyscf import ao2mo, fci, gto, mcscf, scf

from socutils.hci import shci


def converged_selection(solver):
    '''Settings that select the whole symmetry block.'''
    solver.extra = {'davidson': {'tol': 1e-10},
                    'select': {'schedule': [[0, 1e-12]], 'max_iter': 30,
                               'energy_tol': 1e-12,
                               'davidson_tol_loose': 1e-10}}
    return solver


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59',
                        basis='6-31g', symmetry='c2v', verbose=0)
        cls.mf = scf.RHF(cls.mol).run(conv_tol=1e-12)
        cls.norb, cls.nelec = 8, (4, 4)
        mc = mcscf.CASCI(cls.mf, cls.norb, cls.nelec)
        cls.h1, cls.ecore = mc.get_h1eff()
        cls.eri = mc.get_h2eff()
        cls.orbsym = cls.mf.mo_coeff.orbsym[mc.ncore:mc.ncore + cls.norb]

    def test_full_space(self):
        norb, nelec, nroots = self.norb, self.nelec, 3
        ref = fci.direct_spin1.FCI()
        ref.conv_tol = 1e-12
        e_ref, ci_ref = ref.kernel(self.h1, self.eri, norb, nelec,
                                   nroots=nroots, ecore=self.ecore)
        solver = shci.SpatialSHCI(self.mol)
        solver.occslst = 'all'
        solver.davidson_tol = 1e-10
        e, ci = solver.kernel(self.h1, self.eri, norb, nelec, nroots=nroots,
                              ecore=self.ecore)
        self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, 8)
        for k in range(nroots):
            dm1_ref, dm2_ref = ref.make_rdm12(ci_ref[k], norb, nelec)
            dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
            self.assertEqual(dm2.shape, (norb,) * 4)
            self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 5)
            self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 5)
        t_ref = ref.trans_rdm1(ci_ref[0], ci_ref[1], norb, nelec)
        t = solver.trans_rdm1(ci[0], ci[1], norb, nelec)
        # Each state has an arbitrary sign.
        self.assertAlmostEqual(min(abs(t - t_ref).max(), abs(t + t_ref).max()),
                               0, 5)

    def test_selection_with_point_group(self):
        norb, nelec = self.norb, self.nelec
        e_ref = fci.direct_spin1.FCI().kernel(
            self.h1, self.eri, norb, nelec, ecore=self.ecore)[0]
        solver = converged_selection(shci.SpatialSHCI(self.mol))
        e, _ = solver.kernel(self.h1, self.eri, norb, nelec, ecore=self.ecore)
        ndet = len(solver.wavefunction()[0])
        solver = converged_selection(shci.SpatialSHCI(self.mol))
        solver.orbsym = self.orbsym
        e_sym, _ = solver.kernel(self.h1, self.eri, norb, nelec,
                                 ecore=self.ecore)
        self.assertAlmostEqual(e, e_ref, 8)
        self.assertAlmostEqual(e_sym, e_ref, 8)
        # Both runs end in the totally symmetric block of M_S = 0.
        self.assertEqual(len(solver.wavefunction()[0]), ndet)
        self.assertLess(ndet, 70 * 70)

    def test_open_shell(self):
        norb, nelec = self.norb, (5, 3)
        e_ref = fci.direct_spin1.FCI().kernel(
            self.h1, self.eri, norb, nelec, ecore=self.ecore)[0]
        solver = shci.SpatialSHCI(self.mol)
        solver.occslst = 'all'
        e, ci = solver.kernel(self.h1, self.eri, norb, nelec, ecore=self.ecore)
        self.assertAlmostEqual(e, e_ref, 8)
        solver.spin = 2
        e, ci = solver.kernel(self.h1, self.eri, norb, 8, ecore=self.ecore)
        self.assertAlmostEqual(e, e_ref, 8)

    def test_casscf(self):
        ref = mcscf.CASSCF(self.mf, 6, 6).run(conv_tol=1e-9)
        mc = mcscf.CASSCF(self.mf, 6, 6)
        mc.fcisolver = shci.SpatialSHCI(self.mol)
        mc.fcisolver.occslst = 'all'
        mc.conv_tol = 1e-9
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_tot, ref.e_tot, 7)


if __name__ == '__main__':
    print('Tests for hci.shci.SpatialSHCI against pyscf.fci')
    unittest.main()
