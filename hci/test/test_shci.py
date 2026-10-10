'''
The selected-CI driver (socutils.hci.shci, program lib/zshci) against the exact
spinor full CI of socutils (socutils.fci.zfci) on a
small random complex Hamiltonian: energies, dm1, dm2 and transition dm1.

The Hamiltonian has no symmetry, so the states are non-degenerate and the
density matrices are genuinely complex: a density matrix that came back
conjugated or transposed would fail here, which the energy alone cannot tell.

Needs the zshci executable (make in socutils/lib/zshci).  Run:
    python hci/test/test_shci.py
'''

import unittest

import numpy

from socutils.fci import zfci
from socutils.hci import shci


def random_hamiltonian(norb, seed=7):
    '''h1[p,q] and eri[p,q,r,s] = (pq|rs) with the symmetries of a Hermitian
    two-body operator: (pq|rs) = (rs|pq) = conj((qp|sr)).'''
    rng = numpy.random.default_rng(seed)

    def hermitian():
        a = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        return a + a.conj().T

    h1 = hermitian()
    factors = numpy.array([hermitian() for _ in range(2 * norb)])
    eri = 0.1 * numpy.einsum('lpq,lrs->pqrs', factors, factors)
    return h1, eri


def align_phase(a, reference):
    '''a times the phase that makes it closest to reference.'''
    overlap = numpy.vdot(a, reference)
    return a * overlap / abs(overlap)


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.norb, cls.nelec, cls.nroots = 8, 4, 3
        cls.ecore = 0.37
        cls.h1, cls.eri = random_hamiltonian(cls.norb)
        ref = zfci.FCISolver()
        cls.e_ref, cls.ci_ref = ref.kernel(cls.h1, cls.eri, cls.norb, cls.nelec,
                                           nroots=cls.nroots, ecore=cls.ecore)
        cls.ref = ref

    def check(self, solver):
        norb, nelec = self.norb, self.nelec
        e, ci = solver.kernel(self.h1, self.eri, norb, nelec,
                              nroots=self.nroots, ecore=self.ecore)
        self.assertAlmostEqual(abs(numpy.asarray(e) - self.e_ref).max(), 0, 8)
        for k in range(self.nroots):
            dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
            dm1_ref, dm2_ref = self.ref.make_rdm12(self.ci_ref[k], norb, nelec)
            self.assertGreater(abs(dm1_ref.imag).max(), 1e-3)
            self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 6)
            self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 6)
            self.assertAlmostEqual(
                abs(solver.make_rdm1(ci[k], norb, nelec) - dm1_ref).max(), 0, 6)
        for k in range(1, self.nroots):
            # Each state has an arbitrary phase, so a transition density is
            # defined up to one overall phase.
            t_ref = self.ref.trans_rdm1(self.ci_ref[0], self.ci_ref[k], norb, nelec)
            t = solver.trans_rdm1(ci[0], ci[k], norb, nelec)
            self.assertAlmostEqual(abs(align_phase(t, t_ref) - t_ref).max(), 0, 6)
            t_back = solver.trans_rdm1(ci[k], ci[0], norb, nelec)
            self.assertAlmostEqual(abs(t_back - t.conj().T).max(), 0, 12)

    def test_full_space(self):
        solver = shci.SHCI()
        solver.occslst = 'all'
        solver.davidson_tol = 1e-10
        self.check(solver)

    def test_selection_to_convergence(self):
        # A cutoff below every matrix element selects the whole space.
        solver = shci.SHCI()
        solver.schedule = [[0, 1e-12]]
        solver.max_iter = 20
        solver.davidson_tol = 1e-10
        solver.extra = {'davidson': {'tol': 1e-10},
                        'select': {'schedule': [[0, 1e-12]], 'max_iter': 20,
                                   'energy_tol': 1e-12,
                                   'davidson_tol_loose': 1e-10}}
        self.check(solver)


class RealNumbers(unittest.TestCase):
    '''A real Hamiltonian in real arithmetic against the same Hamiltonian in
    complex arithmetic and against zfci.'''

    def test_real_against_complex(self):
        norb, nelec, nroots = 8, 4, 2
        h1, eri = random_hamiltonian(norb, seed=11)
        h1, eri = h1.real.copy(), eri.real.copy()
        ref = zfci.FCISolver()
        e_ref, ci_ref = ref.kernel(h1.astype(complex), eri.astype(complex),
                                   norb, nelec, nroots=nroots)

        results = {}
        for scalar in ('auto', 'complex'):
            solver = shci.SHCI()
            solver.occslst = 'all'
            solver.davidson_tol = 1e-10
            solver.scalar = scalar
            e, ci = solver.kernel(h1, eri, norb, nelec, nroots=nroots)
            results[scalar] = (e, [solver.make_rdm12(c, norb, nelec) for c in ci],
                               solver.wavefunction()[1])
        e_real, dms_real, ci_real = results['auto']
        e_complex, dms_complex, ci_complex = results['complex']
        self.assertTrue(numpy.isrealobj(ci_real))
        self.assertTrue(numpy.iscomplexobj(ci_complex))
        self.assertAlmostEqual(abs(numpy.asarray(e_real) - e_ref).max(), 0, 8)
        self.assertAlmostEqual(abs(numpy.asarray(e_real) - e_complex).max(), 0, 10)
        for k in range(nroots):
            dm1_ref, dm2_ref = ref.make_rdm12(ci_ref[k], norb, nelec)
            self.assertTrue(numpy.isrealobj(dms_real[k][0]))
            self.assertTrue(numpy.isrealobj(dms_real[k][1]))
            self.assertAlmostEqual(abs(dms_real[k][0] - dm1_ref).max(), 0, 6)
            self.assertAlmostEqual(abs(dms_real[k][1] - dm2_ref).max(), 0, 6)
            self.assertAlmostEqual(abs(dms_real[k][0] - dms_complex[k][0]).max(), 0, 8)
            self.assertAlmostEqual(abs(dms_real[k][1] - dms_complex[k][1]).max(), 0, 8)

    def test_real_refuses_complex_integrals(self):
        norb, nelec = 8, 4
        h1, eri = random_hamiltonian(norb, seed=11)
        solver = shci.SHCI()
        solver.occslst = 'all'
        solver.scalar = 'real'
        with self.assertRaises(RuntimeError):
            solver.kernel(h1, eri, norb, nelec)


if __name__ == '__main__':
    print('Tests for hci.shci against zfci')
    unittest.main()
