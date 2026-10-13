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

import os
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


    def test_restart(self):
        # A second call on a slightly different Hamiltonian, started from the
        # wave function of the first, gives what a fresh calculation gives.
        import os
        h1b = self.h1 * 1.02
        solvers = []
        for restart in (False, True):
            solver = shci.SHCI()
            solver.schedule = [[0, 1e-3], [2, 1e-12]]
            solver.davidson_tol = 1e-10
            solver.restart = restart
            solver.kernel(self.h1, self.eri, self.norb, self.nelec,
                          nroots=self.nroots, ecore=self.ecore)
            e, ci = solver.kernel(h1b, self.eri, self.norb, self.nelec,
                                  nroots=self.nroots, ecore=self.ecore)
            solvers.append((solver, e, ci))
        (fresh, e0, ci0), (restarted, e1, ci1) = solvers
        with open(os.path.join(restarted.scratch, 'output.txt')) as f:
            self.assertIn('starting vectors: read from', f.read())
        with open(os.path.join(fresh.scratch, 'output.txt')) as f:
            self.assertNotIn('starting vectors: read from', f.read())
        self.assertAlmostEqual(abs(numpy.asarray(e1) - e0).max(), 0, 8)
        for k in range(self.nroots):
            dm1a, dm2a = fresh.make_rdm12(ci0[k], self.norb, self.nelec)
            dm1b, dm2b = restarted.make_rdm12(ci1[k], self.norb, self.nelec)
            self.assertAlmostEqual(abs(dm1a - dm1b).max(), 0, 6)
            self.assertAlmostEqual(abs(dm2a - dm2b).max(), 0, 6)

    def test_asci_with_pruning(self):
        # Determinants leave and re-enter the space.  Whatever the path, the
        # result must be the eigenstates of the Hamiltonian in the final
        # determinants, and the same for every way of holding the
        # Hamiltonian.
        norb, nelec = self.norb, self.nelec
        select = {'method': 'asci', 'target_dets': 30, 'growth': 1.6,
                  'core_dets': 12, 'prune': True, 'max_iter': 30,
                  'energy_tol': 1e-12, 'davidson_tol_loose': 1e-10}
        results = []
        for extra in ({}, {'incremental': False}, {'hamiltonian': 'direct'},
                      {'hamiltonian': 'connections'},
                      {'select': dict(select, compact_above=0.1)}):
            solver = shci.SHCI()
            solver.davidson_tol = 1e-10
            solver.extra = dict({'select': select}, **extra)
            solver.extra.setdefault('davidson', {'tol': 1e-10})
            e, ci = solver.kernel(self.h1, self.eri, norb, nelec,
                                  nroots=2, ecore=self.ecore)
            occslst, vectors = solver.wavefunction()
            with open(os.path.join(solver.scratch, 'output.txt')) as f:
                output = f.read()
            results.append((e, occslst, solver, ci, output))
        e, occslst, solver, ci, output = results[0]
        self.assertEqual(len(occslst), 30)
        self.assertIn('list compacted', output)
        # Larger than the target: something was taken out along the way.
        removed = int(output.split('list compacted: ')[1].split()[0])
        self.assertGreater(removed, 0)
        self.assertGreater(results[4][4].count('list compacted'), 1)
        self.assertIn('values held for 0 of', results[3][4])

        e_ref, ci_ref = zfci.kernel_dets(self.h1, self.eri, norb, occslst,
                                         ecore=self.ecore, nroots=2)
        self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, 8)
        for k in range(2):
            dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
            dm1_ref, dm2_ref = zfci.make_rdm12_dets(ci_ref[k], norb, occslst)
            self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 6)
            self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 6)
        for other in results[1:]:
            self.assertEqual(sorted(other[1]), sorted(occslst))
            self.assertAlmostEqual(abs(numpy.asarray(other[0]) - e).max(), 0, 8)


    def check_final_space(self, select, nroots=2, smaller_than=None):
        '''Runs a selection that takes determinants out of the space and
        checks that the result is the eigenstates of the Hamiltonian in the
        determinants it ends with.  Returns their number and the output.'''
        norb, nelec = self.norb, self.nelec
        solver = shci.SHCI()
        solver.davidson_tol = 1e-10
        solver.extra = {'select': select, 'davidson': {'tol': 1e-10}}
        e, ci = solver.kernel(self.h1, self.eri, norb, nelec, nroots=nroots,
                              ecore=self.ecore)
        occslst, vectors = solver.wavefunction()
        with open(os.path.join(solver.scratch, 'output.txt')) as f:
            output = f.read()
        e_ref, ci_ref = zfci.kernel_dets(self.h1, self.eri, norb, occslst,
                                         ecore=self.ecore, nroots=nroots)
        self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, 8)
        for k in range(nroots):
            dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
            dm1_ref, dm2_ref = zfci.make_rdm12_dets(ci_ref[k], norb, occslst)
            self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 6)
            self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 6)
        return len(occslst), output

    def test_aci_with_pruning(self):
        base = {'method': 'aci', 'sigma': 0.3, 'max_iter': 30,
                'energy_tol': 1e-12, 'davidson_tol_loose': 1e-10}
        grown, _ = self.check_final_space(base)
        pruned, output = self.check_final_space(dict(base, prune=True))
        self.assertIn('list compacted', output)
        self.assertLess(pruned, grown)

    def test_prune_by_coefficient(self):
        base = {'schedule': [[0, 1e-3]], 'max_iter': 12, 'energy_tol': 1e-12,
                'davidson_tol_loose': 1e-10}
        full, _ = self.check_final_space(base)
        pruned, output = self.check_final_space(
            dict(base, prune_coefficient=0.1))
        self.assertIn('left for their small coefficients', output)
        self.assertLess(pruned, full)


    def test_adaptive_storage(self):
        # "hamiltonian": "adapt" under memory limits from far too small to
        # ample: rows applied directly, rows stored as connections, rows
        # stored with their values, and mixtures of them, always with the
        # energies and density matrices of the stored Hamiltonian.
        norb, nelec = self.norb, self.nelec

        def run(extra):
            solver = shci.SHCI()
            solver.schedule = [[0, 1e-1], [2, 1e-2], [4, 1e-3]]
            solver.davidson_tol = 1e-10
            solver.extra = dict({'davidson': {'tol': 1e-10}}, **extra)
            e, ci = solver.kernel(self.h1, self.eri, norb, nelec, nroots=2,
                                  ecore=self.ecore)
            dms = [solver.make_rdm12(ci[k], norb, nelec) for k in range(2)]
            with open(os.path.join(solver.scratch, 'output.txt')) as f:
                return numpy.asarray(e), dms, f.read()

        e_ref, dms_ref, _ = run({})
        direct_rows, partial_values, all_values = False, False, False
        for kilobytes in (20, 60, 90, 100, 120, 150, 200, 400, 4000):
            e, dms, output = run({'hamiltonian': 'adapt',
                                  'memory': kilobytes / 1048576.0})
            self.assertAlmostEqual(abs(e - e_ref).max(), 0, 8)
            for (dm1, dm2), (dm1_ref, dm2_ref) in zip(dms, dms_ref):
                self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 6)
                self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 6)
            direct_rows = direct_rows or 'are applied directly' in output
            if 'values held for' in output:
                held = int(output.split('values held for ')[1].split()[0])
                partial_values = partial_values or held > 0
            elif 'are applied directly' not in output:
                all_values = True
        self.assertTrue(direct_rows)
        self.assertTrue(all_values)

    def test_relations(self):
        # The three ways of finding related strings ("connect":
        # {"relations": ...}; "bytes" stores no lists of them) with every
        # way of holding the Hamiltonian: the same energies and density
        # matrices.
        norb, nelec = self.norb, self.nelec

        def run(extra):
            solver = shci.SHCI()
            solver.schedule = [[0, 1e-1], [2, 1e-2], [4, 1e-3]]
            solver.davidson_tol = 1e-10
            solver.extra = dict({'davidson': {'tol': 1e-10}}, **extra)
            e, ci = solver.kernel(self.h1, self.eri, norb, nelec, nroots=2,
                                  ecore=self.ecore)
            dms = [solver.make_rdm12(ci[k], norb, nelec) for k in range(2)]
            with open(os.path.join(solver.scratch, 'output.txt')) as f:
                return numpy.asarray(e), dms, f.read()

        e_ref, dms_ref, _ = run({'connect': {'relations': 'subgroups'}})
        for relations in ('keys', 'bytes'):
            for storage in ({}, {'hamiltonian': 'connections'},
                            {'hamiltonian': 'direct'},
                            {'hamiltonian': 'direct', 'incremental': False},
                            {'hamiltonian': 'adapt', 'memory': 1e-4}):
                connect = {'relations': relations}
                if storage.get('incremental') is False:
                    connect['blocks'] = False
                e, dms, output = run(dict(storage, connect=connect))
                self.assertIn('"relations": "%s"' % relations, output)
                self.assertAlmostEqual(abs(e - e_ref).max(), 0, 8)
                for (dm1, dm2), (dm1_ref, dm2_ref) in zip(dms, dms_ref):
                    self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 6)
                    self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 6)

    def test_integral_formats(self):
        # The binary file of antisymmetrized integrals and the text FCIDUMP
        # give the same Hamiltonian, with and without symmetry labels.
        norb, nelec = self.norb, self.nelec
        energies = []
        for fmt in ('hdf5', 'fcidump'):
            solver = shci.SHCI()
            solver.integral_format = fmt
            solver.davidson_tol = 1e-10
            e, _ = solver.kernel(self.h1, self.eri, norb, nelec, nroots=2,
                                 ecore=self.ecore)
            name = 'integrals.h5' if fmt == 'hdf5' else 'FCIDUMP'
            self.assertTrue(os.path.exists(os.path.join(solver.scratch, name)))
            energies.append(numpy.asarray(e))
        self.assertAlmostEqual(abs(energies[0] - energies[1]).max(), 0, 9)


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


class LabelSets(unittest.TestCase):
    '''Two independent conserved labels (an axis modulo 4 and a parity).'''

    def test_two_label_sets(self):
        norb, nelec = 8, 4
        axis = numpy.array([1, -1, 1, -1, 3, -3, 3, -3])
        parity = numpy.array([0, 0, 1, 1, 0, 0, 1, 1])
        h1, eri = random_hamiltonian(norb, seed=3)
        # Keep only what conserves both labels: (pq|rs) moves electrons from
        # q, s to p, r.
        d_axis = axis[:, None] - axis[None, :]
        d_parity = parity[:, None] - parity[None, :]
        h1 = h1 * ((d_axis % 4 == 0) & (d_parity % 2 == 0))
        keep = (((d_axis[:, :, None, None] + d_axis[None, None, :, :]) % 4 == 0)
                & ((d_parity[:, :, None, None] + d_parity[None, None, :, :]) % 2 == 0))
        eri = eri * keep

        def run(orbsym, mod):
            solver = shci.SHCI()
            solver.initial_dets = [[0, 1, 2, 4]]
            solver.orbsym, solver.orbsym_mod = orbsym, mod
            solver.extra = {'davidson': {'tol': 1e-10},
                            'select': {'schedule': [[0, 1e-12]], 'max_iter': 20,
                                       'energy_tol': 1e-12,
                                       'davidson_tol_loose': 1e-10}}
            e, _ = solver.kernel(h1, eri, norb, nelec)
            return e, len(solver.wavefunction()[0])

        e_none, n_none = run(None, 0)
        e_both, n_both = run([axis, parity], [4, 2])
        e_axis, n_axis = run(axis, 4)
        # The Hamiltonian has the symmetry, so the labels change nothing ...
        self.assertAlmostEqual(e_both, e_none, 9)
        self.assertAlmostEqual(e_axis, e_none, 9)
        self.assertEqual(n_both, n_none)
        # ... and the block is a proper part of the full space.
        self.assertLess(n_both, 70)
        e_fci = zfci.FCISolver().kernel(h1, eri, norb, nelec, nroots=20)[0]
        self.assertAlmostEqual(abs(numpy.asarray(e_fci) - e_both).min(), 0, 8)

        # Integrals that break the parity are refused when it is imposed ...
        h1_broken = h1.copy()
        h1_broken[0, 2] += 0.3
        h1_broken[2, 0] += 0.3

        def run_broken(orbsym, mod, tolerance=None):
            solver = shci.SHCI()
            solver.initial_dets = [[0, 1, 2, 4]]
            solver.orbsym, solver.orbsym_mod = orbsym, mod
            solver.schedule = [[0, 1e-12]]
            solver.max_iter = 20
            if tolerance is not None:
                solver.extra = {'symmetry_tolerance': tolerance}
            return solver.kernel(h1_broken, eri, norb, nelec)[0]

        with self.assertRaises(RuntimeError):
            run_broken([axis, parity], [4, 2])
        # ... unless the tolerance allows dropping them, which gives back the
        # symmetric Hamiltonian; without labels they act.
        self.assertAlmostEqual(run_broken([axis, parity], [4, 2], 1.0), e_both, 7)
        self.assertGreater(abs(run_broken(None, 0) - e_both), 1e-4)


if __name__ == '__main__':
    print('Tests for hci.shci against zfci')
    unittest.main()
