'''
The GAS-CI kernels and driver (socutils.hci.gasci, lib/libzgasci.so).

- Exhaustive: H built by acting with the second-quantized operators on the
  bit strings of the determinants that zg_dets lists, against the diagonal
  and sigma of the kernels, for spaces of up to 8 spinors: real and complex,
  full CI, GAS constraints, symmetry labels with and without a modulus, and
  several divisions into sub-groups.
- Spinors: energies, dm1, dm2 and transition dm1 against zfci (full CI) and
  zfci.kernel_dets / SHCI with occslst (constrained spaces); the same problem
  under different divisions into sub-groups; CASSCF against SHCI as solver.
- Spatial orbitals: SpatialGASCI against pyscf.fci.direct_spin1, against
  SpatialSHCI in constrained spaces, with D2h labels, open shell, CASSCF.

Run:
    python hci/test/test_gasci.py
'''

import itertools
import os
import unittest

import numpy

from socutils.hci import gasci


def random_hamiltonian(norb, seed=7):
    '''h1[p,q] and eri[p,q,r,s] = (pq|rs) with the symmetries of a Hermitian
    two-body operator (as in test_shci.py).'''
    rng = numpy.random.default_rng(seed)

    def hermitian():
        a = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        return a + a.conj().T

    h1 = hermitian()
    factors = numpy.array([hermitian() for _ in range(2 * norb)])
    eri = 0.1 * numpy.einsum('lpq,lrs->pqrs', factors, factors)
    return h1, eri


def new_solver(cls=None, **settings):
    '''A solver converged tightly enough for 1e-10 comparisons.'''
    mol = settings.pop('mol', None)
    solver = (cls or gasci.GASCI)(mol)
    solver.conv_tol = 1e-12
    solver.conv_tol_residual = 1e-7
    solver.max_cycle = 400
    solver.max_space = 30
    solver.sigma_kernel = os.environ.get('ZGASCI_TEST_KERNEL', solver.sigma_kernel)
    for key, value in settings.items():
        setattr(solver, key, value)
    return solver


def align_phase(a, reference):
    overlap = numpy.vdot(a, reference)
    return a * overlap / abs(overlap)


# -- the exhaustive reference -------------------------------------------------

def _apply(ops, det):
    '''ops = [(creation?, p), ...] as written, applied right to left on the
    bit string det; returns (sign, det) or None.'''
    sign = 1
    for create, p in reversed(ops):
        occupied = det >> p & 1
        if create == bool(occupied):
            return None
        if bin(det & ((1 << p) - 1)).count('1') % 2:
            sign = -sign
        det ^= 1 << p
    return sign, det


def exact_hamiltonian(h1, eri, dets):
    '''<I|H|J> over the bit strings dets, H = sum h_pq p+ q
    + 1/2 sum (pq|rs) p+ r+ s q.'''
    n = h1.shape[0]
    index = {d: i for i, d in enumerate(dets)}
    hmat = numpy.zeros((len(dets), len(dets)), dtype=numpy.result_type(h1, eri))
    for j, det in enumerate(dets):
        for p, q in itertools.product(range(n), repeat=2):
            if h1[p, q] == 0:
                continue
            out = _apply([(True, p), (False, q)], det)
            if out and out[1] in index:
                hmat[index[out[1]], j] += out[0] * h1[p, q]
        for p, q, r, s in itertools.product(range(n), repeat=4):
            if eri[p, q, r, s] == 0:
                continue
            out = _apply([(True, p), (True, r), (False, s), (False, q)], det)
            if out and out[1] in index:
                hmat[index[out[1]], j] += 0.5 * out[0] * eri[p, q, r, s]
    return hmat


def bit_strings(plan):
    words = plan.dets()
    return [int(w[0]) for w in words]


def all_bit_strings(n, nelec, keep=lambda occ: True):
    return set(sum(1 << p for p in occ)
               for occ in itertools.combinations(range(n), nelec) if keep(occ))


class Exhaustive(unittest.TestCase):
    '''Kernels against H built from the operators, in the spinor order of
    the plan (no reordering).'''

    def check(self, n, nelec, sub_sizes, nsub_a, constraints=(), labels=None,
              mods=(), targets=(), real=False, seed=5):
        h1, eri = random_hamiltonian(n, seed=seed)
        if real:
            h1, eri = h1.real.copy(), eri.real.copy()
        plan = gasci.Plan(n, nelec, sub_sizes, nsub_a, constraints, labels,
                          mods, targets)
        dets = bit_strings(plan)
        self.assertEqual(len(set(dets)), plan.dim)

        # The space: every determinant that satisfies the constraints and
        # the labels, and nothing else.
        bounds = numpy.cumsum([0] + list(sub_sizes))

        def keep(occ):
            for item in constraints:
                subs, lo, hi = item[:3]
                weights = item[3] if len(item) > 3 else [1] * len(subs)
                k = sum(w for p in occ for g, w in zip(subs, weights)
                        if bounds[g] <= p < bounds[g + 1])
                if not lo <= k <= hi:
                    return False
            if labels is not None:
                for s, row in enumerate(numpy.reshape(labels, (-1, n))):
                    total = sum(row[p] for p in occ) - targets[s]
                    if (total % mods[s] if mods[s] else total) != 0:
                        return False
            return True
        self.assertEqual(set(dets), all_bit_strings(n, nelec, keep))

        hmat = exact_hamiltonian(h1, eri, dets)
        v = gasci.antisymmetrized(h1, eri, n, nelec)
        self.assertAlmostEqual(abs(plan.hdiag(v) - hmat.diagonal().real).max(), 0, 11)
        rng = numpy.random.default_rng(1)
        c = rng.standard_normal(plan.dim)
        if not real:
            c = c + 1j * rng.standard_normal(plan.dim)
        self.assertTrue(plan.block_ok)
        for kernel in ('reference', 'block', 'blas'):
            sigma = plan.sigma(v, c, kernel=kernel)
            self.assertEqual(numpy.iscomplexobj(sigma), not real)
            self.assertAlmostEqual(abs(sigma - hmat @ c).max(), 0, 10, msg=kernel)

        # g[P,R] = <b| A+_P A_R |c>: sum_{P,R} v[P,R] g[P,R] = <b|H|c>.
        b = rng.standard_normal(plan.dim) + (0 if real else 1j * rng.standard_normal(plan.dim))
        g = plan.rdm2_pairs(b, c)
        self.assertAlmostEqual(abs(numpy.sum(v * g) - b.conj() @ hmat @ c), 0, 9)
        return plan

    def test_full_ci_real_and_complex(self):
        for real in (False, True):
            for sub_sizes, nsub_a in (([4, 4], 1), ([2, 2, 2, 2], 2),
                                      ([1, 3, 2, 2], 2), ([3, 2, 1, 2], 3),
                                      ([1, 7], 1), ([5, 1, 2], 1)):
                with self.subTest(real=real, sub_sizes=sub_sizes):
                    plan = self.check(8, 4, sub_sizes, nsub_a, real=real)
                    self.assertEqual(plan.dim, 70)

    def test_electron_counts(self):
        for nelec in (2, 3, 5, 7, 8):
            with self.subTest(nelec=nelec):
                self.check(8, nelec, [2, 3, 1, 2], 2)
        self.check(7, 3, [3, 1, 3], 2, real=True)
        self.check(4, 2, [1, 1, 1, 1], 2)

    def test_constraints(self):
        # A RAS-like space and a GAS of three spaces, each half on A and
        # half on B, plus constraints on other unions of sub-groups.
        self.check(8, 4, [2, 2, 2, 2], 2, constraints=[([0, 2], 1, 3), ([1, 3], 1, 2)])
        self.check(8, 4, [1, 2, 1, 1, 2, 1], 3,
                   constraints=[([0, 3], 2, 2), ([1, 4], 1, 3), ([2, 5], 0, 1)])
        self.check(8, 4, [2, 2, 2, 2], 2, constraints=[([0, 2], 1, 3), ([0, 1, 2], 2, 3)],
                   real=True)
        # Weighted: an excitation level, N(1) + 2 N(2, 3) <= 3.
        self.check(8, 4, [2, 2, 2, 2], 2, constraints=[([1, 2, 3], 0, 3, [1, 2, 2])])
        self.check(8, 4, [3, 1, 2, 2], 2, constraints=[([1, 2, 3], 1, 4, [1, 2, 2]),
                                                       ([0], 1, 3)], real=True)

    def test_labels(self):
        jz = [1, -1, 3, -3, 1, -1, 5, -5]
        parity = [0, 1, 1, 0, 0, 1, 1, 0]
        for real in (False, True):
            self.check(8, 4, [2, 2, 2, 2], 2, labels=[jz], mods=[0], targets=[0], real=real)
            self.check(8, 4, [3, 1, 2, 2], 2, labels=[jz], mods=[0], targets=[2], real=real)
            self.check(8, 4, [4, 4], 1, labels=[jz, parity], mods=[4, 2],
                       targets=[2, 1], real=real)
            self.check(8, 3, [2, 2, 2, 2], 3, labels=[jz, parity], mods=[0, 2],
                       targets=[1, 0], constraints=[([0, 3], 1, 2)], real=real)


class BlasKernel(unittest.TestCase):
    '''sigma of the BLAS kernel against the reference kernel, element by
    element, in spaces large enough for several rows and runs of columns;
    with the matrix products forced, the dot products forced, and the default
    choice between them.'''

    settings = ({},
                {'ZGASCI_BLAS_DIRECT': '0'},
                {'ZGASCI_BLAS_DIRECT': '1e300', 'ZGASCI_BLAS_SMALL': '0',
                 'ZGASCI_BLAS_COLS': '5'})

    def compare(self, n, nelec, sub_sizes, nsub_a, constraints=(), labels=None,
                mods=(), targets=(), real=False):
        h1, eri = random_hamiltonian(n, seed=11)
        if real:
            h1, eri = h1.real.copy(), eri.real.copy()
        plan = gasci.Plan(n, nelec, sub_sizes, nsub_a, constraints, labels,
                          mods, targets)
        self.assertTrue(plan.block_ok)
        self.assertGreater(plan.dim, 0)
        v = gasci.antisymmetrized(h1, eri, n, nelec)
        rng = numpy.random.default_rng(3)
        c = rng.standard_normal(plan.dim)
        if not real:
            c = c + 1j * rng.standard_normal(plan.dim)
        reference = plan.sigma(v, c, kernel='reference')
        for setting in self.settings:
            saved = {key: os.environ.get(key) for key in setting}
            os.environ.update(setting)
            try:
                sigma = plan.sigma(v, c, kernel='blas')
            finally:
                for key, value in saved.items():
                    if value is None:
                        del os.environ[key]
                    else:
                        os.environ[key] = value
            self.assertEqual(sigma.dtype, reference.dtype)
            self.assertLess(abs(sigma - reference).max(), 1e-10, msg=str(setting))
        return plan

    def test_full_ci(self):
        for real in (False, True):
            for n, nelec, sub_sizes, nsub_a in (
                    (12, 6, [6, 6], 1), (12, 5, [3, 3, 3, 3], 2),
                    (11, 4, [4, 2, 5], 1), (12, 7, [2, 2, 2, 2, 2, 2], 3),
                    (12, 6, [1, 4, 2, 5], 3), (10, 2, [5, 5], 1),
                    (10, 3, [2, 3, 5], 2), (10, 9, [5, 2, 3], 1),
                    (9, 9, [4, 5], 1)):
                with self.subTest(real=real, n=n, nelec=nelec, sub_sizes=sub_sizes):
                    self.compare(n, nelec, sub_sizes, nsub_a, real=real)

    def test_constraints(self):
        for real in (False, True):
            # GAS spaces half on each side, unions across the sides, a
            # constraint on one side only, weights (an excitation level).
            self.compare(12, 6, [3, 3, 3, 3], 2, real=real,
                         constraints=[([0, 2], 2, 4), ([1, 3], 2, 4)])
            self.compare(12, 5, [2, 2, 2, 2, 2, 2], 3, real=real,
                         constraints=[([0, 3], 2, 4), ([1, 4], 0, 2), ([2, 5], 0, 1)])
            self.compare(12, 6, [3, 3, 3, 3], 2, real=real,
                         constraints=[([0, 1, 2], 3, 5), ([3], 1, 2)])
            self.compare(12, 7, [2, 4, 3, 3], 2, real=real,
                         constraints=[([0, 1], 3, 3)])
            self.compare(12, 6, [2, 2, 2, 2, 2, 2], 3, real=real,
                         constraints=[([1, 2, 4, 5], 0, 4, [1, 2, 1, 2]),
                                      ([0, 3], 2, 4)])

    def test_labels(self):
        jz = [1, -1, 3, -3, 1, -1, 5, -5, 3, -3, 1, -1]
        parity = [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
        for real in (False, True):
            self.compare(12, 6, [6, 6], 1, labels=[jz], mods=[0], targets=[0], real=real)
            self.compare(12, 5, [3, 3, 3, 3], 2, labels=[jz], mods=[0], targets=[1],
                         real=real)
            self.compare(12, 6, [4, 2, 2, 4], 2, labels=[jz, parity], mods=[4, 2],
                         targets=[2, 1], real=real)
            self.compare(12, 7, [2, 2, 2, 2, 2, 2], 3, labels=[jz, parity],
                         mods=[0, 2], targets=[-1, 0], real=real)
            self.compare(12, 6, [3, 3, 3, 3], 2, labels=[jz, parity], mods=[0, 2],
                         targets=[2, 0], constraints=[([0, 2], 2, 5), ([1, 2, 3], 2, 5)],
                         real=real)

    def test_symmetric_integrals(self):
        # Integrals that conserve the labels (as real ones do), over the
        # driver with its reordering of the orbitals.
        n, nelec = 12, 6
        jz = numpy.array([1, -1, 3, -3, 1, -1, 5, -5, 3, -3, 1, -1])
        h1, eri = random_hamiltonian(n, seed=4)
        h1 = h1 * (jz[:, None] == jz[None, :])
        change = jz[:, None] - jz[None, :]
        eri = eri * (change[:, :, None, None] + change[None, None, :, :] == 0)
        for gas in (None, [dict(orbitals=range(4), min=2, max=4)]):
            solver = new_solver(orbsym=jz, target=[0], gas=gas, max_subgroup=3)
            plan = solver.plan(n, nelec)
            v = solver._hamiltonian(h1, eri, n, nelec, plan)
            rng = numpy.random.default_rng(2)
            c = rng.standard_normal(plan.dim) + 1j * rng.standard_normal(plan.dim)
            self.assertLess(abs(plan.sigma(v, c, 'blas') - plan.sigma(v, c, 'reference')).max(),
                            1e-10)


# -- spinors -------------------------------------------------------------------

def filtered_occslst(norb, nelec, constraints=(), labels=None, mods=(), targets=()):
    '''Every determinant (tuple of original orbitals) of the space.'''
    out = []
    for occ in itertools.combinations(range(norb), nelec):
        good = all(c['min'] <= sum(1 for p in occ if p in c['orbitals']) <= c['max']
                   for c in constraints)
        if labels is not None:
            for row, mod, target in zip(labels, mods, targets):
                total = sum(row[p] for p in occ) - target
                good = good and (total % mod if mod else total) == 0
        if good:
            out.append(occ)
    return out


class Spinors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from socutils.fci import zfci
        cls.norb, cls.nelec, cls.nroots = 8, 4, 3
        cls.ecore = 0.37
        cls.h1, cls.eri = random_hamiltonian(cls.norb)
        ref = zfci.FCISolver()
        cls.e_ref, cls.ci_ref = ref.kernel(cls.h1, cls.eri, cls.norb, cls.nelec,
                                           nroots=cls.nroots, ecore=cls.ecore)
        cls.ref = ref

    def compare(self, solver, e_ref, ci_ref, ref_rdm12, ref_trans, h1=None, eri=None,
                nelec=None, nroots=None, places=9):
        norb = self.norb
        nelec = self.nelec if nelec is None else nelec
        nroots = self.nroots if nroots is None else nroots
        h1 = self.h1 if h1 is None else h1
        eri = self.eri if eri is None else eri
        e, ci = solver.kernel(h1, eri, norb, nelec, nroots=nroots, ecore=self.ecore)
        if nroots == 1:
            e, ci, ci_ref = [e], [ci], [ci_ref]
        self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, places)
        out = []
        for k in range(nroots):
            dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
            dm1_ref, dm2_ref = ref_rdm12(ci_ref[k])
            self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 7)
            self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 7)
            self.assertAlmostEqual(abs(solver.make_rdm1(ci[k], norb, nelec) - dm1_ref).max(), 0, 7)
            out.append((dm1, dm2))
        for k in range(1, nroots):
            t_ref = ref_trans(ci_ref[0], ci_ref[k])
            t = solver.trans_rdm1(ci[0], ci[k], norb, nelec)
            self.assertAlmostEqual(abs(align_phase(t, t_ref) - t_ref).max(), 0, 7)
            t_back = solver.trans_rdm1(ci[k], ci[0], norb, nelec)
            self.assertAlmostEqual(abs(t_back - t.conj().T).max(), 0, 12)
        return e, out

    def full_ci_check(self, solver):
        norb, nelec = self.norb, self.nelec
        return self.compare(
            solver, self.e_ref, self.ci_ref,
            lambda c: self.ref.make_rdm12(c, norb, nelec),
            lambda b, k: self.ref.trans_rdm1(b, k, norb, nelec))

    def test_full_ci_divisions(self):
        # The same problem under several divisions into sub-groups: with the
        # Davidson iterations against zfci, and with the full matrix of the
        # kernels against each other to 1e-10.
        divisions = (dict(), dict(max_subgroup=1), dict(max_subgroup=3),
                     dict(subgroups=([[0, 3, 5]], [[1, 2, 4, 6, 7]])),
                     dict(subgroups=([[7], [0, 1], [5]], [[2, 3], [4], [6]])))
        results = []
        for settings in divisions:
            with self.subTest(settings=settings):
                self.full_ci_check(new_solver(direct_dim=0, **settings))
                results.append(self.full_ci_check(new_solver(direct_dim=1000, **settings)))
        e0, dms0 = results[0]
        for e, dms in results[1:]:
            self.assertAlmostEqual(abs(numpy.asarray(e) - e0).max(), 0, 10)
            for (a1, a2), (b1, b2) in zip(dms, dms0):
                self.assertAlmostEqual(abs(a1 - b1).max(), 0, 10)
                self.assertAlmostEqual(abs(a2 - b2).max(), 0, 10)

    def test_wavefunction(self):
        from socutils.fci import zfci
        solver = new_solver()
        solver.subgroups = ([[6, 1], [3]], [[0, 2, 4], [5, 7]])
        e, ci = solver.kernel(self.h1, self.eri, self.norb, self.nelec, nroots=2)
        occslst, coef = solver.wavefunction(ci[0], self.norb, self.nelec)
        e_ref, ci_ref = zfci.kernel_dets(self.h1, self.eri, self.norb, occslst, nroots=2)
        self.assertAlmostEqual(e[0], e_ref[0], 10)
        self.assertAlmostEqual(abs(align_phase(coef, ci_ref[0]) - ci_ref[0]).max(), 0, 8)

    def test_real(self):
        from socutils.fci import zfci
        norb, nelec, nroots = 8, 4, 2
        h1, eri = random_hamiltonian(norb, seed=11)
        h1, eri = h1.real.copy(), eri.real.copy()
        ref = zfci.FCISolver()
        e_ref, ci_ref = ref.kernel(h1.astype(complex), eri.astype(complex), norb,
                                   nelec, nroots=nroots, ecore=self.ecore)
        solver = new_solver()
        solver.direct_dim = 0
        solver.max_subgroup = 2
        e, dms = self.compare(solver, e_ref, ci_ref,
                              lambda c: ref.make_rdm12(c, norb, nelec),
                              lambda b, k: ref.trans_rdm1(b, k, norb, nelec),
                              h1=h1, eri=eri, nroots=nroots)
        self.assertTrue(numpy.isrealobj(solver.ci[0]))
        self.assertTrue(numpy.isrealobj(dms[0][1]))

    def constrained_check(self, solver, constraints, labels=None, mods=(), targets=(),
                          nelec=None, nroots=2):
        from socutils.fci import zfci
        norb = self.norb
        nelec = self.nelec if nelec is None else nelec
        occslst = filtered_occslst(norb, nelec, constraints, labels, mods, targets)
        self.assertLess(len(occslst), 70)
        e_ref, ci_ref = zfci.kernel_dets(self.h1, self.eri, norb, occslst,
                                         ecore=self.ecore, nroots=nroots)
        self.assertEqual(solver.plan(norb, nelec).dim, len(occslst))
        return self.compare(solver, e_ref, ci_ref,
                            lambda c: zfci.make_rdm12_dets(c, norb, occslst),
                            lambda b, k: zfci.trans_rdm1_dets(b, k, norb, occslst),
                            nelec=nelec, nroots=nroots)

    def test_gas(self):
        gas = [dict(orbitals=[0, 1, 2], min=2, max=3),
               dict(orbitals=[3, 4, 5], min=1, max=2),
               dict(orbitals=[6, 7], min=0, max=1)]
        results = []
        for settings in (dict(), dict(max_subgroup=1),
                         dict(subgroups=([[0, 1, 2], [3], [6]], [[4, 5], [7]]))):
            solver = new_solver()
            solver.direct_dim = 0
            solver.gas = gas
            for key, value in settings.items():
                setattr(solver, key, value)
            results.append(self.constrained_check(solver, gas))
        for e, dms in results[1:]:
            self.assertAlmostEqual(abs(numpy.asarray(e) - results[0][0]).max(), 0, 10)
        # A cut through a space is refused.
        solver = new_solver()
        solver.gas = gas
        solver.subgroups = ([[0, 1], [3]], [[2, 4, 5], [6, 7]])
        with self.assertRaises(ValueError):
            solver.plan(self.norb, self.nelec)

    def test_excitation_level(self):
        # Excitations from a reference with 0..1 filled, 2..3 its active
        # space and 4..7 empty, of at most 2: holes(0,1) + N(4..7) <= 2,
        # i.e. N(2,3) + 2 N(4..7) <= 2 + 2 (4 electrons).
        level = [dict(orbitals=list(range(2, 8)), weights=[1, 1, 2, 2, 2, 2],
                      min=0, max=4)]
        def keep(occ):
            holes = sum(1 for p in (0, 1) if p not in occ)
            return holes + sum(1 for p in occ if p >= 4) <= 2
        from socutils.fci import zfci
        occslst = [occ for occ in itertools.combinations(range(8), 4) if keep(occ)]
        e_ref, ci_ref = zfci.kernel_dets(self.h1, self.eri, 8, occslst, ecore=self.ecore,
                                         nroots=2)
        for settings in (dict(), dict(max_subgroup=1)):
            solver = new_solver(restrictions=level, direct_dim=0, **settings)
            self.assertEqual(solver.plan(8, 4).dim, len(occslst))
            self.compare(solver, e_ref, ci_ref,
                         lambda c: zfci.make_rdm12_dets(c, 8, occslst),
                         lambda b, k: zfci.trans_rdm1_dets(b, k, 8, occslst), nroots=2)

    def test_ras(self):
        # RAS1 = 0..2 (at most one hole), RAS3 = 6..7 (at most one particle),
        # given as restrictions in the format of SHCI.
        ras = [dict(orbitals=[0, 1, 2], min=2, max=3),
               dict(orbitals=[6, 7], min=0, max=1)]
        solver = new_solver()
        solver.restrictions = ras
        solver.direct_dim = 0
        self.constrained_check(solver, ras, nroots=3)
        # Cumulative bounds of overlapping sets.
        cum = [dict(orbitals=[0, 1], min=1, max=2),
               dict(orbitals=[0, 1, 2, 3, 4], min=3, max=3)]
        solver = new_solver()
        solver.restrictions = cum
        solver.max_subgroup = 1
        self.constrained_check(solver, cum, nroots=2)

    def test_labels(self):
        jz = [1, -1, 3, -3, 1, -1, 5, -5]
        parity = [0, 1, 1, 0, 0, 1, 1, 0]
        for labels, mods, targets in (([jz], [0], [0]), ([jz], [0], [2]),
                                      ([jz, parity], [4, 2], [2, 1])):
            for settings in (dict(), dict(max_subgroup=1)):
                solver = new_solver()
                solver.direct_dim = 0
                solver.orbsym = labels if len(labels) > 1 else labels[0]
                solver.orbsym_mod = mods
                solver.target = targets
                for key, value in settings.items():
                    setattr(solver, key, value)
                self.constrained_check(solver, (), labels, mods, targets)
        # GAS and labels together, and the default target (that of the
        # lowest nelec spinors: 1 - 1 + 3 - 3 = 0).
        gas = [dict(orbitals=[0, 1, 2, 3], min=1, max=3)]
        solver = new_solver()
        solver.gas = gas
        solver.orbsym = jz
        self.constrained_check(solver, gas, [jz], [0], [0])

    def test_few_electrons(self):
        from socutils.fci import zfci
        for nelec in (1, 2, 7, 8):
            occslst = filtered_occslst(self.norb, nelec)
            nroots = min(2, len(occslst))
            e_ref, ci_ref = zfci.kernel_dets(self.h1, self.eri, self.norb, occslst,
                                             ecore=self.ecore, nroots=nroots)
            solver = new_solver()

            def rdm12(c):
                if nelec == 1:
                    return (zfci.make_rdm1_dets(c, self.norb, occslst),
                            numpy.zeros((self.norb,) * 4))
                return zfci.make_rdm12_dets(c, self.norb, occslst)
            self.compare(solver, e_ref, ci_ref, rdm12,
                         lambda b, k: zfci.trans_rdm1_dets(b, k, self.norb, occslst),
                         nelec=nelec, nroots=nroots)
        solver = new_solver()
        e, ci = solver.kernel(self.h1, self.eri, self.norb, 0, ecore=self.ecore)
        self.assertAlmostEqual(e, self.ecore, 12)

    def test_davidson_larger(self):
        from socutils.fci import zfci
        norb, nelec, nroots = 12, 5, 3
        h1, eri = random_hamiltonian(norb, seed=2)
        # Orbital energies that make the diagonal a fair preconditioner, as
        # in a molecule; the plain random Hamiltonian converges very slowly.
        h1 = h1 + numpy.diag(4.0 * numpy.arange(norb))
        ref = zfci.FCISolver()
        e_ref, _ = ref.kernel(h1, eri, norb, nelec, nroots=nroots)
        energies = []
        for size in (12, 3):
            solver = new_solver()
            solver.max_subgroup = size
            e, ci = solver.kernel(h1, eri, norb, nelec, nroots=nroots)
            self.assertTrue(solver.converged)
            self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, 9)
            energies.append(numpy.asarray(e))
        self.assertAlmostEqual(abs(energies[0] - energies[1]).max(), 0, 10)


# -- against SHCI ---------------------------------------------------------------

def have_zshci():
    from socutils.hci import shci
    return os.path.isfile(shci.DEFAULT_EXE)


@unittest.skipUnless(have_zshci(), 'zshci executable not found')
class AgainstSHCI(unittest.TestCase):
    '''GASCI against SHCI on the same spinor problems: occslst = 'all' (full
    CI), occslst of a constrained space, and a converged selection under
    restrictions.'''

    @classmethod
    def setUpClass(cls):
        cls.norb, cls.nelec = 8, 4
        cls.h1, cls.eri = random_hamiltonian(cls.norb, seed=13)

    def compare(self, shci_solver, solver, nroots=2):
        norb, nelec = self.norb, self.nelec
        e_ref, ci_ref = shci_solver.kernel(self.h1, self.eri, norb, nelec, nroots=nroots)
        e, ci = solver.kernel(self.h1, self.eri, norb, nelec, nroots=nroots)
        self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, 8)
        for k in range(nroots):
            dm1_ref, dm2_ref = shci_solver.make_rdm12(ci_ref[k], norb, nelec)
            dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
            self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 6)
            self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 6)
        t_ref = shci_solver.trans_rdm1(ci_ref[0], ci_ref[1], norb, nelec)
        t = solver.trans_rdm1(ci[0], ci[1], norb, nelec)
        self.assertAlmostEqual(abs(align_phase(t, t_ref) - t_ref).max(), 0, 6)

    def test_full_ci(self):
        from socutils.hci import shci
        ref = shci.SHCI()
        ref.occslst = 'all'
        ref.davidson_tol = 1e-10
        self.compare(ref, new_solver(max_subgroup=2))

    def test_gas(self):
        from socutils.hci import shci
        gas = [dict(orbitals=[0, 1, 2, 3], min=2, max=3),
               dict(orbitals=[4, 5, 6, 7], min=1, max=2)]
        ref = shci.SHCI()
        ref.occslst = filtered_occslst(self.norb, self.nelec, gas)
        ref.davidson_tol = 1e-10
        self.compare(ref, new_solver(gas=gas))
        # The same space reached by selection under restrictions.
        ref = shci.SHCI()
        ref.restrictions = gas
        ref.initial_dets = [[0, 1, 4, 5]]
        ref.davidson_tol = 1e-10
        ref.extra = {'davidson': {'tol': 1e-10},
                     'select': {'schedule': [[0, 1e-12]], 'max_iter': 20,
                                'energy_tol': 1e-12, 'davidson_tol_loose': 1e-10}}
        self.compare(ref, new_solver(restrictions=gas, max_subgroup=1))

    def test_labels(self):
        from socutils.hci import shci
        jz = [1, -1, 3, -3, 1, -1, 5, -5]
        ref = shci.SHCI()
        ref.occslst = filtered_occslst(self.norb, self.nelec, (), [jz], [4], [2])
        ref.davidson_tol = 1e-10
        self.compare(ref, new_solver(orbsym=jz, orbsym_mod=4, target=2))


# -- spinor CASSCF ------------------------------------------------------------------

@unittest.skipUnless(have_zshci(), 'zshci executable not found')
class SpinorCASSCF(unittest.TestCase):
    def test_casscf(self):
        import tempfile
        from pyscf import gto
        from socutils.hci import shci
        from socutils.mcscf import zmcscf
        from socutils.scf import spinor_hf
        here = os.getcwd()
        # zcasci leaves an FCIDUMP in the working directory.
        os.chdir(tempfile.mkdtemp())
        try:
            mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='sto-3g', verbose=0)
            mf = spinor_hf.KRHF(mol).density_fit()
            mf.kernel()
            results = []
            for make in (lambda: shci.SHCI(mol), lambda: new_solver()):
                mc = zmcscf.CASSCF(mf, 8, 6)
                mc.fcisolver = make()
                if isinstance(mc.fcisolver, shci.SHCI):
                    mc.fcisolver.occslst = 'all'
                    mc.fcisolver.davidson_tol = 1e-10
                mc.max_cycle_macro = 60
                mc.kernel()
                self.assertTrue(mc.converged)
                dm1 = mc.fcisolver.make_rdm1(mc.ci, 8, 6)
                results.append((mc.e_tot, numpy.linalg.eigvalsh(dm1)))
        finally:
            os.chdir(here)
        (e_ref, occ_ref), (e, occ) = results
        self.assertAlmostEqual(e, e_ref, 8)
        self.assertAlmostEqual(abs(occ - occ_ref).max(), 0, 5)


# -- spatial orbitals ---------------------------------------------------------------

def flatten(x):
    '''Nested tuples of arrays as one vector.'''
    if isinstance(x, (tuple, list)):
        return numpy.hstack([flatten(y) for y in x])
    return numpy.ravel(x)


def spatial_filtered(norb, nelec, constraints):
    '''Determinants of SpatialSHCI (spin orbitals 2P, 2P + 1) of the space.'''
    nalpha, nbeta = nelec
    out = []
    for alpha in itertools.combinations(range(norb), nalpha):
        for beta in itertools.combinations(range(norb), nbeta):
            good = all(c['min'] <= sum(1 for p in alpha + beta if p in c['orbitals'])
                       <= c['max'] for c in constraints)
            if good:
                out.append([2 * p for p in alpha] + [2 * p + 1 for p in beta])
    return out


class Spatial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyscf import gto, mcscf, scf
        cls.mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59',
                        basis='6-31g', symmetry='c2v', verbose=0)
        cls.mf = scf.RHF(cls.mol).run(conv_tol=1e-12)
        cls.norb, cls.nelec = 8, (4, 4)
        mc = mcscf.CASCI(cls.mf, cls.norb, cls.nelec)
        cls.h1, cls.ecore = mc.get_h1eff()
        cls.eri = mc.get_h2eff()
        cls.orbsym = cls.mf.mo_coeff.orbsym[mc.ncore:mc.ncore + cls.norb]

    def against_direct_spin1(self, solver, nelec, nroots=3):
        from pyscf import fci
        norb = self.norb
        ref = fci.direct_spin1.FCI()
        ref.conv_tol = 1e-12
        e_ref, ci_ref = ref.kernel(self.h1, self.eri, norb, nelec, nroots=nroots,
                                   ecore=self.ecore)
        e, ci = solver.kernel(self.h1, self.eri, norb, nelec, nroots=nroots,
                              ecore=self.ecore)
        self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, 9)
        for k in range(nroots):
            dm1_ref, dm2_ref = ref.make_rdm12(ci_ref[k], norb, nelec)
            dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
            self.assertEqual(dm2.shape, (norb,) * 4)
            self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 6)
            self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 6)
            (a, b), (aa, ab, bb) = solver.make_rdm12s(ci[k], norb, nelec)
            (a0, b0), (aa0, ab0, bb0) = ref.make_rdm12s(ci_ref[k], norb, nelec)
            for x, y in ((a, a0), (b, b0), (aa, aa0), (ab, ab0), (bb, bb0)):
                self.assertAlmostEqual(abs(x - y).max(), 0, 6)
            ss = solver.spin_square(ci[k], norb, nelec)[0]
            self.assertAlmostEqual(ss, ref.spin_square(ci_ref[k], norb, nelec)[0], 6)
        for k in range(1, nroots):
            # Each state has an arbitrary sign.
            sign = None
            for name in ('trans_rdm12s', 'trans_rdm12', 'trans_rdm1'):
                t_ref = getattr(ref, name)(ci_ref[0], ci_ref[k], norb, nelec)
                t = getattr(solver, name)(ci[0], ci[k], norb, nelec)
                t, t_ref = flatten(t), flatten(t_ref)
                if sign is None:
                    sign = 1 if abs(t - t_ref).max() < abs(t + t_ref).max() else -1
                self.assertAlmostEqual(abs(sign * t - t_ref).max(), 0, 6)
        return e

    def test_full_space(self):
        e0 = self.against_direct_spin1(new_solver(gasci.SpatialGASCI), self.nelec)
        e1 = self.against_direct_spin1(
            new_solver(gasci.SpatialGASCI, subgroups=[[0, 5], [1, 2, 3], [4], [6, 7]]),
            self.nelec)
        self.assertAlmostEqual(abs(numpy.asarray(e0) - e1).max(), 0, 10)

    def test_open_shell(self):
        self.against_direct_spin1(new_solver(gasci.SpatialGASCI, max_subgroup=3), (5, 3))
        solver = new_solver(gasci.SpatialGASCI)
        solver.spin = 2
        e, _ = solver.kernel(self.h1, self.eri, self.norb, 8, ecore=self.ecore)
        from pyscf import fci
        e_ref = fci.direct_spin1.FCI().kernel(self.h1, self.eri, self.norb, (5, 3),
                                              ecore=self.ecore)[0]
        self.assertAlmostEqual(e, e_ref, 9)

    def test_point_group(self):
        from pyscf import fci
        norb, nelec = self.norb, self.nelec
        for wfnsym in (0, 3):
            ref = fci.direct_spin1_symm.FCI(self.mol)
            ref.conv_tol = 1e-12
            ref.wfnsym = wfnsym
            e_ref, ci_ref = ref.kernel(self.h1, self.eri, norb, nelec, nroots=2,
                                       ecore=self.ecore, orbsym=self.orbsym)
            solver = new_solver(gasci.SpatialGASCI, orbsym=self.orbsym, wfnsym=wfnsym)
            e, ci = solver.kernel(self.h1, self.eri, norb, nelec, nroots=2,
                                  ecore=self.ecore)
            self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, 9)
            self.assertLess(solver.plan(norb, nelec).dim, 70 * 70 // 2)
            for k in range(2):
                dm1_ref, dm2_ref = ref.make_rdm12(ci_ref[k], norb, nelec)
                dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
                self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 5)
                self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 5)

    @unittest.skipUnless(have_zshci(), 'zshci executable not found')
    def test_gas_against_spatial_shci(self):
        from socutils.hci import shci
        norb = self.norb
        gas = [dict(orbitals=[0, 1, 2], min=5, max=6),
               dict(orbitals=[3, 4], min=2, max=4),
               dict(orbitals=[5, 6, 7], min=0, max=1)]
        for nelec in ((4, 4), (5, 3)):
            ref = shci.SpatialSHCI(self.mol)
            ref.occslst = spatial_filtered(norb, nelec, gas)
            ref.davidson_tol = 1e-10
            e_ref, ci_ref = ref.kernel(self.h1, self.eri, norb, nelec, nroots=2,
                                       ecore=self.ecore)
            results = []
            for settings in (dict(gas=gas), dict(restrictions=gas, max_subgroup=1),
                             dict(gas=gas, orbsym=self.orbsym,
                                  wfnsym=0 if nelec == (4, 4) else None)):
                solver = new_solver(gasci.SpatialGASCI, **settings)
                e, ci = solver.kernel(self.h1, self.eri, norb, nelec, nroots=2,
                                      ecore=self.ecore)
                if 'orbsym' in settings:
                    # The lowest states of the space happen to be of that
                    # symmetry.
                    self.assertAlmostEqual(e[0], e_ref[0], 8)
                    continue
                self.assertEqual(solver.plan(norb, nelec).dim, len(ref.occslst))
                self.assertAlmostEqual(abs(numpy.asarray(e) - e_ref).max(), 0, 8)
                for k in range(2):
                    dm1_ref, dm2_ref = ref.make_rdm12(ci_ref[k], norb, nelec)
                    dm1, dm2 = solver.make_rdm12(ci[k], norb, nelec)
                    self.assertAlmostEqual(abs(dm1 - dm1_ref).max(), 0, 6)
                    self.assertAlmostEqual(abs(dm2 - dm2_ref).max(), 0, 6)
                results.append(numpy.asarray(e))
            self.assertAlmostEqual(abs(results[0] - results[1]).max(), 0, 10)

    def test_casscf(self):
        from pyscf import mcscf
        ref = mcscf.CASSCF(self.mf, 6, 6).run(conv_tol=1e-10)
        mc = mcscf.CASSCF(self.mf, 6, 6)
        mc.fcisolver = new_solver(gasci.SpatialGASCI, mol=self.mol)
        mc.conv_tol = 1e-10
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_tot, ref.e_tot, 8)
        # A RAS-type CASSCF runs too and lies above the full one.
        mc = mcscf.CASSCF(self.mf, 6, 6)
        mc.fcisolver = new_solver(gasci.SpatialGASCI,
                                  gas=[dict(orbitals=[0, 1, 2], min=5, max=6)])
        mc.conv_tol = 1e-9
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertGreater(mc.e_tot, ref.e_tot)


if __name__ == '__main__':
    print('Tests for hci.gasci')
    unittest.main()
