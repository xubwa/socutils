'''
Product spaces of socutils.hci.gasci (lib/zgasci/PRODUCT_SPACE.md):
ProductPlan, ProductSpaceCI (spatial orbitals) and ProductSpinorCI.

- All strings: the same space as full CI (SpatialGASCI, GASCI); sigma
  element by element and energies.
- Random subsets: sigma against H built from the operators on the bit
  strings (test_gasci.exact_hamiltonian), real and complex, with labels;
  against pyscf.fci.selected_ci.contract_2e.
- Energies, roots, 1- and 2-RDMs and spin_square against
  pyscf.fci.selected_ci (kernel_fixed_space) on N2 / cc-pVDZ.
- The density mode of the BLAS kernel (zg_rdm2_blas) against the reference
  zg_rdm2 in combinatorial spaces.

Run:
    python hci/test/test_product_space.py
'''

import itertools
import os
import sys
import unittest

import numpy

from socutils.hci import gasci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gasci import exact_hamiltonian, random_hamiltonian  # noqa: E402


def sub_strings(n, counts):
    '''All strings over n spinors with one of the electron counts.'''
    return [sum(1 << p for p in occ) for k in counts
            for occ in itertools.combinations(range(n), k)]


def subset(strs, size, rng):
    strs = list(strs)
    return [strs[i] for i in sorted(rng.choice(len(strs), size, replace=False))]


def bit_strings(plan):
    return [sum(int(w) << (64 * j) for j, w in enumerate(row)) for row in plan.dets()]


def spin_orbital_integrals(h1, eri, norb):
    '''Spin-orbital h and (pq|rs) over alpha 0 .. norb - 1, beta norb ..
    2 norb - 1 (the order of the spatial product plans).'''
    n = 2 * norb
    h = numpy.zeros((n, n))
    h[:norb, :norb] = h[norb:, norb:] = h1
    g = numpy.zeros((n, n, n, n))
    for a in (slice(0, norb), slice(norb, n)):
        for b in (slice(0, norb), slice(norb, n)):
            g[a, a, b, b] = eri
    return h, g


def random_spatial(norb, seed=3):
    rng = numpy.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = h1 + h1.T
    f = rng.standard_normal((2 * norb, norb, norb))
    f = f + f.transpose(0, 2, 1)
    eri = 0.1 * numpy.einsum('lpq,lrs->pqrs', f, f)
    return h1, eri


def same_order(plan_a, plan_b):
    '''index such that the determinant k of plan_a is index[k] of plan_b.'''
    where = {d: i for i, d in enumerate(bit_strings(plan_b))}
    return numpy.array([where[d] for d in bit_strings(plan_a)])


class AllStrings(unittest.TestCase):
    '''All strings of the sides: the full CI space.'''

    def test_spatial(self):
        norb = 6
        h1, eri = random_spatial(norb)
        for nelec, orbsym in (((3, 3), None), ((4, 2), None),
                              ((3, 3), [0, 1, 2, 3, 0, 1])):
            with self.subTest(nelec=nelec, orbsym=orbsym):
                prod = gasci.ProductSpaceCI(ci_strs=(sub_strings(norb, [nelec[0]]),
                                                     sub_strings(norb, [nelec[1]])))
                full = gasci.SpatialGASCI()
                for s in (prod, full):
                    s.orbsym = orbsym
                    s.direct_dim = 1000
                pp, pf = prod.plan(norb, nelec), full.plan(norb, nelec)
                self.assertEqual(pp.dim, pf.dim)
                index = same_order(pp, pf)
                v = prod._hamiltonian(h1, eri, norb, sum(nelec), pp)
                vf = full._hamiltonian(h1, eri, norb, sum(nelec), pf)
                c = numpy.random.default_rng(1).standard_normal(pp.dim)
                cf = numpy.empty_like(c)
                cf[index] = c
                self.assertLess(abs(pp.sigma(v, c) - pf.sigma(vf, cf, 'reference')[index]).max(),
                                1e-12)
                self.assertLess(abs(pp.hdiag(v) - pf.hdiag(vf)[index]).max(), 1e-12)
                e, _ = prod.kernel(h1, eri, norb, nelec, nroots=3)
                ef, _ = full.kernel(h1, eri, norb, nelec, nroots=3)
                self.assertLess(abs(numpy.asarray(e) - ef).max(), 1e-12)

    def test_spinors(self):
        n, nelec = 10, 4
        h1, eri = random_hamiltonian(n, seed=5)
        jz = [1, -1, 3, -3, 5, 1, -1, 3, -3, -5]
        parity = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        for labels in (None, [jz, parity]):
            with self.subTest(labels=labels is not None):
                strs = [sub_strings(5, range(6))] * 2
                prod = gasci.ProductSpinorCI(ci_strs=strs)
                full = gasci.GASCI()
                full.subgroups = ([list(range(5))], [list(range(5, 10))])
                for s in (prod, full):
                    s.direct_dim = 1000
                    if labels is not None:
                        s.orbsym, s.orbsym_mod, s.target = labels, [0, 2], [0, 1]
                pp, pf = prod.plan(n, nelec), full.plan(n, nelec)
                self.assertEqual(pp.dim, pf.dim)
                index = same_order(pp, pf)
                v = prod._hamiltonian(h1, eri, n, nelec, pp)
                rng = numpy.random.default_rng(2)
                c = rng.standard_normal(pp.dim) + 1j * rng.standard_normal(pp.dim)
                cf = numpy.empty_like(c)
                cf[index] = c
                self.assertLess(abs(pp.sigma(v, c) - pf.sigma(v, cf, 'reference')[index]).max(),
                                1e-12)
                e, _ = prod.kernel(h1, eri, n, nelec, nroots=2)
                ef, _ = full.kernel(h1, eri, n, nelec, nroots=2)
                self.assertLess(abs(numpy.asarray(e) - ef).max(), 1e-12)


class Subsets(unittest.TestCase):
    '''sigma, hdiag and the pair density against H built determinant by
    determinant from the operators.'''

    def check(self, plan, h, g, v, real):
        dets = bit_strings(plan)
        self.assertEqual(len(set(dets)), plan.dim)
        hmat = exact_hamiltonian(h, g, dets)
        rng = numpy.random.default_rng(4)
        c = rng.standard_normal(plan.dim)
        b = rng.standard_normal(plan.dim)
        if not real:
            c = c + 1j * rng.standard_normal(plan.dim)
            b = b + 1j * rng.standard_normal(plan.dim)
        self.assertLess(abs(plan.hdiag(v) - hmat.diagonal().real).max(), 1e-12)
        ref = hmat @ c
        settings = ({}, {'ZGASCI_BLAS_DIRECT': '0', 'ZGASCI_ROWS_DENSE': '0'},
                    {'ZGASCI_BLAS_DIRECT': '1e300', 'ZGASCI_BLAS_SMALL': '0',
                     'ZGASCI_BLAS_COLS': '3'})
        for setting in settings:
            saved = {key: os.environ.get(key) for key in setting}
            os.environ.update(setting)
            try:
                sigma = plan.sigma(v, c)
            finally:
                for key, value in saved.items():
                    if value is None:
                        del os.environ[key]
                    else:
                        os.environ[key] = value
            self.assertLess(abs(sigma - ref).max(), 1e-12 * max(1, abs(ref).max()),
                            msg=str(setting))
        gp = plan.rdm2_pairs(b, c)
        self.assertLess(abs(numpy.sum(v * gp) - b.conj() @ hmat @ c), 1e-10)
        return hmat

    def test_spatial(self):
        norb = 6
        h1, eri = random_spatial(norb, seed=8)
        h, g = spin_orbital_integrals(h1, eri, norb)
        rng = numpy.random.default_rng(6)
        for nelec, sizes, orbsym in (((3, 3), (11, 9), None), ((3, 2), (14, 7), None),
                                     ((2, 2), (9, 12), [0, 3, 1, 2, 3, 0])):
            with self.subTest(nelec=nelec, orbsym=orbsym):
                sa = subset(sub_strings(norb, [nelec[0]]), sizes[0], rng)
                sb = subset(sub_strings(norb, [nelec[1]]), sizes[1], rng)
                solver = gasci.ProductSpaceCI(ci_strs=(sa, sb))
                solver.orbsym = orbsym
                plan = solver.plan(norb, nelec)
                if orbsym is None:
                    self.assertEqual(plan.dim, sizes[0] * sizes[1])
                v = solver._hamiltonian(h1, eri, norb, sum(nelec), plan)
                self.check(plan, h, g, v, real=True)

    def test_spinors(self):
        n, nelec, nside_a = 10, 4, 4
        h1, eri = random_hamiltonian(n, seed=9)
        jz = [1, -1, 3, -3, 5, 1, -1, 3, -3, -5]
        parity = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        rng = numpy.random.default_rng(7)
        for labels, real in ((None, False), ([jz, parity], False), ([jz], True)):
            with self.subTest(labels=labels is not None, real=real):
                sa = subset(sub_strings(nside_a, range(5)), 11, rng)
                sb = subset(sub_strings(n - nside_a, range(5)), 25, rng)
                solver = gasci.ProductSpinorCI(ci_strs=(sa, sb), nside_a=nside_a)
                if labels is not None:
                    solver.orbsym = labels
                    solver.orbsym_mod = [0, 2][:len(labels)]
                    solver.target = [0, 1][:len(labels)]
                hh, gg = (h1.real.copy(), eri.real.copy()) if real else (h1, eri)
                plan = solver.plan(n, nelec)
                v = solver._hamiltonian(hh, gg, n, nelec, plan)
                hmat = self.check(plan, hh, gg, v, real)
                e, c = solver.kernel(hh, gg, n, nelec, nroots=2)
                self.assertLess(abs(numpy.asarray(e) - numpy.linalg.eigvalsh(hmat)[:2]).max(),
                                1e-9)

    def test_wide_side(self):
        # 66 spinors on side A (strings of two words, occupied spinors on
        # both sides of bit 64), 6 on side B, random v: against the
        # combinatorial plan of the same space (A spinors 0 .. 60 empty).
        n, nside_a, nelec = 72, 66, 4
        jz = [(p * p + 3 * p) % 5 - 2 for p in range(n)]
        npair = n * (n - 1) // 2
        comb = gasci.Plan(n, nelec, [16, 16, 16, 13, 5, 6], 5, [([0, 1, 2, 3], 0, 0)],
                          [jz], [0], [0])
        sa = [s << 61 for s in sub_strings(5, range(5))]
        sb = sub_strings(6, range(5))
        prod = gasci.ProductPlan(n, nelec, nside_a, gasci.string_words(sa, nside_a),
                                 gasci.string_words(sb, n - nside_a), [jz], [0], [0])
        self.assertEqual(prod.side_words, [2, 1])
        self.assertEqual(prod.dim, comb.dim)
        index = same_order(prod, comb)
        rng = numpy.random.default_rng(5)
        for real in (False, True):
            v = rng.standard_normal((npair, npair))
            c = rng.standard_normal(prod.dim)
            if not real:
                v = v + 1j * rng.standard_normal((npair, npair))
                c = c + 1j * rng.standard_normal(prod.dim)
            cc = numpy.empty_like(c)
            cc[index] = c
            ref = comb.sigma(v, cc, 'reference')[index]
            self.assertLess(abs(prod.sigma(v, c) - ref).max(), 1e-12 * abs(ref).max())
            self.assertLess(abs(prod.hdiag(v) - comb.hdiag(v)[index]).max(), 1e-12)
            self.assertLess(abs(prod.rdm2_pairs(c, c) - comb.rdm2_pairs(cc, cc)).max(),
                            1e-11)

    def test_against_selected_ci_contract(self):
        from pyscf.fci import direct_spin1, selected_ci
        from pyscf.fci import cistring
        norb, nelec = 8, (4, 3)
        h1, eri = random_spatial(norb, seed=12)
        rng = numpy.random.default_rng(13)
        sa = numpy.sort(rng.choice(cistring.make_strings(range(norb), 4), 37, replace=False))
        sb = numpy.sort(rng.choice(cistring.make_strings(range(norb), 3), 29, replace=False))
        solver = gasci.ProductSpaceCI(ci_strs=(sa, sb))
        plan = solver.plan(norb, nelec)
        self.assertTrue((gasci.words_to_ints(plan.strings(0)) == sa).all())
        v = solver._hamiltonian(h1, eri, norb, sum(nelec), plan)
        c = rng.standard_normal(plan.dim)
        h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, .5)
        ref = selected_ci.contract_2e(h2e, selected_ci._as_SCIvector(c.reshape(37, 29), (sa, sb)),
                                      norb, nelec)
        self.assertLess(abs(plan.sigma(v, c) - numpy.asarray(ref).ravel()).max(), 1e-12)


class AgainstSelectedCI(unittest.TestCase):
    '''N2 / cc-pVDZ, 10 electrons in 12 orbitals above a frozen core, product
    spaces of chosen and random strings, against
    pyscf.fci.selected_ci.kernel_fixed_space.'''

    @classmethod
    def setUpClass(cls):
        from pyscf import gto, mcscf, scf
        from pyscf.fci import cistring
        mol = gto.M(atom='N 0 0 0; N 0 0 1.12', basis='cc-pvdz', symmetry='d2h',
                    verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        cls.norb, cls.nelec = 12, (5, 5)
        mc = mcscf.CASCI(mf, cls.norb, cls.nelec)
        cls.h1, cls.ecore = mc.get_h1eff()
        cls.eri = mc.get_h2eff()
        cls.orbsym = numpy.asarray(mf.mo_coeff.orbsym[mc.ncore:mc.ncore + cls.norb])
        # Strings by the orbital-energy sum of their holes and particles (the
        # lowest first), as an approximate weight, and random ones.
        strs = cistring.make_strings(range(cls.norb), 5)
        mo_e = mf.mo_energy[mc.ncore:mc.ncore + cls.norb]
        occ = [[p for p in range(cls.norb) if s >> p & 1] for s in strs]
        # (Rounded: degenerate orbitals tie, and the space does not depend on
        # the last digits of their energies.)
        cost = numpy.round(numpy.array([sum(mo_e[o]) for o in occ]), 6)
        cls.top = numpy.sort(strs[numpy.argsort(cost, kind='stable')[:180]])
        rng = numpy.random.default_rng(1)
        cls.rand_a = numpy.sort(rng.choice(strs, 150, replace=False))
        cls.rand_b = numpy.sort(numpy.union1d(rng.choice(strs, 120, replace=False),
                                              cls.top[:20]))

    def solve(self, sa, sb, nroots):
        '''Energies against selected_ci: kernel_fixed_space for one root;
        for several, ARPACK (eigsh) on selected_ci.contract_2e, since the
        Davidson of selected_ci starts from a singlet guess and does not
        converge close roots tightly.'''
        from pyscf.fci import direct_spin1, selected_ci
        from scipy.sparse.linalg import LinearOperator, eigsh
        norb, nelec = self.norb, self.nelec
        if nroots == 1:
            ref = selected_ci.SCI()
            ref.conv_tol = 1e-12
            e_ref, c_ref = selected_ci.kernel_fixed_space(
                ref, self.h1, self.eri, norb, nelec, (sa, sb), ecore=self.ecore)
            e_ref, c_ref = [e_ref], [c_ref]
        else:
            h2e = direct_spin1.absorb_h1e(self.h1, self.eri, norb, nelec, .5)
            link = selected_ci._all_linkstr_index((sa, sb), norb, nelec)
            dim = len(sa) * len(sb)

            def hop(x):
                x = selected_ci._as_SCIvector(numpy.reshape(x, (len(sa), len(sb))), (sa, sb))
                return numpy.asarray(selected_ci.contract_2e(h2e, x, norb, nelec, link)).ravel()
            e_ref, vecs = eigsh(LinearOperator((dim, dim), matvec=hop, dtype=float),
                                k=nroots + 2, which='SA', tol=1e-14,
                                v0=numpy.random.default_rng(5).standard_normal(dim))
            order = numpy.argsort(e_ref)[:nroots]
            e_ref = e_ref[order] + self.ecore
            # (selected_ci.make_rdm2s needs C-contiguous vectors.)
            c_ref = [selected_ci._as_SCIvector(
                numpy.ascontiguousarray(vecs[:, k].reshape(len(sa), len(sb))), (sa, sb))
                for k in order]
        solver = gasci.ProductSpaceCI(ci_strs=(sa, sb))
        solver.conv_tol = 1e-12
        solver.conv_tol_residual = 1e-7
        solver.max_space = 30
        solver.max_cycle = 300
        e, c = solver.kernel(self.h1, self.eri, norb, nelec, nroots=nroots, ecore=self.ecore)
        if nroots == 1:
            e, c = [e], [c]
        self.assertLess(abs(numpy.asarray(e) - e_ref).max(), 1e-10,
                        msg='%s %s' % (numpy.asarray(e), e_ref))
        return solver, c, c_ref

    def test_top_strings(self):
        from pyscf.fci import selected_ci
        norb, nelec = self.norb, self.nelec
        solver, c, c_ref = self.solve(self.top, self.top, nroots=3)
        # Density matrices of the same vectors (the excited roots are only
        # converged to about 1e-4 in the vector: close roots).
        for k in range(3):
            x = numpy.asarray(c_ref[k]).ravel()
            dm1s = solver.make_rdm1s(x, norb, nelec)
            ref1s = selected_ci.make_rdm1s(c_ref[k], norb, nelec)
            for a, b in zip(dm1s, ref1s):
                self.assertLess(abs(a - b).max(), 1e-12)
            self.assertLess(abs(solver.make_rdm1(x, norb, nelec)
                                - selected_ci.make_rdm1(c_ref[k], norb, nelec)).max(), 1e-12)
            dm2s = solver.make_rdm12s(x, norb, nelec)[1]
            ref2s = selected_ci.make_rdm2s(c_ref[k], norb, nelec)
            for a, b in zip(dm2s, ref2s):
                self.assertLess(abs(a - b).max(), 1e-12)
            ss = solver.spin_square(x, norb, nelec)
            ss_ref = selected_ci.spin_square(c_ref[k], norb, nelec)
            self.assertAlmostEqual(ss[0], ss_ref[0], 11)
            self.assertAlmostEqual(ss[1], ss_ref[1], 11)
        # The ground state of either solver.
        dm1 = solver.make_rdm1(c[0], norb, nelec)
        self.assertLess(abs(dm1 - selected_ci.make_rdm1(c_ref[0], norb, nelec)).max(), 1e-7)

    def test_random_strings(self):
        from pyscf.fci import selected_ci
        norb, nelec = self.norb, self.nelec
        solver, c, c_ref = self.solve(self.rand_a, self.rand_b, nroots=1)
        dm1a, dm1b = solver.make_rdm1s(c[0], norb, nelec)
        ra, rb = selected_ci.make_rdm1s(c_ref[0], norb, nelec)
        self.assertLess(max(abs(dm1a - ra).max(), abs(dm1b - rb).max()), 1e-7)
        self.assertAlmostEqual(solver.spin_square(c[0], norb, nelec)[0],
                               selected_ci.spin_square(c_ref[0], norb, nelec)[0], 7)
        # The vector is C[a, b] over the sorted lists, as selected_ci's.
        m = solver.to_matrix(c[0], norb, nelec)
        overlap = abs(numpy.vdot(m.ravel(), numpy.asarray(c_ref[0]).ravel()))
        self.assertAlmostEqual(overlap, 1, 9)

    def test_symmetry(self):
        # D2h labels: the block of the ground state (Ag) has its energy.
        solver = gasci.ProductSpaceCI(ci_strs=(self.top, self.top))
        solver.conv_tol = 1e-12
        e0, _ = solver.kernel(self.h1, self.eri, self.norb, self.nelec, ecore=self.ecore)
        sym = gasci.ProductSpaceCI(ci_strs=(self.top, self.top))
        sym.conv_tol = 1e-12
        sym.orbsym, sym.wfnsym = self.orbsym, 0
        e1, c1 = sym.kernel(self.h1, self.eri, self.norb, self.nelec, ecore=self.ecore)
        self.assertLess(sym.plan(self.norb, self.nelec).dim, len(self.top) ** 2)
        self.assertAlmostEqual(e0, e1, 10)

    def test_warm_start(self):
        # A space and a larger one: the vector of the first is a start of the
        # second; to_matrix / from_matrix round trip.
        small = self.top[:120]
        solver = gasci.ProductSpaceCI(ci_strs=(small, small))
        solver.conv_tol = 1e-12
        e_small, c_small = solver.kernel(self.h1, self.eri, self.norb, self.nelec,
                                         ecore=self.ecore)
        m = solver.to_matrix(c_small, self.norb, self.nelec)
        back = solver.from_matrix(m, (small, small), self.norb, self.nelec)
        self.assertLess(abs(back - c_small).max(), 1e-15)
        e, _ = solver.kernel(self.h1, self.eri, self.norb, self.nelec, ecore=self.ecore,
                             ci_strs=(self.top, self.top), ci0=(small, small, m))
        cold = gasci.ProductSpaceCI(ci_strs=(self.top, self.top))
        cold.conv_tol = 1e-12
        e_cold, _ = cold.kernel(self.h1, self.eri, self.norb, self.nelec, ecore=self.ecore)
        self.assertLess(e, e_small)
        self.assertAlmostEqual(e, e_cold, 9)


class SpinFlip(unittest.TestCase):
    '''spin_flip = +1 / -1 with the same alpha and beta strings: the
    eigenvalues of the even and the odd part of the full product space, and
    sigma_flip against sigma on vectors of the subspace.'''

    def run_case(self, nstr, orbsym, seed):
        norb, nelec = 8, (4, 4)
        h1, eri = random_spatial(norb, seed=seed)
        rng = numpy.random.default_rng(seed)
        strs = subset(sub_strings(norb, [4]), nstr, rng)
        full = gasci.ProductSpaceCI(ci_strs=(strs, strs))
        full.orbsym = orbsym
        plan = full.plan(norb, nelec)
        self.assertTrue(plan.mirror)
        v = full._hamiltonian(h1, eri, norb, 8, plan)
        hmat = numpy.array([plan.sigma(v, x) for x in numpy.eye(plan.dim)]).T
        e_all, vecs = numpy.linalg.eigh(hmat)
        flip = plan.flip_index()
        parity_of = numpy.einsum('ik,ik->k', vecs[flip], vecs)
        self.assertLess(abs(abs(parity_of) - 1).max(), 1e-8)
        for parity in (1, -1):
            c = rng.standard_normal(plan.dim)
            c = c + parity * c[flip]
            self.assertLess(abs(plan.sigma_flip(v, c, parity) - plan.sigma(v, c)).max(), 1e-12)
            solver = gasci.ProductSpaceCI(ci_strs=(strs, strs))
            solver.orbsym = orbsym
            solver.spin_flip = parity
            solver.conv_tol = 1e-12
            solver.conv_tol_residual = 1e-7
            solver.max_space = 30
            solver.max_cycle = 300
            e, c = solver.kernel(h1, eri, norb, nelec, nroots=3)
            expect = e_all[parity_of * parity > 0][:3]
            self.assertLess(abs(numpy.asarray(e) - expect).max(), 1e-10)
            self.assertLess(abs(c[0][flip] - parity * c[0]).max(), 1e-10)

    def test_davidson(self):
        self.run_case(40, None, 21)
        self.run_case(48, [0, 1, 2, 3, 0, 1, 2, 3], 22)

    def test_direct(self):
        self.run_case(14, None, 23)

    def test_not_mirror(self):
        solver = gasci.ProductSpaceCI(ci_strs=([15, 23, 27], [15, 23, 29]))
        solver.spin_flip = 1
        h1, eri = random_spatial(6, seed=1)
        with self.assertRaises(ValueError):
            solver.kernel(h1, eri, 6, (4, 4))


class DensityMode(unittest.TestCase):
    '''zg_rdm2_blas against the reference zg_rdm2 in combinatorial spaces.'''

    def compare(self, n, nelec, sub_sizes, nsub_a, constraints=(), labels=None,
                mods=(), targets=(), real=False):
        plan = gasci.Plan(n, nelec, sub_sizes, nsub_a, constraints, labels, mods, targets)
        rng = numpy.random.default_rng(5)
        b, c = rng.standard_normal((2, plan.dim))
        if not real:
            b = b + 1j * rng.standard_normal(plan.dim)
            c = c + 1j * rng.standard_normal(plan.dim)
        ref = plan.rdm2_pairs(b, c)
        self.assertLess(abs(plan.rdm2_pairs(b, c, kernel='blas') - ref).max(), 1e-11)

    def test_spaces(self):
        jz = [1, -1, 3, -3, 1, -1, 5, -5, 3, -3]
        for real in (False, True):
            self.compare(10, 4, [5, 5], 1, real=real)
            self.compare(10, 5, [2, 3, 2, 3], 2, real=real)
            self.compare(10, 4, [3, 2, 3, 2], 2, constraints=[([0, 2], 1, 3)], real=real)
            self.compare(10, 4, [5, 5], 1, labels=[jz], mods=[0], targets=[0], real=real)


if __name__ == '__main__':
    unittest.main()
