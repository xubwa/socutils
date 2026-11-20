#!/usr/bin/env python
'''Finite-field g tensor against the Kramers-doublet mapping.

Same OH radical CASCI(3 in 4) and in-memory store as test_gtensor.py.  The
finite-field route rebuilds the active-space Hamiltonian, adds -B.mu, and
reads G = g g^T from the doublet splittings alone -- no pseudo-spin -- so it
must reproduce the G of :func:`observables.g_tensor` up to the second-order
Zeeman mixing, (mu_B B / gap)^2, for both doublets and for a molecular axis
that is not along any Cartesian direction.
'''

import unittest
import numpy
from socutils.prop import observables
from socutils.prop.test.test_gtensor import oh, casci_pi, MemStore


def setUpModule():
    global mc, store, n
    xyz, n = oh([0.68, 0.57, 0.45])
    mol, mc = casci_pi(xyz)
    store = MemStore(mol)


class GTensorFiniteField(unittest.TestCase):
    def test_matches_mapping(self):
        g_map = observables.g_tensor(mc, store, verbose=0)
        g_ff = observables.g_tensor_ff(mc, store, verbose=0)           # default 0.1 T
        self.assertEqual(g_ff.shape, (2, 3, 3))
        for k in range(2):
            G_map = g_map[k] @ g_map[k].T
            G_ff = g_ff[k] @ g_ff[k].T
            # relative to |G|, but not below the scale of a free electron's G ~ 4: the
            # Omega = 1/2 level has g ~ 0 and its G is itself ~1e-5
            self.assertLess(abs(G_ff - G_map).max(), 1e-7 * max(abs(G_map).max(), 1.0))
        # Omega = 3/2: one principal value along the bond, the same axis as the mapping gives
        w, v = numpy.linalg.eigh(g_ff[0])
        self.assertAlmostEqual(abs(v[:, 2] @ n), 1, 8)

    def test_error_is_second_order_zeeman(self):
        '''The residual against the mapping is the second-order Zeeman mixing, so it
        scales as B^2: a factor ~100 between 1 T and 0.1 T.'''
        g_map = observables.g_tensor(mc, store, verbose=0)
        G_map = g_map @ g_map.transpose(0, 2, 1)
        err = []
        for field in (1.0, 0.1):
            g = observables.g_tensor_ff(mc, store, field=field, verbose=0)
            err.append(abs(g @ g.transpose(0, 2, 1) - G_map).max())
        self.assertLess(err[0], 1e-5 * abs(G_map).max())
        self.assertGreater(err[0] / err[1], 50)
        self.assertLess(err[0] / err[1], 200)

    def test_rebuilt_hamiltonian_must_match(self):
        import copy
        bad = copy.copy(mc)
        bad.e_states = numpy.asarray(getattr(mc, 'e_states', None) or mc.e_tot) + 1e-6
        bad.e_tot = bad.e_states
        with self.assertRaises(ValueError):
            observables.g_tensor_ff(bad, store, verbose=0)


class HyperfineFiniteField(unittest.TestCase):
    def test_matches_mapping(self):
        """A A^T and g A^T from the doublet splittings alone equal the
        mapping's, g A^T including its sign (the stand-in hyperfine operator is the
        spin, parallel to mu)."""
        g = observables.g_tensor(mc, store, verbose=0)
        A = observables.hyperfine_tensor(mc, store, verbose=0)
        G, AAt, gAt = observables.hyperfine_tensor_ff(mc, store, verbose=0)
        self.assertEqual(AAt.shape, (2, 2, 3, 3))
        for k in range(2):
            G_map = g[k] @ g[k].T
            self.assertLess(abs(G[k] - G_map).max(), 1e-7 * max(abs(G_map).max(), 1.0))
            for atom in range(2):
                AAt_map = A[k, atom] @ A[k, atom].T
                gAt_map = g[k] @ A[k, atom].T
                self.assertLess(abs(AAt[k, atom] - AAt_map).max(), 1e-6 * max(abs(AAt_map).max(), 1.0))
                self.assertLess(abs(gAt[k, atom] - gAt_map).max(), 1e-6 * max(abs(gAt_map).max(), 1.0))

    def test_step_is_the_first_crossing(self):
        """hyperfine_step against the spectrum of H = mu_B B n.g.S + A_zz I_z S_z
        (I = 7/2, mu_B = 1/2, S = sigma/2) for an axial doublet, at two angles:
        the first level crossing sits at B_1 and nowhere below it, and B_1 grows
        as 1/cos(theta)."""
        gz, az = 1.3, 2.0e-6
        g = numpy.diag([0.0, 0.0, gz])
        a = numpy.diag([0.0, 0.0, az])
        G, gAt = g @ g.T, g @ a.T
        sz = numpy.diag([0.5, -0.5])
        iz = numpy.diag(numpy.arange(3.5, -3.6, -1.0))
        steps = []
        for theta in (0.0, 1.0):
            n = numpy.array([numpy.sin(theta), 0.0, numpy.cos(theta)])
            b1 = observables.hyperfine_step(G, gAt, n)
            steps.append(b1)

            def min_gap(B):
                h = 0.5 * B * (n @ g)[2] * numpy.kron(sz, numpy.eye(8)) + az * numpy.kron(sz, iz)
                return numpy.diff(numpy.sort(numpy.linalg.eigvalsh(h))).min()
            self.assertLess(min_gap(b1), 1e-12 * az)
            for B in numpy.linspace(0.05, 0.95, 19) * b1:
                self.assertGreater(min_gap(B), 1e-3 * az)
        self.assertAlmostEqual(steps[1] / steps[0], 1 / numpy.cos(1.0), 10)
        self.assertAlmostEqual(steps[0], az / gz, 15)


if __name__ == '__main__':
    unittest.main()
