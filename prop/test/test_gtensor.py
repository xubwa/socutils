#!/usr/bin/env python
'''g tensor and hyperfine-type tensors from the Kramers-doublet block.

OH radical, CASCI(3 in 4) over the pi spinors of the closed-shell OH-
reference, spin-orbit coupling from pyscf's one-electron spinor X2C.  The
magnetic-moment operator is the bare ``mu = (L + g_e S)/2`` in the spinor
AO basis, wrapped in an in-memory store with the CfourPropStore surface.

What is checked is the pseudo-spin machinery, not a picture-changed
number: the lowest level is a doublet (Omega = 3/2) with one principal g
close to the case-(a) value ``2(Lambda + g_e Sigma) = 4.002`` along the
bond and two vanishing ones; the second level (Omega = 1/2) has the
famous near-zero g; the tensors are identical when the molecule is
rotated so its axis lies in the xy plane, orthogonal to the spinor
quantisation axis; and a second time-odd operator (the spin alone,
standing in for a hyperfine operator) lands in the frame fixed by g with
its sign relative to g preserved.
'''

import unittest
import numpy
from pyscf import gto
from pyscf.x2c import x2c
from pyscf.data import nist
from socutils.scf import spinor_hf
from socutils.mcscf import zcasci
from socutils.prop import observables
from socutils.prop.densities import level_densities

BOHR = 0.529177
G_E = nist.G_ELECTRON


class MemStore:
    '''Time-odd operators from pyscf integrals, CfourPropStore surface.'''
    def __init__(self, mol):
        self.mol = mol
        c2 = numpy.vstack(mol.sph2spinor_coeff())          # (2 nao, n2c), alpha rows then beta
        nao = mol.nao_nr()
        s = mol.intor('int1e_ovlp')
        z = numpy.zeros_like(s)
        sigma_so = [numpy.block([[z, s], [s, z]]),
                    numpy.block([[z, -1j * s], [1j * s, z]]),
                    numpy.block([[s, z], [z, -s]])]
        spin = numpy.array([0.5 * c2.conj().T @ m @ c2 for m in sigma_so])   # S_a = sigma_a / 2
        with mol.with_common_orig((0, 0, 0)):
            orb = -1j * mol.intor('int1e_cg_irxp_spinor', comp=3)            # L = -i (i r x nabla)
        mu = 0.5 * (orb + G_E * spin)
        self._ops = {}
        for k, c in enumerate('XYZ'):
            self._ops['UTM10' + c] = mu[k]
        # 'SOM01' stands in for a hyperfine operator: the spin, per atom, with CFOUR's -Tr sign
        for a in range(mol.natm):
            for k in range(3):
                self._ops['SOM01', a, k] = -spin[k]

    def op(self, name, atom=None, comp=None, check_hermi=True):
        return self._ops[name.upper()] if atom is None else self._ops[name.upper(), atom, comp]

    def meta(self, name):
        from socutils.prop import cfour
        return cfour.meta(name)


def oh(axis):
    axis = numpy.asarray(axis, float) / numpy.linalg.norm(axis)
    xyz = numpy.array([[0, 0, 0], 0.97 / BOHR * axis])
    mass = numpy.array([15.995, 1.008])
    xyz = xyz - (mass[:, None] * xyz).sum(0) / mass.sum()
    return xyz, axis


def casci_pi(xyz):
    '''CASCI(3 in 4) over the pi spinors of OH, orbitals from closed-shell OH-.'''
    kw = dict(unit='Bohr', basis='sto-3g', verbose=0)
    mol_a = gto.M(atom=[('O', xyz[0]), ('H', xyz[1])], charge=-1, spin=0, **kw)
    mfa = spinor_hf.SpinorSCF(mol_a)
    mfa.with_x2c = x2c.SpinorX2CHelper(mol_a)
    mfa.conv_tol = 1e-12
    mfa.kernel()
    mol = gto.M(atom=[('O', xyz[0]), ('H', xyz[1])], charge=0, spin=1, **kw)
    mf = spinor_hf.SpinorSCF(mol)
    mf.with_x2c = x2c.SpinorX2CHelper(mol)
    mf.mo_coeff = mfa.mo_coeff
    mf.mo_occ = numpy.zeros(mfa.mo_occ.size); mf.mo_occ[:9] = 1
    mf.mo_energy = mfa.mo_energy; mf.converged = True
    mc = zcasci.CASCI(mf, 4, 3)
    mc.fcisolver.nroots = 4
    mc.kernel()
    return mol, mc


def run(axis):
    xyz, n = oh(axis)
    mol, mc = casci_pi(xyz)
    store = MemStore(mol)
    g = observables.g_tensor(mc, store, verbose=0)
    A = observables.hyperfine_tensor(mc, store, verbose=0)
    return n, mc, g, A


class GTensorOH(unittest.TestCase):
    def test_levels_and_frames(self):
        n, mc, g, A = run([0.68, 0.57, 0.45])
        levels = list(level_densities(mc))
        self.assertEqual([m['degeneracy'] for _, _, _, m in levels], [2, 2])
        # transition-density block is Hermitian and its diagonal equals make_rdm1
        gam = levels[0][1]
        self.assertAlmostEqual(abs(gam[0, 1] - gam[1, 0].conj().T).max(), 0, 10)
        ci0 = mc.ci[0]
        rdm = mc.fcisolver.make_rdm1(ci0, mc.ncas, mc.nelecas)
        self.assertAlmostEqual(abs(gam[0, 0, mc.ncore:mc.ncore + 4, mc.ncore:mc.ncore + 4] - rdm).max(), 0, 10)

        self.assertEqual(g.shape, (2, 3, 3))
        w, v = numpy.linalg.eigh(g[0])
        self.assertAlmostEqual(w[2], 2 * (1 + G_E / 2), delta=0.02)     # Omega = 3/2: 2(Lambda + g_e Sigma)
        self.assertLess(abs(w[:2]).max(), 1e-6)
        self.assertAlmostEqual(abs(v[:, 2] @ n), 1, 8)                   # principal axis along the bond
        w1 = numpy.linalg.eigh(g[1])[0]
        self.assertLess(abs(w1).max(), 0.05)                              # Omega = 1/2: g ~ 0

        # the stand-in hyperfine (the spin) in the g frame: symmetric, along the bond,
        # positive relative to g (S is parallel to mu), magnitude 2 Sigma = 1
        a = A[0, 0]
        self.assertAlmostEqual(abs(a - a.T).max(), 0, 8)
        self.assertAlmostEqual(n @ a @ n, 1.0, delta=0.02)
        self.assertGreater(n @ a @ n, 0)

    def test_invariant_matches_frame_route(self):
        """The basis-independent invariant reproduces the doublet frame route.

        Only to 1e-8: the invariant is quadratic in the block, so taking its
        square root costs half the digits on a vanishing principal value --
        which is why g_tensor uses the SVD-based frame route for a doublet
        and the invariant only for larger multiplets.
        """
        from socutils.prop import cfour
        xyz, _ = oh([0.68, 0.57, 0.45])
        mol, mc = casci_pi(xyz)
        store = MemStore(mol)
        ops = numpy.array([store.op('UTM10' + c) for c in 'XYZ'])
        blocks, _ = observables.level_block(mc, store, ops, cfour.meta('UTM10X'))
        for M in blocks:
            P, _ = observables.pseudo_spin_frame(observables.pseudo_spin_tensor(M))
            g_inv = observables._sym_sqrt(observables.spin_invariant(M))
            self.assertAlmostEqual(abs(2 * P - g_inv).max(), 0, 7)

    def test_orientation_invariance(self):
        n1, _, g1, A1 = run([0.68, 0.57, 0.45])
        n2, _, g2, A2 = run([numpy.cos(0.3), numpy.sin(0.3), 0.0])       # axis in the xy plane
        for t1, t2 in ((g1[0], g2[0]), (g1[1], g2[1])):
            self.assertAlmostEqual(abs(observables.g_principal(t1)[0]
                                       - observables.g_principal(t2)[0]).max(), 0, 7)
        for t1, t2 in ((A1[0, 0], A2[0, 0]), (A1[1, 0], A2[1, 0])):
            self.assertAlmostEqual(abs(observables.hyperfine_principal(t1)[0]
                                       - observables.hyperfine_principal(t2)[0]).max(), 0, 7)
        self.assertAlmostEqual(n1 @ A1[0, 0] @ n1, n2 @ A2[0, 0] @ n2, 7)


def rotation(axis, angle):
    axis = numpy.asarray(axis, float) / numpy.linalg.norm(axis)
    k = numpy.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return numpy.eye(3) + numpy.sin(angle) * k + (1 - numpy.cos(angle)) * k @ k


class PrincipalValues(unittest.TestCase):
    """Synthetic tensors with known principal values and axes.

    g is symmetric by convention, so its principal values are eigenvalues.
    A couples a laboratory index to a pseudo-spin index, ``A -> Q A R^T``
    under a laboratory rotation Q and a doublet-basis rotation R, so the
    only invariants are the singular values; the eigenvalues of the
    symmetric part are *not*, and the diagonal elements in the g frame are
    the observed couplings only when A shares the g axes.
    """
    def setUp(self):
        self.Q = rotation([1, 2, 3], 0.7)
        self.gd = numpy.array([1.1, 0.35, 0.24])
        self.g = self.Q @ numpy.diag(self.gd) @ self.Q.T

    def test_g_principal(self):
        w, v = observables.g_principal(self.g)
        self.assertAlmostEqual(abs(w - self.gd).max(), 0, 12)
        for k in range(3):
            self.assertAlmostEqual(abs(v[:, k] @ self.Q[:, k]), 1, 12)
        self.assertAlmostEqual(abs(v @ numpy.diag(w) @ v.T - self.g).max(), 0, 12)
        # stacked levels, nan passthrough
        w2, v2 = observables.g_principal(numpy.array([self.g, numpy.full((3, 3), numpy.nan)]))
        self.assertAlmostEqual(abs(w2[0] - self.gd).max(), 0, 12)
        self.assertTrue(numpy.isnan(w2[1]).all() and numpy.isnan(v2[1]).all())

    def test_hyperfine_principal_is_the_svd_invariant(self):
        s = numpy.array([25.0, 4.8, 4.1])
        U = rotation([0, 1, 0], 0.4)
        V = rotation([1, 0, 1], -1.1)
        a = U @ numpy.diag(s) @ V.T
        w, nuc, spin = observables.hyperfine_principal(a)
        self.assertAlmostEqual(abs(w - s).max(), 0, 12)
        for k in range(3):
            self.assertAlmostEqual(abs(nuc[:, k] @ U[:, k]), 1, 12)
            self.assertAlmostEqual(abs(spin[:, k] @ V[:, k]), 1, 12)
        # invariant under a laboratory rotation and a change of doublet basis
        R = rotation([2, -1, 0.5], 2.0)
        w2 = observables.hyperfine_principal(self.Q @ a @ R.T)[0]
        self.assertAlmostEqual(abs(w2 - s).max(), 0, 12)
        # the eigenvalues of the symmetric part are not: they move with R
        e1 = numpy.sort(abs(numpy.linalg.eigvalsh(0.5 * (a + a.T))))
        e2 = numpy.sort(abs(numpy.linalg.eigvalsh(0.5 * (a @ R.T + R @ a.T))))
        self.assertGreater(abs(e1 - e2).max(), 1.0)

    def test_hyperfine_along_g(self):
        # A sharing the g axes: the couplings along g are the diagonal of A in the g frame
        ad = numpy.array([25.0, 4.8, 4.1])
        a = self.Q @ numpy.diag(ad) @ self.Q.T
        k, axes = observables.hyperfine_along_g(a[None], self.g)
        self.assertAlmostEqual(abs(k[0] - ad).max(), 0, 12)
        # A with its axes 10 degrees off g's in the XY plane: the diagonal element under-reads
        # the coupling |A e_X| = sqrt(A_XX^2 + A_YX^2), and the column norm gets it exactly
        P = rotation(self.Q[:, 2], numpy.radians(10))
        a2 = P @ a @ P.T
        k2, _ = observables.hyperfine_along_g(a2[None], self.g)
        a2g = self.Q.T @ a2 @ self.Q
        self.assertAlmostEqual(abs(k2[0] - numpy.linalg.norm(a2g, axis=0)).max(), 0, 12)
        self.assertLess(abs(a2g[0, 0]), k2[0, 0])
        self.assertAlmostEqual(k2[0, 0], 25.0 * numpy.sqrt(numpy.cos(numpy.radians(10))**2
                                                           + (4.8 / 25.0)**2 * numpy.sin(numpy.radians(10))**2), 12)
        # stacked levels, one g per level, natm nuclei
        A = numpy.array([[a, a2], [a2, a]])
        G = numpy.array([self.g, self.g])
        k3, _ = observables.hyperfine_along_g(A, G)
        self.assertEqual(k3.shape, (2, 2, 3))
        self.assertAlmostEqual(abs(k3[0, 0] - k[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(k3[1, 0] - k2[0]).max(), 0, 12)


if __name__ == '__main__':
    unittest.main()
