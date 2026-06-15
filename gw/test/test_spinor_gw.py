#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Tests for the relativistic (spinor) G0W0-AC module.

Two things are checked:

1. Reduction to the non-relativistic limit.  A :class:`SpinorSCF` reference
   built without spin-orbit coupling is the spinor image of a closed-shell RHF
   calculation: every spatial orbital becomes a Kramers doublet.  The spinor
   G0W0 quasiparticle energies must reproduce the restricted RHF-G0W0 valence
   energies, doubly degenerate.

2. Complex-conjugate symmetry.  Applying an arbitrary 2x2 *complex* unitary
   rotation to a degenerate Kramers pair makes the reference orbitals (and
   hence the DF tensor Lpq) genuinely complex without changing the physics.
   The quasiparticle energy of the rotated pair must be invariant.  A wrong
   complex conjugate anywhere in the response or self-energy would break this.
'''

import unittest
import copy
import numpy as np

from pyscf import gto, scf, gw

from socutils.scf import spinor_hf
from socutils.gw.spinor_gw_ac import SpinorGWAC


def setUpModule():
    global mol, rhf, mf
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='def2-svp', verbose=0)

    rhf = scf.RHF(mol).density_fit()
    rhf.conv_tol = 1e-12
    rhf.kernel()

    mf = spinor_hf.SpinorSCF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()


def tearDownModule():
    global mol, rhf, mf
    del mol, rhf, mf


def _random_complex_unitary(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    # fix the phase so q is a well-defined unitary
    q = q @ np.diag(np.exp(1j * np.angle(np.diag(r))))
    return q


class KnownValues(unittest.TestCase):
    def test_scf_reduces_to_rhf(self):
        # sanity: the non-relativistic spinor HF reproduces RHF
        self.assertAlmostEqual(mf.e_tot, rhf.e_tot, 9)

    def test_gw_reduces_to_rhf_gw(self):
        nocc_r = mol.nelectron // 2          # RHF occupied count
        nocc = mol.nelectron                 # spinor occupied count

        # restricted RHF-G0W0 reference (valence window)
        gw_r = gw.GW(rhf, freq_int='ac')
        gw_r.ac = 'pade'
        gw_r.linearized = False
        gw_r.orbs = list(range(nocc_r - 1, nocc_r + 1))
        gw_r.kernel()
        homo_r = gw_r.mo_energy[nocc_r - 1]
        lumo_r = gw_r.mo_energy[nocc_r]

        # spinor G0W0
        mygw = SpinorGWAC(mf)
        mygw.ac = 'pade'
        mygw.linearized = False
        mygw.kernel(orbs=range(nocc - 2, nocc + 2))
        homo_s = mygw.mo_energy[nocc - 2:nocc]
        lumo_s = mygw.mo_energy[nocc:nocc + 2]

        # spinor HOMO/LUMO are Kramers doublets degenerate with the RHF values
        self.assertAlmostEqual(homo_s[0], homo_s[1], 6)
        self.assertAlmostEqual(lumo_s[0], lumo_s[1], 6)
        self.assertAlmostEqual(homo_s[0], homo_r, 4)
        self.assertAlmostEqual(lumo_s[0], lumo_r, 4)

    def test_complex_conjugate_symmetry(self):
        nocc = mol.nelectron

        mygw = SpinorGWAC(mf)
        mygw.ac = 'pade'
        mygw.linearized = False
        mygw.kernel(orbs=range(nocc - 2, nocc + 2))
        e_ref = mygw.mo_energy.copy()

        # rotate the occupied Kramers HOMO pair by a complex 2x2 unitary.  The
        # rotated MOs remain degenerate eigenvectors (so mo_energy is
        # unchanged) but are now genuinely complex.
        pair = [nocc - 2, nocc - 1]
        U = _random_complex_unitary(2, seed=42)
        C_rot = mf.mo_coeff.copy()
        C_rot[:, pair] = mf.mo_coeff[:, pair] @ U
        self.assertGreater(np.linalg.norm(C_rot.imag), 1e-3)  # really complex

        mf_rot = copy.copy(mf)
        mf_rot.mo_coeff = C_rot

        gw_rot = SpinorGWAC(mf_rot)
        gw_rot.ac = 'pade'
        gw_rot.linearized = False
        gw_rot.kernel(orbs=range(nocc - 2, nocc + 2))
        e_rot = gw_rot.mo_energy

        # the rotated Kramers pair and the LUMO pair are clean doublets;
        # their quasiparticle energies must be invariant.
        for p in [nocc - 2, nocc - 1, nocc, nocc + 1]:
            self.assertAlmostEqual(e_rot[p], e_ref[p], 6)

    def test_parallel_consistency(self):
        # the threaded frequency loop must give the same answer as serial
        nocc = mol.nelectron
        e1 = SpinorGWAC(mf)
        e1.ac = 'pade'
        e1.nthreads = 1
        e1.kernel(orbs=range(nocc - 2, nocc + 2))

        e4 = SpinorGWAC(mf)
        e4.ac = 'pade'
        e4.nthreads = 4
        e4.kernel(orbs=range(nocc - 2, nocc + 2))

        for p in range(nocc - 2, nocc + 2):
            self.assertAlmostEqual(e1.mo_energy[p], e4.mo_energy[p], 9)


class KnownValuesFrozen(unittest.TestCase):
    '''Frozen core / frozen virtual against restricted RHF-G0W0.'''

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
        cls.rhf = scf.RHF(cls.mol).density_fit()
        cls.rhf.conv_tol = 1e-12
        cls.rhf.kernel()
        cls.mf = spinor_hf.SpinorSCF(cls.mol).density_fit()
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

    def test_frozen_core_matches_rhf(self):
        nocc_r = self.mol.nelectron // 2
        nocc = self.mol.nelectron

        # RHF-G0W0 freezing the 1s spatial orbital
        gw_r = gw.GW(self.rhf, freq_int='ac', frozen=1)
        gw_r.ac = 'pade'
        gw_r.orbs = list(range(nocc_r - 1, nocc_r + 1))
        gw_r.kernel()

        # spinor-G0W0 freezing the 1s Kramers pair (2 lowest spinors)
        mygw = SpinorGWAC(self.mf, frozen=2)
        mygw.ac = 'pade'
        mygw.kernel(orbs=range(nocc - 2, nocc + 2))

        self.assertEqual(mygw.nocc, nocc - 2)
        self.assertAlmostEqual(mygw.mo_energy[nocc - 1], gw_r.mo_energy[nocc_r - 1], 4)
        self.assertAlmostEqual(mygw.mo_energy[nocc], gw_r.mo_energy[nocc_r], 4)

    def test_frozen_virtual_list(self):
        # frozen given as a list freezes exactly those orbitals (core + virtual)
        nocc = self.mol.nelectron
        nmo_full = len(self.mf.mo_energy)
        frozen = [0, 1] + list(range(nmo_full - 4, nmo_full))
        mygw = SpinorGWAC(self.mf, frozen=frozen)
        mygw.ac = 'pade'
        mygw.kernel(orbs=range(nocc - 2, nocc + 2))
        self.assertEqual(mygw.nmo, nmo_full - len(frozen))
        self.assertTrue(np.isfinite(mygw.mo_energy[nocc - 1]))
        # a frozen orbital must be rejected when requested in orbs
        with self.assertRaises(RuntimeError):
            SpinorGWAC(self.mf, frozen=frozen).kernel(orbs=[0])


class KnownValuesSOC(unittest.TestCase):
    '''Genuinely relativistic reference (X2CAMF spin-orbit coupling).

    Requires the bundled x2camf / zquatev C backends to be compiled
    (`make` in the repo root).
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='Ar 0 0 0', basis='cc-pvdz', verbose=0)
        cls.mf = (spinor_hf.SpinorSCF(cls.mol).x2camf()
                  .density_fit(auxbasis='cc-pvdz-jkfit'))
        cls.mf.conv_tol = 1e-11
        cls.mf.kernel()

    def _occ_degenerate_pair(self):
        '''Lowest isolated 2-fold degenerate occupied pair above the frozen
        core (a Kramers doublet, e.g. 3p_1/2 of Ar).'''
        e = self.mf.mo_energy
        nocc = self.mol.nelectron
        for i in range(7, nocc):
            deg = abs(e[i] - e[i - 1]) < 1e-7
            iso_lo = abs(e[i - 1] - e[i - 2]) > 1e-5
            iso_hi = (i + 1 >= len(e)) or abs(e[i] - e[i + 1]) > 1e-5
            if deg and iso_lo and iso_hi:
                return [i - 1, i]
        raise self.skipTest('no isolated occupied Kramers pair found')

    def test_soc_gw_runs(self):
        nocc = self.mol.nelectron
        mygw = SpinorGWAC(self.mf)
        mygw.ac = 'pade'
        mygw.frozen = 5
        mygw.kernel(orbs=range(10, nocc))
        qpe = mygw.mo_energy[10:nocc]
        self.assertTrue(np.all(np.isfinite(qpe)))
        # SOC-split 3p HOMO must lie above the 3s level
        self.assertGreater(qpe.max(), -0.7)

    def test_complex_conjugate_symmetry_soc(self):
        '''Complex-conjugate symmetry on a relativistic reference.

        On a SOC Kramers doublet the diagonal self-energy has the form
        a*I + (off-diagonal); in an arbitrary basis the *individual* diagonal
        energies are basis-dependent, but the trace over the complete
        degenerate manifold is exactly invariant under a unitary rotation
        within it.  Rotating by a 2x2 complex unitary (which makes the MOs
        genuinely complex) must leave that trace unchanged.
        '''
        import copy
        nocc = self.mol.nelectron
        pair = self._occ_degenerate_pair()

        def run(mo_coeff):
            m = copy.copy(self.mf)
            m.mo_coeff = mo_coeff
            g = SpinorGWAC(m)
            g.ac = 'pade'
            g.frozen = 5
            g.kernel(orbs=range(10, nocc))
            return g.mo_energy

        e_ref = run(self.mf.mo_coeff)
        U = _random_complex_unitary(2, seed=7)
        C_rot = self.mf.mo_coeff.copy()
        C_rot[:, pair] = self.mf.mo_coeff[:, pair] @ U
        self.assertGreater(np.linalg.norm(C_rot.imag), 1e-3)
        e_rot = run(C_rot)

        self.assertTrue(np.all(np.isfinite(e_rot)))
        # the trace is preserved to the analytic-continuation / Newton noise
        # floor (~1e-7); that is 7 significant figures of invariance under a
        # genuinely complex rotation.
        self.assertAlmostEqual(e_rot[pair].sum(), e_ref[pair].sum(), 6)


if __name__ == '__main__':
    print('Tests for relativistic spinor G0W0-AC')
    unittest.main()
