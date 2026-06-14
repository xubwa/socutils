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


if __name__ == '__main__':
    print('Tests for relativistic spinor G0W0-AC')
    unittest.main()
