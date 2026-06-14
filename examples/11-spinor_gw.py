#!/usr/bin/env python
'''
Relativistic (spinor) G0W0 quasiparticle energies.

``socutils.gw.SpinorGWAC`` is the spinor analogue of PySCF's restricted
``gw.gw_ac``.  It takes any spinor mean-field reference (``SpinorSCF`` or an
X2C/X2CAMF subclass), builds the complex density-fitting tensor in the spinor
MO basis, and computes the diagonal G0W0 self-energy on the imaginary axis
followed by a Pade analytic continuation to the real axis.

This example does two things on H2 (a light, effectively non-relativistic
system) so the result can be checked against the ordinary RHF-G0W0 numbers:

1. Compare the spinor-G0W0 valence quasiparticle energies with restricted
   RHF-G0W0.  Each RHF level shows up as a doubly degenerate Kramers pair.

2. Demonstrate the complex-conjugate symmetry of the implementation: rotate
   the occupied Kramers (HOMO) pair by an arbitrary 2x2 *complex* unitary.
   This makes the reference orbitals genuinely complex without changing the
   physics, and the quasiparticle energies of the rotated pair stay put.
'''

import copy
import numpy as np
from pyscf import gto, scf, gw

from socutils.scf import spinor_hf
from socutils.gw import SpinorGWAC

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='def2-svp', verbose=0)
nocc_r = mol.nelectron // 2      # RHF occupied orbitals
nocc = mol.nelectron             # occupied spinors

# ---------------------------------------------------------------------------
# 1. restricted RHF-G0W0 reference
# ---------------------------------------------------------------------------
rhf = scf.RHF(mol).density_fit()
rhf.kernel()

gw_r = gw.GW(rhf, freq_int='ac')
gw_r.ac = 'pade'
gw_r.orbs = list(range(nocc_r - 1, nocc_r + 1))
gw_r.kernel()
print('RHF   E      = %.10f' % rhf.e_tot)
print('RHF-GW  HOMO = %.6f   LUMO = %.6f'
      % (gw_r.mo_energy[nocc_r - 1], gw_r.mo_energy[nocc_r]))

# ---------------------------------------------------------------------------
# 2. spinor (relativistic-ready) G0W0
# ---------------------------------------------------------------------------
mf = spinor_hf.SpinorSCF(mol).density_fit()
mf.kernel()

mygw = SpinorGWAC(mf)
mygw.ac = 'pade'
mygw.kernel(orbs=range(nocc - 2, nocc + 2))
print('spinor E     = %.10f' % mf.e_tot)
print('spinor-GW HOMO pair =', np.round(mygw.mo_energy[nocc - 2:nocc], 6))
print('spinor-GW LUMO pair =', np.round(mygw.mo_energy[nocc:nocc + 2], 6))

# ---------------------------------------------------------------------------
# 3. complex-conjugate symmetry check
#    rotate the occupied Kramers pair by a 2x2 complex unitary
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
q, r = np.linalg.qr(a)
U = q @ np.diag(np.exp(1j * np.angle(np.diag(r))))   # random 2x2 unitary

pair = [nocc - 2, nocc - 1]
C_rot = mf.mo_coeff.copy()
C_rot[:, pair] = mf.mo_coeff[:, pair] @ U
print('\nimag(mo_coeff) norm after complex rotation = %.4f'
      % np.linalg.norm(C_rot.imag))

mf_rot = copy.copy(mf)
mf_rot.mo_coeff = C_rot

gw_rot = SpinorGWAC(mf_rot)
gw_rot.ac = 'pade'
gw_rot.kernel(orbs=range(nocc - 2, nocc + 2))
print('spinor-GW HOMO pair (complex ref) =',
      np.round(gw_rot.mo_energy[nocc - 2:nocc], 6))
print('max |dE| of rotated pair = %.2e'
      % np.max(np.abs(gw_rot.mo_energy[pair] - mygw.mo_energy[pair])))
