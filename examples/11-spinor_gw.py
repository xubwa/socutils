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

# ---------------------------------------------------------------------------
# 4. genuinely relativistic reference (X2CAMF spin-orbit coupling)
#    Requires the bundled x2camf C backend (build it with `make` in the repo
#    root, see x2camf_c/README.md).
# ---------------------------------------------------------------------------
mol_h = gto.M(atom='Ar 0 0 0', basis='cc-pvdz', verbose=0)
mf_soc = (spinor_hf.SpinorSCF(mol_h).x2camf()
          .density_fit(auxbasis='cc-pvdz-jkfit'))
mf_soc.kernel()
no = mol_h.nelectron
gw_soc = SpinorGWAC(mf_soc)
gw_soc.ac = 'pade'
gw_soc.frozen = 5
gw_soc.kernel(orbs=range(10, no + 4))
print('\nX2CAMF spinor-HF E = %.8f' % mf_soc.e_tot)
print('SOC spinor-GW HOMO (3p_3/2) =',
      np.round(gw_soc.mo_energy[no - 4:no], 5))
print('SOC spinor-GW LUMO          =',
      np.round(gw_soc.mo_energy[no:no + 4], 5))

# complex-conjugate symmetry on the SOC reference.  On a Kramers doublet the
# diagonal self-energy is a*I + (off-diagonal), so the individual diagonal QP
# energies are basis dependent, but the trace over the complete degenerate
# manifold is invariant under a within-manifold complex rotation.
e_soc = mf_soc.mo_energy
soc_pair = next([i - 1, i] for i in range(7, no)
                if abs(e_soc[i] - e_soc[i - 1]) < 1e-7
                and abs(e_soc[i - 1] - e_soc[i - 2]) > 1e-5
                and abs(e_soc[i] - e_soc[i + 1]) > 1e-5)
C_soc = mf_soc.mo_coeff.copy()
C_soc[:, soc_pair] = mf_soc.mo_coeff[:, soc_pair] @ U
mf_soc_rot = copy.copy(mf_soc)
mf_soc_rot.mo_coeff = C_soc
gw_soc_rot = SpinorGWAC(mf_soc_rot)
gw_soc_rot.ac = 'pade'
gw_soc_rot.frozen = 5
gw_soc_rot.kernel(orbs=range(10, no))
tr0 = gw_soc.mo_energy[soc_pair].sum()
tr1 = gw_soc_rot.mo_energy[soc_pair].sum()
print('SOC doublet %s trace QP: orig %.8f  rot %.8f  |dE| %.2e'
      % (soc_pair, tr0, tr1, abs(tr0 - tr1)))

