#!/usr/bin/env python
'''
Zero-field splitting of O2 X 3Sigma_g- from spin-orbit coupling:
X2CAMF Hamiltonian, state-averaged spinor CASSCF(8e, 12 spinors) over the
six lowest states (X 3Sigma_g-, a 1Delta_g, b 1Sigma_g+), then multi-state
UC-NEVPT2.  The X2CAMF Hamiltonian has no electronic spin-spin term, so the
splitting is the spin-orbit part of the experimental 2*lambda (about 3.96
cm^-1 in total, of which spin-orbit is roughly 2.4 cm^-1).

Usage:  python 01-o2_zfs.py [basis]

The socutils CASSCF needs a density-fitted Kramers-restricted reference
(KRHF + zquatev); zmr then uses exact (non-DF) integrals in those orbitals.
socutils writes an FCIDUMP file to the working directory.
'''
import sys
import time

import numpy
from pyscf import gto
from socutils.fci import zfci
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf

from socutils.zmr.active import symmetry
from socutils.zmr.interface import socutils_x2c
from socutils.zmr.uc.nevpt2 import UCNEVPT2

HARTREE2WAVENUMBER = 219474.6313632
basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'

mol = gto.M(atom='O 0 0 -0.6035; O 0 0 0.6035', basis=basis, spin=2, verbose=0)

# orbitals: SA-CASSCF(8e, 12 spinors: 2p sigma, pi, pi*, sigma*), 6 states
mf_df = spinor_hf.KRHF(mol).x2camf().density_fit()
mf_df.verbose = 0
mf_df.kernel()
mc = zmcscf.CASSCF(mf_df, 12, 8)
mc.fcisolver = zfci.FCISolver(mol)
mc = mc.state_average_([1 / 6.] * 6)
mc.verbose = 0
mc.kernel()

# exact integrals in the CASSCF orbitals: 1s frozen, 2s correlated core
mf = spinor_hf.SCF(mol).x2camf()
mf.verbose = 0
eris = socutils_x2c.from_scf(mf, ncore=4, nact=12, nfrozen=4, mo_coeff=mc.mo_coeff, mol=mol)
# J_z-adapted active spinors: block diagonal sector Hamiltonians, same energies
eris, sym = symmetry.adapt(eris)
if sym is not None:
    print('active-space symmetry: %d generators, largest discarded integral %.1e Eh'
          % (sym.ngen, sym.error))

numpy.set_printoptions(linewidth=200, precision=2, suppress=True)
print('basis %s, %d spinors' % (basis, mc.mo_coeff.shape[1]))
for e0 in ('average', 'state'):
    t0 = time.time()
    nev = UCNEVPT2(eris, 8, nstate=6, e0=e0,
                   method={'ccav': 'laplace', 'cavv': 'laplace'}, nquad=40)
    e = nev.kernel()
    e_cas = numpy.linalg.eigvalsh(nev.hmodel)
    print('e0=%-7s CASCI  (cm-1)' % e0, (e_cas - e_cas[0]) * HARTREE2WAVENUMBER)
    print('e0=%-7s NEVPT2 (cm-1)' % e0, (e - e[0]) * HARTREE2WAVENUMBER,
          ' E2(X) = %.6f  [%.0f s]' % (e[0] - e_cas[0], time.time() - t0))
