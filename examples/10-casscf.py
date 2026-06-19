#!/usr/bin/env python
'''
CASSCF on a two-component (spinor) reference.

zmcscf.CASSCF additionally optimizes the orbitals: kernel() drives the
super-CI orbital optimizer directly, alternating CI and orbital steps.

The orbital optimizer uses a Kramers-paired eigensolver and therefore requires
the bundled zquatev solver to be compiled (run `make` at the repo root);
kernel() raises a clear error if it is missing.  It also builds its integrals
from a density-fitted reference, so the mean field must be density-fitted
(.density_fit()).
'''
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

# the CASSCF orbital optimizer needs a density-fitted reference
mf = spinor_hf.SCF(mol).x2camf().density_fit()
mf.kernel()

mc = zmcscf.CASSCF(mf, 8, 6)   # 6 electrons in 8 active spinor orbitals
mc.kernel()
print('E(CASSCF) = %.10f' % mc.e_tot)
