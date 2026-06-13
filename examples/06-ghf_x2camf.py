#!/usr/bin/env python
'''
GHF (spin-orbital) driver with an X2CAMF spin-orbit Hamiltonian.

ghf.GHF is the spin-orbital analogue of spinor_hf.SCF: same .x2camf()/.x2cmp()
shortcuts, but the SCF runs in a spin-orbital rather than a j-adapted spinor
basis.  The spinor and GHF results agree to numerical precision.
'''
from pyscf import gto
from socutils.scf import ghf, spinor_hf

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

e_ghf = ghf.GHF(mol).x2camf().kernel()
e_spinor = spinor_hf.SCF(mol).x2camf().kernel()
print('E(GHF X2CAMF)    = %.10f' % e_ghf)
print('E(spinor X2CAMF) = %.10f' % e_spinor)
print('difference       = %.2e' % abs(e_ghf - e_spinor))
