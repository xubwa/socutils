#!/usr/bin/env python
'''
The x2cmp (molecular picture-change) flavor, and toggling the two-electron
spin-orbit terms.

.x2cmp() attaches the molecular picture-change flavor; .x2camf() the atomic
mean-field flavor.  Both accept with_gaunt / with_breit / with_pcc / with_aoc
to control which two-electron relativistic corrections enter the mean field.
'''
from pyscf import gto
from socutils.scf import spinor_hf

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

# molecular picture-change flavor (Gaunt + Breit on by default)
e_mp = spinor_hf.SCF(mol).x2cmp().kernel()
print('E(x2cmp, Gaunt+Breit)   = %.10f' % e_mp)

# Dirac-Coulomb only: no Gaunt, no Breit
e_dc = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False).kernel()
print('E(x2camf, Coulomb only) = %.10f' % e_dc)
