#!/usr/bin/env python
'''
Density fitting (resolution of the identity) for the two-electron integrals.

Any spinor SCF can use density fitting through the .density_fit() shortcut,
exactly as in PySCF.  The shortcuts chain, so it composes with .x2camf().
'''
from pyscf import gto
from socutils.scf import spinor_hf

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

# default auxiliary basis
e = spinor_hf.SCF(mol).x2camf().density_fit().kernel()
print('E(DF X2CAMF) = %.10f' % e)

# explicit auxiliary basis
e2 = spinor_hf.SCF(mol).x2camf().density_fit(auxbasis='ccpvdz-jkfit').kernel()
print('E(DF X2CAMF, ccpvdz-jkfit) = %.10f' % e2)
