#!/usr/bin/env python
'''
Four-component Dirac-Hartree-Fock reference methods.

socutils provides specialized four-component DHF drivers built on PySCF's dhf
module.  Shown here is the spin-free (scalar-relativistic) SpinFreeDHF, which
keeps the four-component structure but removes spin-orbit coupling -- useful
for separating scalar-relativistic from spin-orbit effects.

See also scf.linear_dhf.SymmDHF (symmetry-adapted 4c DHF) and
scf.frac_dhf.FRAC_RDHF (fractional open-shell occupation, needs zquatev).
'''
from pyscf import gto
from socutils.scf import sfdhf

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

mf = sfdhf.SpinFreeDHF(mol)
e = mf.kernel()
print('E(spin-free 4c DHF) = %.10f' % e)
