#!/usr/bin/env python
'''
Double-group symmetry for an atom.

An atom is treated with the D∞h double group: the Fock matrix is blocked by
m_j and parity, and the irreps are named by m_j ('1/2g', '-3/2u', ...).
irrep_nelec fixes the number of electrons of named irreps.
'''
from pyscf import gto
from socutils.scf import double_group_symm

mol = gto.M(atom='Ne 0 0 0', basis='ccpvdz', symmetry=True, verbose=4)

mf = double_group_symm.SymAdaptedRHF(mol).x2camf()
e = mf.kernel()
print('E(Ne, D∞h double group) = %.10f' % e)
print('irreps:', mf.irrep_name)
