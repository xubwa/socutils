#!/usr/bin/env python
'''
Double-group symmetry for a linear molecule.

The molecule has to be in its symmetry frame (main axis along z);
socutils.symm.to_symm_frame moves it there and symmetrizes the coordinates.
The irreps are named by m_j ('1/2', '-3/2', ...; with g/u for D∞h).
All vectors, tensors and m_j labels refer to the symmetry frame.
'''
from pyscf import gto
from socutils.symm import to_symm_frame
from socutils.scf import double_group_symm

mol = gto.M(atom='H 0 0 0; F 0.5 0.3 0.7', basis='ccpvdz', symmetry=True, verbose=4)
mol = to_symm_frame(mol)

mf = double_group_symm.SymAdaptedRHF(mol).x2camf()
e = mf.kernel()
print('E(HF, C∞v double group) = %.10f' % e)
print('irreps:', mf.irrep_name)

# an open-shell Omega component: HF+ X 2Pi_3/2 with M_J = -3/2 (hole in m_j = +3/2)
mol_c = mol.copy()
mol_c.charge, mol_c.spin = 1, 1
mol_c.build()
mf = double_group_symm.SymAdaptedUHF(mol_c).x2camf()
mf.irrep_nelec = {'1/2': 4, '-1/2': 4, '3/2': 0, '-3/2': 1}
mf.kernel()
