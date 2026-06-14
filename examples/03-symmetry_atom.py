#!/usr/bin/env python
'''
Symmetry-adapted spinor SCF for an atom (spherical symmetry).

SymmSpinorSCF block-diagonalizes the Fock matrix by spinor irrep.  With
symmetry='sph' the irreps are labelled by the (j, m_j) spinor labels.
'''
from pyscf import gto
from socutils.scf import spinor_hf

mol = gto.M(atom='Ne 0 0 0', basis='ccpvdz', verbose=4)

mf = spinor_hf.SymmSpinorSCF(mol, symmetry='sph').x2camf()
e = mf.kernel()
print('E(Ne, sph symmetry) = %.10f' % e)
print('irreps:', sorted(set(mf.irrep_ao.keys())))   # 's1/2,1/2', 'p3/2,-1/2', ...
