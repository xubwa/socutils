#!/usr/bin/env python
'''
Symmetry-adapted spinor SCF for a linear molecule.

With symmetry='linear' the irreps are labelled by m_j ('1/2', '-3/2', ...).

IMPORTANT: the molecule MUST be placed along the z axis.  The irrep assignment
assumes the molecular axis is z and does NOT check or reorient the geometry, so
a molecule on another axis is silently mis-classified.
'''
from pyscf import gto
from socutils.scf import spinor_hf

# HF aligned along z, as required for symmetry='linear'
mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

mf = spinor_hf.SymmSpinorSCF(mol, symmetry='linear').x2camf()
e = mf.kernel()
print('E(HF, linear symmetry) = %.10f' % e)
print('irreps:', sorted(set(mf.irrep_ao.keys())))   # '1/2', '-1/2', '3/2', ...
