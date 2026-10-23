#!/usr/bin/env python
'''
Non-collinear and multi-collinear two-component DFT.

The two-component Kohn-Sham classes (GHF style socutils.dft.gks.GKS, spinor
style socutils.dft.dft.SpinorDFT) take the XC functional of the density and
the magnetization vector m in one of three schemes, set by `collinear`:

  'col'   collinear: e(rho_alpha, rho_beta) of the spin-diagonal blocks only.
          Not invariant under spin rotation; its TDDFT kernel has no spin-flip
          channel (the triplets of a closed shell split 1 + 2).
  'ncol'  locally collinear (Kubler; GGA after Scalmani-Frisch): s = |m| and
          the gradients / tau projected on the local axis m/|m|.
  'mcol'  multi-collinear (Pu, Li, Liu et al., JCTC 19, 2006 (2023)): average
          of the collinear functional over spin-projection directions.  Its
          kernel is well defined at m = 0 -- the choice for TDDFT.

For LDA, ncol and mcol are the same functional (evaluated in closed form).
For GGA/MGGA, mcol samples `spin_samples` Lebedev directions on the sphere
(half of them are evaluated, in the C kernel lib/libsocdft.so when built --
`make`).  mcfun is not needed; `mf._numint.mcol_backend = 'mcfun'` switches to
it as a reference.
'''

from pyscf import gto
from socutils.dft import gks
from socutils.dft import dft as spinor_dft

au2ev = 27.211386245988

# Open shell: the three schemes on a triplet
mol = gto.M(atom='O 0 0 0; O 0 0 1.21', basis='cc-pvdz', spin=2, verbose=0)
for col in ('col', 'ncol', 'mcol'):
    mf = gks.GKS(mol, xc='pbe')
    mf.collinear = col
    mf.spin_samples = 50
    print('O2  PBE %-4s  E = %.10f' % (col, mf.kernel()))

# Spinor reference, multi-collinear (add .x2camf() for spin-orbit coupling)
mf = spinor_dft.SpinorDFT(mol, xc='pbe')
mf._numint.collinear = 'mcol'
mf._numint.spin_samples = 50
print('O2  PBE mcol, spinor basis  E = %.10f' % mf.kernel())

# Closed shell TDA: only the non-collinear kernels give degenerate triplets
mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='6-31g', verbose=0)
for col in ('col', 'mcol'):
    mf = gks.GKS(mol, xc='pbe')
    mf.collinear = col
    mf.spin_samples = 50
    mf.kernel()
    td = mf.TDA()
    td.nstates = 6
    print('H2O PBE %-4s TDA (eV):' % col, (td.kernel()[0] * au2ev).round(4))
