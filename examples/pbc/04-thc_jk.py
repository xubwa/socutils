#!/usr/bin/env python
'''
J and K without the three-index tensor: tensor hypercontraction (THCJK).

RSDF stores naux x nao^2 / 2 numbers, which is what limits the cell size.
socutils.pbc.thc.THCJK builds J and K from much smaller pieces:

  * the basis is split at the exponent alpha_c into compact and smooth functions;
  * pairs of smooth functions are interpolated on N_mu points of the uniform
    grid (THC / ISDF), nmu_factor points per smooth function;
  * pairs with a compact function are fitted on the auxiliary functions of
    their two atoms (within local_rcut) plus a far-field correction;
  * J is integrated on the uniform grid, with atomic grids for the compact part.

It replaces RSDF in the mean field through socutils.pbc.split_direct.periodic.
THC is meant for large cells: in a cell much smaller than the extent of the
basis functions the smooth pairs are few and nearly linearly dependent.

The build sizes all its blocks to cell.max_memory (MB; per MPI rank in a
parallel run): more memory means fewer, larger blocks and a faster build.
'''
import numpy
from pyscf.pbc import gto
from pyscf.pbc.tools import pbc as pbctools
from socutils.scf import spinor_hf
from socutils.pbc import spinor_scf, split_direct
from socutils.pbc.rsdf import RSDF
from socutils.pbc.thc import THCJK
from socutils.cd import accd

# LiH, 2x2x2 supercell of the primitive cell (16 atoms)
prim = gto.Cell()
prim.a = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * 4.083 / 2
prim.atom = 'Li 0 0 0; H 2.0415 2.0415 2.0415'
prim.basis = 'cc-pvdz'
prim.exp_to_discard = 0.1
prim.nucmod = 'G'
prim.precision = 1e-10
prim.verbose = 0
prim.build()
cell = pbctools.super_cell(prim, [2, 2, 2])
cell.max_memory = 16000          # MB: the budget of the THC build
cell.verbose = 4

# The pairs with a compact function are fitted locally, to first order in the fitting error: use a large
# auxiliary basis (here acCD of the orbital basis; an even-tempered aug_etb(beta=2) is not enough: 2.5e-3 Eh).
auxbasis = accd.accd_basis(prim, verbose=0)
nuc_zeta = spinor_scf.nuclear_zeta(cell)

jk = THCJK(cell, auxbasis, alpha_c=4., nmu_factor=16., nuc_zeta=nuc_zeta, local_rcut=6.)   # local_rcut in Angstrom
jk.verbose = True                # prints the steps of the build and their memory
jk.build()

mf = split_direct.periodic(spinor_hf.KRHF(cell), jk).x2camf()
e_thc = mf.kernel()

# the same calculation with RSDF
rsdf = RSDF(cell, auxbasis, nuc_zeta=nuc_zeta).build()
mf_ref = spinor_scf.periodic(spinor_hf.KRHF(cell), rsdf).x2camf()
mf_ref.verbose = 0
e_rsdf = mf_ref.kernel()

print('E(THC)  = %.8f' % e_thc)
print('E(RSDF) = %.8f   difference %.1e Eh' % (e_rsdf, e_thc - e_rsdf))
print('THC: %d interpolation points for %d smooth functions; RSDF: %d vectors of %d AO pairs'
      % (len(jk.th.pts), len(jk.th.smooth), rsdf.naux_kept, cell.nao * (cell.nao + 1) // 2))
