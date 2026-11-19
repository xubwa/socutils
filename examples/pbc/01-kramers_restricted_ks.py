#!/usr/bin/env python
'''
Gamma-point Kramers-restricted Kohn-Sham of a crystal.

spinor_ks.periodic_ks is the Kohn-Sham counterpart of spinor_scf.periodic: the
same RSDF object gives J (and the exact exchange of a hybrid), the
exchange-correlation term is integrated on Becke grids around the atoms and
their periodic images.
'''
import numpy
from pyscf.pbc import gto
from socutils.scf import spinor_hf
from socutils.pbc import spinor_scf, spinor_ks
from socutils.pbc.rsdf import RSDF
from socutils.cd import accd

cell = gto.Cell()
cell.a = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * 4.083 / 2
cell.atom = 'Li 0 0 0; H 2.0415 2.0415 2.0415'
cell.basis = 'cc-pvdz'
cell.exp_to_discard = 0.1
cell.nucmod = 'G'
cell.precision = 1e-10
cell.verbose = 4
cell.build()

rsdf = RSDF(cell, accd.accd_basis(cell, verbose=0), nuc_zeta=spinor_scf.nuclear_zeta(cell)).build()

mf = spinor_ks.periodic_ks(spinor_hf.KRHF(cell), rsdf, xc='PBE').x2camf()
mf.grids_level = 4
e_pbe = mf.kernel()

# a hybrid functional reuses the same vectors for the exact exchange
mf = spinor_ks.periodic_ks(spinor_hf.KRHF(cell), rsdf, xc='PBE0').x2camf()
mf.grids_level = 4
e_pbe0 = mf.kernel()

print('E(PBE)  = %.10f' % e_pbe)
print('E(PBE0) = %.10f' % e_pbe0)
