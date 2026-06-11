#!/usr/bin/env python

'''
RASCI with SelectedCI.

A RAS-type CI is just a CI in a restricted determinant list, so it can be
simulated by combining SelectedCI with gen_ras_occslst, which generates
all determinants with at most max_hole holes in RAS1 and at most max_elec
electrons in RAS3 (RAS2 is unrestricted).

The RAS blocks are contiguous subsets of the active orbitals: with
nras = (n1, n2, n3), RAS1 covers active spinor orbitals 0..n1-1, RAS2 the
next n2 and RAS3 the last n3.  Order the active orbitals accordingly
(e.g. with mc.sort_mo) before the calculation.
'''

import math
from pyscf import gto, scf
from socutils.mcscf import zcasci
from socutils.fci import zfci

mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g', verbose=0)
mf = scf.X2C(mol).run()

#
# 10 active spinor orbitals, 6 electrons, partitioned as
#   RAS1: orbitals 0-3 (occupied in the reference)
#   RAS2: orbitals 4-7
#   RAS3: orbitals 8-9
#
ncas, nelecas = 10, 6
nras = (4, 4, 2)

for max_hole, max_elec in [(1, 1), (2, 2), (4, 2)]:
    occslst = zfci.gen_ras_occslst(nras, nelecas,
                                   max_hole=max_hole, max_elec=max_elec)
    mc = zcasci.CASCI(mf, ncas, nelecas)
    mc.fcisolver = zfci.SelectedCI(mol, occslst=occslst)
    mc.kernel()
    print('RASCI(h%d, e%d)  %4d dets  E = %.12f'
          % (max_hole, max_elec, len(occslst), mc.e_tot))

#
# max_hole = n1 and max_elec = n3 removes all restrictions, so the last
# entry above is identical to CASCI(12, 6):
#
mc = zcasci.CASCI(mf, ncas, nelecas)
mc.fcisolver = zfci.FCISolver(mol)
mc.kernel()
print('CASCI reference %4d dets  E = %.12f'
      % (math.comb(ncas, nelecas), mc.e_tot))
