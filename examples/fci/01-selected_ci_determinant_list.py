#!/usr/bin/env python

'''
CI in an arbitrary list of determinants (selected CI).

zfci.SelectedCI diagonalizes the Hamiltonian in the space spanned by a
user-supplied list of determinants, given through the attribute occslst.

The format of occslst
---------------------
occslst is a (ndet, nelec) nested list (or numpy array):

  * each row is one determinant;
  * a row lists the indices of the occupied spinor orbitals of that
    determinant, counted inside the active space (0-based, from 0 to
    ncas-1; orbital 0 is the first active spinor orbital, not the first
    MO of the molecule);
  * every row has nelec entries, e.g. for a CAS(4e, 8so) active space

        occslst = [[0, 1, 2, 3],   # "HF" determinant |0 1 2 3>
                   [0, 1, 2, 4],   # single excitation 3 -> 4
                   [0, 1, 4, 5],   # double excitation 2,3 -> 4,5
                   ...]

  * the order of the indices within a row does not matter (a determinant
    is always defined with its creation operators in ascending orbital
    order), but the order of the rows does: the CI coefficients in the
    resulting CI vector follow the order of occslst.

This is the same way determinants are specified in a Dice input file
(e.g. the initial determinants of a ZSHCI calculation), so a determinant
list produced by or prepared for Dice can be used directly.
'''

import numpy
from pyscf import gto, scf
from pyscf.fci import cistring
from socutils.scf import spinor_hf
from socutils.mcscf import zcasci
from socutils.fci import zfci

mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g', verbose=0)
mf = spinor_hf.SCF(mol).x2camf().run()

ncas, nelecas = 8, 4

#
# 1. A hand-picked determinant list: the HF determinant plus the
#    paired double excitations into the next Kramers pair.
#
occslst = [[0, 1, 2, 3],
           [0, 1, 4, 5],
           [2, 3, 4, 5]]
mc = zcasci.CASCI(mf, ncas, nelecas)
mc.fcisolver = zfci.SelectedCI(mol, occslst=occslst)
mc.kernel()
print('E(3 determinants)   = %.12f' % mc.e_tot)
# CI coefficients follow the order of occslst
for occ, c in zip(occslst, mc.ci):
    print('  %s  % .6f %+.6fj' % (occ, c.real, c.imag))

#
# 2. All single and double excitations from the HF determinant
#    (a CISD calculation inside the active space).
#
hf_det = [0, 1, 2, 3]
occslst = [hf_det]
for occ in cistring.gen_occslst(range(ncas), nelecas):
    ndiff = len(set(hf_det) - set(occ))
    if 1 <= ndiff <= 2:
        occslst.append(list(occ))
mc = zcasci.CASCI(mf, ncas, nelecas)
mc.fcisolver = zfci.SelectedCI(mol, occslst=occslst)
mc.kernel()
print('E(CAS-CISD, %2d dets) = %.12f' % (len(occslst), mc.e_tot))

#
# 3. The complete determinant space reproduces full CI.
#
occslst = cistring.gen_occslst(range(ncas), nelecas)
mc = zcasci.CASCI(mf, ncas, nelecas)
mc.fcisolver = zfci.SelectedCI(mol, occslst=occslst)
mc.kernel()
print('E(all %d dets)       = %.12f' % (len(occslst), mc.e_tot))

mc = zcasci.CASCI(mf, ncas, nelecas)
mc.fcisolver = zfci.FCISolver(mol)
mc.kernel()
print('E(full CI reference) = %.12f' % mc.e_tot)
