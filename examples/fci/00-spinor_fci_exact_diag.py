#!/usr/bin/env python

'''
Exact full CI for spinor (relativistic) Hamiltonians by explicit
construction and diagonalization of the Hamiltonian matrix.

zfci.FCISolver is a drop-in replacement for the Dice-based SHCI interface
(socutils.hci.shci) and for pyscf.fci.fci_dhf_slow (Davidson) when the
active space is small enough for exact diagonalization.  All roots are
obtained from a single diagonalization, with no Davidson convergence
issues for the (nearly) degenerate roots caused by Kramers degeneracy.
'''

import numpy
from pyscf import gto, scf
from socutils.mcscf import zcasci
from socutils.fci import zfci

mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='sto-3g', verbose=0)
mf = scf.X2C(mol).run()
print('E(X2C-HF) = %.12f' % mf.e_tot)

#
# CASCI with an active space of 8 spinor orbitals and 4 electrons.
# Note ncas and ncore count spinor orbitals (= 2x the number of Kramers
# pairs) and nelecas is the total number of electrons.
#
mc = zcasci.CASCI(mf, 8, 4)
mc.fcisolver = zfci.FCISolver(mol)
mc.kernel()
print('E(CASCI)  = %.12f' % mc.e_tot)

#
# Several roots from one diagonalization; mc.e_tot is then the state
# average.  The individual energies are kept in mc.fcisolver.eci.
#
mc = zcasci.CASCI(mf, 8, 4)
mc.fcisolver = zfci.FCISolver(mol)
mc.fcisolver.nroots = 4
mc.kernel()
for i, e in enumerate(mc.fcisolver.eci):
    print('state %d  E = %.12f' % (i, e.real))

#
# The solver can also be used standalone with a given set of integrals,
# following the same conventions as pyscf.fci.fci_dhf_slow: h1e is the
# (norb, norb) complex one-electron Hamiltonian in the spinor MO basis and
# eri the (norb, norb, norb, norb) complex two-electron integrals in
# chemists' notation.
#
h1e, ecore = mc.get_h1eff()
eri = mc.get_h2eff().reshape(8, 8, 8, 8)
e, civec = zfci.kernel(h1e, eri, 8, 4, ecore=ecore)
print('E(standalone kernel) = %.12f' % e.real)
