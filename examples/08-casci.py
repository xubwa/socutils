#!/usr/bin/env python
'''
CASCI on a two-component (spinor) reference.

The active space is counted in spinor (spin-orbital) orbitals: ncas active
spinors holding nelecas electrons (nelecas <= ncas).  The default CI solver is
pyscf.fci.fci_dhf_slow.FCISolver.  kernel() returns (e_tot, e_cas, ci, ...).
'''
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import zcasci

mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

mf = spinor_hf.SCF(mol).x2camf()
mf.kernel()

# 6 electrons in 8 active spinor orbitals
mc = zcasci.CASCI(mf, 8, 6)
mc.kernel()
print('E(CASCI) = %.10f' % mc.e_tot)
print('E(CAS)   = %.10f' % mc.e_cas)
