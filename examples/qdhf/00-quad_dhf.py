#!/usr/bin/env python

'''
Four-component Dirac-Hartree-Fock (Dirac-Coulomb) in quadruple precision.

qdhf.DHF(mol)                       quad precision integrals + quad precision SCF
qdhf.DHF(mol, integrals='libcint')  libcint (double) integrals, quad precision SCF

The result e_tot is an mpmath number with the binary128 precision
(34 significant digits).
'''

from pyscf import gto, scf
from socutils import qdhf

mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=4)

# reference: the double precision DHF of PySCF
e_double = scf.DHF(mol).run(conv_tol=1e-12).e_tot

mf = qdhf.DHF(mol)
mf.kernel()
print('pyscf.scf.DHF  (double)            %.16f' % e_double)
print('qdhf.DHF       (quad integrals)    %s' % mf.e_tot_str)

mf_libcint = qdhf.DHF(mol, integrals='libcint')
mf_libcint.kernel()
print('qdhf.DHF       (libcint integrals) %s' % mf_libcint.e_tot_str)
print('integral rounding error of libcint in the energy: %.3g'
      % float(mf_libcint.e_tot - mf.e_tot))

# orbital energies (QArray -> mpmath numbers)
n2c = mol.nao_2c()
e_mo = mf.mo_energy.to_mpmath()
for i in range(n2c, n2c + mol.nelectron):
    print('PES orbital %2d  %s' % (i - n2c, qdhf.qarray.qmp.nstr(e_mo[i], 34)))
