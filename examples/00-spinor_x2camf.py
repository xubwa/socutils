#!/usr/bin/env python
'''
Basic spinor Hartree-Fock with an X2CAMF spin-orbit Hamiltonian.

spinor_hf.SCF(mol).x2camf() is the spinor analogue of PySCF's
scf.RHF(mol).x2c(): it attaches an atomic-mean-field X2C spin-orbit
Hamiltonian (default flavor 'x2camf', Gaunt + Breit on) through with_x2c.

Requires the optional x2camf package for the SOC integrals.
'''
from pyscf import gto
from socutils.scf import spinor_hf

mol = gto.M(
    atom='H 0 0 0; F 0 0 0.917',
    basis='ccpvdz',
    verbose=4,
)

mf = spinor_hf.SCF(mol).x2camf()
e = mf.kernel()
print('E(X2CAMF spinor HF) = %.10f' % e)
