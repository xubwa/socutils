#!/usr/bin/env python

'''
Quad precision nonrelativistic HF (qdhf.RHF) and the nonrelativistic limit
of quad precision DHF.

The relativistic correction E_DHF(c) - E_HF vanishes as A/c^2 when the speed
of light c goes to infinity.  c^2 (E_DHF(c) - E_HF) converges to the leading
relativistic coefficient A.  In double precision this difference is lost in
rounding errors for large c; in quadruple precision it is resolved.

The speed of light can be given as a string, which is parsed in quad
precision.
'''

from pyscf import gto, lib, scf
from socutils import qdhf
from socutils.qdhf.qarray import qmp

mol = gto.M(atom='He', basis='cc-pvdz', verbose=0)

e_hf = qdhf.RHF(mol).kernel()
e_hf_double = scf.RHF(mol).run(conv_tol=1e-12).e_tot
print('E(HF, quad)   = %s' % qmp.nstr(e_hf, 34))
print('E(HF, double) = %.16f' % e_hf_double)
print()
print('       c   c^2 (E_DHF - E_HF) quad         c^2 (E_DHF - E_HF) double')
for c in ['137.035999084', '1e3', '1e4', '1e5', '1e6']:
    mf = qdhf.DHF(mol, light_speed=c)
    mf.init_guess = 'double'
    e_dhf = mf.kernel()

    # double precision DHF with the same speed of light
    c0 = lib.param.LIGHT_SPEED
    lib.param.LIGHT_SPEED = float(c)
    e_double = scf.DHF(mol).run(conv_tol=1e-12).e_tot
    lib.param.LIGHT_SPEED = c0

    c2 = qmp.mpf(c)**2
    print('%8s  %-30s  %.10e' % (c, qmp.nstr((e_dhf - e_hf) * c2, 25),
                                 (e_double - e_hf_double) * float(c)**2))
