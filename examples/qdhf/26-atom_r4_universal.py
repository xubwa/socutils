#!/usr/bin/env python

'''
Second-order (R^-4) spurious interaction of two closed-shell atoms in
sf-X2C-1e ('1e').  In the complete basis, the decoupling X responds to the
bare nuclear field F = Z_B / R^2 of the far atom; to order F^2 c^-4

    h^(2) = F^2 c^-4 [ -1/8 + (1/8) p_F^2 / p^2 ]

so that, for atoms with an isotropic momentum distribution (<p_F^2/p^2> = 1/3),

    E_spur = -(N_A Z_B^2 + N_B Z_A^2) / (12 c^4 R^4).

He...He: c^4 R^4 E = -4/3;  He...Ne: -(2*100 + 10*4)/12 = -20;
Ne...Ne: -(10*100*2)/12 = -166.67.  Printed: c^4 R^4 dE for approx='1e'
and 'atom1e' (no X response).

Usage:  python 26-atom_r4_universal.py [basis ...]
'''

import sys
from pyscf import gto, lib
from socutils import qdhf
from socutils.qdhf.qarray import qmp

bases = sys.argv[1:] or ['cc-pvdz', 'cc-pvtz']
bohr = 1 / 0.52917721092
c4 = qmp.mpf(lib.param.LIGHT_SPEED)**4
pairs = [('He', 'He', 2, 2, 2, 2), ('He', 'Ne', 2, 2, 10, 10), ('Ne', 'Ne', 10, 10, 10, 10)]


def energy(geom, basis, approx):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = qdhf.SFX2C1E_RHF(mol)
    mf.approx = approx
    mf.init_guess = 'double'
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.max_cycle = 100
    mf.verbose = 0
    return mf.kernel()


print('c^4 R^4 dE (a.u.), sf-X2C-1e;  prediction -(N_A Z_B^2 + N_B Z_A^2)/12')
for a, b, za, na, zb, nb in pairs:
    pred = -(na * zb**2 + nb * za**2) / 12.
    for basis in bases:
        for approx in ('1e', 'atom1e'):
            e0 = energy('%s 0 0 0' % a, basis, approx) + energy('%s 0 0 0' % b, basis, approx)
            vals = []
            for r in (160, 320):
                de = energy('%s 0 0 0; %s 0 0 %g' % (a, b, r), basis, approx) - e0
                vals.append(float(de * (qmp.mpf(r) * bohr)**4 * c4))
            print('%s...%s %-10s %-7s  R=160: % .5f   R=320: % .5f   prediction % .4f' % (
                a, b, basis, approx, vals[0], vals[1], pred), flush=True)
