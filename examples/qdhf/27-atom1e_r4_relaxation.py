#!/usr/bin/env python

'''
Origin of the R^-4 tail of sf-X2C-1e(atom1e) for He...He.  The decoupling
(atomic X, molecular R) does not depend on the far nuclear potential, so h is
linear in it and the only second-order term is orbital relaxation under the
first-order error operator.  Its leading part is the uniform-field (dipole)
piece F dz, dz = PC(z) - z, F = Z_B / R^2:

    E2 = e2 F^2 per atom,  e2 = (1/2) d^2E/dlambda^2 for h + lambda dz
    R^4 E_spur = 2 e2 Z^2   (both atoms)

e2 from finite fields (lambda = +-1e-3, quad precision), to be compared with
the supermolecular R^4 dE of 11-he2_r4_basis.out.

Usage:  python 27-atom1e_r4_relaxation.py [basis ...]
'''

import sys
from pyscf import gto
from socutils.qdhf import x2c, dhf
from socutils.qdhf.qarray import hmp

bases = sys.argv[1:] or ['cc-pvdz', 'cc-pvtz', 'cc-pvqz', 'aug-cc-pvqz', 'cc-pv5z']
lam = hmp.mpf('1e-3')


def energy(mol, h):
    ham = dhf.build_nr_hamiltonian(mol, 'RHF', 0)
    ham.hcore = h
    mf = dhf.RHF(mol)
    mf.build_hamiltonian = lambda: ham
    mf.init_guess = 'double'
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.max_cycle = 100
    mf.verbose = 0
    return mf.kernel()


print('He atom1e: R^4 E_spur predicted from the polarization by dz = PC(z) - z')
for basis in bases:
    mol = gto.M(atom='He 0 0 0', basis=basis, verbose=0)
    xmol, contr = x2c.uncontract_mol(mol)
    d = x2c.get_xr(xmol, approx='atom1e')
    qi = d['qi']
    z = qi.dip_x(2)
    zy = qi.dip_y(2)
    wz = zy[0::3, 0::3] + zy[1::3, 1::3] + zy[2::3, 2::3]
    dz = contr.T @ (x2c.picture_change(d, z, wz) - z) @ contr
    h0 = contr.T @ x2c.hcore_from_xr(d) @ contr
    e0 = energy(mol, h0)
    ep = energy(mol, h0 + dz * lam)
    em = energy(mol, h0 + dz * (-lam))
    e2 = (ep + em - 2 * e0) / (2 * lam**2)
    print('%-12s e2 % .6e   predicted R^4 E (Z_B = 2, both atoms) % .6e' % (basis, float(e2), float(2 * e2 * 4)),
          flush=True)
