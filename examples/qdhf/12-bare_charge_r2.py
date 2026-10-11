#!/usr/bin/env python

'''
Spurious R^-2 tail of a dipolar molecule (HF) in sf-X2C-1e, separated from
all physical interactions:

A far bare nucleus C (charge Z_C, no basis functions; mol.charge keeps the
electron count of HF) enters the one-electron Hamiltonian either
  (a) through the X2C-1e decoupling (as a nucleus of the molecule), or
  (b) untransformed, h = h_X2C[HF] + V_C   (consistent with untransformed
      electron-electron interaction of a real far atom).
dE(R) = E_a - E_b removes the physical induction (identical in a and b) and
leaves the spurious term.  R^2 dE -> its R^-2 coefficient.

approx='atom1e': X does not depend on V_C; the tail is the first-order
  picture-change error of V_C and should vanish in the basis-set limit.
approx='1e': X depends on V_C; an additional 1/c^4 term from the
  nonlinearity of the decoupling (grad V_A . grad V_C) is expected to
  survive the basis-set limit.

Usage:  python 12-bare_charge_r2.py
'''

import sys
import time
import numpy
from pyscf import gto, lib
from socutils.qdhf import x2c, dhf
from socutils.qdhf import integrals as qints
from socutils.qdhf.qarray import qmp

bohr = 1 / 0.52917721092
geom_a = 'F 0 0 0; H 0 0 0.917'


def etb(beta, ranges):
    out = {}
    for el, rg in ranges.items():
        etbs = []
        for l, (a0, a1) in enumerate(rg):
            n = int(numpy.floor(numpy.log(a1/a0) / numpy.log(beta))) + 1
            etbs.append((l, n, a0, beta))
        out[el] = gto.etbs(etbs)
    return out


def scf(mol, h, light_speed):
    qi = qints.QuadIntegrals(mol)
    ham = dhf.DirectRHFHamiltonian(mol, qi, h, qi.ovlp_x(), qmp.mpf(qints.energy_nuc(mol)))
    mf = dhf.RHF(mol)
    mf.build_hamiltonian = lambda: ham
    mf.conv_tol = 1e-30
    mf.conv_tol_grad = 1e-15
    mf.lindep = 1e-28
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: SCF not converged', flush=True)
    return e


def spurious(basis, r, approx, light_speed=None, zc=2):
    bas = dict(basis) if isinstance(basis, dict) else {'F': basis, 'H': basis}
    mol = gto.M(atom=geom_a + '; He 0 0 %g' % r, basis=bas, charge=zc, verbose=0)
    mol_a = gto.M(atom=geom_a, basis=bas, verbose=0)
    ha = x2c.get_hcore(mol_a, light_speed, approx)
    h_a = x2c.get_hcore(mol, light_speed, approx)
    v_c = qints.QuadIntegrals(mol).nuc_x(atoms=[2])
    return scf(mol, h_a, light_speed) - scf(mol, ha + v_c, light_speed)


bases = [('cc-pVDZ', 'cc-pvdz'), ('cc-pVTZ', 'cc-pvtz'), ('aug-cc-pVTZ', 'aug-cc-pvtz'),
         ('cc-pVQZ', 'cc-pvqz'), ('aug-cc-pVQZ', 'aug-cc-pvqz'),
         ('ETB sp b=2.0 wide', etb(2.0, {'F': [(0.005, 1e7), (0.004, 1e4)],
                                         'H': [(0.002, 1e5), (0.005, 1e3)]})),
         ('ETB spd b=2.0 wide', etb(2.0, {'F': [(0.005, 1e7), (0.004, 1e4), (0.01, 50.)],
                                          'H': [(0.002, 1e5), (0.005, 1e3), (0.01, 10.)]}))]
approxes = sys.argv[1].split(',') if len(sys.argv) > 1 else ['atom1e', '1e']
c0 = qmp.mpf(lib.param.LIGHT_SPEED)
print('HF + bare charge Z=2 on the +z axis (H side): R^2 dE_spurious (a.u.), c^2 and c^4 scaled')
for approx in approxes:
    for label, basis in bases:
        t0 = time.time()
        out = []
        for r in (160, 320):
            de = spurious(basis, r, approx)
            out.append(de * (qmp.mpf(r) * bohr)**2)
        print('%-8s %-20s R=160: % .6e  R=320: % .6e   c^2*: % .6e  c^4*: % .5f  (%.0f s)' % (
            approx, label, float(out[0]), float(out[1]), float(out[1] * c0**2),
            float(out[1] * c0**4), time.time() - t0), flush=True)
