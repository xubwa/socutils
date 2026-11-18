#!/usr/bin/env python
'''
Dependence of rs1e on the range-separation parameter omega: bond-stretching
energy of HX, E(R_ref + 0.6 A) - E(R_ref), and the absolute energy at R_ref,
against sf-X2C-1e (omega -> 0) and against the spin-free X2CMP (sfmp), which
contains the two-electron picture change (quad precision, closed-shell RHF).

Usage:  python 44-rs1e_omega_scan.py [X] [basis] [R_ref]
'''
import sys
from pyscf import gto
from socutils import qdhf

el = sys.argv[1] if len(sys.argv) > 1 else 'Cl'
basis = sys.argv[2] if len(sys.argv) > 2 else 'cc-pvdz'
r0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.275
omegas = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0]


def energy(approx, omega, r):
    mol = gto.M(atom='H 0 0 0; %s 0 0 %.6f' % (el, r), basis=basis, verbose=0)
    mf = qdhf.SFX2C1E_RHF(mol)
    mf.approx = approx
    if omega is not None:
        mf.omega = omega
    mf.init_guess = 'double'
    mf.conv_tol = 1e-20
    mf.conv_tol_grad = 1e-10
    mf.verbose = 0
    e = mf.kernel()
    assert mf.converged
    return e


ref = [energy('1e', None, r0), energy('1e', None, r0 + 0.6)]
print('H%s %s, R_ref = %.3f A.  dE = E(R_ref + 0.6 A) - E(R_ref)' % (el, basis, r0))
print('sf-X2C-1e: dE = %.6f mEh' % (float(ref[1] - ref[0]) * 1e3))
print('%8s %22s %26s' % ('omega', 'dE - dE(1e) / microEh', 'E(R_ref) - E_1e / mEh'))
for w in omegas:
    e = [energy('rs1e', w, r0), energy('rs1e', w, r0 + 0.6)]
    print('%8.3f %22.3f %26.4f' % (w, float((e[1] - e[0]) - (ref[1] - ref[0])) * 1e6,
                                   float(e[0] - ref[0]) * 1e3), flush=True)
