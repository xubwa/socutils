#!/usr/bin/env python

'''
Is the spurious R^-4 tail of He...He in sf-X2C-1e a finite-basis artifact?

R^4 dE(R) is evaluated at R = 160 and 320 A for a sequence of basis sets
approaching completeness.  If it converges to a nonzero value, the R^-4 term
(which scales as 1/c^4) is intrinsic to the one-electron decoupling: the
exact FW Hamiltonian of the bare nuclear potential contains terms nonlinear
in V, e.g. (grad V)^2 / c^4, and (grad V_B)^2 ~ (Z_B/R^2)^2 is not screened
by the electrons of B, which do not enter the decoupling.

Usage:  python 11-he2_r4_basis.py [approx list, e.g. 1e,atom1e] [number of basis sets]
'''

import time
import numpy
from pyscf import gto
from socutils import qdhf
from socutils.qdhf.qarray import qmp

bohr = 1 / 0.52917721092


def etb(beta, ranges):
    etbs = []
    for l, (a0, a1) in enumerate(ranges):
        n = int(numpy.floor(numpy.log(a1/a0) / numpy.log(beta))) + 1
        etbs.append((l, n, a0, beta))
    return {'He': gto.etbs(etbs)}


def energy(geom, basis, approx):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = qdhf.SFX2C1E_RHF(mol)
    mf.approx = approx
    mf.direct = True
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.lindep = 1e-28
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: not converged', geom, flush=True)
    return e, mol.nao


bases = [('cc-pVDZ', 'cc-pvdz'), ('cc-pVTZ', 'cc-pvtz'), ('cc-pVQZ', 'cc-pvqz'),
         ('aug-cc-pVQZ', 'aug-cc-pvqz'), ('cc-pV5Z', 'cc-pv5z'),
         ('ETB sp b=2.0', etb(2.0, [(0.02, 1e5), (0.05, 100.)])),
         ('ETB sp b=1.6', etb(1.6, [(0.01, 1e6), (0.02, 300.)])),
         ('ETB spd b=2.0', etb(2.0, [(0.02, 1e5), (0.05, 100.), (0.1, 20.)])),
         ('ETB spd b=1.6', etb(1.6, [(0.01, 1e6), (0.02, 300.), (0.05, 50.)]))]

print('He...He  R^4 dE (a.u., R in bohr); c^4 R^4 dE in ()')
print('%-16s %4s   %-28s %-28s %s' % ('basis', 'nao', 'R=160 A', 'R=320 A', 'approx'))
c4 = qmp.mpf(137.03599967994)**4
import sys
approxes = sys.argv[1].split(',') if len(sys.argv) > 1 else ['1e', 'atom1e']
nbas = int(sys.argv[2]) if len(sys.argv) > 2 else len(bases)
for approx in approxes:
    for label, basis in bases[:nbas]:
        t0 = time.time()
        e1, nao = energy('He 0 0 0', basis, approx)
        row = []
        for r in (160, 320):
            e2, _ = energy('He 0 0 0; He 0 0 %g' % r, basis, approx)
            row.append((e2 - 2 * e1) * (qmp.mpf(r) * bohr)**4)
        print('%-16s %4d   % .8e (% .5f)   % .8e (% .5f)   %s  %.0fs' % (
            label, nao, float(row[0]), float(row[0] * c4), float(row[1]), float(row[1] * c4),
            approx, time.time() - t0), flush=True)
