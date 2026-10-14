#!/usr/bin/env python

'''
Does rs1e (or sap1e) change relative energies computed with sf-X2C-1e?
Bond-stretching curves of HX: E(R) - E(R_ref) for each Hamiltonian and the
deviation from sf-X2C-1e (quad precision, contracted basis, closed-shell RHF).

Usage:  python 24-rs1e_relative_energies.py [X] [basis] [R_ref]
'''

import sys
import time
from pyscf import gto
from socutils import qdhf
from socutils.qdhf.qarray import qmp

el = sys.argv[1] if len(sys.argv) > 1 else 'Cl'
basis = sys.argv[2] if len(sys.argv) > 2 else 'cc-pvdz'
r0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1.275
shifts = [-0.1, 0.0, 0.1, 0.3, 0.6]
methods = [('1e', None), ('rs1e', 1.0), ('rs1e', 0.5), ('sap1e', None)]
t0 = time.time()


def energy(approx, omega, r):
    mol = gto.M(atom='H 0 0 0; %s 0 0 %.6f' % (el, r), basis=basis, verbose=0)
    mf = qdhf.SFX2C1E_RHF(mol)
    mf.approx = approx
    if omega is not None:
        mf.omega = omega
    mf.init_guess = 'double'
    mf.conv_tol = 1e-24
    mf.conv_tol_grad = 1e-12
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING not converged', approx, omega, r, flush=True)
    return e


res = {}
for m in methods:
    res[m] = [energy(m[0], m[1], r0 + d) for d in shifts]
    print('  %s done (%.0f s)' % (str(m), time.time() - t0), flush=True)
iref = shifts.index(0.0)
print('\nH%s (%s), E(R) - E(R_ref), R_ref = %.3f A, in mHartree; deviation from sf-X2C-1e in microHartree'
      % (el, basis, r0))
print('  dR/A  ' + '%-14s' % 'sf-X2C-1e' + ''.join('%-24s' % ('%s w=%s' % m if m[1] else m[0])
                                                   for m in methods[1:]))
for i, d in enumerate(shifts):
    rel = {m: (res[m][i] - res[m][iref]) for m in methods}
    line = '%+6.2f  % .6f    ' % (d, float(rel[methods[0]]) * 1e3)
    for m in methods[1:]:
        line += '% .6f (% .3f)   ' % (float(rel[m]) * 1e3, float(rel[m] - rel[methods[0]]) * 1e6)
    print(line, flush=True)
print('absolute E(R_ref) - E_1e(R_ref) / mHartree: ' +
      '  '.join('%s %.3f' % (m, float(res[m][iref] - res[methods[0]][iref]) * 1e3) for m in methods[1:]))
