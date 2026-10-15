#!/usr/bin/env python

'''
Asymptotics of the He...He interaction in sf-X2C-1e at a small speed of
light (default c = 3 a.u., Z/c = 0.67), where the complete-basis residual of
'atom1e' (orbital relaxation under the error operator, ~ c^-8) becomes large
enough to see.  dE(R) and the local exponent for 'atom1e' and '1e'.

Usage:  python 28-atom1e_small_c.py [c] [basis label]
'''

import sys
import time
import numpy
from pyscf import gto
from socutils import qdhf
from socutils.qdhf.qarray import qmp

c = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
which = sys.argv[2] if len(sys.argv) > 2 else 'ETB sp'
distances = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
t0 = time.time()


def etb(beta, ranges):
    etbs = []
    for l, (a0, a1) in enumerate(ranges):
        n = int(numpy.floor(numpy.log(a1 / a0) / numpy.log(beta))) + 1
        etbs.append((l, n, a0, beta))
    return {'He': gto.etbs(etbs)}


bases = {'aug-cc-pVQZ': 'aug-cc-pvqz',
         'ETB sp': etb(2.0, [(0.02, 1e8), (0.05, 1e3)]),
         'ETB spd': etb(2.0, [(0.02, 1e8), (0.05, 1e3), (0.1, 100.)])}
basis = bases[which]


def energy(geom, approx):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = qdhf.SFX2C1E_RHF(mol, light_speed=repr(c))
    mf.approx = approx
    mf.direct = True
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.lindep = 1e-28
    mf.max_cycle = 100
    mf.init_guess = 'double'
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING not converged', approx, geom, flush=True)
    return e


res = {}
for approx in ('atom1e', '1e'):
    e0 = 2 * energy('He 0 0 0', approx)
    res[approx] = [energy('He 0 0 0; He 0 0 %g' % r, approx) - e0 for r in distances]
    print('  %s done (%.0f s)' % (approx, time.time() - t0), flush=True)
print('\nHe...He sf-X2C-1e, c = %g a.u., %s;  dE(R) / Hartree, local exponent n in ()' % (c, which))
print('   R/A  %-30s %-30s' % ('atom1e', '1e'))
for i, r in enumerate(distances):
    line = '%6g  ' % r
    for approx in ('atom1e', '1e'):
        de = res[approx][i]
        n = ''
        if i > 0 and de * res[approx][i-1] > 0:
            n = '(%6.3f)' % -float(qmp.log(de / res[approx][i-1]) / qmp.log(qmp.mpf(r) / distances[i-1]))
        line += '% .8e %-12s  ' % (float(de), n)
    print(line, flush=True)
