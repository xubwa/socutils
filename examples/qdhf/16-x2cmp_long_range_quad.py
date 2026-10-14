#!/usr/bin/env python

'''
Long-range behaviour of the spin-orbit coupled two-component Hamiltonians
decoupled from the 4c Dirac-Coulomb Hamiltonian, in quad precision
(socutils.qdhf.x2c4c):

    NR     nonrelativistic GHF
    1e     X2C-1e (with spin-orbit coupling)
    x2cmp  x2cmp model potential (Dirac-Coulomb), atomic DHF densities

All in the decontracted basis.  The quad implementation reproduces the
production socutils x2cmp (double precision, same basis) to 1e-12 Hartree.

Usage:  python 16-x2cmp_long_range_quad.py [basis]
'''

import sys
import time
from pyscf import gto
from socutils.qdhf import x2c4c
from socutils.qdhf.qarray import qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
distances = [20, 40, 80, 160, 320, 640, 1280, 2560]
systems = [('He2...He', 'He 0 0 -1.5; He 0 0 1.5'),
           ('He...He', 'He 0 0 0')]
methods = [None, '1e', 'x2cmp']
names = {None: 'NR', '1e': 'X2C-1e(SO)', 'x2cmp': 'x2cmp'}
cache = {}
t0 = time.time()


def energy(approx, geom):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = x2c4c.GHF2c(mol, approx)
    mf.cache = cache
    mf.verbose = 0
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.max_cycle = 100
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: %s not converged at %s' % (names[approx], geom), flush=True)
    return e


for label, geom_a in systems:
    res = {}
    for m in methods:
        e0 = energy(m, geom_a) + energy(m, 'He 0 0 0')
        res[m] = []
        for r in distances:
            res[m].append(energy(m, geom_a + '; He 0 0 %g' % r) - e0)
            print('  %s %-10s R=%5g  dE % .6e  (%.0f s)' % (label, names[m], r, float(res[m][-1]),
                                                       time.time() - t0), flush=True)
    print('\n%s (%s)  dE(R) / Hartree, local exponent n in ()' % (label, basis))
    print('   R/A  ' + ''.join('%-28s' % names[m] for m in methods))
    for i, r in enumerate(distances):
        line = '%6g  ' % r
        for m in methods:
            de = res[m][i]
            n = ''
            if i > 0 and de * res[m][i-1] > 0:
                n = '(%6.3f)' % -float(qmp.log(de / res[m][i-1]) / qmp.log(qmp.mpf(r) / distances[i-1]))
            line += '% .6e %-9s    ' % (float(de), n)
        print(line, flush=True)
