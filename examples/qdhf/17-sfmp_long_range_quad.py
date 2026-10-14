#!/usr/bin/env python

'''
Long-range behaviour of the spin-free x2cmp (sfmp) Hamiltonian in quad
precision (socutils.qdhf.sfmp), compared with NR-HF and sf-X2C-1e.

The quad sfmp reproduces the production socutils sfmp (double precision,
same contracted basis) to 1e-13 Hartree in the total energy
(HF...He at 10 A: -102.96549059972534 vs -102.9654905997254).

Usage:  python 17-sfmp_long_range_quad.py [basis]
'''

import sys
import time
from pyscf import gto
from socutils import qdhf
from socutils.qdhf import sfmp
from socutils.qdhf.qarray import qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
distances = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
systems = [('HF...He', 'F 0 0 0; H 0 0 0.917'),
           ('He2...He', 'He 0 0 -1.5; He 0 0 1.5')]
methods = ['HF', 'sfX2C1e', 'sfmp']
t0 = time.time()


def energy(method, geom):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    if method == 'HF':
        mf = qdhf.RHF(mol)
    elif method == 'sfX2C1e':
        mf = qdhf.SFX2C1E_RHF(mol)
        mf.init_guess = 'double'
    else:
        mf = sfmp.SFMP_RHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.max_cycle = 100
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: %s not converged at %s' % (method, geom), flush=True)
    return e


for label, geom_a in systems:
    res = {}
    for m in methods:
        e0 = energy(m, geom_a) + energy(m, 'He 0 0 0')
        res[m] = []
        for r in distances:
            res[m].append(energy(m, geom_a + '; He 0 0 %g' % r) - e0)
            print('  %s %-8s R=%5g  dE % .6e  (%.0f s)' % (label, m, r, float(res[m][-1]),
                                                     time.time() - t0), flush=True)
    print('\n%s (%s)  dE(R) / Hartree, local exponent n in ()' % (label, basis))
    print('   R/A  ' + ''.join('%-28s' % m for m in methods) + 'sfmp/HF')
    for i, r in enumerate(distances):
        line = '%6g  ' % r
        for m in methods:
            de = res[m][i]
            n = ''
            if i > 0 and de * res[m][i-1] > 0:
                n = '(%6.3f)' % -float(qmp.log(de / res[m][i-1]) / qmp.log(qmp.mpf(r) / distances[i-1]))
            line += '% .6e %-9s    ' % (float(de), n)
        line += '%.8f' % float(res['sfmp'][i] / res['HF'][i])
        print(line, flush=True)
