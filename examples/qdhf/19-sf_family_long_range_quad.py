#!/usr/bin/env python

'''
Spin-free family in quad precision: the parent spin-free DHF and the
spin-free amfX2C (atomic 2e Fock matrices only in the atomic diagonal
blocks; Knecht, Repisky, Jensen, Saue, JCP 157, 114106 (2022)).  NR-HF,
sf-X2C-1e and sfmp (spin-free X2CMP) are in 17-sfmp_long_range_quad.

Usage:  python 19-sf_family_long_range_quad.py [basis] [system ...]
'''

import sys
import time
from pyscf import gto
from socutils.qdhf import sfmp
from socutils.qdhf.qarray import qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
distances = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
all_systems = {'He2...He': 'He 0 0 -1.5; He 0 0 1.5',
               'He...He': 'He 0 0 0',
               'HF...He': 'F 0 0 0; H 0 0 0.917'}
labels = sys.argv[2:] if len(sys.argv) > 2 else ['He2...He', 'He...He', 'HF...He']
methods = ['sfDHF', 'sfamfX2C']
t0 = time.time()


def energy(method, geom):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = sfmp.SFDHF(mol) if method == 'sfDHF' else sfmp.SFAMF_RHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.max_cycle = 100
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: %s not converged at %s' % (method, geom), flush=True)
    return e


for label in labels:
    geom_a = all_systems[label]
    res = {}
    for m in methods:
        e0 = energy(m, geom_a) + energy(m, 'He 0 0 0')
        res[m] = []
        for r in distances:
            res[m].append(energy(m, geom_a + '; He 0 0 %g' % r) - e0)
            print('  %s %-9s R=%5g  dE % .6e  (%.0f s)' % (label, m, r, float(res[m][-1]),
                                                      time.time() - t0), flush=True)
    print('\n%s (%s)  dE(R) / Hartree, local exponent n in ()' % (label, basis))
    print('   R/A  ' + ''.join('%-28s' % m for m in methods))
    for i, r in enumerate(distances):
        line = '%6g  ' % r
        for m in methods:
            de = res[m][i]
            n = ''
            if i > 0 and de * res[m][i-1] > 0:
                n = '(%6.3f)' % -float(qmp.log(de / res[m][i-1]) / qmp.log(qmp.mpf(r) / distances[i-1]))
            line += '% .6e %-9s    ' % (float(de), n)
        print(line, flush=True)
