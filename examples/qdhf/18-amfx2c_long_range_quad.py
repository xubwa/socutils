#!/usr/bin/env python

'''
Spin-orbit coupled family in quad precision (decontracted basis):

    DHF      parent 4c Dirac-Coulomb Hartree-Fock
    NR       nonrelativistic GHF
    1e       X2C-1e
    amfx2c   amfX2C (Knecht, Repisky, Jensen, Saue, JCP 157, 114106 (2022)):
             atomic 2e Fock matrices only in the atomic diagonal blocks
    x2cmp    x2cmp = HF eamfX2C: molecular 2e Fock of the superposed atomic
             4c density matrices

Usage:  python 18-amfx2c_long_range_quad.py [basis]
'''

import sys
import time
from pyscf import gto
from socutils.qdhf import x2c4c, x2c, dhf
from socutils.qdhf.qarray import qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
distances = [20, 40, 80, 160, 320, 640, 1280, 2560]
systems = [('He2...He', 'He 0 0 -1.5; He 0 0 1.5'),
           ('He...He', 'He 0 0 0')]
methods = ['DHF', None, '1e', 'amfx2c', 'x2cmp']
names = {'DHF': 'DHF', None: 'NR', '1e': 'X2C-1e', 'amfx2c': 'amfX2C', 'x2cmp': 'x2cmp/eamfX2C'}
cache = {}
t0 = time.time()


def energy(approx, geom):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    if approx == 'DHF':
        xmol, _ = x2c.uncontract_mol(mol)
        mf = dhf.DHF(xmol)
    else:
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
            print('  %s %-13s R=%5g  dE % .6e  (%.0f s)' % (label, names[m], r, float(res[m][-1]),
                                                       time.time() - t0), flush=True)
    print('\n%s (%s)  dE(R) / Hartree, local exponent n in ()' % (label, basis))
    print('   R/A  ' + ''.join('%-26s' % names[m] for m in methods))
    for i, r in enumerate(distances):
        line = '%6g  ' % r
        for m in methods:
            de = res[m][i]
            n = ''
            if i > 0 and de * res[m][i-1] > 0:
                n = '(%6.3f)' % -float(qmp.log(de / res[m][i-1]) / qmp.log(qmp.mpf(r) / distances[i-1]))
            line += '% .5e %-9s  ' % (float(de), n)
        print(line, flush=True)
