#!/usr/bin/env python

'''
Asymptotics of the SAP-screened X2C-1e (sap1e) and of rs1e at very large separations,
compared with NR-HF.  sap1e: X from V + V_scr (superposition of atomic potentials), V_scr subtracted untransformed.

The physical tails (R^-6 for HF...He, R^-8 for He2...He, exponential for
He...He) fall below 1e-26 Hartree beyond ~1000 A, so any residual spurious
R^-2..R^-4 term of rs1e would dominate there.  In double precision the total
energies have ~1e-15 noise and the interaction energy is invisible beyond
~10-20 A.

Usage:  python 09-rs1e_asymptotics.py [basis]
'''

import sys
from pyscf import gto
from socutils import qdhf
from socutils.qdhf.qarray import qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
distances = [20, 40, 80, 160, 320, 640, 1280, 2560]
systems = [
    ('HF...He', 'F 0 0 0; H 0 0 0.917'),
    ('He2...He', 'He 0 0 -1.5; He 0 0 1.5'),
    ('He...He', 'He 0 0 0'),
]
methods = ['sap1e', 'rs1e', 'HF']


def energy(method, geom):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    if method == 'HF':
        mf = qdhf.RHF(mol)
    elif method == 'DHF':
        mf = qdhf.DHF(mol)
    else:
        mf = qdhf.SFX2C1E_RHF(mol)
        mf.approx = method if method in ('rs1e', 'sap1e') else '1e'
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-17
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: %s not converged at %s' % (method, geom), flush=True)
    return e


for label, geom_a in systems:
    res = {}
    for m in methods:
        e0 = energy(m, geom_a) + energy(m, 'He')
        res[m] = [energy(m, geom_a + '; He 0 0 %g' % r) - e0 for r in distances]
    print('\n%s   dE(R) / Hartree   (local exponent n)' % label)
    print('   R/A  ' + ''.join('%-30s' % m for m in methods) + 'sap1e/HF')
    for i, r in enumerate(distances):
        line = '%6g  ' % r
        for m in methods:
            de = res[m][i]
            n = ''
            if i > 0 and de * res[m][i-1] > 0:
                n = '(%6.3f)' % -float(qmp.log(de / res[m][i-1]) / qmp.log(qmp.mpf(r) / distances[i-1]))
            line += '% .6e %-9s     ' % (float(de), n)
        ratio = res['sap1e'][i] / res['HF'][i] if res['HF'][i] != 0 else 0
        line += '%.8f' % float(ratio)
        print(line, flush=True)
