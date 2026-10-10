#!/usr/bin/env python

'''
Long-range behavior of the range-separated X2C-1e (approx='rs1e', Q. Sun,
pyscf branch range-separated-x2c1e) compared with X2C-1e and NR-HF.

rs1e: V = V_sr + V_lr per nucleus, V_lr = -Z erf(omega r)/r (Gaussian
countercharge).  X is built from V_sr; V_lr is not picture-change
transformed, like the electron-electron interaction.  The far nucleus then
enters the block of the other fragment only through V_sr ~ erfc(omega R)/R,
so both the picture-change error Delta = PC(V_B) - V_B and the polarization
of X by V_B (see 06-x2c_decomposition.py) vanish.

Usage:  python 08-rs1e_long_range.py [basis]
'''

import sys
from pyscf import gto
from socutils import qdhf
from socutils.qdhf.qarray import qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
bohr = 1 / 0.52917721092
distances = [10, 20, 40, 80, 160, 320]
systems = [
    ('He...He', 'He 0 0 0'),
    ('He2...He', 'He 0 0 -1.5; He 0 0 1.5'),
    ('HF...He', 'F 0 0 0; H 0 0 0.917'),
]
methods = [('HF', None, None), ('sfX2C1e', '1e', None),
           ('rs1e(w=1)', 'rs1e', 1.0), ('rs1e(w=0.5)', 'rs1e', 0.5), ('rs1e(w=2)', 'rs1e', 2.0)]

def energy(geom, approx, omega):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    if approx is None:
        mf = qdhf.RHF(mol)
    else:
        mf = qdhf.SFX2C1E_RHF(mol)
        mf.approx = approx
        if omega is not None:
            mf.omega = omega
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-17
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: not converged', geom, approx, omega)
    return e

for label, geom_a in systems:
    print('\n%s   dE(R) in Hartree, local exponent n in ()' % label)
    print('%-12s' % 'method' + ''.join('     R=%-4g A          ' % r for r in distances))
    for name, approx, omega in methods:
        e0 = energy(geom_a, approx, omega) + energy('He', approx, omega)
        line = '%-12s' % name
        prev = None
        for r in distances:
            de = energy(geom_a + '; He 0 0 %g' % r, approx, omega) - e0
            if prev is not None and de * prev[1] > 0:
                n = '(%5.2f)' % -float(qmp.log(de / prev[1]) / qmp.log(qmp.mpf(r) / prev[0]))
            else:
                n = '       '
            line += ' % .3e %s' % (float(de), n)
            prev = (r, de)
        print(line, flush=True)
