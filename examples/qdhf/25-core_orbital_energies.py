#!/usr/bin/env python

'''
Core orbital energies with sf-X2C-1e, rs1e (omega = 1, 0.5), sap1e and sfmp
(quad precision, decontracted basis, closed-shell RHF), compared with the
parent spin-free DHF where it fits in memory.

Usage:  python 25-core_orbital_energies.py [geom] [basis] [ncore] [ref]
        ref = sfdhf | sfmp
'''

import sys
import numpy
from pyscf import gto
from socutils import qdhf
from socutils.qdhf import sfmp

geom = sys.argv[1] if len(sys.argv) > 1 else 'H 0 0 0; F 0 0 0.917'
basis = sys.argv[2] if len(sys.argv) > 2 else 'unc-cc-pvdz'
ncore = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ref = sys.argv[4] if len(sys.argv) > 4 else 'sfdhf'
mol = gto.M(atom=geom, basis=basis, verbose=0)


def occupied_energies(mf):
    e = numpy.array([float(x) for x in mf.mo_energy.to_mpmath()])
    occ = numpy.asarray(mf.mo_occ)
    return e[occ > 0]


def run(name):
    if name == 'sfdhf':
        mf = sfmp.SFDHF(mol)
    elif name == 'sfmp':
        mf = sfmp.SFMP_RHF(mol)
    else:
        approx, omega = name.split(':') if ':' in name else (name, None)
        mf = qdhf.SFX2C1E_RHF(mol)
        mf.approx = approx
        if omega is not None:
            mf.omega = float(omega)
        mf.init_guess = 'double'
    mf.verbose = 0
    mf.conv_tol = 1e-24
    mf.conv_tol_grad = 1e-12
    mf.kernel()
    return occupied_energies(mf)


methods = ['1e', 'rs1e:1.0', 'rs1e:0.5', 'sap1e', 'sfmp']
if ref == 'sfdhf':
    methods = methods + ['sfdhf']
eps = {m: run(m) for m in methods}
print('%s (%s): occupied orbital energies / Hartree; differences from %s in mHartree' % (geom, basis, ref))
print('orb  %-14s ' % ref + ''.join('%-12s' % m for m in methods if m != ref))
for i in range(len(eps[ref])):
    tag = 'core' if i < ncore else 'val '
    print('%2d %s % .8f  ' % (i, tag, eps[ref][i]) +
          ''.join('% 10.4f  ' % ((eps[m][i] - eps[ref][i]) * 1e3) for m in methods if m != ref), flush=True)
