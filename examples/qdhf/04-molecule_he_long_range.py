#!/usr/bin/env python

'''
Molecule + far He atom along several directions:

    dE(R) = E(M...He, R) - E(M) - E(He)

The spurious long-range tail of the one-electron X2C decoupling is the
interaction of the bare nuclear charge Z of the far atom with the lowest
non-vanishing multipole of the picture-change difference of the molecular
charge distribution (X2C-transformed minus untransformed):

    dipolar molecule (HF):  Z dmu cos(theta) / R^2     (sign flips with direction)
    He2 (D_inf_h):           Z dTheta P2(cos theta) / R^3
    CH4 (T_d):               Z dOmega_xyz xyz/R^7 -> R^-4, zero along the S4 (x) axis,
                             opposite signs towards an H and towards a face;
                             plus the isotropic field-squared R^-4 term of atoms

Physical HF/DHF tails: R^-6 (dipole-induced dipole), R^-10 (CH4 octupole).

Usage:  python 04-molecule_he_long_range.py {hf,ch4} [basis] [methods]
        methods: comma separated subset of HF,DHF,sfX2C1e,sfX2C1e(atom)
'''

import sys
import numpy
from pyscf import gto
from socutils import qdhf
from socutils.qdhf.qarray import qmp

molecules = {
    # F at the origin, H on +z
    'hf': ('F 0 0 0; H 0 0 0.917',
           {'+z (H side)': (0, 0, 1), '-z (F side)': (0, 0, -1)}),
    # C at the origin, H at (a,a,a), ...
    'ch4': ('C 0 0 0; H {a} {a} {a}; H {a} -{a} -{a}; H -{a} {a} -{a}; H -{a} -{a} {a}'
            .format(a=1.087/3**.5),
            {'+111 (towards H)': (1, 1, 1), '-111 (towards face)': (-1, -1, -1),
             'x (S4 axis)': (1, 0, 0)}),
}

name = sys.argv[1] if len(sys.argv) > 1 else 'hf'
basis = sys.argv[2] if len(sys.argv) > 2 else 'cc-pvdz'
methods = sys.argv[3].split(',') if len(sys.argv) > 3 else ['HF', 'sfX2C1e', 'sfX2C1e(atom)']
distances = [10, 20, 40, 80, 160]   # Angstrom
bohr = 1 / 0.52917721092
geom, directions = molecules[name]

def run(method, mol):
    if method == 'HF':
        mf = qdhf.RHF(mol)
    elif method == 'DHF':
        mf = qdhf.DHF(mol)
    else:
        mf = qdhf.SFX2C1E_RHF(mol)
        if method == 'sfX2C1e(atom)':
            mf.approx = 'atom1e'
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-17
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: %s not converged' % method)
    return e

he = gto.M(atom='He', basis=basis, verbose=0)
mol0 = gto.M(atom=geom, basis=basis, verbose=0)
ref = {m: run(m, he) + run(m, mol0) for m in methods}

for method in methods:
    for label, v in directions.items():
        v = numpy.asarray(v, dtype=float)
        v /= numpy.linalg.norm(v)
        print('\n%s  %s...He  direction %s' % (method, name.upper(), label))
        print('   R/A   dE(quad)                     n        R^2 dE        R^3 dE        R^4 dE')
        prev = None
        for r in distances:
            x = v * r
            mol = gto.M(atom=geom + '; He %.12f %.12f %.12f' % tuple(x),
                        basis=basis, verbose=0)
            de = run(method, mol) - ref[method]
            if prev is not None and de * prev[1] > 0:
                n = '%8.4f' % -float(qmp.log(de / prev[1]) / qmp.log(qmp.mpf(r) / prev[0]))
            else:
                n = '      --'
            rb = qmp.mpf(r) * bohr
            print('%6g  %-26s %s  % .6e  % .6e  % .6e' % (
                r, qmp.nstr(de, 16), n, float(de*rb**2), float(de*rb**3), float(de*rb**4)),
                flush=True)
            prev = (r, de)
