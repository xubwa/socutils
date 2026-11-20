#!/usr/bin/env python

'''
He2 (fixed bond length) + He pulled apart along the molecular axis:

    dE(R) = E(He2...He, R) - E(He2) - E(He)

R is measured from the midpoint of He2.  Quad precision NR-HF, 4c DHF and
sf-X2C-1e HF (molecular '1e' and atomic 'atom1e' decoupling).

Findings (cc-pVDZ, d = 3 A, see 03-he2_he_long_range.cc-pvdz.out):

* HF and DHF decay as R^-8: the permanent quadrupole of He2 (from the
  overlap of the two atoms) induces a dipole on the far He,
  dE = -alpha/2 (3 Theta/R^4)^2.  This is physical.
* sf-X2C-1e ('1e' and 'atom1e') decays as R^-3 with a positive sign
  (collinear).  This is a charge-quadrupole term: the nucleus of the far He
  sees the picture-change transformed (X2C) charge distribution of He2,
  while its electrons see the untransformed one.  The difference is
  neutral but carries a quadrupole ~ 1/c^2, so the interaction with the
  bare charge Z gives ~ Z dTheta P2(cos theta) / R^3.  Perpendicular
  geometry gives R^3 dE = -1/2 x collinear (checked to 1e-4).
  For two atoms (spherical, 02-he2_long_range.py) this term vanishes and
  the leading spurious term is R^-4.

Usage:  python 03-he2_he_long_range.py [basis] [He2 bond length in A]
'''

import sys
from pyscf import gto
from socutils import qdhf
from socutils.qdhf.qarray import qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
d = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
distances = [5, 10, 20, 40, 80, 160]   # Angstrom
bohr = 1 / 0.52917721092

def make(name, mol):
    if name == 'HF':
        mf = qdhf.RHF(mol)
    elif name == 'DHF':
        mf = qdhf.DHF(mol)
    else:
        mf = qdhf.SFX2C1E_RHF(mol)
        if name == 'sfX2C1e(atom)':
            mf.approx = 'atom1e'
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-17
    mf.max_cycle = 80
    mf.verbose = 0
    return mf

def run(name, mol):
    mf = make(name, mol)
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: %s not converged for %s' % (name, mol.atom))
    return e, mf.e_tot_double

names = ['HF', 'DHF', 'sfX2C1e', 'sfX2C1e(atom)']
he = gto.M(atom='He 0 0 0', basis=basis, verbose=0)
he2 = gto.M(atom='He 0 0 %g; He 0 0 %g' % (-d/2, d/2), basis=basis, verbose=0)
ref = {}
for name in names:
    ref[name] = [run(name, he), run(name, he2)]
    print('%-14s E(He) = %s   E(He2) = %s' % (
        name, qmp.nstr(ref[name][0][0], 34), qmp.nstr(ref[name][1][0], 34)), flush=True)

results = {name: [] for name in names}
for r in distances:
    mol = gto.M(atom='He 0 0 %g; He 0 0 %g; He 0 0 %g' % (-d/2, d/2, r),
                basis=basis, verbose=0)
    for name in names:
        e, e_double = run(name, mol)
        de = e - ref[name][0][0] - ref[name][1][0]
        de_double = e_double - ref[name][0][1] - ref[name][1][1]
        results[name].append((r, de, de_double))
        print('R = %5g A  %-14s dE(quad) = % .6e   dE(double) = % .3e'
              % (r, name, float(de), de_double), flush=True)

print('\nHe2(d=%g A)...He interaction energies (Hartree), local exponent n of '
      'dE ~ R^-n' % d)
for name in names:
    print('\n%s' % name)
    print('   R/A          dE(quad)                       n        R^4 dE (a.u.)    dE(double)')
    prev = None
    for r, de, de_double in results[name]:
        if prev is not None and de != 0 and prev[1] != 0 and (de / prev[1]) > 0:
            n = '%8.4f' % -float(qmp.log(de / prev[1]) / qmp.log(qmp.mpf(r) / prev[0]))
        else:
            n = '      --'
        r4 = float(de * (qmp.mpf(r) * bohr)**4)
        print('%6g  %-30s %s  % .10e  % .3e' % (r, qmp.nstr(de, 20), n, r4, de_double))
        prev = (r, de)
