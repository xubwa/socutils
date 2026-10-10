#!/usr/bin/env python

'''
Long-range behavior of the He-He interaction energy

    dE(R) = E(He2, R) - 2 E(He)

for the nonrelativistic HF, the four-component Dirac-Coulomb HF and the
spin-free X2C-1e HF, all in quadruple precision.

Two neutral closed-shell atoms interact at the HF level only through the
overlap of their densities, so dE decays exponentially (Gaussian basis:
even faster).  The one-electron X2C decoupling of the *molecular*
one-electron Hamiltonian, however, contains the bare nuclear potential of
the other atom.  Its field Z/R^2 polarizes the decoupling (X and R matrices)
of each atom, which gives a spurious contribution ~ 1/R^4 to dE.  With the
atomic decoupling ('atom1e', block-diagonal X) the X matrix does not see
the other nucleus.

At large R the spurious term is far below the double precision resolution
of the total energy (~1e-15 Hartree); quad precision resolves it.

Usage:  python 02-he2_long_range.py [basis]
'''

import sys
import numpy
from pyscf import gto
from socutils import qdhf
from socutils.qdhf.qarray import qmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'unc-cc-pvdz'
distances = [5, 10, 20, 40, 80, 160]   # Angstrom

methods = {
    'HF': lambda mol: qdhf.RHF(mol),
    'DHF': lambda mol: qdhf.DHF(mol),
    'sfX2C1e': lambda mol: qdhf.SFX2C1E_RHF(mol),
    'sfX2C1e(atom)': lambda mol: _x2c_atom(mol),
}

def _x2c_atom(mol):
    mf = qdhf.SFX2C1E_RHF(mol)
    mf.approx = 'atom1e'
    return mf

def run(make, mol):
    mf = make(mol)
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: not converged', mol.atom)
    return e, mf.e_tot_double

he = gto.M(atom='He 0 0 0', basis=basis, verbose=0)
e_atom = {}
for name, make in methods.items():
    e_atom[name] = run(make, he)
    print('E(He) %-14s %s' % (name, qmp.nstr(e_atom[name][0], 34)))

results = {name: [] for name in methods}
for r in distances:
    mol = gto.M(atom='He 0 0 0; He 0 0 %g' % r, basis=basis, verbose=0)
    for name, make in methods.items():
        e, e_double = run(make, mol)
        de = e - 2 * e_atom[name][0]
        de_double = e_double - 2 * e_atom[name][1]
        results[name].append((r, de, de_double))
        print('R = %5g A  %-14s dE(quad) = % .6e   dE(double) = % .3e'
              % (r, name, float(de), de_double), flush=True)

bohr = 1 / 0.52917721092
print('\nInteraction energies dE(R) in Hartree (quad precision), '
      'and the local exponent n of dE ~ R^-n')
for name in methods:
    print('\n%s' % name)
    print('   R/A          dE(quad)                       n        R^4 dE (a.u.)    dE(double)')
    prev = None
    for r, de, de_double in results[name]:
        if prev is not None and de != 0 and prev[1] != 0 and (de / prev[1]) > 0:
            n = -float(qmp.log(de / prev[1]) / qmp.log(qmp.mpf(r) / prev[0]))
            n = '%8.4f' % n
        else:
            n = '      --'
        r4 = float(de * (qmp.mpf(r) * bohr)**4)
        print('%6g  %-30s %s  % .10e  % .3e' % (r, qmp.nstr(de, 20), n, r4, de_double))
        prev = (r, de)
