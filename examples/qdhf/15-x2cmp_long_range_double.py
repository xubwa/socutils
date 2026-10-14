#!/usr/bin/env python

'''
Long-range behaviour of the production spin-free x2cmp Hamiltonian ('sfmp') (socutils.somf.x2cmp,
double precision) for HF...He, compared with X2C-1e in the same spinor
framework and with nonrelativistic HF.

sf-X2C-1e has a spurious R^-2 tail for HF...He (~2e-6 R^-2 a.u. in
cc-pVDZ, i.e. 1e-9 Hartree at 20 A and 6e-11 at 80 A), well above the
~1e-14 numerical noise of double precision, so the absence of the tail in
x2cmp can be tested directly with the production code at 10-80 A.  The
physical interaction is the R^-6 induction + dispersion (~1e-12 at 40 A).

flavors:  NR      spinor HF without relativity
          sf1e    spin-free X2C-1e                (amf_type 'sf1e')
          sfmp    spin-free x2cmp                 (amf_type 'sfmp')

Usage:  python 15-x2cmp_long_range_double.py [basis]
'''

import os
import sys
import contextlib
import numpy
from pyscf import gto, lib
from socutils.scf import spinor_hf

# pyscf.lib.einsum relies on numpy.einsum_path internals that changed in
# numpy 2.5; fall back to numpy.einsum (same result) in this environment.
lib.einsum = lambda subscripts, *ops, **kw: numpy.einsum(subscripts, *ops, optimize=True)

basis = sys.argv[1] if len(sys.argv) > 1 else 'cc-pvdz'
flavors = ['NR', 'sf1e', 'sfmp']
distances = [5, 10, 20, 40, 80, 160]
geom_a = 'F 0 0 0; H 0 0 0.917'
os.environ['X2CAMF_TIMING'] = '0'


def energy(flavor, geom):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = spinor_hf.SCF(mol)
    if flavor != 'NR':
        mf.x2cmp(flavor=flavor, with_gaunt=False, with_breit=False, with_pcc=False)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.verbose = 0
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        e = mf.kernel()
    if not mf.converged:
        print('WARNING: %s not converged at %s' % (flavor, geom), flush=True)
    return e


print('HF...He  (%s)   dE(R) / Hartree, local exponent n in ()' % basis)
res = {}
for fl in flavors:
    e0 = energy(fl, geom_a) + energy(fl, 'He 0 0 0')
    res[fl] = [energy(fl, geom_a + '; He 0 0 %g' % r) - e0 for r in distances]
print('  R/A  ' + ''.join('%-24s' % f for f in flavors))
for i, r in enumerate(distances):
    line = '%5g  ' % r
    for fl in flavors:
        de = res[fl][i]
        n = ''
        if i > 0 and de * res[fl][i-1] > 0:
            n = '(%5.2f)' % (-numpy.log(de / res[fl][i-1]) / numpy.log(r / distances[i-1]))
        line += '% .4e %-9s ' % (de, n)
    print(line, flush=True)
print('\nratio to NR:')
for i, r in enumerate(distances):
    print('%5g  ' % r + ''.join('%-24.6f' % (res[fl][i] / res['NR'][i]) for fl in flavors), flush=True)
