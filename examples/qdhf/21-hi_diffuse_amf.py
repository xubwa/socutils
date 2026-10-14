#!/usr/bin/env python

'''
Numerical stability of the atomic-mean-field X2C flavors for HI with
increasingly diffuse basis sets (production socutils, double precision).

basis level k: ANO-RCC plus k even-tempered s and p functions (ratio 2.5)
below the most diffuse s/p exponent on both atoms.

flavors (Dirac-Coulomb, no Gaunt/Breit):
    1e         X2C-1e
    x2camf     X2C-1e + atomic spin-orbit AMF ('x2camf')
    dirac_amf  amfX2C as implemented ('dirac_amf', pcc=True)
    amf_fixed  amfX2C with the pure 2e atomic correction (20-amfx2c_production_check)
    x2cmp      x2cmp = HF eamfX2C
reference: 4c DHF (pyscf), same contracted basis.

Usage:  python 21-hi_diffuse_amf.py [kmax]
'''

import os
import sys
import time
import contextlib
import importlib.util
import numpy
from pyscf import gto, scf, lib

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('amfchk', os.path.join(here, '20-amfx2c_production_check.py'))
amfchk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amfchk)

kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
geom = 'H 0 0 0; I 0 0 1.609'
flavors = ['1e', 'x2camf', 'dirac_amf', 'amf_fixed', 'x2cmp']


def make_basis(k):
    bas = {}
    for el in ('H', 'I'):
        b = gto.basis.load('ano-rcc', el)
        extra = []
        for l in (0, 1):
            emin = min(p[0] for s in b if s[0] == l for p in s[1:])
            extra += [[l, [emin / 2.5**(i+1), 1.]] for i in range(k)]
        bas[el] = b + extra
    return bas


def quiet(f):
    with open(os.devnull, 'w') as fn, contextlib.redirect_stdout(fn):
        return f()


def dhf_energy(mol):
    mf = scf.DHF(mol)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    e = mf.kernel()
    return e, mf.converged


t0 = time.time()
print('HI (1.609 A), ANO-RCC + k diffuse s,p; E(flavor) - E(DHF) / Hartree')
print('%2s %5s %10s %18s  ' % ('k', 'nao', 'min eig S', 'E(DHF)') + ''.join('%-14s' % f for f in flavors))
for k in range(kmax + 1):
    mol = gto.M(atom=geom, basis=make_basis(k), verbose=0)
    smin = numpy.linalg.eigvalsh(mol.intor('int1e_ovlp')).min()
    e4, conv4 = quiet(lambda: dhf_energy(mol))
    line = '%2d %5d %10.2e %18.10f%s ' % (k, mol.nao, smin, e4, '' if conv4 else '*')
    for fl in flavors:
        try:
            e = quiet(lambda: amfchk.energy(mol, fl))
            line += '% .6e  ' % (e - e4)
        except Exception as ex:
            line += '%-14s' % ('ERR ' + type(ex).__name__)
    print(line + '(%.0f s)' % (time.time() - t0), flush=True)
