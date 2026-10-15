#!/usr/bin/env python

'''
c scaling of the He...He R^-4 coefficient in sf-X2C-1e, '1e' and 'atom1e',
at fixed R = 320 A (R^-4 checked separately), c in the perturbative regime.

    R^4 dE = a4/c^4 + a6/c^6 + a8/c^8

'1e':     a4 -> -4/3 (X responds to the bare nuclear field)
'atom1e': finite-basis a4 (RI error of dz, ~1e-7 in this basis) and the
          complete-basis a8 (relaxation under the c^-4 part of dz).

Usage:  python 29-atom1e_c_scaling.py [c ...]
'''

import sys
import time
import numpy
from pyscf import gto, lib
from socutils import qdhf
from socutils.qdhf.qarray import qmp

bohr = 1 / 0.52917721092
cs = [float(x) for x in sys.argv[1:]] or [10., 14., 20., 28., 40., float(lib.param.LIGHT_SPEED)]
r = 320
t0 = time.time()


def etb(beta, ranges):
    etbs = []
    for l, (a0, a1) in enumerate(ranges):
        n = int(numpy.floor(numpy.log(a1 / a0) / numpy.log(beta))) + 1
        etbs.append((l, n, a0, beta))
    return {'He': gto.etbs(etbs)}


basis = etb(2.0, [(0.02, 1e8), (0.05, 1e3)])


def energy(geom, approx, c):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = qdhf.SFX2C1E_RHF(mol, light_speed=repr(c))
    mf.approx = approx
    mf.direct = True
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.lindep = 1e-28
    mf.max_cycle = 100
    mf.init_guess = 'double'
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING not converged', approx, c, geom, flush=True)
    return e


res = {'1e': [], 'atom1e': []}
print('He...He sf-X2C-1e, ETB sp (b=2.0, s 0.02..1e8, p 0.05..1e3), R = %d A' % r)
print('%10s %-7s %16s %14s %14s' % ('c', 'approx', 'R^4 dE', 'c^4 R^4 dE', 'c^8 R^4 dE'))
for c in cs:
    for approx in ('1e', 'atom1e'):
        e0 = 2 * energy('He 0 0 0', approx, c)
        de = energy('He 0 0 0; He 0 0 %g' % r, approx, c) - e0
        y = de * (qmp.mpf(r) * bohr)**4
        res[approx].append(y)
        print('%10.4f %-7s % .8e % .6e % .6e  (%.0f s)' % (c, approx, float(y), float(y * c**4),
                                                         float(y * c**8), time.time() - t0), flush=True)
for approx in ('1e', 'atom1e'):
    a = qmp.matrix([[qmp.mpf(c)**(-4), qmp.mpf(c)**(-6), qmp.mpf(c)**(-8)] for c in cs])
    b = qmp.matrix(res[approx])
    coef = qmp.qr_solve(a, b)[0]
    print('fit %-7s a4 % .6e  a6 % .6e  a8 % .6e' % (approx, *[float(x) for x in coef]), flush=True)
