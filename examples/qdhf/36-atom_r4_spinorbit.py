#!/usr/bin/env python

'''
Second-order (R^-4) spurious interaction of two closed-shell atoms:
X2C-1e with spin-orbit coupling (socutils.qdhf.x2c4c, full W) against
sf-X2C-1e (W = p.Vp), both decoupled in the decontracted basis.

    E_spur = -kappa (N_A Z_B^2 + N_B Z_A^2) / (c^4 R^4)

kappa = 1/12 (spin-free): h^(2) = F^2 c^-4 [-1/8 + p_F^2 / (8 p^2)].
kappa = 1/16 (spin-orbit): the scalar part of h^(2) vanishes; the term is the
second-order response to the unscreened spin-orbit field of the bare nucleus,
-F (sigma x p)_F / (4 c^2), closed by the sum rule (sigma x p) = i [H, sigma x r].

Printed: c^4 R^4 dE / (N_A Z_B^2 + N_B Z_A^2), i.e. -kappa.

Usage:  python 36-atom_r4_spinorbit.py [part]     part: std | std2 | et | c  (run one at a time: about 10 GB each)
'''

import sys
import time
from pyscf import gto, lib
from socutils import qdhf
from socutils.qdhf import x2c4c
from socutils.qdhf.qarray import qmp

part = sys.argv[1] if len(sys.argv) > 1 else 'std'
bohr = 1 / 0.52917721092
t0 = time.time()


def et(n, a0, b):
    return [a0 * b**i for i in range(n)]


ET_HE = [[0, [e, 1.]] for e in et(10, 0.06, 2.6)] + [[1, [e, 1.]] for e in et(6, 0.12, 2.6)] + \
        [[2, [e, 1.]] for e in et(3, 0.3, 2.6)]


def energy(geom, basis, so, c):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    if so:
        mf = x2c4c.GHF2c(mol, '1e')
    else:
        mf = qdhf.SFX2C1E_RHF(mol)
        mf.init_guess = 'double'
    mf.light_speed = c
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING not converged', geom, flush=True)
    return e


def run(a, b, za, zb, basis, label, c=None, rs=(160, 320)):
    cc = qmp.mpf(lib.param.LIGHT_SPEED if c is None else c)
    norm = za * zb**2 + zb * za**2
    for so in (False, True):
        e0 = energy('%s 0 0 0' % a, basis, so, c) + energy('%s 0 0 0' % b, basis, so, c)
        vals = []
        for r in rs:
            de = energy('%s 0 0 0; %s 0 0 %g' % (a, b, r), basis, so, c) - e0
            vals.append(float(de * (qmp.mpf(r) * bohr)**4 * cc**4 / norm))
        print('%s...%s %-12s c=%-8g %s  ' % (a, b, label, float(cc), 'SO' if so else 'sf') +
              '  '.join('R=%g: % .5f' % (r, v) for r, v in zip(rs, vals)) +
              '   (-1/12 = -0.08333, -1/16 = -0.06250)  [%.0f s]' % (time.time() - t0), flush=True)


if part == 'std':
    for a, b, za, zb in (('He', 'He', 2, 2), ('He', 'Ne', 2, 10), ('Ne', 'Ne', 10, 10)):
        for basis in ('unc-cc-pvdz', 'unc-cc-pvtz'):
            run(a, b, za, zb, basis, basis)
elif part == 'std2':
    run('He', 'Ne', 2, 10, 'unc-cc-pvtz', 'unc-cc-pvtz')
    for basis in ('unc-cc-pvdz', 'unc-cc-pvtz'):
        run('Ne', 'Ne', 10, 10, basis, basis)
elif part == 'et':
    run('He', 'He', 2, 2, {'He': ET_HE}, 'ET 10s6p3d')
elif part == 'c':
    for c in (10, 20, 40):
        run('He', 'He', 2, 2, {'He': ET_HE}, 'ET 10s6p3d', c=c, rs=(160,))
