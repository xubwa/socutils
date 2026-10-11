#!/usr/bin/env python

'''
Speed-of-light scaling of the long-range tails of the model Hamiltonians of
the original test script (nonrelativistic framework, X = 1):

    R from  S~ = S + T/(2c^2),  R = S^-1/2 (S^-1/2 S~ S^-1/2)^-1/2 S^1/2

    VRcomm : h = T + (R^-T V R + R^T V R^-1)/2
    RVR    : h = T + R^T V R
    sfX2C1e: full sf-X2C-1e (reference)

With R = 1 - S^-1 T/(4c^2) + O(c^-4), the O(c^-2) terms of R^-T V R and
R^T V R^-1 cancel in VRcomm, so its tails should scale as c^-4, while the
first-order picture-change error of sf-X2C-1e scales as c^-2.

Usage:  python 13-vr_commutator_cscaling.py [basis]
'''

import sys
from pyscf import gto, lib
from socutils import qdhf
from socutils.qdhf import x2c, dhf, qarray
from socutils.qdhf import integrals as qints
from socutils.qdhf.qarray import QArray, qmp, hmp

basis = sys.argv[1] if len(sys.argv) > 1 else 'unc-cc-pvdz'
bohr = 1 / 0.52917721092
c0 = qmp.mpf(lib.param.LIGHT_SPEED)


def model_hcore(mol, kind, c):
    qi = qints.QuadIntegrals(mol)
    s = qi.ovlp_x()
    v = qi.nuc_x()
    sy = qi.ovlp_y()
    t = (sy[0::3, 0::3] + sy[1::3, 1::3] + sy[2::3, 2::3]) * hmp.mpf(0.5)
    r = x2c._get_r(s, s + t * (1 / (2 * c**2)))
    if kind == 'RVR':
        return qi, t + r.T @ v @ r
    rinv = qarray.solve(r, QArray.eye(r.shape[0]))
    return qi, t + (rinv.T @ v @ r + r.T @ v @ rinv) * hmp.mpf(0.5)


def energy(geom, kind, c):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    if kind == 'sfX2C1e':
        mf = qdhf.SFX2C1E_RHF(mol, light_speed=qmp.nstr(c, 34))
        mf.init_guess = 'double'
    else:
        qi, h = model_hcore(mol, kind, c)
        ham = dhf.DirectRHFHamiltonian(mol, qi, h, qi.ovlp_x(),
                                       qmp.mpf(qints.energy_nuc(mol)))
        mf = dhf.RHF(mol)
        mf.build_hamiltonian = lambda: ham
        mf.init_guess = 'double'
    mf.conv_tol = 1e-32
    mf.conv_tol_grad = 1e-16
    mf.max_cycle = 100
    mf.verbose = 0
    e = mf.kernel()
    if not mf.converged:
        print('WARNING: not converged', kind, geom, flush=True)
    return e


systems = [('HF...He', 'F 0 0 0; H 0 0 0.917', 2),
           ('He...He', 'He 0 0 0', 4)]
r = 320
for label, geom_a, n in systems:
    print('\n%s  R = %d A   R^%d dE (a.u.) scaled by c^2 and c^4' % (label, r, n))
    for kind in ('VRcomm', 'RVR', 'sfX2C1e'):
        for k in (1, 2, 4):
            c = c0 * k
            de = energy(geom_a + '; He 0 0 %g' % r, kind, c) - \
                energy(geom_a, kind, c) - energy('He', kind, c)
            rn = de * (qmp.mpf(r) * bohr)**n
            print('%-8s c/c0=%d   R^n dE = % .6e   c^2* = % .6e   c^4* = % .6e' % (
                kind, k, float(rn), float(rn * c**2), float(rn * c**4)), flush=True)
