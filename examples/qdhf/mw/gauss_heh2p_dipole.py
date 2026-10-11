#!/usr/bin/env python
'''
Picture-change error of the dipole operator for a one-electron model system
(HeH2+, fixed nuclei He at 0 and H at +z R_AB, Gaussian nuclear charges
exp(-zeta r^2)), sf-X2C-1e (approx='1e', i.e. the exact one-electron
decoupling within the basis), quad precision, for a sequence of basis sets.

    dM(c) = <phi|PC(z) - z|phi>,  PC(z) = R^T (z + X^T W_z X / 4c^2) R

phi = ground state of h_X2C.  dM(c) is computed at c = c0*k and fitted to
a2/c^2 + a4/c^4 + a6/c^6 + a8/c^8.  In the complete basis a2 = 0 and a4 is
given by the operator formula evaluated with multiwavelets (mw_heh2p_dipole.py).

Usage:  python gauss_heh2p_dipole.py [basis ...]
  basis: a pyscf basis name, or etN (even-tempered s,p,d sets, see etbasis)
Env: QDHF_ZETA (nuclear exponent, default 1e4), QDHF_RAB (bohr, default 2.0)
'''
import os
import sys
import numpy
from pyscf import gto, lib
from pyscf.gto import mole
from socutils.qdhf import x2c, qarray
from socutils.qdhf.qarray import QArray, qmp, hmp

zeta = float(os.environ.get('QDHF_ZETA', '1e4'))
rab = float(os.environ.get('QDHF_RAB', '2.0'))
c0 = qmp.mpf(lib.param.LIGHT_SPEED)
kcs = [8, 16, 32, 64, 128]


def etbasis(name):
    '''etN: even-tempered, beta = 2.0 (s), 2.2 (p), 2.5 (d) on He and H,
    s exponents from 0.02 up (N = number of s functions, N/2 p, N/4 d).
    etfN: as etN plus N/6 f (beta 2.5 from 0.2) and N/8 g (beta 2.5 from 0.4)'''
    withf = name.startswith('etf')
    ns = int(name[3:] if withf else name[2:])
    s = [0.02 * 2.0**k for k in range(ns)]
    p = [0.04 * 2.2**k for k in range(max(ns // 2, 1))]
    d = [0.1 * 2.5**k for k in range(max(ns // 4, 1))]
    b = [[0, [e, 1.]] for e in s] + [[1, [e, 1.]] for e in p] + [[2, [e, 1.]] for e in d]
    if withf:
        b += [[3, [0.2 * 2.5**k, 1.]] for k in range(max(ns // 6, 1))]
        b += [[4, [0.4 * 2.5**k, 1.]] for k in range(max(ns // 8, 1))]
    return {'He': b, 'H': b}


def make_mol(basis):
    b = etbasis(basis) if basis.startswith('et') else basis
    # nucmod as a function: kept by mol.copy()/build() in x2c.uncontract_mol
    def nucmod(z, prop=None):
        return zeta
    mol = gto.M(atom='He 0 0 0; H 0 0 %.10f' % rab, unit='B', basis=b,
                charge=2, spin=1, verbose=0, nucmod=nucmod)
    return mol


def ground(h, s):
    sm = x2c._sym_pow(s, qmp.mpf(-0.5))
    e, v = qarray.eigh(sm @ h @ sm)
    v = sm @ v.real
    return e.to_mpmath()[0], v[:, 0:1]


def run(basis):
    mol = make_mol(basis)
    xmol, _ = x2c.uncontract_mol(mol)
    for ia in range(xmol.natm):
        assert xmol._atm[ia, mole.NUC_MOD_OF] == mole.NUC_GAUSS
        assert xmol._env[xmol._atm[ia, mole.PTR_ZETA]] == zeta
    out = []
    d = None
    for kc in kcs:
        c = c0 * kc
        d = x2c.get_xr(xmol, light_speed=qmp.nstr(c, 34), approx='1e', qi=d['qi'] if d else None)
        qi = d['qi']
        if kc == kcs[0]:
            oz = qi.dip_x(2)
            dy = qi.dip_y(2)
            wz = dy[0::3, 0::3] + dy[1::3, 1::3] + dy[2::3, 2::3]
            e_nr, cnr = ground(d['t'] + d['v'], d['s'])
            mu_nr = (cnr.T @ oz @ cnr).to_mpmath()[0, 0]
            # c^-2 coefficient in this basis: RI error 1/4 <W_z - T S^-1 z - z S^-1 T>_NR
            si = qarray.solve(d['s'], QArray.eye(d['s'].shape[0]))
            ri = wz - d['t'] @ si @ oz - oz @ si @ d['t']
            a2_ri = (cnr.T @ ri @ cnr).to_mpmath()[0, 0] / 4
        h = x2c.hcore_from_xr(d)
        e, cg = ground(h, d['s'])
        pc = x2c.picture_change(d, oz, wz)
        dm = (cg.T @ (pc - oz) @ cg).to_mpmath()[0, 0]
        out.append((c, e, dm))
    # least squares fit dM = sum_k a_{2k} c^{-2k}, k = 1..4 (exactly determined + 1)
    # in scaled variables x = (c0/c)^2 (well conditioned), a_2k = b_k c0^2k
    A = qmp.matrix([[(c0 / o[0])**(2 * k) for k in range(1, 5)] for o in out])
    b = qmp.matrix([o[2] for o in out])
    bk = qmp.qr_solve(A, b)[0]
    a = [bk[k - 1] * c0**(2 * k) for k in range(1, 5)]
    return xmol.nao, e_nr, mu_nr, out, a, a2_ri


if __name__ == '__main__':
    bases = sys.argv[1:] or ['cc-pvdz', 'cc-pvtz', 'aug-cc-pvtz']
    print('HeH2+ (one electron), R_AB = %.4f bohr, Gaussian nuclei zeta = %g' % (rab, zeta))
    print('dM(c) = <PC(z) - z>, fit dM = a2/c^2 + a4/c^4 + a6/c^6 + a8/c^8, c = c0*k, k = %s' % kcs)
    for basis in bases:
        nao, e_nr, mu_nr, out, a, a2_ri = run(basis)
        print('basis %-12s nao %4d  E_NR % .15f  <z>_NR % .15f' % (basis, nao, float(e_nr), float(mu_nr)))
        for c, e, dm in out:
            print('    c/c0 %4.0f  E_X2C % .15f  dM % .12e  c^4 dM % .10f' % (
                float(c / c0), float(e), float(dm), float(dm * c**4)))
        print('  RI error a2 = 1/4 <W_z - T S^-1 z - z S^-1 T>_NR = % .10e' % float(a2_ri))
        print('  -> a2 % .10e  a4 % .10e  a6 % .6e  a8 % .3e' % tuple(float(x) for x in a), flush=True)
