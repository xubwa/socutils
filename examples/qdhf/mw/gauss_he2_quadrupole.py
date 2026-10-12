#!/usr/bin/env python
'''
He2 (He at z = -+1.5 Angstrom, Gaussian nuclei exp(-zeta r^2)), quad sf-X2C-1e
(approx='1e', X from the molecular one-electron Dirac Hamiltonian of He2):
c-scaling of the picture-change error of the quadrupole

    O = r^2 P2(cos theta) = z^2 - (x^2+y^2)/2   (origin at the midpoint)
    dM2(c) = Tr[D(c) (PC(O) - O)],  PC(O) = R^T (O + X^T W_O X / 4c^2) R

Far charge Z_B on +z:  R^3 E_spurious -> -Z_B dM2  (script 14: Z_B = 2).

O and W_O = <p chi|O|p chi> are taken from the potential of two point charges
1 at z = +-Rf (dipole and odd multipoles cancel exactly):
    -(Rf^3/2) (V(Rf) + 2/Rf) = O + O(Rf^-2),  Richardson in Rf^-2 (Rf = 1000, 2000 bohr).

dM2_fixed(c) = Tr[D0 (PC(O) - O)] (D0: NR-HF density) is fitted to
a2/c^2 + ... + a8/c^8 at c = c0*k.  The density relaxation
    dM2(c) - dM2_fixed(c) = Tr[(D(c) - D0)(PC(O) - O)] = r4/c^4 + O(c^-6)
uses the sf-X2C-1e RHF density D(c) at each c (SCF in double precision with the
quad h_X2C(c), tight convergence).  Its noise grows like c^4 (double precision
SCF), so r4 is the Richardson extrapolation of c^4 (dM2 - dM2_fixed) from
k = 8, 16 (values for all k are printed).  a4(SCF) = a4(fixed) + r4.
In the complete basis r4 = Tr[D2 A2] = 0 because the c^-2 operator A2 = lap O/8 = 0.

The CBS formula evaluated with matrices in each basis (Loewdin basis, D0):
    a4_H1 = (1/32) Tr D0 [T,[T,O]],  a4_H2kin = 4 a4_H1,
    a4_W  = (1/8) Tr D0 W_V T^-1 [O,T]
    kin   = -Tr D0 (2 <dz|dz> - <dx|dx> - <dy|dy>)  (CBS form of Tr D0 [T,[T,O]])
    a2_RI = (1/4) Tr D0 (W_O - T O - O T)

Usage: python gauss_he2_quadrupole.py [basis ...]
  basis: pyscf name, or heN (even-tempered s,p,d,f,g on He, see etbasis)
Env: QDHF_ZETA (default 1e4), QDHF_KCS (default 8,16,32,64,128)
'''
import os
import sys
import time
import numpy
from pyscf import gto, scf, lib
from pyscf.gto import mole
from socutils.qdhf import x2c, qarray
from socutils.qdhf.qarray import QArray, qmp, hmp

zeta = float(os.environ.get('QDHF_ZETA', '1e4'))
kcs = [int(x) for x in os.environ.get('QDHF_KCS', '8,16,32,64,128').split(',')]
c0 = qmp.mpf(lib.param.LIGHT_SPEED)
rfs = (1000.0, 2000.0)


def etbasis(name):
    '''heN: s N+6 (0.04 * 2^k), p N/2 (0.08 * 2.3^k), d N/3, f N/4, g N/6
    (0.15, 0.3, 0.5 * 2.5^k)'''
    n = int(name[2:])
    b = [[0, [0.04 * 2.0**k, 1.]] for k in range(n + 6)]
    b += [[1, [0.08 * 2.3**k, 1.]] for k in range(n // 2)]
    b += [[2, [0.15 * 2.5**k, 1.]] for k in range(n // 3)]
    b += [[3, [0.3 * 2.5**k, 1.]] for k in range(n // 4)]
    b += [[4, [0.5 * 2.5**k, 1.]] for k in range(n // 6)]
    return {'He': b}


def make_mol(basis):
    def nucmod(z, prop=None):
        return zeta
    b = etbasis(basis) if basis.startswith('he') else basis
    return gto.M(atom='He 0 0 -1.5; He 0 0 1.5', basis=b, nucmod=nucmod, verbose=0,
                 max_memory=8000)


def tr(a, b):
    return qmp.re(qarray.trace_dot(a, b))


def scf_density(mol, h=None, dm0=None):
    mf = scf.RHF(mol)
    if h is not None:
        mf.get_hcore = lambda *args: h
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 100
    mf.kernel(dm0)
    assert mf.converged
    return mf.make_rdm1(), mf.energy_elec()[0]


def run(basis):
    t0 = time.time()
    mol = make_mol(basis)
    xmol, contr = x2c.uncontract_mol(mol)
    for ia in range(xmol.natm):
        assert xmol._env[xmol._atm[ia, mole.PTR_ZETA]] == zeta
    d = x2c.get_xr(xmol, light_speed=qmp.nstr(c0 * kcs[0], 34), approx='1e')
    qi = d['qi']
    s, t = d['s'], d['t']
    # O and W_O from two far point charges, Richardson in Rf^-2
    os_, ws_ = [], []
    for rf in rfs:
        vc, vyc = qi.charge_potential([hmp.mpf(1), hmp.mpf(1)], [[0, 0, rf], [0, 0, -rf]], [0., 0.])
        wc = vyc[0::3, 0::3] + vyc[1::3, 1::3] + vyc[2::3, 2::3]
        k = hmp.mpf(-rf**3 / 2)
        os_.append((vc + s * hmp.mpf(2 / rf)) * k)
        ws_.append((wc + t * hmp.mpf(4 / rf)) * k)
    w1, w2 = hmp.mpf(rfs[0]**2), hmp.mpf(rfs[1]**2)
    O = (os_[1] * w2 - os_[0] * w1) * (1 / (w2 - w1))
    WO = (ws_[1] * w2 - ws_[0] * w1) * (1 / (w2 - w1))
    O_raw = os_[1]

    def tox(dm):
        return contr @ QArray.from_numpy(dm) @ contr.T

    dm0, e_nr = scf_density(mol)
    d0 = tox(dm0)
    q_nr = tr(d0, O)
    out = []
    dm = dm0
    for kc in kcs:
        c = c0 * kc
        if kc != kcs[0]:
            d = x2c.get_xr(xmol, light_speed=qmp.nstr(c, 34), approx='1e', qi=qi)
        h = (contr.T @ x2c.hcore_from_xr(d) @ contr).to_numpy().real
        dm, e = scf_density(mol, h, dm)
        dx = tox(dm)
        pc = x2c.picture_change(d, O, WO) - O
        pc_raw = x2c.picture_change(d, O_raw, ws_[1]) - O_raw
        out.append((c, e, tr(dx, pc), tr(d0, pc), tr(dx, pc_raw)))

    def fit(col):
        nk = min(4, len(out) - 1)
        A = qmp.matrix([[(c0 / o[0])**(2 * k) for k in range(1, nk + 1)] for o in out])
        b = qmp.matrix([o[col] for o in out])
        bk = qmp.qr_solve(A, b)[0]
        return [bk[k - 1] * c0**(2 * k) for k in range(1, nk + 1)]

    # CBS formula in this basis (Loewdin), NR density
    sm = x2c._sym_pow(s, qmp.mpf(-0.5))
    sp = x2c._sym_pow(s, qmp.mpf(0.5))
    T, Wv, Ol, WOl = (sm @ m @ sm for m in (t, d['w'], O, WO))
    Ti = qarray.solve(T, QArray.eye(T.shape[0]))
    do = sp @ d0 @ sp
    TO = T @ Ol - Ol @ T
    kmat = tr(do, T @ TO - TO @ T)
    wmat = tr(do, Wv @ Ti @ (Ol @ T - T @ Ol))
    a2ri = tr(do, WOl - T @ Ol - Ol @ T) / 4
    sy = qi.ovlp_y()
    kin = -tr(d0, sy[2::3, 2::3] * 2 - sy[0::3, 0::3] - sy[1::3, 1::3])
    r4k = [(o[2] - o[3]) * o[0]**4 for o in out]
    r4 = (r4k[1] * 4 - r4k[0]) / 3
    return dict(nao=xmol.nao, e_nr=e_nr, q_nr=q_nr, out=out, fixed=fit(3), r4k=r4k, r4=r4,
                raw=fit(4), kmat=kmat, wmat=wmat, a2ri=a2ri, kin=kin, time=time.time() - t0)


if __name__ == '__main__':
    bases = sys.argv[1:] or ['cc-pvdz']
    print('He2 (+-1.5 A), Gaussian nuclei zeta = %g, approx=1e.  O = z^2-(x^2+y^2)/2 about the midpoint' % zeta)
    print('dM2(c) = Tr[D(c)(PC(O)-O)], c = c0*k, k = %s;  R^3 E_far = -Z_B dM2' % kcs)
    for b in bases:
        r = run(b)
        print('basis %-12s nao %4d  E_elec(NR-HF) % .12f  <O>_NR % .10f  (%.0f s)' % (
            b, r['nao'], r['e_nr'], float(r['q_nr']), r['time']))
        for c, e, dm, dmf, dmr in r['out']:
            print('    c/c0 %4.0f  E_elec % .12f  dM2 % .12e  dM2(fixed D0) % .12e  dM2(Rf=2000 only) % .6e' % (
                float(c / c0), e, float(dm), float(dmf), float(dmr)))
        a = r['fixed']
        print('  fit fixed D0: a2 % .8e  a4 % .8e  a6 % .3e  a8 % .2e' % tuple(float(x) for x in a))
        print('  relaxation c^4 (dM2 - dM2_fixed) at k=%s: %s -> r4 % .6e' % (
            kcs, ' '.join('% .5e' % float(x) for x in r['r4k']), float(r['r4'])))
        print('  RESULT %-12s a2 % .8e  a4(fixed) % .8e  a4(SCF) % .8e   at c0 as R^3 E (Z_B=2): '
              'c^-2 % .4e  c^-4 % .4e' % (b, float(a[0]), float(a[1]), float(a[1] + r['r4']),
                                           float(-2 * a[0] / c0**2), float(-2 * (a[1] + r['r4']) / c0**4)))
        print('  fit with Rf=2000 only (no Richardson, SCF D): a2 % .8e  a4 % .8e' % tuple(
            float(x) for x in r['raw'][:2]))
        print('  in-basis CBS formula (D0): a2_RI % .8e  a4_H1 % .8e  a4_H2kin % .8e  a4_W % .8e  a4 % .8e'
              '  kin(CBS form) %.8e vs Tr D[T,[T,O]] %.8e' % (
                  float(r['a2ri']), float(r['kmat'] / 32), float(r['kmat'] / 8), float(r['wmat'] / 8),
                  float(r['kmat'] * 5 / 32 + r['wmat'] / 8), float(r['kin']), float(r['kmat'])),
              flush=True)
