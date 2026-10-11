#!/usr/bin/env python

'''
Term-by-term decomposition of the picture-change error of a far nuclear
potential in sf-X2C-1e, following the H^1..H^4 partition of the draft, in a
non-orthogonal basis.

Fragment A (HF molecule) and a far bare point charge C (Z_C = 2 on the +z
axis, no basis functions).  All matrices are taken in the Loewdin
orthonormalized decontracted basis of A (tilde quantities), where

    Delta = R (V + X^T W X / 4c^2) R - V
          = H1 + H2 + H3 + H4                                     (exact)
    H1 = 1/2 ((R V - V R) R^-1 + R^-1 (V R - R V))       = 1/2 [[R,V],R^-1]
    H2 = 1/4c^2 R ([X^T,V] T X + X^T T [V,X]) R
    H3 = 1/4c^2 R X^T (W~ - V T - T V) X R     W~ = <f|T^V^ + V^T^|f> (exact)
    H4 = 1/4c^2 R X^T W' X R                   W' = 1/2 <f|lap V|f> (Darwin)

with V = V_C, W = <p f|V_C|p f>, W' = 2 pi Z_C f(C) f(C)^T and W~ = W - W'.
The products V T, T V are matrix products in the orthonormal basis, i.e.
V S^-1 T in the AO basis: H3 is the resolution-of-identity error.

The first-order contribution of each term to the interaction energy is
E_k = Tr[D H_k] (D: monomer density of A), reported as R^2 E_k (R in bohr).

approx='atom1e': X and R do not depend on V_C, sum_k E_k is the full
  first-order spurious energy.
approx='1e': X and R are those of A + C; the remainder
  E_X = Tr[D (h[A+C] - h[A] - V_C)] - sum_k E_k is the response of the
  decoupling (X, R) itself to V_C.

Usage:  python 14-h1h4_decomposition.py [basis ...]
'''

import sys
import numpy
from pyscf import gto, lib
from socutils.qdhf import x2c, qarray
from socutils.qdhf import integrals as qints
from socutils.qdhf.qarray import QArray, qmp, hmp

bohr = 1 / 0.52917721092
geom_a = 'F 0 0 0; H 0 0 0.917'
zc = 2
c0 = qmp.mpf(lib.param.LIGHT_SPEED)


def trace(a, b):
    return qmp.re(qarray.trace_dot(a, b))


def decompose(basis, r_ang, approx, c):
    mol = gto.M(atom=geom_a, basis=basis, verbose=0)
    mf = x2c.SFX2C1E_RHF(mol, light_speed=qmp.nstr(c, 34))
    mf.approx = approx
    mf.init_guess = 'double'
    mf.conv_tol = 1e-28
    mf.conv_tol_grad = 1e-15
    mf.verbose = 0
    mf.kernel()
    xmol, contr = x2c.uncontract_mol(mol)
    dm = contr @ mf.make_rdm1() @ contr.T

    d_a = x2c.get_xr(xmol, light_speed=qmp.nstr(c, 34), approx=approx)
    qi = d_a['qi']
    s, t = d_a['s'], d_a['t']
    rc = r_ang * bohr
    vc, vyc = qi.charge_potential([hmp.mpf(zc)], [[0, 0, rc]], [0.])
    wc = vyc[0::3, 0::3] + vyc[1::3, 1::3] + vyc[2::3, 2::3]
    f = QArray.from_mpf(qints.ao_values(xmol, [0, 0, rc]).reshape(-1, 1))
    wp = f @ f.T * (2 * hmp.pi * zc)

    if approx == 'atom1e':
        x, r = d_a['x'], d_a['r']
    else:
        x = x2c.xmatrix(t, d_a['v'] + vc, d_a['w'] + wc, s, c)
        r = x2c._get_r(s, s + x.T @ t @ x * (1 / (2 * c**2)))

    sm = x2c._sym_pow(s, qmp.mpf(-0.5))
    sp = x2c._sym_pow(s, qmp.mpf(0.5))
    V = sm @ vc @ sm
    T = sm @ t @ sm
    W = sm @ wc @ sm
    Wp = sm @ wp @ sm
    Wt = W - Wp
    X = sp @ x @ sm
    R = sp @ r @ sm
    R = (R + R.T) * hmp.mpf(0.5)
    Ri = qarray.solve(R, QArray.eye(R.shape[0]))
    k = 1 / (4 * c**2)
    delta = R @ (V + X.T @ W @ X * k) @ R - V
    h1 = ((R @ V - V @ R) @ Ri + Ri @ (V @ R - R @ V)) * hmp.mpf(0.5)
    h2 = R @ ((X.T @ V - V @ X.T) @ T @ X + X.T @ T @ (V @ X - X @ V)) @ R * k
    h3 = R @ X.T @ (Wt - V @ T - T @ V) @ X @ R * k
    h4 = R @ X.T @ Wp @ X @ R * k
    do = sp @ dm @ sp
    r2 = qmp.mpf(rc)**2
    e = [trace(do, h) * r2 for h in (h1, h2, h3, h4)]
    resid = (delta - h1 - h2 - h3 - h4).max_abs()
    ex = None
    if approx == '1e':
        h_a = x2c.hcore_from_xr(d_a)
        d_ac = dict(d_a, x=x, r=r, v=d_a['v'] + vc, w=d_a['w'] + wc)
        h_ac = x2c.hcore_from_xr(d_ac)
        ex = trace(dm, h_ac - h_a - vc) * r2 - sum(e)
    return e, resid, ex


if __name__ == '__main__':
    bases = sys.argv[1:] if len(sys.argv) > 1 else ['cc-pvdz', 'cc-pvtz', 'aug-cc-pvtz']
    print('HF + bare charge Z=2 at R on +z (H side).  R^2 E_k (a.u.), E_k = Tr[D H_k]')
    print('%-12s %-7s %4s %4s  %-13s %-13s %-13s %-13s %-13s %-9s' % (
        'basis', 'approx', 'c/c0', 'R/A', 'H1', 'H2', 'H3', 'H4', 'X-response', '|resid|'))
    for basis in bases:
        for approx in ('atom1e', '1e'):
            for kc in (1, 2, 4):
                for r_ang in ((160, 320) if kc == 1 else (320,)):
                    e, resid, ex = decompose(basis, r_ang, approx, c0 * kc)
                    exs = '% .6e' % float(ex) if ex is not None else '      --     '
                    print('%-12s %-7s %4d %4d  % .6e % .6e % .6e % .6e %s %.1e' % (
                        basis, approx, kc, r_ang, *[float(v) for v in e], exs, float(resid)),
                        flush=True)
