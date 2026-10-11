#!/usr/bin/env python
'''
Split of the finite-basis c^-4 coefficient a4 of dM = <PC(z) - z> (HeH2+,
see gauss_heh2p_dipole.py) into

    a4_CBSf = (1/8) <W_V T^-1 [z,T]>     (the complete-basis formula, evaluated
              with matrix products in the Loewdin basis, NR ground state psi0)
    a4_K    = (1/32) <[T,[T,z]]> + 1/2 Re <[X2^T, z] T>,  X2 = T^-1 W_V/4 - (V + T)/2
              (the c^-4 expansion term without the explicit Delta = W_z - Tz - zT
              pieces; still contains finite-basis [V,z] != 0 and [T,[T,z]] != 0)
In the complete basis [T,[T,z]] = [V,z] = Delta = 0 and a4, a4_K, a4_CBSf all
tend to (1/8) <W_V T^-1 d_z> (mw_heh2p_dipole.py).
Also checks X2 against c^2 (X~ - 1) from the exact X matrix at large c.

Usage: python gauss_heh2p_a4split.py [basis ...]
'''
import sys
from socutils.qdhf import x2c, qarray
from socutils.qdhf.qarray import QArray, qmp, hmp
import gauss_heh2p_dipole as g


def split(basis):
    nao, e_nr, mu_nr, out, a, a2_ri = g.run(basis)
    mol = g.make_mol(basis)
    xmol, _ = x2c.uncontract_mol(mol)
    cbig = g.c0 * 1024
    d = x2c.get_xr(xmol, light_speed=qmp.nstr(cbig, 34), approx='1e')
    qi = d['qi']
    s, t, v, w = d['s'], d['t'], d['v'], d['w']
    oz = qi.dip_x(2)
    sm = x2c._sym_pow(s, qmp.mpf(-0.5))
    sp = x2c._sym_pow(s, qmp.mpf(0.5))
    T, V, W, O = (sm @ m @ sm for m in (t, v, w, oz))
    n = T.shape[0]
    I = QArray.eye(n)
    Ti = qarray.solve(T, I)
    X2 = Ti @ W * hmp.mpf(0.25) - (V + T) * hmp.mpf(0.5)
    Xl = sp @ d['x'] @ sm
    x2_err = ((Xl - I) * cbig**2 - X2).max_abs()
    x2_ref = X2.max_abs()
    e, cv = qarray.eigh(T + V)
    p = cv.real[:, 0:1]
    TO = T @ O - O @ T
    k1 = (p.T @ (T @ TO - TO @ T) @ p).to_mpmath()[0, 0] / 32
    m = (X2.T @ O - O @ X2.T) @ T
    k2 = (p.T @ m @ p).to_mpmath()[0, 0] / 2
    # same pieces in the CBS-reduced form: (5/32)<[T,[T,z]]> + (1/8)<W_V T^-1 [z,T]>
    k3 = (p.T @ W @ Ti @ (O @ T - T @ O) @ p).to_mpmath()[0, 0] / 8
    return a, a2_ri, k1, k2, k3, x2_err, x2_ref


if __name__ == '__main__':
    bases = sys.argv[1:] or ['cc-pvtz']
    print('HeH2+ R_AB = %.4f, zeta = %g.  a4 = a4_K + a4_RI (see docstring)' % (g.rab, g.zeta))
    print('%-12s %-12s %-12s %-12s %-12s %-12s %-12s %s' % (
        'basis', 'a2', 'a4(fit)', 'a4_CBSf', 'a4_K', '[T,[T,z]]/32', 'a4-a4_CBSf', '|X2 err|/|X2|'))
    for b in bases:
        a, a2_ri, k1, k2, k3, xe, xr = split(b)
        print('%-12s % .5e % .5e % .5e % .5e % .5e % .5e %.1e' % (
            b, float(a[0]), float(a[1]), float(k3), float(k1 + k2), float(k1),
            float(a[1] - k3), float(xe / xr)), flush=True)
