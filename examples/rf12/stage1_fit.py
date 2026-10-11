#!/usr/bin/env python
'''
Asymptotic fits of the stage-1 partial-wave data.

    python stage1_fit.py run.json nr.json [lo hi]

1. Energy increments: least squares in (l+1/2)^-n, n = 3..7, over l in
   [lo, hi]; the leading coefficient A3 is compared with the short-distance
   model value -3 Z alpha / (64 pi) (relmodel; large-large channel only,
   inference).
2. Coalescence (r1 = r2 = r): a_l(LL singlet)/Phi0 fitted with
   L * c_l[ln r12] + s * c_l[r12] + t (l+1/2)^-3 ; L compared with the
   model Z alpha / (4 pi), s with the Kato 1/2.
Fits are done in mpmath (40 digits) by plain normal equations on scaled
columns; the spread over fitting windows is printed as the error estimate.
'''
import sys
import json
import mpmath
from socutils.rf12 import angular as A
mp = mpmath.mp
mp.dps = 40


def lsq(cols, y):
    n = len(cols)
    M = mp.matrix(n, n)
    b = mp.matrix(n, 1)
    for i in range(n):
        for j in range(n):
            M[i, j] = mp.fsum(ci * cj for ci, cj in zip(cols[i], cols[j]))
        b[i] = mp.fsum(ci * yi for ci, yi in zip(cols[i], y))
    return mp.lu_solve(M, b)


def c_ln(l):
    # Legendre coefficient of ln(r12/r) = (1/2) ln(2(1-x)) at r1 = r2 = r, l >= 1
    return -mp.mpf(2 * l + 1) / (2 * l * (l + 1))


def load(fn):
    d = json.load(open(fn))
    by_l = {}
    for k in d['kappas']:
        by_l.setdefault(k['l'], []).append(k)
    return d, by_l


def main():
    run, nr = sys.argv[1], sys.argv[2]
    d, by_l = load(run)
    dn, by_ln = load(nr)
    cp = mp.mpf(d['cprime'])
    za = 1 / cp
    dE = {l: mp.fsum(mp.mpf(k['e2']) for k in ks) for l, ks in by_l.items()}
    lmax = max(dE)
    print('# %s   Z alpha = %s' % (run, mpmath.nstr(za, 8)))
    print('# energy increments: dE_l = sum_n A_n (l+1/2)^-n')
    print('# model A3 (LL channel, inference) = -3 Za/(64 pi) = %s' % mpmath.nstr(-3 * za / (64 * mp.pi), 8))
    for nmax in (5, 6, 7):
        for lo in (10, 14, 18):
            ls = [l for l in range(lo, lmax + 1)]
            x = [l + mp.mpf(1) / 2 for l in ls]
            cols = [[xi ** -n for xi in x] for n in range(3, nmax + 1)]
            sol = lsq(cols, [dE[l] for l in ls])
            print('  n=3..%d  l=%2d..%2d   A3=%s  A3/(Za)=%s  A4=%s' % (
                nmax, lo, lmax, mpmath.nstr(sol[0], 8), mpmath.nstr(sol[0] / za, 6), mpmath.nstr(sol[1], 6)))
    # coalescence
    print('# coalescence: a_l/Phi0 = L c_l[ln r12] + s r c_l[r12/r] + t (l+1/2)^-3 (+u (l+1/2)^-4)')
    print('# model L = Za/(4 pi) = %s   Kato s = 1/2' % mpmath.nstr(za / (4 * mp.pi), 8))
    for ir, r in enumerate(d['rcoal']):
        rr = mp.mpf(r)
        G0 = A.reference_tensor('LL')
        phi0 = (mp.mpf(d['Pa'][ir]) ** 2 * A.singlet(G0)).real
        a = {}
        for l, ks in by_l.items():
            a[l] = mp.fsum((mp.mpf(k['coal']['LL'][ir]) * A.singlet(A.coalescence_tensor(k['kappa'], 'LL'))).real
                           for k in ks) / phi0
        for lo in (10, 14, 18):
            for nt in (3, 4):
                ls = list(range(lo, lmax + 1))
                cols = [[c_ln(l) for l in ls], [rr * A.legendre_r12(l) for l in ls]]
                cols += [[(l + mp.mpf(1) / 2) ** -n for l in ls] for n in range(3, nt + 1)]
                sol = lsq(cols, [a[l] for l in ls])
                print('  r=%-4s l=%2d..%2d nt=%d  L=%s  L/(Za/4pi)=%s  s=%s' % (
                    r, lo, lmax, nt, mpmath.nstr(sol[0], 8), mpmath.nstr(sol[0] / (za / (4 * mp.pi)), 6),
                    mpmath.nstr(sol[1], 6)))


if __name__ == '__main__':
    main()
