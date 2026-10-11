#!/usr/bin/env python
'''
Collect the stage-1 numbers into the tables of rf12/notes/stage1_results.md.

    python stage1_summary.py RUNDIR      (e.g. examples/rf12/data)
'''
import sys
import json
import mpmath
from socutils.rf12 import angular as A
from socutils.rf12 import relmodel
mp = mpmath.mp
mp.dps = 30
R = sys.argv[1].rstrip('/') + '/'


def load(fn):
    d = json.load(open(R + fn))
    by = {}
    for k in d['kappas']:
        by.setdefault(k['l'], []).append(k)
    return d, by


def inc(by, key=None):
    if key is None:
        return {l: mp.fsum(mp.mpf(k['e2']) for k in ks) for l, ks in by.items()}
    return {l: mp.fsum(mp.mpf(k['e2ch'][key]) for k in ks) for l, ks in by.items()}


def pexp(f, l):
    return -mp.log(f[l + 1] / f[l]) / mp.log((l + 1.5) / (l + 0.5))


def lsq(cols, y):
    n = len(cols)
    M = mp.matrix(n, n)
    b = mp.matrix(n, 1)
    for i in range(n):
        for j in range(n):
            M[i, j] = mp.fsum(a * c for a, c in zip(cols[i], cols[j]))
        b[i] = mp.fsum(a * c for a, c in zip(cols[i], y))
    return mp.lu_solve(M, b)


def fitA3(f, lo=12, hi=30, nmax=6):
    ls = [l for l in range(lo, hi + 1) if l in f]
    cols = [[(l + mp.mpf(0.5)) ** -n for l in ls] for n in range(3, nmax + 1)]
    return lsq(cols, [f[l] for l in ls])[0]


def coal(d, by, comp, ir, part='s'):
    proj = A.singlet if part == 's' else A.triplet0
    out = {}
    for l, ks in by.items():
        tot = 0
        for k in ks:
            g = proj(A.coalescence_tensor(k['kappa'], comp))
            if d['Qa'] is None and l > 0:
                g += proj(A.coalescence_tensor(l, comp))
            tot += mp.mpf(k['coal'][comp][ir]) * g
        out[l] = tot
    G0 = A.reference_tensor('LL')
    phi0 = (mp.mpf(d['Pa'][ir]) ** 2 * A.singlet(G0)).real
    return {l: v / phi0 for l, v in out.items()}


def f(x, n=6):
    return mpmath.nstr(x, n)


# ---- NR anchor
d, by = load('nr_dx.json')
e = inc(by)
print('## NR anchor (Schroedinger, xfine grid)')
print('| l | dE_l | S_l = -(256/45)(l+1/2)^4 dE_l | Schwartz 1-(5/4)(l+1/2)^-2 | p_l | s_l (cusp, r=1) | p(amp) |')
print('|---|---|---|---|---|---|---|')
a = coal(d, by, 'LL', 2)
for l in (5, 10, 15, 20, 25, 29):
    x = l + mp.mpf(0.5)
    print('| %d | %s | %s | %s | %s | %s | %s |' % (
        l, f(e[l], 10), f(-e[l] * x ** 4 * 256 / 45, 8), f(1 - mp.mpf(5) / 4 / x ** 2, 8), f(pexp(e, l), 5),
        f((a[l] / (A.legendre_r12(l))).real, 5), f(pexp({k: abs(v) for k, v in a.items()}, l), 4)))
print('\nE2 (sum l<=30) = %s' % f(mp.fsum(e.values()), 12))

# ---- relativistic
print('\n## Dirac-Coulomb no-pair')
rows = []
nr_f = inc(load('nr_double_fine.json')[1])
nr_x = inc(load('nr_dx.json')[1])
for Z, fn, nrr in ((2, 'z2_dx.json', nr_x), (10, 'z10_ch.json', nr_x), (30, 'z30_ch.json', nr_f),
                   (50, 'z50_ch.json', nr_f), (80, 'z80_ch.json', nr_f)):
    d, by = load(fn)
    za = 1 / mp.mpf(d['cprime'])
    e = inc(by)
    dd = {l: e[l] - nrr[l] for l in e}
    sign = [l for l in range(1, 30) if dd[l] > 0 and dd[l + 1] < 0]
    row = dict(Z=Z, za=za, E2=mp.fsum(e.values()), p29=pexp(e, 29), cross=sign[0] + 1 if sign else None,
               dpt=max((dd[l] * (l + 0.5) ** 2 / za ** 2 for l in range(2, 8))))
    if 'e2ch' in by[1][0] and by[1][0]['e2ch']:
        ll = inc(by, 'LL')
        row['A3LL'] = fitA3({l: ll[l] - nrr[l] for l in ll}) / za
        row['A3LS'] = fitA3(inc(by, 'LS')) / za ** 3
        row['A3SS'] = fitA3(inc(by, 'SS')) / za ** 3
    row['A3'] = fitA3(dd) / za
    aLL = coal(d, by, 'LL', 2)
    row['ampLL'] = pexp({k: abs(v) for k, v in aLL.items()}, 28)
    row['ratio'] = aLL[29].real / relmodel.legendre_coeff(29, 1.0, float(mp.mpf(d['cprime'])))
    # log coefficient with the model removing the non-log part: L_eff from the ratio
    rows.append(row)
print('| Z | Z alpha | E2 (l<=30) | max_(l=2..7) (l+1/2)^2 d_l/(Za)^2 | sign change of d_l at l | p_l (l=29) | A3/(Za) total | A3(LL-NR)/(Za) | A3(LS)/(Za)^3 | A3(SS)/(Za)^3 | amp. exponent LL (l=28) | a_29/model (r=1) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print('| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        r['Z'], f(r['za'], 5), f(r['E2'], 10), f(r['dpt'], 4), r['cross'], f(r['p29'], 4), f(r['A3'], 4),
        f(r.get('A3LL', ''), 4) if 'A3LL' in r else '', f(r['A3LS'], 4) if 'A3LS' in r else '',
        f(r['A3SS'], 4) if 'A3SS' in r else '', f(r['ampLL'], 4), f(r['ratio'], 5)))
print('\nreference values: 3/64 = %s ;  -1/(16 pi) = %s' % (f(mp.mpf(3) / 64), f(-1 / (16 * mp.pi))))
