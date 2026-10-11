#!/usr/bin/env python
'''
c -> infinity anchors (quad runs against the Schroedinger run on the same grid)

    python stage1_anchor.py anchor.json nr.json
prints dE_l(DC) - dE_l(NR), (l+1/2)^2 d_l/(Z alpha)^2 (-> 3/64 in the DPT
regime), and the local exponents of dE_l and of d_l.
'''
import sys
import json
import mpmath
mp = mpmath.mp
mp.dps = 34


def inc(fn):
    d = json.load(open(fn))
    o = {}
    for k in d['kappas']:
        o[k['l']] = o.get(k['l'], 0) + mp.mpf(k['e2'])
    return d, o


d, e = inc(sys.argv[1])
_, n = inc(sys.argv[2])
za2 = 1 / mp.mpf(d['cprime']) ** 2
print('# %s  c/Z = %s  (Z alpha)^2 = %s' % (sys.argv[1], d['cprime'][:16], mpmath.nstr(za2, 6)))
print('# 1s: %s' % d['e1s'][:36])
ls = sorted(l for l in e if l in n)
print('%3s %24s %24s %14s %16s %8s %8s' % ('l', 'dE_l(DC)', 'dE_l(NR)', 'd_l/dE_l(NR)', '(l+.5)^2 d/Za^2', 'p(dE)', 'p(d)'))
for l in ls:
    dd = e[l] - n[l]
    x = l + mp.mpf(0.5)
    pe = pd = ''
    if l + 1 in e and l + 1 in n:
        r = mp.log((l + 1.5) / x)
        pe = mpmath.nstr(-mp.log(e[l + 1] / e[l]) / r, 6)
        d1 = e[l + 1] - n[l + 1]
        if dd != 0 and d1 / dd > 0:
            pd = mpmath.nstr(-mp.log(d1 / dd) / r, 6)
    print('%3d %24s %24s %14s %16s %8s %8s' % (l, mpmath.nstr(e[l], 18), mpmath.nstr(n[l], 18),
                                                mpmath.nstr(dd / n[l], 6), mpmath.nstr(dd * x ** 2 / za2, 8), pe, pd))
print('# 3/64 = %s' % mpmath.nstr(mp.mpf(3) / 64, 8))
