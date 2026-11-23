"""Inventory of the relativistic families of the CFOUR x2cso GENBAS to be extracted into socutils
(user, 2026-11-05: "先把手里 genbas 的基组提取出来", "DE4 的也整理出来。出现两次的对比一下").

Families: names ending in -SO, -SF, -X2C, _X2C, -DE4 (cc-type P/PC/PWC, with or without aug, and
Dyall-type DY...).  For every (family, element) all occurrences are parsed and compared:
same primitives? same coefficients (up to the sign of a column)?  CFOUR uses the FIRST occurrence."""
import re, sys, json
import numpy as np
G = '/home/xwang405/apps/cfour-x2cso/basis/GENBAS'
lines = open(G).read().splitlines()
pat = re.compile(r'^([A-Z][A-Z]?):(\S+)$')
SUF = re.compile(r'^(.*?)(-SO|-SF|-X2C|_X2C|-DE4)$')
idx = {}
for i, ln in enumerate(lines):
    m = pat.match(ln.strip()) if ln and not ln[0].isspace() else None
    if not m: continue
    el, name = m.groups()
    if not SUF.match(name): continue
    idx.setdefault((name, el), []).append(i)
def parse(i):
    comment = lines[i + 1].strip()
    nl = int(lines[i + 3].split()[0])
    ls, nc, npr = ([int(x) for x in lines[i + 4 + k].split()[:nl]] for k in range(3))
    # data until the next entry header (a line starting with a letter + ':'), stop at blank-separated next name
    j = i + 7; toks = []
    need = sum(p + p * c for p, c in zip(npr, nc))
    while len(toks) < need:
        toks += lines[j].split(); j += 1
    it = iter(toks); shells = {}
    for l, c, p in zip(ls, nc, npr):
        ex = np.array([float(next(it)) for _ in range(p)])
        co = np.array([float(next(it)) for _ in range(p * c)]).reshape(p, c)
        if l in shells:                     # some entries list an l in several blocks (e.g. aug _X2C)
            e0, c0 = shells[l]
            ex = np.concatenate([e0, ex])
            co = np.block([[c0, np.zeros((c0.shape[0], co.shape[1]))], [np.zeros((co.shape[0], c0.shape[1])), co]])
        shells[l] = (ex, co)
    return comment, shells
def same(a, b):
    if sorted(a) != sorted(b): return 'different l'
    for l in a:
        ea, ca = a[l]; eb, cb = b[l]
        if len(ea) != len(eb) or ca.shape != cb.shape: return 'different size (l=%d: %s vs %s)' % (l, ca.shape, cb.shape)
        if not np.allclose(ea, eb, rtol=1e-8, atol=0): return 'different exponents (l=%d, max rel diff %.1e)' % (l, np.max(abs(ea / eb - 1)))
        d = np.minimum(abs(ca - cb), abs(ca + cb)).max()
        if d > 1e-6: return 'different coefficients (l=%d, max %.1e)' % (l, d)
    return 'identical'
fam = {}
dups = []
for (name, el), occ in sorted(idx.items()):
    fam.setdefault(name, []).append(el)
    if len(occ) > 1:
        parsed = [parse(i) for i in occ]
        res = [same(parsed[0][1], p[1]) for p in parsed[1:]]
        dups.append((name, el, occ, [p[0] for p in parsed], res))
print('families (%d):' % len(fam))
for name in sorted(fam, key=lambda n: (SUF.match(n).group(2), n)):
    print('  %-16s %3d  %s' % (name, len(fam[name]), ' '.join(fam[name])))
print('\nentries occurring more than once (%d):' % len(dups))
for name, el, occ, comments, res in dups:
    print('  %s:%s  lines %s' % (el, name, ', '.join(str(o + 1) for o in occ)))
    for k, (c, r) in enumerate(zip(comments, ['(first, used by CFOUR)'] + res)):
        print('      #%d %-40s | %s' % (k + 1, r, c[:70]))
# -X2C vs _X2C for the same element and level
print('\n-X2C vs _X2C (same element, same level):')
for (name, el), occ in sorted(idx.items()):
    if name.endswith('-X2C'):
        alt = name[:-4] + '_X2C'
        if (alt, el) in idx:
            r = same(parse(occ[0])[1], parse(idx[(alt, el)][0])[1])
            print('  %-3s %-14s vs %-14s %s' % (el, name, alt, r))
json.dump({'%s:%s' % (el, name): occ for (name, el), occ in idx.items()}, open('index.json', 'w'))
