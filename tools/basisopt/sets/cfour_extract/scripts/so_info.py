"""Spin-orbit information lines of the -SO entries of a CFOUR GENBAS, derived from the
contraction coefficients themselves (read only; nothing is written to the library).

Layout of a j-adapted l > 0: columns [n for j = l-1/2][n for j = l+1/2][shared functions].
Two indicators per l:
  A  half the number of contracted columns (more than one nonzero coefficient), if even;
  B  the largest n for which every pair (k, k+n) overlaps > 0.90, each column's best partner in
     the other block is its own, and the pair is not two different single primitives.
Proposed n: A if all its pairs (k, k+n) overlap > 0.80, otherwise B.  For the proposed n the
report gives the smallest pair overlap and the largest overlap between two DIFFERENT shells of
the same kappa (should be near 0 for atomic spinors; near 1 means two contracted functions are
almost the same function, i.e. near linear dependence in the entry)."""
import re, sys
import numpy as np
G = sys.argv[1] if len(sys.argv) > 1 else '/home/xwang405/apps/cfour-x2cso/basis/GENBAS'
LN = 'spdfghik'
lines = open(G, errors='replace').read().split('\n')
name_re = re.compile(r'^[A-Z]{1,2}:\S*-SO\S*\s*$')
fl = lambda x: float(x.replace('D', 'E').replace('d', 'e'))

def parse(i):
    j = i + 2
    while not lines[j].strip():
        j += 1
    ls = [int(x) for x in lines[j + 1].split()]
    nc = [int(x) for x in lines[j + 2].split()]
    npr = [int(x) for x in lines[j + 3].split()]
    k = j + 4
    out = []
    for l_, c_, p_ in zip(ls, nc, npr):
        ex, co = [], []
        while len(ex) < p_:
            ex += [fl(x) for x in lines[k].split()]; k += 1
        while len(co) < p_ * c_:
            co += [fl(x) for x in lines[k].split()]; k += 1
        out.append((l_, np.array(ex), np.array(co).reshape(p_, c_)))
    return out

def analyse(l, ex, c):
    a, b = ex[:, None], ex[None, :]
    s = (2 * np.sqrt(a * b) / (a + b)) ** (l + 1.5)
    nrm = np.sqrt(np.einsum('pi,pq,qi->i', c, s, c))
    cn = c / np.where(nrm > 0, nrm, 1)
    o = abs(cn.T @ s @ cn)
    nz = [np.flatnonzero(abs(c[:, k]) > 1e-12) for k in range(c.shape[1])]
    ncon = sum(len(z) > 1 for z in nz)
    single_pair = lambda i, j: len(nz[i]) == 1 and len(nz[j]) == 1 and nz[i][0] != nz[j][0]
    pairs = lambda n: np.array([o[k, k + n] for k in range(n)])
    nA = ncon // 2 if ncon % 2 == 0 else None
    nB = 0
    for n in range(c.shape[1] // 2, 0, -1):
        if pairs(n).min() < 0.90 or any(single_pair(k, k + n) for k in range(n)):
            continue
        cross = o[:n, n:2 * n]
        if all(np.argmax(cross[k]) == k for k in range(n)) and all(np.argmax(cross[:, k]) == k for k in range(n)):
            nB = n; break
    n = nA if (nA and pairs(nA).min() > 0.80) else nB
    if n:
        blk = np.concatenate([o[:n, :n][np.triu_indices(n, 1)], o[n:2 * n, n:2 * n][np.triu_indices(n, 1)]]) if n > 1 else np.array([0.0])
        return n, nA, nB, pairs(n).min(), blk.max(), c.shape[1] - 2 * n
    return 0, nA, nB, 1.0, 0.0, c.shape[1]

def line_of(shells):
    parts, notes, pmin, bmax = [], [], 1.0, 0.0
    for l, ex, c in shells:
        if l == 0:
            continue
        n, nA, nB, p, b, rest = analyse(l, ex, c)
        if n:
            parts.append((l, n)); pmin = min(pmin, p); bmax = max(bmax, b)
            if nA != nB:
                notes.append('%s: A=%s B=%s' % (LN[l], nA, nB))
            if b > 0.5:
                notes.append('%s: two shells of one kappa overlap %.2f' % (LN[l], b))
    return ('%d  %s' % (len(parts), '  '.join('%d %d' % p for p in parts)) if parts else '0'), pmin, bmax, notes

norm = lambda s: ' '.join(s.split())
rows = []
for i, line in enumerate(lines):
    if name_re.match(line):
        name, given = line.strip(), lines[i + 1].strip()
        sh = parse(i)
        mine, pmin, bmax, notes = line_of(sh)
        sizes = ' '.join('%s:%d/%d' % (LN[l], len(ex), c.shape[1]) for l, ex, c in sh)
        rows.append((name, given, mine, pmin, bmax, notes, sizes))
known = [r for r in rows if r[1]]
print('GENBAS: %s' % G)
print('%d entries with -SO in the name; %d carry an information line; the rule reproduces %d of them; %d have an empty line.' % (
    len(rows), len(known), sum(norm(r[1]) == norm(r[2]) for r in known), len(rows) - len(known)))
print('\n== Entries WITH a line that the coefficients do not support')
for r in known:
    if norm(r[1]) != norm(r[2]):
        print('%-22s library %-18r coefficients say %-14r  [prim/columns %s]' % (r[0], r[1], r[2], r[6]))
print('\n== Entries WITHOUT a line: proposed line, smallest overlap of a j pair, largest overlap of two shells of one kappa, remarks')
for r in rows:
    if not r[1]:
        print('%-22s %-22s pair %.3f  same-kappa %.2f  %s' % (r[0], r[2], r[3], r[4], '; '.join(r[5])))
print('\n== Entries with a line and near linear dependence (two shells of one kappa overlap > 0.5)')
for r in known:
    if r[4] > 0.5:
        print('%-22s %-22s same-kappa %.2f' % (r[0], r[1], r[4]))
