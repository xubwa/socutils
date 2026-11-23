"""A new GENBAS with j-adapted (X2CAMF + Gaunt, configuration average, Gaussian nucleus) sets
for Th, U, O and F.

O, F   Dunning's cc-pVnZ and aug-cc-pVnZ primitives, contraction pattern and correlating
       functions unchanged, plus ONE tight p (optimised on the 2p spin-orbit splitting) that
       enters the contracted 2p only.  Names PVnZ-TP-SOG, APVnZ-TP-SOG.
Th, U  Dyall's v2z / v3z / v4z primitives and correlating functions; the occupied atomic
       spinors are contracted, the primitives of Dyall's archive (5f block, valence set) are
       left uncontracted, and a contracted shell that the uncontracted primitives already span
       is dropped (smallest eigenvalue of the normalised overlap of contracted + shared
       functions of one kappa below THR).  Names DYVnZ-SOG.
Every entry is checked: smallest overlap eigenvalue per l and kappa, and the occupied spinor
energies in the contracted space against the uncontracted atom."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audit'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'so_info'))
import dirac_basis
from socutils.tools.basisopt import engine, contract, genbas
HERE = os.path.dirname(os.path.abspath(__file__))
D = '/home/xwang405/apps/DIRAC-23.0-Source/basis/'
THR = 1e-5
LN = 'spdfghik'
TIGHT_P = {'O': {'d': 117.3083, 't': 225.2700, 'q': 419.8531}, 'F': {'d': 149.4919, 't': 286.1936, 'q': 531.7950}}
# Dyall's archive, 5f block, set 1 (valence): uncontracted SCF primitives counted from the smallest exponent
KEEP_5F = {2: {0: [2, 4], 1: [2, 4], 2: [1, 3], 3: [1, 2, 3]},
           3: {0: [1, 3, 4, 6], 1: [2, 3, 5, 6], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4, 5]},
           4: {0: [1, 2, 4, 5, 6, 8], 1: [2, 3, 4, 6, 7, 8], 2: [1, 2, 3, 4, 5, 6], 3: [1, 2, 3, 4]}}

def min_eigs(entry):
    """{(l, two_j): smallest eigenvalue of the normalised overlap of contracted + shared functions}"""
    so, out = entry.so_info(), {}
    for l, (ex, c) in entry.shells.items():
        a, b = ex[:, None], ex[None, :]
        s = (2 * np.sqrt(a * b) / (a + b)) ** (l + 1.5)
        n = so.get(l, 0)
        blocks = [(2 * l + 1, list(range(c.shape[1])))] if (l == 0 or n == 0) else [
            (2 * l - 1, list(range(0, n)) + list(range(2 * n, c.shape[1]))), (2 * l + 1, list(range(n, c.shape[1])))]
        for tj, cols in blocks:
            cc = c[:, cols]; cc = cc / np.sqrt(np.einsum('pi,pq,qi->i', cc, s, cc))
            out[(l, tj)] = float(np.linalg.eigvalsh(cc.T @ s @ cc)[0])
    return out

def drop_redundant(entry, thr=THR):
    """Remove the outermost contracted shell of an l (both kappas; s: the single block) while
    the set is nearly dependent.  Returns the list of what was dropped."""
    dropped = []
    while True:
        bad = [(v, l) for (l, tj), v in min_eigs(entry).items() if v < thr]
        if not bad:
            break
        l = min(bad)[1]
        ex, c = entry.shells[l]
        so = entry.so_info()
        if l == 0:
            con = [j for j in range(c.shape[1]) if (abs(c[:, j]) > 1e-12).sum() > 1]
            if not con: break
            c = np.delete(c, con[-1], axis=1); dropped.append('%ds' % len(con))
        else:
            n = so.get(l, 0)
            if n == 0: break
            c = np.delete(c, [n - 1, 2 * n - 1], axis=1); dropped.append('%d%s' % (n + l, LN[l]))
            so[l] = n - 1
            parts = [(k, v) for k, v in sorted(so.items()) if v > 0]
            entry.comment = '%d  %s' % (len(parts), '  '.join('%d %d' % p for p in parts))
        entry.shells[l] = (ex, c)
    return dropped

def check(entry, el):
    rows = contract.contraction_error(entry, el, 'x2camf-g', gau_nuc=True)
    return max(abs(b - a) for *_, a, b in rows), rows

def report(entry, el, dropped):
    err, rows = check(entry, el)
    me = min_eigs(entry)
    so = entry.so_info()
    size = ''.join('%d%s' % (c.shape[1] - so.get(l, 0), LN[l]) for l, (x, c) in sorted(entry.shells.items()))
    prim = ''.join('%d%s' % (len(x), LN[l]) for l, (x, c) in sorted(entry.shells.items()))
    nfun = sum((2 * l + 1) * (c.shape[1] - so.get(l, 0)) for l, (x, c) in entry.shells.items())
    eps = {(n, l, tj): (a, b) for n, l, tj, a, b in rows}
    spl = ''
    for (n, l, tj), (a, b) in sorted(eps.items()):
        if tj == 2 * l + 1 and (n, l, 2 * l - 1) in eps and n == max(k[0] for k in eps if k[1] == l):
            d0 = a - eps[(n, l, 2 * l - 1)][0]; d1 = b - eps[(n, l, 2 * l - 1)][1]
            spl += '  %d%s splitting ratio %.5f' % (n, LN[l], d1 / d0)
    print('%-18s %s -> %s (%d functions)  line %r  dropped: %s' % (entry.name, prim, size, nfun, entry.comment if so else '(none)', ', '.join(dropped) or '-'))
    print('      smallest overlap eigenvalue %.1e (%s)   largest spinor-energy error %.1e Eh%s' % (
        min(me.values()), '%s%d/2' % (LN[min(me, key=me.get)[0]], min(me, key=me.get)[1]), err, spl), flush=True)

out = os.path.join(HERE, 'GENBAS.thuof')
if os.path.exists(out): os.remove(out)
os.makedirs(os.path.join(HERE, 'pyscf'), exist_ok=True)
py = {}
def emit(entry, el):
    genbas.append(out, entry)
    py.setdefault(entry.name.split(':')[1], {})[el] = genbas.to_pyscf(entry, kappa=True)

for el in ('O', 'F'):
    for z in 'dtq':
        for aug in ('', 'aug-'):
            prims, pattern = contract.pattern_from_pyscf(el, '%scc-pv%sz' % (aug, z))
            prims, pattern = contract.add_primitives(prims, pattern, {1: [TIGHT_P[el][z]]})
            e = contract.contract(el, prims, pattern, kind='sog', gau_nuc=True, name='%sPV%sZ-TP-SOG' % ('A' if aug else '', z.upper()))
            dropped = drop_redundant(e)
            report(e, el, dropped); emit(e, el)
for el, Z in (('Th', 90), ('U', 92)):
    for n, zl in ((2, 'D'), (3, 'T'), (4, 'Q')):
        d = dirac_basis.read(D + 'dyall.v%dz' % n)[Z]
        prims = {l: sorted(v, reverse=True) for l, v in d.items()}
        pattern = contract.pattern_occupied(el, prims, KEEP_5F[n], gau_nuc=True)
        e = contract.contract(el, prims, pattern, kind='sog', gau_nuc=True, name='DYV%sZ-SOG' % zl, conv=2e-8, max_iter=200)
        before = min(min_eigs(e).values())
        dropped = drop_redundant(e)
        print('%s:DYV%sZ-SOG  occupied shells contracted: %s; smallest eigenvalue before dropping %.1e' % (
            el.upper(), zl, {LN[l]: pattern[l][0] for l in sorted(pattern) if pattern[l][0]}, before))
        report(e, el, dropped); emit(e, el)
for name, d in py.items():
    genbas.write_pyscf(os.path.join(HERE, 'pyscf', name.lower().replace('-', '_') + '.py'), d,
                       header='%s: j-adapted, X2CAMF + Gaunt atomic contraction, Gaussian nucleus (draft of 2026-10-13)' % name)
print('written:', out)
