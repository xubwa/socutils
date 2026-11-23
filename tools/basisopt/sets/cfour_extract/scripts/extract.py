"""Extract the relativistic families of the CFOUR x2cso GENBAS into socutils basis modules (staging
directory ./modules; nothing is written into socutils yet).

* families: names ending in -SO, -SF, -X2C, _X2C, -DE4 (see inventory.py)
* one module per family, one variable per element, PySCF internal format; j-adapted (-SO) sets with
  kappa (columns split by the information line; an empty line is taken from
  so_info/proposed_lines.tsv)
* duplicated entries: the first occurrence (what CFOUR uses), unless it is malformed
* every entry is checked: the number of numbers in its body against its header (malformed entries
  are reported and not used), smallest eigenvalue of the normalised overlap per l (and kappa)
* module names: A... -> aug-, P/PC/PWC -> cc-p/cc-pc/cc-pwc, DY... -> dy-, suffix lower case,
  _X2C written as -x2c; where an element has both spellings the -X2C entry is used"""
import os, re, json, sys
import numpy as np
exec(open('inventory.py').read().split('fam = {}')[0])     # lines, idx, SUF, parse
from socutils.tools.basisopt import genbas
heads = sorted(i for occ in idx.values() for i in occ)
allheads = [i for i, ln in enumerate(lines) if ln and not ln[0].isspace() and re.match(r'^[A-Z][A-Z]?:\S+\s*$', ln)]
nexthead = {h: allheads[k + 1] if k + 1 < len(allheads) else len(lines) for k, h in enumerate(allheads)}
def wellformed(i):
    nl = int(lines[i + 3].split()[0])
    ls, nc, npr = ([int(x) for x in lines[i + 4 + k].split()[:nl]] for k in range(3))
    need = sum(p + p * c for p, c in zip(npr, nc))
    have = sum(len(l.split()) for l in lines[i + 7:nexthead[i]])
    return need == have, need, have
PROP = {}
for ln in open('../so_info/proposed_lines.tsv').read().splitlines()[1:]:
    f = ln.split('\t')
    PROP[f[0]] = f[1]
def so_split(comment):
    t = comment.split()
    if not t or not all(x.lstrip('-').isdigit() for x in t): return None
    n = int(t[0]); pairs = list(map(int, t[1:1 + 2 * n]))
    return dict(zip(pairs[0::2], pairs[1::2]))
def modname(name):
    m = SUF.match(name); base, suf = m.group(1), m.group(2)
    s = {'-SO': '-so', '-SF': '-sf', '-X2C': '-x2c', '_X2C': '-x2c', '-DE4': '-de4'}[suf]
    aug = base.startswith('A') and not base.startswith('ANO')
    b = base[1:] if aug else base
    if b.startswith('DY'):
        core = 'dy-' + b[2:].lower()
    else:
        mm = re.match(r'^(PWC|PC|P)V(.*)$', b)
        core = 'cc-%sv%s' % (mm.group(1).lower(), mm.group(2).lower()) if mm else b.lower()
    return ('aug-' if aug else '') + core + s, suf
def ovl(ex, l):
    a, b = ex[:, None], ex[None, :]
    return (2 * np.sqrt(a * b) / (a + b)) ** (l + 1.5)
def mineig(ex, c, l):
    s = ovl(ex, l); cc = c / np.sqrt(np.einsum('pi,pq,qi->i', c, s, c))
    return float(np.linalg.eigvalsh(cc.T @ s @ cc)[0])
mods, report, used = {}, [], {}
for (name, el), occ in sorted(idx.items()):
    mn, suf = modname(name)
    if suf == '_X2C' and (name[:-4] + '-X2C', el) in idx:
        report.append('SPELLING %s:%s also exists as %s; the -X2C entry is used' % (el, name, name[:-4] + '-X2C'))
        continue
    pick = None
    for k, i in enumerate(occ):
        ok, need, have = wellformed(i)
        if ok: pick = (k, i); break
        report.append('MALFORMED %s:%s occurrence %d (line %d): header asks for %d numbers, body has %d' % (el, name, k + 1, i + 1, need, have))
    if pick is None:
        report.append('SKIPPED %s:%s - no well-formed occurrence' % (el, name)); continue
    k, i = pick
    comment, shells = parse(i)
    bad = [l for l, (ex, c) in shells.items() if not np.all(np.isfinite(ex)) or np.any(ex <= 0) or not np.all(np.isfinite(c))]
    if bad:
        report.append('MALFORMED %s:%s occurrence %d (line %d): zero/invalid exponents for l = %s (body does not match its header layout); not extracted' % (el, name, k + 1, i + 1, bad)); continue
    elsym = el.capitalize()
    eigs = []
    info = comment
    if suf == '-SO':
        so = so_split(comment)
        if so is None:
            line = PROP.get('%s:%s' % (el, name))
            if line is None:
                report.append('SKIPPED %s:%s - empty information line, no proposed line' % (el, name)); continue
            so = so_split(line); info = line
            report.append('INFO-LINE %s:%s empty in GENBAS, used %r' % (el, name, line))
        for l, (ex, c) in shells.items():
            n = so.get(l, 0)
            if l == 0 or n == 0:
                eigs.append(mineig(ex, c, l))
            else:
                for cols in (list(range(0, n)), list(range(n, 2 * n))):
                    eigs.append(mineig(ex, c[:, cols + list(range(2 * n, c.shape[1]))], l))
    else:
        eigs = [mineig(ex, c, l) for l, (ex, c) in shells.items()]
    ent = genbas.Entry('%s:%s' % (el, name), info, {l: shells[l] for l in sorted(shells)})
    basis = genbas.to_pyscf(ent, kappa=(suf == '-SO'))
    mods.setdefault(mn, {})[elsym] = basis
    used['%s:%s' % (el, name)] = {'module': mn, 'occurrence': k + 1, 'line': i + 1, 'min_overlap_eig': min(eigs), 'comment': comment}
    if min(eigs) < 1e-6:
        report.append('NEAR-DEPENDENT %s:%s smallest overlap eigenvalue %.1e' % (el, name, min(eigs)))
os.makedirs('modules', exist_ok=True)
for mn, d in sorted(mods.items()):
    srcs = sorted({k.split(':')[1] for k, v in used.items() if v['module'] == mn})
    genbas.write_pyscf(os.path.join('modules', mn + '.py'), d,
                       header='%s: extracted from the CFOUR x2cso GENBAS (entries %s), 2026-11-05' % (mn, ', '.join(srcs)))
json.dump(used, open('extracted.json', 'w'), indent=1)
open('report.txt', 'w').write('\n'.join(report) + '\n')
print('%d modules, %d entries; report lines %d' % (len(mods), len(used), len(report)))
for r in report: print(r)
