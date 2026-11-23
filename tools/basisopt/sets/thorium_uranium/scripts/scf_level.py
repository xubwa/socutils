"""SCF error of Dyall's and Peterson's primitive sets for heavy atoms (4c DC, Gaussian
nucleus, configuration average), relative to Dyall's v4z primitives of the same atom."""
import sys, os
sys.path.insert(0, '../audit'); sys.path.insert(0, '../ccrepo')
import numpy as np, dirac_basis, ccrepo
from socutils.tools.basisopt import engine
D = '/home/xwang405/apps/DIRAC-23.0-Source/basis/'
NAME = {'Th': 'thorium', 'U': 'uranium', 'Rn': 'radon', 'Hg': 'mercury'}
LMAX = {'Th': 3, 'U': 3, 'Rn': 3, 'Hg': 3}
def pet(el, name):
    t = ccrepo.get(NAME[el], name)
    if t is None: return None
    out = {}
    for l in sorted(t):
        if l > LMAX[el]: continue
        e, cols = t[l]
        hf = sorted({e[i] for col in cols if sum(abs(c) > 1e-12 for c in col) > 1 for i, c in enumerate(col) if abs(c) > 1e-12}, reverse=True)
        if hf: out[l] = hf
    return out
for el, Z in (('Th', 90), ('U', 92), ('Rn', 86)):
    sets = {'dyall.v%dz' % n: {l: sorted(v, reverse=True) for l, v in dirac_basis.read(D + 'dyall.v%dz' % n)[Z].items() if l <= LMAX[el]} for n in (2, 3, 4)}
    for nm in ('cc-pVDZ-DK3', 'cc-pVTZ-DK3', 'cc-pVQZ-DK3'):
        p = pet(el, nm)
        if p: sets[nm] = p
    res = {}
    for nm, pr in sets.items():
        try:
            res[nm] = engine.atom(el, pr, 'dc', aoc=True, gau_nuc=True, conv=2e-8, max_iter=300).e_scf
        except Exception as ex:
            res[nm] = float('nan')
        print('%-2s %-12s %-16s E %.6f' % (el, nm, ''.join('%d%s' % (len(pr[l]), 'spdf'[l]) for l in sorted(pr)), res[nm]), flush=True)
    ref = res['dyall.v4z']
    print('   ' + '   '.join('%s %+.3f mEh' % (nm, 1e3 * (e - ref)) for nm, e in res.items()), flush=True)
