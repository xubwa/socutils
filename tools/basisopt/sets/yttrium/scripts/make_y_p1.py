"""Y:DYVDZ-SF-P1 and Y:DYVTZ-SF-P1 (requested by the Eu:YSO session, user decisions A3/B1, 2026-11-10).

Primitives: dyall.v2z (DZ, 21s14p10d2f) and dyall.v3z (TZ, 28s20p13d4f2g without the 4s4p
outer-core correlating f 1.115 -> 28s20p13d3f2g).  Contraction: spin-free X2CAMF atomic SCF,
Gaussian nucleus, average of 4d1 5s2; one contracted function per occupied shell (1s-5s, 2p-4p,
3d-4d).  Uncontracted primitives (counted from the smallest exponent): Dyall's 4d-block valence
rules (DZ: 2nd s, 2nd p, 1st+2nd d; TZ: 1st+3rd s, 2nd+3rd p, 1st-3rd d) plus the most diffuse p
("-P1": the 4d1 5s2 contraction has no 5p, and without it the 5s->5p polarization of Y+ is lost,
alpha -11 % at DZ).  f, g (unoccupied) uncontracted."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'audit'))
import dirac_basis
from socutils.tools.basisopt import contract, genbas
B = '/home/xwang405/apps/DIRAC-23.0-Source/basis/'
SETS = {'DZ': ('dyall.v2z', {0: [2], 1: [1, 2], 2: [1, 2]}, []),
        'TZ': ('dyall.v3z', {0: [1, 3], 1: [1, 2, 3], 2: [1, 2, 3]}, [1.1153443])}
out = os.path.join(HERE, 'GENBAS.y')
have = [l.strip() for l in open(out) if l.startswith('Y:')]
for z, (f, keep, drop) in SETS.items():
    prims = {l: sorted(v, reverse=True) for l, v in dirac_basis.read(B + f)[39].items()}
    if drop:
        prims[3] = [e for e in prims[3] if all(abs(e / d - 1) > 1e-6 for d in drop)]
    pattern = contract.pattern_occupied('Y', prims, keep, gau_nuc=True)
    name = 'DYV%s-SF-P1' % z
    e = contract.contract('Y', prims, pattern, kind='sf', gau_nuc=True, name=name, conv=1e-9, max_iter=300)
    nfun = sum((2 * l + 1) * c.shape[1] for l, (x, c) in e.shells.items())
    print('== Y:%s  %s -> %s  (%d functions)  E_SCF(4d1 5s2 av.) %.8f' % (name,
          ''.join('%d%s' % (len(x), 'spdfg'[l]) for l, (x, c) in sorted(e.shells.items())),
          ''.join('%d%s' % (c.shape[1], 'spdfg'[l]) for l, (x, c) in sorted(e.shells.items())), nfun, e.scf.e_scf))
    for l, (x, c) in sorted(e.shells.items()):
        free = [x[i] for i in pattern[l][1]] if pattern.get(l, (0, []))[0] else []
        print('   %s: free %s' % ('spdfg'[l], ' '.join('%.5g' % v for v in free) if free else '(all)' if not pattern.get(l, (0,))[0] else '-'))
    eu = __import__('socutils.tools.basisopt.engine', fromlist=['atom']).atom('Y', prims, 'sfx2c1e', gau_nuc=True, max_iter=300)
    print('   E_SCF uncontracted (same SCF in primitives) %.8f' % eu.e_scf)
    seen = set()
    for n, l, tj, x0, x1 in contract.contraction_error(e, 'Y', 'sfx2c1e', gau_nuc=True):
        if (n, l) in seen: continue
        seen.add((n, l))
        print('   %d%s  uncontracted %12.6f  contracted %12.6f  diff %.1e' % (n, 'spdfg'[l], x0, x1, x1 - x0))
    e.name = 'Y:' + name
    if e.name not in have:
        genbas.append(out, e)
    genbas.write_pyscf(os.path.join(HERE, 'dyv%s_sf_p1.py' % z.lower()), {'Y': genbas.to_pyscf(e)},
                       header='Y:%s: %s, spin-free X2CAMF contraction (4d1 5s2 average), Gaussian nucleus; '
                              'Dyall valence rule + most diffuse p uncontracted. Source: y_sf/make_y_p1.py' % (name, f))
