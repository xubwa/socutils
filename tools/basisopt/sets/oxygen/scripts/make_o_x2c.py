"""O sets for the O2 zero-field-splitting basis-set extrapolation (request of the o2 zfs run
coordination session, user's words: "ccpvxz-x2c ... 在 2p 上加上那个我们认为对 so 重要的那个 function").

For X = D, T, Q, 5 and with / without aug:
  primitives      Dunning's (aug-)cc-pVXZ, unchanged, plus ONE tight p per level, the one of ccJ-pVXZ
                  (Benedikt et al.; 6.0 x the largest cc p): D 106.2, T 206.76, Q 380.52, 5 1173.0.
                  (Our exponents optimised on the 2p splitting, 117.31 / 225.27 / 419.85 / 1317.60,
                  give the same splitting to 5e-4; opt/ccj_vs_opt.out.  User chose ccJ's, 2026-10-19.)
  contraction     Dunning's pattern (contracted 1s, 2s, 2p; his uncontracted functions unchanged);
                  the tight p enters the contracted 2p only (no extra function), coefficients
                  recomputed from the atomic SCF of O (3P configuration average), Gaussian nucleus:
     -X2C-TP      spin-free X2C-1e (one contraction per l; the "cc-pVXZ-X2C" convention)
     -TP-SOG      j-adapted, X2CAMF with Gaunt (one contraction per kappa; information line "1  1 1")
Same rule at every level.  Checks: contraction error of the occupied spinor energies against the
uncontracted primitives, 2p splitting ratio (SOG), smallest overlap eigenvalue."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from socutils.tools.basisopt import contract, genbas
src = open(os.path.join(HERE, '..', 'thuof', 'make_thuof.py')).read()
exec(src[src.index('THR = 1e-5'):src.index('def check(')])          # min_eigs, drop_redundant, LN
TIGHT = {'d': 106.2, 't': 206.76, 'q': 380.52, '5': 1173.0}      # the tight p of ccJ-pVXZ (6.0 x the largest cc p); user's choice 2026-10-19
out = os.path.join(HERE, 'GENBAS.o_x2c')
if os.path.exists(out): os.remove(out)
py = {}
for z in 'dtq5':
    for aug in ('', 'aug-'):
        prims, pattern = contract.pattern_from_pyscf('O', '%scc-pv%sz' % (aug, z))
        prims, pattern = contract.add_primitives(prims, pattern, {1: [TIGHT[z]]})
        for kind, nm in (('sf', '%sPV%sZ-X2C-TP' % ('A' if aug else '', z.upper())), ('sog', '%sPV%sZ-TP-SOG' % ('A' if aug else '', z.upper()))):
            e = contract.contract('O', prims, pattern, kind=kind, gau_nuc=True, name=nm)
            dropped = drop_redundant(e)
            so = e.so_info()
            nfun = sum((2 * l + 1) * (c.shape[1] - so.get(l, 0)) for l, (x, c) in e.shells.items())
            ham = 'sfx2c1e' if kind == 'sf' else 'x2camf-g'
            rows = contract.contraction_error(e, 'O', ham, gau_nuc=True)
            err = max(abs(b - a) for *_, a, b in rows)
            eps = {(n, l, tj): (a, b) for n, l, tj, a, b in rows}
            spl = ''
            if (2, 1, 1) in eps and (2, 1, 3) in eps and kind != 'sf':
                d0 = eps[(2, 1, 3)][0] - eps[(2, 1, 1)][0]; d1 = eps[(2, 1, 3)][1] - eps[(2, 1, 1)][1]
                spl = '  2p splitting ratio %.5f' % (d1 / d0)
            size = ''.join('%d%s' % (c.shape[1] - so.get(l, 0), LN[l]) for l, (x, c) in sorted(e.shells.items()))
            prim = ''.join('%d%s' % (len(x), LN[l]) for l, (x, c) in sorted(e.shells.items()))
            print('O:%-14s %-14s -> %-12s %3d functions (%3d spinors)  smallest overlap eigenvalue %.1e  largest spinor-energy error %.1e Eh%s  dropped %s' % (
                nm, prim, size, nfun, 2 * nfun, min(min_eigs(e).values()), err, spl, ', '.join(dropped) or '-'), flush=True)
            e.name = 'O:' + nm
            genbas.append(out, e)
            py.setdefault(nm, {})['O'] = genbas.to_pyscf(e, kappa=(kind != 'sf'))
for nm, d in py.items():
    genbas.write_pyscf(os.path.join(HERE, 'pyscf', nm.lower().replace('-', '_') + '.py'), d,
                       header='O:%s: (aug-)cc-pVXZ primitives + the tight p of ccJ-pVXZ in the contracted 2p; %s; Gaussian nucleus (2026-10-19)' % (
                           nm, 'spin-free X2C-1e contraction' if 'X2C' in nm else 'j-adapted X2CAMF + Gaunt contraction'))
print('written:', out)
