"""Core-valence companions O:PWCV{T,Q}Z-X2C-TP (request of o2 zfs run coordination, 2026-10-20;
user: "就是在 QZ 上做 core valence 呗"): Dunning cc-pwCVXZ primitives and contraction pattern,
spin-free X2C-1e recontraction, plus the ccJ-pVXZ tight p, exactly as the PVXZ-X2C-TP sets below.

O sets for the O2 zero-field-splitting basis-set extrapolation (request of the o2 zfs run
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
out = os.path.join(HERE, 'GENBAS.o_x2c_cv')
if os.path.exists(out): os.remove(out)
py = {}
for z in 'tq':
    # Dunning's construction of cc-pwCVXZ: the cc-pVXZ contracted set plus the core-valence
    # functions as extra uncontracted primitives.  The contraction is that of PVXZ-X2C-TP (atomic
    # SCF in the cc-pVXZ primitives + tight p); an SCF in all pwCV primitives does not converge
    # at QZ (s 71.21 next to 69.32, ratio 1.03).
    vp, vpat = contract.pattern_from_pyscf('O', 'cc-pv%sz' % z)
    cp, cpat = contract.pattern_from_pyscf('O', 'cc-pwcv%sz' % z)
    vp, vpat = contract.add_primitives(vp, vpat, {1: [TIGHT[z]]})
    nm = 'PWCV%sZ-X2C-TP' % z.upper()
    e = contract.contract('O', vp, vpat, kind='sf', gau_nuc=True, name=nm)
    added = {}
    for l, exps in cp.items():
        new_e = [x for x in exps if all(abs(x / y - 1) > 1e-6 for y in vp.get(l, []))]
        if not new_e:
            continue
        added[l] = new_e
        if l in e.shells:
            ex, c = e.shells[l]
            k = len(new_e)
            c = np.hstack([np.vstack([c, np.zeros((k, c.shape[1]))]), np.vstack([np.zeros((len(ex), k)), np.eye(k)])])
            e.shells[l] = (np.concatenate([ex, new_e]), c)
        else:
            e.shells[l] = (np.array(new_e), np.eye(len(new_e)))
    so = e.so_info()
    nfun = sum((2 * l + 1) * (c.shape[1] - so.get(l, 0)) for l, (x, c) in e.shells.items())
    size = ''.join('%d%s' % (c.shape[1], LN[l]) for l, (x, c) in sorted(e.shells.items()))
    prim = ''.join('%d%s' % (len(x), LN[l]) for l, (x, c) in sorted(e.shells.items()))
    print('O:%-14s %-14s -> %-12s %3d functions (%3d spinors)  smallest overlap eigenvalue %.1e  core-valence functions added: %s' % (
        nm, prim, size, nfun, 2 * nfun, min(min_eigs(e).values()),
        '; '.join('%s %s' % (LN[l], ' '.join('%.4g' % x for x in v)) for l, v in sorted(added.items()))), flush=True)
    e.name = 'O:' + nm
    genbas.append(out, e)
    py[nm] = {'O': genbas.to_pyscf(e)}
for nm, d in py.items():
    genbas.write_pyscf(os.path.join(HERE, 'pyscf', nm.lower().replace('-', '_') + '.py'), d,
                       header='O:%s: (aug-)cc-pVXZ primitives + the tight p of ccJ-pVXZ in the contracted 2p; %s; Gaussian nucleus (2026-10-19)' % (
                           nm, 'spin-free X2C-1e contraction of the cc-pVXZ part, core-valence functions uncontracted'))
print('written:', out)
