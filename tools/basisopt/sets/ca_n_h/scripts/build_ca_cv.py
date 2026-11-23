"""Ca aug-cc-pwCV{T,Q}Z X2C sets (sf and sog) by Peterson's device on top of the valence sets
of build_ca_val.py.  Pins = the core-correlating exponents Hill & Peterson add in
cc-pwCV{T,Q}Z-X2C.
  s, p (occupied): each pin replaces the nearest HF exponent; the HF exponents within
     [min(pin)/4, 4 max(pin)] (pins and valence free exponents excluded) are re-optimised on
     the SF-X2C-1e atomic HF energy with neighbouring ratios >= 1.6 (penalty); pins are free
     functions.
  d, f, g (unoccupied, fully uncontracted): pins added; valence exponents within a ratio of
     1.6 of a pin removed."""
import sys, numpy as np, scipy.optimize
from socutils.tools.basisopt import contract, genbas, engine
exec(open('build_ca_val.py').read().split("for z in 'TQ5':")[0])
RATIO = 1.6
import os
DROP = os.environ.get('DROP', '1') == '1'
WIN = float(os.environ.get('WIN', '4'))
PINS = {'T': {0: [1.627, 0.2898], 1: [1.712, 0.2805], 2: [4.404, 1.324], 3: [0.6292]},
        'Q': {0: [5.463, 1.73, 0.7772], 1: [6.858, 1.956, 0.828], 2: [2.139, 1.426, 0.5333],
              3: [1.991, 0.7507, 0.3792], 4: [0.6106]}}
def energy(p):
    r = engine.atom('Ca', p, 'sfx2c1e', gau_nuc=True, max_iter=1000)
    return r.e_scf if r.converged else 0.0
def make(z):
    prims, pat, _ = build(z)
    free = {l: [prims[l][i] for i in pat[l][1]] for l in prims}
    new, opt = {}, {}
    for l, ex in prims.items():
        pins = PINS[z].get(l, [])
        if l >= 2:
            new[l] = sorted([e for e in ex if all(max(e / q, q / e) >= RATIO for q in pins)] + pins, reverse=True)
            continue
        keep = list(ex)
        for q in (pins if DROP else []):
            cand = [e for e in keep if e not in free[l]]
            keep.remove(min(cand, key=lambda e: abs(np.log(e / q))))
        new[l] = sorted(keep + pins, reverse=True)
        lo, hi = (min(pins) / WIN, WIN * max(pins)) if pins else (0, 0)
        opt[l] = [e for e in new[l] if lo <= e <= hi and e not in pins and e not in free[l]]
    e0 = energy(new)
    keys = [(l, e) for l in opt for e in opt[l]]
    def assemble(x):
        p = {l: list(v) for l, v in new.items()}
        for (l, e), v in zip(keys, x):
            p[l][p[l].index(e)] = np.exp(v)
        return {l: sorted(v, reverse=True) for l, v in p.items()}
    def f(x):
        p = assemble(x); pen = 0.0
        for l in (0, 1):
            r = np.array(p[l][:-1]) / np.array(p[l][1:])
            pen += np.sum(np.clip(np.log(RATIO) - np.log(r), 0, None) ** 2) * 10.0
        return energy(p) + pen
    x0 = np.log([e for l, e in keys])
    res = scipy.optimize.minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-9, 'maxiter': 3000})
    p = assemble(res.x)
    for (l, e), v in zip(keys, res.x):
        print('  %s: %.5g -> %.5g' % ('spdfg'[l], e, np.exp(v)))
    eref = energy(prims)
    print('  E_SCF valence prims %.8f | pins inserted %.8f | re-optimised %.8f  (dE vs valence %.4f mEh)' % (eref, e0, energy(p), (energy(p) - eref) * 1e3))
    cvfree = {l: sorted(set(free[l]) | set(PINS[z].get(l, [])), reverse=True) for l in (0, 1)}
    pat2 = {0: (4, sorted(p[0].index(e) for e in cvfree[0])), 1: (2, sorted(p[1].index(e) for e in cvfree[1]))}
    for l in p:
        if l >= 2: pat2[l] = (0, list(range(len(p[l]))))
    for l in sorted(p):
        r = np.array(p[l][:-1]) / np.array(p[l][1:])
        print('  %s %2d prims, min ratio %.2f, free %s' % ('spdfg'[l], len(p[l]), r.min() if len(r) else 0,
              ' '.join('%.4g' % p[l][i] for i in pat2[l][1])))
    return p, pat2
if os.environ.get('TRY'):
    make(os.environ['TRY']); sys.exit()
for z in sys.argv[1:] or 'TQ':
    print('== Ca aug-cc-pwCV%sZ' % z, flush=True)
    p, pat2 = make(z)
    for kind in ('sf', 'sog'):
        e = contract.contract('Ca', p, pat2, kind=kind, aoc=True, gau_nuc=True, max_iter=1000, name='APWCV%sZ-%s' % (z, kind.upper()))
        genbas.append('GENBAS.canh_cv', e)
        print('  %-4s E_SCF %.8f  functions %s' % (kind, e.scf.e_scf, ' '.join('%d%s' % (e.shells[l][1].shape[1], 'spdfgh'[l]) for l in sorted(e.shells))), flush=True)
