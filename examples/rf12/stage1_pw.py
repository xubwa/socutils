#!/usr/bin/env python
'''
Stage 1: partial-wave increments of the second-order (1/Z) pair energy of
1s1/2^2 He-like ions, Dirac-Coulomb no-pair vs Schroedinger, and the
partial-wave amplitudes of the first-order pair function at coalescence.

    python stage1_pw.py --Z 80 --backend quad --lmax 30 --out z80_quad.json
    python stage1_pw.py --cprime 1e4 ...      # scaled speed of light c/Z
    python stage1_pw.py --nr ...              # Schroedinger reference

Results (strings, full precision) go to a JSON file; analysis in
stage1_analyze.py.
'''
import argparse
import json
import time
from socutils.rf12 import pairpw, bspline, angular
from socutils.rf12 import xp as X

C_CODATA2018 = '137.035999084'

GRIDS = {
    # piecewise geometric: rmin..0.08 (12%), 0.08..8 (78%), 8..rmax
    'fine': dict(nb=300, k=12, rmin=1e-3, rmax=40, nq=22, segments=[(0.08, 0.12), (8., 0.78)]),
    'xfine': dict(nb=360, k=12, rmin=1e-3, rmax=40, nq=22, segments=[(0.08, 0.12), (8., 0.78)]),
    'med':  dict(nb=250, k=10, rmin=1e-3, rmax=40, nq=20, segments=[(0.08, 0.12), (8., 0.78)]),
    'coarse': dict(nb=200, k=10, rmin=1e-3, rmax=40, nq=20, segments=[(0.08, 0.12), (8., 0.78)]),
    'test': dict(nb=60, k=8, rmin=1e-4, rmax=40, nq=16),
}
RCOAL = ['0.25', '0.5', '1', '2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Z', type=int)
    ap.add_argument('--cprime', type=str)
    ap.add_argument('--nr', action='store_true')
    ap.add_argument('--backend', default='double')
    ap.add_argument('--grid', default='fine')
    ap.add_argument('--lmin', type=int, default=0)
    ap.add_argument('--lmax', type=int, default=25)
    ap.add_argument('--out', required=True)
    ap.add_argument('--channels', action='store_true',
                    help='also split E2 into LL, SS, LS component channels')
    a = ap.parse_args()

    grid = bspline.Grid(prec=200, **GRIDS[a.grid])
    ctx = grid.ctx
    if a.nr:
        c = None
    elif a.cprime:
        c = ctx.mpf(a.cprime)
    else:
        c = ctx.mpf(C_CODATA2018) / a.Z
    xp = X.Quad() if a.backend == 'quad' else X.Double()
    sp = pairpw.Space(grid, xp, c)
    t0 = time.time()
    ref = pairpw.KappaStates(sp, -1)
    # anchor: the lowest positive-energy kappa=-1 state must be the exact
    # point-nucleus 1s (catches variational collapse of the radial basis)
    e1s = xp.to_mp(ref.e[0:1], ctx)[0]
    exact = -ctx.mpf(1) / 2 if c is None else c * c * (ctx.sqrt(1 - 1 / c ** 2) - 1)
    print('1s: %s  exact %s  diff %.2e' % (mpmath_str(e1s, 20), mpmath_str(exact, 20), float(e1s - exact)))
    if abs(e1s - exact) > 1e-6:
        raise RuntimeError('radial basis does not reproduce the 1s state (collapse?)')
    Pa, Qa = ref.at(RCOAL)
    tostr = lambda v: mpmath_str(v)
    out = dict(Z=a.Z, cprime=None if c is None else tostr(c), backend=a.backend,
               grid=GRIDS[a.grid], rcoal=RCOAL, e1s=tostr(xp.to_mp(ref.e[0:1], ctx)[0]),
               Pa=[tostr(v) for v in xp.to_mp(Pa[0], ctx)],
               Qa=None if Qa is None else [tostr(v) for v in xp.to_mp(Qa[0], ctx)],
               nneg=getattr(ref, 'nneg', 0), kappas=[])
    for l in range(a.lmin, a.lmax + 1):
        kaps = [-1] if l == 0 else ([-l - 1] if c is None else [l, -l - 1])
        for kap in kaps:
            t = time.time()
            r, _ = pairpw.pair_kappa(sp, ref, kap, rcoal=[ctx.mpf(x) for x in RCOAL], channels=a.channels)
            out['kappas'].append(dict(kappa=kap, l=l, e2=tostr(r.e2), nstates=r.nstates,
                                      e2ch={k: tostr(v) for k, v in r.e2ch.items()},
                                      coal={k: [tostr(v) for v in vs] for k, vs in r.coal.items()}))
            print('l=%2d kappa=%3d  E2=%s  (%.1fs)' % (l, kap, mpmath_str(r.e2, 25), time.time() - t), flush=True)
            with open(a.out, 'w') as f:
                json.dump(out, f, indent=1)
    print('total %.1fs' % (time.time() - t0))


def mpmath_str(v, n=40):
    import mpmath
    return mpmath.nstr(v, n, strip_zeros=False) if not isinstance(v, str) else v


if __name__ == '__main__':
    main()
