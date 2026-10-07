"""Optimisation of added (tight) primitive exponents with the X2CAMF atomic SCF as engine.

Objectives (smaller is better):
  so      sum_i w_i (D_i / D_i^ref - 1)^2 over the target shells, D = eps(n l j=l+1/2) -
          eps(n l j=l-1/2); the reference is the same basis with the tight region of the
          optimised angular momenta saturated by an even-tempered series.
  energy  the atomic SCF energy.

Command line, e.g.
  python -m socutils.tools.basisopt.optimize O aug-cc-pvtz --add p:1 --ham dcb
"""
import argparse

import numpy as np
from scipy.optimize import minimize

from . import engine

CM = 219474.6313705
LNAMES = 'spdfghi'


def saturated(prims, ls, ratio=2.0, top=None, z=1):
    """Copy of prims with an even-tempered tight series (given ratio) added for each l in ls,
    up to `top` (default 2e3 Z^2 for l >= 1, 1e5 Z^2 for s)."""
    out = {l: list(v) for l, v in prims.items()}
    for l in ls:
        limit = top if top else (1e5 if l == 0 else 2e3) * z * z
        x = max(out[l])
        while x * ratio < limit:
            x *= ratio
            out[l].insert(0, x)
    return out


def geometric_start(prims, add):
    """Continuation of each series with its last ratio: {l: [new exponents, descending]}."""
    start = {}
    for l, n in add.items():
        e = sorted(prims[l], reverse=True)
        ratio = e[0] / e[1]
        start[l] = [e[0] * ratio ** (k + 1) for k in range(n)][::-1]
    return start


def valence_targets(result):
    """[(n, l)] of the highest occupied shell of each l >= 1."""
    top = {}
    for n, l, two_j, f, e in result.occupied():
        if l >= 1:
            top[l] = max(top.get(l, 0), n)
    return [(n, l) for l, n in sorted(top.items())]


def all_targets(result):
    return sorted({(n, l) for n, l, two_j, f, e in result.occupied() if l >= 1})


class TightOptimizer:
    """Optimise `add` = {l: number of new primitives} on top of `prims` = {l: exponents}."""

    def __init__(self, element, prims, add, hamiltonian='dcb', objective='so', targets='valence',
                 weights=None, gau_nuc=True, sat_ratio=2.0, verbose=1, **atom_kwargs):
        self.element = element
        self.prims = {l: sorted(v, reverse=True) for l, v in prims.items()}
        self.add = {l: n for l, n in add.items() if n > 0}
        self.objective = objective
        self.verbose = verbose
        self.kw = dict(hamiltonian=hamiltonian, gau_nuc=gau_nuc, **atom_kwargs)
        self.base = self.run(self.prims)
        if targets == 'valence':
            targets = valence_targets(self.base)
        elif targets == 'all':
            targets = all_targets(self.base)
        self.targets = list(targets)
        self.weights = list(weights) if weights else [1.0] * len(self.targets)
        z = engine.charge(element)
        self.reference = self.run(saturated(self.prims, self.add, sat_ratio, z=z))
        self.so_ref = [self.reference.so_splitting(n, l) for n, l in self.targets]
        self.neval = 0

    def run(self, prims):
        return engine.atom(self.element, prims, **self.kw)

    def with_added(self, new):
        return {l: sorted(list(v) + list(new.get(l, ())), reverse=True)
                for l, v in self.prims.items()}

    def unpack(self, x):
        new, i = {}, 0
        for l, n in sorted(self.add.items()):
            new[l] = sorted(np.exp(x[i:i + n]), reverse=True)
            i += n
        return new

    def so_ratios(self, result):
        return [result.so_splitting(n, l) / ref for (n, l), ref in zip(self.targets, self.so_ref)]

    def value(self, result):
        if self.objective == 'energy':
            return result.e_scf
        return sum(w * (r - 1) ** 2 for w, r in zip(self.weights, self.so_ratios(result)))

    def __call__(self, x):
        new = self.unpack(x)
        # keep the new primitives apart from each other and from the existing ones
        for l, e in new.items():
            allexp = sorted(list(self.prims[l]) + list(e), reverse=True)
            if min(a / b for a, b in zip(allexp, allexp[1:])) < 1.3:
                return 1e3
        self.neval += 1
        try:
            return self.value(self.run(self.with_added(new)))
        except Exception:
            return 1e3

    def kernel(self, start=None, xtol=3e-3):
        start = start or geometric_start(self.prims, self.add)
        x0 = np.log(np.concatenate([sorted(start[l], reverse=True) for l in sorted(self.add)]))
        best = None
        # the surface has several minima when two functions are added; try shifted starts
        for shift in ((0.0, 0.4, -0.4) if len(x0) > 1 else (0.0,)):
            res = minimize(self, x0 + shift, method='Nelder-Mead',
                           options=dict(xatol=xtol, fatol=1e-10 if self.objective == "so" else 1e-9,
                                        maxiter=150 * len(x0)))
            if best is None or res.fun < best.fun:
                best = res
        self.new = self.unpack(best.x)
        self.result = self.run(self.with_added(self.new))
        if self.verbose:
            print(self.report(start))
        return self.new

    def report(self, start=None):
        lines = ['%s: %s, objective %s, %d SCF runs' % (
            self.element, self.kw['hamiltonian'], self.objective, self.neval)]
        rows = [('base', self.base)]
        if start:
            rows.append(('geometric', self.run(self.with_added(start))))
        rows += [('optimised', self.result), ('saturated', self.reference)]
        head = '  %-10s %16s' % ('', 'E(SCF)')
        for n, l in self.targets:
            head += '  D(%d%s)/cm-1   ratio' % (n, LNAMES[l])
        lines.append(head)
        for name, r in rows:
            line = '  %-10s %16.8f' % (name, r.e_scf)
            for (n, l), ratio in zip(self.targets, self.so_ratios(r)):
                line += '  %11.4f %7.5f' % (r.so_splitting(n, l) * CM, ratio)
            lines.append(line)
        for l in sorted(self.add):
            lines.append('  new %s exponents: %s   (largest before: %.6g; geometric: %s)' % (
                LNAMES[l], ' '.join('%.6g' % e for e in self.new[l]), self.prims[l][0],
                ' '.join('%.6g' % e for e in (start or {}).get(l, ()))))
        return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('element')
    ap.add_argument('basis', help='PySCF basis-set name; its primitives are the fixed part')
    ap.add_argument('--add', required=True, help='new primitives per l, e.g. p:2,d:1')
    ap.add_argument('--ham', default='dcb', choices=sorted(engine.HAMILTONIANS))
    ap.add_argument('--objective', default='so', choices=('so', 'energy'))
    ap.add_argument('--targets', default='valence',
                    help="'valence', 'all', or shells such as 2p,3p,3d")
    ap.add_argument('--weights', default=None, help='weights of the targets, e.g. 1,4')
    ap.add_argument('--point-nucleus', action='store_true')
    args = ap.parse_args()
    add = {LNAMES.index(i.split(':')[0]): int(i.split(':')[1]) for i in args.add.split(',')}
    targets = args.targets
    if targets not in ('valence', 'all'):
        targets = [(int(t[:-1]), LNAMES.index(t[-1])) for t in targets.split(',')]
    weights = [float(w) for w in args.weights.split(',')] if args.weights else None
    opt = TightOptimizer(args.element, engine.prims_from_pyscf(args.element, args.basis), add,
                         hamiltonian=args.ham, objective=args.objective, targets=targets,
                         weights=weights, gau_nuc=not args.point_nucleus)
    opt.kernel()


if __name__ == '__main__':
    main()
