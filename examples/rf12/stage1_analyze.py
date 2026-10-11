#!/usr/bin/env python
'''
Analysis of stage1_pw.py output.

    python stage1_analyze.py run.json [--ref nr.json]

Prints, per partial wave l (large-component orbital angular momentum,
kappa = l and -(l+1) summed):
  dE_l                    second-order increment
  S_l = -(256/45)(l+1/2)^4 dE_l         -> 1 - (5/4)(l+1/2)^-2 (Schwartz, NR)
  p_l = local exponent  -dln|dE|/dln(l+1/2) between l and l+1
  with --ref (NR on the same grid):
  d_l = dE_l(rel) - dE_l(NR), its local exponent and (l+1/2)^2 d_l / (Z alpha)^2
        -> 3/64 if the Kutzelnigg-2008 coefficient (with psi0) holds
  coalescence: s_l = a_l / (Phi0(r,r) * r * c_l),  a_l = LL spin-singlet
        l-component of the first-order pair function at r1 = r2 = r z,
        c_l = Legendre coefficient of r12/r; s_l -> 1/2 is the Kato cusp.
'''
import sys
import json
import argparse
import mpmath
from socutils.rf12 import angular as A
from socutils.rf12 import relmodel

mp = mpmath.mp
mp.dps = 40


def load(fn):
    d = json.load(open(fn))
    by_l = {}
    for k in d['kappas']:
        by_l.setdefault(k['l'], []).append(k)
    return d, by_l


def increments(by_l):
    return {l: mp.fsum(mp.mpf(k['e2']) for k in ks) for l, ks in sorted(by_l.items())}


def coal(d, by_l, comp, ir, part):
    '''l-components of u^{comp}(r z, r z) spin part (singlet/triplet0),
    relative units (common factor 1/(sqrt2 r^2) dropped)'''
    proj = A.singlet if part == 's' else A.triplet0
    out = {}
    for l, ks in sorted(by_l.items()):
        tot = 0
        for k in ks:
            if comp not in k['coal']:
                continue
            G = A.coalescence_tensor(k['kappa'], comp)
            g = proj(G)
            if d['Qa'] is None and l > 0:
                # Schroedinger run: one radial set per l carries both
                # j = l +- 1/2 spin-angular parts
                g += proj(A.coalescence_tensor(l, comp))
            tot += mp.mpf(k['coal'][comp][ir]) * g
        out[l] = tot
    return out


def ref_value(d, comp, ir, part):
    proj = A.singlet if part == 's' else A.triplet0
    X = {'L': d['Pa'], 'S': d['Qa']}
    if X[comp[0]] is None or X[comp[1]] is None:
        return 0
    G0 = A.reference_tensor(comp)
    return mp.mpf(X[comp[0]][ir]) * mp.mpf(X[comp[1]][ir]) * proj(G0)


def local_exp(f, l):
    if l + 1 not in f or f[l] == 0 or f[l + 1] == 0:
        return None
    return -mp.log(abs(f[l + 1] / f[l])) / mp.log((l + mp.mpf(1.5)) / (l + mp.mpf(0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--ref')
    a = ap.parse_args()
    d, by_l = load(a.run)
    dE = increments(by_l)
    zalpha2 = None if d['cprime'] is None else 1 / mp.mpf(d['cprime']) ** 2
    print('# %s  backend=%s  c/Z=%s  (Z alpha)^2=%s  e1s=%s' % (
        a.run, d['backend'], d['cprime'], mpmath.nstr(zalpha2, 8) if zalpha2 else '0', d['e1s'][:25]))
    print('# E2 summed to l=%d: %s' % (max(dE), mpmath.nstr(mp.fsum(dE.values()), 16)))
    ref = None
    if a.ref:
        dr, by_lr = load(a.ref)
        ref = increments(by_lr)
    print('%3s %24s %14s %10s' % ('l', 'dE_l', 'S_l', 'p_l') +
          ('' if ref is None else ' %14s %10s %14s' % ('d_l', 'p(d)', 'd(l+.5)^2/Za2')))
    dd = {l: dE[l] - ref[l] for l in dE if ref and l in ref}
    for l in dE:
        x = l + mp.mpf(0.5)
        S = -dE[l] * x ** 4 * 256 / 45
        p = local_exp(dE, l)
        line = '%3d %24s %14s %10s' % (l, mpmath.nstr(dE[l], 16), mpmath.nstr(S, 10),
                                       mpmath.nstr(p, 6) if p else '')
        if ref and l in ref:
            pd = local_exp(dd, l)
            line += ' %14s %10s %14s' % (mpmath.nstr(dd[l], 8), mpmath.nstr(pd, 6) if pd else '',
                                         mpmath.nstr(dd[l] * x ** 2 / zalpha2, 8) if zalpha2 else '')
        print(line)
    # coalescence
    for ir, r in enumerate(d['rcoal']):
        rr = mp.mpf(r)
        print('# coalescence r1 = r2 = %s' % r)
        comps = ['LL'] + (['SS', 'LS'] if d['Qa'] is not None else [])
        hdr = '%3s %12s %10s' % ('l', 's_l(LL)', 'a/model')
        cols = {}
        for comp in comps:
            for part in 'st':
                c = coal(d, by_l, comp, ir, part)
                if max(abs(v) for v in c.values()) > 1e-30:
                    cols[comp + part] = c
                    hdr += ' %14s %8s' % ('a(' + comp + part + ')', 'p')
        print(hdr)
        phi0 = ref_value(d, 'LL', ir, 's')
        for l in sorted(by_l):
            s = (cols['LLs'][l] / (phi0 * rr * A.legendre_r12(l))).real
            mdl = ''
            if l > 0:
                cp = None if d['cprime'] is None else float(mp.mpf(d['cprime']))
                mdl = mpmath.nstr((cols['LLs'][l] / phi0).real / relmodel.legendre_coeff(l, float(rr), cp), 6)
            line = '%3d %12s %10s' % (l, mpmath.nstr(s, 8), mdl)
            for name, c in cols.items():
                p = local_exp(c, l)
                v = c[l] / phi0
                v = v.real if abs(v.imag) <= abs(v.real) else v.imag
                line += ' %14s %8s' % (mpmath.nstr(v, 7), mpmath.nstr(p, 5) if p else '')
            print(line)
        for comp in comps:
            for part in 'st':
                v = ref_value(d, comp, ir, part)
                if v != 0:
                    print('#   Phi0 %s%s / Phi0 LLs = %s' % (comp, part, mpmath.nstr(v / phi0, 8)))


if __name__ == '__main__':
    main()
