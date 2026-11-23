"""Primitives and contraction pattern of a BSE set: {l: exps}, {l: (ncontracted, [free indices])}."""
import numpy as np, basis_set_exchange as bse
def src(name, el='Ca'):
    b = bse.get_basis(name, elements=[el])['elements'][str(bse.lut.element_Z_from_sym(el))]
    cols = {}
    for s in b['electron_shells']:
        ex = [float(x) for x in s['exponents']]
        for i, l in enumerate(s['angular_momentum']):
            co = s['coefficients'] if len(s['angular_momentum']) == 1 else [s['coefficients'][i]]
            for c in co:
                cols.setdefault(l, []).append([e for e, x in zip(ex, c) if abs(float(x)) > 1e-12])
    prims, pat = {}, {}
    for l, cl in cols.items():
        exps = sorted({e for c in cl for e in c}, reverse=True)
        free = sorted({exps.index(c[0]) for c in cl if len(c) == 1})
        prims[l], pat[l] = exps, (sum(len(c) > 1 for c in cl), free)
    return prims, pat
if __name__ == '__main__':
    for n in ['cc-pVTZ-X2C', 'aug-cc-pVTZ-X2C', 'cc-pVQZ-X2C', 'aug-cc-pVQZ-X2C', 'cc-pwCVTZ-X2C', 'cc-pwCVQZ-X2C', 'aug-cc-pwCVTZ-X2C', 'aug-cc-pwCVQZ-X2C']:
        p, t = src(n)
        print('==', n)
        for l in sorted(p):
            print('  %s n=%2d contr=%d free=%s : %s' % ('spdfgh'[l], len(p[l]), t[l][0], t[l][1],
                  ' '.join('%.4g' % p[l][i] for i in t[l][1])))
