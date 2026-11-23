"""Exponent patterns of the cc sets for Ca, N, H (BSE): per-l exponents, contraction counts,
smallest neighbour ratio, and the minimum overlap eigenvalue of the uncontracted shell."""
import numpy as np, basis_set_exchange as bse
from math import gamma
L = 'spdfghi'
def mineig(ex, l):
    ex = np.array(ex); a = ex[:, None] + ex[None, :]
    S = (2 * np.sqrt(np.outer(ex, ex)) / a) ** (l + 1.5)
    return np.linalg.eigvalsh(S)[0]
def show(name, el):
    try: b = bse.get_basis(name, elements=[el], uncontract_general=False)
    except Exception as e: print('%-18s %-2s MISSING' % (name, el)); return
    sh = {}
    for s in b['elements'][str(bse.lut.element_Z_from_sym(el))]['electron_shells']:
        for i, l in enumerate(s['angular_momentum']):
            ex = [float(x) for x in s['exponents']]
            co = s['coefficients'] if len(s['angular_momentum']) == 1 else [s['coefficients'][i]]
            d = sh.setdefault(l, [set(), 0, 0])
            d[0].update(ex)
            for c in co:
                nz = sum(abs(float(x)) > 0 for x in c)
                if nz == 1: d[2] += 1
                else: d[1] += 1
    print('== %s %s' % (name, el))
    for l in sorted(sh):
        ex = sorted(sh[l][0], reverse=True)
        r = [ex[i] / ex[i + 1] for i in range(len(ex) - 1)]
        print('  %s nprim %2d ncontr %d nfree %d  minratio %5s  mineig %.1e | %s' % (
            L[l], len(ex), sh[l][1], sh[l][2], '%.2f' % min(r) if r else '-', mineig(ex, l),
            ' '.join('%.4g' % e for e in (ex if len(ex) <= 8 else ex[-8:]))))
import sys
for el, names in [('Ca', ['cc-pVTZ', 'cc-pVQZ', 'cc-pV5Z', 'cc-pVTZ-X2C', 'cc-pVQZ-X2C',
                          'aug-cc-pVTZ-X2C', 'aug-cc-pVQZ-X2C', 'cc-pCVTZ', 'cc-pCVQZ', 'cc-pCV5Z', 'cc-pwCVTZ-X2C']),
                  ('N', ['aug-cc-pVTZ', 'aug-cc-pVQZ', 'aug-cc-pV5Z']),
                  ('H', ['aug-cc-pVTZ', 'aug-cc-pVQZ', 'aug-cc-pV5Z'])]:
    for n in names: show(n, el)
