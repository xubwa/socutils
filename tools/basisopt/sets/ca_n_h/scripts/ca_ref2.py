"""Reference by refinement of cc-pVQZ-X2C: geometric midpoints inserted in s and/or p, and tight
extensions; the energy change shows how far QZ-X2C is from the HF limit."""
import numpy as np
from socutils.tools.basisopt import engine
exec(open('ca_scf.py').read().split('for name in')[0])
q = prims('cc-pVQZ-X2C')
def mid(ex):
    out = []
    for a, b in zip(ex[:-1], ex[1:]): out += [a, np.sqrt(a * b)]
    return out + [ex[-1]]
def tight(ex, k=2):
    r = ex[0] / ex[1]; return [ex[0] * r ** i for i in range(k, 0, -1)] + ex
base = engine.atom('Ca', q, 'sfx2c1e', gau_nuc=True).e_scf
print('QZ-X2C %.8f' % base)
for lab, p in [('s mid', {**q, 0: mid(q[0])}), ('p mid', {**q, 1: mid(q[1])}),
               ('s+p mid', {**q, 0: mid(q[0]), 1: mid(q[1])}),
               ('s+p tight2', {**q, 0: tight(q[0]), 1: tight(q[1])}),
               ('s+p mid + tight', {**q, 0: tight(mid(q[0])), 1: tight(mid(q[1]))})]:
    r = engine.atom('Ca', p, 'sfx2c1e', gau_nuc=True)
    print('%-16s %ds %dp  E = %.8f  dE vs QZ-X2C %8.4f mEh %s' % (lab, len(p[0]), len(p[1]), r.e_scf, (r.e_scf - base) * 1e3, r.converged))
