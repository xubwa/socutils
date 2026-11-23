"""One refinement of cc-pVQZ-X2C per process (the library exits on failure): midpoints in the
outer k s/p primitives, or tight extensions."""
import sys, numpy as np
from socutils.tools.basisopt import engine
exec(open('ca_scf.py').read().split('for name in')[0])
q = prims('cc-pVQZ-X2C')
def mid(ex, k):           # midpoints between the k+1 most diffuse... from position len-k on
    n = len(ex); out = list(ex[:n - k - 1])
    for a, b in zip(ex[n - k - 1:-1], ex[n - k:]): out += [a, np.sqrt(a * b)]
    return out + [ex[-1]]
def tight(ex, k):
    r = ex[0] / ex[1]; return [ex[0] * r ** i for i in range(k, 0, -1)] + ex
mode, l, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = dict(q); p[l] = mid(q[l], k) if mode == 'mid' else tight(q[l], k)
r = engine.atom('Ca', p, 'sfx2c1e', gau_nuc=True)
print('%-5s l=%d k=%2d n=%d E = %.8f conv %s' % (mode, l, k, len(p[l]), r.e_scf, r.converged))
