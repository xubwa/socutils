"""Compare the duplicated / differing entries by the space their contracted functions span
(primitive metric, per l): same span = only the representation differs."""
import json, sys
import numpy as np
exec(open('inventory.py').read().split('fam = {}')[0])     # lines, idx, parse, same
def ovl(ex, l):
    a, b = ex[:, None], ex[None, :]
    return (2 * np.sqrt(a * b) / (a + b)) ** (l + 1.5)
def span_cmp(A, B):
    out = []
    for l in sorted(set(A) | set(B)):
        if l not in A or l not in B:
            out.append('l=%d only in %s' % (l, 'first' if l in A else 'second')); continue
        ea, ca = A[l]; eb, cb = B[l]
        ex = np.array(sorted(set(np.round(np.concatenate([ea, eb]), 10)), reverse=True))
        def embed(e, c):
            m = np.zeros((len(ex), c.shape[1]))
            for i, x in enumerate(e):
                m[np.argmin(abs(ex - x))] += c[i]
            return m
        s = ovl(ex, l); a = embed(ea, ca); b = embed(eb, cb)
        def orth(m):
            g = m.T @ s @ m; w, v = np.linalg.eigh(g); keep = w > 1e-10 * w.max()
            return m @ v[:, keep] / np.sqrt(w[keep])
        qa, qb = orth(a), orth(b)
        sv = np.linalg.svd(qa.T @ s @ qb, compute_uv=False)
        out.append('l=%d dim %d/%d, prims %d/%d, smallest cos %.6f' % (l, qa.shape[1], qb.shape[1], len(ea), len(eb), sv.min() if len(sv) else 0))
    return '; '.join(out)
for key in sys.argv[1:]:
    el, name = key.split(':', 1)
    occ = idx[(name, el)]
    p = [parse(i) for i in occ]
    print(key, '| first: %s | second: %s' % (p[0][0][:40], p[1][0][:40]))
    print('   ', span_cmp(p[0][1], p[1][1]))
