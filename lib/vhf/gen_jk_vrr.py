#!/usr/bin/env python
'''
Generator of lib/vhf/jk_vrr_gen.c: per angular class (la lb | lc ld) with
l <= LMAX, the primitive part of the Head-Gordon-Pople scheme of jk_gen.c
unrolled into straight-line C,

  per primitive quartet: F_m(T), m <= L (Taylor tables), the vertical
  recursions [e0|00]^(m) (Obara-Saika on centre A) and their transfer to the
  ket [e0|f0]^(m) (on centre C), only the (e, f, m) the targets need (deg e in
  la .. la + lb, deg f in lc .. lc + ld, m = 0), in dependency order;

  the bra primitives as the inner (vectorised) loop, the bra then the ket
  contraction into SK[(kc * ncb + cb) * NT + e * NFT + f] (e, f local to the
  target degree bands).

The contracted horizontal recursions, the Cartesian-to-spherical transform
and the output layout are jk_gen.c's.  A = the bra shell of the higher l (the
pair's shells swapped if needed), likewise C; PA = P - R_A, QC = Q - R_C.

    python gen_jk_vrr.py > jk_vrr_gen.c
'''
import sys

LMAX = 2


def comps(d):
    return [(x, y, d - x - y) for x in range(d, -1, -1) for y in range(d - x, -1, -1)]


def band(lo, hi):
    '''Cartesian components of degree lo .. hi, degree-ordered'''
    out = []
    for d in range(lo, hi + 1):
        out += comps(d)
    return out


def sub(e, k, n=1):
    e = list(e)
    e[k] -= n
    return tuple(e)


def first_nz(e):
    return next(k for k in range(3) if e[k])


def name(e, f, m):
    return 'x%d%d%d_%d%d%d_%d' % (e + f + (m,))


def plan(la, lb, lc, ld):
    '''the nodes (e, f, m) in dependency order and their expressions'''
    exprs = {}
    order = []
    X = 'xyz'

    def visit(e, f, m):
        key = (e, f, m)
        if key in exprs:
            return
        if any(f):
            k = first_nz(f)
            f1 = sub(f, k)
            terms = []
            visit(e, f1, m)
            visit(e, f1, m + 1)
            terms.append('QC%s * %s' % (X[k], name(e, f1, m)))
            terms.append('WQ%s * %s' % (X[k], name(e, f1, m + 1)))
            if f[k] >= 2:
                f2 = sub(f, k, 2)
                visit(e, f2, m)
                visit(e, f2, m + 1)
                terms.append('%d * h2q * (%s - rq * %s)' % (f[k] - 1, name(e, f2, m), name(e, f2, m + 1)))
            if e[k] >= 1:
                e1 = sub(e, k)
                visit(e1, f1, m + 1)
                terms.append('%d * h2pq * %s' % (e[k], name(e1, f1, m + 1)))
            exprs[key] = ' + '.join(terms)
        elif any(e):
            k = first_nz(e)
            e1 = sub(e, k)
            visit(e1, f, m)
            visit(e1, f, m + 1)
            terms = ['PA%s * %s' % (X[k], name(e1, f, m)), 'WP%s * %s' % (X[k], name(e1, f, m + 1))]
            if e[k] >= 2:
                e2 = sub(e, k, 2)
                visit(e2, f, m)
                visit(e2, f, m + 1)
                terms.append('%d * h2p * (%s - rp * %s)' % (e[k] - 1, name(e2, f, m), name(e2, f, m + 1)))
            exprs[key] = ' + '.join(terms)
        else:
            exprs[key] = 'F[%d]' % m
        order.append(key)

    E, Fd = band(la, la + lb), band(lc, lc + ld)
    for e in E:
        for f in Fd:
            visit(e, f, 0)
    return order, exprs, E, Fd


def kernel(la, lb, lc, ld):
    order, exprs, E, Fd = plan(la, lb, lc, ld)
    L = la + lb + lc + ld
    nt = len(E) * len(Fd)
    fn = 'jkv_%d%d%d%d' % (la, lb, lc, ld)
    o = []
    o.append('/* (%d%d|%d%d): %d VRR nodes, %d targets */' % (la, lb, lc, ld, len(order), nt))
    o.append('static int %s(double *SK, const GPair *B, int bs, const GPair *K, int ks, double cut,' % fn)
    o.append('                const double *tab, double *wk)')
    o.append('{')
    o.append('        const int nb = B->n, nk = K->n, ncb = B->nc, nck = K->nc;')
    o.append('        const double *pax = bs ? B->rjx : B->rix, *pay = bs ? B->rjy : B->riy, *paz = bs ? B->rjz : B->riz;')
    o.append('        const double *qcx = ks ? K->rjx : K->rix, *qcy = ks ? K->rjy : K->riy, *qcz = ks ? K->rjz : K->riz;')
    o.append('        double *v = wk, *w = v + (size_t)%d * nb;' % nt)
    o.append('        int t, b, cb, kc, i, any = 0;')
    o.append('        memset(SK, 0, sizeof(double) * nck * ncb * %d);' % nt)
    o.append('        for (t = 0; t < nk; t++) {')
    o.append('                if (K->bnd[t] < cut) continue;')
    o.append('                any = 1;')
    o.append('                const double q = K->p[t], qx = K->px[t], qy = K->py[t], qz = K->pz[t];')
    o.append('                const double QCx = qcx[t], QCy = qcy[t], QCz = qcz[t];')
    o.append('                const double h2q = .5 / q;')
    o.append('                (void)QCx; (void)QCy; (void)QCz; (void)h2q;')
    o.append('#pragma omp simd')
    o.append('                for (b = 0; b < nb; b++) {')
    o.append('                        const double p = B->p[b], s = 1. / (p + q);')
    o.append('                        const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;')
    o.append('                        const double rho = p * q * s, T = rho * (dx * dx + dy * dy + dz * dz);')
    o.append('                        double F[%d];' % (L + 1))
    o.append('                        boys_n(tab, T, %d, F);' % L)
    o.append('                        const double PAx = pax[b], PAy = pay[b], PAz = paz[b];')
    o.append('                        const double WPx = -q * s * dx, WPy = -q * s * dy, WPz = -q * s * dz;')
    o.append('                        const double WQx = p * s * dx, WQy = p * s * dy, WQz = p * s * dz;')
    o.append('                        const double h2p = .5 / p, rp = rho / p, rq = rho / q, h2pq = .5 * s;')
    o.append('                        (void)PAx; (void)PAy; (void)PAz; (void)WPx; (void)WPy; (void)WPz;')
    o.append('                        (void)WQx; (void)WQy; (void)WQz; (void)h2p; (void)rp; (void)rq; (void)h2pq;')
    o.append('                        const double g = B->kp[b] * sqrt(s);')
    for key in order:
        o.append('                        const double %s = %s;' % (name(*key), exprs[key]))
    i = 0
    for e in E:
        for f in Fd:
            o.append('                        v[%d * nb + b] = g * %s;' % (i, name(e, f, 0)))
            i += 1
    o.append('                }')
    o.append('                for (cb = 0; cb < ncb; cb++) {')
    o.append('                        const double *c = B->c + (size_t)cb * B->ld;')
    o.append('                        for (i = 0; i < %d; i++) {' % nt)
    o.append('                                const double *vi = v + (size_t)i * nb;')
    o.append('                                double y = 0;')
    o.append('                                for (b = 0; b < nb; b++) y += c[b] * vi[b];')
    o.append('                                w[cb * %d + i] = y;' % nt)
    o.append('                        }')
    o.append('                }')
    o.append('                const double kq = K->kp[t];')
    o.append('                for (kc = 0; kc < nck; kc++) {')
    o.append('                        const double f = K->c[(size_t)kc * K->ld + t] * kq;')
    o.append('                        double *so = SK + (size_t)kc * ncb * %d;' % nt)
    o.append('                        for (i = 0; i < ncb * %d; i++) so[i] += f * w[i];' % nt)
    o.append('                }')
    o.append('        }')
    o.append('        return any;')
    o.append('}')
    return '\n'.join(o), fn, nt


def classes():
    '''canonical classes: la >= lb, lc >= ld, l <= LMAX'''
    pairs = [(a, b) for a in range(LMAX + 1) for b in range(a + 1)]
    return [(a, b, c, d) for (a, b) in pairs for (c, d) in pairs]


def main():
    out = ['/* Generated by gen_jk_vrr.py (LMAX = %d); do not edit. */' % LMAX,
           '/* included by jk_gen.c, which defines GPair, boys_n */', '']
    table = []
    for cls in classes():
        code, fn, nt = kernel(*cls)
        out.append(code)
        out.append('')
        table.append((cls, fn, nt))
    n = LMAX + 1
    out.append('typedef int (*jkv_fn)(double *, const GPair *, int, const GPair *, int, double,')
    out.append('                      const double *, double *);')
    out.append('#define JKV_LMAX %d' % LMAX)
    out.append('static const jkv_fn jkv_table[%d][%d][%d][%d] = {' % (n, n, n, n))
    ent = {}
    for cls, fn, nt in table:
        ent[cls] = fn
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    out.append('        [%d][%d][%d][%d] = %s,' % (a, b, c, d, ent.get((a, b, c, d), 'NULL')))
    out.append('};')
    out.append('static const int jkv_nt[%d][%d][%d][%d] = {' % (n, n, n, n))
    for cls, fn, nt in table:
        out.append('        [%d][%d][%d][%d] = %d,' % (cls + (nt,)))
    out.append('};')
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
