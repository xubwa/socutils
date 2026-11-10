#!/usr/bin/env python
'''
Generator of lib/vhf/rij_direct_gen.c: unrolled kernels of the direct RI-J
engine of rij_direct.c (included there).

  rjr_<L>(R0, r, X, Y, Z), L <= RLMAX: the Hermite tensor R^(0)_tuv of degree
  <= L by the McMurchie-Davidson recursion (R^(n)_{t+1} = t R^(n+1)_{t-1} +
  X R^(n+1)_t along the first nonzero component) from the given
  r[n] = R^(n)_000, n = 0..L: pref (-2 theta)^n F_n(theta R^2) for the
  Coulomb integrals of two Hermite Gaussians, or (-1)^n (2n-1)!! / R^(2n+1)
  for the classical multipole tensor d^n (1 / R);

  rjx_<lp>_<lq>(R0, rp, xq): xq_s += sum_t R_{t+s} rp_t over the degree-lq
  Hermite indices s only (a spherical auxiliary function is a pure degree-l
  Hermite Gaussian), t over all of degree <= lp;

  rjy_<lp>_<lq>(R0, hq, vp): vp_t += sum_s R_{t+s} hq_s, s as above.

Hermite order: by degree, then t ascending, then u ascending (as gen_md_j.py);
the degree-l block of an array of degree <= L starts at NH(l - 1).

    python gen_rij_direct.py > rij_direct_gen.c
'''
import sys

LMAXP = 4       # bra pair degree (d d)
LMAXQ = 4       # auxiliary angular momentum (g)
RLMAX = LMAXP + LMAXQ


def order(L):
    out = []
    for d in range(L + 1):
        for t in range(d + 1):
            for u in range(d - t + 1):
                out.append((t, u, d - t - u))
    return out


def top(l):
    return order(l)[nh(l - 1):]


def nh(d):
    return (d + 1) * (d + 2) * (d + 3) // 6 if d >= 0 else 0


def rjr(L):
    idx = {h: i for i, h in enumerate(order(L))}
    o = []
    w = o.append
    w('static void rjr_%d(double *R0, const double *r, double X, double Y, double Z)' % L)
    w('{')
    for n in range(L + 1):
        w('        const double r%d_0 = r[%d];' % (n, n))
    XYZ = 'XYZ'
    for (t, u, v) in order(L)[1:]:
        c = (t, u, v)
        k = 0 if t else (1 if u else 2)
        m1 = list(c)
        m1[k] -= 1
        m2 = list(c)
        m2[k] -= 2
        i, i1 = idx[c], idx[tuple(m1)]
        cc = c[k] - 1
        i2 = idx[tuple(m2)] if c[k] >= 2 else None
        d = t + u + v
        for n in range(L - d + 1):
            e = '%s * r%d_%d' % (XYZ[k], n + 1, i1)
            if cc > 0:
                e += ' + %d * r%d_%d' % (cc, n + 1, i2)
            w('        const double r%d_%d = %s;' % (n, i, e))
    for i in range(len(order(L))):
        w('        R0[%d] = r0_%d;' % (i, i))
    w('}')
    return '\n'.join(o)


def rjx(lp, lq):
    idx = {h: i for i, h in enumerate(order(lp + lq))}
    P, S = order(lp), top(lq)
    o = []
    w = o.append
    w('static void rjx_%d_%d(const double *R0, const double *rp, double *xq)' % (lp, lq))
    w('{')
    for b, hb in enumerate(S):
        terms = ['R0[%d] * rp[%d]' % (idx[(ha[0] + hb[0], ha[1] + hb[1], ha[2] + hb[2])], a)
                 for a, ha in enumerate(P)]
        w('        xq[%d] += %s;' % (b, ' + '.join(terms)))
    w('}')
    return '\n'.join(o)


def rjy(lp, lq):
    idx = {h: i for i, h in enumerate(order(lp + lq))}
    P, S = order(lp), top(lq)
    o = []
    w = o.append
    w('static void rjy_%d_%d(const double *R0, const double *hq, double *vp)' % (lp, lq))
    w('{')
    for a, ha in enumerate(P):
        terms = ['R0[%d] * hq[%d]' % (idx[(ha[0] + hb[0], ha[1] + hb[1], ha[2] + hb[2])], b)
                 for b, hb in enumerate(S)]
        w('        vp[%d] += %s;' % (a, ' + '.join(terms)))
    w('}')
    return '\n'.join(o)


NL = 8          # SIMD lanes of the batched kernels


def rjrv(L):
    '''rjr over NL lanes: R0[i * NL + lane], r[n * NL + lane]'''
    idx = {h: i for i, h in enumerate(order(L))}
    o = []
    w = o.append
    w('static void rjrv_%d(double *restrict R0, const double *restrict r, const double *restrict X, '
      'const double *restrict Y, const double *restrict Z)' % L)
    w('{')
    w('        int l;')
    w('#pragma omp simd')
    w('        for (l = 0; l < RJ_NL; l++) {')
    for n in range(L + 1):
        w('                const double r%d_0 = r[%d * RJ_NL + l];' % (n, n))
    XYZ = ('X[l]', 'Y[l]', 'Z[l]')
    for (t, u, v) in order(L)[1:]:
        c = (t, u, v)
        k = 0 if t else (1 if u else 2)
        m1 = list(c)
        m1[k] -= 1
        m2 = list(c)
        m2[k] -= 2
        i, i1 = idx[c], idx[tuple(m1)]
        cc = c[k] - 1
        i2 = idx[tuple(m2)] if c[k] >= 2 else None
        d = t + u + v
        for n in range(L - d + 1):
            e = '%s * r%d_%d' % (XYZ[k], n + 1, i1)
            if cc > 0:
                e += ' + %d * r%d_%d' % (cc, n + 1, i2)
            w('                const double r%d_%d = %s;' % (n, i, e))
    for i in range(len(order(L))):
        w('                R0[%d * RJ_NL + l] = r0_%d;' % (i, i))
    w('        }')
    w('}')
    return '\n'.join(o)


def rjxv(lp, lq):
    '''out[s * NL + lane] = sum_t R_{t+s} rp_t, s over the degree-lq indices'''
    idx = {h: i for i, h in enumerate(order(lp + lq))}
    P, S = order(lp), top(lq)
    o = []
    w = o.append
    w('static void rjxv_%d_%d(const double *restrict R0, const double *restrict rp, double *restrict out)' % (lp, lq))
    w('{')
    w('        int l;')
    w('#pragma omp simd')
    w('        for (l = 0; l < RJ_NL; l++) {')
    for b, hb in enumerate(S):
        terms = ['R0[%d * RJ_NL + l] * rp[%d]' % (idx[(ha[0] + hb[0], ha[1] + hb[1], ha[2] + hb[2])], a)
                 for a, ha in enumerate(P)]
        w('                out[%d * RJ_NL + l] = %s;' % (b, ' + '.join(terms)))
    w('        }')
    w('}')
    return '\n'.join(o)


def rjyv(lp, lq):
    '''vl[t * NL + lane] += sum_s R_{t+s} hq[s * NL + lane]'''
    idx = {h: i for i, h in enumerate(order(lp + lq))}
    P, S = order(lp), top(lq)
    o = []
    w = o.append
    w('static void rjyv_%d_%d(const double *restrict R0, const double *restrict hq, double *restrict vl)' % (lp, lq))
    w('{')
    w('        int l;')
    w('#pragma omp simd')
    w('        for (l = 0; l < RJ_NL; l++) {')
    for a, ha in enumerate(P):
        terms = ['R0[%d * RJ_NL + l] * hq[%d * RJ_NL + l]' % (idx[(ha[0] + hb[0], ha[1] + hb[1], ha[2] + hb[2])], b)
                 for b, hb in enumerate(S)]
        w('                vl[%d * RJ_NL + l] += %s;' % (a, ' + '.join(terms)))
    w('        }')
    w('}')
    return '\n'.join(o)


def nearv(L):
    '''r[n * NL + lane] = pref (-2 theta)^n F_n(theta r2) for n <= L over NL lanes, the Boys functions from
    rij_direct.c's table (T < BOYS_TMAX: every near-field pair)'''
    o = []
    w = o.append
    w('static void nearv_%d(double *restrict r, double p, const double *restrict q, const double *restrict r2)' % L)
    w('{')
    w('        int l;')
    w('        const double *bt = &boys_tab[0][0];')
    w('#pragma omp simd')
    w('        for (l = 0; l < RJ_NL; l++) {')
    w('                const double pq = p * q[l], s = p + q[l], theta = pq / s, t = theta * r2[l];')
    w('                const int k = (int)(t * 20 + .5);')
    w('                const double dl = k * .05 - t;')
    w('                const int o = k * BOYS_MT + %d;' % L)
    w('                const double f%d = bt[o] + dl * (bt[o + 1] + dl * .5 * (bt[o + 2] + dl * (1. / 3) * (bt[o + 3]'
      ' + dl * .25 * (bt[o + 4] + dl * .2 * (bt[o + 5] + dl * (1. / 6) * bt[o + 6])))));' % L)
    if L > 0:
        w('                const double et = boys_exp[k] * (1 + dl * (1 + dl * .5 * (1 + dl * (1. / 3) * (1 + dl * .25'
          ' * (1 + dl * .2 * (1 + dl * (1. / 6)))))));')
    for n in range(L, 0, -1):
        w('                const double f%d = (2 * t * f%d + et) * %.17g;' % (n - 1, n, 1. / (2 * n - 1)))
    if L > 0:
        w('                const double m2t = -2 * theta;')
    w('                double g = PI_FAC / (pq * sqrt(s));')
    for n in range(L + 1):
        w('                r[%d * RJ_NL + l] = g * f%d;' % (n, n))
        if n < L:
            w('                g *= m2t;')
    w('        }')
    w('}')
    return '\n'.join(o)


def main():
    out = ['/* Generated by gen_rij_direct.py; do not edit.  Included by rij_direct.c. */', '']
    for L in range(RLMAX + 1):
        out.append(rjr(L))
        out.append('')
    for lp in range(LMAXP + 1):
        for lq in range(LMAXQ + 1):
            out.append(rjx(lp, lq))
            out.append('')
            out.append(rjy(lp, lq))
            out.append('')
    out.append('#define RJ_NL %d' % NL)
    for L in range(RLMAX + 1):
        out.append(nearv(L))
        out.append('')
    for L in range(RLMAX + 1):
        out.append(rjrv(L))
        out.append('')
    for lp in range(LMAXP + 1):
        for lq in range(LMAXQ + 1):
            out.append(rjxv(lp, lq))
            out.append('')
            out.append(rjyv(lp, lq))
            out.append('')
    H = order(RLMAX)
    out.append('#define RJ_LMAXP %d' % LMAXP)
    out.append('#define RJ_LMAXQ %d' % LMAXQ)
    out.append('#define RJ_NH %d' % len(H))
    out.append('static const int rj_t[%d] = {%s};' % (len(H), ', '.join(str(h[0]) for h in H)))
    out.append('static const int rj_u[%d] = {%s};' % (len(H), ', '.join(str(h[1]) for h in H)))
    out.append('static const int rj_v[%d] = {%s};' % (len(H), ', '.join(str(h[2]) for h in H)))
    out.append('typedef void (*rjr_fn)(double *, const double *, double, double, double);')
    out.append('static const rjr_fn rjr_table[%d] = {%s};' % (RLMAX + 1, ', '.join('rjr_%d' % L for L in range(RLMAX + 1))))
    out.append('typedef void (*rjc_fn)(const double *, const double *, double *);')
    out.append('typedef void (*nearv_fn)(double *, double, const double *, const double *);')
    out.append('static const nearv_fn nearv_table[%d] = {%s};' % (RLMAX + 1, ', '.join('nearv_%d' % L for L in range(RLMAX + 1))))
    out.append('typedef void (*rjrv_fn)(double *, const double *, const double *, const double *, const double *);')
    out.append('static const rjrv_fn rjrv_table[%d] = {%s};' % (RLMAX + 1, ', '.join('rjrv_%d' % L for L in range(RLMAX + 1))))
    for nm in ('rjx', 'rjy', 'rjxv', 'rjyv'):
        out.append('static const rjc_fn %s_table[%d][%d] = {' % (nm, LMAXP + 1, LMAXQ + 1))
        for lp in range(LMAXP + 1):
            out.append('        {%s},' % ', '.join('%s_%d_%d' % (nm, lp, lq) for lq in range(LMAXQ + 1)))
        out.append('};')
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
