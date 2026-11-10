#!/usr/bin/env python
'''
Generates lib/dft/cosx_r_gen.h: for every Hermite degree L <= LMAX one AVX-512
kernel of lib/dft/cosx.c,

    static void cxr_L(double *A, int nab, const double *E, const double *F, const double *s,
                      const double *X, const double *Y, const double *Z, int n)

which, per chunk of 8 grid points held in registers, builds the Hermite
Coulomb tensor R^(0)_tuv (t + u + v <= L) by the straight-line recursion
R^(m)_{t+1,u,v} = t R^(m+1)_{t-1,u,v} + X R^(m+1)_{t,u,v} from the seeds
R^(m)_000 = s[m] F[m] (F: (L + 1, n) Boys values) and accumulates
A[ab, g] += sum_i E[ab, i] R_i(g) for ab < nab (A rows of n points).  The
index order of the Hermite components is that of cosx.c's tables (by degree,
t descending, then u descending).

    python lib/dft/gen_cosx_r.py > lib/dft/cosx_r_gen.h
'''
import sys

LMAX = 8


def nh(d):
    return (d + 1) * (d + 2) * (d + 3) // 6


def tables(dmax):
    idx, comps = {}, []
    for d in range(dmax + 1):
        for t in range(d, -1, -1):
            for u in range(d - t, -1, -1):
                v = d - t - u
                idx[(t, u, v)] = len(comps)
                comps.append((t, u, v))
    return idx, comps


def gen(L, idx, comps):
    o = []
    w = o.append
    w('static void cxr_%d(double *A, int nab, const double *E, const double *F, const double *s,' % L)
    w('                  const double *X, const double *Y, const double *Z, int n)')
    w('{')
    w('        int g0, ab;')
    w('        for (g0 = 0; g0 < n; g0 += 8) {')
    w('                const __mmask8 mk = n - g0 >= 8 ? 0xff : (__mmask8)((1u << (n - g0)) - 1);')
    if L > 0:
        w('                const __m512d x = _mm512_maskz_loadu_pd(mk, X + g0), y = _mm512_maskz_loadu_pd(mk, Y + g0),'
          ' z = _mm512_maskz_loadu_pd(mk, Z + g0);')
    n_ = nh(L)
    # level m holds R^(m)_i for deg(i) <= L - m; names r{m}_{i}
    for m in range(L, -1, -1):
        w('                const __m512d r%d_0 = _mm512_mul_pd(_mm512_set1_pd(s[%d]), _mm512_maskz_loadu_pd(mk, F + %d * n + g0));'
          % (m, m, m))
        for i in range(1, nh(L - m)):
            t, u, v = comps[i]
            c = [t, u, v]
            k = 0 if t else (1 if u else 2)
            m1 = list(c)
            m1[k] -= 1
            xv = 'xyz'[k]
            a = 'r%d_%d' % (m + 1, idx[tuple(m1)])
            if c[k] >= 2:
                m2 = list(c)
                m2[k] -= 2
                b = 'r%d_%d' % (m + 1, idx[tuple(m2)])
                if c[k] - 1 == 1:
                    w('                const __m512d r%d_%d = _mm512_fmadd_pd(%s, %s, %s);' % (m, i, xv, a, b))
                else:
                    w('                const __m512d r%d_%d = _mm512_fmadd_pd(%s, %s, _mm512_mul_pd(_mm512_set1_pd(%d.), %s));'
                      % (m, i, xv, a, c[k] - 1, b))
            else:
                w('                const __m512d r%d_%d = _mm512_mul_pd(%s, %s);' % (m, i, xv, a))
    # 4 rows of A at a time, two partial sums each: 8 independent FMA chains
    w('                for (ab = 0; ab + 4 <= nab; ab += 4) {')
    w('                        const double *e = E + (size_t)ab * %d;' % n_)
    w('                        double *a = A + (size_t)ab * n + g0;')
    for q in range(4):
        w('                        __m512d p%d = _mm512_maskz_loadu_pd(mk, a + %d * (size_t)n), q%d = _mm512_setzero_pd();' % (q, q, q))
    for i in range(n_):
        for q in range(4):
            acc = 'p%d' % q if i % 2 == 0 else 'q%d' % q
            w('                        %s = _mm512_fmadd_pd(_mm512_set1_pd(e[%d]), r0_%d, %s);' % (acc, q * n_ + i, i, acc))
    for q in range(4):
        w('                        _mm512_mask_storeu_pd(a + %d * (size_t)n, mk, _mm512_add_pd(p%d, q%d));' % (q, q, q))
    w('                }')
    w('                for (; ab < nab; ab++) {')
    w('                        const double *e = E + (size_t)ab * %d;' % n_)
    w('                        double *a = A + (size_t)ab * n + g0;')
    w('                        __m512d p0 = _mm512_maskz_loadu_pd(mk, a), p1 = _mm512_setzero_pd(), '
      'p2 = _mm512_setzero_pd(), p3 = _mm512_setzero_pd();')
    for i in range(n_):
        w('                        p%d = _mm512_fmadd_pd(_mm512_set1_pd(e[%d]), r0_%d, p%d);' % (i % 4, i, i, i % 4))
    w('                        _mm512_mask_storeu_pd(a, mk, _mm512_add_pd(_mm512_add_pd(p0, p1), _mm512_add_pd(p2, p3)));')
    w('                }')
    w('        }')
    w('}')
    return '\n'.join(o)


def gen_full(L, idx, comps):
    """cxp_L: per 8-point chunk T, the Boys values, R and A += E R in registers"""
    o = []
    w = o.append
    n_ = nh(L)
    w('static void cxp_%d(double *A, int nab, const double *E, double p, const double *P, const double *xyz, int n)' % L)
    w('{')
    w('        int g0, ab;')
    w('        const __m512d vp = _mm512_set1_pd(p), px = _mm512_set1_pd(P[0]), py = _mm512_set1_pd(P[1]),'
      ' pz = _mm512_set1_pd(P[2]);')
    w('        const __m512d tsw = _mm512_set1_pd(BOYS_TSW), one = _mm512_set1_pd(1.);')
    w('        const __m512i ibase = _mm512_set1_epi64(%d);' % L)
    w('        double sc[%d];' % (L + 1))
    w('        sc[0] = 1;')
    for m in range(1, L + 1):
        w('        sc[%d] = sc[%d] * (-2 * p);' % (m, m - 1))
    w('        for (g0 = 0; g0 < n; g0 += 8) {')
    w('                const __mmask8 mk = n - g0 >= 8 ? 0xff : (__mmask8)((1u << (n - g0)) - 1);')
    w('                const __m512d x = _mm512_sub_pd(px, _mm512_mask_loadu_pd(px, mk, xyz + g0));')
    w('                const __m512d y = _mm512_sub_pd(py, _mm512_mask_loadu_pd(py, mk, xyz + n + g0));')
    w('                const __m512d z = _mm512_sub_pd(pz, _mm512_mask_loadu_pd(pz, mk, xyz + 2 * n + g0));')
    w('                const __m512d t = _mm512_mul_pd(vp, _mm512_fmadd_pd(x, x, _mm512_fmadd_pd(y, y, _mm512_mul_pd(z, z))));')
    w('                const __mmask8 far = _mm512_cmp_pd_mask(t, tsw, _CMP_GE_OQ);')
    for m in range(L + 1):
        w('                __m512d f%d;' % m)
    w('                if (far != 0xff) {')
    w('                        const __m512d tc = _mm512_min_pd(t, tsw);')
    w('                        const __m512i k = _mm512_cvttpd_epi64(_mm512_fmadd_pd(tc, _mm512_set1_pd(32.), _mm512_set1_pd(.5)));')
    w('                        const __m512d dl = _mm512_fmsub_pd(_mm512_cvtepi64_pd(k), _mm512_set1_pd(1. / 32), tc);')
    w('                        const __m512i ix = _mm512_add_epi64(_mm512_mullo_epi64(k, _mm512_set1_epi64(BOYS_MT)), ibase);')
    w('                        const double *tb = boys_tab[0];')
    for j in range(5):
        w('                        const __m512d b%d = _mm512_i64gather_pd(%s, tb, 8);'
          % (j, 'ix' if j == 0 else '_mm512_add_epi64(ix, _mm512_set1_epi64(%d))' % j))
    w('                        f%d = _mm512_fmadd_pd(dl, _mm512_fmadd_pd(_mm512_mul_pd(dl, _mm512_set1_pd(.5)), '
      '_mm512_fmadd_pd(_mm512_mul_pd(dl, _mm512_set1_pd(1. / 3)), _mm512_fmadd_pd(_mm512_mul_pd(dl, _mm512_set1_pd(.25)), '
      'b4, b3), b2), b1), b0);' % L)
    if L > 0:
        w('                        const __m512d et = _ZGVeN8v_exp(_mm512_sub_pd(_mm512_setzero_pd(), tc));')
        w('                        const __m512d t2 = _mm512_add_pd(t, t);')
        for m in range(L, 0, -1):
            w('                        f%d = _mm512_mul_pd(_mm512_fmadd_pd(t2, f%d, et), _mm512_set1_pd(%.17g));'
              % (m - 1, m, 1. / (2 * m - 1)))
    w('                }')
    w('                if (far) {')
    w('                        __m512d r = _mm512_rsqrt14_pd(_mm512_max_pd(t, one));')
    w('                        const __m512d hh = _mm512_mul_pd(_mm512_max_pd(t, one), _mm512_set1_pd(.5));')
    w('                        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hh, _mm512_mul_pd(r, r), _mm512_set1_pd(1.5)));')
    w('                        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hh, _mm512_mul_pd(r, r), _mm512_set1_pd(1.5)));')
    if L > 0:
        w('                        const __m512d h = _mm512_mul_pd(_mm512_mul_pd(r, r), _mm512_set1_pd(.5));')
    w('                        __m512d fa = _mm512_mul_pd(r, _mm512_set1_pd(.886226925452758014));')
    for m in range(L + 1):
        w('                        f%d = far == 0xff ? fa : _mm512_mask_blend_pd(far, f%d, fa);' % (m, m))
        if m < L:
            w('                        fa = _mm512_mul_pd(fa, _mm512_mul_pd(h, _mm512_set1_pd(%d.)));' % (2 * m + 1))
    w('                }')
    for m in range(L, -1, -1):
        w('                const __m512d r%d_0 = _mm512_mul_pd(_mm512_set1_pd(sc[%d]), f%d);' % (m, m, m))
        for i in range(1, nh(L - m)):
            t, u, v = comps[i]
            c = [t, u, v]
            k = 0 if t else (1 if u else 2)
            m1 = list(c)
            m1[k] -= 1
            xv = 'xyz'[k]
            a = 'r%d_%d' % (m + 1, idx[tuple(m1)])
            if c[k] >= 2:
                m2 = list(c)
                m2[k] -= 2
                b = 'r%d_%d' % (m + 1, idx[tuple(m2)])
                if c[k] - 1 == 1:
                    w('                const __m512d r%d_%d = _mm512_fmadd_pd(%s, %s, %s);' % (m, i, xv, a, b))
                else:
                    w('                const __m512d r%d_%d = _mm512_fmadd_pd(%s, %s, _mm512_mul_pd(_mm512_set1_pd(%d.), %s));'
                      % (m, i, xv, a, c[k] - 1, b))
            else:
                w('                const __m512d r%d_%d = _mm512_mul_pd(%s, %s);' % (m, i, xv, a))
    if L == 0:
        w('                (void)x; (void)y; (void)z;')
    emit_er(w, n_)
    w('        }')
    w('}')
    return '\n'.join(o)


def emit_er(w, n_):
    w('                for (ab = 0; ab + 4 <= nab; ab += 4) {')
    w('                        const double *e = E + (size_t)ab * %d;' % n_)
    w('                        double *a = A + (size_t)ab * n + g0;')
    for q in range(4):
        w('                        __m512d p%d = _mm512_maskz_loadu_pd(mk, a + %d * (size_t)n), q%d = _mm512_setzero_pd();' % (q, q, q))
    for i in range(n_):
        for q in range(4):
            acc = 'p%d' % q if i % 2 == 0 else 'q%d' % q
            w('                        %s = _mm512_fmadd_pd(_mm512_set1_pd(e[%d]), r0_%d, %s);' % (acc, q * n_ + i, i, acc))
    for q in range(4):
        w('                        _mm512_mask_storeu_pd(a + %d * (size_t)n, mk, _mm512_add_pd(p%d, q%d));' % (q, q, q))
    w('                }')
    w('                for (; ab < nab; ab++) {')
    w('                        const double *e = E + (size_t)ab * %d;' % n_)
    w('                        double *a = A + (size_t)ab * n + g0;')
    w('                        __m512d p0 = _mm512_maskz_loadu_pd(mk, a), p1 = _mm512_setzero_pd(), '
      'p2 = _mm512_setzero_pd(), p3 = _mm512_setzero_pd();')
    for i in range(n_):
        w('                        p%d = _mm512_fmadd_pd(_mm512_set1_pd(e[%d]), r0_%d, p%d);' % (i % 4, i, i, i % 4))
    w('                        _mm512_mask_storeu_pd(a, mk, _mm512_add_pd(_mm512_add_pd(p0, p1), _mm512_add_pd(p2, p3)));')
    w('                }')


def main():
    idx, comps = tables(LMAX)
    out = ['/* generated by lib/dft/gen_cosx_r.py: the fused Hermite R / E R kernels of cosx.c (AVX-512) */',
           '#define CXR_LMAX %d' % LMAX]
    for L in range(LMAX + 1):
        out.append(gen(L, idx, comps))
    out.append('typedef void (*cxr_fn)(double *, int, const double *, const double *, const double *,')
    out.append('                       const double *, const double *, const double *, int);')
    out.append('static const cxr_fn cxr_table[CXR_LMAX + 1] = {%s};' % ', '.join('cxr_%d' % L for L in range(LMAX + 1)))
    out.append('__m512d _ZGVeN8v_exp(__m512d);         /* libmvec */')
    for L in range(LMAX + 1):
        out.append(gen_full(L, idx, comps))
    out.append('typedef void (*cxp_fn)(double *, int, const double *, double, const double *, const double *, int);')
    out.append('static const cxp_fn cxp_table[CXR_LMAX + 1] = {%s};' % ', '.join('cxp_%d' % L for L in range(LMAX + 1)))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
