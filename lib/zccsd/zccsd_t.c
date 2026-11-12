/*
 * zccsd_t.c -- the (T) energy of the spinor CCSD from batches of its
 * intermediates (zccsdt/zccsd/triples.py).
 *
 * For virtual a > b > c and occupied i > j > k, with pairs in
 * numpy.tril_indices(n, -1) order,
 *
 *   x_{a,(bc)}[i, (jk)] = sum_e <bc||ei> t2[j,k,a,e] - sum_m t2[i,m,b,c] <ma||jk>
 *   Z[i, (jk)]  = x_{a,(bc)} - x_{b,(ac)} + x_{c,(ab)}
 *   w[ijk]      = Z[i, (jk)] - Z[j, (ik)] + Z[k, (ij)]
 *
 * and the same two antisymmetrizations of the disconnected part
 *
 *   y_{a,(bc)}[i, (jk)] = t1[i,a] <bc||jk> + f[a,i] t2[j,k,b,c]
 *
 * give v = w + (antisymmetrized y), and
 *
 *   E(T) = sum_{a>b>c} sum_{i>j>k} conj(v) w / (e_i + e_j + e_k - e_a - e_b - e_c).
 *
 * zc_t_batch adds up the triples (a, b, c) of one a and a range b0 <= b < b1,
 * c < b, from the GEMM results of a batch, matrices whose rows are (pair or
 * virtual, i) and whose columns are (virtual, (jk)):
 *
 *   X1[(p - p0, i), c1 + (jk)] = x_{a,(bc)},  p = b (b - 1) / 2 + c the pair
 *       (b, c); s1 elements per row (several a side by side: c1 picks one)
 *   G1[(x, i), (y - b0, (jk))] = x_{y,(a x)} for x < b1 and y >= b0; n1
 *       elements per row
 *   G2[(x - b0, i), (y, (jk))] = x_{y,(a x)} for b0 <= x < b1 and y < b0; n2
 *       elements per row
 *
 * (x_{b,(ac)} = G1[c, b]; x_{c,(ab)} = G1[b, c] or G2[b, c]).  The pairs
 * (b, c) are the tasks of the threads: Z of one pair is assembled in the
 * cache of its thread.
 *
 * With m3 the three arrays come as the three real products of the "3M"
 * scheme, p1 = Ar Br, p2 = Ai Bi, p3 = (Ar + Ai) (Br + Bi), summed over the
 * particle and the hole term: x = (p1 - p2) + i (p3 - p1 - p2), taken after
 * the three real Z have been put together.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef double complex zc;

/* VpT[(b>c), (j>k)] = <bc||jk>, T2pT[(b>c), (j>k)] = t2[j,k,b,c], t1[i,a],
 * fvo[a,i]; out[0], out[1]: real and imaginary part of the sum.  X1, G1, G2:
 * one complex array each (m3 = 0) or three real ones. */
void zc_t_batch(const void **X1p, int64_t s1, int64_t c1, const void **G1p, int64_t n1,
                const void **G2p, int64_t n2, int m3, int64_t a, int64_t b0, int64_t b1,
                int64_t no, int64_t nv, const zc *t1, const zc *fvo, const zc *VpT,
                const zc *T2pT, const double *eo, const double *ev, double *out)
{
    const zc *X1 = X1p[0], *G1 = G1p[0], *G2 = G2p[0];
    const int64_t npo = no * (no - 1) / 2;
    const int64_t p0 = b0 * (b0 - 1) / 2, pa = a * (a - 1) / 2;
    /* the pairs (b, c), c < b, of the range are the tasks, so that the
     * threads are busy also when the range holds a few b only; several
     * pairs per task where the arrays of a pair are small (few occupied
     * spinors) */
    const int64_t bmin = b0 > 1 ? b0 : 1;
    const int64_t pbeg = bmin * (bmin - 1) / 2, pend = b1 * (b1 - 1) / 2;
    const int64_t chunk = 1 + 8192 / (no * npo);
    double er = 0.0, ei = 0.0;
    int has_f = 0;                                  /* f[a,i] = 0 for canonical orbitals */
    for (int64_t x = 0; x < no * nv && !has_f; x++)
        has_f = creal(fvo[x]) != 0.0 || cimag(fvo[x]) != 0.0;
    #pragma omp parallel reduction(+:er, ei)
    {
        zc *Z = malloc(sizeof(zc) * no * npo);      /* Z[i, (jk)] */
        double *zp = malloc(sizeof(double) * 3 * npo);
        zc *tx = malloc(sizeof(zc) * 6 * no);       /* t1[k,x] and f[x,k] for x = a, b, c */
        zc *ta = tx, *tb = tx + no, *tc = tx + 2 * no;
        zc *fa = tx + 3 * no, *fb = tx + 4 * no, *fc = tx + 5 * no;
        #pragma omp for schedule(dynamic, chunk)
        for (int64_t pbc = pbeg; pbc < pend; pbc++) {
            int64_t b = (int64_t)((1.0 + sqrt(1.0 + 8.0 * (double)pbc)) / 2.0);
            while (b * (b - 1) / 2 > pbc)
                b--;
            while ((b + 1) * b / 2 <= pbc)
                b++;
            const int64_t c = pbc - b * (b - 1) / 2;
            /* Z[i, (jk)] = x_{a,(bc)} - x_{b,(ac)} + x_{c,(ab)} */
            for (int64_t i = 0; i < no; i++) {
                const int64_t o1 = ((pbc - p0) * no + i) * s1 + c1;
                const int64_t o2 = (c * no + i) * n1 + (b - b0) * npo;
                const int64_t o3 = c >= b0 ? (b * no + i) * n1 + (c - b0) * npo
                                           : ((b - b0) * no + i) * n2 + c * npo;
                zc *z = Z + i * npo;
                if (m3) {
                    for (int k = 0; k < 3; k++) {
                        const double *r1 = (const double *)X1p[k] + o1;
                        const double *r2 = (const double *)G1p[k] + o2;
                        const double *r3 = (const double *)(c >= b0 ? G1p[k] : G2p[k]) + o3;
                        double *d = zp + k * npo;
                        for (int64_t q = 0; q < npo; q++)
                            d[q] = r1[q] - r2[q] + r3[q];
                    }
                    const double *d0 = zp, *d1 = zp + npo, *d2 = zp + 2 * npo;
                    for (int64_t q = 0; q < npo; q++)
                        z[q] = (d0[q] - d1[q]) + (d2[q] - d0[q] - d1[q]) * I;
                } else {
                    const zc *r1 = X1 + o1, *r2 = G1 + o2, *r3 = (c >= b0 ? G1 : G2) + o3;
                    for (int64_t q = 0; q < npo; q++)
                        z[q] = r1[q] - r2[q] + r3[q];
                }
            }
            /* the disconnected part is put together with w, from the columns
             * a, b, c of t1 (and of f) and the rows (bc), (ac), (ab) of
             * <bc||jk> (and of t2) */
            const zc *vbc = VpT + pbc * npo, *tbc = T2pT + pbc * npo;
            const zc *vac = VpT + (pa + c) * npo, *tac = T2pT + (pa + c) * npo;
            const zc *vab = VpT + (pa + b) * npo, *tab = T2pT + (pa + b) * npo;
            for (int64_t k = 0; k < no; k++) {
                ta[k] = t1[k * nv + a];
                tb[k] = t1[k * nv + b];
                tc[k] = t1[k * nv + c];
                fa[k] = fvo[a * no + k];
                fb[k] = fvo[b * no + k];
                fc[k] = fvo[c * no + k];
            }
            /* w[ijk] = Z[i,(jk)] - Z[j,(ik)] + Z[k,(ij)]; the same combination
             * of y[i,(jk)] = t1[i,a] <bc||jk> - t1[i,b] <ac||jk> + t1[i,c] <ab||jk>
             * (and of the f t2 terms) is added to it in u */
            const double eabc = ev[a] + ev[b] + ev[c];
            for (int64_t i = 2; i < no; i++)
                for (int64_t j = 1; j < i; j++) {
                    const int64_t pij = i * (i - 1) / 2 + j;
                    const int64_t oj = j * (j - 1) / 2, oi = i * (i - 1) / 2;
                    const zc *zi = Z + i * npo + oj, *zj = Z + j * npo + oi;
                    const zc *zk = Z + pij;                         /* Z[k, (ij)]: stride npo */
                    const zc tai = ta[i], tbi = tb[i], tci = tc[i];
                    const zc taj = ta[j], tbj = tb[j], tcj = tc[j];
                    const zc vbcij = vbc[pij], vacij = vac[pij], vabij = vab[pij];
                    const double dij = eo[i] + eo[j] - eabc;
                    if (!has_f) {
                        #pragma omp simd reduction(+:er, ei)
                        for (int64_t k = 0; k < j; k++) {
                            const zc w = zi[k] - zj[k] + zk[k * npo];
                            const zc u = w + tai * vbc[oj + k] - tbi * vac[oj + k]
                                         + tci * vab[oj + k]
                                         - taj * vbc[oi + k] + tbj * vac[oi + k]
                                         - tcj * vab[oi + k]
                                         + ta[k] * vbcij - tb[k] * vacij + tc[k] * vabij;
                            const double rd = 1.0 / (dij + eo[k]);
                            er += (creal(u) * creal(w) + cimag(u) * cimag(w)) * rd;
                            ei += (creal(u) * cimag(w) - cimag(u) * creal(w)) * rd;
                        }
                    } else {
                        const zc fai = fa[i], fbi = fb[i], fci = fc[i];
                        const zc faj = fa[j], fbj = fb[j], fcj = fc[j];
                        const zc tbcij = tbc[pij], tacij = tac[pij], tabij = tab[pij];
                        #pragma omp simd reduction(+:er, ei)
                        for (int64_t k = 0; k < j; k++) {
                            const zc w = zi[k] - zj[k] + zk[k * npo];
                            const zc u = w + tai * vbc[oj + k] - tbi * vac[oj + k]
                                         + tci * vab[oj + k]
                                         - taj * vbc[oi + k] + tbj * vac[oi + k]
                                         - tcj * vab[oi + k]
                                         + ta[k] * vbcij - tb[k] * vacij + tc[k] * vabij
                                         + fai * tbc[oj + k] - fbi * tac[oj + k]
                                         + fci * tab[oj + k]
                                         - faj * tbc[oi + k] + fbj * tac[oi + k]
                                         - fcj * tab[oi + k]
                                         + fa[k] * tbcij - fb[k] * tacij + fc[k] * tabij;
                            const double rd = 1.0 / (dij + eo[k]);
                            er += (creal(u) * creal(w) + cimag(u) * cimag(w)) * rd;
                            ei += (creal(u) * cimag(w) - cimag(u) * creal(w)) * rd;
                        }
                    }
                }
        }
        free(Z);
        free(zp);
        free(tx);
    }
    out[0] = er;
    out[1] = ei;
}

/* ------------------------------------------------------------------ operands
 *
 * The two arrays the GEMMs of (T) read, written block by block:
 *
 *   A[(b>c), i, (e | m)] = (<bc||ei> | t2[i,m,b,c])
 *   B[a, (j>k), (e | m)] = (t2[j,k,a,e] | -<ma||jk>)
 *
 * so that x_{a,(bc)}[i, (jk)] = sum over (e | m) of A B is one product for
 * the particle and the hole term.  The kernels fill one part of the last
 * index each: ld elements per row, the part starts at off.  With m3 the
 * arrays are the three real parts of the 3M scheme (Re, Im, Re + Im: out0,
 * out1, out2 double), else complex (out0). */

#define PUT(x, val)                                         \
    do {                                                    \
        const zc v_ = (val);                                \
        if (m3) {                                           \
            ((double *)out0)[x] = creal(v_);                \
            ((double *)out1)[x] = cimag(v_);                \
            ((double *)out2)[x] = creal(v_) + cimag(v_);    \
        } else                                              \
            ((zc *)out0)[x] = v_;                           \
    } while (0)

/* A[p, i, off + e] = <bc||ei> = -conj(<ie||bc>) for one occupied i and np
 * pairs p = (b>c): src[e ld_src + p] = <ie||bc> for the pair p of the block
 * (the slab of the integrals as the CCSD keeps them, from its first pair
 * on).  Blocks of pairs, so that the rows that are being written stay in
 * the cache while the slab is read by rows. */
void zc_t_ovvv(const zc *src, int64_t ld_src, int64_t nv, int64_t np, int64_t no, int64_t i,
               int64_t ld, int64_t off, int m3, void *out0, void *out1, void *out2)
{
    enum { BLK = 256 };
    #pragma omp parallel for schedule(static)
    for (int64_t p0 = 0; p0 < np; p0 += BLK) {
        const int64_t p1 = p0 + BLK < np ? p0 + BLK : np;
        for (int64_t e = 0; e < nv; e++) {
            const zc *row = src + e * ld_src;
            for (int64_t p = p0; p < p1; p++)
                PUT((p * no + i) * ld + off + e, -conj(row[p]));
        }
    }
}

/* B[a - a0, (j>k), off + e] = t2[j,k,a,e] for a0 <= a < a1 from
 * t2p[(j>k), (a>e)]. */
void zc_t_t2u(const zc *t2p, int64_t npo, int64_t nv, int64_t a0, int64_t a1,
              int64_t ld, int64_t off, int m3, void *out0, void *out1, void *out2)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t a = a0; a < a1; a++)
        for (int64_t p = 0; p < npo; p++) {
            const zc *r = t2p + p * npv;
            const int64_t x = ((a - a0) * npo + p) * ld + off;
            for (int64_t e = 0; e < a; e++)
                PUT(x + e, r[a * (a - 1) / 2 + e]);
            PUT(x + a, 0.0);
            for (int64_t e = a + 1; e < nv; e++)
                PUT(x + e, -r[e * (e - 1) / 2 + a]);
        }
}

/* A[q - q0, i, off + m] = t2[i,m,b,c] for the pairs q0 <= q = (b>c) < q1 from
 * t2p[(i>m), (b>c)]. */
void zc_t_t2x(const zc *t2p, int64_t no, int64_t npv, int64_t q0, int64_t q1,
              int64_t ld, int64_t off, int m3, void *out0, void *out1, void *out2)
{
    #pragma omp parallel for schedule(static)
    for (int64_t q = q0; q < q1; q++)
        for (int64_t i = 0; i < no; i++) {
            const int64_t x = ((q - q0) * no + i) * ld + off;
            for (int64_t m = 0; m < i; m++)
                PUT(x + m, t2p[(i * (i - 1) / 2 + m) * npv + q]);
            PUT(x + i, 0.0);
            for (int64_t m = i + 1; m < no; m++)
                PUT(x + m, -t2p[(m * (m - 1) / 2 + i) * npv + q]);
        }
}

/* ----------------------------------------- the (T) terms of the Lambda equations
 *
 * E(T) depends on t1 and t2 and on the Fock matrix: the Lambda equations of
 * CCSD(T) have its derivatives with respect to the amplitudes as source
 * terms, and the one-particle density has (T) terms of its own
 * (zccsdt/zccsd/lam_t.py has the equations).  With w and u = w + (the
 * disconnected part) of the virtual triple a > b > c, both divided by the
 * denominator, all of them are sums over the triples, each in three roles:
 * one of its virtual indices s, the other two as a pair p > q, and the sign
 * of the permutation (s, p, q) of (a, b, c) - + for (a, (bc)), - for
 * (b, (ac)), + for (c, (ab)):
 *
 *   l1[i, s]       += sign sum_{j>k} conj(w[ijk]) <pq||jk>
 *   dvo[s, k]      += sign sum_{i>j} conj(t2[i,j,p,q]) w[ijk]
 *   goo[i]         += sum_{k>l} conj(u[ikl]) w[ikl]
 *   gvv[s]         += sum_{i>j>k} u[ijk] conj(w[ijk])
 *   lt[(pq), (ij)] += sign sum_k f[s,k] conj(w[ijk])
 *
 * and two contractions of
 *
 *   M[i, (jk)] = conj(w + u)[ijk]      (all i, pairs j > k; antisymmetric in
 *                                       the three, zero where i is j or k)
 *
 * with <ke||pq> over k and with <mn||js> over m > n: o^3 v^4 / 4 and
 * o^4 v^3 / 4 multiply-adds, as many as the x took.  These two are products
 * of the shape of those that made the x, result and operand exchanged, and
 * are left to GEMM: each of the three x of a triple belongs to that triple
 * alone, so zc_t_batch_lam writes sign M over them - M over x_{a,(bc)}, -M
 * over x_{b,(ac)}, M over x_{c,(ab)} (with m3: its real part, its imaginary
 * part and their sum over the three real arrays) -, zeroes the x_{y,(ay)} of
 * G1, which belong to no triple, and the caller multiplies the arrays of the
 * batch with the integrals of A and B (triples.py).  The other sums are done
 * here.
 */

static inline zc dotu(const zc *x, const zc *y, int64_t n)
{
    double sr = 0.0, si = 0.0;
    #pragma omp simd reduction(+:sr, si)
    for (int64_t k = 0; k < n; k++) {
        const double xr = creal(x[k]), xi = cimag(x[k]), yr = creal(y[k]), yi = cimag(y[k]);
        sr += xr * yr - xi * yi;
        si += xr * yi + xi * yr;
    }
    return sr + si * I;
}

/* For the three virtuals s = a, b, c of a triple with the vectors xa, xb, xc
 * over the occupied spinors: x[k] conj(w[ijk]) to the pair (ij), -x[j] to
 * (ik), x[i] to (jk) of the three rows buf[s, (pair)]; wv = conj(w[ij, k < j])
 * over the denominator. */
static inline void role_sums(const zc *wv, const zc *xa, const zc *xb, const zc *xc,
                             int64_t i, int64_t j, int64_t npo, zc *buf)
{
    const int64_t pij = i * (i - 1) / 2 + j, oj = j * (j - 1) / 2, oi = i * (i - 1) / 2;
    const zc xai = xa[i], xbi = xb[i], xci = xc[i];
    const zc xaj = xa[j], xbj = xb[j], xcj = xc[j];
    zc *f1 = buf, *f2 = buf + npo, *f3 = buf + 2 * npo;
    f1[pij] += dotu(wv, xa, j);
    f2[pij] += dotu(wv, xb, j);
    f3[pij] += dotu(wv, xc, j);
    for (int64_t k = 0; k < j; k++) {
        const zc wc = wv[k];
        f1[oi + k] -= xaj * wc;
        f2[oi + k] -= xbj * wc;
        f3[oi + k] -= xcj * wc;
        f1[oj + k] += xai * wc;
        f2[oj + k] += xbi * wc;
        f3[oj + k] += xci * wc;
    }
}

/* The three rows of role_sums to the rows (bc), -(ac), (ab) of out[(pair of
 * virtuals), (pair)]: (bc) is the pair's own, (ac) and (ab) are shared with
 * the other pairs of a. */
static inline void role_rows(const zc *buf, zc *out, int64_t pbc, int64_t pac, int64_t pab,
                             int64_t npo)
{
    zc *r = out + pbc * npo;
    for (int64_t q = 0; q < npo; q++)
        r[q] += buf[q];
    #pragma omp critical(zc_t_lam_rows)
    {
        zc *rc = out + pac * npo, *rb = out + pab * npo;
        for (int64_t q = 0; q < npo; q++) {
            rc[q] -= buf[npo + q];
            rb[q] += buf[2 * npo + q];
        }
    }
}

/* The arguments of zc_t_batch (X1, G1, G2 are overwritten), and the sums:
 * l1[i,a], dvo[a,i], goo[i], gvv[a] and lt[(a>b), (i>j)], which are added
 * to; g3[(a>b), (i>j)] (or NULL): the density of <bc||jk>, the sum of
 * lt with t1[k,s] in the place of f[s,k]. */
void zc_t_batch_lam(void **X1p, int64_t s1, int64_t c1, void **G1p, int64_t n1, void **G2p,
                    int64_t n2, int m3, int64_t a, int64_t b0, int64_t b1, int64_t no,
                    int64_t nv, const zc *t1, const zc *fvo, const zc *VpT, const zc *T2pT,
                    const double *eo, const double *ev, double *out, zc *l1, zc *dvo,
                    zc *goo, zc *gvv, zc *lt, zc *g3)
{
    const int64_t npo = no * (no - 1) / 2;
    const int64_t p0 = b0 * (b0 - 1) / 2, pa = a * (a - 1) / 2;
    const int64_t bmin = b0 > 1 ? b0 : 1;
    const int64_t pbeg = bmin * (bmin - 1) / 2, pend = b1 * (b1 - 1) / 2;
    const int64_t chunk = 1 + 8192 / (no * npo);
    const int64_t ns = b1 + 1;          /* the virtuals of the sums: those below b1, and a */
    double er = 0.0, ei = 0.0;
    int has_f = 0;
    for (int64_t x = 0; x < no * nv && !has_f; x++)
        has_f = creal(fvo[x]) != 0.0 || cimag(fvo[x]) != 0.0;
    #pragma omp parallel reduction(+:er, ei)
    {
        zc *Z = malloc(sizeof(zc) * no * npo);
        double *zp = malloc(sizeof(double) * 3 * npo);
        zc *tx = malloc(sizeof(zc) * 6 * no);
        zc *ta = tx, *tb = tx + no, *tc = tx + 2 * no;
        zc *fa = tx + 3 * no, *fb = tx + 4 * no, *fc = tx + 5 * no;
        zc *wv = malloc(sizeof(zc) * 2 * no), *uv = wv + no;    /* the k of one (i, j) */
        /* the sums of one pair (b, c) over the pairs of occupied spinors,
         * for the rows (bc), (ac), (ab) of <bc||jk> (l1) and of t2 (dvo) */
        zc *acc = malloc(sizeof(zc) * 6 * no);
        zc *la = acc, *lb = acc + no, *lc = acc + 2 * no;
        zc *da = acc + 3 * no, *db = acc + 4 * no, *dc = acc + 5 * no;
        zc *fr = has_f ? malloc(sizeof(zc) * 3 * npo) : NULL;   /* the f terms of the rows */
        zc *tr = g3 ? malloc(sizeof(zc) * 3 * npo) : NULL;      /* the t1 terms of the rows */
        /* the sums of this thread */
        zc *l1t = calloc(no * ns, sizeof(zc)), *dvt = calloc(ns * no, sizeof(zc));
        zc *got = calloc(no, sizeof(zc)), *gvt = calloc(ns, sizeof(zc));
        #pragma omp for schedule(static)
        for (int64_t y = b0; y < b1; y++)
            for (int64_t i = 0; i < no; i++) {
                const int64_t off = (y * no + i) * n1 + (y - b0) * npo;
                if (m3)
                    for (int k = 0; k < 3; k++)
                        memset((double *)G1p[k] + off, 0, sizeof(double) * npo);
                else
                    memset((zc *)G1p[0] + off, 0, sizeof(zc) * npo);
            }
        #pragma omp for schedule(dynamic, chunk)
        for (int64_t pbc = pbeg; pbc < pend; pbc++) {
            int64_t b = (int64_t)((1.0 + sqrt(1.0 + 8.0 * (double)pbc)) / 2.0);
            while (b * (b - 1) / 2 > pbc)
                b--;
            while ((b + 1) * b / 2 <= pbc)
                b++;
            const int64_t c = pbc - b * (b - 1) / 2;
            /* Z[i, (jk)] = x_{a,(bc)} - x_{b,(ac)} + x_{c,(ab)} */
            for (int64_t i = 0; i < no; i++) {
                const int64_t o1 = ((pbc - p0) * no + i) * s1 + c1;
                const int64_t o2 = (c * no + i) * n1 + (b - b0) * npo;
                const int64_t o3 = c >= b0 ? (b * no + i) * n1 + (c - b0) * npo
                                           : ((b - b0) * no + i) * n2 + c * npo;
                zc *z = Z + i * npo;
                if (m3) {
                    for (int k = 0; k < 3; k++) {
                        const double *r1 = (const double *)X1p[k] + o1;
                        const double *r2 = (const double *)G1p[k] + o2;
                        const double *r3 = (const double *)(c >= b0 ? G1p[k] : G2p[k]) + o3;
                        double *d = zp + k * npo;
                        for (int64_t q = 0; q < npo; q++)
                            d[q] = r1[q] - r2[q] + r3[q];
                    }
                    const double *d0 = zp, *d1 = zp + npo, *d2 = zp + 2 * npo;
                    for (int64_t q = 0; q < npo; q++)
                        z[q] = (d0[q] - d1[q]) + (d2[q] - d0[q] - d1[q]) * I;
                } else {
                    const zc *r1 = (const zc *)X1p[0] + o1, *r2 = (const zc *)G1p[0] + o2;
                    const zc *r3 = (const zc *)(c >= b0 ? G1p[0] : G2p[0]) + o3;
                    for (int64_t q = 0; q < npo; q++)
                        z[q] = r1[q] - r2[q] + r3[q];
                }
            }
            const zc *vbc = VpT + pbc * npo, *tbc = T2pT + pbc * npo;
            const zc *vac = VpT + (pa + c) * npo, *tac = T2pT + (pa + c) * npo;
            const zc *vab = VpT + (pa + b) * npo, *tab = T2pT + (pa + b) * npo;
            for (int64_t k = 0; k < no; k++) {
                ta[k] = t1[k * nv + a];
                tb[k] = t1[k * nv + b];
                tc[k] = t1[k * nv + c];
                fa[k] = fvo[a * no + k];
                fb[k] = fvo[b * no + k];
                fc[k] = fvo[c * no + k];
            }
            memset(acc, 0, sizeof(zc) * 6 * no);
            if (has_f)
                memset(fr, 0, sizeof(zc) * 3 * npo);
            if (g3)
                memset(tr, 0, sizeof(zc) * 3 * npo);
            zc xs = 0.0;                                /* sum of conj(u) w / D^2 */
            const double eabc = ev[a] + ev[b] + ev[c];
            for (int64_t i = 2; i < no; i++)
                for (int64_t j = 1; j < i; j++) {
                    const int64_t pij = i * (i - 1) / 2 + j;
                    const int64_t oj = j * (j - 1) / 2, oi = i * (i - 1) / 2;
                    zc *zi = Z + i * npo + oj, *zj = Z + j * npo + oi;
                    zc *zk = Z + pij;                               /* Z[k, (ij)]: stride npo */
                    const zc tai = ta[i], tbi = tb[i], tci = tc[i];
                    const zc taj = ta[j], tbj = tb[j], tcj = tc[j];
                    const zc vbcij = vbc[pij], vacij = vac[pij], vabij = vab[pij];
                    const zc tbcij = tbc[pij], tacij = tac[pij], tabij = tab[pij];
                    const double dij = eo[i] + eo[j] - eabc;
                    for (int64_t k = 0; k < j; k++) {
                        const zc w = zi[k] - zj[k] + zk[k * npo];
                        wv[k] = w;
                        uv[k] = w + tai * vbc[oj + k] - tbi * vac[oj + k] + tci * vab[oj + k]
                                - taj * vbc[oi + k] + tbj * vac[oi + k] - tcj * vab[oi + k]
                                + ta[k] * vbcij - tb[k] * vacij + tc[k] * vabij;
                    }
                    if (has_f) {
                        const zc fai = fa[i], fbi = fb[i], fci = fc[i];
                        const zc faj = fa[j], fbj = fb[j], fcj = fc[j];
                        for (int64_t k = 0; k < j; k++)
                            uv[k] += fai * tbc[oj + k] - fbi * tac[oj + k] + fci * tab[oj + k]
                                     - faj * tbc[oi + k] + fbj * tac[oi + k] - fcj * tab[oi + k]
                                     + fa[k] * tbcij - fb[k] * tacij + fc[k] * tabij;
                    }
                    /* the energy and the diagonals of the density; M[i,(jk)] = m,
                     * M[j,(ik)] = -m, M[k,(ij)] = m over the three Z of (ijk),
                     * which are read here only; conj(w) / D for the other sums */
                    zc g = 0.0;
                    for (int64_t k = 0; k < j; k++) {
                        const zc w = wv[k], u = uv[k];
                        const double rd = 1.0 / (dij + eo[k]);
                        er += (creal(u) * creal(w) + cimag(u) * cimag(w)) * rd;
                        ei += (creal(u) * cimag(w) - cimag(u) * creal(w)) * rd;
                        const zc e3 = conj(u) * w * (rd * rd);
                        const zc m = conj(w + u) * rd;
                        g += e3;
                        got[k] += e3;
                        zi[k] = m;
                        zj[k] = -m;
                        zk[k * npo] = m;
                        wv[k] = conj(w) * rd;
                    }
                    got[i] += g;
                    got[j] += g;
                    xs += g;
                    /* conj(w[ijk]) with the pair (jk) for i, -(ik) for j, (ij) for k */
                    la[i] += dotu(wv, vbc + oj, j);
                    lb[i] += dotu(wv, vac + oj, j);
                    lc[i] += dotu(wv, vab + oj, j);
                    la[j] -= dotu(wv, vbc + oi, j);
                    lb[j] -= dotu(wv, vac + oi, j);
                    lc[j] -= dotu(wv, vab + oi, j);
                    da[i] += dotu(wv, tbc + oj, j);
                    db[i] += dotu(wv, tac + oj, j);
                    dc[i] += dotu(wv, tab + oj, j);
                    da[j] -= dotu(wv, tbc + oi, j);
                    db[j] -= dotu(wv, tac + oi, j);
                    dc[j] -= dotu(wv, tab + oi, j);
                    for (int64_t k = 0; k < j; k++) {
                        const zc wc = wv[k];
                        la[k] += wc * vbcij;
                        lb[k] += wc * vacij;
                        lc[k] += wc * vabij;
                        da[k] += wc * tbcij;
                        db[k] += wc * tacij;
                        dc[k] += wc * tabij;
                    }
                    if (has_f)          /* f[s,k] conj(w[ijk]) */
                        role_sums(wv, fa, fb, fc, i, j, npo, fr);
                    if (g3)             /* t1[k,s] conj(w[ijk]) */
                        role_sums(wv, ta, tb, tc, i, j, npo, tr);
                }
            /* M vanishes where its single index is one of the pair: these Z
             * belong to no (ijk) */
            for (int64_t i = 1; i < no; i++)
                for (int64_t j = 0; j < i; j++) {
                    const int64_t pij = i * (i - 1) / 2 + j;
                    Z[i * npo + pij] = 0.0;
                    Z[j * npo + pij] = 0.0;
                }
            /* M, -M, M in place of the three x */
            for (int64_t i = 0; i < no; i++) {
                const int64_t o1 = ((pbc - p0) * no + i) * s1 + c1;
                const int64_t o2 = (c * no + i) * n1 + (b - b0) * npo;
                const int64_t o3 = c >= b0 ? (b * no + i) * n1 + (c - b0) * npo
                                           : ((b - b0) * no + i) * n2 + c * npo;
                const zc *z = Z + i * npo;
                if (m3) {
                    void **G3p = c >= b0 ? G1p : G2p;
                    double *x0 = (double *)X1p[0] + o1, *x1 = (double *)X1p[1] + o1;
                    double *x2 = (double *)X1p[2] + o1;
                    double *g0 = (double *)G1p[0] + o2, *g1 = (double *)G1p[1] + o2;
                    double *g2 = (double *)G1p[2] + o2;
                    double *h0 = (double *)G3p[0] + o3, *h1 = (double *)G3p[1] + o3;
                    double *h2 = (double *)G3p[2] + o3;
                    for (int64_t q = 0; q < npo; q++) {
                        const double re = creal(z[q]), im = cimag(z[q]);
                        x0[q] = re;
                        x1[q] = im;
                        x2[q] = re + im;
                        g0[q] = -re;
                        g1[q] = -im;
                        g2[q] = -(re + im);
                        h0[q] = re;
                        h1[q] = im;
                        h2[q] = re + im;
                    }
                } else {
                    zc *r1 = (zc *)X1p[0] + o1, *r2 = (zc *)G1p[0] + o2;
                    zc *r3 = (zc *)(c >= b0 ? G1p[0] : G2p[0]) + o3;
                    for (int64_t q = 0; q < npo; q++) {
                        r1[q] = z[q];
                        r2[q] = -z[q];
                        r3[q] = z[q];
                    }
                }
            }
            /* the sums of the pair with the signs of the three roles; a is
             * the last of the ns virtuals */
            for (int64_t x = 0; x < no; x++) {
                l1t[x * ns + b1] += la[x];
                l1t[x * ns + b] -= lb[x];
                l1t[x * ns + c] += lc[x];
                dvt[b1 * no + x] += conj(da[x]);
                dvt[b * no + x] -= conj(db[x]);
                dvt[c * no + x] += conj(dc[x]);
            }
            gvt[b1] += conj(xs);
            gvt[b] += conj(xs);
            gvt[c] += conj(xs);
            if (has_f)
                role_rows(fr, lt, pbc, pa + c, pa + b, npo);
            if (g3)
                role_rows(tr, g3, pbc, pa + c, pa + b, npo);
        }
        #pragma omp critical(zc_t_lam_sums)
        {
            for (int64_t x = 0; x < no; x++) {
                for (int64_t s = 0; s < b1; s++)
                    l1[x * nv + s] += l1t[x * ns + s];
                l1[x * nv + a] += l1t[x * ns + b1];
                dvo[a * no + x] += dvt[b1 * no + x];
                goo[x] += got[x];
            }
            for (int64_t s = 0; s < b1; s++) {
                for (int64_t x = 0; x < no; x++)
                    dvo[s * no + x] += dvt[s * no + x];
                gvv[s] += gvt[s];
            }
            gvv[a] += gvt[b1];
        }
        free(Z);
        free(zp);
        free(tx);
        free(wv);
        free(acc);
        free(fr);
        free(tr);
        free(l1t);
        free(dvt);
        free(got);
        free(gvt);
    }
    out[0] = er;
    out[1] = ei;
}

/* The product of the triples of a batch with <ke||pq>, as it comes from GEMM
 * transposed - Q[e, (s - sa, (ij))], ld elements per row; complex (q0), or
 * with m3 its three real products -, antisymmetrized over the virtual pair
 * into lt[(a>b), (i>j)]:
 *
 *   lt[(s>e), (ij)] += Q[e, (s, (ij))],    lt[(e>s), (ij)] -= Q[e, (s, (ij))]
 *
 * for sa <= s < sb.  Two s of the block meet in one row: the s one after the
 * other, the rows of one s on the threads. */
void zc_t_lam_particle(const void *q0, const void *q1, const void *q2, int m3, int64_t ld,
                       int64_t nv, int64_t npo, int64_t sa, int64_t sb, zc *lt)
{
    /* a few rows are not worth a team of threads */
    #pragma omp parallel if(nv * npo > 65536)
    for (int64_t s = sa; s < sb; s++) {
        #pragma omp for schedule(static)
        for (int64_t e = 0; e < nv; e++) {
            if (e == s)
                continue;
            const int64_t hi = e < s ? s : e, lo = e < s ? e : s;
            const double sg = e < s ? 1.0 : -1.0;
            const int64_t off = e * ld + (s - sa) * npo;
            zc *r = lt + (hi * (hi - 1) / 2 + lo) * npo;
            if (m3) {
                const double *d0 = (const double *)q0 + off, *d1 = (const double *)q1 + off;
                const double *d2 = (const double *)q2 + off;
                for (int64_t x = 0; x < npo; x++)
                    r[x] += sg * ((d0[x] - d1[x]) + (d2[x] - d0[x] - d1[x]) * I);
            } else {
                const zc *q = (const zc *)q0 + off;
                for (int64_t x = 0; x < npo; x++)
                    r[x] += sg * q[x];
            }
        }
    }
}

/* The product of the triples of a batch with <mn||js> - Y[(r - ra, i), off + j]
 * in rows of ld elements, for the pairs ra <= r < rb of virtual spinors;
 * complex, or its three real products -, antisymmetrized over the occupied
 * pair into lt[(a>b), (i>j)]: lt[r, (ij)] += Y[(r, i), j] - Y[(r, j), i]. */
void zc_t_lam_hole(const void *y0, const void *y1, const void *y2, int m3, int64_t no,
                   int64_t ld, int64_t off, int64_t ra, int64_t rb, zc *lt)
{
    const int64_t npo = no * (no - 1) / 2;
    #pragma omp parallel for schedule(static) if((rb - ra) * no * no > 65536)
    for (int64_t r = ra; r < rb; r++) {
        const int64_t base = (r - ra) * no * ld + off;
        zc *row = lt + r * npo;
        if (m3) {
            const double *d0 = (const double *)y0 + base, *d1 = (const double *)y1 + base;
            const double *d2 = (const double *)y2 + base;
            for (int64_t i = 1; i < no; i++)
                for (int64_t j = 0; j < i; j++) {
                    const int64_t x = i * ld + j, y = j * ld + i;
                    const double p1 = d0[x] - d0[y], p2 = d1[x] - d1[y], p3 = d2[x] - d2[y];
                    row[i * (i - 1) / 2 + j] += (p1 - p2) + (p3 - p1 - p2) * I;
                }
        } else {
            const zc *yy = (const zc *)y0 + base;
            for (int64_t i = 1; i < no; i++)
                for (int64_t j = 0; j < i; j++)
                    row[i * (i - 1) / 2 + j] += yy[i * ld + j] - yy[j * ld + i];
        }
    }
}

/* g2[j, s, (mn)] -= q[nv + j, (s - sa, (mn))] for all occupied j and
 * sa <= s < sb: the rows of the product A^T M that belong to the t2 in A
 * (q: no + nv rows of ld elements; complex, or its three real products) are
 * minus the density of <js||mn>. */
void zc_t_den_hole(const void *q0, const void *q1, const void *q2, int m3, int64_t ld,
                   int64_t nv, int64_t no, int64_t npo, int64_t sa, int64_t sb, zc *g2)
{
    const int64_t nc = (sb - sa) * npo;
    #pragma omp parallel for schedule(static) if(no * nc > 65536)
    for (int64_t j = 0; j < no; j++) {
        zc *d = g2 + (j * nv + sa) * npo;
        const int64_t off = (nv + j) * ld;
        if (m3) {
            const double *d0 = (const double *)q0 + off, *d1 = (const double *)q1 + off;
            const double *d2 = (const double *)q2 + off;
            for (int64_t x = 0; x < nc; x++)
                d[x] -= (d0[x] - d1[x]) + (d2[x] - d0[x] - d1[x]) * I;
        } else {
            const zc *q = (const zc *)q0 + off;
            for (int64_t x = 0; x < nc; x++)
                d[x] -= q[x];
        }
    }
}

/* dst[r, c] += y[r, c] for nrow rows and the first ncol of the ld columns of
 * y (complex, or its three real products): the product of the triples of a
 * batch with t2, added to the rows of the density of <bc||ei>. */
void zc_t_den_rows(const void *y0, const void *y1, const void *y2, int m3, int64_t nrow,
                   int64_t ld, int64_t ncol, zc *dst)
{
    #pragma omp parallel for schedule(static) if(nrow * ncol > 65536)
    for (int64_t r = 0; r < nrow; r++) {
        zc *d = dst + r * ncol;
        if (m3) {
            const double *d0 = (const double *)y0 + r * ld, *d1 = (const double *)y1 + r * ld;
            const double *d2 = (const double *)y2 + r * ld;
            for (int64_t x = 0; x < ncol; x++)
                d[x] += (d0[x] - d1[x]) + (d2[x] - d0[x] - d1[x]) * I;
        } else {
            const zc *yy = (const zc *)y0 + r * ld;
            for (int64_t x = 0; x < ncol; x++)
                d[x] += yy[x];
        }
    }
}

/* dst[k, e, q] = src[q - q0, k, e] for the pairs q0 <= q < q1: the density of
 * <bc||ek> from the layout of the batches (rows of pairs) into that of the
 * stored integrals, [k, e, (b>c)] with npv pairs. */
void zc_t_den_sort(const zc *src, int64_t npv, int64_t no, int64_t nv, int64_t q0, int64_t q1,
                   zc *dst)
{
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < no; k++)
        for (int64_t qa = q0; qa < q1; qa += 64) {      /* a tile of pairs in the cache */
            const int64_t qb = qa + 64 < q1 ? qa + 64 : q1;
            for (int64_t e = 0; e < nv; e++) {
                zc *d = dst + (k * nv + e) * npv;
                for (int64_t q = qa; q < qb; q++)
                    d[q] = src[((q - q0) * no + k) * nv + e];
            }
        }
}
