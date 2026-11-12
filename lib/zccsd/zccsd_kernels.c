/*
 * zccsd_kernels.c -- kernels of the spinor CCSD amplitude equations
 * (zccsdt/zccsd/amps.py): the passes over arrays of o^2 v^2 elements that
 * numpy does with fancy indexing on one thread, and the products with
 * <ma||ef> that want the packed integrals unpacked in the cache.
 *
 * Pairs are i > j and a > b in numpy.tril_indices(n, -1) order,
 * (i, j) -> i (i - 1) / 2 + j.  The "ring layout" of a four-index array is
 * the matrix [(i a), (j b)]: row i nv + a, column j nv + b.
 *
 * Compiled with -fcx-limited-range: complex products are four real
 * multiplications, without the checks for infinities of Annex G.
 */

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef double complex zc;

/* ------------------------------------------------------------------ vectors */

/* dst = src, or its complex conjugate, on all threads (first touch included). */
void zc_copy(const zc *src, int64_t n, int conj_src, zc *dst)
{
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < n; k++)
        dst[k] = conj_src ? conj(src[k]) : src[k];
}

/* out = sum_k x[k] y[k], not conjugated. */
void zc_dotu(const zc *x, const zc *y, int64_t n, zc *out)
{
    double re = 0.0, im = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:re, im)
    for (int64_t k = 0; k < n; k++) {
        const zc p = x[k] * y[k];
        re += creal(p);
        im += cimag(p);
    }
    *out = re + im * I;
}

/* sum_k |x[k] - y[k]|^2 */
double zc_diff_norm2(const zc *x, const zc *y, int64_t n)
{
    double s = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:s)
    for (int64_t k = 0; k < n; k++) {
        const zc d = x[k] - y[k];
        s += creal(d) * creal(d) + cimag(d) * cimag(d);
    }
    return s;
}

/* The parts of a complex m x n matrix x (leading dimension ld) that a
 * product by three real multiplications takes: xr = Re x, xi = Im x,
 * xs = Re x + Im x, contiguous m x n arrays. */
void zc_split3(const zc *x, int64_t m, int64_t n, int64_t ld, double *xr, double *xi,
               double *xs)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; i++) {
        const zc *row = x + i * ld;
        double *r = xr + i * n, *q = xi + i * n, *t = xs + i * n;
        for (int64_t j = 0; j < n; j++) {
            r[j] = creal(row[j]);
            q[j] = cimag(row[j]);
            t[j] = r[j] + q[j];
        }
    }
}

/* The complex product from its three real ones, p1 = Ar Br, p2 = Ai Bi,
 * p3 = (Ar + Ai) (Br + Bi):  c = alpha ((p1 - p2) + i (p3 - p1 - p2)),
 * added to c with acc. */
void zc_combine3(const double *p1, const double *p2, const double *p3, int64_t n,
                 double alpha, int acc, zc *c)
{
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < n; k++) {
        const zc val = alpha * ((p1[k] - p2[k]) + (p3[k] - p1[k] - p2[k]) * I);
        c[k] = acc ? c[k] + val : val;
    }
}

/* zc_combine3 for m rows of n elements that are blocks of larger row-major
 * arrays: the rows of the three products ldp elements apart, those of c ldc
 * apart (the columns of one range of a matrix that is filled range by range). */
void zc_combine3_ld(const double *p1, const double *p2, const double *p3, int64_t m, int64_t n,
                    int64_t ldp, double alpha, int acc, zc *c, int64_t ldc)
{
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < m; r++) {
        const double *a = p1 + r * ldp, *b = p2 + r * ldp, *d = p3 + r * ldp;
        zc *x = c + r * ldc;
        for (int64_t k = 0; k < n; k++) {
            const zc val = alpha * ((a[k] - b[k]) + (d[k] - a[k] - b[k]) * I);
            x[k] = acc ? x[k] + val : val;
        }
    }
}

/* ---------------------------------------------------------------- amplitudes */

/* R[(i>j), (a>b)] += fac (x[i,a] y[j,b] - x[i,b] y[j,a] - x[j,a] y[i,b] + x[j,b] y[i,a]):
 * the antisymmetrized product of two o x v matrices (tau from t2: x = y = t1,
 * fac = 1/2). */
void zc_outer_pp(const zc *x, const zc *y, int64_t no, int64_t nv, double fac, zc *R)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 1; i < no; i++)
        for (int64_t j = 0; j < i; j++) {
            zc *r = R + (i * (i - 1) / 2 + j) * npv;
            const zc *xi = x + i * nv, *xj = x + j * nv;
            const zc *yi = y + i * nv, *yj = y + j * nv;
            for (int64_t a = 1; a < nv; a++) {
                const zc p = fac * xi[a], q = fac * xj[a];
                const zc s = fac * yi[a], t = fac * yj[a];
                for (int64_t b = 0; b < a; b++)
                    r[b] += p * yj[b] - xi[b] * t - q * yi[b] + xj[b] * s;
                r += a;
            }
        }
}

/* The columns j0 <= j < j1 of t2 in the ring layout,
 *     out[(n f), (j b)] = t2[n, j, f, b],
 * from t2p[(i>j), (a>b)]: no nv rows, (j1 - j0) nv columns.  The matrix is
 * symmetric, so its transpose gives the rows (j b) as well. */
void zc_ring_cols(const zc *t2p, int64_t no, int64_t nv, int64_t j0, int64_t j1, zc *out)
{
    const int64_t npv = nv * (nv - 1) / 2, nc = (j1 - j0) * nv;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < no; n++)
        for (int64_t j = j0; j < j1; j++) {
            zc *blk = out + n * nv * nc + (j - j0) * nv;        /* [f][b] at blk[f nc + b] */
            if (n == j) {
                for (int64_t f = 0; f < nv; f++)
                    memset(blk + f * nc, 0, sizeof(zc) * nv);
                continue;
            }
            const int64_t hi = n > j ? n : j, lo = n > j ? j : n;
            const double sgn = n > j ? 1.0 : -1.0;
            const zc *r = t2p + (hi * (hi - 1) / 2 + lo) * npv;
            for (int64_t f = 0; f < nv; f++) {
                zc *row = blk + f * nc;
                const zc *rf = r + f * (f - 1) / 2;
                for (int64_t b = 0; b < f; b++)
                    row[b] = sgn * rf[b];
                row[f] = 0.0;
                for (int64_t b = f + 1; b < nv; b++)
                    row[b] = -sgn * r[b * (b - 1) / 2 + f];
            }
        }
}

/* zc_ring_cols in the three real parts of zc_split3. */
void zc_ring_cols3(const zc *t2p, int64_t no, int64_t nv, int64_t j0, int64_t j1,
                   double *tr, double *ti, double *ts)
{
    const int64_t npv = nv * (nv - 1) / 2, nc = (j1 - j0) * nv;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t n = 0; n < no; n++)
        for (int64_t j = j0; j < j1; j++) {
            const int64_t off = n * nv * nc + (j - j0) * nv;
            double *br = tr + off, *bi = ti + off, *bs = ts + off;
            if (n == j) {
                for (int64_t f = 0; f < nv; f++) {
                    memset(br + f * nc, 0, sizeof(double) * nv);
                    memset(bi + f * nc, 0, sizeof(double) * nv);
                    memset(bs + f * nc, 0, sizeof(double) * nv);
                }
                continue;
            }
            const int64_t hi = n > j ? n : j, lo = n > j ? j : n;
            const double sgn = n > j ? 1.0 : -1.0;
            const zc *r = t2p + (hi * (hi - 1) / 2 + lo) * npv;
            for (int64_t f = 0; f < nv; f++) {
                double *rr = br + f * nc, *ri = bi + f * nc, *rs = bs + f * nc;
                const zc *rf = r + f * (f - 1) / 2;
                for (int64_t b = 0; b < f; b++) {
                    rr[b] = sgn * creal(rf[b]);
                    ri[b] = sgn * cimag(rf[b]);
                    rs[b] = rr[b] + ri[b];
                }
                rr[f] = ri[f] = rs[f] = 0.0;
                for (int64_t b = f + 1; b < nv; b++) {
                    const zc val = r[b * (b - 1) / 2 + f];
                    rr[b] = -sgn * creal(val);
                    ri[b] = -sgn * cimag(val);
                    rs[b] = rr[b] + ri[b];
                }
            }
        }
}

/* R[(i>j), (a>b)] += alpha (x[i,a,j,b] - x[j,a,i,b] - x[i,b,j,a] + x[j,b,i,a])
 * from the columns j0 <= j < j1 of x in the ring layout: x has no nv rows of
 * nc elements, and the column (j, b) is its element (j - j0) nv + b (a
 * pointer into a wider matrix selects other columns of it).  Every pair
 * takes the terms whose third index lies in the range, so that the calls
 * for all ranges add up to the full antisymmetrization P(ij) P(ab). */
void zc_asym_pack_cols(const zc *x, int64_t nc, int64_t no, int64_t nv, int64_t j0,
                       int64_t j1, double alpha, zc *R)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel
    {
        zc *D = malloc(sizeof(zc) * nv * nv);
        #pragma omp for schedule(dynamic, 1)
        for (int64_t i = 1; i < no; i++)
            for (int64_t j = 0; j < i; j++) {
                const int inj = j >= j0 && j < j1, ini = i >= j0 && i < j1;
                if (!inj && !ini)
                    continue;
                /* D[a, b] = x[i,a,j,b] - x[j,a,i,b] */
                const zc *xi = x + i * nv * nc + (j - j0) * nv;
                const zc *xj = x + j * nv * nc + (i - j0) * nv;
                for (int64_t a = 0; a < nv; a++) {
                    zc *d = D + a * nv;
                    const zc *p = xi + a * nc, *q = xj + a * nc;
                    if (inj && ini)
                        for (int64_t b = 0; b < nv; b++)
                            d[b] = p[b] - q[b];
                    else if (inj)
                        memcpy(d, p, sizeof(zc) * nv);
                    else
                        for (int64_t b = 0; b < nv; b++)
                            d[b] = -q[b];
                }
                zc *r = R + (i * (i - 1) / 2 + j) * npv;
                for (int64_t a = 1; a < nv; a++) {
                    const zc *d = D + a * nv;
                    for (int64_t b = 0; b < a; b++)
                        r[b] += alpha * (d[b] - D[b * nv + a]);
                    r += a;
                }
            }
        free(D);
    }
}

/* R[p, (a>b)] += alpha (x[p,a,b] - x[p,b,a]) for np matrices x whose element
 * (p, a, b) is x[p sp + a sa + b]: sp = nv nv, sa = nv for the layout
 * [p, a, b], sp = nv, sa = np nv for [a, p, b]. */
void zc_asym_pack2(const zc *x, int64_t np, int64_t nv, int64_t sp, int64_t sa,
                   double alpha, zc *R)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel
    {
        zc *D = malloc(sizeof(zc) * nv * nv);
        #pragma omp for schedule(static)
        for (int64_t p = 0; p < np; p++) {
            for (int64_t a = 0; a < nv; a++)
                memcpy(D + a * nv, x + p * sp + a * sa, sizeof(zc) * nv);
            zc *r = R + p * npv;
            for (int64_t a = 1; a < nv; a++) {
                const zc *d = D + a * nv;
                for (int64_t b = 0; b < a; b++)
                    r[b] += alpha * (d[b] - D[b * nv + a]);
                r += a;
            }
        }
        free(D);
    }
}

/* R[p, (a>b)] = x[p,a,b] - x[p,b,a]: zc_asym_pack2 that writes R instead of
 * adding to it (pages of R that have not been touched are touched here, by
 * the threads). */
void zc_asym_pack2_set(const zc *x, int64_t np, int64_t nv, int64_t sp, int64_t sa, zc *R)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel
    {
        zc *D = malloc(sizeof(zc) * nv * nv);
        #pragma omp for schedule(static)
        for (int64_t p = 0; p < np; p++) {
            for (int64_t a = 0; a < nv; a++)
                memcpy(D + a * nv, x + p * sp + a * sa, sizeof(zc) * nv);
            zc *r = R + p * npv;
            for (int64_t a = 1; a < nv; a++) {
                const zc *d = D + a * nv;
                for (int64_t b = 0; b < a; b++)
                    r[b] = d[b] - D[b * nv + a];
                r += a;
            }
        }
        free(D);
    }
}

/* out[f, a] += sum_r sum_c M_r[a, c] q[f, r, c], M_r the antisymmetric matrix
 * of the packed row G[r, (a>c)]: nr rows and nf vectors per row, the element
 * (f, r, c) of q at q[f sf + r sr + c].  Nothing is unpacked: a packed element
 * meets two elements of every vector. */
void zc_tril_matvec(const zc *G, int64_t nr, int64_t nv, const zc *q, int64_t nf,
                    int64_t sf, int64_t sr, zc *out)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel
    {
        zc *acc = calloc((size_t)(nf * nv), sizeof(zc));
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < nr; r++) {
            const zc *g = G + r * npv;
            for (int64_t f = 0; f < nf; f++) {
                const zc *x = q + f * sf + r * sr;
                zc *y = acc + f * nv;
                const zc *ga = g;
                for (int64_t a = 1; a < nv; a++) {
                    const zc xa = x[a];
                    double re = 0.0, im = 0.0;
                    #pragma omp simd reduction(+:re, im)
                    for (int64_t c = 0; c < a; c++) {
                        const zc p = ga[c] * x[c];
                        re += creal(p);
                        im += cimag(p);
                        y[c] -= ga[c] * xa;
                    }
                    y[a] += re + im * I;
                    ga += a;
                }
            }
        }
        #pragma omp critical
        for (int64_t k = 0; k < nf * nv; k++)
            out[k] += acc[k];
        free(acc);
    }
}

/* R[p, (a>b)] += alpha x[p,a,b] for np matrices x whose element (p, a, b)
 * is x[p sp + a sa + b]: sp = nv nv, sa = nv for the layout [p, a, b],
 * sp = nv, sa = np nv for [a, p, b]. */
void zc_pack_tril_add(const zc *x, int64_t np, int64_t nv, int64_t sp, int64_t sa,
                      double alpha, zc *R)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < np; p++) {
        zc *r = R + p * npv;
        const zc *xp = x + p * sp;
        for (int64_t a = 1; a < nv; a++) {
            const zc *d = xp + a * sa;
            for (int64_t b = 0; b < a; b++)
                r[b] += alpha * d[b];
            r += a;
        }
    }
}

/* x[p, a, b] = -x[p, b, a] = in[p, (a>b)], zero diagonal; the element
 * (p, a, b) is x[p sp + a sa + b] as in zc_pack_tril_add. */
void zc_unpack_tril(const zc *in, int64_t np, int64_t nv, int64_t sp, int64_t sa, zc *x)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < np; p++) {
        const zc *r = in + p * npv;
        zc *xp = x + p * sp;
        for (int64_t a = 0; a < nv; a++) {
            zc *row = xp + a * sa;
            const zc *ra = r + a * (a - 1) / 2;
            for (int64_t b = 0; b < a; b++)
                row[b] = ra[b];
            row[a] = 0.0;
            for (int64_t b = a + 1; b < nv; b++)
                row[b] = -r[b * (b - 1) / 2 + a];
        }
    }
}

/* t2p[(i>j), (a>b)] = t2[i, j, a, b]. */
void zc_pack_t2(const zc *t2, int64_t no, int64_t nv, zc *t2p)
{
    const int64_t npv = nv * (nv - 1) / 2, nvv = nv * nv;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 1; i < no; i++)
        for (int64_t j = 0; j < i; j++) {
            zc *r = t2p + (i * (i - 1) / 2 + j) * npv;
            const zc *x = t2 + (i * no + j) * nvv;
            for (int64_t a = 1; a < nv; a++) {
                memcpy(r, x + a * nv, sizeof(zc) * a);
                r += a;
            }
        }
}

/* t2[i, j, a, b], antisymmetric in both pairs, from t2p[(i>j), (a>b)]. */
void zc_unpack_t2(const zc *t2p, int64_t no, int64_t nv, zc *t2)
{
    const int64_t npv = nv * (nv - 1) / 2, nvv = nv * nv;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < no; i++) {
        memset(t2 + (i * no + i) * nvv, 0, sizeof(zc) * nvv);
        for (int64_t j = 0; j < i; j++) {
            const zc *r = t2p + (i * (i - 1) / 2 + j) * npv;
            zc *tij = t2 + (i * no + j) * nvv, *tji = t2 + (j * no + i) * nvv;
            for (int64_t a = 0; a < nv; a++) {
                zc *p = tij + a * nv, *q = tji + a * nv;
                const zc *ra = r + a * (a - 1) / 2;
                for (int64_t b = 0; b < a; b++) {
                    p[b] = ra[b];
                    q[b] = -ra[b];
                }
                p[a] = q[a] = 0.0;
                for (int64_t b = a + 1; b < nv; b++) {
                    const zc val = r[b * (b - 1) / 2 + a];
                    p[b] = -val;
                    q[b] = val;
                }
            }
        }
    }
}

/* R[(i>j), (a>b)] /= eo[i] + eo[j] - ev[a] - ev[b]. */
void zc_div_packed(zc *R, const double *eo, const double *ev, int64_t no, int64_t nv)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 1; i < no; i++)
        for (int64_t j = 0; j < i; j++) {
            zc *r = R + (i * (i - 1) / 2 + j) * npv;
            const double eij = eo[i] + eo[j];
            for (int64_t a = 1; a < nv; a++) {
                const double ea = eij - ev[a];
                for (int64_t b = 0; b < a; b++)
                    r[b] /= ea - ev[b];
                r += a;
            }
        }
}

/* out[i, a] += sum_{m e} t2[i, m, a, e] F[m, e] from the packed amplitudes. */
void zc_t2p_matvec(const zc *t2p, const zc *F, int64_t no, int64_t nv, zc *out)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel
    {
        zc *loc = calloc(no * nv, sizeof(zc));
        #pragma omp for schedule(dynamic, 1) nowait
        for (int64_t i = 1; i < no; i++)
            for (int64_t m = 0; m < i; m++) {
                const zc *r = t2p + (i * (i - 1) / 2 + m) * npv;
                zc *oi = loc + i * nv, *om = loc + m * nv;
                const zc *Fi = F + i * nv, *Fm = F + m * nv;
                for (int64_t a = 1; a < nv; a++) {
                    const zc fma = Fm[a], fia = Fi[a];
                    zc si = 0.0, sm = 0.0;
                    for (int64_t e = 0; e < a; e++) {
                        const zc val = r[e];
                        si += val * Fm[e];
                        sm += val * Fi[e];
                        oi[e] -= val * fma;
                        om[e] += val * fia;
                    }
                    oi[a] += si;
                    om[a] -= sm;
                    r += a;
                }
            }
        #pragma omp critical
        for (int64_t x = 0; x < no * nv; x++)
            out[x] += loc[x];
        free(loc);
    }
}

/* R[rows[k]] += signs[k] Y[k] (the complex conjugate of Y[k] with conj_y)
 * for n rows of ncol elements; rows[k] < 0 is skipped, the rows are distinct. */
void zc_add_rows(zc *R, int64_t ncol, const int64_t *rows, const double *signs,
                 const zc *Y, int64_t n, int conj_y)
{
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < n; k++) {
        if (rows[k] < 0)
            continue;
        zc *r = R + rows[k] * ncol;
        const zc *y = Y + k * ncol;
        const double s = signs[k];
        if (conj_y)
            for (int64_t q = 0; q < ncol; q++)
                r[q] += s * conj(y[q]);
        else
            for (int64_t q = 0; q < ncol; q++)
                r[q] += s * y[q];
    }
}

/* ------------------------------------------------------------------ <ma||ef> */

/* The <mb||ef> t1 terms of one occupied m from the integrals packed over
 * e > f, Om[b, (e>f)] = <mb||ef>:
 *
 *     A[b, e, j] = sum_f <mb||ef> t1[j, f]
 *     Wm[e, (j b)] += A[b, e, j]      (the rows (m e) of W in the ring layout)
 *     fv[b, e]      = A[b, e, m]
 *     x1[j, b]      = sum_e t1[m, e] A[b, e, j]
 *
 * A few rows b per task.  A row is the lower triangle L[e, f] (e > f) of an
 * antisymmetric matrix; its product with t1 is taken from the packed row
 * itself, like a packed symmetric matrix-vector product: for every f the
 * segment L[f, e < f] gives the dot product A[f, j] += sum_e L[f, e] t1[j, e]
 * and the update A[e, j] -= L[f, e] t1[j, f] in one pass over contiguous
 * real and imaginary parts.  The results of the task go to W in runs of
 * consecutive b.  Nothing of nv^2 elements is unpacked. */
void zc_ovvv_t1(const zc *Om, const zc *t1, int64_t nv, int64_t no, int64_t m,
                zc *Wm, zc *fv, zc *x1)
{
    const int64_t npv = nv * (nv - 1) / 2, n1 = no * nv;
    int64_t ch = (1 << 19) / (16 * n1);             /* rows per task: a buffer of 512 kB */
    ch = ch < 1 ? 1 : ch > 8 ? 8 : ch;
    double *tr = malloc(sizeof(double) * n1);
    double *ti = malloc(sizeof(double) * n1);
    for (int64_t x = 0; x < n1; x++) {
        tr[x] = creal(t1[x]);
        ti[x] = cimag(t1[x]);
    }
    #pragma omp parallel
    {
        double *lr = malloc(sizeof(double) * npv);
        double *li = malloc(sizeof(double) * npv);
        double *ar = malloc(sizeof(double) * n1);   /* A[j][e] of the row b */
        double *ai = malloc(sizeof(double) * n1);
        zc *buf = malloc(sizeof(zc) * n1 * ch);
        #pragma omp for schedule(dynamic, 1)
        for (int64_t b0 = 0; b0 < nv; b0 += ch) {
            const int64_t c = (b0 + ch < nv ? b0 + ch : nv) - b0;
            for (int64_t b = b0; b < b0 + c; b++) {
                const zc *row = Om + b * npv;
                for (int64_t q = 0; q < npv; q++) {
                    lr[q] = creal(row[q]);
                    li[q] = cimag(row[q]);
                }
                memset(ar, 0, sizeof(double) * n1);
                memset(ai, 0, sizeof(double) * n1);
                for (int64_t f = 1; f < nv; f++) {
                    const double *pr = lr + f * (f - 1) / 2, *pi = li + f * (f - 1) / 2;
                    for (int64_t j = 0; j < no; j++) {
                        const double *yr = tr + j * nv, *yi = ti + j * nv;
                        double *zr = ar + j * nv, *zi = ai + j * nv;
                        const double tfr = yr[f], tfi = yi[f];
                        double sr = 0.0, si = 0.0;
                        #pragma omp simd reduction(+:sr, si)
                        for (int64_t e = 0; e < f; e++) {
                            sr += pr[e] * yr[e] - pi[e] * yi[e];
                            si += pr[e] * yi[e] + pi[e] * yr[e];
                            zr[e] -= pr[e] * tfr - pi[e] * tfi;
                            zi[e] -= pr[e] * tfi + pi[e] * tfr;
                        }
                        zr[f] += sr;
                        zi[f] += si;
                    }
                }
                const double *mr = tr + m * nv, *mi = ti + m * nv;
                for (int64_t j = 0; j < no; j++) {
                    const double *zr = ar + j * nv, *zi = ai + j * nv;
                    double sr = 0.0, si = 0.0;
                    #pragma omp simd reduction(+:sr, si)
                    for (int64_t e = 0; e < nv; e++) {
                        sr += mr[e] * zr[e] - mi[e] * zi[e];
                        si += mr[e] * zi[e] + mi[e] * zr[e];
                    }
                    x1[j * nv + b] = sr + si * I;
                    for (int64_t e = 0; e < nv; e++)
                        buf[(e * no + j) * c + (b - b0)] = zr[e] + zi[e] * I;
                }
                for (int64_t e = 0; e < nv; e++)
                    fv[b * nv + e] = ar[m * nv + e] + ai[m * nv + e] * I;
            }
            for (int64_t e = 0; e < nv; e++)
                for (int64_t j = 0; j < no; j++) {
                    zc *w = Wm + e * n1 + j * nv + b0;
                    const zc *s = buf + (e * no + j) * c;
                    for (int64_t x = 0; x < c; x++)
                        w[x] += s[x];
                }
        }
        free(lr);
        free(li);
        free(ar);
        free(ai);
        free(buf);
    }
    free(tr);
    free(ti);
}

/* ------------------------------------------------------------------ AO ladder */

/* Rows of the two AO operators of the ladder from a block of AO integrals
 *
 *   E[f, x, t, y] = (f0 + f, x | t0 + t, y),   f < nf, t < nt, x and y < m
 *
 * (m covers the first and the third index: f0 + nf <= m, t0 + nt <= m).  With
 * M[x, y] = E[f, x, t, y] for one pair of first and third index,
 *
 *   same: M[x, y] - M[y, x] over x > y    the row of the pair (f0 + f > t0 + t)
 *   plus: M[x, y] + M[y, x] over x >= y   the row of the pair (f0 + f >= t0 + t)
 *
 * the antisymmetrized and the symmetrized integral on the pairs (x, y) in
 * numpy.tril_indices order: m (m - 1) / 2 and m (m + 1) / 2 elements, the
 * first elements of rows of lds and ldp elements.  The rows of the pairs
 * that exist follow one another in the order (f, t).
 *
 * packed: the third index runs over all m functions (t0 = 0, nt = m) and the
 * integrals come with their symmetry in the last two indices,
 * E[f, x, (t >= y)] with m (m + 1) / 2 elements for every (f, x). */
void zc_ladder_rows(const double *E, int packed, int64_t nf, int64_t f0, int64_t nt,
                    int64_t t0, int64_t m, double *same, int64_t lds, double *plus,
                    int64_t ldp)
{
    const int64_t mm = m * (m + 1) / 2;
    int64_t *offs = malloc(sizeof(int64_t) * 2 * (nf + 1));
    int64_t *offp = offs + nf + 1;
    offs[0] = offp[0] = 0;
    for (int64_t f = 0; f < nf; f++) {
        int64_t ns = f0 + f - t0, np = f0 + f + 1 - t0;         /* t0 + t < f0 + f; <= */
        ns = ns < 0 ? 0 : (ns > nt ? nt : ns);
        np = np < 0 ? 0 : (np > nt ? nt : np);
        offs[f + 1] = offs[f] + ns;
        offp[f + 1] = offp[f] + np;
    }
    #pragma omp parallel
    {
        double *M = malloc(sizeof(double) * m * m);
        #pragma omp for schedule(dynamic, 4)
        for (int64_t u = 0; u < nf * nt; u++) {
            const int64_t f = u / nt, t = u % nt;
            if (t >= offp[f + 1] - offp[f])
                continue;                                       /* t0 + t > f0 + f */
            if (packed)
                for (int64_t x = 0; x < m; x++) {
                    const double *e = E + (f * m + x) * mm;
                    double *mx = M + x * m;
                    memcpy(mx, e + t * (t + 1) / 2, sizeof(double) * (t + 1));
                    for (int64_t y = t + 1; y < m; y++)
                        mx[y] = e[y * (y + 1) / 2 + t];
                }
            else
                for (int64_t x = 0; x < m; x++)
                    memcpy(M + x * m, E + ((f * m + x) * nt + t) * m, sizeof(double) * m);
            double *p = plus + (offp[f] + t) * ldp;
            double *a = t < offs[f + 1] - offs[f] ? same + (offs[f] + t) * lds : NULL;
            int64_t q = 0, r = 0;
            for (int64_t x = 0; x < m; x++) {
                const double *mx = M + x * m;
                if (a)
                    for (int64_t y = 0; y < x; y++)
                        a[q++] = mx[y] - M[y * m + x];
                for (int64_t y = 0; y < x; y++)
                    p[r++] = mx[y] + M[y * m + x];
                p[r++] = 2.0 * mx[x];
            }
        }
        free(M);
    }
    free(offs);
}

/* The back-transformed amplitudes X[p] of np pairs, matrices over (spin, AO)
 * with n2 = 2 nao functions, as the real rows the AO operators act on.  With
 * aa, bb, ab the spin blocks of X[p]:
 *
 *   AR[k np + p, (mu>nu)], k = 0, 1, 2: aa, bb and (ab - ab^T) / 2
 *   SR[p, (mu>=nu)]:                    (ab + ab^T) / 2
 *
 * real parts; the imaginary parts follow in the rows 3 np + ... of AR and
 * np + ... of SR.  (mu>nu) in numpy.tril_indices(nao, -1) order, (mu>=nu)
 * in numpy.tril_indices(nao) order.  The element (p, x, y) is
 * X[p sp + x sx + y]: sp = n2 n2, sx = n2 for the layout [p, x, y],
 * sp = n2, sx = np n2 for [x, p, y]. */
void zc_ladder_pack(const zc *X, int64_t np, int64_t nao, int64_t sp, int64_t sx,
                    double *AR, double *SR)
{
    const int64_t n2 = 2 * nao, npair = nao * (nao - 1) / 2, nsym = nao * (nao + 1) / 2;
    #pragma omp parallel
    {
        zc *x = malloc(sizeof(zc) * n2 * n2);
        #pragma omp for schedule(static)
        for (int64_t p = 0; p < np; p++) {
            for (int64_t r = 0; r < n2; r++)
                memcpy(x + r * n2, X + p * sp + r * sx, sizeof(zc) * n2);
            double *aar = AR + p * npair, *aai = AR + (3 * np + p) * npair;
            double *bbr = AR + (np + p) * npair, *bbi = AR + (4 * np + p) * npair;
            double *abr = AR + (2 * np + p) * npair, *abi = AR + (5 * np + p) * npair;
            double *sr = SR + p * nsym, *si = SR + (np + p) * nsym;
            int64_t q = 0, s = 0;
            for (int64_t mu = 0; mu < nao; mu++) {
                const zc *xa = x + mu * n2, *xb = x + (nao + mu) * n2 + nao;
                for (int64_t nu = 0; nu < mu; nu++, q++, s++) {
                    const zc u = xa[nao + nu], w = x[nu * n2 + nao + mu];
                    aar[q] = creal(xa[nu]);
                    aai[q] = cimag(xa[nu]);
                    bbr[q] = creal(xb[nu]);
                    bbi[q] = cimag(xb[nu]);
                    abr[q] = 0.5 * creal(u - w);
                    abi[q] = 0.5 * cimag(u - w);
                    sr[s] = 0.5 * creal(u + w);
                    si[s] = 0.5 * cimag(u + w);
                }
                sr[s] = creal(xa[nao + mu]);
                si[s] = cimag(xa[nao + mu]);
                s++;
            }
        }
        free(x);
    }
}

/* The inverse layout for the results GA = AR W_same and GS = SR W_plus:
 * G[p] with the blocks aa and bb antisymmetric, ab = GS + GA_ab (symmetric
 * plus antisymmetric part) and ba = -ab^T; strides as in zc_ladder_pack. */
void zc_ladder_unpack(const double *GA, const double *GS, int64_t np, int64_t nao,
                      int64_t sp, int64_t sx, zc *G)
{
    const int64_t n2 = 2 * nao, npair = nao * (nao - 1) / 2, nsym = nao * (nao + 1) / 2;
    #pragma omp parallel
    {
        zc *g = malloc(sizeof(zc) * n2 * n2);
        #pragma omp for schedule(static)
        for (int64_t p = 0; p < np; p++) {
            const double *aar = GA + p * npair, *aai = GA + (3 * np + p) * npair;
            const double *bbr = GA + (np + p) * npair, *bbi = GA + (4 * np + p) * npair;
            const double *abr = GA + (2 * np + p) * npair, *abi = GA + (5 * np + p) * npair;
            const double *sr = GS + p * nsym, *si = GS + (np + p) * nsym;
            int64_t q = 0, s = 0;
            for (int64_t mu = 0; mu < nao; mu++) {
                for (int64_t nu = 0; nu < mu; nu++, q++, s++) {
                    const zc a = aar[q] + aai[q] * I, b = bbr[q] + bbi[q] * I;
                    const zc A = abr[q] + abi[q] * I, S = sr[s] + si[s] * I;
                    g[mu * n2 + nu] = a;
                    g[nu * n2 + mu] = -a;
                    g[(nao + mu) * n2 + nao + nu] = b;
                    g[(nao + nu) * n2 + nao + mu] = -b;
                    g[mu * n2 + nao + nu] = S + A;                  /* ab[mu, nu] */
                    g[nu * n2 + nao + mu] = S - A;                  /* ab[nu, mu] */
                    g[(nao + nu) * n2 + mu] = -(S + A);             /* ba = -ab^T */
                    g[(nao + mu) * n2 + nu] = -(S - A);
                }
                const zc S = sr[s] + si[s] * I;
                s++;
                g[mu * n2 + mu] = 0.0;
                g[(nao + mu) * n2 + nao + mu] = 0.0;
                g[mu * n2 + nao + mu] = S;
                g[(nao + mu) * n2 + mu] = -S;
            }
            for (int64_t r = 0; r < n2; r++)
                memcpy(G + p * sp + r * sx, g + r * n2, sizeof(zc) * n2);
        }
        free(g);
    }
}

/* ------------------------------------------------------------ Kramers pairs
 *
 * Spinors in time-reversal pairs, ordered [unbarred, barred] among the
 * occupied and among the virtual ones (zccsdt/zccsd/kramers.py): K maps the
 * index p of the first half to p + n/2 with the sign +1 and back with -1,
 * and every tensor obeys X[Kp, Kq, ...] = s_p s_q ... conj(X[p, q, ...]). */

/* The rows (i, a) with barred i of a matrix in the ring layout from those
 * with unbarred i, i0 <= i < i1: X has ld elements per row, its columns are
 * (jj, b) for 2 nj occupied jj - nj unbarred ones and then their partners -
 * and all virtual b:
 *   X[(i + no/2, Ka), (K jj, Kb)] = s_a s_jj s_b conj(X[(i, a), (jj, b)]). */
void zc_kfill_ring(zc *X, int64_t no, int64_t nv, int64_t i0, int64_t i1, int64_t nj,
                   int64_t ld)
{
    const int64_t ho = no / 2, hv = nv / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = i0; i < i1; i++)
        for (int64_t a = 0; a < nv; a++) {
            const zc *src = X + (i * nv + a) * ld;
            const int64_t ka = a < hv ? a + hv : a - hv;
            const double sa = a < hv ? 1.0 : -1.0;
            zc *dst = X + ((i + ho) * nv + ka) * ld;
            for (int64_t jj = 0; jj < 2 * nj; jj++) {
                const zc *s0 = src + jj * nv;
                zc *d0 = dst + (jj < nj ? jj + nj : jj - nj) * nv;
                const double sj = jj < nj ? sa : -sa;
                for (int64_t b = 0; b < hv; b++)
                    d0[b + hv] = sj * conj(s0[b]);
                for (int64_t b = hv; b < nv; b++)
                    d0[b - hv] = -sj * conj(s0[b]);
            }
        }
}

/* R[drow[p], q] += srow[p] sgv[q] conj(L[p, kpv[q]]) for np rows of npv
 * elements: the time-reversed image of rows of a packed array (kpv, sgv:
 * partner and sign of the virtual pairs); drow[p] < 0 is skipped. */
void zc_kfill_packed(const zc *L, int64_t np, int64_t npv, const int64_t *drow,
                     const double *srow, const int64_t *kpv, const double *sgv, zc *R)
{
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < np; p++) {
        if (drow[p] < 0)
            continue;
        zc *r = R + drow[p] * npv;
        const zc *l = L + p * npv;
        const double sp = srow[p];
        for (int64_t q = 0; q < npv; q++)
            r[q] += sp * sgv[q] * conj(l[kpv[q]]);
    }
}

/* X += its time-reversed image, in place, for a packed array of np rows of
 * npv elements:
 *   X[P, Q] += so[P] sv[Q] conj(X[kpo[P], kpv[Q]])    (the values before the call)
 * kpo, so and kpv, sv: partner and sign of the occupied and of the virtual
 * pairs (partners have the same sign).  For an array that holds the part of
 * a symmetric quantity that was summed over the unbarred half of an index. */
void zc_ksym_packed(zc *X, int64_t np, int64_t npv, const int64_t *kpo, const double *so,
                    const int64_t *kpv, const double *sv)
{
    #pragma omp parallel
    {
        zc *a = malloc(sizeof(zc) * 2 * npv), *b = a + npv;
        #pragma omp for schedule(dynamic, 8)
        for (int64_t p = 0; p < np; p++) {
            const int64_t q = kpo[p];
            if (q < p)
                continue;                               /* done with its partner */
            zc *xp = X + p * npv, *xq = X + q * npv;
            memcpy(a, xp, sizeof(zc) * npv);
            if (q != p)
                memcpy(b, xq, sizeof(zc) * npv);
            const zc *other = q != p ? b : a;
            for (int64_t c = 0; c < npv; c++)
                xp[c] = a[c] + so[p] * sv[c] * conj(other[kpv[c]]);
            if (q != p)
                for (int64_t c = 0; c < npv; c++)
                    xq[c] = b[c] + so[q] * sv[c] * conj(a[kpv[c]]);
        }
        free(a);
    }
}

/* R[rows[k], q] += signs[k] sgv[q] Z[zrow[k], kpv[q]] for n rows: zc_add_rows
 * for the time-reversed image of the rows of Z (not conjugated); rows < 0
 * are skipped. */
void zc_add_rows_k(zc *R, int64_t npv, const int64_t *rows, const double *signs,
                   const zc *Z, const int64_t *zrow, int64_t n, const int64_t *kpv,
                   const double *sgv)
{
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < n; k++) {
        if (rows[k] < 0)
            continue;
        zc *r = R + rows[k] * npv;
        const zc *z = Z + zrow[k] * npv;
        const double sk = signs[k];
        for (int64_t q = 0; q < npv; q++)
            r[q] += sk * sgv[q] * z[kpv[q]];
    }
}
