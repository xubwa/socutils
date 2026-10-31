/*
 * Memory-bound kernels of the route-1 THC exchange (socutils.pbc.thc, get_k_sparse), OpenMP.
 *
 *   SOCTHCpair_accum  H[i_p, :] += B[p, :] * E[j_p, :]  (and H[j_p] += B[p] * E[i_p] for i_p != j_p)
 *                     threads own column ranges of H, so there are no write conflicts
 *   SOCTHCunpack_sym  W[k] = 0;  W[k, i_p, j_p] = W[k, j_p, i_p] = V[k, p]      (packed pairs -> symmetric)
 *   SOCTHCembed       W[k, idx[a], idx[b]] = S[k, a, b]
 *   SOCTHCswap02      dst[i2, i1, i0, :] = src[i0, i1, i2, :]
 *   SOCTHCadd_cols    M[:, cols] += S
 *   SOCTHCadd_rows    M[rows] += S
 *   SOCTHCrowdot / SOCTHCscale_rows   row-wise dot products and row scaling (GAPW J grid contractions)
 *   SOCTHCft_post     weighted re/im layout of an F-order ft_ao block (streamed THC A)
 *   SOCTHCsquare      in-place square (Gram columns of the streamed THC build)
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void SOCTHCpair_accum(double *H, const double *B, const double *E, const int *pi, const int *pj, const int8_t *off,
                      int np, int nmu)
{
#pragma omp parallel
{
        int nt = 1, it = 0;
#ifdef _OPENMP
        nt = omp_get_num_threads();
        it = omp_get_thread_num();
#endif
        size_t m0 = (size_t)nmu * it / nt;
        size_t m1 = (size_t)nmu * (it + 1) / nt;
        size_t p, m;
        for (p = 0; p < (size_t)np; p++) {
                const double *b = B + p * nmu;
                double *h = H + (size_t)pi[p] * nmu;
                const double *e = E + (size_t)pj[p] * nmu;
                for (m = m0; m < m1; m++) {
                        h[m] += b[m] * e[m];
                }
                if (off[p]) {
                        h = H + (size_t)pj[p] * nmu;
                        e = E + (size_t)pi[p] * nmu;
                        for (m = m0; m < m1; m++) {
                                h[m] += b[m] * e[m];
                        }
                }
        }
}
}

void SOCTHCunpack_sym(double *W, const double *V, const int *ii, const int *jj, int nb, int npair, int n)
{
        size_t nn = (size_t)n * n;
        int k;
#pragma omp parallel for schedule(static)
        for (k = 0; k < nb; k++) {
                memset(W + k * nn, 0, sizeof(double) * nn);
        }
        size_t chunk = 4096;
        size_t nchunk = ((size_t)npair + chunk - 1) / chunk;
        size_t t;
#pragma omp parallel for schedule(static)
        for (t = 0; t < (size_t)nb * nchunk; t++) {
                size_t kk = t / nchunk;
                size_t p0 = (t % nchunk) * chunk;
                size_t p1 = p0 + chunk < (size_t)npair ? p0 + chunk : (size_t)npair;
                double *w = W + kk * nn;
                const double *v = V + kk * npair;
                size_t p;
                for (p = p0; p < p1; p++) {
                        w[(size_t)ii[p] * n + jj[p]] = v[p];
                        w[(size_t)jj[p] * n + ii[p]] = v[p];
                }
        }
}

void SOCTHCembed(double *W, const double *S, const int *idx, int nb, int ns, int n)
{
        size_t nn = (size_t)n * n, ss = (size_t)ns * ns;
        size_t t;
#pragma omp parallel for schedule(static)
        for (t = 0; t < (size_t)nb * ns; t++) {
                size_t k = t / ns, a = t % ns, b;
                double *w = W + k * nn + (size_t)idx[a] * n;
                const double *s = S + k * ss + a * ns;
                for (b = 0; b < (size_t)ns; b++) {
                        w[idx[b]] = s[b];
                }
        }
}

/* dst[i2, i1, i0, :] = src[i0, i1, i2, :]  (src (n0, n1, n2, L) C order; rows of length L copied in parallel) */
void SOCTHCswap02(double *dst, const double *src, int n0, int n1, int n2, int L)
{
        size_t t, nt = (size_t)n0 * n1 * n2;
#pragma omp parallel for schedule(static)
        for (t = 0; t < nt; t++) {
                size_t i0 = t / ((size_t)n1 * n2);
                size_t i1 = (t / n2) % n1;
                size_t i2 = t % n2;
                memcpy(dst + ((i2 * n1 + i1) * n0 + i0) * (size_t)L, src + t * (size_t)L, sizeof(double) * L);
        }
}

/* M[:, cols] += S  (M (nb, n), S (nb, ncol), C order; columns distinct within a call) */
void SOCTHCadd_cols(double *M, const double *S, const int *cols, int nb, int ncol, int n)
{
        size_t t, nt = (size_t)nb;
#pragma omp parallel for schedule(static)
        for (t = 0; t < nt; t++) {
                double *m = M + t * (size_t)n;
                const double *s = S + t * (size_t)ncol;
                int j;
                for (j = 0; j < ncol; j++) {
                        m[cols[j]] += s[j];
                }
        }
}

/* M[rows] += S  (M (n, L), S (nr, L), C order; rows distinct within a call) */
void SOCTHCadd_rows(double *M, const double *S, const int *rows, int nr, int L)
{
        size_t t;
#pragma omp parallel for schedule(static)
        for (t = 0; t < (size_t)nr; t++) {
                double *m = M + (size_t)rows[t] * L;
                const double *s = S + t * (size_t)L;
                int j;
                for (j = 0; j < L; j++) {
                        m[j] += s[j];
                }
        }
}

/* out[r] = sum_j A[r, j] B[r, j]  (A, B (n, m) C order) */
void SOCTHCrowdot(double *out, const double *A, const double *B, int n, int m)
{
        size_t r;
#pragma omp parallel for schedule(static)
        for (r = 0; r < (size_t)n; r++) {
                const double *a = A + r * m, *b = B + r * m;
                double s = 0;
                int j;
                for (j = 0; j < m; j++) {
                        s += a[j] * b[j];
                }
                out[r] = s;
        }
}

/* out = w[:, None] * A  (A, out (n, m) C order) */
void SOCTHCscale_rows(double *out, const double *A, const double *w, int n, int m)
{
        size_t r;
#pragma omp parallel for schedule(static)
        for (r = 0; r < (size_t)n; r++) {
                const double *a = A + r * m;
                double *o = out + r * m, c = w[r];
                int j;
                for (j = 0; j < m; j++) {
                        o[j] = c * a[j];
                }
        }
}

/* out[2i + r, j] = w[i] * (Re, Im)[r] P[i, j]  for P (g, n) complex in Fortran order (pyscf ft_ao), out (2g, n)
 * C order: the weighted real/imaginary layout of the streamed THC A GEMM, one tiled pass */
void SOCTHCft_post(double *out, const double *P, const double *w, int g, int n)
{
        const int TB = 64;
        int nti = (g + TB - 1) / TB, ntj = (n + TB - 1) / TB;
        int t;
#pragma omp parallel for schedule(dynamic)
        for (t = 0; t < nti * ntj; t++) {
                int i0 = (t / ntj) * TB, j0 = (t % ntj) * TB;
                int i1 = i0 + TB < g ? i0 + TB : g, j1 = j0 + TB < n ? j0 + TB : n;
                int i, j;
                for (j = j0; j < j1; j++) {
                        const double *p = P + 2 * ((size_t)j * g);
                        for (i = i0; i < i1; i++) {
                                out[(size_t)(2 * i) * n + j] = w[i] * p[2 * i];
                                out[(size_t)(2 * i + 1) * n + j] = w[i] * p[2 * i + 1];
                        }
                }
        }
}

/* a[k] = a[k]^2 in place (n doubles) */
void SOCTHCsquare(double *a, size_t n)
{
        size_t k;
#pragma omp parallel for schedule(static)
        for (k = 0; k < n; k++) {
                a[k] *= a[k];
        }
}
