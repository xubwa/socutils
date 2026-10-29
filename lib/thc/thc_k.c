/*
 * Memory-bound kernels of the route-1 THC exchange (socutils.pbc.thc, get_k_sparse), OpenMP.
 *
 *   SOCTHCpair_accum  H[i_p, :] += B[p, :] * E[j_p, :]  (and H[j_p] += B[p] * E[i_p] for i_p != j_p)
 *                     threads own column ranges of H, so there are no write conflicts
 *   SOCTHCunpack_sym  W[k] = 0;  W[k, i_p, j_p] = W[k, j_p, i_p] = V[k, p]      (packed pairs -> symmetric)
 *   SOCTHCembed       W[k, idx[a], idx[b]] = S[k, a, b]
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
