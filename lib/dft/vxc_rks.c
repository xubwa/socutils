/*
 * The SCF XC potential of one symmetric density (LDA / GGA) on a pyscf AO
 * block, cache-blocked (socutils.dft.vxc_fast): the block is cut into
 * sub-blocks of gs grid points, each handled by one thread on the AOs
 * significant on it, max_g |phi_k| > cutoff (value and, for GGA, gradient),
 * so that one read of the block's AO values feeds all the work of a pass and
 * the products run over those AOs only:
 *
 *   pass 1 (SOCDFTvxc_rho):  c = D_sub phi_0,  rho = sum_mu phi_0 c,
 *                            d_k rho = 2 sum_mu d_k phi c;
 *                            the significant AOs of every sub-block are kept
 *   pass 2 (SOCDFTvxc_mat):  aow = sum_k w_k d_k phi (k = 0 the value; GGA w_0
 *                            halved by the caller),  V_sub += phi_0 aow^T
 *                            into per-thread V (nthreads, nao, nao)
 *
 * (GGA: the caller adds V + V^T.)  ao is the AO stack in memory order (ncomp,
 * nao, ngrid), cs doubles between components; idx (nsub, nao) / nl (nsub) the
 * AOs of every sub-block from pass 1.  xctype 0 LDA, 1 GGA.
 *
 * BLAS: none is linked; the caller passes dgemm_ of the BLAS in the process
 * (pyscf's) and, for MKL, MKL_Set_Num_Threads_Local (by value; NULL for other
 * BLAS, which must then be thread-safe and sequential here), with which the
 * workers run MKL on one thread for the duration.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef void (*dgemm_t)(const char *ta, const char *tb, const int *m, const int *n, const int *k,
                        const double *alpha, const double *a, const int *lda, const double *b,
                        const int *ldb, const double *beta, double *c, const int *ldc);
typedef int (*mkl_local_t)(int nt);

/* the significant AOs of sub-block [g0, g0 + gs): ix, and the gathered values
 * a[k][i][g] of all components */
static int select_gather(double *a, int *ix, const double *ao, long cs, int ngrid, int nao, int ncomp,
                         int g0, int gs, double cutoff)
{
        int mu, k, g, nl = 0;
        for (mu = 0; mu < nao; mu++) {
                double m = 0;
                for (k = 0; k < ncomp; k++) {
                        const double *p = ao + k * cs + (size_t)mu * ngrid + g0;
                        for (g = 0; g < gs; g++) {
                                const double v = fabs(p[g]);
                                if (v > m) m = v;
                        }
                }
                if (m > cutoff) ix[nl++] = mu;
        }
        for (k = 0; k < ncomp; k++)
                for (mu = 0; mu < nl; mu++)
                        memcpy(a + ((size_t)k * nl + mu) * gs, ao + k * cs + (size_t)ix[mu] * ngrid + g0,
                               sizeof(double) * gs);
        return nl;
}

static void gather(double *a, const int *ix, int nl, const double *ao, long cs, int ngrid, int ncomp,
                   int g0, int gs)
{
        int mu, k;
        for (k = 0; k < ncomp; k++)
                for (mu = 0; mu < nl; mu++)
                        memcpy(a + ((size_t)k * nl + mu) * gs, ao + k * cs + (size_t)ix[mu] * ngrid + g0,
                               sizeof(double) * gs);
}

/* rho (nvar, ngrid): LDA 1 row, GGA 4 (rho, nabla rho) */
void SOCDFTvxc_rho(double *rho, int *idx, int *nl_out, double *ao, long cs, double *dm, int nao,
                   int ngrid, int gs, double cutoff, int xctype, dgemm_t dgemm_,
                   mkl_local_t mkl_set_num_threads_local)
{
        const int ncomp = xctype == 0 ? 1 : 4;
        const int nsub = (ngrid + gs - 1) / gs;
        const double one = 1, zero = 0;
#pragma omp parallel
{
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *a = malloc(sizeof(double) * ncomp * nao * gs);
        double *c = malloc(sizeof(double) * nao * gs);
        double *d = malloc(sizeof(double) * nao * nao);
        int s;
#pragma omp for schedule(dynamic, 4)
        for (s = 0; s < nsub; s++) {
                const int g0 = s * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                int *ix = idx + (size_t)s * nao;
                int i, j, g, k;
                const int nl = select_gather(a, ix, ao, cs, ngrid, nao, ncomp, g0, n, cutoff);
                nl_out[s] = nl;
                for (k = 0; k < ncomp; k++) memset(rho + (size_t)k * ngrid + g0, 0, sizeof(double) * n);
                if (nl == 0) continue;
                for (i = 0; i < nl; i++)
                        for (j = 0; j < nl; j++) d[i * nl + j] = dm[(size_t)ix[i] * nao + ix[j]];
                /* c (nl, n) = D_sub phi_0 (column-major view: phi_0^T D_sub) */
                dgemm_("N", "N", &n, &nl, &nl, &one, a, &n, d, &nl, &zero, c, &n);
                double *r0 = rho + g0;
                for (i = 0; i < nl; i++) {
                        const double *p = a + (size_t)i * n, *q = c + (size_t)i * n;
                        for (g = 0; g < n; g++) r0[g] += p[g] * q[g];
                }
                for (k = 1; k < ncomp; k++) {
                        double *rk = rho + (size_t)k * ngrid + g0;
                        for (i = 0; i < nl; i++) {
                                const double *p = a + ((size_t)k * nl + i) * n, *q = c + (size_t)i * n;
                                for (g = 0; g < n; g++) rk[g] += 2 * p[g] * q[g];
                        }
                }
        }
        free(a);
        free(c);
        free(d);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}

/* vbuf (nthreads, nao, nao) += phi_0 aow^T of every sub-block */
void SOCDFTvxc_mat(double *vbuf, double *ao, long cs, double *wv, int *idx, int *nl_in, int nao,
                   int ngrid, int gs, int xctype, dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        const int ncomp = xctype == 0 ? 1 : 4;
        const int nsub = (ngrid + gs - 1) / gs;
        const double one = 1, zero = 0;
        const size_t nn = (size_t)nao * nao;
#pragma omp parallel
{
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *V = vbuf + omp_get_thread_num() * nn;
        double *a = malloc(sizeof(double) * ncomp * nao * gs);
        double *w = malloc(sizeof(double) * nao * gs);
        double *v = malloc(sizeof(double) * nao * nao);
        int s;
#pragma omp for schedule(dynamic, 4)
        for (s = 0; s < nsub; s++) {
                const int g0 = s * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                const int nl = nl_in[s];
                const int *ix = idx + (size_t)s * nao;
                int i, j, g, k;
                if (nl == 0) continue;
                gather(a, ix, nl, ao, cs, ngrid, ncomp, g0, n);
                const double *w0 = wv + g0;
                for (i = 0; i < nl; i++) {
                        double *o = w + (size_t)i * n;
                        const double *p = a + (size_t)i * n;
                        for (g = 0; g < n; g++) o[g] = w0[g] * p[g];
                        for (k = 1; k < ncomp; k++) {
                                const double *pk = a + ((size_t)k * nl + i) * n, *wk = wv + (size_t)k * ngrid + g0;
                                for (g = 0; g < n; g++) o[g] += wk[g] * pk[g];
                        }
                }
                /* v (nl, nl) = phi_0 aow^T, v[i][j] = sum_g aow_i phi_j */
                dgemm_("T", "N", &nl, &nl, &n, &one, a, &n, w, &n, &zero, v, &nl);
                for (i = 0; i < nl; i++) {
                        double *row = V + (size_t)ix[i] * nao;
                        for (j = 0; j < nl; j++) row[ix[j]] += v[i * nl + j];
                }
        }
        free(a);
        free(w);
        free(v);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}
