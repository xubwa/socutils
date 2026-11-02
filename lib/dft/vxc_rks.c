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

#define MASK_BLK 56                     /* pyscf's BLKSIZE: rows of the non0tab mask */

/* as select_gather, the candidates restricted to the shells non0tab flags on
 * the 56-point rows of the sub-block (gs a multiple of 56, g0 too) */
static int select_gather_mask(double *a, int *ix, const double *ao, long cs, int ngrid, int nao, int ncomp,
                              int g0, int gs, double cutoff, const unsigned char *mask, const int *ao_loc,
                              int nbas)
{
        const int r0 = g0 / MASK_BLK, r1 = (g0 + gs + MASK_BLK - 1) / MASK_BLK;
        int sh, mu, k, g, r, nl = 0;
        for (sh = 0; sh < nbas; sh++) {
                int on = 0;
                for (r = r0; r < r1 && !on; r++) on = mask[(size_t)r * nbas + sh] != 0;
                if (!on) continue;
                for (mu = ao_loc[sh]; mu < ao_loc[sh + 1]; mu++) {
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
        }
        for (k = 0; k < ncomp; k++)
                for (mu = 0; mu < nl; mu++)
                        memcpy(a + ((size_t)k * nl + mu) * gs, ao + k * cs + (size_t)ix[mu] * ngrid + g0,
                               sizeof(double) * gs);
        (void)nao;
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

static void rho_impl(double *rho, int *idx, int *nl_out, double *ao, long cs, double *dm, int nao,
                     int ngrid, int gs, double cutoff, int xctype, dgemm_t dgemm_,
                     mkl_local_t mkl_set_num_threads_local, const unsigned char *mask, const int *ao_loc,
                     int nbas);

/* rho (nvar, ngrid): LDA 1 row, GGA 4 (rho, nabla rho) */
void SOCDFTvxc_rho(double *rho, int *idx, int *nl_out, double *ao, long cs, double *dm, int nao,
                   int ngrid, int gs, double cutoff, int xctype, dgemm_t dgemm_,
                   mkl_local_t mkl_set_num_threads_local)
{
        rho_impl(rho, idx, nl_out, ao, cs, dm, nao, ngrid, gs, cutoff, xctype, dgemm_,
                 mkl_set_num_threads_local, NULL, NULL, 0);
}

/* the same with the AO candidates of every sub-block from the block's non0tab
 * mask (rows of 56 points, (ngrid / 56, nbas)) and ao_loc; gs a multiple of 56 */
void SOCDFTvxc_rho_mask(double *rho, int *idx, int *nl_out, double *ao, long cs, double *dm, int nao,
                        int ngrid, int gs, double cutoff, int xctype, dgemm_t dgemm_,
                        mkl_local_t mkl_set_num_threads_local, unsigned char *mask, int *ao_loc, int nbas)
{
        rho_impl(rho, idx, nl_out, ao, cs, dm, nao, ngrid, gs, cutoff, xctype, dgemm_,
                 mkl_set_num_threads_local, mask, ao_loc, nbas);
}

static void rho_impl(double *rho, int *idx, int *nl_out, double *ao, long cs, double *dm, int nao,
                     int ngrid, int gs, double cutoff, int xctype, dgemm_t dgemm_,
                     mkl_local_t mkl_set_num_threads_local, const unsigned char *mask, const int *ao_loc,
                     int nbas)
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
                const int nl = mask ? select_gather_mask(a, ix, ao, cs, ngrid, nao, ncomp, g0, n, cutoff,
                                                         mask, ao_loc, nbas)
                                    : select_gather(a, ix, ao, cs, ngrid, nao, ncomp, g0, n, cutoff);
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

/*
 * One pass over the grid (SOCDFTvxc_fused): per sub-block of gs points (a
 * multiple of 56) one thread evaluates the AOs (pyscf's GTOval_sph /
 * GTOval_sph_deriv1 with the grid's non0tab rows), keeps those of the flagged
 * shells with max_g |phi_k| > cutoff, forms rho (and nabla rho), evaluates the
 * functional on the sub-block (pyscf's LIBXC_eval_xc; its OpenMP region runs
 * serially inside ours), forms wv = w (v_rho, 2 v_sigma nabla rho) and adds
 * phi_0 aow^T into its V; no AO array is stored and the AOs are evaluated
 * once.  vbuf (nthreads, nao, nao) (GGA: the caller adds V + V^T), ex / ne
 * (nthreads) the per-thread E_xc and electron counts.  coords (3, ngrid),
 * non0tab (ngrid / 56, nbas).  xctype 0 LDA, 1 GGA.
 */
#include <stdint.h>
typedef void (*gtoval_t)(int ngrids, int *shls_slice, int *ao_loc, double *ao, double *coord,
                         uint8_t *non0table, int *atm, int natm, int *bas, int nbas, double *env);
typedef void (*libxc_t)(int nfn, void *fn_obj, const double *fac, int spin, int deriv, int nvar, int np,
                        int outlen, double *rho, double *out);

void SOCDFTvxc_fused(double *vbuf, double *ex, double *ne, double *dm, double *coords, double *weights,
                     uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                     int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                     libxc_t xcf, int nfn, void *xc_arr, double *facs,
                     dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        const int nao = ao_loc[nbas], ncomp = xctype == 0 ? 1 : 4;
        const int nsub = (ngrid + gs - 1) / gs, outlen = xctype == 0 ? 2 : 3;
        const double one = 1, zero = 0;
        const size_t nn = (size_t)nao * nao;
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *V = vbuf + tid * nn;
        double *full = malloc(sizeof(double) * ncomp * nao * gs);
        double *a = malloc(sizeof(double) * ncomp * nao * gs);
        double *c = malloc(sizeof(double) * nao * gs);
        double *d = malloc(sizeof(double) * nn);
        double *xyz = malloc(sizeof(double) * 3 * gs);
        double *rho = malloc(sizeof(double) * ncomp * gs);
        double *out = malloc(sizeof(double) * outlen * gs);
        double *wv = malloc(sizeof(double) * ncomp * gs);
        int *ix = malloc(sizeof(int) * nao);
        int shls[2] = {0, nbas};
        double exs = 0, nes = 0;
        int s;
#pragma omp for schedule(dynamic, 4)
        for (s = 0; s < nsub; s++) {
                const int g0 = s * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                const uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int i, j, g, k;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                eval(n, shls, ao_loc, full, xyz, (uint8_t *)msk, atm, natm, bas, nbas, env);
                const int nl = select_gather_mask(a, ix, full, (long)nao * n, n, nao, ncomp, 0, n, cutoff,
                                                  msk, ao_loc, nbas);
                if (nl == 0) continue;
                for (i = 0; i < nl; i++)
                        for (j = 0; j < nl; j++) d[i * nl + j] = dm[(size_t)ix[i] * nao + ix[j]];
                /* c (nl, n) = D_sub phi_0; rho, nabla rho */
                dgemm_("N", "N", &n, &nl, &nl, &one, a, &n, d, &nl, &zero, c, &n);
                memset(rho, 0, sizeof(double) * ncomp * n);
                for (i = 0; i < nl; i++) {
                        const double *p = a + (size_t)i * n, *q = c + (size_t)i * n;
                        for (g = 0; g < n; g++) rho[g] += p[g] * q[g];
                }
                for (k = 1; k < ncomp; k++) {
                        double *rk = rho + (size_t)k * n;
                        for (i = 0; i < nl; i++) {
                                const double *p = a + ((size_t)k * nl + i) * n, *q = c + (size_t)i * n;
                                for (g = 0; g < n; g++) rk[g] += 2 * p[g] * q[g];
                        }
                }
                /* the functional: out (outlen, n) = exc, v_rho (, v_sigma); LIBXC_eval_xc
                 * accumulates the functionals of the mixture into out */
                memset(out, 0, sizeof(double) * outlen * n);
                xcf(nfn, xc_arr, facs, 0, 1, ncomp, n, outlen, rho, out);
                const double *w = weights + g0;
                for (g = 0; g < n; g++) {
                        const double den = w[g] * rho[g];
                        nes += den;
                        exs += den * out[g];
                        wv[g] = w[g] * out[n + g];
                }
                if (ncomp == 4) {
                        for (g = 0; g < n; g++) wv[g] *= .5;            /* V + V^T by the caller */
                        for (k = 1; k < 4; k++)
                                for (g = 0; g < n; g++)
                                        wv[k * n + g] = 2 * w[g] * out[2 * n + g] * rho[k * n + g];
                }
                /* aow (nl, n) = sum_k wv_k d_k phi into c; V_sub += phi_0 aow^T */
                for (i = 0; i < nl; i++) {
                        double *o = c + (size_t)i * n;
                        const double *p = a + (size_t)i * n;
                        for (g = 0; g < n; g++) o[g] = wv[g] * p[g];
                        for (k = 1; k < ncomp; k++) {
                                const double *pk = a + ((size_t)k * nl + i) * n, *wk = wv + (size_t)k * n;
                                for (g = 0; g < n; g++) o[g] += wk[g] * pk[g];
                        }
                }
                dgemm_("T", "N", &nl, &nl, &n, &one, a, &n, c, &n, &zero, d, &nl);
                for (i = 0; i < nl; i++) {
                        double *row = V + (size_t)ix[i] * nao;
                        for (j = 0; j < nl; j++) row[ix[j]] += d[i * nl + j];
                }
        }
        ex[tid] = exs;
        ne[tid] = nes;
        free(full); free(a); free(c); free(d); free(xyz); free(rho); free(out); free(wv); free(ix);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}

/*
 * XC part of the RKS gradient in one pass over the grid (SOCDFTxc_grad_fused),
 * as SOCDFTrks_xc_grad (lib/dft/xc_hess.c) per sub-block in cache: the AOs to
 * the second derivatives (GGA; first, LDA) by pyscf's GTOval with the grid's
 * non0tab rows, rho and the functional on the sub-block (LIBXC_eval_xc), then
 *
 *     out[x][mu] += sum_g d_x phi_mu U_mu + (w0 d_x phi_mu + sum_k w_k d_x d_k phi_mu) V_mu
 *
 * with V = D_sub phi_0, U = D_sub aow, aow = sum_k w_k d_k phi (GGA w0 halved);
 * out (nthreads, 3, nao), the gradient -2 sum_{mu in A} out[x][mu].
 */
static const int D2C[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};   /* pyscf's d_x d_k component */

void SOCDFTxc_grad_fused(double *outb, double *dm, double *coords, double *weights,
                         uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                         int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                         libxc_t xcf, int nfn, void *xc_arr, double *facs,
                         dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        const int nao = ao_loc[nbas];
        const int nvar = xctype == 0 ? 1 : 4;           /* rho (, nabla rho) */
        const int ncomp = xctype == 0 ? 4 : 10;         /* AO components evaluated */
        const int nsub = (ngrid + gs - 1) / gs, outlen = xctype == 0 ? 2 : 3;
        const double one = 1, zero = 0;
        const size_t nn = (size_t)nao * nao;
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *out = outb + (size_t)tid * 3 * nao;
        double *full = malloc(sizeof(double) * ncomp * nao * gs);
        double *a = malloc(sizeof(double) * ncomp * nao * gs);
        double *c = malloc(sizeof(double) * nao * gs);          /* V = D phi_0 */
        double *aw = malloc(sizeof(double) * nao * gs);
        double *u = malloc(sizeof(double) * nao * gs);          /* U = D aow */
        double *d = malloc(sizeof(double) * nn);
        double *xyz = malloc(sizeof(double) * 3 * gs);
        double *rho = malloc(sizeof(double) * nvar * gs);
        double *xo = malloc(sizeof(double) * outlen * gs);
        double *wv = malloc(sizeof(double) * nvar * gs);
        int *ix = malloc(sizeof(int) * nao);
        int shls[2] = {0, nbas};
        int s;
#pragma omp for schedule(dynamic, 4)
        for (s = 0; s < nsub; s++) {
                const int g0 = s * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                const uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int i, j, g, k, x;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                eval(n, shls, ao_loc, full, xyz, (uint8_t *)msk, atm, natm, bas, nbas, env);
                const int nl = select_gather_mask(a, ix, full, (long)nao * n, n, nao, ncomp, 0, n, cutoff,
                                                  msk, ao_loc, nbas);
                if (nl == 0) continue;
                for (i = 0; i < nl; i++)
                        for (j = 0; j < nl; j++) d[i * nl + j] = dm[(size_t)ix[i] * nao + ix[j]];
                dgemm_("N", "N", &n, &nl, &nl, &one, a, &n, d, &nl, &zero, c, &n);
                memset(rho, 0, sizeof(double) * nvar * n);
                for (i = 0; i < nl; i++) {
                        const double *p = a + (size_t)i * n, *q = c + (size_t)i * n;
                        for (g = 0; g < n; g++) rho[g] += p[g] * q[g];
                }
                for (k = 1; k < nvar; k++) {
                        double *rk = rho + (size_t)k * n;
                        for (i = 0; i < nl; i++) {
                                const double *p = a + ((size_t)k * nl + i) * n, *q = c + (size_t)i * n;
                                for (g = 0; g < n; g++) rk[g] += 2 * p[g] * q[g];
                        }
                }
                memset(xo, 0, sizeof(double) * outlen * n);
                xcf(nfn, xc_arr, facs, 0, 1, nvar, n, outlen, rho, xo);
                const double *w = weights + g0;
                for (g = 0; g < n; g++) wv[g] = w[g] * xo[n + g];
                if (nvar == 4) {
                        for (g = 0; g < n; g++) wv[g] *= .5;
                        for (k = 1; k < 4; k++)
                                for (g = 0; g < n; g++) wv[k * n + g] = 2 * w[g] * xo[2 * n + g] * rho[k * n + g];
                }
                /* aow = sum_k wv_k d_k phi; U = D_sub aow */
                for (i = 0; i < nl; i++) {
                        double *o = aw + (size_t)i * n;
                        const double *p = a + (size_t)i * n;
                        for (g = 0; g < n; g++) o[g] = wv[g] * p[g];
                        for (k = 1; k < nvar; k++) {
                                const double *pk = a + ((size_t)k * nl + i) * n, *wk = wv + (size_t)k * n;
                                for (g = 0; g < n; g++) o[g] += wk[g] * pk[g];
                        }
                }
                dgemm_("N", "N", &n, &nl, &nl, &one, aw, &n, d, &nl, &zero, u, &n);
                for (i = 0; i < nl; i++) {
                        const double *ui = u + (size_t)i * n, *vi = c + (size_t)i * n;
                        for (x = 0; x < 3; x++) {
                                const double *ax = a + ((size_t)(1 + x) * nl + i) * n;
                                double sm = 0;
                                if (nvar == 1) {
                                        for (g = 0; g < n; g++) sm += ax[g] * ui[g];
                                } else {
                                        const double *a0 = a + ((size_t)D2C[x][0] * nl + i) * n;
                                        const double *a1 = a + ((size_t)D2C[x][1] * nl + i) * n;
                                        const double *a2 = a + ((size_t)D2C[x][2] * nl + i) * n;
                                        const double *w1 = wv + n, *w2 = wv + 2 * n, *w3 = wv + 3 * n;
                                        for (g = 0; g < n; g++)
                                                sm += ax[g] * (ui[g] + wv[g] * vi[g])
                                                    + (a0[g] * w1[g] + a1[g] * w2[g] + a2[g] * w3[g]) * vi[g];
                                }
                                out[(size_t)x * nao + ix[i]] += sm;
                        }
                }
        }
        free(full); free(a); free(c); free(aw); free(u); free(d); free(xyz); free(rho); free(xo); free(wv);
        free(ix);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}
