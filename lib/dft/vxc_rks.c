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
 * non0tab (ngrid / 56, nbas).  xctype 0 LDA, 1 GGA, 2 meta-GGA (SOCDFTvxc_fused_mgga).
 */
#include <stdint.h>
typedef void (*gtoval_t)(int ngrids, int *shls_slice, int *ao_loc, double *ao, double *coord,
                         uint8_t *non0table, int *atm, int natm, int *bas, int nbas, double *env);
typedef void (*libxc_t)(int nfn, void *fn_obj, const double *fac, int spin, int deriv, int nvar, int np,
                        int outlen, double *rho, double *out);

/*
 * The AOs of a sub-block of n points (n <= gs) for the shells non0tab flags on
 * its 56-point rows only: eval per run of consecutive flagged shells (pyscf's
 * GTOval reads non0tab by the absolute shell index) into full, run after run
 * (layout (ncomp, nao_run, n) each), so that neither the evaluation nor the
 * memory traffic touches the other shells; then those with max_g |phi| >
 * cutoff into a (ncomp, nl, n), their indices into ix.  src: nao size_t.
 */
void SOCDFTgtoval_sph(int ngrids, int *shls_slice, int *ao_loc, double *ao, double *coord, uint8_t *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env);
void SOCDFTgtoval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc, double *ao, double *coord,
                             uint8_t *non0table, int *atm, int natm, int *bas, int nbas, double *env);
void SOCDFTgtoval_sph_deriv2(int ngrids, int *shls_slice, int *ao_loc, double *ao, double *coord,
                             uint8_t *non0table, int *atm, int natm, int *bas, int nbas, double *env);
void SOCDFTgtoval_ld(int deriv, int ngrids, int sh0, int sh1, int *ao_loc, double *ao, size_t ld, size_t row0,
                     double *coord, uint8_t *non0table, int *atm, int *bas, int nbas, double *env, double *rowmax);

/* eval_select with gtoval.c's evaluator: the flagged runs straight into a in the layout
 * (ncomp, ntot, n), the AOs below the cutoff squeezed out in place (no copy when all stay) */
static int eval_select_own(double *a, int *ix, int *runs, int n, double *xyz, uint8_t *msk, int ncomp,
                           double cutoff, int *ao_loc, int *atm, int *bas, int nbas, double *env)
{
        const int nrow = (n + MASK_BLK - 1) / MASK_BLK;
        int sh = 0, nrun = 0, r, k, g, i;
        size_t ntot = 0;
        while (sh < nbas) {
                int on = 0;
                for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                if (!on) { sh++; continue; }
                const int s0 = sh;
                for (sh++; sh < nbas; sh++) {
                        on = 0;
                        for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                        if (!on) break;
                }
                runs[2 * nrun] = s0;
                runs[2 * nrun + 1] = sh;
                nrun++;
                ntot += ao_loc[sh] - ao_loc[s0];
        }
        if (ntot == 0) return 0;
        size_t row = 0;
        /* the row maxima over the components from the evaluator (masked blocks: zero) */
        double *rmx = malloc(sizeof(double) * ntot);
        memset(rmx, 0, sizeof(double) * ntot);
        for (i = 0; i < nrun; i++) {
                SOCDFTgtoval_ld(ncomp == 10 ? 2 : (ncomp > 1), n, runs[2 * i], runs[2 * i + 1], ao_loc, a, ntot, row, xyz, msk, atm, bas,
                                nbas, env, rmx);
                row += ao_loc[runs[2 * i + 1]] - ao_loc[runs[2 * i]];
        }
        /* the kept AOs (max_g over the components above the cutoff), squeezed in place */
        int nl = 0;
        row = 0;
        for (i = 0; i < nrun; i++) {
                int mu;
                for (mu = ao_loc[runs[2 * i]]; mu < ao_loc[runs[2 * i + 1]]; mu++, row++) {
                        if (rmx[row] > cutoff) {
                                ix[nl] = mu;
                                runs[2 * nbas + nl] = (int)row;       /* the source row */
                                nl++;
                        }
                }
        }
        free(rmx);
        if ((size_t)nl < ntot) {
                const int *srow = runs + 2 * nbas;
                for (k = 0; k < ncomp; k++)
                        for (i = 0; i < nl; i++) {
                                double *dst = a + ((size_t)k * nl + i) * n;
                                const double *src_ = a + ((size_t)k * ntot + srow[i]) * n;
                                if (dst != src_) memmove(dst, src_, sizeof(double) * n);
                        }
        }
        return nl;
}

static int eval_select(double *a, int *ix, size_t *src, double *full, int n, double *xyz, uint8_t *msk,
                       int ncomp, double cutoff, gtoval_t eval, int *ao_loc, int *atm, int natm, int *bas,
                       int nbas, double *env)
{
        if (eval == SOCDFTgtoval_sph || eval == SOCDFTgtoval_sph_deriv1 || eval == SOCDFTgtoval_sph_deriv2)
                /* src (2 nao size_t) as the runs (2 nbas) + the source rows (nao) */
                return eval_select_own(a, ix, (int *)src, n, xyz, msk, ncomp, cutoff, ao_loc, atm, bas, nbas, env);
        const int nrow = (n + MASK_BLK - 1) / MASK_BLK;
        int sh = 0, nl = 0, r, k, g;
        size_t off = 0;
        int *str = (int *)(src + ao_loc[nbas]);         /* the comp stride of each kept AO, after src */
        while (sh < nbas) {
                int on = 0;
                for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                if (!on) { sh++; continue; }
                const int s0 = sh;
                for (sh++; sh < nbas; sh++) {
                        on = 0;
                        for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                        if (!on) break;
                }
                int sl[2] = {s0, sh};
                const int nr = ao_loc[sh] - ao_loc[s0];
                double *f = full + off;
                eval(n, sl, ao_loc, f, xyz, msk, atm, natm, bas, nbas, env);
                int mu;
                for (mu = 0; mu < nr; mu++) {
                        double m = 0;
                        for (k = 0; k < ncomp; k++) {
                                const double *p = f + ((size_t)k * nr + mu) * n;
                                for (g = 0; g < n; g++) {
                                        const double v = fabs(p[g]);
                                        if (v > m) m = v;
                                }
                        }
                        if (m > cutoff) {
                                ix[nl] = ao_loc[s0] + mu;
                                src[nl] = off + (size_t)mu * n;
                                str[nl] = nr * n;
                                nl++;
                        }
                }
                off += (size_t)ncomp * nr * n;
        }
        int i;
        for (k = 0; k < ncomp; k++)
                for (i = 0; i < nl; i++)
                        memcpy(a + ((size_t)k * nl + i) * n, full + src[i] + (size_t)k * str[i], sizeof(double) * n);
        return nl;
}

typedef void (*dsyrk_t)(const char *uplo, const char *trans, const int *n, const int *k, const double *alpha,
                        const double *a, const int *lda, const double *beta, double *c, const int *ldc);

/*
 * rho (nvar, n) of a sub-block from its AOs a (ncomp, nl, n) (ncomp 1 LDA, 4 GGA / MGGA): rho, nabla rho
 * (GGA, MGGA) and tau = 1/2 sum_k nabla_k phi D nabla_k phi (MGGA).  With the occupied orbitals (orb
 * (nao, nocc) = C sqrt(occ), row-major) and nocc < nl (MGGA), by c_k = C_sub^T d_k phi (nocc rows
 * instead of nl): rho = sum_i c_0 c_0, d_k rho = 2 sum_i c_0 c_k, tau = 1/2 sum_k sum_i c_k c_k;
 * else by c_k = D_sub d_k phi (c (ncomp, nl, n) the first row c_0 = D_sub phi_0 kept, it is the
 * gradient's V).  d (nl, nl) D_sub or (orbital route) C_sub, c scratch of ncomp nao n.
 */
static void sub_rho(double *rho, double *c, double *d, const double *a, const int *ix, int nl, int n, int nvar,
                    int ncomp, const double *dm, int nao, const double *orb, int nocc, dgemm_t dgemm_)
{
        const double one = 1, zero = 0;
        int i, j, g, k;
        if (orb && nvar == 5 && nocc < nl) {
                for (i = 0; i < nl; i++)
                        memcpy(d + (size_t)i * nocc, orb + (size_t)ix[i] * nocc, sizeof(double) * nocc);
                /* c_k (n, nocc) column-major = a_k (n, nl) C_sub (nl, nocc) */
                for (k = 0; k < 4; k++)
                        dgemm_("N", "T", &n, &nocc, &nl, &one, a + (size_t)k * nl * n, &n, d, &nocc, &zero,
                               c + (size_t)k * nocc * n, &n);
                memset(rho, 0, sizeof(double) * 5 * n);
                double *r4 = rho + 4 * n;
                for (i = 0; i < nocc; i++) {
                        const double *c0 = c + (size_t)i * n;
                        const double *c1 = c + ((size_t)nocc + i) * n;
                        const double *c2 = c + ((size_t)2 * nocc + i) * n;
                        const double *c3 = c + ((size_t)3 * nocc + i) * n;
                        for (g = 0; g < n; g++) {
                                rho[g] += c0[g] * c0[g];
                                rho[n + g] += 2 * c0[g] * c1[g];
                                rho[2 * n + g] += 2 * c0[g] * c2[g];
                                rho[3 * n + g] += 2 * c0[g] * c3[g];
                                r4[g] += .5 * (c1[g] * c1[g] + c2[g] * c2[g] + c3[g] * c3[g]);
                        }
                }
                return;
        }
        for (i = 0; i < nl; i++)
                for (j = 0; j < nl; j++) d[i * nl + j] = dm[(size_t)ix[i] * nao + ix[j]];
        /* c (nl, n) = D_sub phi_0; rho, nabla rho */
        dgemm_("N", "N", &n, &nl, &nl, &one, a, &n, d, &nl, &zero, c, &n);
        memset(rho, 0, sizeof(double) * nvar * n);
        for (i = 0; i < nl; i++) {
                const double *p = a + (size_t)i * n, *q = c + (size_t)i * n;
                for (g = 0; g < n; g++) rho[g] += p[g] * q[g];
        }
        for (k = 1; k < (nvar < 4 ? nvar : 4); k++) {
                double *rk = rho + (size_t)k * n;
                for (i = 0; i < nl; i++) {
                        const double *p = a + ((size_t)k * nl + i) * n, *q = c + (size_t)i * n;
                        for (g = 0; g < n; g++) rk[g] += 2 * p[g] * q[g];
                }
        }
        if (nvar == 5) {
                /* c_k = D_sub d_k phi, k = 1..3 */
                double *ck = c + (size_t)nl * n;
                for (k = 1; k < 4; k++)
                        dgemm_("N", "N", &n, &nl, &nl, &one, a + (size_t)k * nl * n, &n, d, &nl, &zero,
                               ck + (size_t)(k - 1) * nl * n, &n);
                double *r4 = rho + 4 * n;
                for (k = 1; k < 4; k++)
                        for (i = 0; i < nl; i++) {
                                const double *p = a + ((size_t)k * nl + i) * n;
                                const double *q = ck + ((size_t)(k - 1) * nl + i) * n;
                                for (g = 0; g < n; g++) r4[g] += .5 * p[g] * q[g];
                        }
        }
        (void)ncomp;
}

/*
 * The tau term of a sub-block's V (MGGA): T = sum_k d_k phi diag(wt) d_k phi^T (wt = w v_tau / 2 for
 * the 1/2 in tau), symmetric, by DSYRK on b = sqrt(+-wt) d_k phi stacked over k (3n, nl) column-major
 * (half the flops of a GEMM), the points of either sign separately; added to V as T_ij (i < j) at
 * (ix_i, ix_j) and T_ii / 2 (the caller's V + V^T completes it).  b scratch (3 n nl), t (nl, nl).
 */
static void sub_tau_v(double *V, double *b, double *t, const double *a, const double *wt, double *sw,
                      const int *ix, int nl, int n, int nao, dsyrk_t dsyrk_)
{
        const int n3 = 3 * n;
        const double one = 1, mone = -1, zero = 0;
        int i, j, g, k, sgn, done = 0;
        for (sgn = 0; sgn < 2; sgn++) {
                int any = 0;
                for (g = 0; g < n; g++) {
                        const double x = sgn == 0 ? wt[g] : -wt[g];
                        sw[g] = x > 0 ? sqrt(x) : 0;
                        any |= x > 0;
                }
                if (!any) continue;
                for (i = 0; i < nl; i++)
                        for (k = 0; k < 3; k++) {
                                const double *p = a + ((size_t)(k + 1) * nl + i) * n;
                                double *o = b + (size_t)i * n3 + (size_t)k * n;
                                for (g = 0; g < n; g++) o[g] = sw[g] * p[g];
                        }
                /* t (nl, nl) column-major upper += +- b^T b */
                dsyrk_("U", "T", &nl, &n3, sgn == 0 ? &one : &mone, b, &n3, done ? &one : &zero, t, &nl);
                done = 1;
        }
        if (!done) return;
        for (j = 0; j < nl; j++) {
                const double *tj = t + (size_t)j * nl;
                for (i = 0; i < j; i++) V[(size_t)ix[i] * nao + ix[j]] += tj[i];
                V[(size_t)ix[j] * nao + ix[j]] += .5 * tj[j];
        }
}

static void vxc_fused_impl(double *vbuf, double *ex, double *ne, double *dm, double *coords, double *weights,
                           uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                           int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                           libxc_t xcf, int nfn, void *xc_arr, double *facs,
                           dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local,
                           dsyrk_t dsyrk_, double *orb, int nocc)
{
        const int nao = ao_loc[nbas], ncomp = xctype == 0 ? 1 : 4;
        const int nvar = xctype == 0 ? 1 : (xctype == 1 ? 4 : 5);
        const int nsub = (ngrid + gs - 1) / gs, outlen = xctype == 0 ? 2 : (xctype == 1 ? 3 : 4);
        const double one = 1, zero = 0;
        const size_t nn = (size_t)nao * nao;
        /* c holds D_sub d_k phi (MGGA: 4 rows of AOs) or the orbital products (4 nocc rows) */
        const size_t nc = (size_t)(xctype == 2 ? 4 : 1) * (nao > nocc ? nao : nocc) * gs;
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *V = vbuf + tid * nn;
        double *full = malloc(sizeof(double) * ncomp * nao * gs);
        double *a = malloc(sizeof(double) * ncomp * nao * gs);
        double *c = malloc(sizeof(double) * nc);
        double *d = malloc(sizeof(double) * (nn > (size_t)nao * nocc ? nn : (size_t)nao * nocc));
        double *xyz = malloc(sizeof(double) * 3 * gs);
        double *rho = malloc(sizeof(double) * nvar * gs);
        double *out = malloc(sizeof(double) * outlen * gs);
        double *wv = malloc(sizeof(double) * nvar * gs);
        double *sw = malloc(sizeof(double) * gs);
        int *ix = malloc(sizeof(int) * nao);
        size_t *src = malloc(sizeof(size_t) * 2 * nao);
        double exs = 0, nes = 0;
        int s;
#pragma omp for schedule(dynamic, 4)
        for (s = 0; s < nsub; s++) {
                const int g0 = s * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                const uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int i, j, g, k;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select(a, ix, src, full, n, xyz, (uint8_t *)msk, ncomp, cutoff, eval,
                                           ao_loc, atm, natm, bas, nbas, env);
                if (nl == 0) continue;
                sub_rho(rho, c, d, a, ix, nl, n, nvar, ncomp, dm, nao, orb, nocc, dgemm_);
                /* the functional: out (outlen, n) = exc, v_rho (, v_sigma (, v_tau)); LIBXC_eval_xc
                 * accumulates the functionals of the mixture into out */
                memset(out, 0, sizeof(double) * outlen * n);
                xcf(nfn, xc_arr, facs, 0, 1, nvar, n, outlen, rho, out);
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
                if (nvar == 5) {
                        double *wt = wv + 4 * n;
                        for (g = 0; g < n; g++) wt[g] = .5 * w[g] * out[3 * n + g];
                        /* b (3 n nl) in c (>= 4 nl n), t in d */
                        sub_tau_v(V, c, d, a, wt, sw, ix, nl, n, nao, dsyrk_);
                }
        }
        ex[tid] = exs;
        ne[tid] = nes;
        free(full); free(a); free(c); free(d); free(xyz); free(rho); free(out); free(wv); free(sw); free(ix);
        free(src);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}

void SOCDFTvxc_fused(double *vbuf, double *ex, double *ne, double *dm, double *coords, double *weights,
                     uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                     int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                     libxc_t xcf, int nfn, void *xc_arr, double *facs,
                     dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        vxc_fused_impl(vbuf, ex, ne, dm, coords, weights, non0tab, ngrid, gs, cutoff, xctype, eval, ao_loc, atm,
                       natm, bas, nbas, env, xcf, nfn, xc_arr, facs, dgemm_, mkl_set_num_threads_local, NULL, NULL,
                       0);
}

/*
 * SOCDFTvxc_fused for LDA / GGA / meta-GGA (xctype 2: rho (5, n) = rho, nabla rho, tau; out exc, v_rho,
 * v_sigma, v_tau): the tau term of V by DSYRK (dsyrk_ of the process' BLAS); orb (nao, nocc) row-major
 * C sqrt(occ) of the occupied orbitals or NULL (then rho from D), used for the meta-GGA rho on the
 * sub-blocks with nocc < nl.
 */
void SOCDFTvxc_fused_mgga(double *vbuf, double *ex, double *ne, double *dm, double *coords, double *weights,
                          uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                          int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                          libxc_t xcf, int nfn, void *xc_arr, double *facs,
                          dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local,
                          dsyrk_t dsyrk_, double *orb, int nocc)
{
        vxc_fused_impl(vbuf, ex, ne, dm, coords, weights, non0tab, ngrid, gs, cutoff, xctype, eval, ao_loc, atm,
                       natm, bas, nbas, env, xcf, nfn, xc_arr, facs, dgemm_, mkl_set_num_threads_local, dsyrk_,
                       orb, orb ? nocc : 0);
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
 * meta-GGA (xctype 2) also + sum_k d_x d_k phi_mu T_k,mu, T_k = (w v_tau / 2) D_sub d_k phi;
 * out (nthreads, 3, nao), the gradient -2 sum_{mu in A} out[x][mu].
 */
static const int D2C[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};   /* pyscf's d_x d_k component */

/* pg != NULL (grid response, socutils.grad.grid_response): also pg (3, ngrid) = the per-point sums over
 * all AOs of the integrand of out (the gradient's point term of the owner atom is +2 sum_{g of A} pg), and
 * eg (ngrid) = exc rho, the energy density for the derivative of the partition weights */
static void xc_grad_fused_impl(double *outb, double *dm, double *coords, double *weights,
                               uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                               int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                               libxc_t xcf, int nfn, void *xc_arr, double *facs,
                               dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local, double *pg, double *eg);

void SOCDFTxc_grad_fused(double *outb, double *dm, double *coords, double *weights,
                         uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                         int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                         libxc_t xcf, int nfn, void *xc_arr, double *facs,
                         dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        xc_grad_fused_impl(outb, dm, coords, weights, non0tab, ngrid, gs, cutoff, xctype, eval, ao_loc, atm, natm,
                           bas, nbas, env, xcf, nfn, xc_arr, facs, dgemm_, mkl_set_num_threads_local, NULL, NULL);
}

void SOCDFTxc_grad_fused_gr(double *outb, double *dm, double *coords, double *weights,
                            uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                            int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                            libxc_t xcf, int nfn, void *xc_arr, double *facs,
                            dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local, double *pg, double *eg)
{
        xc_grad_fused_impl(outb, dm, coords, weights, non0tab, ngrid, gs, cutoff, xctype, eval, ao_loc, atm, natm,
                           bas, nbas, env, xcf, nfn, xc_arr, facs, dgemm_, mkl_set_num_threads_local, pg, eg);
}

static void xc_grad_fused_impl(double *outb, double *dm, double *coords, double *weights,
                               uint8_t *non0tab, int ngrid, int gs, double cutoff, int xctype, gtoval_t eval,
                               int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                               libxc_t xcf, int nfn, void *xc_arr, double *facs,
                               dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local, double *pg, double *eg)
{
        const int nao = ao_loc[nbas];
        const int nvar = xctype == 0 ? 1 : (xctype == 1 ? 4 : 5);  /* rho (, nabla rho (, tau)) */
        const int ncomp = xctype == 0 ? 4 : 10;         /* AO components evaluated */
        const int nsub = (ngrid + gs - 1) / gs, outlen = xctype == 0 ? 2 : (xctype == 1 ? 3 : 4);
        const double one = 1, zero = 0;
        const size_t nn = (size_t)nao * nao;
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *out = outb + (size_t)tid * 3 * nao;
        double *full = malloc(sizeof(double) * ncomp * nao * gs);
        double *a = malloc(sizeof(double) * ncomp * nao * gs);
        /* V = D phi_0 (MGGA: then D d_k phi, k = 1..3, for tau and its term) */
        double *c = malloc(sizeof(double) * (xctype == 2 ? 4 : 1) * nao * gs);
        double *aw = malloc(sizeof(double) * nao * gs);
        double *u = malloc(sizeof(double) * nao * gs);          /* U = D aow */
        double *d = malloc(sizeof(double) * nn);
        double *xyz = malloc(sizeof(double) * 3 * gs);
        double *rho = malloc(sizeof(double) * nvar * gs);
        double *xo = malloc(sizeof(double) * outlen * gs);
        double *wv = malloc(sizeof(double) * nvar * gs);
        int *ix = malloc(sizeof(int) * nao);
        size_t *src = malloc(sizeof(size_t) * 2 * nao);
        int s;
#pragma omp for schedule(dynamic, 4)
        for (s = 0; s < nsub; s++) {
                const int g0 = s * gs, n = g0 + gs <= ngrid ? gs : ngrid - g0;
                const uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int i, j, g, k, x;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select(a, ix, src, full, n, xyz, (uint8_t *)msk, ncomp, cutoff, eval,
                                           ao_loc, atm, natm, bas, nbas, env);
                if (pg) {
                        for (x = 0; x < 3; x++) memset(pg + (size_t)x * ngrid + g0, 0, sizeof(double) * n);
                        memset(eg + g0, 0, sizeof(double) * n);
                }
                if (nl == 0) continue;
                /* rho from D (d = D_sub, c = D_sub d_k phi) */
                sub_rho(rho, c, d, a, ix, nl, n, nvar, ncomp, dm, nao, NULL, 0, dgemm_);
                memset(xo, 0, sizeof(double) * outlen * n);
                xcf(nfn, xc_arr, facs, 0, 1, nvar, n, outlen, rho, xo);
                const double *w = weights + g0;
                for (g = 0; g < n; g++) wv[g] = w[g] * xo[n + g];
                if (eg)
                        for (g = 0; g < n; g++) eg[g0 + g] = xo[g] * rho[g];
                if (nvar >= 4) {
                        for (g = 0; g < n; g++) wv[g] *= .5;
                        for (k = 1; k < 4; k++)
                                for (g = 0; g < n; g++) wv[k * n + g] = 2 * w[g] * xo[2 * n + g] * rho[k * n + g];
                }
                if (nvar == 5) {
                        /* T_k = wt D d_k phi (wt = w v_tau / 2) in place of D d_k phi */
                        for (g = 0; g < n; g++) wv[4 * n + g] = .5 * w[g] * xo[3 * n + g];
                        const double *wt = wv + 4 * n;
                        double *ck = c + (size_t)nl * n;
                        for (i = 0; i < 3 * nl; i++) {
                                double *t = ck + (size_t)i * n;
                                for (g = 0; g < n; g++) t[g] *= wt[g];
                        }
                }
                /* aow = sum_k wv_k d_k phi; U = D_sub aow */
                for (i = 0; i < nl; i++) {
                        double *o = aw + (size_t)i * n;
                        const double *p = a + (size_t)i * n;
                        for (g = 0; g < n; g++) o[g] = wv[g] * p[g];
                        for (k = 1; k < (nvar < 4 ? nvar : 4); k++) {
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
                                if (pg) {
                                        /* the same sums, also point by point */
                                        double *px = pg + (size_t)x * ngrid + g0;
                                        if (nvar == 1) {
                                                for (g = 0; g < n; g++) {
                                                        const double t = ax[g] * ui[g];
                                                        sm += t;
                                                        px[g] += t;
                                                }
                                        } else {
                                                const double *a0 = a + ((size_t)D2C[x][0] * nl + i) * n;
                                                const double *a1 = a + ((size_t)D2C[x][1] * nl + i) * n;
                                                const double *a2 = a + ((size_t)D2C[x][2] * nl + i) * n;
                                                const double *w1 = wv + n, *w2 = wv + 2 * n, *w3 = wv + 3 * n;
                                                for (g = 0; g < n; g++) {
                                                        const double t = ax[g] * (ui[g] + wv[g] * vi[g])
                                                                       + (a0[g] * w1[g] + a1[g] * w2[g] + a2[g] * w3[g]) * vi[g];
                                                        sm += t;
                                                        px[g] += t;
                                                }
                                        }
                                } else if (nvar == 1) {
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
                                if (nvar == 5) {
                                        /* + sum_k d_x d_k phi T_k */
                                        const double *a0 = a + ((size_t)D2C[x][0] * nl + i) * n;
                                        const double *a1 = a + ((size_t)D2C[x][1] * nl + i) * n;
                                        const double *a2 = a + ((size_t)D2C[x][2] * nl + i) * n;
                                        const double *t0 = c + ((size_t)nl + i) * n;
                                        const double *t1 = c + ((size_t)2 * nl + i) * n;
                                        const double *t2 = c + ((size_t)3 * nl + i) * n;
                                        double st = 0;
                                        if (pg) {
                                                double *px = pg + (size_t)x * ngrid + g0;
                                                for (g = 0; g < n; g++) {
                                                        const double t = a0[g] * t0[g] + a1[g] * t1[g] + a2[g] * t2[g];
                                                        st += t;
                                                        px[g] += t;
                                                }
                                        } else {
                                                for (g = 0; g < n; g++) st += a0[g] * t0[g] + a1[g] * t1[g] + a2[g] * t2[g];
                                        }
                                        sm += st;
                                }
                                out[(size_t)x * nao + ix[i]] += sm;
                        }
                }
        }
        free(full); free(a); free(c); free(aw); free(u); free(d); free(xyz); free(rho); free(xo); free(wv); free(src);
        free(ix);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}

#include "vxc_2c.h"     /* SOCDFTvxc_fused_2c: the two-component (GKS / spinor) pass */
#include "fxc_2c_mo.h"  /* SOCDFTfxc2c_mo: the closed-shell two-component f_xc response, MO form */
