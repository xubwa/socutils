/*
 * Elementwise grid work of the RKS XC response for many first-order
 * densities at once (socutils.hessian.rhf_fast, CPHF), between the two
 * batched GEMMs  c = D phi^T  (all densities stacked) and  V = aow phi,
 * fused in one pass over c:
 *
 *   rho1[k,g] = s_k sum_mu c[m,mu,g] ao[k,mu,g]        (s_0 = 1, s_k = 2: nabla rho)
 *   wv[x,g]   = sum_y fw[x,y,g] rho1[y,g]              (fw = fxc * weight, x = 0 row
 *                                                       halved for GGA by the caller)
 *   c[m,mu,g] <- aow[m,mu,g] = sum_k wv[k,g] ao[k,mu,g]
 *
 * c is (ndm, nao, ngrid) (in: D phi^T, out: aow), ao the AO derivative stack in
 * memory order (ncomp, nao, ngrid), fw (ncomp, ncomp, ngrid); ncomp = 1 (LDA)
 * or 4 (GGA).  OpenMP over (chunk of grid points, density), chunk-major so that
 * the threads share the chunk of ao (ncomp x nao x chunk); the chunk length is
 * chosen from the L2 size and nao (grid_chunk); the chunk of
 * c of one density stays in cache between its rho1 and aow passes.
 *
 * SOCDFTgather_dm / SOCDFTscatter_add_dm: the sub-matrices on the AOs idx of
 * all densities, out[m][i][j] = d[m][idx_i][idx_j], and v[m][idx_i][idx_j] +=
 * x[m][i][j].
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#define GCHUNK_MAX 256

/* L2 cache size in bytes: SOCUTILS_L2_BYTES, else sysconf, else sysfs, else 512 kB */
static long l2_bytes(void)
{
        static long cached = 0;
        if (cached > 0) return cached;
        long v = 0;
        const char *e = getenv("SOCUTILS_L2_BYTES");
        if (e) v = atol(e);
#ifdef _SC_LEVEL2_CACHE_SIZE
        if (v <= 0) v = sysconf(_SC_LEVEL2_CACHE_SIZE);
#endif
        if (v <= 0) {
                FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
                if (f) {
                        char unit = 0;
                        long x = 0;
                        if (fscanf(f, "%ld%c", &x, &unit) >= 1)
                                v = x * (unit == 'K' ? 1024 : unit == 'M' ? 1024 * 1024 : 1);
                        fclose(f);
                }
        }
        if (v <= 0) v = 512 * 1024;
        cached = v;
        return v;
}

/* grid chunk: the chunk of the AO stack (ncomp x nao) plus that of c (nao) of one
 * task in half of L2, a multiple of 8 in [8, GCHUNK_MAX] */
static int grid_chunk(int ncomp, int nao)
{
        long gc = l2_bytes() / 2 / (8L * (ncomp + 1) * (nao > 0 ? nao : 1));
        gc = gc / 8 * 8;
        if (gc < 8) gc = 8;
        if (gc > GCHUNK_MAX) gc = GCHUNK_MAX;
        return (int)gc;
}

void SOCDFTfxc_fused(double *c, double *ao, double *fw, int ndm, int ncomp, int nao, int ngrid)
{
        const size_t G = ngrid;
        const int GCHUNK = grid_chunk(ncomp, nao);
        const int nchunk = (ngrid + GCHUNK - 1) / GCHUNK;
        const long ntask = (long)nchunk * ndm;
        long it;
#pragma omp parallel for schedule(dynamic, 2)
        for (it = 0; it < ntask; it++) {
                /* chunk-major: consecutive tasks share the chunk of ao */
                const int ic = it / ndm, m = it % ndm;
                const int g0 = ic * GCHUNK;
                const int ng = (g0 + GCHUNK < ngrid ? g0 + GCHUNK : ngrid) - g0;
                double r[4][GCHUNK_MAX], w[4][GCHUNK_MAX];
                int k, x, mu, g;
                {
                        for (k = 0; k < ncomp; k++)
                                for (g = 0; g < ng; g++) r[k][g] = 0;
                        for (mu = 0; mu < nao; mu++) {
                                const double *cm = c + ((size_t)m * nao + mu) * G + g0;
                                for (k = 0; k < ncomp; k++) {
                                        const double *a = ao + ((size_t)k * nao + mu) * G + g0;
#pragma omp simd
                                        for (g = 0; g < ng; g++) r[k][g] += cm[g] * a[g];
                                }
                        }
                        for (k = 1; k < ncomp; k++)
                                for (g = 0; g < ng; g++) r[k][g] *= 2;
                        for (x = 0; x < ncomp; x++) {
                                for (g = 0; g < ng; g++) w[x][g] = 0;
                                for (k = 0; k < ncomp; k++) {
                                        const double *f = fw + ((size_t)x * ncomp + k) * G + g0;
#pragma omp simd
                                        for (g = 0; g < ng; g++) w[x][g] += f[g] * r[k][g];
                                }
                        }
                        for (mu = 0; mu < nao; mu++) {
                                double *o = c + ((size_t)m * nao + mu) * G + g0;
                                const double *a0 = ao + (size_t)mu * G + g0;
#pragma omp simd
                                for (g = 0; g < ng; g++) o[g] = w[0][g] * a0[g];
                                for (k = 1; k < ncomp; k++) {
                                        const double *a = ao + ((size_t)k * nao + mu) * G + g0;
#pragma omp simd
                                        for (g = 0; g < ng; g++) o[g] += w[k][g] * a[g];
                                }
                        }
                }
        }
}

/*
 * SOCDFTfxc_fused for meta-GGA: the AOs as rows of one GEMM operand, b (nao, 4, ngrid) =
 * [phi_0 | d_x phi | d_y phi | d_z phi] per AO, and c (ndm, nao, 4, ngrid) = D b (in) /
 * the left operand of V += c b^T (out), so that the value, gradient and tau terms of V are
 * one GEMM over 4 ngrid columns:
 *
 *   rho1 = (sum_mu c_0 b_0, 2 sum_mu c_0 b_k, 1/2 sum_k sum_mu c_k b_k),
 *   wv[x] = sum_y fw[x,y] rho1[y]          (fw (5, 5, ngrid); the caller halves row 0
 *                                           and quarters row 4 for V + V^T),
 *   c_0 <- sum_{k<4} wv[k] b_k,  c_k <- wv[4] b_k (k = 1..3).
 */
void SOCDFTfxc_fused_mgga(double *c, double *b, double *fw, int ndm, int nao, int ngrid)
{
        const size_t G = ngrid;
        const int GCHUNK = grid_chunk(4, nao);
        const int nchunk = (ngrid + GCHUNK - 1) / GCHUNK;
        const long ntask = (long)nchunk * ndm;
        long it;
#pragma omp parallel for schedule(dynamic, 2)
        for (it = 0; it < ntask; it++) {
                const int ic = it / ndm, m = it % ndm;
                const int g0 = ic * GCHUNK;
                const int ng = (g0 + GCHUNK < ngrid ? g0 + GCHUNK : ngrid) - g0;
                double r[5][GCHUNK_MAX], w[5][GCHUNK_MAX];
                int k, x, mu, g;
                for (k = 0; k < 5; k++)
                        for (g = 0; g < ng; g++) r[k][g] = 0;
                for (mu = 0; mu < nao; mu++) {
                        const double *cm = c + ((size_t)m * nao + mu) * 4 * G + g0;
                        const double *a = b + (size_t)mu * 4 * G + g0;
#pragma omp simd
                        for (g = 0; g < ng; g++) {
                                r[0][g] += cm[g] * a[g];
                                r[1][g] += cm[g] * a[G + g];
                                r[2][g] += cm[g] * a[2 * G + g];
                                r[3][g] += cm[g] * a[3 * G + g];
                                r[4][g] += cm[G + g] * a[G + g] + cm[2 * G + g] * a[2 * G + g]
                                         + cm[3 * G + g] * a[3 * G + g];
                        }
                }
                for (g = 0; g < ng; g++) {
                        r[1][g] *= 2;
                        r[2][g] *= 2;
                        r[3][g] *= 2;
                        r[4][g] *= .5;
                }
                for (x = 0; x < 5; x++) {
                        for (g = 0; g < ng; g++) w[x][g] = 0;
                        for (k = 0; k < 5; k++) {
                                const double *f = fw + ((size_t)x * 5 + k) * G + g0;
#pragma omp simd
                                for (g = 0; g < ng; g++) w[x][g] += f[g] * r[k][g];
                        }
                }
                for (mu = 0; mu < nao; mu++) {
                        double *o = c + ((size_t)m * nao + mu) * 4 * G + g0;
                        const double *a = b + (size_t)mu * 4 * G + g0;
#pragma omp simd
                        for (g = 0; g < ng; g++) {
                                o[g] = w[0][g] * a[g] + w[1][g] * a[G + g] + w[2][g] * a[2 * G + g]
                                     + w[3][g] * a[3 * G + g];
                                o[G + g] = w[4][g] * a[G + g];
                                o[2 * G + g] = w[4][g] * a[2 * G + g];
                                o[3 * G + g] = w[4][g] * a[3 * G + g];
                        }
                }
        }
}

/* b[i][k][g] = ao[k][idx_i][s0 + g] (idx NULL: all AOs), (nl, ncomp, gs): the AO
 * components of a sub-block as the rows of one GEMM operand (ao as in SOCDFTgather_ao) */
void SOCDFTgather_ao_rows(double *b, double *ao, int *idx, int ncomp, int nao, int nl,
                          int ngrid, int s0, int gs, long cs)
{
        int i;
        (void)nao;
#pragma omp parallel for schedule(static)
        for (i = 0; i < nl; i++) {
                const size_t mu = idx ? (size_t)idx[i] : (size_t)i;
                int k;
                for (k = 0; k < ncomp; k++)
                        memcpy(b + ((size_t)i * ncomp + k) * gs, ao + k * cs + mu * ngrid + s0,
                               sizeof(double) * gs);
        }
}

void SOCDFTgather_dm(double *out, double *d, int *idx, int nl, int ndm, int nao)
{
        const size_t nn = (size_t)nao * nao, ll = (size_t)nl * nl;
        long it;
#pragma omp parallel for schedule(static)
        for (it = 0; it < (long)ndm * nl; it++) {
                const int m = it / nl, i = it % nl;
                const double *src = d + m * nn + (size_t)idx[i] * nao;
                double *o = out + m * ll + (size_t)i * nl;
                int j;
                for (j = 0; j < nl; j++) o[j] = src[idx[j]];
        }
}

void SOCDFTscatter_add_dm(double *v, double *x, int *idx, int nl, int ndm, int nao)
{
        const size_t nn = (size_t)nao * nao, ll = (size_t)nl * nl;
        long it;
#pragma omp parallel for schedule(static)
        for (it = 0; it < (long)ndm * nl; it++) {
                const int m = it / nl, i = it % nl;
                double *dst = v + m * nn + (size_t)idx[i] * nao;
                const double *src = x + m * ll + (size_t)i * nl;
                int j;
                for (j = 0; j < nl; j++) dst[idx[j]] += src[j];
        }
}

/*
 * aow[m,mu,g] = sum_k wv[m,k,g] ao[k,mu,g] for many weight sets m (e.g. the
 * 3 natm first-order XC potentials of the CPHF right-hand sides); layouts as
 * above, wv (ndm, ncomp, ngrid).  OpenMP over (grid chunk, m), chunk-major.
 */
void SOCDFTaow_multi(double *aow, double *wv, double *ao, int ndm, int ncomp, int nao, int ngrid)
{
        const size_t G = ngrid;
        const int GCHUNK = grid_chunk(ncomp, nao);
        const int nchunk = (ngrid + GCHUNK - 1) / GCHUNK;
        const long ntask = (long)nchunk * ndm;
        long it;
#pragma omp parallel for schedule(dynamic, 2)
        for (it = 0; it < ntask; it++) {
                const int ic = it / ndm, m = it % ndm;
                const int g0 = ic * GCHUNK;
                const int ng = (g0 + GCHUNK < ngrid ? g0 + GCHUNK : ngrid) - g0;
                const double *w = wv + (size_t)m * ncomp * G + g0;
                int k, mu, g;
                for (mu = 0; mu < nao; mu++) {
                        double *o = aow + ((size_t)m * nao + mu) * G + g0;
                        const double *a0 = ao + (size_t)mu * G + g0;
#pragma omp simd
                        for (g = 0; g < ng; g++) o[g] = w[g] * a0[g];
                        for (k = 1; k < ncomp; k++) {
                                const double *a = ao + ((size_t)k * nao + mu) * G + g0;
                                const double *wk = w + (size_t)k * G;
#pragma omp simd
                                for (g = 0; g < ng; g++) o[g] += wk[g] * a[g];
                        }
                }
        }
}

/* out[k][i][g] = ao[k][idx_i][s0 + g] (idx NULL: all AOs), (ncomp, nl, gs) from
 * the AO stack (ncomp, nao, ngrid) with cs doubles between components: a
 * sub-block in contiguous memory. */
void SOCDFTgather_ao(double *out, double *ao, int *idx, int ncomp, int nao, int nl,
                     int ngrid, int s0, int gs, long cs)
{
        long it;
#pragma omp parallel for schedule(static)
        for (it = 0; it < (long)ncomp * nl; it++) {
                const int k = it / nl, i = it % nl;
                const int mu = idx ? idx[i] : i;
                memcpy(out + it * gs, ao + (size_t)k * cs + (size_t)mu * ngrid + s0, sizeof(double) * gs);
        }
}

/* out[r][g] = src[r][s0 + g], (nrow, gs) from (nrow, ngrid) */
void SOCDFTslice_grid(double *out, double *src, long nrow, int ngrid, int s0, int gs)
{
        long r;
#pragma omp parallel for schedule(static)
        for (r = 0; r < nrow; r++)
                memcpy(out + r * gs, src + r * ngrid + s0, sizeof(double) * gs);
}

/* wv[(a * 3 + i)][x][g] = scale sum_y wf[x][y][g] R[i][y][a][g], row x = 0 halved
 * if halve0: the first-order XC potentials of all (atom, direction) from the
 * per-atom density derivatives R (3, nvar, natm, ngrid) and fxc * weight. */
void SOCDFTfxc_contract_R(double *wv, double *wf, double *R, int natm, int nvar,
                          int ngrid, double scale, int halve0)
{
        const size_t G = ngrid;
        long it;
#pragma omp parallel for schedule(static)
        for (it = 0; it < (long)natm * 3; it++) {
                const int a = it / 3, i = it % 3;
                double *o = wv + (size_t)it * nvar * G;
                int x, y;
                size_t g;
                for (x = 0; x < nvar; x++) {
                        double *ox = o + (size_t)x * G;
                        const double s = (halve0 && x == 0) ? .5 * scale : scale;
                        for (g = 0; g < G; g++) ox[g] = 0;
                        for (y = 0; y < nvar; y++) {
                                const double *f = wf + ((size_t)x * nvar + y) * G;
                                const double *r = R + (((size_t)i * nvar + y) * natm + a) * G;
#pragma omp simd
                                for (g = 0; g < G; g++) ox[g] += f[g] * r[g];
                        }
                        for (g = 0; g < G; g++) ox[g] *= s;
                }
        }
}

/* out[a][i][k][g] = R[i][k][a][g]: the per-atom density derivatives (3, nvar,
 * natm, ngrid) as rows (atom, direction) x columns (component, grid point). */
void SOCDFTreorder_R(double *out, double *R, int natm, int nvar, int ngrid)
{
        const size_t G = ngrid;
        long it;
#pragma omp parallel for schedule(static)
        for (it = 0; it < (long)natm * 3 * nvar; it++) {
                const int a = it / (3 * nvar), i = (it / nvar) % 3, k = it % nvar;
                memcpy(out + (size_t)it * G, R + (((size_t)i * nvar + k) * natm + a) * G, sizeof(double) * G);
        }
}

/* out[k][i] = [A[k][idx_i][s0:s0+gs], B[k][idx_i][s0:s0+gs]] for k < 3: rows of
 * the two operands of one GEMM over 2 gs columns; A, B the first of three
 * components, ca / cb doubles apart, each (nao, ngrid). */
void SOCDFTgather_rows2(double *out, double *A, long ca, double *B, long cb, int *idx,
                        int nao, int nl, int ngrid, int s0, int gs)
{
        long it;
#pragma omp parallel for schedule(static)
        for (it = 0; it < 3L * nl; it++) {
                const int k = it / nl, i = it % nl;
                const int mu = idx ? idx[i] : i;
                double *o = out + (size_t)it * 2 * gs;
                memcpy(o, A + (size_t)k * ca + (size_t)mu * ngrid + s0, sizeof(double) * gs);
                memcpy(o + gs, B + (size_t)k * cb + (size_t)mu * ngrid + s0, sizeof(double) * gs);
        }
}

/*
 * SOCDFTxcd1_occ: the first-order XC matrices of one sub-block projected on the
 * occupied orbitals, (V_r + V_r^T) C for every row r = (atom, x) of ws, where
 * V_r[mu][nu] = sum_g aow_r[mu][g] phi_0[nu][g], aow_r = sum_k ws[r][k] phi_k:
 *
 *      T_r = [aow_r | phi_0] . [psi_0 ; Y_r],   Y_r[g][i] = sum_k ws[r][k][g] psi_k[g][i]
 *
 * one (nl x 2 gs) . (2 gs x nocc) GEMM per r (OpenMP over r, single-threaded
 * dgemm), aow_r and Y_r built in the thread's buffers.  psi (nvar, gs, nocc) =
 * phi_k^T C on the sub-block's AOs, ao (nvar, nl, gs), ws (nr, nvar, gs).
 * out (*, nao, nocc): out[orow[r]][idx[mu]] += T_r[mu].
 *
 * nvar 5 (meta-GGA; ws row 4 the tau weight u_r, psi and ao then of the 4 components
 * phi_0, nabla phi only): V_r also has the symmetric sum_j (u_r d_j phi) d_j phi^T,
 * whose share of (V_r + V_r^T) C, 2 sum_j d_j phi u_r psi_j, comes from 3 gs more columns:
 *
 *      T_r = [aow_r | phi_0 | d_1 phi | d_2 phi | d_3 phi] . [psi_0 ; Y_r ; 2 u_r psi_1 ; ..]
 */
typedef void (*xcd1_dgemm_t)(const char *ta, const char *tb, const int *m, const int *n, const int *k,
                             const double *alpha, const double *a, const int *lda, const double *b,
                             const int *ldb, const double *beta, double *c, const int *ldc);
typedef int (*xcd1_mkl_local_t)(int nt);

void SOCDFTxcd1_occ(double *out, double *ws, double *ao, double *psi, int *idx, int *orow, int nr,
                    int nvar, int nl, int gs, int nocc, int nao, xcd1_dgemm_t dgemm_,
                    xcd1_mkl_local_t mkl_set_num_threads_local)
{
        const size_t G = gs;
        const int mg = nvar == 5;
        const int nk = mg ? 4 : nvar;                           /* AO components in ao / psi */
        const int nb = mg ? 5 : 2;                              /* column blocks of the GEMM */
        const int k2 = nb * gs;
        const double one = 1, zero = 0;
#pragma omp parallel
{
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        double *A = malloc(sizeof(double) * nl * nb * G);       /* rows mu: [aow_r | phi_0 (| nabla phi)] */
        double *B = malloc(sizeof(double) * nb * G * nocc);     /* [psi_0 ; Y_r (; 2 u psi_j)] */
        double *T = malloc(sizeof(double) * nl * nocc);
        int mu, g, k, i, r;
        for (mu = 0; mu < nl; mu++) {
                memcpy(A + (size_t)mu * k2 + G, ao + (size_t)mu * G, sizeof(double) * G);
                if (mg)
                        for (k = 1; k < 4; k++)
                                memcpy(A + (size_t)mu * k2 + (size_t)(k + 1) * G, ao + ((size_t)k * nl + mu) * G,
                                       sizeof(double) * G);
        }
        memcpy(B, psi, sizeof(double) * G * nocc);
#pragma omp for schedule(dynamic, 1)
        for (r = 0; r < nr; r++) {
                const double *w = ws + (size_t)r * nvar * G;
                for (mu = 0; mu < nl; mu++) {
                        double *o = A + (size_t)mu * k2;
                        const double *a0 = ao + (size_t)mu * G;
#pragma omp simd
                        for (g = 0; g < gs; g++) o[g] = w[g] * a0[g];
                        for (k = 1; k < nk; k++) {
                                const double *a = ao + ((size_t)k * nl + mu) * G;
                                const double *wk = w + (size_t)k * G;
#pragma omp simd
                                for (g = 0; g < gs; g++) o[g] += wk[g] * a[g];
                        }
                }
                double *Y = B + G * nocc;
                for (g = 0; g < gs; g++) {
                        double *y = Y + (size_t)g * nocc;
                        const double *p0 = psi + (size_t)g * nocc;
                        const double w0 = w[g];
#pragma omp simd
                        for (i = 0; i < nocc; i++) y[i] = w0 * p0[i];
                        for (k = 1; k < nk; k++) {
                                const double *pk = psi + ((size_t)k * G + g) * nocc;
                                const double wk = w[(size_t)k * G + g];
#pragma omp simd
                                for (i = 0; i < nocc; i++) y[i] += wk * pk[i];
                        }
                        if (mg) {
                                const double u2 = 2 * w[4 * G + g];
                                for (k = 1; k < 4; k++) {
                                        double *z = B + ((size_t)(k + 1) * G + g) * nocc;
                                        const double *pk = psi + ((size_t)k * G + g) * nocc;
#pragma omp simd
                                        for (i = 0; i < nocc; i++) z[i] = u2 * pk[i];
                                }
                        }
                }
                /* row-major T (nl, nocc) = A (nl, nb gs) . B (nb gs, nocc) */
                dgemm_("N", "N", &nocc, &nl, &k2, &one, B, &nocc, A, &k2, &zero, T, &nocc);
                double *dst = out + (size_t)orow[r] * nao * nocc;
                for (mu = 0; mu < nl; mu++) {
                        double *d = dst + (size_t)idx[mu] * nocc;
                        const double *t = T + (size_t)mu * nocc;
#pragma omp simd
                        for (i = 0; i < nocc; i++) d[i] += t[i];
                }
        }
        free(A);
        free(B);
        free(T);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
}
