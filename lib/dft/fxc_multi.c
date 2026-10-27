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
 * or 4 (GGA).  OpenMP over (chunk of GCHUNK grid points, density), chunk-major
 * so that the threads share the chunk of ao (ncomp x nao x GCHUNK); the chunk of
 * c of one density stays in cache between its rho1 and aow passes.
 *
 * SOCDFTgather_dm / SOCDFTscatter_add_dm: the sub-matrices on the AOs idx of
 * all densities, out[m][i][j] = d[m][idx_i][idx_j], and v[m][idx_i][idx_j] +=
 * x[m][i][j].
 */
#include <stdlib.h>
#include <string.h>

#ifndef GCHUNK
#define GCHUNK 64
#endif

void SOCDFTfxc_fused(double *c, double *ao, double *fw, int ndm, int ncomp, int nao, int ngrid)
{
        const size_t G = ngrid;
        const int nchunk = (ngrid + GCHUNK - 1) / GCHUNK;
        const long ntask = (long)nchunk * ndm;
        long it;
#pragma omp parallel for schedule(dynamic, 2)
        for (it = 0; it < ntask; it++) {
                /* chunk-major: consecutive tasks share the chunk of ao */
                const int ic = it / ndm, m = it % ndm;
                const int g0 = ic * GCHUNK;
                const int ng = (g0 + GCHUNK < ngrid ? g0 + GCHUNK : ngrid) - g0;
                double r[4][GCHUNK], w[4][GCHUNK];
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
