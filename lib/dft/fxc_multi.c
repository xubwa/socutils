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
