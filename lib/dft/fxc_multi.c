/*
 * Elementwise grid work of the RKS XC response for many first-order
 * densities at once (socutils.hessian.rhf_fast, CPHF), between the two
 * batched GEMMs  c = D phi^T  (all densities stacked) and  V = aow phi:
 *
 *   rho1[m,k,g] = s_k sum_mu c[m,mu,g] ao[k,mu,g]   (s_0 = 1, s_k = 2: nabla rho)
 *   aow[m,mu,g] = sum_k wv[m,k,g] ao[k,mu,g]
 *
 * ao is the AO derivative stack in memory order (ncomp, nao, ngrid), c and aow
 * (ndm, nao, ngrid), rho1 / wv (ndm, ncomp, ngrid).  ncomp = 1 (LDA) or 4 (GGA).
 * OpenMP over chunks of GCHUNK grid points, so that the chunk of ao stays in
 * cache while all densities pass over it (ao is read once, not ndm times).
 */
#include <stdlib.h>
#include <string.h>

#ifndef GCHUNK
#define GCHUNK 128
#endif

void SOCDFTfxc_rho1(double *rho1, double *c, double *ao, int ndm, int ncomp, int nao, int ngrid)
{
        const size_t G = ngrid;
        const int nchunk = (ngrid + GCHUNK - 1) / GCHUNK;
        int ic;
#pragma omp parallel for schedule(static)
        for (ic = 0; ic < nchunk; ic++) {
                const int g0 = ic * GCHUNK;
                const int ng = (g0 + GCHUNK < ngrid ? g0 + GCHUNK : ngrid) - g0;
                int m, k, mu, g;
                for (m = 0; m < ndm; m++) {
                        double r[4][GCHUNK];
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
                        for (k = 0; k < ncomp; k++) {
                                const double s = k ? 2. : 1.;
                                double *rk = rho1 + ((size_t)m * ncomp + k) * G + g0;
                                for (g = 0; g < ng; g++) rk[g] = s * r[k][g];
                        }
                }
        }
}

void SOCDFTfxc_aow(double *aow, double *wv, double *ao, int ndm, int ncomp, int nao, int ngrid)
{
        const size_t G = ngrid;
        const int nchunk = (ngrid + GCHUNK - 1) / GCHUNK;
        int ic;
#pragma omp parallel for schedule(static)
        for (ic = 0; ic < nchunk; ic++) {
                const int g0 = ic * GCHUNK;
                const int ng = (g0 + GCHUNK < ngrid ? g0 + GCHUNK : ngrid) - g0;
                int m, k, mu, g;
                for (m = 0; m < ndm; m++) {
                        const double *w = wv + (size_t)m * ncomp * G + g0;
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
}
