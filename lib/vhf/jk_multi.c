/*
 * Direct J / K for many density matrices at once (CPHF, TDDFT responses):
 *
 *     J_ab = sum_cd (ab|cd) D_dc,     K_ad = sum_bc (ab|cd) D_bc,
 *
 * over the unique shell quartets (i >= j, k >= l, ij >= kl), one libcint
 * evaluation per quartet shared by all densities.  The densities are stored
 * as (nao, nao, ndm), density index innermost, so that each integral updates
 * J and K of all densities with contiguous vector operations (the eight
 * permutations of the quartet, any symmetry of D).  Quartets are skipped when
 * q[ij] q[kl] (Schwarz) times the largest density element on the six index
 * pairs is below tol.  Per-thread J / K, reduced at the end.  Spherical AOs.
 */
#include <stdlib.h>
#include <string.h>
#include "cint.h"

int int2e_sph(double *out, int *dims, int *shls, int *atm, int natm,
              int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

static const int PERMS[8][4] = {
        {0, 1, 2, 3}, {1, 0, 2, 3}, {0, 1, 3, 2}, {1, 0, 3, 2},
        {2, 3, 0, 1}, {3, 2, 0, 1}, {2, 3, 1, 0}, {3, 2, 1, 0}};

static inline double dmax2(double a, double b) { return a > b ? a : b; }

static inline void shell_pair(int ij, int *i, int *j)
{
        int a = (int)((__builtin_sqrt(1. + 8. * ij) - 1) / 2);
        while (a * (a + 1) / 2 > ij) a--;
        while ((a + 1) * (a + 2) / 2 <= ij) a++;
        *i = a;
        *j = ij - a * (a + 1) / 2;
}

void SOCVHFjk_multi(double *vj, double *vk, double *dms, int ndm,
                    int with_j, int with_k,
                    double *qcond, double *dcond, double tol, CINTOpt *opt,
                    int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        const size_t nv = nao * nao * ndm;
        int dmax = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        int cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)dmax * dmax * dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        if (with_j) memset(vj, 0, sizeof(double) * nv);
        if (with_k) memset(vk, 0, sizeof(double) * nv);
#pragma omp parallel
{
        double *jloc = with_j ? calloc(nv, sizeof(double)) : NULL;
        double *kloc = with_k ? calloc(nv, sizeof(double)) : NULL;
        double *buf = malloc(sizeof(double) * (nbuf + cache_size));
        double *cache = buf + nbuf;
        int ij, kl, ish, jsh, ksh, lsh, p, m;
        int shls[4];
#pragma omp for schedule(dynamic, 2)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        double d = dmax2(dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]),
                                         dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                               dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh])));
                        if (q4 * d < tol) {
                                continue;
                        }
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        if (!int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                                continue;
                        }
                        int di = ao_loc[ish + 1] - ao_loc[ish];
                        int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                        int dl = ao_loc[lsh + 1] - ao_loc[lsh];
                        double fac = 1.;
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        int i0, j0, k0, l0;
                        for (l0 = 0; l0 < dl; l0++) {
                        for (k0 = 0; k0 < dk; k0++) {
                        for (j0 = 0; j0 < dj; j0++) {
                        for (i0 = 0; i0 < di; i0++) {
                                double g = buf[((l0 * dk + k0) * dj + j0) * di + i0] * fac;
                                if (g == 0) continue;
                                size_t ao[4] = {ao_loc[ish] + i0, ao_loc[jsh] + j0,
                                                ao_loc[ksh] + k0, ao_loc[lsh] + l0};
                                for (p = 0; p < 8; p++) {
                                        size_t a = ao[PERMS[p][0]], b = ao[PERMS[p][1]];
                                        size_t c = ao[PERMS[p][2]], e = ao[PERMS[p][3]];
                                        if (with_j) {
                                                double *v = jloc + (a * nao + b) * ndm;
                                                double *dd = dms + (e * nao + c) * ndm;
#pragma omp simd
                                                for (m = 0; m < ndm; m++) v[m] += g * dd[m];
                                        }
                                        if (with_k) {
                                                double *v = kloc + (a * nao + e) * ndm;
                                                double *dd = dms + (b * nao + c) * ndm;
#pragma omp simd
                                                for (m = 0; m < ndm; m++) v[m] += g * dd[m];
                                        }
                                }
                        } } } }
                }
        }
#pragma omp critical
        {
                size_t n;
                if (with_j) for (n = 0; n < nv; n++) vj[n] += jloc[n];
                if (with_k) for (n = 0; n < nv; n++) vk[n] += kloc[n];
        }
        free(jloc);
        free(kloc);
        free(buf);
}
}

/*
 * The same for symmetric densities (D_cd = D_dc; CPHF): J and K are then
 * symmetric, the eight permutations collapse to two J and four K updates per
 * integral, W += ..., and J = W + W^T at the end.
 */
void SOCVHFjk_multi_sym(double *vj, double *vk, double *dms, int ndm,
                        int with_j, int with_k,
                        double *qcond, double *dcond, double tol, CINTOpt *opt,
                        int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        const size_t nv = nao * nao * ndm;
        int dmax = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        int cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)dmax * dmax * dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        if (with_j) memset(vj, 0, sizeof(double) * nv);
        if (with_k) memset(vk, 0, sizeof(double) * nv);
#pragma omp parallel
{
        double *jloc = with_j ? calloc(nv, sizeof(double)) : NULL;
        double *kloc = with_k ? calloc(nv, sizeof(double)) : NULL;
        double *buf = malloc(sizeof(double) * (nbuf + cache_size));
        double *cache = buf + nbuf;
        int ij, kl, ish, jsh, ksh, lsh, m;
        int shls[4];
#pragma omp for schedule(dynamic, 2)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        double d = dmax2(dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]),
                                         dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                               dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh])));
                        if (q4 * d < tol) {
                                continue;
                        }
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        if (!int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                                continue;
                        }
                        int di = ao_loc[ish + 1] - ao_loc[ish];
                        int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                        int dl = ao_loc[lsh + 1] - ao_loc[lsh];
                        double fac = 1.;
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        int i0, j0, k0, l0;
                        for (l0 = 0; l0 < dl; l0++) {
                        for (k0 = 0; k0 < dk; k0++) {
                        for (j0 = 0; j0 < dj; j0++) {
                        for (i0 = 0; i0 < di; i0++) {
                                double g = buf[((l0 * dk + k0) * dj + j0) * di + i0] * fac;
                                if (g == 0) continue;
                                size_t a = ao_loc[ish] + i0, b = ao_loc[jsh] + j0;
                                size_t c = ao_loc[ksh] + k0, e = ao_loc[lsh] + l0;
                                if (with_j) {
                                        double g2 = 2 * g;
                                        double *v0 = jloc + (a * nao + b) * ndm;
                                        double *d0 = dms + (c * nao + e) * ndm;
                                        double *v1 = jloc + (c * nao + e) * ndm;
                                        double *d1 = dms + (a * nao + b) * ndm;
#pragma omp simd
                                        for (m = 0; m < ndm; m++) {
                                                v0[m] += g2 * d0[m];
                                                v1[m] += g2 * d1[m];
                                        }
                                }
                                if (with_k) {
                                        /* K_il += D_jk, K_jl += D_ik, K_ik += D_jl, K_jk += D_il */
                                        double *v0 = kloc + (a * nao + e) * ndm;
                                        double *v1 = kloc + (b * nao + e) * ndm;
                                        double *v2 = kloc + (a * nao + c) * ndm;
                                        double *v3 = kloc + (b * nao + c) * ndm;
                                        double *d0 = dms + (b * nao + c) * ndm;
                                        double *d1 = dms + (a * nao + c) * ndm;
                                        double *d2 = dms + (b * nao + e) * ndm;
                                        double *d3 = dms + (a * nao + e) * ndm;
#pragma omp simd
                                        for (m = 0; m < ndm; m++) {
                                                v0[m] += g * d0[m];
                                                v1[m] += g * d1[m];
                                                v2[m] += g * d2[m];
                                                v3[m] += g * d3[m];
                                        }
                                }
                        } } } }
                }
        }
#pragma omp critical
        {
                size_t n;
                if (with_j) for (n = 0; n < nv; n++) vj[n] += jloc[n];
                if (with_k) for (n = 0; n < nv; n++) vk[n] += kloc[n];
        }
        free(jloc);
        free(kloc);
        free(buf);
}
        /* V = W + W^T over the two AO indices */
        size_t p;
#pragma omp parallel for schedule(dynamic, 4)
        for (p = 0; p < nao; p++) {
                size_t q;
                int m;
                for (q = 0; q <= p; q++) {
                        if (with_j) {
                                double *x = vj + (p * nao + q) * ndm;
                                double *y = vj + (q * nao + p) * ndm;
                                for (m = 0; m < ndm; m++) {
                                        double s = x[m] + y[m];
                                        x[m] = s;
                                        y[m] = s;
                                }
                        }
                        if (with_k) {
                                double *x = vk + (p * nao + q) * ndm;
                                double *y = vk + (q * nao + p) * ndm;
                                for (m = 0; m < ndm; m++) {
                                        double s = x[m] + y[m];
                                        x[m] = s;
                                        y[m] = s;
                                }
                        }
                }
        }
}
