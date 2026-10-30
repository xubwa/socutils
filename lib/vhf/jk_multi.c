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
#include <omp.h>
#include "cint.h"

int int2e_sph(double *out, int *dims, int *shls, int *atm, int natm,
              int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
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
        size_t cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
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


/* out[0:n] = sum_t bufs[t][0:n], each thread of the enclosing parallel region
 * summing its own slice (instead of the buffers one after another) */
static void reduce_thread_buffers(double *out, double **bufs, int nbuf, size_t n)
{
        const int t = omp_get_thread_num(), nt = omp_get_num_threads();
        const size_t chunk = (n + nt - 1) / nt;
        const size_t s0 = chunk * t < n ? chunk * t : n;
        const size_t s1 = s0 + chunk < n ? s0 + chunk : n;
        size_t i;
        int k;
        for (i = s0; i < s1; i++) out[i] = 0;
        for (k = 0; k < nbuf; k++) {
                const double *b = bufs[k];
                for (i = s0; i < s1; i++) out[i] += b[i];
        }
}

/*
 * The same for symmetric densities (D_cd = D_dc; CPHF): J and K are then
 * symmetric, the eight permutations collapse to two J and four K updates per
 * integral, W += ..., and J = W + W^T at the end.  J and K are screened
 * separately (J by the densities on (ij), (kl), K by those on (ik), (il),
 * (jk), (jl)).  The K rows of the bra shells i, j (columns < ao_loc[i+1],
 * since l <= k <= i) and J_ij are accumulated in local buffers over the ket
 * loop and added to the thread's matrices once per bra pair; the thread
 * matrices are reduced in parallel.
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
        size_t cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)dmax * dmax * dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        const int nthmax = omp_get_max_threads();
        double **jbufs = calloc(nthmax, sizeof(double *));
        double **kbufs = calloc(nthmax, sizeof(double *));
        int nth_used = 1;
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
#pragma omp single
        nth_used = omp_get_num_threads();
        double *jloc = with_j ? calloc(nv, sizeof(double)) : NULL;
        double *kloc = with_k ? calloc(nv, sizeof(double)) : NULL;
        jbufs[tid] = jloc;
        kbufs[tid] = kloc;
        double *buf = malloc(sizeof(double) * (nbuf + cache_size));
        double *cache = buf + nbuf;
        double *kri = with_k ? malloc(sizeof(double) * dmax * nao * ndm) : NULL;
        double *krj = with_k ? malloc(sizeof(double) * dmax * nao * ndm) : NULL;
        double *jij = with_j ? malloc(sizeof(double) * dmax * dmax * ndm) : NULL;
        int it, ij, kl, ish, jsh, ksh, lsh, m;
        int shls[4];
        /* the work of pair ij grows with ij (kl <= ij): largest first */
#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < npair; it++) {
                ij = npair - 1 - it;
                shell_pair(ij, &ish, &jsh);
                const int di = ao_loc[ish + 1] - ao_loc[ish];
                const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
                const size_t ncol = ao_loc[ish + 1];
                double *krb = ish == jsh ? kri : krj;     /* rows of shell j */
                int touched = 0;
                if (with_k) {
                        memset(kri, 0, sizeof(double) * di * ncol * ndm);
                        if (ish != jsh) memset(krj, 0, sizeof(double) * dj * ncol * ndm);
                }
                if (with_j) memset(jij, 0, sizeof(double) * di * dj * ndm);
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        int do_j = with_j && q4 * dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]) >= tol;
                        int do_k = with_k && q4 * dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                                        dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh])) >= tol;
                        if (!do_j && !do_k) {
                                continue;
                        }
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        if (!int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                                continue;
                        }
                        touched = 1;
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
                                size_t a = a0 + i0, b = b0 + j0;
                                size_t c = ao_loc[ksh] + k0, e = ao_loc[lsh] + l0;
                                if (do_j) {
                                        double g2 = 2 * g;
                                        double *v0 = jij + ((size_t)i0 * dj + j0) * ndm;
                                        double *d0 = dms + (c * nao + e) * ndm;
                                        double *v1 = jloc + (c * nao + e) * ndm;
                                        double *d1 = dms + (a * nao + b) * ndm;
#pragma omp simd
                                        for (m = 0; m < ndm; m++) {
                                                v0[m] += g2 * d0[m];
                                                v1[m] += g2 * d1[m];
                                        }
                                }
                                if (do_k) {
                                        /* K_ae += D_bc, K_be += D_ac, K_ac += D_be, K_bc += D_ae */
                                        double *v0 = kri + ((size_t)i0 * ncol + e) * ndm;
                                        double *v1 = krb + ((size_t)j0 * ncol + e) * ndm;
                                        double *v2 = kri + ((size_t)i0 * ncol + c) * ndm;
                                        double *v3 = krb + ((size_t)j0 * ncol + c) * ndm;
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
                if (!touched) {
                        continue;
                }
                /* flush the bra-pair buffers */
                int i0, j0;
                size_t q;
                if (with_j) {
                        for (i0 = 0; i0 < di; i0++)
                                for (j0 = 0; j0 < dj; j0++) {
                                        double *o = jloc + ((a0 + i0) * nao + b0 + j0) * ndm;
                                        double *src = jij + ((size_t)i0 * dj + j0) * ndm;
                                        for (m = 0; m < ndm; m++) o[m] += src[m];
                                }
                }
                if (with_k) {
                        for (i0 = 0; i0 < di; i0++) {
                                double *o = kloc + (a0 + i0) * nao * ndm;
                                double *src = kri + (size_t)i0 * ncol * ndm;
                                for (q = 0; q < ncol * ndm; q++) o[q] += src[q];
                        }
                        if (ish != jsh) {
                                for (j0 = 0; j0 < dj; j0++) {
                                        double *o = kloc + (b0 + j0) * nao * ndm;
                                        double *src = krj + (size_t)j0 * ncol * ndm;
                                        for (q = 0; q < ncol * ndm; q++) o[q] += src[q];
                                }
                        }
                }
        }
#pragma omp barrier
        if (with_j) reduce_thread_buffers(vj, jbufs, nth_used, nv);
        if (with_k) reduce_thread_buffers(vk, kbufs, nth_used, nv);
#pragma omp barrier
        free(jloc);
        free(kloc);
        free(buf);
        free(kri);
        free(krj);
        free(jij);
}
        free(jbufs);
        free(kbufs);
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
