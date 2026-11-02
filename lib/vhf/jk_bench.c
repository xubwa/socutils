/*
 * Cost of the integrals of one angular class in the J / K builds: the unique
 * shell quartets (i >= j, k >= l, ij >= kl) of jk_multi_sym with q_ij q_kl >=
 * tol whose (l_i l_j | l_k l_l) matches cls (-1: any), evaluated by one engine
 * with nothing else done (no digestion, no density screening; the kernels get
 * dmax = 1), the OpenMP loop as jk_multi_sym's.
 *
 *   engine 0: libcint int2e_sph;  1: jk_ssss.c (s quartets only)
 *
 * stats[0] quartets, stats[1] primitive quartets (product of the nprim),
 * stats[2] contracted functions quartets, stats[3] wall seconds of the loop,
 * stats[4] wall seconds of the engine setup; returns the sum of all integrals
 * (a checksum, equal across engines).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "cint.h"

int int2e_sph(double *out, int *dims, int *shls, int *atm, int natm,
              int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env);
typedef struct SSCtx SSCtx;
SSCtx *SOCVHFss_ctx_build(int *atm, int natm, int *bas, int nbas, double *env);
void SOCVHFss_ctx_free(SSCtx *cx);
size_t SOCVHFss_wksize(int *bas, int nbas);
int SOCVHFss_int2e(double *out, const int *shls, const SSCtx *cx, double dmax, double prim_tol, double *wk);

static inline void shell_pair(int ij, int *i, int *j)
{
        int a = (int)((sqrt(1. + 8. * ij) - 1) / 2);
        while (a * (a + 1) / 2 > ij) a--;
        while ((a + 1) * (a + 2) / 2 <= ij) a++;
        *i = a;
        *j = ij - a * (a + 1) / 2;
}

double SOCVHFjk_bench_class(int engine, int *cls, double *qcond, double tol, CINTOpt *opt,
                            int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                            double *stats)
{
        int dmax = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        const size_t cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)dmax * dmax * dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        double t0 = omp_get_wtime();
        SSCtx *scx = engine == 1 ? SOCVHFss_ctx_build(atm, natm, bas, nbas, env) : NULL;
        const size_t sswk = engine == 1 ? SOCVHFss_wksize(bas, nbas) : 0;
        stats[4] = omp_get_wtime() - t0;
        double nq = 0, npq = 0, nfq = 0, sum = 0;
        t0 = omp_get_wtime();
#pragma omp parallel reduction(+:nq, npq, nfq, sum)
{
        double *buf = malloc(sizeof(double) * (nbuf + cache_size + sswk));
        double *cache = buf + nbuf, *swk = cache + cache_size;
        int it, ij, kl, ish, jsh, ksh, lsh, n;
        int shls[4];
#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < npair; it++) {
                ij = npair - 1 - it;
                shell_pair(ij, &ish, &jsh);
                if ((cls[0] >= 0 && bas[ish * BAS_SLOTS + ANG_OF] != cls[0])
                    || (cls[1] >= 0 && bas[jsh * BAS_SLOTS + ANG_OF] != cls[1])) continue;
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        if ((cls[2] >= 0 && bas[ksh * BAS_SLOTS + ANG_OF] != cls[2])
                            || (cls[3] >= 0 && bas[lsh * BAS_SLOTS + ANG_OF] != cls[3])) continue;
                        if (qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh] < tol) continue;
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        const int nf = (ao_loc[ish + 1] - ao_loc[ish]) * (ao_loc[jsh + 1] - ao_loc[jsh])
                                     * (ao_loc[ksh + 1] - ao_loc[ksh]) * (ao_loc[lsh + 1] - ao_loc[lsh]);
                        int ok;
                        if (engine == 1) {
                                ok = SOCVHFss_int2e(buf, shls, scx, 1., tol * 1e-2, swk);
                        } else {
                                ok = int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache);
                        }
                        nq += 1;
                        npq += (double)bas[ish * BAS_SLOTS + NPRIM_OF] * bas[jsh * BAS_SLOTS + NPRIM_OF]
                             * bas[ksh * BAS_SLOTS + NPRIM_OF] * bas[lsh * BAS_SLOTS + NPRIM_OF];
                        nfq += nf;
                        if (ok) for (n = 0; n < nf; n++) sum += buf[n];
                }
        }
        free(buf);
}
        stats[0] = nq;
        stats[1] = npq;
        stats[2] = nfq;
        stats[3] = omp_get_wtime() - t0;
        SOCVHFss_ctx_free(scx);
        return sum;
}
