/*
 * Cost of the integrals of one angular class in the J / K builds: the unique
 * shell quartets (i >= j, k >= l, ij >= kl) of jk_multi_sym with q_ij q_kl >=
 * tol whose (l_i l_j | l_k l_l) matches cls (-1: any), evaluated by one engine
 * with nothing else done (no digestion, no density screening; the kernels get
 * dmax = 1), the OpenMP loop as jk_multi_sym's.
 *
 *   engine 0: libcint int2e_sph;  1: jk_ssss.c (s / p quartets with at most two p shells);
 *   2: jk_gen.c (l <= 2);  3: jk_gen.c's lanes (GL_NL quartets of one ket type per call, per row)
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
typedef struct GenCtx GenCtx;
GenCtx *SOCVHFgen_ctx_build(int *atm, int natm, int *bas, int nbas, double *env);
void SOCVHFgen_ctx_free(GenCtx *cx);
size_t SOCVHFgen_wksize(int *bas, int nbas);
int SOCVHFgen_int2e(double *out, const int *shls, const GenCtx *cx, double dmax, double prim_tol,
                    double *wk, const int *bas);
int SOCVHFgen_shell_types(int *type, const int *bas, int nbas);
int SOCVHFgen_lane_ok(int ksh, int lsh, const int *bas);
size_t SOCVHFgen_lane_wksize(const int *bas, int nbas, int nbuf);
int SOCVHFgen_int2e_lanes(double *out, size_t ostride, int ish, int jsh, const int *ksh, const int *lsh, int nl,
                          const GenCtx *cx, double dmax, double prim_tol, double *wk,
                          const int *atm, const int *bas, const double *env);
#define BL_NL 8                 /* = GL_NL of jk_gen.h */

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
        GenCtx *gcx = engine >= 2 ? SOCVHFgen_ctx_build(atm, natm, bas, nbas, env) : NULL;
        size_t sswk = engine == 1 ? SOCVHFss_wksize(bas, nbas) : (engine >= 2 ? SOCVHFgen_wksize(bas, nbas) : 0);
        if (engine == 3 && SOCVHFgen_lane_wksize(bas, nbas, nbuf) > sswk) sswk = SOCVHFgen_lane_wksize(bas, nbas, nbuf);
        int *stype = malloc(sizeof(int) * nbas);
        const int ntype = SOCVHFgen_shell_types(stype, bas, nbas);
        stats[4] = omp_get_wtime() - t0;
        double nq = 0, npq = 0, nfq = 0, sum = 0;
        t0 = omp_get_wtime();
#pragma omp parallel reduction(+:nq, npq, nfq, sum)
{
        double *buf = malloc(sizeof(double) * (nbuf * BL_NL + cache_size + sswk));
        double *cache = buf + nbuf * BL_NL, *swk = cache + cache_size;
        const int nbk = engine == 3 ? ntype * ntype : 1;
        int *bkk = malloc(sizeof(int) * nbk * BL_NL), *bkl = malloc(sizeof(int) * nbk * BL_NL);
        int *bkn = calloc(nbk, sizeof(int));
        int it, ij, kl, ish, jsh, ksh, lsh, n, kb;
        int shls[4];
#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < npair; it++) {
                ij = npair - 1 - it;
                shell_pair(ij, &ish, &jsh);
                if ((cls[0] >= 0 && bas[ish * BAS_SLOTS + ANG_OF] != cls[0])
                    || (cls[1] >= 0 && bas[jsh * BAS_SLOTS + ANG_OF] != cls[1])) continue;
                const int row_lanes = engine == 3 && bas[ish * BAS_SLOTS + ANG_OF] <= 2
                                    && bas[jsh * BAS_SLOTS + ANG_OF] <= 2 && bas[ish * BAS_SLOTS + NCTR_OF] <= 4
                                    && bas[jsh * BAS_SLOTS + NCTR_OF] <= 4;
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        if ((cls[2] >= 0 && bas[ksh * BAS_SLOTS + ANG_OF] != cls[2])
                            || (cls[3] >= 0 && bas[lsh * BAS_SLOTS + ANG_OF] != cls[3])) continue;
                        if (qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh] < tol) continue;
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        const int nf = (ao_loc[ish + 1] - ao_loc[ish]) * (ao_loc[jsh + 1] - ao_loc[jsh])
                                     * (ao_loc[ksh + 1] - ao_loc[ksh]) * (ao_loc[lsh + 1] - ao_loc[lsh]);
                        int ok;
                        if (row_lanes && SOCVHFgen_lane_ok(ksh, lsh, bas)) {
                                nq += 1;
                                npq += (double)bas[ish * BAS_SLOTS + NPRIM_OF] * bas[jsh * BAS_SLOTS + NPRIM_OF]
                                     * bas[ksh * BAS_SLOTS + NPRIM_OF] * bas[lsh * BAS_SLOTS + NPRIM_OF];
                                nfq += nf;
                                kb = stype[ksh] * ntype + stype[lsh];
                                bkk[kb * BL_NL + bkn[kb]] = ksh;
                                bkl[kb * BL_NL + bkn[kb]] = lsh;
                                if (++bkn[kb] == BL_NL) {
                                        if (SOCVHFgen_int2e_lanes(buf, nbuf, ish, jsh, bkk + kb * BL_NL,
                                                                  bkl + kb * BL_NL, BL_NL, gcx, 1., tol * 1e-2, swk,
                                                                  atm, bas, env)) {
                                                int l_;
                                                for (l_ = 0; l_ < BL_NL; l_++)
                                                        for (n = 0; n < nf; n++) sum += buf[l_ * nbuf + n];
                                        }
                                        bkn[kb] = 0;
                                }
                                continue;
                        }
                        if (engine == 1) {
                                ok = SOCVHFss_int2e(buf, shls, scx, 1., tol * 1e-2, swk);
                        } else if (engine == 2) {
                                ok = SOCVHFgen_int2e(buf, shls, gcx, 1., tol * 1e-2, swk, bas);
                        } else {
                                ok = int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache);
                        }
                        nq += 1;
                        npq += (double)bas[ish * BAS_SLOTS + NPRIM_OF] * bas[jsh * BAS_SLOTS + NPRIM_OF]
                             * bas[ksh * BAS_SLOTS + NPRIM_OF] * bas[lsh * BAS_SLOTS + NPRIM_OF];
                        nfq += nf;
                        if (ok) for (n = 0; n < nf; n++) sum += buf[n];
                }
                for (kb = 0; row_lanes && kb < nbk; kb++) {
                        if (bkn[kb] == 0) continue;
                        const int k0 = bkk[kb * BL_NL], l0 = bkl[kb * BL_NL];
                        const int nf = (ao_loc[ish + 1] - ao_loc[ish]) * (ao_loc[jsh + 1] - ao_loc[jsh])
                                     * (ao_loc[k0 + 1] - ao_loc[k0]) * (ao_loc[l0 + 1] - ao_loc[l0]);
                        if (SOCVHFgen_int2e_lanes(buf, nbuf, ish, jsh, bkk + kb * BL_NL, bkl + kb * BL_NL,
                                                  bkn[kb], gcx, 1., tol * 1e-2, swk, atm, bas, env)) {
                                int l_;
                                for (l_ = 0; l_ < bkn[kb]; l_++)
                                        for (n = 0; n < nf; n++) sum += buf[l_ * nbuf + n];
                        }
                        bkn[kb] = 0;
                }
        }
        free(buf);
        free(bkk);
        free(bkl);
        free(bkn);
}
        free(stype);
        stats[0] = nq;
        stats[1] = npq;
        stats[2] = nfq;
        stats[3] = omp_get_wtime() - t0;
        SOCVHFss_ctx_free(scx);
        SOCVHFgen_ctx_free(gcx);
        return sum;
}

static inline double dmax2(double a, double b) { return a > b ? a : b; }

/*
 * How much of jk_multi_sym's work its screening could have skipped: the
 * quartets passing its J / K tests (Schwarz q_ij q_kl times the block maxima
 * dcond of the density), evaluated by libcint, and per total l (L < 9):
 *   out[L * 6 + 0] quartets, 1 their function quartets,
 *   2 / 3 those with max |(ab|cd)| times the same block bound below tol (Schwarz loose),
 *   4 / 5 those with max over the elements of |(ab|cd)| times max(|D_ab|, |D_cd|,
 *         |D_ac|, |D_ad|, |D_bc|, |D_bd|) below tol (Schwarz and the block maxima loose).
 * dm (nao, nao) symmetric.
 */
void SOCVHFjk_screen_waste(double *out, double *dm, double *qcond, double *dcond, double tol, CINTOpt *opt,
                           int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        int dmax = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        const size_t cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)dmax * dmax * dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        memset(out, 0, sizeof(double) * 9 * 6);
#pragma omp parallel
{
        double loc[9 * 6];
        memset(loc, 0, sizeof(loc));
        double *buf = malloc(sizeof(double) * (nbuf + cache_size));
        double *cache = buf + nbuf;
        int it, ij, kl, ish, jsh, ksh, lsh;
        int shls[4];
#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < npair; it++) {
                ij = npair - 1 - it;
                shell_pair(ij, &ish, &jsh);
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        const double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        const double dj = dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]);
                        const double dk = dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                                dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh]));
                        if (q4 * dj < tol && q4 * dk < tol) continue;
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        const int L = bas[ish * BAS_SLOTS + ANG_OF] + bas[jsh * BAS_SLOTS + ANG_OF]
                                    + bas[ksh * BAS_SLOTS + ANG_OF] + bas[lsh * BAS_SLOTS + ANG_OF];
                        const int di = ao_loc[ish + 1] - ao_loc[ish], dj_ = ao_loc[jsh + 1] - ao_loc[jsh];
                        const int dk_ = ao_loc[ksh + 1] - ao_loc[ksh], dl = ao_loc[lsh + 1] - ao_loc[lsh];
                        const double nf = (double)di * dj_ * dk_ * dl;
                        double gmax = 0, emax = 0;
                        if (int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                                int a, b, c, d;
                                for (d = 0; d < dl; d++)
                                for (c = 0; c < dk_; c++)
                                for (b = 0; b < dj_; b++)
                                for (a = 0; a < di; a++) {
                                        const double g = fabs(buf[((d * dk_ + c) * dj_ + b) * di + a]);
                                        if (g > gmax) gmax = g;
                                        const size_t A = ao_loc[ish] + a, B = ao_loc[jsh] + b;
                                        const size_t C = ao_loc[ksh] + c, D = ao_loc[lsh] + d;
                                        double dd = dmax2(dmax2(fabs(dm[A * nao + B]), fabs(dm[C * nao + D])),
                                                          dmax2(dmax2(fabs(dm[A * nao + C]), fabs(dm[A * nao + D])),
                                                                dmax2(fabs(dm[B * nao + C]), fabs(dm[B * nao + D]))));
                                        if (g * dd > emax) emax = g * dd;
                                }
                        }
                        double *o = loc + (L < 9 ? L : 8) * 6;
                        o[0] += 1;
                        o[1] += nf;
                        if (gmax * dmax2(dj, dk) < tol) { o[2] += 1; o[3] += nf; }
                        if (emax < tol) { o[4] += 1; o[5] += nf; }
                }
        }
        free(buf);
#pragma omp critical
        for (i = 0; i < 9 * 6; i++) out[i] += loc[i];
}
}
