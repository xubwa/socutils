/*
 * Second nuclear derivatives of a two-electron energy at fixed densities,
 *
 *     E = sum_t c_t sum_{abcd} g(ab|cd) A_t[p_t] B_t[q_t],
 *
 * (p_t, q_t: index pairs of the four positions a, b, c, d), over the unique
 * shell quartets (i >= j, k >= l, ij >= kl) only.  For every quartet all the
 * second-derivative integrals are taken at once: by translational invariance
 * (sum of the derivatives of the four centres = 0) six blocks suffice,
 *
 *     mu mu  : int2e_ipip1  (i j|k l)     nu nu : int2e_ipip1  (j i|k l)
 *     la la  : int2e_ipip1  (k l|i j)     mu nu : int2e_ipvip1 (i j|k l)
 *     mu la  : int2e_ip1ip2 (i j|k l)     nu la : int2e_ip1ip2 (j i|k l),
 *
 * and they are contracted on the fly with the density weight symmetrized
 * over the eight permutations of the quartet, Gamma_sym(mu nu la si) =
 * sum_perm sum_t c_t A_t B_t, directly into the (natm, natm, 3, 3) Hessian
 * (nuclear derivative = -nabla, two of them: +).  Quartets are skipped when
 * the Schwarz-type estimate q[ij] q[kl] times the density bound is below tol.
 * Spherical AOs.
 */
#include <stdlib.h>
#include <string.h>
#include "cint.h"

#define MAXTERM 64
#define MAXGROUP 8

int int2e_ipip1_sph(double *out, int *dims, int *shls, int *atm, int natm,
                    int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
int int2e_ipvip1_sph(double *out, int *dims, int *shls, int *atm, int natm,
                     int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
int int2e_ip1ip2_sph(double *out, int *dims, int *shls, int *atm, int natm,
                     int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

/* the eight index permutations of g(ab|cd) = g(ba|cd) = g(ab|dc) = g(cd|ab) */
static const int PERMS[8][4] = {
        {0, 1, 2, 3}, {1, 0, 2, 3}, {0, 1, 3, 2}, {1, 0, 3, 2},
        {2, 3, 0, 1}, {3, 2, 0, 1}, {2, 3, 1, 0}, {3, 2, 1, 0}};

static double dmax3(double a, double b, double c)
{
        double m = a > b ? a : b;
        return m > c ? m : c;
}

void SOCVHFhess2e_contract(double *hess, int natm_out,
                           int nterm, double **mats_a, int *pair_a,
                           double **mats_b, int *pair_b, double *coef,
                           int ngroup, int *group,
                           double *qcond, double *dcond, double tol,
                           CINTOpt *opt_ipip, CINTOpt *opt_ipvip, CINTOpt *opt_ip1ip2,
                           int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        /* one Hessian per group of terms (e.g. J and K), from the same integrals */
        const size_t nh1 = (size_t)natm_out * natm_out * 9;
        const size_t nh = nh1 * ngroup;
        int dmax = 0;
        int i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        double csum = 0;
        for (i = 0; i < nterm; i++) {
                csum += coef[i] > 0 ? coef[i] : -coef[i];
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        int cache_size = GTOmax_cache_size(int2e_ipip1_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        int c2 = GTOmax_cache_size(int2e_ipvip1_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        int c3 = GTOmax_cache_size(int2e_ip1ip2_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        cache_size = cache_size > c2 ? cache_size : c2;
        cache_size = cache_size > c3 ? cache_size : c3;
        const size_t nbuf = (size_t)9 * dmax * dmax * dmax * dmax;
        int npair = nbas * (nbas + 1) / 2;

        memset(hess, 0, sizeof(double) * nh);
#pragma omp parallel
{
        double *hloc = calloc(nh, sizeof(double));
        double *buf = malloc(sizeof(double) * (nbuf * 6 + cache_size));
        double *cache = buf + nbuf * 6;
        double *b00 = buf, *b11 = buf + nbuf, *b22 = buf + nbuf * 2;
        double *b01 = buf + nbuf * 3, *b02 = buf + nbuf * 4, *b12 = buf + nbuf * 5;
        int ij, kl, ish, jsh, ksh, lsh, t, p, x, y, q;
        int shls[4];
        double S[MAXGROUP][6][9];
        double F[4][4][9];
        double gam[MAXGROUP];
        int g;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < npair; ij++) {
                ish = (int)((-1 + __builtin_sqrt(1. + 8. * ij)) / 2);
                while (ish * (ish + 1) / 2 > ij) ish--;
                while ((ish + 1) * (ish + 2) / 2 <= ij) ish++;
                jsh = ij - ish * (ish + 1) / 2;
                for (kl = 0; kl <= ij; kl++) {
                        ksh = (int)((-1 + __builtin_sqrt(1. + 8. * kl)) / 2);
                        while (ksh * (ksh + 1) / 2 > kl) ksh--;
                        while ((ksh + 1) * (ksh + 2) / 2 <= kl) ksh++;
                        lsh = kl - ksh * (ksh + 1) / 2;
                        /* screening */
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        double dbound = csum * dmax3(dcond[ish * nbas + jsh] * dcond[ksh * nbas + lsh],
                                                     dcond[ish * nbas + ksh] * dcond[jsh * nbas + lsh],
                                                     dcond[ish * nbas + lsh] * dcond[jsh * nbas + ksh]);
                        if (q4 * dbound < tol) {
                                continue;
                        }
                        int di = ao_loc[ish + 1] - ao_loc[ish];
                        int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                        int dl = ao_loc[lsh + 1] - ao_loc[lsh];
                        int nf = di * dj * dk * dl;
                        int ok = 0;
/* an all-zero quartet need not be written by libcint: zero the buffer then */
#define CALL(fn, b, opt) do { \
                        if (fn(b, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) ok = 1; \
                        else memset(b, 0, sizeof(double) * 9 * nf); } while (0)
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        CALL(int2e_ipip1_sph, b00, opt_ipip);
                        CALL(int2e_ipvip1_sph, b01, opt_ipvip);
                        CALL(int2e_ip1ip2_sph, b02, opt_ip1ip2);
                        shls[0] = jsh; shls[1] = ish;
                        CALL(int2e_ipip1_sph, b11, opt_ipip);
                        CALL(int2e_ip1ip2_sph, b12, opt_ip1ip2);
                        shls[0] = ksh; shls[1] = lsh; shls[2] = ish; shls[3] = jsh;
                        CALL(int2e_ipip1_sph, b22, opt_ipip);
#undef CALL
                        if (!ok) {
                                continue;
                        }
                        double fac = 1.;
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        memset(S, 0, sizeof(S));
                        int i0, j0, k0, l0;
                        for (l0 = 0; l0 < dl; l0++) {
                        for (k0 = 0; k0 < dk; k0++) {
                        for (j0 = 0; j0 < dj; j0++) {
                        for (i0 = 0; i0 < di; i0++) {
                                size_t ao[4] = {ao_loc[ish] + i0, ao_loc[jsh] + j0,
                                                ao_loc[ksh] + k0, ao_loc[lsh] + l0};
                                for (g = 0; g < ngroup; g++) gam[g] = 0;
                                for (p = 0; p < 8; p++) {
                                        size_t a4[4];
                                        for (q = 0; q < 4; q++) a4[q] = ao[PERMS[p][q]];
                                        for (t = 0; t < nterm; t++) {
                                                gam[group[t]] += coef[t]
                                                     * mats_a[t][a4[pair_a[2*t]] * nao + a4[pair_a[2*t+1]]]
                                                     * mats_b[t][a4[pair_b[2*t]] * nao + a4[pair_b[2*t+1]]];
                                        }
                                }
                                /* element offsets, buffers in libcint (Fortran) order */
                                size_t o00 = ((l0 * dk + k0) * dj + j0) * di + i0;          /* (i j|k l) */
                                size_t o11 = ((l0 * dk + k0) * di + i0) * dj + j0;          /* (j i|k l) */
                                size_t o22 = ((j0 * di + i0) * dl + l0) * dk + k0;          /* (k l|i j) */
                                for (g = 0; g < ngroup; g++) {
                                double gm = gam[g] * fac;
                                for (x = 0; x < 9; x++) {
                                        S[g][0][x] += b00[x * nf + o00] * gm;
                                        S[g][1][x] += b11[x * nf + o11] * gm;
                                        S[g][2][x] += b22[x * nf + o22] * gm;
                                        S[g][3][x] += b01[x * nf + o00] * gm;
                                        S[g][4][x] += b02[x * nf + o00] * gm;
                                        S[g][5][x] += b12[x * nf + o11] * gm;
                                } }
                        } } } }
                        int at[4] = {bas[ATOM_OF + ish * BAS_SLOTS], bas[ATOM_OF + jsh * BAS_SLOTS],
                                     bas[ATOM_OF + ksh * BAS_SLOTS], bas[ATOM_OF + lsh * BAS_SLOTS]};
                        for (g = 0; g < ngroup; g++) {
                        /* the 4 x 4 position blocks from translational invariance */
                        for (x = 0; x < 9; x++) {
                                int xt = (x % 3) * 3 + x / 3;      /* transpose */
                                F[0][0][x] = S[g][0][x];
                                F[1][1][x] = S[g][1][x];
                                F[2][2][x] = S[g][2][x];
                                F[0][1][x] = S[g][3][x];  F[1][0][xt] = S[g][3][x];
                                F[0][2][x] = S[g][4][x];  F[2][0][xt] = S[g][4][x];
                                F[1][2][x] = S[g][5][x];  F[2][1][xt] = S[g][5][x];
                        }
                        for (p = 0; p < 3; p++) {
                                for (x = 0; x < 9; x++) {
                                        F[p][3][x] = -(F[p][0][x] + F[p][1][x] + F[p][2][x]);
                                }
                        }
                        for (y = 0; y < 4; y++) {
                                for (x = 0; x < 9; x++) {
                                        F[3][y][x] = -(F[0][y][x] + F[1][y][x] + F[2][y][x]);
                                }
                        }
                        for (p = 0; p < 4; p++) {
                        for (y = 0; y < 4; y++) {
                                double *h = hloc + g * nh1 + ((size_t)at[p] * natm_out + at[y]) * 9;
                                for (x = 0; x < 9; x++) {
                                        h[x] += F[p][y][x];
                                }
                        } }
                        }
                }
        }
#pragma omp critical
        for (size_t n = 0; n < nh; n++) {
                hess[n] += hloc[n];
        }
        free(hloc);
        free(buf);
}
}
