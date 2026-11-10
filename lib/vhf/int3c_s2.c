/*
 * Three-centre Coulomb integrals (ij|P) of a list of significant AO shell
 * pairs (ish >= jsh), for socutils.scf.rik.cholesky_eri: the lower-triangle
 * AO pairs of every listed shell pair, all auxiliary shells, written as rows
 *
 *     out[row(i, j), P],   row = pair_row[k] + the pair's lower-triangle
 *                           functions (i >= j) in pack_tril order
 *
 * of an (nrow, naux) C-ordered array (the screened analogue of pyscf's
 * aux_e2(..., aosym='s2ij'), which loops over every pair).  pair_row[k]: the
 * first row of shell pair k; for ish > jsh its di dj functions are rows
 * pair_row[k] + a dj + b (a over i, b over j), for ish == jsh the di (di+1)/2
 * functions a >= b, row pair_row[k] + a (a+1) / 2 + b.  The caller scatters
 * the rows to the pack_tril columns.  Combined basis (pyscf gto.conc_env of
 * mol, auxmol): AO shells [0, nbas), auxiliary shells after.  Spherical AOs.
 */
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cint.h"

int int3c2e_sph(double *out, int *dims, int *shls, int *atm, int natm,
                int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env);

void SOCVHFint3c2e_s2_pairs(double *out, int *pairs, long *pair_row, int npairs,
                            CINTOpt *opt, int *ao_loc, int *atm, int natm_tot,
                            int *bas, int nbas, int nauxbas, double *env)
{
        const int nbtot = nbas + nauxbas;
        const size_t nao = ao_loc[nbas];
        const size_t naux = ao_loc[nbtot] - nao;
        int dmax = 0, daux = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                if (d > dmax) dmax = d;
        }
        for (i = nbas; i < nbtot; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                if (d > daux) daux = d;
        }
        int shls_slice[6] = {0, nbas, 0, nbas, nbas, nbtot};
        size_t cache_size = GTOmax_cache_size(int3c2e_sph, shls_slice, 3,
                                              atm, natm_tot, bas, nbtot, env);
#pragma omp parallel
{
        double *buf = malloc(sizeof(double) * (size_t)dmax * dmax * daux);
        double *cache = malloc(sizeof(double) * cache_size);
        int k;
#pragma omp for schedule(dynamic, 1)
        for (k = 0; k < npairs; k++) {
                const int ish = pairs[2 * k], jsh = pairs[2 * k + 1];
                const int di = ao_loc[ish + 1] - ao_loc[ish];
                const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                const int dij = di * dj;
                double *o = out + pair_row[k] * naux;
                int ksh;
                for (ksh = nbas; ksh < nbtot; ksh++) {
                        const int p0 = ao_loc[ksh] - nao, dp = ao_loc[ksh + 1] - ao_loc[ksh];
                        int shls[3] = {ish, jsh, ksh};
                        int a, b, p;
                        if (!int3c2e_sph(buf, NULL, shls, atm, natm_tot, bas, nbtot, env, opt, cache)) {
                                if (ish != jsh) {
                                        for (a = 0; a < dij; a++)
                                                memset(o + a * naux + p0, 0, sizeof(double) * dp);
                                } else {
                                        for (a = 0; a < di * (di + 1) / 2; a++)
                                                memset(o + a * naux + p0, 0, sizeof(double) * dp);
                                }
                                continue;
                        }
                        /* buf[p, b, a] (i fastest) */
                        if (ish != jsh) {
                                for (a = 0; a < di; a++)
                                for (b = 0; b < dj; b++) {
                                        double *r = o + (size_t)(a * dj + b) * naux + p0;
                                        for (p = 0; p < dp; p++)
                                                r[p] = buf[(size_t)p * dij + b * di + a];
                                }
                        } else {
                                for (a = 0; a < di; a++)
                                for (b = 0; b <= a; b++) {
                                        double *r = o + (size_t)(a * (a + 1) / 2 + b) * naux + p0;
                                        for (p = 0; p < dp; p++)
                                                r[p] = buf[(size_t)p * dij + b * di + a];
                                }
                        }
                }
        }
        free(buf);
        free(cache);
}
}
