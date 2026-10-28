/*
 * Screened assembly of one-electron operators that are sums of atom-centred terms,
 *     out[c][comp][mu,nu] += sum_K ( mu nu | k_K )_{c,comp} ,
 * with libcint real (spherical) integrals, without storing the three-index tensor.
 *
 *  OCJ_3c_drv   : k_K a generally contracted (nctr columns) s shell on atom K;
 *                 int3c2e_sph, int3c2e_ipvip1_sph            (erf/r of unit-charge Gaussians)
 *                 int3c1e[_r2m_origk]_sph, int3c1e_ipvip1[_r2m_origk]_sph   (local Gaussians)
 *  OCJ_rinv_drv : q_K / |r - R_K|:  int1e_rinv_sph, int1e_iprinvip_sph
 *
 * Screening per shell pair (i, j) and atom K: the K term decays as exp(-z_K |r - R_K|^2), so
 *     a_i z/(a_i + z) |A_i - R_K|^2 + a_j z/(a_j + z) |A_j - R_K|^2 > cut   ->  skipped
 * (a: smallest exponent of the shell).  OpenMP over shell pairs i >= j; the (j, i) block is
 * written as the transpose (component tmap[comp] for the derivative integrals, where
 * <d_a i|V|d_b j> = <d_b j|V|d_a i>), so every block has a single writer.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ATM_SLOTS       6
#define BAS_SLOTS       8
#define PTR_RINV_ORIG   4

typedef long CACHE_SIZE_T;
typedef CACHE_SIZE_T (*Intor)(double *out, int *dims, int *shls, int *atm, int natm,
                              int *bas, int nbas, double *env, void *opt, double *cache);

static double pair_exponent(double ai, const double *ri, double aj, const double *rj,
                            double z, const double *rk)
{
        double di = 0, dj = 0, t;
        int x;
        for (x = 0; x < 3; x++) {
                t = ri[x] - rk[x]; di += t * t;
                t = rj[x] - rk[x]; dj += t * t;
        }
        return ai * z / (ai + z) * di + aj * z / (aj + z) * dj;
}

/* buf: libcint layout [i + di*(j + dj*(c + nctr*comp))];  out: [comp][c][n][n] */
static void add_block(double *out, int n, int i0, int j0, int di, int dj, double *buf,
                      int nctr, int comp, int *tmap, double scale, int sym)
{
        int c, x, ii, jj;
        size_t nn = (size_t)n * n;
        for (x = 0; x < comp; x++) {
        for (c = 0; c < nctr; c++) {
                double *o = out + nn * ((size_t)x * nctr + c);
                double *b = buf + (size_t)di * dj * (c + nctr * x);
                for (jj = 0; jj < dj; jj++) {
                for (ii = 0; ii < di; ii++) {
                        o[(size_t)(i0 + ii) * n + j0 + jj] += scale * b[ii + di * jj];
                } }
                if (sym) {
                        double *ot = out + nn * ((size_t)tmap[x] * nctr + c);
                        for (jj = 0; jj < dj; jj++) {
                        for (ii = 0; ii < di; ii++) {
                                ot[(size_t)(j0 + jj) * n + i0 + ii] += scale * b[ii + di * jj];
                        } }
                }
        } }
}

static int pair_index_i(int ij)
{
        int ish = (int)((sqrt(8. * ij + 1) - 1) / 2);
        while (ish * (ish + 1) / 2 > ij) ish--;
        while ((ish + 1) * (ish + 2) / 2 <= ij) ish++;
        return ish;
}

static int max_shell_dim(int *ao_loc, int nbas)
{
        int i, d, m = 0;
        for (i = 0; i < nbas; i++) {
                d = ao_loc[i + 1] - ao_loc[i];
                if (d > m) m = d;
        }
        return m;
}

void OCJ_3c_drv(Intor intor, double *out, int nctr, int comp, int *tmap, int nao, int *ao_loc,
                int nbas_ao, double *amin, int natmk, int *ksh, double *kcoord, double *kzeta,
                double cut, int *atm, int natm, int *bas, int nbas, double *env, long *counter)
{
        int npair = nbas_ao * (nbas_ao + 1) / 2;
        int maxd = max_shell_dim(ao_loc, nbas_ao);
        long nkept = 0;
#pragma omp parallel reduction(+:nkept)
{
        int ij, ish, jsh, K, di, dj, shls[3];
        double *cache = NULL;
        CACHE_SIZE_T ncache = 0, need;
        double *buf = malloc(sizeof(double) * maxd * maxd * nctr * comp);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < npair; ij++) {
                ish = pair_index_i(ij);
                jsh = ij - ish * (ish + 1) / 2;
                di = ao_loc[ish + 1] - ao_loc[ish];
                dj = ao_loc[jsh + 1] - ao_loc[jsh];
                double *ri = env + atm[bas[ish * BAS_SLOTS] * ATM_SLOTS + 1];
                double *rj = env + atm[bas[jsh * BAS_SLOTS] * ATM_SLOTS + 1];
                for (K = 0; K < natmk; K++) {
                        if (pair_exponent(amin[ish], ri, amin[jsh], rj, kzeta[K], kcoord + 3 * K) > cut) {
                                continue;
                        }
                        nkept++;
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh[K];
                        need = (*intor)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
                        if (need > ncache) {
                                free(cache);
                                ncache = need;
                                cache = malloc(sizeof(double) * ncache);
                        }
                        (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, NULL, cache);
                        add_block(out, nao, ao_loc[ish], ao_loc[jsh], di, dj, buf, nctr, comp, tmap,
                                  1., ish != jsh);
                }
        }
        free(buf);
        free(cache);
}
        *counter = nkept;
}

void OCJ_rinv_drv(Intor intor, double *out, int comp, int *tmap, int nao, int *ao_loc,
                  int nbas_ao, double *amin, int natmk, double *kcoord, double *kzeta,
                  double *kcharge, double cut, int *atm, int natm, int *bas, int nbas,
                  double *env, int nenv)
{
        int npair = nbas_ao * (nbas_ao + 1) / 2;
        int maxd = max_shell_dim(ao_loc, nbas_ao);
#pragma omp parallel
{
        int ij, ish, jsh, K, di, dj, shls[2];
        double *cache = NULL;
        CACHE_SIZE_T ncache = 0, need;
        double *envt = malloc(sizeof(double) * nenv);
        memcpy(envt, env, sizeof(double) * nenv);
        double *buf = malloc(sizeof(double) * maxd * maxd * comp);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < npair; ij++) {
                ish = pair_index_i(ij);
                jsh = ij - ish * (ish + 1) / 2;
                di = ao_loc[ish + 1] - ao_loc[ish];
                dj = ao_loc[jsh + 1] - ao_loc[jsh];
                double *ri = env + atm[bas[ish * BAS_SLOTS] * ATM_SLOTS + 1];
                double *rj = env + atm[bas[jsh * BAS_SLOTS] * ATM_SLOTS + 1];
                for (K = 0; K < natmk; K++) {
                        if (pair_exponent(amin[ish], ri, amin[jsh], rj, kzeta[K], kcoord + 3 * K) > cut) {
                                continue;
                        }
                        memcpy(envt + PTR_RINV_ORIG, kcoord + 3 * K, sizeof(double) * 3);
                        shls[0] = ish; shls[1] = jsh;
                        need = (*intor)(NULL, NULL, shls, atm, natm, bas, nbas, envt, NULL, NULL);
                        if (need > ncache) {
                                free(cache);
                                ncache = need;
                                cache = malloc(sizeof(double) * ncache);
                        }
                        (*intor)(buf, NULL, shls, atm, natm, bas, nbas, envt, NULL, cache);
                        add_block(out, nao, ao_loc[ish], ao_loc[jsh], di, dj, buf, 1, comp, tmap,
                                  kcharge[K], ish != jsh);
                }
        }
        free(buf);
        free(cache);
        free(envt);
}
}
