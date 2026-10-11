/* Copyright 2026 The socutils developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * F12 two-electron integral classes for socutils.rf12 that libcint (built
 * with WITH_F12) does not export, assembled exactly the way libcint builds
 * its own F12 classes (src/cint2e_f12.c): an existing, kernel-independent
 * gout routine of libcint combined with the Slater (stg) or Yukawa (yp)
 * EnvVars initializer.  Requires libcint >= 6 compiled with -DWITH_F12=1.
 *
 *   int2e_{stg,yp}_ipvip1ipvip2_sph  (nabla i nabla j | K | nabla k nabla l), 81 comps
 *   int2e_{stg,yp}_ipip1ipip2_sph    (nabla nabla i j | K | nabla nabla k l), 81 comps
 *
 * plus RF12_contract2e: a shell-quartet driver that contracts the bra
 * functions of both electrons (i and k) with coefficient arrays on the fly.
 */
#include <stdlib.h>
#include <string.h>
#include "cint.h"




void CINTinit_int2e_stg_EnvVars(CINTEnvVars *envs, FINT *ng, FINT *shls,
                                FINT *atm, FINT natm, FINT *bas, FINT nbas, double *env);
void CINTinit_int2e_yp_EnvVars(CINTEnvVars *envs, FINT *ng, FINT *shls,
                               FINT *atm, FINT natm, FINT *bas, FINT nbas, double *env);
void CINTall_2e_stg_optimizer(CINTOpt **opt, FINT *ng,
                              FINT *atm, FINT natm, FINT *bas, FINT nbas, double *env);
CACHE_SIZE_T CINT2e_drv(double *out, FINT *dims, CINTEnvVars *envs, CINTOpt *opt,
                        double *cache, void (*f_c2s)());
void c2s_sph_2e1(double *opij, double *gctr, FINT *dims, CINTEnvVars *envs, double *cache);
void CINTgout2e_int2e_ipvip1ipvip2(double *gout, double *g, FINT *idx, CINTEnvVars *envs, FINT gout_empty);
void CINTgout2e_int2e_ipip1ipip2(double *gout, double *g, FINT *idx, CINTEnvVars *envs, FINT gout_empty);

#define F12CLASS(KER, NAME, NG0, NG1, NG2, NG3) \
void int2e_##KER##_##NAME##_optimizer(CINTOpt **opt, FINT *atm, FINT natm, \
                                      FINT *bas, FINT nbas, double *env) { \
        FINT ng[] = {NG0, NG1, NG2, NG3, 4, 1, 1, 81}; \
        CINTall_2e_stg_optimizer(opt, ng, atm, natm, bas, nbas, env); \
} \
CACHE_SIZE_T int2e_##KER##_##NAME##_sph(double *out, FINT *dims, FINT *shls, FINT *atm, FINT natm, \
                       FINT *bas, FINT nbas, double *env, CINTOpt *opt, double *cache) { \
        FINT ng[] = {NG0, NG1, NG2, NG3, 4, 1, 1, 81}; \
        CINTEnvVars envs; \
        CINTinit_int2e_##KER##_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env); \
        envs.f_gout = &CINTgout2e_int2e_##NAME; \
        return CINT2e_drv(out, dims, &envs, opt, cache, &c2s_sph_2e1); \
}

/* ng layouts copied from libcint src/autocode/hess.c */
F12CLASS(stg, ipvip1ipvip2, 1, 1, 1, 1)
F12CLASS(yp,  ipvip1ipvip2, 1, 1, 1, 1)
F12CLASS(stg, ipip1ipip2,   2, 0, 2, 0)
F12CLASS(yp,  ipip1ipip2,   2, 0, 2, 0)

/*
 * RF12_contract2e
 *
 *   T[a, cj, j, b, cl, l] = sum_{ci, i, ck, k} A[a, ci, i] B[b, ck, k]
 *                                           * (ci i  cj j | K | ck k  cl l)
 *
 * (complex A, B given as separate real/imaginary arrays; integrals real.)
 * The integral class has ncomp = nci*ncj*nck*ncl components; libcint stores
 * a shell-quartet block as out[comp][l][k][j][i] (i fastest) with comp the
 * row-major index of (ci, cj, ck, cl) for the classes used here (checked
 * against finite differences in rf12/test).  Output T is complex,
 * layout [a][cj][j][b][cl][l], accumulated (caller zeroes it).
 */
typedef CACHE_SIZE_T (*intor_t)(double *out, FINT *dims, FINT *shls, FINT *atm, FINT natm,
                                FINT *bas, FINT nbas, double *env, CINTOpt *opt, double *cache);

void RF12_contract2e(intor_t intor, CINTOpt *opt, int nci, int ncj, int nck, int ncl,
                     double *Are, double *Aim, int na,
                     double *Bre, double *Bim, int nb,
                     double *Tre, double *Tim, int *ao_loc,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int nao = ao_loc[nbas];
        int ncomp = nci * ncj * nck * ncl;
        long sT_l = 1, sT_cl = nao, sT_b = (long)ncl * nao;
        long sT_j = (long)nb * sT_b, sT_cj = (long)nao * sT_j, sT_a = (long)ncj * sT_cj;
        long sA_i = 1, sA_ci = nao, sA_a = (long)nci * nao;
        long sB_k = 1, sB_ck = nao, sB_b = (long)nck * nao;
#pragma omp parallel
{
        int ish, jsh, ksh, lsh;
        int shls[4];
        int dmax = 0;
        for (ish = 0; ish < nbas; ish++) {
                int d = ao_loc[ish+1] - ao_loc[ish];
                if (d > dmax) dmax = d;
        }
        long blk = (long)dmax * dmax * dmax * dmax * ncomp;
        double *buf = malloc(sizeof(double) * blk);
        /* cache size as in pyscf GTOmax_cache_size: query libcint with out = NULL */
        long csize = 0;
        for (ish = 0; ish < nbas; ish++) {
                int s4[4] = {ish, ish, ish, ish};
                long n = (*intor)(NULL, NULL, s4, atm, natm, bas, nbas, env, NULL, NULL);
                if (n > csize) csize = n;
        }
        double *cache = malloc(sizeof(double) * (csize + 1000));
        /* partial sums: P[a][cj][j][ck][k][cl][l] for one (ish, jsh, ksh, lsh) after i contraction */
        long psize = (long)na * ncj * dmax * nck * dmax * ncl * dmax;
        double *Pre = malloc(sizeof(double) * psize);
        double *Pim = malloc(sizeof(double) * psize);
        /* thread-private accumulation of T for the j-shell slice handled by this thread */
#pragma omp for schedule(dynamic)
        for (jsh = 0; jsh < nbas; jsh++) {
        for (lsh = 0; lsh < nbas; lsh++) {
        for (ish = 0; ish < nbas; ish++) {
        for (ksh = 0; ksh < nbas; ksh++) {
                shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                int di = ao_loc[ish+1] - ao_loc[ish];
                int dj = ao_loc[jsh+1] - ao_loc[jsh];
                int dk = ao_loc[ksh+1] - ao_loc[ksh];
                int dl = ao_loc[lsh+1] - ao_loc[lsh];
                int i0 = ao_loc[ish], j0 = ao_loc[jsh], k0 = ao_loc[ksh], l0 = ao_loc[lsh];
                if (!(*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                        continue;
                }
                long nblk = (long)di * dj * dk * dl;
                int a, ci, cj, ck, cl, i, j, k, l, b;
                /* step 1: contract i -> a:  P[a][cj][j][ck][k][cl][l] */
                long sPl = 1, sPcl = dl, sPk = (long)ncl * dl, sPck = (long)dk * sPk;
                long sPj = (long)nck * sPck, sPcj = (long)dj * sPj, sPa = (long)ncj * sPcj;
                memset(Pre, 0, sizeof(double) * na * sPa);
                memset(Pim, 0, sizeof(double) * na * sPa);
                for (ci = 0; ci < nci; ci++) {
                for (cj = 0; cj < ncj; cj++) {
                for (ck = 0; ck < nck; ck++) {
                for (cl = 0; cl < ncl; cl++) {
                        int comp = ((ci * ncj + cj) * nck + ck) * ncl + cl;
                        double *g = buf + comp * nblk;
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                double *gi = g + ((long)(l * dk + k) * dj + j) * di;
                                for (a = 0; a < na; a++) {
                                        double *ar = Are + a * sA_a + ci * sA_ci + i0;
                                        double *aI = Aim + a * sA_a + ci * sA_ci + i0;
                                        double sr = 0, si = 0;
                                        for (i = 0; i < di; i++) {
                                                sr += ar[i] * gi[i];
                                                si += aI[i] * gi[i];
                                        }
                                        long p = a * sPa + cj * sPcj + j * sPj + ck * sPck + k * sPk + cl * sPcl + l;
                                        Pre[p] += sr;
                                        Pim[p] += si;
                                }
                        } } }
                } } } }
                /* step 2: contract k -> b and accumulate into T */
                for (a = 0; a < na; a++) {
                for (cj = 0; cj < ncj; cj++) {
                for (j = 0; j < dj; j++) {
                for (b = 0; b < nb; b++) {
                for (cl = 0; cl < ncl; cl++) {
                for (l = 0; l < dl; l++) {
                        double sr = 0, si = 0;
                        for (ck = 0; ck < nck; ck++) {
                                double *br = Bre + b * sB_b + ck * sB_ck + k0;
                                double *bI = Bim + b * sB_b + ck * sB_ck + k0;
                                long p0 = a * sPa + cj * sPcj + j * sPj + ck * sPck + cl * sPcl + l;
                                for (k = 0; k < dk; k++) {
                                        double pr = Pre[p0 + k * sPk];
                                        double pi = Pim[p0 + k * sPk];
                                        sr += pr * br[k] - pi * bI[k];
                                        si += pr * bI[k] + pi * br[k];
                                }
                        }
                        long t = a * sT_a + cj * sT_cj + (long)(j0 + j) * sT_j + b * sT_b + cl * sT_cl + (l0 + l);
                        /* each (j, l) element is owned by one thread (jsh loop) */
                        Tre[t] += sr;
                        Tim[t] += si;
                } } } } } }
        } } } }
        free(Pim);
        free(Pre);
        free(cache);
        free(buf);
}
}
