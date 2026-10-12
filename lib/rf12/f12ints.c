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
 *   T[a, cy1, y1, b, cy2, y2] = sum A[a, cx1, x1] B[b, cx2, x2] (ci i cj j | K | ck k cl l)
 *
 * mode bit 0: electron 1 contracts x1 = i (0) or x1 = j (1), the other is open (y1);
 * mode bit 1: electron 2 contracts x2 = k (0) or x2 = l (1).
 * (real AOs: (ij|kl) = (ji|kl), so derivative classes acting on the first
 * function, e.g. ipip1, can be used for either function of a pair.)  Complex A, B
 * as separate real/imaginary arrays, real integrals.  libcint stores a
 * shell-quartet block as out[comp][l][k][j][i] (i fastest) with comp the
 * row-major index of (ci, cj, ck, cl) (checked against finite differences in
 * rf12/test/test_ints4c.py).  T is accumulated (caller zeroes it).
 */
typedef CACHE_SIZE_T (*intor_t)(double *out, FINT *dims, FINT *shls, FINT *atm, FINT natm,
                                FINT *bas, FINT nbas, double *env, CINTOpt *opt, double *cache);

void RF12_contract2e(intor_t intor, CINTOpt *opt, int mode, int nci, int ncj, int nck, int ncl,
                     double *Are, double *Aim, int na,
                     double *Bre, double *Bim, int nb,
                     double *Tre, double *Tim, int *ao_loc,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int nao = ao_loc[nbas];
        int ncomp = nci * ncj * nck * ncl;
        /* contracted (x1 on electron 1, x2 on electron 2), open (y1, y2) */
        int m1 = mode & 1, m2 = (mode >> 1) & 1;
        int ncx1 = m1 ? ncj : nci, ncy1 = m1 ? nci : ncj;
        int ncx2 = m2 ? ncl : nck, ncy2 = m2 ? nck : ncl;
        long sT_y2 = 1, sT_cy2 = nao, sT_b = (long)ncy2 * nao;
        long sT_y1 = (long)nb * sT_b, sT_cy1 = (long)nao * sT_y1, sT_a = (long)ncy1 * sT_cy1;
        long sA_ci = nao, sA_a = (long)ncx1 * nao;
        long sB_ck = nao, sB_b = (long)ncx2 * nao;
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
        long csize = 0;
        for (ish = 0; ish < nbas; ish++) {
                int s4[4] = {ish, ish, ish, ish};
                long n = (*intor)(NULL, NULL, s4, atm, natm, bas, nbas, env, NULL, NULL);
                if (n > csize) csize = n;
        }
        double *cache = malloc(sizeof(double) * (csize + 1000));
        long psize = (long)na * ncy1 * dmax * ncx2 * dmax * ncy2 * dmax;
        double *Pre = malloc(sizeof(double) * psize);
        double *Pim = malloc(sizeof(double) * psize);
        int y1sh, y2sh, x1sh, x2sh;
        /* the open electron-1 shell is distributed over threads, so each
         * thread owns disjoint slices of T */
#pragma omp for schedule(dynamic)
        for (y1sh = 0; y1sh < nbas; y1sh++) {
        for (y2sh = 0; y2sh < nbas; y2sh++) {
        for (x1sh = 0; x1sh < nbas; x1sh++) {
        for (x2sh = 0; x2sh < nbas; x2sh++) {
                if (m1) { ish = y1sh; jsh = x1sh; } else { ish = x1sh; jsh = y1sh; }
                if (m2) { ksh = y2sh; lsh = x2sh; } else { ksh = x2sh; lsh = y2sh; }
                shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                int di = ao_loc[ish+1] - ao_loc[ish];
                int dj = ao_loc[jsh+1] - ao_loc[jsh];
                int dk = ao_loc[ksh+1] - ao_loc[ksh];
                int dl = ao_loc[lsh+1] - ao_loc[lsh];
                if (!(*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                        continue;
                }
                long nblk = (long)di * dj * dk * dl;
                int dx1 = m1 ? dj : di, dy1 = m1 ? di : dj;
                int dx2 = m2 ? dl : dk, dy2 = m2 ? dk : dl;
                int x10 = ao_loc[x1sh], y10 = ao_loc[y1sh], x20 = ao_loc[x2sh], y20 = ao_loc[y2sh];
                int a, ci, cj, ck, cl, i, j, k, l, b, cx1, cy1, cx2, cy2, x1, y1, x2, y2;
                long sPy2 = 1, sPcy2 = dy2, sPx2 = (long)ncy2 * dy2, sPcx2 = (long)dx2 * sPx2;
                long sPy1 = (long)ncx2 * sPcx2, sPcy1 = (long)dy1 * sPy1, sPa = (long)ncy1 * sPcy1;
                memset(Pre, 0, sizeof(double) * na * sPa);
                memset(Pim, 0, sizeof(double) * na * sPa);
                /* step 1: contract x1 -> a */
                for (ci = 0; ci < nci; ci++) {
                for (cj = 0; cj < ncj; cj++) {
                for (ck = 0; ck < nck; ck++) {
                for (cl = 0; cl < ncl; cl++) {
                        int comp = ((ci * ncj + cj) * nck + ck) * ncl + cl;
                        double *g = buf + comp * nblk;
                        cx1 = m1 ? cj : ci; cy1 = m1 ? ci : cj;
                        cx2 = m2 ? cl : ck; cy2 = m2 ? ck : cl;
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++) {
                                double v = g[((long)(l * dk + k) * dj + j) * di + i];
                                if (v == 0) continue;
                                x1 = m1 ? j : i; y1 = m1 ? i : j;
                                x2 = m2 ? l : k; y2 = m2 ? k : l;
                                long p0 = cy1 * sPcy1 + y1 * sPy1 + cx2 * sPcx2 + x2 * sPx2 + cy2 * sPcy2 + y2;
                                double *ar = Are + cx1 * sA_ci + x10 + x1;
                                double *aI = Aim + cx1 * sA_ci + x10 + x1;
                                for (a = 0; a < na; a++) {
                                        Pre[a * sPa + p0] += ar[a * sA_a] * v;
                                        Pim[a * sPa + p0] += aI[a * sA_a] * v;
                                }
                        } } } }
                } } } }
                /* step 2: contract x2 -> b and accumulate */
                for (a = 0; a < na; a++) {
                for (cy1 = 0; cy1 < ncy1; cy1++) {
                for (y1 = 0; y1 < dy1; y1++) {
                for (b = 0; b < nb; b++) {
                for (cy2 = 0; cy2 < ncy2; cy2++) {
                for (y2 = 0; y2 < dy2; y2++) {
                        double sr = 0, si = 0;
                        for (cx2 = 0; cx2 < ncx2; cx2++) {
                                double *br = Bre + b * sB_b + cx2 * sB_ck + x20;
                                double *bI = Bim + b * sB_b + cx2 * sB_ck + x20;
                                long p0 = a * sPa + cy1 * sPcy1 + y1 * sPy1 + cx2 * sPcx2 + cy2 * sPcy2 + y2;
                                for (x2 = 0; x2 < dx2; x2++) {
                                        double pr = Pre[p0 + x2 * sPx2];
                                        double pi = Pim[p0 + x2 * sPx2];
                                        sr += pr * br[x2] - pi * bI[x2];
                                        si += pr * bI[x2] + pi * br[x2];
                                }
                        }
                        long t = a * sT_a + cy1 * sT_cy1 + (long)(y10 + y1) * sT_y1 + b * sT_b + cy2 * sT_cy2 + (y20 + y2);
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
