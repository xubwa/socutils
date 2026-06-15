/* Optimized spinor (j-spinor) AO->MO e1 transform.
 *
 * Reuses the non-relativistic AO permutational-symmetry fill
 * (AO2MOnr_e1fill_drv + AO2MOfill_nr_s2ij / _s4 from libao2mo) to compute the
 * real AO integrals with i>=j (and, for s4, k>=l) symmetry -- ~4x fewer shell
 * quartets than the stock s1-only nrr path -- then performs the complex
 * two-component (alpha + beta j-spinor) MO contraction here.
 *
 * Author: prototype for socutils.
 */
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "r_ao2mo.h"

#define MAX(X,Y)  ((X) > (Y) ? (X) : (Y))

/* from libao2mo (non-relativistic AO fill with permutational symmetry) */
void AO2MOnr_e1fill_drv(int (*intor)(), void (*fill)(), double *eri,
                        int klsh_start, int klsh_count, int nkl, int ncomp,
                        int *ao_loc, CINTOpt *cintopt, CVHFOpt *vhfopt,
                        int *atm, int natm, int *bas, int nbas, double *env);

/* Complex two-component MO contraction: full nao x nao real AO block -> MO.
 * (identical to AO2MOmmm_nrr_iltj in nrr_ao2mo.c) */
int AO2MOmmm_nrr_iltj(double complex *vout, double *eri,
                      struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int n2c = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        int i;
        double *buf1 = malloc(sizeof(double)*n2c*i_count*3);
        double *buf2 = buf1 + n2c*i_count;
        double *buf3 = buf2 + n2c*i_count;
        double *bufr, *bufi;
        double *mo1 = malloc(sizeof(double) * n2c*MAX(i_count,j_count)*2);
        double *mo2, *mo_r, *mo_i;
        double *eri_r = malloc(sizeof(double) * n2c*n2c*3);
        double *eri_i = eri_r + n2c*n2c;
        double *eri1  = eri_i + n2c*n2c;
        double *vout1, *vout2, *vout3;

        mo_r = envs->mo_r + i_start * n2c;
        mo_i = envs->mo_i + i_start * n2c;
        mo2 = mo1 + n2c*i_count;
        for (i = 0; i < n2c*i_count; i++) {
                mo1[i] = mo_r[i] - mo_i[i];
                mo2[i] =-mo_i[i] - mo_r[i];
        }
        for (i = 0; i < n2c*n2c; i++) {
                eri_r[i] = eri[i];
                eri_i[i] = 0.0;
                eri1 [i] = eri_r[i] + eri_i[i];
        }
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri1, &n2c, mo_r, &n2c, &D0, buf1, &n2c);
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri_r, &n2c, mo2, &n2c, &D0, buf2, &n2c);
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri_i, &n2c, mo1, &n2c, &D0, buf3, &n2c);
        free(eri_r);

        bufr = buf3;
        bufi = buf2;
        for (i = 0; i < n2c*i_count; i++) {
                buf3[i] = buf1[i] - buf3[i];
                buf2[i] = buf1[i] + buf2[i];
        }
        for (i = 0; i < n2c*i_count; i++) {
                buf1[i] = bufr[i] + bufi[i];
        }
        mo_r = envs->mo_r + j_start * n2c;
        mo_i = envs->mo_i + j_start * n2c;
        mo2 = mo1 + n2c*j_count;
        for (i = 0; i < n2c*j_count; i++) {
                mo1[i] = mo_r[i] + mo_i[i];
                mo2[i] = mo_i[i] - mo_r[i];
        }
        vout1 = malloc(sizeof(double)*i_count*j_count*3);
        vout2 = vout1 + i_count * j_count;
        vout3 = vout2 + i_count * j_count;
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo_r, &n2c, buf1, &n2c, &D0, vout1, &j_count);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo2, &n2c, bufr, &n2c, &D0, vout2, &j_count);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo1, &n2c, bufi, &n2c, &D0, vout3, &j_count);
        for (i = 0; i < i_count*j_count; i++) {
                vout[i] = (vout1[i]-vout3[i]) + (vout1[i]+vout2[i])*_Complex_I;
        }
        free(vout1);
        free(buf1);
        free(mo1);
        return 0;
}

/* unpack tril AO block (i>=j) -> full nao x nao, then complex MO contraction */
static void transe1_nrr_s2ij(int (*fmmm)(), double complex *vout, double *vin,
                             double *buf, int row_id, struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao_pair = (size_t)nao * (nao + 1) / 2;
        NPdunpack_tril(nao, vin + nao_pair * row_id, buf, 0);
        (*fmmm)(vout + ij_pair * row_id, buf, envs, 0);
}

void AO2MOnrr_opt_e1_drv(int (*intor)(), void (*fill)(), int (*fmmm)(),
                         double complex *eri, double complex *mo_a,
                         double complex *mo_b,
                         int klsh_start, int klsh_count, int nkl, int ncomp,
                         int *orbs_slice, int *tao, int *ao_loc,
                         CINTOpt *cintopt, CVHFOpt *vhfopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        const int i_start = orbs_slice[0];
        const int i_count = orbs_slice[1] - orbs_slice[0];
        const int j_start = orbs_slice[2];
        const int j_count = orbs_slice[3] - orbs_slice[2];
        const int ij_count = i_count * j_count;
        const int nao = ao_loc[nbas];
        const int nmo = MAX(orbs_slice[1], orbs_slice[3]);
        size_t i;

        double *mo_ra = malloc(sizeof(double) * nao * nmo);
        double *mo_ia = malloc(sizeof(double) * nao * nmo);
        double *mo_rb = malloc(sizeof(double) * nao * nmo);
        double *mo_ib = malloc(sizeof(double) * nao * nmo);
        for (i = 0; i < (size_t)nao*nmo; i++) {
                mo_ra[i] = creal(mo_a[i]);  mo_ia[i] = cimag(mo_a[i]);
                mo_rb[i] = creal(mo_b[i]);  mo_ib[i] = cimag(mo_b[i]);
        }
        struct _AO2MOEnvs envs_a = {natm, nbas, atm, bas, env, nao,
                klsh_start, klsh_count, i_start, i_count, j_start, j_count,
                ncomp, tao, ao_loc, mo_a, mo_ra, mo_ia, cintopt, vhfopt};
        struct _AO2MOEnvs envs_b = {natm, nbas, atm, bas, env, nao,
                klsh_start, klsh_count, i_start, i_count, j_start, j_count,
                ncomp, tao, ao_loc, mo_b, mo_rb, mo_ib, cintopt, vhfopt};

        /* real AO integrals with permutational symmetry (reused nr fill) */
        double *eri_ao = malloc(sizeof(double) * (size_t)nao*nao*nkl*ncomp);
        if (eri_ao == NULL) {
                fprintf(stderr, "malloc failed in AO2MOnrr_opt_e1_drv\n");
                exit(1);
        }
        AO2MOnr_e1fill_drv(intor, fill, eri_ao, klsh_start, klsh_count,
                           nkl, ncomp, ao_loc, cintopt, vhfopt,
                           atm, natm, bas, nbas, env);

        size_t off = (size_t)ncomp * nkl * ij_count;
#pragma omp parallel default(none) \
        shared(fmmm, eri, eri_ao, nkl, ncomp, envs_a, envs_b, nao, off) \
        private(i)
{
        double *buf = malloc(sizeof(double) * (size_t)nao * nao);
#pragma omp for schedule(static)
        for (i = 0; i < (size_t)nkl*ncomp; i++) {
                transe1_nrr_s2ij(fmmm, eri,       eri_ao, buf, i, &envs_a);
                transe1_nrr_s2ij(fmmm, eri + off, eri_ao, buf, i, &envs_b);
        }
        free(buf);
}
        free(eri_ao);
        free(mo_ra); free(mo_ia); free(mo_rb); free(mo_ib);
}
