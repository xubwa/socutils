/*
 * x2c_dhf_sph_pcc.c -- C translation of src/dhf_sph_pcc.cpp (X2CAMF):
 * the x2c2e picture-change correction drivers (x2c2ePCC / x2c2ePCC_K /
 * h_x2c2e) and the Coulomb-only / J / K Fock builders.
 *
 * These are DHF_SPH members (take DHF_SPH* self).  They call helpers
 * implemented in x2c_dhf_sph.c (evaluateFock_2e / _K, evaluateDensity_spinor)
 * as link-time references.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "x2c_dhf_sph.h"
#include "x2c_general.h"
#include "x2c_mat.h"

/* ------------------------------------------------------------------ */
/* DHF_SPH::x2c2ePCC                                                   */
/* ------------------------------------------------------------------ */
vmat_t dhf_sph_x2c2ePCC(DHF_SPH* self, bool amfi4c, vmat_t* coeff2c)
{
    if (self->printLevel >= 4)
        printf("Running DHF_SPH::x2c2ePCC\n");
    if (!self->converged)
    {
        printf("SCF did not converge. x2c2ePCC cannot be used!\n");
        /* exit(99); */
    }

    int occMax_irrep = self->occMax_irrep;
    vmat_t fock_pcc = vmat_new(occMax_irrep);
    vmat_t fock_4c_2e = vmat_new(occMax_irrep);
    vmat_t fock_x2c2e = vmat_new(occMax_irrep);
    vmat_t fock_x2c2e_2e = vmat_new(occMax_irrep);
    vmat_t JK_x2c2c = vmat_new(occMax_irrep);
    vmat_t coeff_2c = vmat_new(occMax_irrep);
    vmat_t density_2c = vmat_new(occMax_irrep);
    vmat_t h1e_x2c2e = vmat_new(occMax_irrep);
    vmat_t h1e_x2c1e = vmat_new(occMax_irrep);
    vmat_t XXX = vmat_new(occMax_irrep);
    vmat_t RRR = vmat_new(occMax_irrep);
    vmat_t XXX_1e = vmat_new(occMax_irrep);
    vmat_t RRR_1e = vmat_new(occMax_irrep);
    vmat_t overlap_2c = vmat_new(occMax_irrep);
    vmat_t overlap_h_i_2c = vmat_new(occMax_irrep);

    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        mat_assign(&overlap_2c.m[ir],
                   mat_block(self->overlap_4c.m[ir], 0, 0,
                             self->overlap_4c.m[ir].rows / 2,
                             self->overlap_4c.m[ir].cols / 2));
        mat_assign(&overlap_h_i_2c.m[ir],
                   matrix_half_inverse(overlap_2c.m[ir]));

        mat_assign(&XXX.m[ir], X2C_get_X_from_coeff(self->coeff.m[ir]));
        mat_assign(&RRR.m[ir],
                   X2C_get_R_4c(self->overlap_4c.m[ir], XXX.m[ir]));

        mat_assign(&h1e_x2c2e.m[ir],
                   X2C_transform_4c_2c(self->h1e_4c.m[ir], XXX.m[ir],
                                       RRR.m[ir]));

        if (coeff2c == NULL)
        {
            mat_assign(&fock_x2c2e.m[ir],
                       X2C_transform_4c_2c(self->fock_4c.m[ir], XXX.m[ir],
                                           RRR.m[ir]));
            double* ene_mo_tmp =
                (double*)malloc((size_t)fock_x2c2e.m[ir].rows *
                                sizeof(double));
            eigensolverG(fock_x2c2e.m[ir], overlap_h_i_2c.m[ir], ene_mo_tmp,
                         &coeff_2c.m[ir]);
            free(ene_mo_tmp);
        }
        else
        {
            mat_assign(&coeff_2c.m[ir], mat_clone(coeff2c->m[ir]));
        }
        mat_assign(&density_2c.m[ir],
                   dhf_sph_evaluateDensity_spinor(coeff_2c.m[ir],
                                                  self->occNumber.v[ir],
                                                  true));

        /* X2C1E */
        mat_assign(&XXX_1e.m[ir],
                   X2C_get_X(self->overlap.m[ir], self->kinetic.m[ir],
                             self->WWW.m[ir], self->Vnuc.m[ir]));
        mat_assign(&RRR_1e.m[ir],
                   X2C_get_R(self->overlap.m[ir], self->kinetic.m[ir],
                             XXX_1e.m[ir]));
        mat_assign(&h1e_x2c1e.m[ir],
                   X2C_evaluate_h1e_x2c(self->overlap.m[ir],
                                        self->kinetic.m[ir], self->WWW.m[ir],
                                        self->Vnuc.m[ir], XXX_1e.m[ir],
                                        RRR_1e.m[ir]));
    }

    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        dhf_sph_evaluateFock_2e(self, &fock_4c_2e.m[ir], false, &self->density,
                                self->irrep_list[ir].size, ir);
        dhf_sph_evaluateFock_2e(self, &JK_x2c2c.m[ir], true, &density_2c,
                                self->irrep_list[ir].size, ir);
        mat_assign(&fock_x2c2e_2e.m[ir],
                   X2C_transform_4c_2c(fock_4c_2e.m[ir], XXX.m[ir],
                                       RRR.m[ir]));
    }
    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        /* fock_pcc(ir) = fock_x2c2e_2e - JK_x2c2c + h1e_x2c2e - h1e_x2c1e; */
        mat_t tmp = mat_clone(fock_x2c2e_2e.m[ir]);
        mat_add_inplace(tmp, JK_x2c2c.m[ir], -1.0);
        mat_add_inplace(tmp, h1e_x2c2e.m[ir], 1.0);
        mat_add_inplace(tmp, h1e_x2c1e.m[ir], -1.0);
        mat_assign(&fock_pcc.m[ir], tmp);
    }

    vmat_free(&self->x2cXXX);
    self->x2cXXX = XXX;
    vmat_free(&self->x2cRRR);
    self->x2cRRR = RRR;
    self->X_calculated = true;

    /* Special case for H-like atoms. */
    if (fabs(self->nelec - 1.0) < 1e-5)
    {
        for (int ir = 0; ir < occMax_irrep; ir++)
        {
            mat_assign(&fock_pcc.m[ir],
                       mat_new(fock_pcc.m[ir].rows, fock_pcc.m[ir].cols));
            mat_assign(&fock_4c_2e.m[ir],
                       mat_new(fock_4c_2e.m[ir].rows,
                               fock_4c_2e.m[ir].cols));
        }
    }

    /* cleanup temporaries not transferred to the caller */
    vmat_free(&fock_x2c2e);
    vmat_free(&fock_x2c2e_2e);
    vmat_free(&JK_x2c2c);
    vmat_free(&coeff_2c);
    vmat_free(&density_2c);
    vmat_free(&h1e_x2c2e);
    vmat_free(&h1e_x2c1e);
    vmat_free(&XXX_1e);
    vmat_free(&RRR_1e);
    vmat_free(&overlap_2c);
    vmat_free(&overlap_h_i_2c);

    if (amfi4c)
    {
        vmat_free(&fock_pcc);
        return fock_4c_2e;
    }
    else
    {
        vmat_free(&fock_4c_2e);
        return fock_pcc;
    }
}

/* ------------------------------------------------------------------ */
/* DHF_SPH::x2c2ePCC_K                                                 */
/* ------------------------------------------------------------------ */
vmat_t dhf_sph_x2c2ePCC_K(DHF_SPH* self, bool amfi4c, vmat_t* coeff2c)
{
    if (self->printLevel >= 4)
        printf("Running DHF_SPH::x2c2ePCC_K\n");
    if (!self->converged)
    {
        printf("WARNING: SCF did not converge. Use x2c2ePCC with caution.\n");
    }

    int occMax_irrep = self->occMax_irrep;
    vmat_t fock_pcc = vmat_new(occMax_irrep);
    vmat_t fock_4c_2e = vmat_new(occMax_irrep);
    vmat_t fock_x2c2e = vmat_new(occMax_irrep);
    vmat_t fock_x2c2e_2e = vmat_new(occMax_irrep);
    vmat_t JK_x2c2c = vmat_new(occMax_irrep);
    vmat_t coeff_2c = vmat_new(occMax_irrep);
    vmat_t density_2c = vmat_new(occMax_irrep);
    vmat_t h1e_x2c2e = vmat_new(occMax_irrep);
    vmat_t h1e_x2c1e = vmat_new(occMax_irrep);
    vmat_t XXX = vmat_new(occMax_irrep);
    vmat_t RRR = vmat_new(occMax_irrep);
    vmat_t XXX_1e = vmat_new(occMax_irrep);
    vmat_t RRR_1e = vmat_new(occMax_irrep);
    vmat_t overlap_2c = vmat_new(occMax_irrep);
    vmat_t overlap_h_i_2c = vmat_new(occMax_irrep);

    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        mat_assign(&overlap_2c.m[ir],
                   mat_block(self->overlap_4c.m[ir], 0, 0,
                             self->overlap_4c.m[ir].rows / 2,
                             self->overlap_4c.m[ir].cols / 2));
        mat_assign(&overlap_h_i_2c.m[ir],
                   matrix_half_inverse(overlap_2c.m[ir]));

        mat_assign(&XXX.m[ir], X2C_get_X_from_coeff(self->coeff.m[ir]));
        mat_assign(&RRR.m[ir],
                   X2C_get_R_4c(self->overlap_4c.m[ir], XXX.m[ir]));

        mat_assign(&h1e_x2c2e.m[ir],
                   X2C_transform_4c_2c(self->h1e_4c.m[ir], XXX.m[ir],
                                       RRR.m[ir]));

        if (coeff2c == NULL)
        {
            mat_assign(&fock_x2c2e.m[ir],
                       X2C_transform_4c_2c(self->fock_4c.m[ir], XXX.m[ir],
                                           RRR.m[ir]));
            double* ene_mo_tmp =
                (double*)malloc((size_t)fock_x2c2e.m[ir].rows *
                                sizeof(double));
            eigensolverG(fock_x2c2e.m[ir], overlap_h_i_2c.m[ir], ene_mo_tmp,
                         &coeff_2c.m[ir]);
            free(ene_mo_tmp);
        }
        else
        {
            mat_assign(&coeff_2c.m[ir], mat_clone(coeff2c->m[ir]));
        }
        mat_assign(&density_2c.m[ir],
                   dhf_sph_evaluateDensity_spinor(coeff_2c.m[ir],
                                                  self->occNumber.v[ir],
                                                  true));

        /* X2C1E */
        mat_assign(&XXX_1e.m[ir],
                   X2C_get_X(self->overlap.m[ir], self->kinetic.m[ir],
                             self->WWW.m[ir], self->Vnuc.m[ir]));
        mat_assign(&RRR_1e.m[ir],
                   X2C_get_R(self->overlap.m[ir], self->kinetic.m[ir],
                             XXX_1e.m[ir]));
        mat_assign(&h1e_x2c1e.m[ir],
                   X2C_evaluate_h1e_x2c(self->overlap.m[ir],
                                        self->kinetic.m[ir], self->WWW.m[ir],
                                        self->Vnuc.m[ir], XXX_1e.m[ir],
                                        RRR_1e.m[ir]));
    }

    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        dhf_sph_evaluateFock_K(self, &fock_4c_2e.m[ir], false, &self->density,
                               self->irrep_list[ir].size, ir);
        dhf_sph_evaluateFock_K(self, &JK_x2c2c.m[ir], true, &density_2c,
                               self->irrep_list[ir].size, ir);
        mat_assign(&fock_x2c2e_2e.m[ir],
                   X2C_transform_4c_2c(fock_4c_2e.m[ir], XXX.m[ir],
                                       RRR.m[ir]));
    }
    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        mat_t tmp = mat_clone(fock_x2c2e_2e.m[ir]);
        mat_add_inplace(tmp, JK_x2c2c.m[ir], -1.0);
        mat_add_inplace(tmp, h1e_x2c2e.m[ir], 1.0);
        mat_add_inplace(tmp, h1e_x2c1e.m[ir], -1.0);
        mat_assign(&fock_pcc.m[ir], tmp);
    }

    vmat_free(&self->x2cXXX);
    self->x2cXXX = XXX;
    vmat_free(&self->x2cRRR);
    self->x2cRRR = RRR;
    self->X_calculated = true;

    /* Special case for H-like atoms. */
    if (fabs(self->nelec - 1.0) < 1e-5)
    {
        for (int ir = 0; ir < occMax_irrep; ir++)
        {
            mat_assign(&fock_pcc.m[ir],
                       mat_new(fock_pcc.m[ir].rows, fock_pcc.m[ir].cols));
            mat_assign(&fock_4c_2e.m[ir],
                       mat_new(fock_4c_2e.m[ir].rows,
                               fock_4c_2e.m[ir].cols));
        }
    }

    vmat_free(&fock_x2c2e);
    vmat_free(&fock_x2c2e_2e);
    vmat_free(&JK_x2c2c);
    vmat_free(&coeff_2c);
    vmat_free(&density_2c);
    vmat_free(&h1e_x2c2e);
    vmat_free(&h1e_x2c1e);
    vmat_free(&XXX_1e);
    vmat_free(&RRR_1e);
    vmat_free(&overlap_2c);
    vmat_free(&overlap_h_i_2c);

    if (amfi4c)
    {
        vmat_free(&fock_pcc);
        return fock_4c_2e;
    }
    else
    {
        vmat_free(&fock_4c_2e);
        return fock_pcc;
    }
}

/* ------------------------------------------------------------------ */
/* DHF_SPH::h_x2c2e                                                    */
/* ------------------------------------------------------------------ */
vmat_t dhf_sph_h_x2c2e(DHF_SPH* self, vmat_t* coeff2c)
{
    if (!self->converged)
    {
        printf("SCF did not converge. x2c2ePCC cannot be used!\n");
        /* exit(99); */
    }

    int occMax_irrep = self->occMax_irrep;
    vmat_t fock_pcc = vmat_new(occMax_irrep);
    vmat_t fock_4c_2e = vmat_new(occMax_irrep);
    vmat_t fock_x2c2e = vmat_new(occMax_irrep);
    vmat_t fock_x2c2e_2e = vmat_new(occMax_irrep);
    vmat_t JK_x2c2c = vmat_new(occMax_irrep);
    vmat_t coeff_2c = vmat_new(occMax_irrep);
    vmat_t density_2c = vmat_new(occMax_irrep);
    vmat_t h1e_x2c2e = vmat_new(occMax_irrep);
    vmat_t h1e_x2c1e = vmat_new(occMax_irrep);
    vmat_t XXX = vmat_new(occMax_irrep);
    vmat_t RRR = vmat_new(occMax_irrep);
    vmat_t XXX_1e = vmat_new(occMax_irrep);
    vmat_t RRR_1e = vmat_new(occMax_irrep);
    vmat_t overlap_2c = vmat_new(occMax_irrep);
    vmat_t overlap_h_i_2c = vmat_new(occMax_irrep);

    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        mat_assign(&overlap_2c.m[ir],
                   mat_block(self->overlap_4c.m[ir], 0, 0,
                             self->overlap_4c.m[ir].rows / 2,
                             self->overlap_4c.m[ir].cols / 2));
        mat_assign(&overlap_h_i_2c.m[ir],
                   matrix_half_inverse(overlap_2c.m[ir]));
        mat_assign(&XXX.m[ir], X2C_get_X_from_coeff(self->coeff.m[ir]));
        mat_assign(&RRR.m[ir],
                   X2C_get_R_4c(self->overlap_4c.m[ir], XXX.m[ir]));

        mat_assign(&h1e_x2c2e.m[ir],
                   X2C_transform_4c_2c(self->h1e_4c.m[ir], XXX.m[ir],
                                       RRR.m[ir]));
        mat_assign(&fock_x2c2e.m[ir],
                   X2C_transform_4c_2c(self->fock_4c.m[ir], XXX.m[ir],
                                       RRR.m[ir]));
        if (coeff2c == NULL)
        {
            double* ene_mo_tmp =
                (double*)malloc((size_t)fock_x2c2e.m[ir].rows *
                                sizeof(double));
            eigensolverG(fock_x2c2e.m[ir], overlap_h_i_2c.m[ir], ene_mo_tmp,
                         &coeff_2c.m[ir]);
            free(ene_mo_tmp);
        }
        else
            mat_assign(&coeff_2c.m[ir], mat_clone(coeff2c->m[ir]));

        mat_assign(&density_2c.m[ir],
                   dhf_sph_evaluateDensity_spinor(coeff_2c.m[ir],
                                                  self->occNumber.v[ir],
                                                  true));

        /* X2C1E */
        mat_assign(&XXX_1e.m[ir],
                   X2C_get_X(self->overlap.m[ir], self->kinetic.m[ir],
                             self->WWW.m[ir], self->Vnuc.m[ir]));
        mat_assign(&RRR_1e.m[ir],
                   X2C_get_R(self->overlap.m[ir], self->kinetic.m[ir],
                             XXX_1e.m[ir]));
        mat_assign(&h1e_x2c1e.m[ir],
                   X2C_evaluate_h1e_x2c(self->overlap.m[ir],
                                        self->kinetic.m[ir], self->WWW.m[ir],
                                        self->Vnuc.m[ir], XXX_1e.m[ir],
                                        RRR_1e.m[ir]));
    }

    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        dhf_sph_evaluateFock_2e(self, &fock_4c_2e.m[ir], false, &self->density,
                                self->irrep_list[ir].size, ir);
        dhf_sph_evaluateFock_2e(self, &JK_x2c2c.m[ir], true, &density_2c,
                                self->irrep_list[ir].size, ir);
        mat_assign(&fock_x2c2e_2e.m[ir],
                   X2C_transform_4c_2c(fock_4c_2e.m[ir], XXX.m[ir],
                                       RRR.m[ir]));
    }
    for (int ir = 0; ir < occMax_irrep; ir++)
    {
        mat_assign(&fock_pcc.m[ir], mat_clone(h1e_x2c2e.m[ir]));
    }

    vmat_free(&self->x2cXXX);
    self->x2cXXX = XXX;
    vmat_free(&self->x2cRRR);
    self->x2cRRR = RRR;
    self->X_calculated = true;

    vmat_free(&fock_4c_2e);
    vmat_free(&fock_x2c2e);
    vmat_free(&fock_x2c2e_2e);
    vmat_free(&JK_x2c2c);
    vmat_free(&coeff_2c);
    vmat_free(&density_2c);
    vmat_free(&h1e_x2c2e);
    vmat_free(&h1e_x2c1e);
    vmat_free(&XXX_1e);
    vmat_free(&RRR_1e);
    vmat_free(&overlap_2c);
    vmat_free(&overlap_h_i_2c);

    return fock_pcc;
}

/* ------------------------------------------------------------------ */
/* DHF_SPH::evaluateFock_2e                                            */
/* ------------------------------------------------------------------ */
void dhf_sph_evaluateFock_2e(DHF_SPH* self, mat_t* fock, bool twoC,
                             const vmat_t* den, int size, int Iirrep)
{
    int ir = self->all2compact[Iirrep];
    if (!twoC)
    {
        mat_assign(fock, mat_new(size * 2, size * 2));
#pragma omp parallel for
        for (int mm = 0; mm < size; mm++)
            for (int nn = 0; nn <= mm; nn++)
            {
                M(*fock, mm, nn) = 0.0;
                M(*fock, mm + size, nn) = 0.0;
                if (mm != nn) M(*fock, nn + size, mm) = 0.0;
                M(*fock, mm + size, nn + size) = 0.0;
                for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
                {
                    int Jirrep = self->compact2all[jr];
                    double twojP1 = self->irrep_list[Jirrep].two_j + 1;
                    int size_tmp2 = self->irrep_list[Jirrep].size;
                    for (int ss = 0; ss < size_tmp2; ss++)
                        for (int rr = 0; rr < size_tmp2; rr++)
                        {
                            int emn = mm * size + nn,
                                esr = ss * size_tmp2 + rr,
                                emr = mm * size_tmp2 + rr,
                                esn = ss * size + nn;
                            M(*fock, mm, nn) +=
                                twojP1 * M(den->m[Jirrep], ss, rr) *
                                    self->h2eLLLL_JK.J[ir][jr][emn][esr] +
                                twojP1 *
                                    M(den->m[Jirrep], size_tmp2 + ss,
                                      size_tmp2 + rr) *
                                    self->h2eSSLL_JK.J[jr][ir][esr][emn];
                            M(*fock, mm + size, nn) -=
                                twojP1 *
                                M(den->m[Jirrep], ss, size_tmp2 + rr) *
                                self->h2eSSLL_JK.K[ir][jr][emr][esn];
                            if (mm != nn)
                            {
                                int enr = nn * size_tmp2 + rr,
                                    esm = ss * size + mm;
                                M(*fock, nn + size, mm) -=
                                    twojP1 *
                                    M(den->m[Jirrep], ss, size_tmp2 + rr) *
                                    self->h2eSSLL_JK.K[ir][jr][enr][esm];
                            }
                            M(*fock, mm + size, nn + size) +=
                                twojP1 *
                                    M(den->m[Jirrep], size_tmp2 + ss,
                                      size_tmp2 + rr) *
                                    self->h2eSSSS_JK.J[ir][jr][emn][esr] +
                                twojP1 * M(den->m[Jirrep], ss, rr) *
                                    self->h2eSSLL_JK.J[ir][jr][emn][esr];
                            if (self->with_gaunt)
                            {
                                int enm = nn * size + mm,
                                    ers = rr * size_tmp2 + ss;
                                M(*fock, mm, nn) -=
                                    twojP1 *
                                    M(den->m[Jirrep], size_tmp2 + ss,
                                      size_tmp2 + rr) *
                                    self->gauntLSSL_JK.K[ir][jr][emr][esn];
                                M(*fock, mm + size, nn + size) -=
                                    twojP1 * M(den->m[Jirrep], ss, rr) *
                                    self->gauntLSSL_JK.K[jr][ir][esn][emr];
                                M(*fock, mm + size, nn) +=
                                    twojP1 *
                                        M(den->m[Jirrep], size_tmp2 + ss,
                                          rr) *
                                        self->gauntLSLS_JK
                                            .J[ir][jr][enm][ers] +
                                    twojP1 *
                                        M(den->m[Jirrep], ss,
                                          size_tmp2 + rr) *
                                        self->gauntLSSL_JK
                                            .J[jr][ir][esr][emn];
                                if (mm != nn)
                                {
                                    int ern = rr * size + nn,
                                        ems = mm * size_tmp2 + ss;
                                    (void)ern;
                                    (void)ems;
                                    M(*fock, nn + size, mm) +=
                                        twojP1 *
                                            M(den->m[Jirrep], size_tmp2 + ss,
                                              rr) *
                                            self->gauntLSLS_JK
                                                .J[ir][jr][emn][ers] +
                                        twojP1 *
                                            M(den->m[Jirrep], ss,
                                              size_tmp2 + rr) *
                                            self->gauntLSSL_JK
                                                .J[jr][ir][esr][enm];
                                }
                            }
                        }
                }
                M(*fock, nn, mm) = M(*fock, mm, nn);
                M(*fock, nn + size, mm + size) =
                    M(*fock, mm + size, nn + size);
                M(*fock, nn, mm + size) = M(*fock, mm + size, nn);
                M(*fock, mm, nn + size) = M(*fock, nn + size, mm);
            }
    }
    else
    {
        mat_assign(fock, mat_new(size, size));
#pragma omp parallel for
        for (int mm = 0; mm < size; mm++)
            for (int nn = 0; nn <= mm; nn++)
            {
                M(*fock, mm, nn) = 0.0;
                for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
                {
                    int Jirrep = self->compact2all[jr];
                    double twojP1 = self->irrep_list[Jirrep].two_j + 1;
                    int size_tmp2 = self->irrep_list[Jirrep].size;
                    for (int ss = 0; ss < size_tmp2; ss++)
                        for (int rr = 0; rr < size_tmp2; rr++)
                        {
                            int emn = mm * size + nn,
                                esr = ss * size_tmp2 + rr;
                            M(*fock, mm, nn) +=
                                twojP1 * M(den->m[Jirrep], ss, rr) *
                                self->h2eLLLL_JK.J[ir][jr][emn][esr];
                        }
                }
                M(*fock, nn, mm) = M(*fock, mm, nn);
            }
    }
}

/* ------------------------------------------------------------------ */
/* DHF_SPH::evaluateFock_J                                             */
/* ------------------------------------------------------------------ */
void dhf_sph_evaluateFock_J(DHF_SPH* self, mat_t* fock, bool twoC,
                            const vmat_t* den, int size, int Iirrep)
{
    (void)self;
    (void)fock;
    (void)twoC;
    (void)den;
    (void)size;
    (void)Iirrep;
    printf("evaluateFock_J is closed now\n");
    exit(99);
}

/* ------------------------------------------------------------------ */
/* DHF_SPH::evaluateFock_K                                             */
/* ------------------------------------------------------------------ */
void dhf_sph_evaluateFock_K(DHF_SPH* self, mat_t* fock, bool twoC,
                            const vmat_t* den, int size, int Iirrep)
{
    int ir = self->all2compact[Iirrep];
    if (!twoC)
    {
        mat_assign(fock, mat_new(size * 2, size * 2));
#pragma omp parallel for
        for (int NN = 0; NN < size * (size + 1) / 2; NN++)
        {
            int tmp_i = (int)(sqrt(NN * 2.0)), mm, nn;
            if (tmp_i * (tmp_i + 1) / 2 > NN)
            {
                mm = tmp_i - 1;
            }
            else
            {
                mm = tmp_i;
            }
            nn = NN - mm * (mm + 1) / 2;

            for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
            {
                int Jirrep = self->compact2all[jr];
                double twojP1 = self->irrep_list[Jirrep].two_j + 1;
                int size_tmp2 = self->irrep_list[Jirrep].size;
                for (int ss = 0; ss < size_tmp2; ss++)
                    for (int rr = 0; rr < size_tmp2; rr++)
                    {
                        int emn = mm * size + nn, esr = ss * size_tmp2 + rr,
                            emr = mm * size_tmp2 + rr, esn = ss * size + nn;
                        (void)emn;
                        (void)esr;
                        M(*fock, mm, nn) -=
                            twojP1 * M(den->m[Jirrep], ss, rr) *
                            self->h2eLLLL_JK.K[ir][jr][emr][esn];
                        M(*fock, mm + size, nn) -=
                            twojP1 *
                            M(den->m[Jirrep], ss, size_tmp2 + rr) *
                            self->h2eSSLL_JK.K[ir][jr][emr][esn];
                        if (mm != nn)
                        {
                            int enr = nn * size_tmp2 + rr,
                                esm = ss * size + mm;
                            M(*fock, nn + size, mm) -=
                                twojP1 *
                                M(den->m[Jirrep], ss, size_tmp2 + rr) *
                                self->h2eSSLL_JK.K[ir][jr][enr][esm];
                        }
                        M(*fock, mm + size, nn + size) -=
                            twojP1 *
                            M(den->m[Jirrep], size_tmp2 + ss,
                              size_tmp2 + rr) *
                            self->h2eSSSS_JK.K[ir][jr][emr][esn];
                        if (self->with_gaunt)
                        {
                            int enm = nn * size + mm,
                                ers = rr * size_tmp2 + ss;
                            M(*fock, mm, nn) -=
                                twojP1 *
                                M(den->m[Jirrep], size_tmp2 + ss,
                                  size_tmp2 + rr) *
                                self->gauntLSSL_JK.K[ir][jr][emr][esn];
                            M(*fock, mm + size, nn + size) -=
                                twojP1 * M(den->m[Jirrep], ss, rr) *
                                self->gauntLSSL_JK.K[jr][ir][esn][emr];
                            M(*fock, mm + size, nn) +=
                                twojP1 *
                                    M(den->m[Jirrep], size_tmp2 + ss, rr) *
                                    self->gauntLSLS_JK.J[ir][jr][enm][ers] +
                                twojP1 *
                                    M(den->m[Jirrep], ss, size_tmp2 + rr) *
                                    self->gauntLSSL_JK.J[jr][ir][esr][emn];
                            if (mm != nn)
                            {
                                int ern = rr * size + nn,
                                    ems = mm * size_tmp2 + ss;
                                (void)ern;
                                (void)ems;
                                M(*fock, nn + size, mm) +=
                                    twojP1 *
                                        M(den->m[Jirrep], size_tmp2 + ss,
                                          rr) *
                                        self->gauntLSLS_JK
                                            .J[ir][jr][emn][ers] +
                                    twojP1 *
                                        M(den->m[Jirrep], ss,
                                          size_tmp2 + rr) *
                                        self->gauntLSSL_JK
                                            .J[jr][ir][esr][enm];
                            }
                        }
                    }
            }
            M(*fock, nn, mm) = M(*fock, mm, nn);
            M(*fock, nn + size, mm + size) = M(*fock, mm + size, nn + size);
            M(*fock, nn, mm + size) = M(*fock, mm + size, nn);
            M(*fock, mm, nn + size) = M(*fock, nn + size, mm);
        }
    }
    else
    {
        mat_assign(fock, mat_new(size, size));
#pragma omp parallel for
        for (int NN = 0; NN < size * (size + 1) / 2; NN++)
        {
            int tmp_i = (int)(sqrt(NN * 2.0)), mm, nn;
            if (tmp_i * (tmp_i + 1) / 2 > NN)
            {
                mm = tmp_i - 1;
            }
            else
            {
                mm = tmp_i;
            }
            nn = NN - mm * (mm + 1) / 2;
            for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
            {
                int Jirrep = self->compact2all[jr];
                double twojP1 = self->irrep_list[Jirrep].two_j + 1;
                int size_tmp2 = self->irrep_list[Jirrep].size;
                for (int ss = 0; ss < size_tmp2; ss++)
                    for (int rr = 0; rr < size_tmp2; rr++)
                    {
                        int emr = mm * size_tmp2 + rr, esn = ss * size + nn;
                        M(*fock, mm, nn) -=
                            twojP1 * M(den->m[Jirrep], ss, rr) *
                            self->h2eLLLL_JK.K[ir][jr][emr][esn];
                    }
            }
            M(*fock, nn, mm) = M(*fock, mm, nn);
        }
    }
}
