/*
 * x2c_dhf_sph.c -- C translation of src/dhf_sph.cpp (X2CAMF):
 * DHF_SPH constructor, SCF, Fock builds, occupation table, amfi evaluation
 * and the getters.  The DHF_SPH_CA / PCC / basisGenerator members live in
 * other translation units.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "x2c_dhf_sph.h"

/* timing helpers from the C++ code: keep the call sites, drop the prints */
static void countTime(void)
{
}
static void printTime(const char* msg)
{
    (void)msg;
}

/* ----------------------------------------------------------------------
   construction
   ---------------------------------------------------------------------- */
DHF_SPH* dhf_sph_new(INT_SPH* int_sph, const char* filename, int printLevel,
                     bool spinFree, bool twoC, bool with_gaunt,
                     bool with_gauge, bool allInt, bool gaussian_nuc)
{
    DHF_SPH* self = (DHF_SPH*)calloc(1, sizeof(DHF_SPH));

    /* member initializers / defaults from include/dhf_sph.h */
    self->maxIter = 100;
    self->size_DIIS = 8;
    self->convControl = 1e-8;
    self->converged = false;
    self->renormalizedSmall = false;
    self->with_gaunt = with_gaunt;
    self->with_gauge = with_gauge;
    self->X_calculated = false;
    self->ca = NULL;

    self->irrep_list = int_sph->irrep_list;
    self->shell_list = int_sph->shell_list;
    self->size_shell = int_sph->size_shell;
    self->printLevel = printLevel;

    /* method banner */
    {
        char method[512];
        method[0] = '\0';
        if (spinFree) strcat(method, "SF-");
        if (twoC)
            strcat(method, "2c-");
        else
            strcat(method, "4c-");
        strcat(method, "HF");
        if (with_gaunt) strcat(method, " with Gaunt");
        if (with_gauge) strcat(method, " with gauge");
        if (gaussian_nuc) strcat(method, " with Gaussian nuclear model");
        printf("Initializing %s for %s atom.\n", method, int_sph->atomName);
    }

    self->Nirrep = int_sph->Nirrep;
    self->size_basis_spinor = int_sph->size_gtou_spinor;

    self->occNumber = vvecd_new(self->Nirrep);
    self->occMax_irrep = 0;
    dhf_sph_setOCC(self, filename, int_sph->atomName);

    if (allInt)
    {
        self->occMax_irrep = self->Nirrep;
    }
    self->occMax_irrep_compact =
        self->irrep_list[self->occMax_irrep - 1].l * 2 + 1;
    self->Nirrep_compact = self->irrep_list[self->Nirrep - 1].l * 2 + 1;
    self->compact2all = (int*)calloc((size_t)self->Nirrep_compact, sizeof(int));
    self->all2compact = (int*)calloc((size_t)self->Nirrep, sizeof(int));
    for (int ir = 0; ir < self->Nirrep; ir++)
    {
        if (self->irrep_list[ir].two_j - 2 * self->irrep_list[ir].l > 0)
            self->all2compact[ir] = 2 * self->irrep_list[ir].l;
        else
            self->all2compact[ir] = 2 * self->irrep_list[ir].l - 1;
    }
    {
        int tmp_i = 0;
        for (int ir = 0; ir < self->Nirrep;
             ir += self->irrep_list[ir].two_j + 1)
        {
            self->compact2all[tmp_i] = ir;
            tmp_i++;
        }
    }

    self->nelec = 0.0;
    for (int ii = 0; ii < self->Nirrep; ii++)
        for (int jj = 0; jj < self->occNumber.v[ii].n; jj++)
            self->nelec += self->occNumber.v[ii].d[jj];

    if (self->printLevel >= 4)
    {
        printf("Occupation number vector:\n");
        printf("l\t2j\t2mj\tOcc\n");
        for (int ii = 0; ii < self->Nirrep; ii++)
        {
            printf("%d\t%d\t%d\t", self->irrep_list[ii].l,
                   self->irrep_list[ii].two_j, self->irrep_list[ii].two_mj);
            for (int kk = 0; kk < self->occNumber.v[ii].n; kk++)
                printf("%g ", self->occNumber.v[ii].d[kk]);
            printf("\n");
        }
        printf("Highest occupied irrep: %d\n", self->occMax_irrep);
        printf("Total number of electrons: %g\n\n", self->nelec);
    }

    /* approximate maximum memory cost for SCF-amfi integrals */
    {
        double numberOfDouble = 0;
        for (int ir = 0; ir < self->Nirrep;
             ir += self->irrep_list[ir].two_j + 1)
            for (int jr = 0; jr < self->Nirrep;
                 jr += self->irrep_list[jr].two_j + 1)
            {
                numberOfDouble += 2.0 * self->irrep_list[ir].size *
                                  self->irrep_list[ir].size *
                                  self->irrep_list[jr].size *
                                  self->irrep_list[jr].size;
            }
        numberOfDouble *= 5.0;
        if (with_gaunt) numberOfDouble = numberOfDouble / 5.0 * 9.0;
        if (with_gauge) numberOfDouble = numberOfDouble / 9.0 * 12.0;
        if (self->printLevel >= 1)
            printf("Maximum memory cost (2e part) in SCF and amfi "
                   "calculation: %g GB.\n",
                   numberOfDouble * sizeof(double) / pow(1024.0, 3));
    }

    countTime();
    self->overlap = int_sph_get_h1e(int_sph, "overlap");
    self->kinetic = int_sph_get_h1e(int_sph, "kinetic");
    if (gaussian_nuc)
    {
        printf("Using Gaussian nuclear model!\n\n");
        self->Vnuc = int_sph_get_h1e(int_sph, "nucGau_attra");
        if (spinFree)
            self->WWW = int_sph_get_h1e(int_sph, "s_p_nucGau_s_p_sf");
        else
            self->WWW = int_sph_get_h1e(int_sph, "s_p_nucGau_s_p");
    }
    else
    {
        self->Vnuc = int_sph_get_h1e(int_sph, "nuc_attra");
        if (spinFree)
            self->WWW = int_sph_get_h1e(int_sph, "s_p_nuc_s_p_sf");
        else
            self->WWW = int_sph_get_h1e(int_sph, "s_p_nuc_s_p");
    }
    countTime();
    if (self->printLevel >= 1) printTime("1e-integrals");

    countTime();
    if (twoC)
        self->h2eLLLL_JK = int_sph_get_h2e_JK_compact(
            int_sph, "LLLL", self->irrep_list[self->occMax_irrep - 1].l);
    else
        int_sph_get_h2e_JK_direct(int_sph, &self->h2eLLLL_JK, &self->h2eSSLL_JK,
                                  &self->h2eSSSS_JK,
                                  self->irrep_list[self->occMax_irrep - 1].l,
                                  spinFree);
    countTime();
    if (self->printLevel >= 1) printTime("2e-Coulomb-integrals");

    if (with_gaunt && !twoC)
    {
        countTime();
        /* Always calculate all Gaunt integrals for amfi integrals */
        if (spinFree)
        {
            printf("ATTENTION! Spin-free Gaunt integrals are used!\n");
            self->gauntLSLS_JK =
                int_sph_get_h2e_JK_gauntSF_compact(int_sph, "LSLS", -1);
            self->gauntLSSL_JK =
                int_sph_get_h2e_JK_gauntSF_compact(int_sph, "LSSL", -1);
        }
        else
            int_sph_get_h2e_JK_gaunt_direct(int_sph, &self->gauntLSLS_JK,
                                            &self->gauntLSSL_JK, -1, false);
        countTime();
        if (self->printLevel >= 1) printTime("2e-Gaunt-integrals");
    }
    if (with_gauge && !twoC)
    {
        if (!with_gaunt)
        {
            printf("ERROR: When gauge term is included, the Gaunt term must "
                   "be included.\n");
            exit(99);
        }
        int2eJK tmp1, tmp2;
        int_sph_get_h2e_JK_gauge_direct(int_sph, &tmp1, &tmp2, -1, false);
        for (int ir = 0; ir < self->Nirrep_compact; ir++)
            for (int jr = 0; jr < self->Nirrep_compact; jr++)
            {
                int size_i = self->irrep_list[self->compact2all[ir]].size,
                    size_j = self->irrep_list[self->compact2all[jr]].size;
                for (int mm = 0; mm < size_i; mm++)
                    for (int nn = 0; nn < size_i; nn++)
                        for (int ss = 0; ss < size_j; ss++)
                            for (int rr = 0; rr < size_j; rr++)
                            {
                                int emn = mm * size_i + nn,
                                    esr = ss * size_j + rr,
                                    emr = mm * size_j + rr,
                                    esn = ss * size_i + nn;
                                self->gauntLSLS_JK.J[ir][jr][emn][esr] -=
                                    tmp1.J[ir][jr][emn][esr];
                                self->gauntLSLS_JK.K[ir][jr][emr][esn] -=
                                    tmp1.K[ir][jr][emr][esn];
                                self->gauntLSSL_JK.J[ir][jr][emn][esr] -=
                                    tmp2.J[ir][jr][emn][esr];
                                self->gauntLSSL_JK.K[ir][jr][emr][esn] -=
                                    tmp2.K[ir][jr][emr][esn];
                            }
            }
        {
            int* sizes = (int*)malloc((size_t)self->Nirrep_compact *
                                      sizeof(int));
            for (int ir = 0; ir < self->Nirrep_compact; ir++)
                sizes[ir] = self->irrep_list[self->compact2all[ir]].size;
            int2e_free(&tmp1, sizes, self->Nirrep_compact);
            int2e_free(&tmp2, sizes, self->Nirrep_compact);
            free(sizes);
        }
        countTime();
        if (self->printLevel >= 1) printTime("2e-gauge-integrals");
    }
    dhf_sph_symmetrize_h2e(self, twoC);

    self->fock_4c = vmat_new(self->occMax_irrep);
    self->h1e_4c = vmat_new(self->occMax_irrep);
    self->overlap_4c = vmat_new(self->occMax_irrep);
    self->overlap_half_i_4c = vmat_new(self->occMax_irrep);
    self->density = vmat_new(self->occMax_irrep);
    self->coeff = vmat_new(self->occMax_irrep);
    self->ene_orb = vvecd_new(self->occMax_irrep);
    self->x2cXXX = vmat_new(self->Nirrep);
    self->x2cRRR = vmat_new(self->Nirrep);
    if (!twoC)
    {
        /*
            overlap_4c = [[S, 0], [0, T/2c^2]]
            h1e_4c = [[V, T], [T, W/4c^2 - T]]
        */
        for (int ii = 0; ii < self->occMax_irrep; ii++)
        {
            int size_tmp = self->irrep_list[ii].size;
            mat_assign(&self->fock_4c.m[ii], mat_new(size_tmp * 2, size_tmp * 2));
            mat_assign(&self->h1e_4c.m[ii], mat_new(size_tmp * 2, size_tmp * 2));
            mat_assign(&self->overlap_4c.m[ii],
                       mat_new(size_tmp * 2, size_tmp * 2));
            for (int mm = 0; mm < size_tmp; mm++)
                for (int nn = 0; nn < size_tmp; nn++)
                {
                    M(self->overlap_4c.m[ii], mm, nn) =
                        M(self->overlap.m[ii], mm, nn);
                    M(self->overlap_4c.m[ii], size_tmp + mm, nn) = 0.0;
                    M(self->overlap_4c.m[ii], mm, size_tmp + nn) = 0.0;
                    M(self->overlap_4c.m[ii], size_tmp + mm, size_tmp + nn) =
                        M(self->kinetic.m[ii], mm, nn) / 2.0 / speedOfLight /
                        speedOfLight;
                    M(self->h1e_4c.m[ii], mm, nn) = M(self->Vnuc.m[ii], mm, nn);
                    M(self->h1e_4c.m[ii], size_tmp + mm, nn) =
                        M(self->kinetic.m[ii], mm, nn);
                    M(self->h1e_4c.m[ii], mm, size_tmp + nn) =
                        M(self->kinetic.m[ii], mm, nn);
                    M(self->h1e_4c.m[ii], size_tmp + mm, size_tmp + nn) =
                        M(self->WWW.m[ii], mm, nn) / 4.0 / speedOfLight /
                            speedOfLight -
                        M(self->kinetic.m[ii], mm, nn);
                }
            mat_assign(&self->overlap_half_i_4c.m[ii],
                       matrix_half_inverse(self->overlap_4c.m[ii]));
        }
    }
    else
    {
        /*
            In X2C-1e, h1e_4c, fock_4c, overlap_4c, and overlap_half_i_4c are
            the corresponding 2-c matrices.

            h1e_4c = [[V, T], [T, W_sf/4c^2 - T]]
        */
        for (int ir = 0; ir < self->occMax_irrep; ir++)
        {
            mat_assign(&self->fock_4c.m[ir],
                       mat_new(self->irrep_list[ir].size,
                               self->irrep_list[ir].size));
            mat_assign(&self->x2cXXX.m[ir],
                       X2C_get_X(self->overlap.m[ir], self->kinetic.m[ir],
                                 self->WWW.m[ir], self->Vnuc.m[ir]));
            mat_assign(&self->x2cRRR.m[ir],
                       X2C_get_R(self->overlap.m[ir], self->kinetic.m[ir],
                                 self->x2cXXX.m[ir]));
            mat_assign(&self->h1e_4c.m[ir],
                       X2C_evaluate_h1e_x2c(self->overlap.m[ir],
                                            self->kinetic.m[ir],
                                            self->WWW.m[ir], self->Vnuc.m[ir],
                                            self->x2cXXX.m[ir],
                                            self->x2cRRR.m[ir]));
            mat_assign(&self->overlap_4c.m[ir], mat_clone(self->overlap.m[ir]));
            mat_assign(&self->overlap_half_i_4c.m[ir],
                       matrix_half_inverse(self->overlap_4c.m[ir]));
        }
    }

    return self;
}

void dhf_sph_free(DHF_SPH* self)
{
    if (self == NULL) return;

    /* per-compact sizes for the int2eJK frees */
    int* sizes = NULL;
    if (self->Nirrep_compact > 0 && self->compact2all != NULL)
    {
        sizes = (int*)malloc((size_t)self->Nirrep_compact * sizeof(int));
        for (int ir = 0; ir < self->Nirrep_compact; ir++)
            sizes[ir] = self->irrep_list[self->compact2all[ir]].size;
    }
    if (sizes != NULL)
    {
        int2e_free(&self->h2eLLLL_JK, sizes, self->Nirrep_compact);
        int2e_free(&self->h2eSSLL_JK, sizes, self->Nirrep_compact);
        int2e_free(&self->h2eSSSS_JK, sizes, self->Nirrep_compact);
        int2e_free(&self->gauntLSLS_JK, sizes, self->Nirrep_compact);
        int2e_free(&self->gauntLSSL_JK, sizes, self->Nirrep_compact);
        free(sizes);
    }

    vmat_free(&self->overlap);
    vmat_free(&self->kinetic);
    vmat_free(&self->WWW);
    vmat_free(&self->Vnuc);
    vmat_free(&self->density);
    vmat_free(&self->fock_4c);
    vmat_free(&self->h1e_4c);
    vmat_free(&self->overlap_4c);
    vmat_free(&self->overlap_half_i_4c);
    vmat_free(&self->x2cXXX);
    vmat_free(&self->x2cRRR);
    vmat_free(&self->coeff);
    vvecd_free(&self->norm_s);
    vvecd_free(&self->occNumber);
    vvecd_free(&self->ene_orb);

    free(self->all2compact);
    free(self->compact2all);

    if (self->ca != NULL)
    {
        DHF_CA_EXT* ca = self->ca;
        free(ca->NN_list);
        free(ca->MM_list);
        free(ca->f_list);
        if (ca->occNumberShells != NULL)
        {
            for (int ii = 0; ii < ca->n_occShells; ii++)
                vvecd_free(&ca->occNumberShells[ii]);
            free(ca->occNumberShells);
        }
        if (ca->densityShells != NULL)
        {
            for (int ii = 0; ii < ca->n_denShells; ii++)
                vmat_free(&ca->densityShells[ii]);
            free(ca->densityShells);
        }
        free(self->ca);
    }

    free(self);
}

/* ----------------------------------------------------------------------
   h2e symmetrization
   ---------------------------------------------------------------------- */
void dhf_sph_symmetrize_h2e(DHF_SPH* self, bool twoC)
{
    if (twoC)
    {
        dhf_sph_symmetrize_JK(self, &self->h2eLLLL_JK,
                              self->occMax_irrep_compact);
    }
    else
    {
        dhf_sph_symmetrize_JK(self, &self->h2eLLLL_JK,
                              self->occMax_irrep_compact);
        dhf_sph_symmetrize_JK(self, &self->h2eSSSS_JK,
                              self->occMax_irrep_compact);
        if (self->with_gaunt)
        {
            dhf_sph_symmetrize_JK_gaunt(self, &self->gauntLSLS_JK,
                                        self->Nirrep_compact);
        }
    }

    return;
}
void dhf_sph_symmetrize_JK(DHF_SPH* self, int2eJK* h2e, int Ncompact)
{
    for (int ir = 0; ir < Ncompact; ir++)
        for (int jr = 0; jr < Ncompact; jr++)
        {
            int size_i = self->irrep_list[self->compact2all[ir]].size,
                size_j = self->irrep_list[self->compact2all[jr]].size;
            double tmpJ[size_i * size_i][size_j * size_j];
            for (int mm = 0; mm < size_i; mm++)
                for (int nn = 0; nn < size_i; nn++)
                    for (int ss = 0; ss < size_j; ss++)
                        for (int rr = 0; rr < size_j; rr++)
                        {
                            int emn = mm * size_i + nn, esr = ss * size_j + rr,
                                emr = mm * size_j + rr, esn = ss * size_i + nn;
                            tmpJ[emn][esr] = h2e->J[ir][jr][emn][esr] -
                                             h2e->K[ir][jr][emr][esn];
                        }
            for (int ii = 0; ii < size_i * size_i; ii++)
                for (int jj = 0; jj < size_j * size_j; jj++)
                    h2e->J[ir][jr][ii][jj] = tmpJ[ii][jj];
        }
    return;
}
void dhf_sph_symmetrize_JK_gaunt(DHF_SPH* self, int2eJK* h2e, int Ncompact)
{
    for (int ir = 0; ir < Ncompact; ir++)
        for (int jr = 0; jr < Ncompact; jr++)
        {
            int size_i = self->irrep_list[self->compact2all[ir]].size,
                size_j = self->irrep_list[self->compact2all[jr]].size;
            double tmpJ1[size_i * size_i][size_j * size_j];
            for (int nn = 0; nn < size_i; nn++)
                for (int mm = 0; mm < size_i; mm++)
                    for (int rr = 0; rr < size_j; rr++)
                        for (int ss = 0; ss < size_j; ss++)
                        {
                            int enm = nn * size_i + mm, ers = rr * size_j + ss,
                                erm = rr * size_i + mm, ens = nn * size_j + ss;
                            tmpJ1[enm][ers] = h2e->J[ir][jr][enm][ers] -
                                              h2e->K[jr][ir][erm][ens];
                        }
            for (int ii = 0; ii < size_i * size_i; ii++)
                for (int jj = 0; jj < size_j * size_j; jj++)
                {
                    h2e->J[ir][jr][ii][jj] = tmpJ1[ii][jj];
                }
        }
    return;
}

/* ----------------------------------------------------------------------
   generalized eigen solver / change evaluation over irreps
   ---------------------------------------------------------------------- */
void dhf_sph_eigensolverG_irrep(DHF_SPH* self, const vmat_t* inputM,
                                const vmat_t* s_h_i, vvecd_t* values,
                                vmat_t* vectors)
{
    for (int ir = 0; ir < self->occMax_irrep;
         ir += self->irrep_list[ir].two_j + 1)
    {
        if (values->v[ir].n != inputM->m[ir].rows)
        {
            vecd_free(&values->v[ir]);
            values->v[ir] = vecd_new(inputM->m[ir].rows);
        }
        eigensolverG(inputM->m[ir], s_h_i->m[ir], values->v[ir].d,
                     &vectors->m[ir]);
    }
    return;
}

double dhf_sph_evaluateChange_irrep(DHF_SPH* self, const vmat_t* M1,
                                    const vmat_t* M2)
{
    vecd_t vecd_tmp = vecd_new(self->occMax_irrep);
    for (int ir = 0; ir < self->occMax_irrep;
         ir += self->irrep_list[ir].two_j + 1)
    {
        vecd_tmp.d[ir] = evaluateChange(M1->m[ir], M2->m[ir]);
    }
    double maxval = vecd_tmp.d[0];
    for (int ii = 1; ii < vecd_tmp.n; ii++)
        if (vecd_tmp.d[ii] > maxval) maxval = vecd_tmp.d[ii];
    vecd_free(&vecd_tmp);
    return maxval;
}

/* ----------------------------------------------------------------------
   DIIS error matrices
   ---------------------------------------------------------------------- */
mat_t dhf_sph_evaluateErrorDIIS_fds(mat_t fock_, mat_t overlap_,
                                    mat_t density_)
{
    mat_t fd = mat_mul(fock_, density_);
    mat_t fds = mat_mul(fd, overlap_);
    mat_t od = mat_mul(overlap_, density_);
    mat_t odf = mat_mul(od, fock_);
    mat_t tmp = mat_sub(fds, odf);
    mat_free(&fd);
    mat_free(&fds);
    mat_free(&od);
    mat_free(&odf);
    int size = fock_.rows;
    mat_t err = mat_new(size * size, 1);
    for (int ii = 0; ii < size; ii++)
        for (int jj = 0; jj < size; jj++)
        {
            M(err, ii * size + jj, 0) = M(tmp, ii, jj);
        }
    mat_free(&tmp);
    return err;
}
mat_t dhf_sph_evaluateErrorDIIS_den(mat_t den_old, mat_t den_new)
{
    mat_t tmp = mat_sub(den_old, den_new);
    int size = den_old.rows;
    mat_t err = mat_new(size * size, 1);
    for (int ii = 0; ii < size; ii++)
        for (int jj = 0; jj < size; jj++)
        {
            M(err, ii * size + jj, 0) = M(tmp, ii, jj);
        }
    mat_free(&tmp);
    return err;
}

/* ----------------------------------------------------------------------
   SCF
   ---------------------------------------------------------------------- */
void dhf_sph_run_scf(DHF_SPH* self, bool twoC, bool renormSmall)
{
    if (self->ca != NULL)
        dhf_sph_ca_run_scf(self, twoC, renormSmall);
    else
        dhf_sph_run_scf_base(self, twoC, renormSmall);
}

void dhf_sph_run_scf_base(DHF_SPH* self, bool twoC, bool renormSmall)
{
    if (renormSmall && !twoC)
    {
        dhf_sph_renormalize_small(self);
    }
    /* per-irrep DIIS history: fixed capacity size_DIIS, with a running count */
    int occMax_irrep = self->occMax_irrep;
    mat_t* error4DIIS = (mat_t*)calloc(
        (size_t)occMax_irrep * (size_t)self->size_DIIS, sizeof(mat_t));
    mat_t* fock4DIIS = (mat_t*)calloc(
        (size_t)occMax_irrep * (size_t)self->size_DIIS, sizeof(mat_t));
    int* nDIIS = (int*)calloc((size_t)occMax_irrep, sizeof(int));

    countTime();
    if (self->printLevel >= 1)
    {
        printf("\n");
        if (twoC)
            printf("Start X2C-1e Hartree-Fock iterations...\n");
        else
            printf("Start Dirac Hartree-Fock iterations...\n");
        printf("with SCF convergence = %g\n", self->convControl);
        printf("\n");
    }

    vmat_t newDen;
    newDen.n = 0;
    newDen.m = NULL;
    {
        dhf_sph_eigensolverG_irrep(self, &self->h1e_4c,
                                   &self->overlap_half_i_4c, &self->ene_orb,
                                   &self->coeff);
        vmat_t d = dhf_sph_evaluateDensity_spinor_irrep(self, twoC);
        vmat_free(&self->density);
        self->density = d;
    }

    for (int iter = 1; iter <= self->maxIter; iter++)
    {
        if (iter <= 2)
        {
            for (int ir = 0; ir < occMax_irrep;
                 ir += self->irrep_list[ir].two_j + 1)
            {
                int size_tmp = self->irrep_list[ir].size;
                dhf_sph_evaluateFock(self, &self->fock_4c.m[ir], twoC,
                                     &self->density, size_tmp, ir);
            }
        }
        else
        {
            int tmp_size = nDIIS[0];
            mat_t B4DIIS = mat_new(tmp_size + 1, tmp_size + 1);
            double* vec_b = (double*)calloc((size_t)(tmp_size + 1),
                                            sizeof(double));
            for (int ii = 0; ii < tmp_size; ii++)
            {
                for (int jj = 0; jj <= ii; jj++)
                {
                    M(B4DIIS, ii, jj) = 0.0;
                    for (int ir = 0; ir < occMax_irrep;
                         ir += self->irrep_list[ir].two_j + 1)
                    {
                        mat_t ei = error4DIIS[ir * self->size_DIIS + ii];
                        mat_t ej = error4DIIS[ir * self->size_DIIS + jj];
                        mat_t eit = mat_transpose(ei);
                        mat_t prod = mat_mul(eit, ej);
                        M(B4DIIS, ii, jj) += M(prod, 0, 0);
                        mat_free(&eit);
                        mat_free(&prod);
                    }
                    M(B4DIIS, jj, ii) = M(B4DIIS, ii, jj);
                }
                M(B4DIIS, tmp_size, ii) = -1.0;
                M(B4DIIS, ii, tmp_size) = -1.0;
                vec_b[ii] = 0.0;
            }
            M(B4DIIS, tmp_size, tmp_size) = 0.0;
            vec_b[tmp_size] = -1.0;
            double* C = (double*)calloc((size_t)(tmp_size + 1), sizeof(double));
            mat_solve_vec(B4DIIS, vec_b, C);
            for (int ir = 0; ir < occMax_irrep;
                 ir += self->irrep_list[ir].two_j + 1)
            {
                mat_assign(&self->fock_4c.m[ir],
                           mat_new(self->fock_4c.m[ir].rows,
                                   self->fock_4c.m[ir].cols));
                for (int ii = 0; ii < tmp_size; ii++)
                {
                    mat_add_inplace(self->fock_4c.m[ir],
                                    fock4DIIS[ir * self->size_DIIS + ii],
                                    C[ii]);
                }
            }
            mat_free(&B4DIIS);
            free(vec_b);
            free(C);
        }

        dhf_sph_eigensolverG_irrep(self, &self->fock_4c,
                                   &self->overlap_half_i_4c, &self->ene_orb,
                                   &self->coeff);
        vmat_free(&newDen);
        newDen = dhf_sph_evaluateDensity_spinor_irrep(self, twoC);
        self->d_density =
            dhf_sph_evaluateChange_irrep(self, &self->density, &newDen);

        if (self->printLevel >= 4)
            printf("Iter #%d maximum density difference: %g\n", iter,
                   self->d_density);

        /* density = newDen; (deep copy ownership transfer) */
        {
            vmat_t old = self->density;
            self->density = vmat_new(newDen.n);
            for (int ii = 0; ii < newDen.n; ii++)
                self->density.m[ii] = mat_clone(newDen.m[ii]);
            vmat_free(&old);
        }
        if (self->d_density < self->convControl || fabs(self->nelec - 1) < 1e-5)
        {
            /* Special case for H atom */
            if (fabs(self->nelec - 1) < 1e-5)
            {
                printf(
                    "Special treatment for fractional occupation of H atom.\n");
                dhf_sph_eigensolverG_irrep(self, &self->h1e_4c,
                                           &self->overlap_half_i_4c,
                                           &self->ene_orb, &self->coeff);
                vmat_t d = dhf_sph_evaluateDensity_spinor_irrep(self, twoC);
                vmat_free(&self->density);
                self->density = d;
            }
            self->converged = true;
            if (self->printLevel >= 1)
                printf("\nSCF converges after %d iterations.\n\n", iter);

            if (self->printLevel >= 4)
            {
                printf("\tOrbital\t\tEnergy(in hartree)\n");
                printf("\t*******\t\t******************\n");
                for (int ir = 0; ir < occMax_irrep;
                     ir += self->irrep_list[ir].two_j + 1)
                    for (int ii = 1; ii <= self->irrep_list[ir].size; ii++)
                    {
                        if (twoC)
                            printf("\t%d\t\t%.15g\n", ii,
                                   self->ene_orb.v[ir].d[ii - 1]);
                        else
                            printf("\t%d\t\t%.15g\n", ii,
                                   self->ene_orb.v[ir]
                                       .d[self->irrep_list[ir].size + ii - 1]);
                    }
            }

            self->ene_scf = 0.0;
            for (int ir = 0; ir < occMax_irrep;
                 ir += self->irrep_list[ir].two_j + 1)
            {
                int size_tmp = self->irrep_list[ir].size;
                if (twoC)
                {
                    for (int ii = 0; ii < size_tmp; ii++)
                        for (int jj = 0; jj < size_tmp; jj++)
                        {
                            self->ene_scf +=
                                0.5 * M(self->density.m[ir], ii, jj) *
                                (M(self->h1e_4c.m[ir], jj, ii) +
                                 M(self->fock_4c.m[ir], jj, ii)) *
                                (self->irrep_list[ir].two_j + 1.0);
                        }
                }
                else
                {
                    for (int ii = 0; ii < size_tmp * 2; ii++)
                        for (int jj = 0; jj < size_tmp * 2; jj++)
                        {
                            self->ene_scf +=
                                0.5 * M(self->density.m[ir], ii, jj) *
                                (M(self->h1e_4c.m[ir], jj, ii) +
                                 M(self->fock_4c.m[ir], jj, ii)) *
                                (self->irrep_list[ir].two_j + 1.0);
                        }
                }
            }
            if (twoC)
                printf("Final X2C-1e HF energy is %.15g hartree.\n",
                       self->ene_scf);
            else
                printf("Final DHF energy is %.15g hartree.\n", self->ene_scf);
            break;
        }
        for (int ir = 0; ir < occMax_irrep;
             ir += self->irrep_list[ir].two_j + 1)
        {
            int size_tmp = self->irrep_list[ir].size;
            dhf_sph_evaluateFock(self, &self->fock_4c.m[ir], twoC,
                                 &self->density, size_tmp, ir);
            if (nDIIS[ir] >= self->size_DIIS)
            {
                /* erase the oldest, shift, push_back */
                mat_free(&error4DIIS[ir * self->size_DIIS + 0]);
                mat_free(&fock4DIIS[ir * self->size_DIIS + 0]);
                for (int kk = 0; kk < self->size_DIIS - 1; kk++)
                {
                    error4DIIS[ir * self->size_DIIS + kk] =
                        error4DIIS[ir * self->size_DIIS + kk + 1];
                    fock4DIIS[ir * self->size_DIIS + kk] =
                        fock4DIIS[ir * self->size_DIIS + kk + 1];
                }
                error4DIIS[ir * self->size_DIIS + self->size_DIIS - 1] =
                    dhf_sph_evaluateErrorDIIS_fds(self->fock_4c.m[ir],
                                                  self->overlap_4c.m[ir],
                                                  self->density.m[ir]);
                fock4DIIS[ir * self->size_DIIS + self->size_DIIS - 1] =
                    mat_clone(self->fock_4c.m[ir]);
            }
            else
            {
                error4DIIS[ir * self->size_DIIS + nDIIS[ir]] =
                    dhf_sph_evaluateErrorDIIS_fds(self->fock_4c.m[ir],
                                                  self->overlap_4c.m[ir],
                                                  self->density.m[ir]);
                fock4DIIS[ir * self->size_DIIS + nDIIS[ir]] =
                    mat_clone(self->fock_4c.m[ir]);
                nDIIS[ir]++;
            }
        }
    }

    for (int ir = 0; ir < occMax_irrep;
         ir += self->irrep_list[ir].two_j + 1)
    {
        for (int jj = 1; jj < self->irrep_list[ir].two_j + 1; jj++)
        {
            mat_assign(&self->fock_4c.m[ir + jj],
                       mat_clone(self->fock_4c.m[ir]));
            vecd_free(&self->ene_orb.v[ir + jj]);
            self->ene_orb.v[ir + jj] = vecd_new(self->ene_orb.v[ir].n);
            for (int kk = 0; kk < self->ene_orb.v[ir].n; kk++)
                self->ene_orb.v[ir + jj].d[kk] = self->ene_orb.v[ir].d[kk];
            mat_assign(&self->coeff.m[ir + jj], mat_clone(self->coeff.m[ir]));
            mat_assign(&self->density.m[ir + jj],
                       mat_clone(self->density.m[ir]));
        }
    }

    countTime();
    if (self->printLevel >= 1) printTime("DHF iterations");

    /* clean up DIIS history */
    for (int ir = 0; ir < occMax_irrep; ir++)
        for (int kk = 0; kk < self->size_DIIS; kk++)
        {
            mat_free(&error4DIIS[ir * self->size_DIIS + kk]);
            mat_free(&fock4DIIS[ir * self->size_DIIS + kk]);
        }
    free(error4DIIS);
    free(fock4DIIS);
    free(nDIIS);
    vmat_free(&newDen);
}

/* ----------------------------------------------------------------------
   small-component renormalization
   ---------------------------------------------------------------------- */
void dhf_sph_renormalize_small(DHF_SPH* self)
{
    self->norm_s = vvecd_new(self->occMax_irrep);
    for (int ii = 0; ii < self->occMax_irrep; ii++)
    {
        self->norm_s.v[ii] = vecd_new(self->irrep_list[ii].size);
        for (int jj = 0; jj < self->irrep_list[ii].size; jj++)
        {
            self->norm_s.v[ii].d[jj] = sqrt(M(self->kinetic.m[ii], jj, jj) /
                                            2.0 / speedOfLight / speedOfLight);
        }
    }
    if (self->printLevel >= 4)
    {
        printf("Renormalizing small component....\n");
        printf("overlap_4c, h1e_4c, overlap_half_i_4c,\n");
        printf("and all h2e will be renormalized.\n\n");
    }

    for (int ii = 0; ii < self->occMax_irrep; ii++)
    {
        int size_tmp = self->irrep_list[ii].size;
        for (int mm = 0; mm < size_tmp; mm++)
            for (int nn = 0; nn < size_tmp; nn++)
            {
                M(self->overlap_4c.m[ii], size_tmp + mm, size_tmp + nn) /=
                    self->norm_s.v[ii].d[mm] * self->norm_s.v[ii].d[nn];
                M(self->h1e_4c.m[ii], size_tmp + mm, nn) /=
                    self->norm_s.v[ii].d[mm];
                M(self->h1e_4c.m[ii], mm, size_tmp + nn) /=
                    self->norm_s.v[ii].d[nn];
                M(self->h1e_4c.m[ii], size_tmp + mm, size_tmp + nn) /=
                    self->norm_s.v[ii].d[mm] * self->norm_s.v[ii].d[nn];
            }
        mat_assign(&self->overlap_half_i_4c.m[ii],
                   matrix_half_inverse(self->overlap_4c.m[ii]));
    }
    for (int ir = 0; ir < self->occMax_irrep_compact; ir++)
        for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
        {
            int Iirrep = self->compact2all[ir], Jirrep = self->compact2all[jr];
            int sizei = self->irrep_list[Iirrep].size,
                sizej = self->irrep_list[Jirrep].size;
            for (int ii = 0; ii < sizei * sizei; ii++)
                for (int jj = 0; jj < sizej * sizej; jj++)
                {
                    int a = ii / sizei, b = ii - a * sizei, c = jj / sizej,
                        d = jj - c * sizej;
                    self->h2eSSLL_JK.J[ir][jr][ii][jj] /=
                        self->norm_s.v[Iirrep].d[a] *
                        self->norm_s.v[Iirrep].d[b];
                    self->h2eSSSS_JK.J[ir][jr][ii][jj] /=
                        self->norm_s.v[Iirrep].d[a] *
                        self->norm_s.v[Iirrep].d[b] *
                        self->norm_s.v[Jirrep].d[c] *
                        self->norm_s.v[Jirrep].d[d];
                }
            for (int ii = 0; ii < sizei * sizej; ii++)
                for (int jj = 0; jj < sizej * sizei; jj++)
                {
                    int a = ii / sizej, b = ii - a * sizej, c = jj / sizei,
                        d = jj - c * sizei;
                    self->h2eSSLL_JK.K[ir][jr][ii][jj] /=
                        self->norm_s.v[Iirrep].d[a] *
                        self->norm_s.v[Jirrep].d[b];
                    self->h2eSSSS_JK.K[ir][jr][ii][jj] /=
                        self->norm_s.v[Iirrep].d[a] *
                        self->norm_s.v[Jirrep].d[b] *
                        self->norm_s.v[Jirrep].d[c] *
                        self->norm_s.v[Iirrep].d[d];
                }
            if (self->with_gaunt)
            {
                for (int ii = 0; ii < sizei * sizei; ii++)
                    for (int jj = 0; jj < sizej * sizej; jj++)
                    {
                        int a = ii / sizei, b = ii - a * sizei, c = jj / sizej,
                            d = jj - c * sizej;
                        (void)a;
                        (void)c;
                        self->gauntLSLS_JK.J[ir][jr][ii][jj] /=
                            self->norm_s.v[Iirrep].d[b] *
                            self->norm_s.v[Jirrep].d[d];
                        self->gauntLSSL_JK.J[ir][jr][ii][jj] /=
                            self->norm_s.v[Iirrep].d[b] *
                            self->norm_s.v[Jirrep].d[c];
                    }
                for (int ii = 0; ii < sizei * sizej; ii++)
                    for (int jj = 0; jj < sizej * sizei; jj++)
                    {
                        int a = ii / sizej, b = ii - a * sizej, c = jj / sizei,
                            d = jj - c * sizei;
                        (void)a;
                        (void)c;
                        self->gauntLSLS_JK.K[ir][jr][ii][jj] /=
                            self->norm_s.v[Jirrep].d[b] *
                            self->norm_s.v[Iirrep].d[d];
                        self->gauntLSSL_JK.K[ir][jr][ii][jj] /=
                            self->norm_s.v[Jirrep].d[b] *
                            self->norm_s.v[Jirrep].d[c];
                    }
            }
        }

    self->renormalizedSmall = true;
}
void dhf_sph_renormalize_h2e(DHF_SPH* self, int2eJK* h2eInput,
                             const char* intType)
{
    for (int ir = 0; ir < self->occMax_irrep_compact; ir++)
        for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
        {
            int Iirrep = self->compact2all[ir], Jirrep = self->compact2all[jr];
            int sizei = self->irrep_list[Iirrep].size,
                sizej = self->irrep_list[Jirrep].size;
            for (int ii = 0; ii < sizei * sizei; ii++)
                for (int jj = 0; jj < sizej * sizej; jj++)
                {
                    int a = ii / sizei, b = ii - a * sizei, c = jj / sizej,
                        d = jj - c * sizej;
                    if (strcmp(intType, "SSLL") == 0)
                        h2eInput->J[ir][jr][ii][jj] /=
                            self->norm_s.v[Iirrep].d[a] *
                            self->norm_s.v[Iirrep].d[b];
                    else if (strcmp(intType, "SSSS") == 0)
                        h2eInput->J[ir][jr][ii][jj] /=
                            self->norm_s.v[Iirrep].d[a] *
                            self->norm_s.v[Iirrep].d[b] *
                            self->norm_s.v[Jirrep].d[c] *
                            self->norm_s.v[Jirrep].d[d];
                    else if (strcmp(intType, "LSLS") == 0)
                        h2eInput->J[ir][jr][ii][jj] /=
                            self->norm_s.v[Iirrep].d[b] *
                            self->norm_s.v[Jirrep].d[d];
                    else if (strcmp(intType, "LSSL") == 0)
                        h2eInput->J[ir][jr][ii][jj] /=
                            self->norm_s.v[Iirrep].d[b] *
                            self->norm_s.v[Jirrep].d[c];
                    else
                    {
                        printf("ERROR: Unkown intType in renormalize_h2e\n");
                        exit(99);
                    }
                }
            for (int ii = 0; ii < sizei * sizej; ii++)
                for (int jj = 0; jj < sizej * sizei; jj++)
                {
                    int a = ii / sizej, b = ii - a * sizej, c = jj / sizei,
                        d = jj - c * sizei;
                    if (strcmp(intType, "SSLL") == 0)
                        h2eInput->K[ir][jr][ii][jj] /=
                            self->norm_s.v[Iirrep].d[a] *
                            self->norm_s.v[Jirrep].d[b];
                    else if (strcmp(intType, "SSSS") == 0)
                        h2eInput->K[ir][jr][ii][jj] /=
                            self->norm_s.v[Iirrep].d[a] *
                            self->norm_s.v[Jirrep].d[b] *
                            self->norm_s.v[Jirrep].d[c] *
                            self->norm_s.v[Iirrep].d[d];
                    else if (strcmp(intType, "LSLS") == 0)
                        h2eInput->K[ir][jr][ii][jj] /=
                            self->norm_s.v[Jirrep].d[b] *
                            self->norm_s.v[Iirrep].d[d];
                    else if (strcmp(intType, "LSSL") == 0)
                        h2eInput->K[ir][jr][ii][jj] /=
                            self->norm_s.v[Jirrep].d[b] *
                            self->norm_s.v[Jirrep].d[c];
                    else
                    {
                        printf("ERROR: Unkown intType in renormalize_h2e\n");
                        exit(99);
                    }
                }
        }
}

/* ----------------------------------------------------------------------
   density evaluation
   ---------------------------------------------------------------------- */
mat_t dhf_sph_evaluateDensity_spinor(mat_t coeff_, vecd_t occNumber_,
                                     bool twoC)
{
    if (!twoC)
    {
        int size = coeff_.cols / 2;
        mat_t den = mat_new(2 * size, 2 * size);
        for (int aa = 0; aa < size; aa++)
            for (int bb = 0; bb < size; bb++)
            {
                for (int ii = 0; ii < occNumber_.n; ii++)
                {
                    M(den, aa, bb) += occNumber_.d[ii] *
                                      M(coeff_, aa, ii + size) *
                                      M(coeff_, bb, ii + size);
                    M(den, size + aa, bb) += occNumber_.d[ii] *
                                             M(coeff_, size + aa, ii + size) *
                                             M(coeff_, bb, ii + size);
                    M(den, aa, size + bb) += occNumber_.d[ii] *
                                             M(coeff_, aa, ii + size) *
                                             M(coeff_, size + bb, ii + size);
                    M(den, size + aa, size + bb) +=
                        occNumber_.d[ii] * M(coeff_, size + aa, ii + size) *
                        M(coeff_, size + bb, ii + size);
                }
            }
        return den;
    }
    else
    {
        int size = coeff_.cols;
        mat_t den = mat_new(size, size);
        for (int aa = 0; aa < size; aa++)
            for (int bb = 0; bb < size; bb++)
                for (int ii = 0; ii < occNumber_.n; ii++)
                    M(den, aa, bb) += occNumber_.d[ii] * M(coeff_, aa, ii) *
                                      M(coeff_, bb, ii);

        return den;
    }
}

vmat_t dhf_sph_evaluateDensity_spinor_irrep(DHF_SPH* self, bool twoC)
{
    vmat_t den = vmat_new(self->occMax_irrep);
    for (int ir = 0; ir < self->occMax_irrep;
         ir += self->irrep_list[ir].two_j + 1)
    {
        mat_assign(&den.m[ir],
                   dhf_sph_evaluateDensity_spinor(
                       self->coeff.m[ir], self->occNumber.v[ir], twoC));
    }

    return den;
}

/* ----------------------------------------------------------------------
   Fock build
   ---------------------------------------------------------------------- */
void dhf_sph_evaluateFock(DHF_SPH* self, mat_t* fock, bool twoC,
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

            M(*fock, mm, nn) = M(self->h1e_4c.m[Iirrep], mm, nn);
            M(*fock, mm + size, nn) = M(self->h1e_4c.m[Iirrep], mm + size, nn);
            if (mm != nn)
                M(*fock, nn + size, mm) =
                    M(self->h1e_4c.m[Iirrep], nn + size, mm);
            M(*fock, mm + size, nn + size) =
                M(self->h1e_4c.m[Iirrep], mm + size, nn + size);
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
                            int enr = nn * size_tmp2 + rr, esm = ss * size + mm;
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
                            int enm = nn * size + mm, ers = rr * size_tmp2 + ss,
                                erm = rr * size + mm, ens = nn * size_tmp2 + ss;
                            (void)erm;
                            (void)ens;
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
                                        M(den->m[Jirrep], size_tmp2 + ss, rr) *
                                        self->gauntLSLS_JK.J[ir][jr][emn][ers] +
                                    twojP1 *
                                        M(den->m[Jirrep], ss, size_tmp2 + rr) *
                                        self->gauntLSSL_JK.J[jr][ir][esr][enm];
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
            M(*fock, mm, nn) = M(self->h1e_4c.m[Iirrep], mm, nn);
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
                        (void)emr;
                        (void)esn;
                        M(*fock, mm, nn) +=
                            twojP1 * M(den->m[Jirrep], ss, rr) *
                            self->h2eLLLL_JK.J[ir][jr][emn][esr];
                    }
            }
            M(*fock, nn, mm) = M(*fock, mm, nn);
        }
    }
}

/* ----------------------------------------------------------------------
   occupation numbers
   ---------------------------------------------------------------------- */
void dhf_sph_setOCC(DHF_SPH* self, const char* filename, const char* atomName)
{
    double vecd_tmp[10];
    for (int ii = 0; ii < 10; ii++) vecd_tmp[ii] = 0.0;
    int int_tmp = 0, int_tmp2;
    bool found = false;
    bool fileFail = false;

    FILE* ifs = fopen(filename, "r");
    if (ifs != NULL)
    {
        char flags[256];
        char target[256];
        sprintf(target, "%%occAMFI_%s", atomName);
        while (fscanf(ifs, "%255s", flags) == 1)
        {
            if (strcmp(flags, target) == 0)
            {
                if (self->printLevel >= 4)
                    printf("Found input occupation number for %s\n", atomName);
                if (fscanf(ifs, "%d", &int_tmp) != 1)
                {
                    fileFail = true;
                    break;
                }
                for (int ii = 0; ii < int_tmp; ii++)
                {
                    if (fscanf(ifs, "%lf", &vecd_tmp[ii]) != 1)
                    {
                        fileFail = true;
                        break;
                    }
                }
                found = true;
                break;
            }
        }
    }
    else
    {
        fileFail = true;
    }

    /* eof / fail / file-missing fallthrough to the default table */
    if (!found || fileFail)
    {
        if (self->printLevel >= 4)
        {
            printf("Did NOT find %%occAMFI in %s\n", filename);
            printf("Using default occupation number for %s\n", atomName);
        }
        if (strcmp(atomName, "H") == 0) {int_tmp = 1; vecd_tmp[0] = 1.0;}
        else if (strcmp(atomName, "HE") == 0) {int_tmp = 1; vecd_tmp[0] = 2.0;}
        else if (strcmp(atomName, "LI") == 0) {int_tmp = 1; vecd_tmp[0] = 3.0;}
        else if (strcmp(atomName, "BE") == 0) {int_tmp = 1; vecd_tmp[0] = 4.0;}
        else if (strcmp(atomName, "B") == 0) {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 1.0/3.0; vecd_tmp[2] = 2.0/3.0;}
        else if (strcmp(atomName, "C") == 0) {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 2.0/3.0; vecd_tmp[2] = 4.0/3.0;}
        else if (strcmp(atomName, "N") == 0) {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 1.0; vecd_tmp[2] = 2.0;}
        else if (strcmp(atomName, "O") == 0) {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 4.0/3.0; vecd_tmp[2] = 8.0/3.0;}
        else if (strcmp(atomName, "F") == 0) {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 5.0/3.0; vecd_tmp[2] = 10.0/3.0;}
        else if (strcmp(atomName, "NE") == 0) {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 2.0; vecd_tmp[2] = 4.0;}
        else if (strcmp(atomName, "NA") == 0) {int_tmp = 3; vecd_tmp[0] = 5.0; vecd_tmp[1] = 2.0; vecd_tmp[2] = 4.0;}
        else if (strcmp(atomName, "MG") == 0) {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 2.0; vecd_tmp[2] = 4.0;}
        else if (strcmp(atomName, "AL") == 0) {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 7.0/3.0; vecd_tmp[2] = 14.0/3.0;}
        else if (strcmp(atomName, "SI") == 0) {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 8.0/3.0; vecd_tmp[2] = 16.0/3.0;}
        else if (strcmp(atomName, "P") == 0) {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 3.0; vecd_tmp[2] = 6.0;}
        else if (strcmp(atomName, "S") == 0) {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 10.0/3.0; vecd_tmp[2] = 20.0/3.0;}
        else if (strcmp(atomName, "CL") == 0) {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 11.0/3.0; vecd_tmp[2] = 22.0/3.0;}
        else if (strcmp(atomName, "AR") == 0) {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0;}
        else if (strcmp(atomName, "K") == 0) {int_tmp = 3; vecd_tmp[0] = 7.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0;}
        else if (strcmp(atomName, "CA") == 0) {int_tmp = 3; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0;}
        else if (strcmp(atomName, "SC") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 1.0/2.5; vecd_tmp[4] = 1.5/2.5;}
        else if (strcmp(atomName, "TI") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 2.0/2.5; vecd_tmp[4] = 3.0/2.5;}
        else if (strcmp(atomName, "V") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 3.0/2.5; vecd_tmp[4] = 4.5/2.5;}
        else if (strcmp(atomName, "CR") == 0) {int_tmp = 5; vecd_tmp[0] = 7.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 2.0; vecd_tmp[4] = 3.0;}
        else if (strcmp(atomName, "MN") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 2.0; vecd_tmp[4] = 3.0;}
        else if (strcmp(atomName, "FE") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 6.0/2.5; vecd_tmp[4] = 9.0/2.5;}
        else if (strcmp(atomName, "CO") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 7.0/2.5; vecd_tmp[4] = 10.5/2.5;}
        else if (strcmp(atomName, "NI") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 8.0/2.5; vecd_tmp[4] = 12.0/2.5;}
        else if (strcmp(atomName, "CU") == 0) {int_tmp = 5; vecd_tmp[0] = 7.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "ZN") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "GA") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 13.0/3.0; vecd_tmp[2] = 26.0/3.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "GE") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 14.0/3.0; vecd_tmp[2] = 28.0/3.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "AS") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 5.0; vecd_tmp[2] = 10.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "SE") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 16.0/3.0; vecd_tmp[2] = 32.0/3.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "BR") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 17.0/3.0; vecd_tmp[2] = 34.0/3.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "KR") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "RB") == 0) {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "SR") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if (strcmp(atomName, "Y") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 11.0/2.5; vecd_tmp[4] = 16.5/2.5;}
        else if (strcmp(atomName, "ZR") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 12.0/2.5; vecd_tmp[4] = 18.0/2.5;}
        else if (strcmp(atomName, "NB") == 0) {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 14.0/2.5; vecd_tmp[4] = 21.0/2.5;}
        else if (strcmp(atomName, "MO") == 0) {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 6.0; vecd_tmp[4] = 9.0;}
        else if (strcmp(atomName, "TC") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 6.0; vecd_tmp[4] = 9.0;}
        else if (strcmp(atomName, "RU") == 0) {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 17.0/2.5; vecd_tmp[4] = 25.5/2.5;}
        else if (strcmp(atomName, "RH") == 0) {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 18.0/2.5; vecd_tmp[4] = 27.0/2.5;}
        else if (strcmp(atomName, "PD") == 0) {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "AG") == 0) {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "CD") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "IN") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 19.0/3.0; vecd_tmp[2] = 38.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "SN") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 20.0/3.0; vecd_tmp[2] = 40.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "SB") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 21.0/3.0; vecd_tmp[2] = 42.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "TE") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 22.0/3.0; vecd_tmp[2] = 44.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "I") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 23.0/3.0; vecd_tmp[2] = 46.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "XE") == 0) {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "CS") == 0) {int_tmp = 5; vecd_tmp[0] = 11.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "BA") == 0) {int_tmp = 5; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if (strcmp(atomName, "LA") == 0) {int_tmp = 5; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 21.0/2.5; vecd_tmp[4] = 31.5/2.5;}
        else if (strcmp(atomName, "CE") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 21.0/2.5; vecd_tmp[4] = 31.5/2.5; vecd_tmp[5] = 3.0/7.0; vecd_tmp[6] = 4.0/7.0;}
        else if (strcmp(atomName, "PR") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 9.0/7.0; vecd_tmp[6] = 12.0/7.0;}
        else if (strcmp(atomName, "ND") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 12.0/7.0; vecd_tmp[6] = 16.0/7.0;}
        else if (strcmp(atomName, "PM") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 15.0/7.0; vecd_tmp[6] = 20.0/7.0;}
        else if (strcmp(atomName, "SM") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 18.0/7.0; vecd_tmp[6] = 24.0/7.0;}
        else if (strcmp(atomName, "EU") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 3.0; vecd_tmp[6] = 4.0;}
        else if (strcmp(atomName, "GD") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 21.0/2.5; vecd_tmp[4] = 31.5/2.5; vecd_tmp[5] = 21.0/7.0; vecd_tmp[6] = 28.0/7.0;}
        else if (strcmp(atomName, "TB") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 27.0/7.0; vecd_tmp[6] = 36.0/7.0;}
        else if (strcmp(atomName, "DY") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 30.0/7.0; vecd_tmp[6] = 40.0/7.0;}
        else if (strcmp(atomName, "HO") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 33.0/7.0; vecd_tmp[6] = 44.0/7.0;}
        else if (strcmp(atomName, "ER") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 36.0/7.0; vecd_tmp[6] = 48.0/7.0;}
        else if (strcmp(atomName, "TM") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 39.0/7.0; vecd_tmp[6] = 52.0/7.0;}
        else if (strcmp(atomName, "YB") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "LU") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 21.0/2.5; vecd_tmp[4] = 31.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "HF") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 22.0/2.5; vecd_tmp[4] = 33.0/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "TA") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 23.0/2.5; vecd_tmp[4] = 34.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "W") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 24.0/2.5; vecd_tmp[4] = 36.0/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "RE") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 10.0; vecd_tmp[4] = 15.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "OS") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 26.0/2.5; vecd_tmp[4] = 39.0/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "IR") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 27.0/2.5; vecd_tmp[4] = 40.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "PT") == 0) {int_tmp = 7; vecd_tmp[0] = 11.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 29.0/2.5; vecd_tmp[4] = 43.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "AU") == 0) {int_tmp = 7; vecd_tmp[0] = 11.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "HG") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "TL") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 25.0/3.0; vecd_tmp[2] = 50.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "PB") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 26.0/3.0; vecd_tmp[2] = 52.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "BI") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 9.0; vecd_tmp[2] = 18.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "PO") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 28.0/3.0; vecd_tmp[2] = 56.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "AT") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 29.0/3.0; vecd_tmp[2] = 58.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "RN") == 0) {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "FR") == 0) {int_tmp = 7; vecd_tmp[0] = 13.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "RA") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "AC") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "TH") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 32.0/2.5; vecd_tmp[4] = 48.0/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if (strcmp(atomName, "PA") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 48.0/7.0; vecd_tmp[6] = 64.0/7.0;}
        else if (strcmp(atomName, "U") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 51.0/7.0; vecd_tmp[6] = 68.0/7.0;}
        else if (strcmp(atomName, "NP") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 54.0/7.0; vecd_tmp[6] = 72.0/7.0;}
        else if (strcmp(atomName, "PU") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 60.0/7.0; vecd_tmp[6] = 80.0/7.0;}
        else if (strcmp(atomName, "AM") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 63.0/7.0; vecd_tmp[6] = 84.0/7.0;}
        else if (strcmp(atomName, "CM") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 63.0/7.0; vecd_tmp[6] = 84.0/7.0;}
        else if (strcmp(atomName, "BK") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 69.0/7.0; vecd_tmp[6] = 92.0/7.0;}
        else if (strcmp(atomName, "CF") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 72.0/7.0; vecd_tmp[6] = 96.0/7.0;}
        else if (strcmp(atomName, "ES") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 75.0/7.0; vecd_tmp[6] = 100.0/7.0;}
        else if (strcmp(atomName, "FM") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 78.0/7.0; vecd_tmp[6] = 104.0/7.0;}
        else if (strcmp(atomName, "MD") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 81.0/7.0; vecd_tmp[6] = 108.0/7.0;}
        else if (strcmp(atomName, "NO") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "LR") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 31.0/3.0; vecd_tmp[2] = 62.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "RF") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 32.0/2.5; vecd_tmp[4] = 48.0/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "DB") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 33.0/2.5; vecd_tmp[4] = 49.5/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "SG") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 34.0/2.5; vecd_tmp[4] = 51.0/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "BH") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 14.0; vecd_tmp[4] = 21.0; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "HS") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 36.0/2.5; vecd_tmp[4] = 54.0/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "MT") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 37.0/2.5; vecd_tmp[4] = 55.5/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "DS") == 0) {int_tmp = 7; vecd_tmp[0] = 13.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 39.0/2.5; vecd_tmp[4] = 58.5/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "RG") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 39.0/2.5; vecd_tmp[4] = 58.5/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "CN") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 16; vecd_tmp[4] = 24; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "NH") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 31.0/3.0; vecd_tmp[2] = 62.0/3.0; vecd_tmp[3] = 16; vecd_tmp[4] = 24; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "FL") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 32.0/3.0; vecd_tmp[2] = 64.0/3.0; vecd_tmp[3] = 16; vecd_tmp[4] = 24; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "MC") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 33.0/3.0; vecd_tmp[2] = 66.0/3.0; vecd_tmp[3] = 16; vecd_tmp[4] = 24; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "LV") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 34.0/3.0; vecd_tmp[2] = 68.0/3.0; vecd_tmp[3] = 16; vecd_tmp[4] = 24; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "TS") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 35.0/3.0; vecd_tmp[2] = 70.0/3.0; vecd_tmp[3] = 16; vecd_tmp[4] = 24; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if (strcmp(atomName, "OG") == 0) {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 12.0; vecd_tmp[2] = 24.0; vecd_tmp[3] = 16.0; vecd_tmp[4] = 24.0; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else
        {
            printf("ERROR: %s does NOT have a default ON. Please input the "
                   "occupation numbers by hand.\n",
                   atomName);
            exit(99);
        }
    }
    int_tmp2 = 0;
    for (int ii = 0; ii < int_tmp; ii++)
    {
        int int_tmp3 =
            (int)(vecd_tmp[ii] / (self->irrep_list[int_tmp2].two_j + 1));
        double d_tmp =
            (double)(vecd_tmp[ii] -
                     int_tmp3 * (self->irrep_list[int_tmp2].two_j + 1)) /
            (double)(self->irrep_list[int_tmp2].two_j + 1);
        for (int jj = 0; jj < self->irrep_list[int_tmp2].two_j + 1; jj++)
        {
            vecd_free(&self->occNumber.v[int_tmp2 + jj]);
            self->occNumber.v[int_tmp2 + jj] =
                vecd_new(self->irrep_list[int_tmp2 + jj].size);
            for (int kk = 0; kk < int_tmp3; kk++)
            {
                self->occNumber.v[int_tmp2 + jj].d[kk] = 1.0;
            }
            if (self->occNumber.v[int_tmp2 + jj].n > int_tmp3)
                self->occNumber.v[int_tmp2 + jj].d[int_tmp3] = d_tmp;
        }
        self->occMax_irrep += self->irrep_list[int_tmp2].two_j + 1;
        int_tmp2 += self->irrep_list[int_tmp2].two_j + 1;
    }
    if (ifs != NULL) fclose(ifs);

    return;
}

/* ----------------------------------------------------------------------
   amfi SOC integrals
   ---------------------------------------------------------------------- */
vmat_t dhf_sph_get_amfi_unc(DHF_SPH* self, INT_SPH* int_sph, bool twoC,
                            const char* Xmethod, bool amfi_with_gaunt,
                            bool amfi_with_gauge, bool amfi4c, bool sd_gaunt)
{
    if (self->printLevel >= 4) printf("Running DHF_SPH::get_amfi_unc\n");
    if (self->with_gaunt && !amfi_with_gaunt)
    {
        if (self->printLevel >= 4)
            printf("\nATTENTION! Since gaunt terms are included in SCF, they "
                   "are automatically calculated in amfi integrals.\n\n");
        amfi_with_gaunt = true;
        if (self->with_gauge && !amfi_with_gauge) amfi_with_gauge = true;
    }
    if ((!self->with_gaunt && amfi_with_gaunt) || (twoC && amfi_with_gaunt))
    {
        countTime();
        int_sph_get_h2e_JK_gaunt_direct(int_sph, &self->gauntLSLS_JK,
                                        &self->gauntLSSL_JK, -1, false);
        countTime();
        if (self->printLevel >= 1) printTime("2e-Gaunt-integrals");
        if (amfi_with_gauge)
        {
            int2eJK tmp1, tmp2;
            int_sph_get_h2e_JK_gauge_direct(int_sph, &tmp1, &tmp2, -1, false);
            for (int ir = 0; ir < self->Nirrep_compact; ir++)
                for (int jr = 0; jr < self->Nirrep_compact; jr++)
                {
                    int size_i = self->irrep_list[self->compact2all[ir]].size,
                        size_j = self->irrep_list[self->compact2all[jr]].size;
                    for (int mm = 0; mm < size_i; mm++)
                        for (int nn = 0; nn < size_i; nn++)
                            for (int ss = 0; ss < size_j; ss++)
                                for (int rr = 0; rr < size_j; rr++)
                                {
                                    int emn = mm * size_i + nn,
                                        esr = ss * size_j + rr,
                                        emr = mm * size_j + rr,
                                        esn = ss * size_i + nn;
                                    self->gauntLSLS_JK.J[ir][jr][emn][esr] -=
                                        tmp1.J[ir][jr][emn][esr];
                                    self->gauntLSLS_JK.K[ir][jr][emr][esn] -=
                                        tmp1.K[ir][jr][emr][esn];
                                    self->gauntLSSL_JK.J[ir][jr][emn][esr] -=
                                        tmp2.J[ir][jr][emn][esr];
                                    self->gauntLSSL_JK.K[ir][jr][emr][esn] -=
                                        tmp2.K[ir][jr][emr][esn];
                                }
                }
            {
                int* sizes = (int*)malloc((size_t)self->Nirrep_compact *
                                          sizeof(int));
                for (int ir = 0; ir < self->Nirrep_compact; ir++)
                    sizes[ir] = self->irrep_list[self->compact2all[ir]].size;
                int2e_free(&tmp1, sizes, self->Nirrep_compact);
                int2e_free(&tmp2, sizes, self->Nirrep_compact);
                free(sizes);
            }
            countTime();
            if (self->printLevel >= 1) printTime("2e-gauge-integrals");
        }
        dhf_sph_symmetrize_JK_gaunt(self, &self->gauntLSLS_JK,
                                    self->Nirrep_compact);
        if (self->renormalizedSmall)
        {
            dhf_sph_renormalize_h2e(self, &self->gauntLSLS_JK, "LSLS");
            dhf_sph_renormalize_h2e(self, &self->gauntLSSL_JK, "LSSL");
        }
    }
    if (amfi_with_gauge && !amfi_with_gaunt)
    {
        printf("ERROR: When gauge term is included, the Gaunt term must be "
               "included.\n");
        exit(99);
    }
    int2eJK gauntLSLS_SD, gauntLSSL_SD;
    gauntLSLS_SD.J = NULL;
    gauntLSLS_SD.K = NULL;
    gauntLSSL_SD.J = NULL;
    gauntLSSL_SD.K = NULL;
    bool gaunt_SD_owned = false;
    if (amfi_with_gaunt)
    {
        if (sd_gaunt)
        {
            if (amfi_with_gauge)
            {
                printf("ERROR: Gauge term is not supported with SD gaunt "
                       "term.\n");
                exit(99);
            }
            if (self->printLevel >= 4)
                printf("Calculate only SD gaunt terms...\n");
            /* Enable SD gaunt */
            int_sph_get_h2e_JK_gaunt_direct(int_sph, &gauntLSLS_SD,
                                            &gauntLSSL_SD, -1, true);
            gaunt_SD_owned = true;
            if (self->renormalizedSmall)
            {
                dhf_sph_renormalize_h2e(self, &gauntLSLS_SD, "LSLS");
                dhf_sph_renormalize_h2e(self, &gauntLSSL_SD, "LSSL");
            }
            dhf_sph_symmetrize_JK_gaunt(self, &gauntLSLS_SD,
                                        self->Nirrep_compact);
            for (int ir = 0; ir < self->Nirrep_compact; ir++)
                for (int jr = 0; jr < self->Nirrep_compact; jr++)
                {
                    int size_i = self->irrep_list[self->compact2all[ir]].size,
                        size_j = self->irrep_list[self->compact2all[jr]].size;
#pragma omp parallel for
                    for (int nn = 0;
                         nn < size_i * size_i * size_j * size_j; nn++)
                    {
                        int kk = nn / (size_j * size_j),
                            ll = nn - kk * size_j * size_j;
                        gauntLSLS_SD.J[ir][jr][kk][ll] =
                            self->gauntLSLS_JK.J[ir][jr][kk][ll] -
                            gauntLSLS_SD.J[ir][jr][kk][ll];
                        gauntLSSL_SD.J[ir][jr][kk][ll] =
                            self->gauntLSSL_JK.J[ir][jr][kk][ll] -
                            gauntLSSL_SD.J[ir][jr][kk][ll];
                        kk = nn / (size_i * size_j);
                        ll = nn - kk * size_i * size_j;
                        gauntLSLS_SD.K[ir][jr][kk][ll] =
                            self->gauntLSLS_JK.K[ir][jr][kk][ll] -
                            gauntLSLS_SD.K[ir][jr][kk][ll];
                        gauntLSSL_SD.K[ir][jr][kk][ll] =
                            self->gauntLSSL_JK.K[ir][jr][kk][ll] -
                            gauntLSSL_SD.K[ir][jr][kk][ll];
                    }
                }
        }
        else
        {
            gauntLSLS_SD = self->gauntLSLS_JK;
            gauntLSSL_SD = self->gauntLSSL_JK;
        }
    }
    int2eJK SSLL_SD, SSSS_SD;
    countTime();
    int_sph_get_h2eSD_JK_direct(int_sph, &SSLL_SD, &SSSS_SD, -1);
    dhf_sph_symmetrize_JK(self, &SSSS_SD, self->Nirrep_compact);
    if (self->renormalizedSmall)
    {
        dhf_sph_renormalize_h2e(self, &SSLL_SD, "SSLL");
        dhf_sph_renormalize_h2e(self, &SSSS_SD, "SSSS");
    }
    countTime();
    if (self->printLevel >= 1) printTime("2e-SO Coulomb integrals");

    vmat_t result;
    if (twoC)
    {
        result = dhf_sph_get_amfi_unc_2c(self, &SSLL_SD, &SSSS_SD,
                                         amfi_with_gaunt, amfi4c);
    }
    else
    {
        if (self->occMax_irrep < self->Nirrep &&
            strcmp(Xmethod, "fullFock") == 0)
        {
            if (self->printLevel >= 4)
                printf("fullFock is used in amfi function with incomplete "
                       "h2e.\n");
            if (self->printLevel >= 4)
                printf("Recalculate h2e and gaunt2e...\n");
            countTime();
            int_sph_get_h2e_JK_direct(int_sph, &self->h2eLLLL_JK,
                                      &self->h2eSSLL_JK, &self->h2eSSSS_JK, -1,
                                      false);
            dhf_sph_symmetrize_JK(self, &self->h2eLLLL_JK, self->Nirrep_compact);
            dhf_sph_symmetrize_JK(self, &self->h2eSSSS_JK, self->Nirrep_compact);
            if (self->renormalizedSmall)
            {
                dhf_sph_renormalize_h2e(self, &self->h2eSSLL_JK, "SSLL");
                dhf_sph_renormalize_h2e(self, &self->h2eSSSS_JK, "SSSS");
            }
            countTime();
            if (self->printLevel >= 1) printTime("Extra 2e-integrals");
        }
        result = dhf_sph_get_amfi_unc_int2e(
            self, &SSLL_SD, &SSSS_SD, &gauntLSLS_SD, &gauntLSSL_SD,
            &self->density, Xmethod, amfi_with_gaunt, amfi4c);
    }

    /* free the SD integrals we own */
    {
        int* sizes = (int*)malloc((size_t)self->Nirrep_compact * sizeof(int));
        for (int ir = 0; ir < self->Nirrep_compact; ir++)
            sizes[ir] = self->irrep_list[self->compact2all[ir]].size;
        int2e_free(&SSLL_SD, sizes, self->Nirrep_compact);
        int2e_free(&SSSS_SD, sizes, self->Nirrep_compact);
        if (gaunt_SD_owned)
        {
            int2e_free(&gauntLSLS_SD, sizes, self->Nirrep_compact);
            int2e_free(&gauntLSSL_SD, sizes, self->Nirrep_compact);
        }
        free(sizes);
    }

    return result;
}

vmat_t dhf_sph_get_amfi_unc_int2e(DHF_SPH* self, const int2eJK* h2eSSLL_SD,
                                  const int2eJK* h2eSSSS_SD,
                                  const int2eJK* gauntLSLS_SD,
                                  const int2eJK* gauntLSSL_SD,
                                  const vmat_t* density_, const char* Xmethod,
                                  bool amfi_with_gaunt, bool amfi4c)
{
    if (!self->converged)
    {
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("!!  WARNING: Dirac HF did NOT converge  !!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
    vmat_t amfi_unc = vmat_new(self->Nirrep);
    vmat_t SO_4c = vmat_new(self->Nirrep);
    vmat_t h1e_4c_full = vmat_new(self->Nirrep);
    vmat_t overlap_4c_full = vmat_new(self->Nirrep);
    /*
        Construct h1e_4c_full and overlap_4c_full
    */
    for (int ir = 0; ir < self->occMax_irrep; ir++)
    {
        mat_assign(&h1e_4c_full.m[ir], mat_clone(self->h1e_4c.m[ir]));
        mat_assign(&overlap_4c_full.m[ir], mat_clone(self->overlap_4c.m[ir]));
    }
    for (int ir = self->occMax_irrep; ir < self->Nirrep; ir++)
    {
        int size_tmp = self->irrep_list[ir].size;
        mat_assign(&h1e_4c_full.m[ir], mat_new(size_tmp * 2, size_tmp * 2));
        mat_assign(&overlap_4c_full.m[ir], mat_new(size_tmp * 2, size_tmp * 2));
        for (int mm = 0; mm < size_tmp; mm++)
            for (int nn = 0; nn < size_tmp; nn++)
            {
                M(overlap_4c_full.m[ir], mm, nn) = M(self->overlap.m[ir], mm, nn);
                M(overlap_4c_full.m[ir], size_tmp + mm, nn) = 0.0;
                M(overlap_4c_full.m[ir], mm, size_tmp + nn) = 0.0;
                M(overlap_4c_full.m[ir], size_tmp + mm, size_tmp + nn) =
                    M(self->kinetic.m[ir], mm, nn) / 2.0 / speedOfLight /
                    speedOfLight;
                M(h1e_4c_full.m[ir], mm, nn) = M(self->Vnuc.m[ir], mm, nn);
                M(h1e_4c_full.m[ir], size_tmp + mm, nn) =
                    M(self->kinetic.m[ir], mm, nn);
                M(h1e_4c_full.m[ir], mm, size_tmp + nn) =
                    M(self->kinetic.m[ir], mm, nn);
                M(h1e_4c_full.m[ir], size_tmp + mm, size_tmp + nn) =
                    M(self->WWW.m[ir], mm, nn) / 4.0 / speedOfLight /
                        speedOfLight -
                    M(self->kinetic.m[ir], mm, nn);
            }
    }

    for (int ir = 0; ir < self->Nirrep; ir++)
    {
        int ir_c = self->all2compact[ir];
        int size_tmp = self->irrep_list[ir].size;
        mat_assign(&SO_4c.m[ir], mat_new(2 * size_tmp, 2 * size_tmp));
        /*
            Evaluate SO integrals in 4c basis
            The structure is the same as 2e Coulomb integrals in fock matrix
        */
        for (int mm = 0; mm < size_tmp; mm++)
            for (int nn = 0; nn <= mm; nn++)
            {
                M(SO_4c.m[ir], mm, nn) = 0.0;
                M(SO_4c.m[ir], mm + size_tmp, nn) = 0.0;
                if (mm != nn) M(SO_4c.m[ir], nn + size_tmp, mm) = 0.0;
                M(SO_4c.m[ir], mm + size_tmp, nn + size_tmp) = 0.0;
                for (int jr = 0; jr < self->occMax_irrep; jr++)
                {
                    int jr_c = self->all2compact[jr];
                    int size_tmp2 = self->irrep_list[jr].size;
                    for (int ss = 0; ss < size_tmp2; ss++)
                        for (int rr = 0; rr < size_tmp2; rr++)
                        {
                            int emn = mm * size_tmp + nn,
                                esr = ss * size_tmp2 + rr,
                                emr = mm * size_tmp2 + rr,
                                esn = ss * size_tmp + nn;
                            M(SO_4c.m[ir], mm, nn) +=
                                M(density_->m[jr], size_tmp2 + ss,
                                  size_tmp2 + rr) *
                                h2eSSLL_SD->J[jr_c][ir_c][esr][emn];
                            M(SO_4c.m[ir], mm + size_tmp, nn) -=
                                M(density_->m[jr], ss, size_tmp2 + rr) *
                                h2eSSLL_SD->K[ir_c][jr_c][emr][esn];
                            if (mm != nn)
                            {
                                int enr = nn * size_tmp2 + rr,
                                    esm = ss * size_tmp + mm;
                                M(SO_4c.m[ir], nn + size_tmp, mm) -=
                                    M(density_->m[jr], ss, size_tmp2 + rr) *
                                    h2eSSLL_SD->K[ir_c][jr_c][enr][esm];
                            }
                            M(SO_4c.m[ir], mm + size_tmp, nn + size_tmp) +=
                                M(density_->m[jr], size_tmp2 + ss,
                                  size_tmp2 + rr) *
                                    h2eSSSS_SD->J[ir_c][jr_c][emn][esr] +
                                M(density_->m[jr], ss, rr) *
                                    h2eSSLL_SD->J[ir_c][jr_c][emn][esr];
                            if (amfi_with_gaunt)
                            {
                                int enm = nn * size_tmp + mm,
                                    ers = rr * size_tmp2 + ss,
                                    erm = rr * size_tmp + mm,
                                    ens = nn * size_tmp2 + ss;
                                (void)erm;
                                (void)ens;
                                M(SO_4c.m[ir], mm, nn) -=
                                    M(density_->m[jr], size_tmp2 + ss,
                                      size_tmp2 + rr) *
                                    gauntLSSL_SD->K[ir_c][jr_c][emr][esn];
                                M(SO_4c.m[ir], mm + size_tmp,
                                  nn + size_tmp) -=
                                    M(density_->m[jr], ss, rr) *
                                    gauntLSSL_SD->K[jr_c][ir_c][esn][emr];
                                M(SO_4c.m[ir], mm + size_tmp, nn) +=
                                    M(density_->m[jr], size_tmp2 + ss, rr) *
                                        gauntLSLS_SD->J[ir_c][jr_c][enm][ers] +
                                    M(density_->m[jr], ss, size_tmp2 + rr) *
                                        gauntLSSL_SD->J[jr_c][ir_c][esr][emn];
                                if (mm != nn)
                                {
                                    int ern = rr * size_tmp + nn,
                                        ems = mm * size_tmp2 + ss;
                                    (void)ern;
                                    (void)ems;
                                    M(SO_4c.m[ir], nn + size_tmp, mm) +=
                                        M(density_->m[jr], size_tmp2 + ss,
                                          rr) *
                                            gauntLSLS_SD
                                                ->J[ir_c][jr_c][emn][ers] +
                                        M(density_->m[jr], ss,
                                          size_tmp2 + rr) *
                                            gauntLSSL_SD
                                                ->J[jr_c][ir_c][esr][enm];
                                }
                            }
                        }
                }
                M(SO_4c.m[ir], nn, mm) = M(SO_4c.m[ir], mm, nn);
                M(SO_4c.m[ir], nn + size_tmp, mm + size_tmp) =
                    M(SO_4c.m[ir], mm + size_tmp, nn + size_tmp);
                M(SO_4c.m[ir], nn, mm + size_tmp) =
                    M(SO_4c.m[ir], mm + size_tmp, nn);
                M(SO_4c.m[ir], mm, nn + size_tmp) =
                    M(SO_4c.m[ir], nn + size_tmp, mm);
            }
        /*
            Evaluate X with various options
        */
        if (strcmp(Xmethod, "h1e") == 0)
        {
            mat_assign(&self->x2cXXX.m[ir],
                       X2C_get_X(self->overlap.m[ir], self->kinetic.m[ir],
                                 self->WWW.m[ir], self->Vnuc.m[ir]));
        }
        else
        {
            if (ir < self->occMax_irrep)
            {
                mat_assign(&self->x2cXXX.m[ir],
                           X2C_get_X_from_coeff(self->coeff.m[ir]));
            }
            else
            {
                if (strcmp(Xmethod, "partialFock") == 0)
                    mat_assign(&self->x2cXXX.m[ir],
                               X2C_get_X(self->overlap.m[ir],
                                         self->kinetic.m[ir], self->WWW.m[ir],
                                         self->Vnuc.m[ir]));
                else if (strcmp(Xmethod, "fullFock") == 0)
                {
                    mat_t fock_tmp = mat_new(2 * size_tmp, 2 * size_tmp);
                    mat_t overlap_half_i_4c_tmp =
                        matrix_half_inverse(overlap_4c_full.m[ir]);
                    for (int mm = 0; mm < size_tmp; mm++)
                        for (int nn = 0; nn <= mm; nn++)
                        {
                            M(fock_tmp, mm, nn) =
                                M(h1e_4c_full.m[ir], mm, nn);
                            M(fock_tmp, mm + size_tmp, nn) =
                                M(h1e_4c_full.m[ir], mm + size_tmp, nn);
                            if (mm != nn)
                                M(fock_tmp, nn + size_tmp, mm) =
                                    M(h1e_4c_full.m[ir], nn + size_tmp, mm);
                            M(fock_tmp, mm + size_tmp, nn + size_tmp) =
                                M(h1e_4c_full.m[ir], mm + size_tmp,
                                  nn + size_tmp);
                            for (int jr = 0; jr < self->occMax_irrep; jr++)
                            {
                                int jr_c = self->all2compact[jr];
                                int size_tmp2 = self->irrep_list[jr].size;
                                for (int ss = 0; ss < size_tmp2; ss++)
                                    for (int rr = 0; rr < size_tmp2; rr++)
                                    {
                                        int emn = mm * size_tmp + nn,
                                            esr = ss * size_tmp2 + rr,
                                            emr = mm * size_tmp2 + rr,
                                            esn = ss * size_tmp + nn;
                                        M(fock_tmp, mm, nn) +=
                                            M(density_->m[jr], ss, rr) *
                                                self->h2eLLLL_JK
                                                    .J[ir_c][jr_c][emn][esr] +
                                            M(density_->m[jr], size_tmp2 + ss,
                                              size_tmp2 + rr) *
                                                self->h2eSSLL_JK
                                                    .J[jr_c][ir_c][esr][emn];
                                        M(fock_tmp, mm + size_tmp, nn) -=
                                            M(density_->m[jr], ss,
                                              size_tmp2 + rr) *
                                            self->h2eSSLL_JK
                                                .K[ir_c][jr_c][emr][esn];
                                        if (mm != nn)
                                        {
                                            int enr = nn * size_tmp2 + rr,
                                                esm = ss * size_tmp + mm;
                                            M(fock_tmp, nn + size_tmp, mm) -=
                                                M(density_->m[jr], ss,
                                                  size_tmp2 + rr) *
                                                self->h2eSSLL_JK
                                                    .K[ir_c][jr_c][enr][esm];
                                        }
                                        M(fock_tmp, mm + size_tmp,
                                          nn + size_tmp) +=
                                            M(density_->m[jr], size_tmp2 + ss,
                                              size_tmp2 + rr) *
                                                self->h2eSSSS_JK
                                                    .J[ir_c][jr_c][emn][esr] +
                                            M(density_->m[jr], ss, rr) *
                                                self->h2eSSLL_JK
                                                    .J[ir_c][jr_c][emn][esr];
                                        if (self->with_gaunt)
                                        {
                                            int enm = nn * size_tmp + mm,
                                                ers = rr * size_tmp2 + ss,
                                                erm = rr * size_tmp + mm,
                                                ens = nn * size_tmp2 + ss;
                                            (void)erm;
                                            (void)ens;
                                            M(fock_tmp, mm, nn) -=
                                                M(density_->m[jr],
                                                  size_tmp2 + ss,
                                                  size_tmp2 + rr) *
                                                self->gauntLSSL_JK
                                                    .K[ir_c][jr_c][emr][esn];
                                            M(fock_tmp, mm + size_tmp,
                                              nn + size_tmp) -=
                                                M(density_->m[jr], ss, rr) *
                                                self->gauntLSSL_JK
                                                    .K[jr_c][ir_c][esn][emr];
                                            M(fock_tmp, mm + size_tmp, nn) +=
                                                M(density_->m[jr],
                                                  size_tmp2 + ss, rr) *
                                                    self->gauntLSLS_JK
                                                        .J[ir_c][jr_c][enm]
                                                          [ers] +
                                                M(density_->m[jr], ss,
                                                  size_tmp2 + rr) *
                                                    self->gauntLSSL_JK
                                                        .J[jr_c][ir_c][esr]
                                                          [emn];
                                            if (mm != nn)
                                            {
                                                int ern = rr * size_tmp + nn,
                                                    ems = mm * size_tmp2 + ss;
                                                (void)ern;
                                                (void)ems;
                                                M(fock_tmp, nn + size_tmp,
                                                  mm) +=
                                                    M(density_->m[jr],
                                                      size_tmp2 + ss, rr) *
                                                        self->gauntLSLS_JK
                                                            .J[ir_c][jr_c][emn]
                                                              [ers] +
                                                    M(density_->m[jr], ss,
                                                      size_tmp2 + rr) *
                                                        self->gauntLSSL_JK
                                                            .J[jr_c][ir_c][esr]
                                                              [enm];
                                            }
                                        }
                                    }
                            }
                            M(fock_tmp, nn, mm) = M(fock_tmp, mm, nn);
                            M(fock_tmp, nn + size_tmp, mm + size_tmp) =
                                M(fock_tmp, mm + size_tmp, nn + size_tmp);
                            M(fock_tmp, nn, mm + size_tmp) =
                                M(fock_tmp, mm + size_tmp, nn);
                            M(fock_tmp, mm, nn + size_tmp) =
                                M(fock_tmp, nn + size_tmp, mm);
                        }
                    mat_t coeff_tmp = mat_empty();
                    double* ene_orb_tmp =
                        (double*)calloc((size_t)fock_tmp.rows, sizeof(double));
                    eigensolverG(fock_tmp, overlap_half_i_4c_tmp, ene_orb_tmp,
                                 &coeff_tmp);
                    mat_assign(&self->x2cXXX.m[ir],
                               X2C_get_X_from_coeff(coeff_tmp));
                    mat_free(&fock_tmp);
                    mat_free(&overlap_half_i_4c_tmp);
                    mat_free(&coeff_tmp);
                    free(ene_orb_tmp);
                }
                else
                {
                    printf("ERROR: unknown Xmethod in get_amfi\n");
                    exit(99);
                }
            }
        }
        /*
            Evaluate R and amfi
        */
        mat_assign(&self->x2cRRR.m[ir],
                   X2C_get_R_4c(overlap_4c_full.m[ir], self->x2cXXX.m[ir]));
        {
            mat_t b00 = mat_block(SO_4c.m[ir], 0, 0, size_tmp, size_tmp);
            mat_t b0s =
                mat_block(SO_4c.m[ir], 0, size_tmp, size_tmp, size_tmp);
            mat_t bs0 =
                mat_block(SO_4c.m[ir], size_tmp, 0, size_tmp, size_tmp);
            mat_t bss = mat_block(SO_4c.m[ir], size_tmp, size_tmp, size_tmp,
                                  size_tmp);
            mat_t XT = mat_transpose(self->x2cXXX.m[ir]);
            mat_t t1 = mat_mul(b0s, self->x2cXXX.m[ir]);
            mat_t t2 = mat_mul(XT, bs0);
            mat_t t3a = mat_mul(XT, bss);
            mat_t t3 = mat_mul(t3a, self->x2cXXX.m[ir]);
            mat_t sum = mat_add(b00, t1);
            mat_add_inplace(sum, t2, 1.0);
            mat_add_inplace(sum, t3, 1.0);
            mat_t RT = mat_transpose(self->x2cRRR.m[ir]);
            mat_t rs = mat_mul(RT, sum);
            mat_assign(&amfi_unc.m[ir], mat_mul(rs, self->x2cRRR.m[ir]));
            mat_free(&b00);
            mat_free(&b0s);
            mat_free(&bs0);
            mat_free(&bss);
            mat_free(&XT);
            mat_free(&t1);
            mat_free(&t2);
            mat_free(&t3a);
            mat_free(&t3);
            mat_free(&sum);
            mat_free(&RT);
            mat_free(&rs);
        }
    }

    self->X_calculated = true;
    vmat_free(&h1e_4c_full);
    vmat_free(&overlap_4c_full);
    if (amfi4c)
    {
        vmat_free(&amfi_unc);
        return SO_4c;
    }
    else
    {
        vmat_free(&SO_4c);
        return amfi_unc;
    }
}

vmat_t dhf_sph_get_amfi_unc_2c(DHF_SPH* self, const int2eJK* h2eSSLL_SD,
                               const int2eJK* h2eSSSS_SD,
                               bool amfi_with_gaunt, bool amfi4c)
{
    if (!self->converged)
    {
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("!!  WARNING: 2-c HF did NOT converge  !!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }

    vmat_t amfi_unc = vmat_new(self->Nirrep);
    vmat_t SO_4c = vmat_new(self->Nirrep);
    vmat_t h1e_2c_full = vmat_new(self->Nirrep);
    vmat_t overlap_2c_full = vmat_new(self->Nirrep);
    /*
        Construct h1e_2c_full and overlap_2c_full
    */
    for (int ir = 0; ir < self->occMax_irrep; ir++)
    {
        mat_assign(&h1e_2c_full.m[ir], mat_clone(self->h1e_4c.m[ir]));
        mat_assign(&overlap_2c_full.m[ir], mat_clone(self->overlap_4c.m[ir]));
    }
    for (int ir = self->occMax_irrep; ir < self->Nirrep; ir++)
    {
        mat_assign(&self->x2cXXX.m[ir],
                   X2C_get_X(self->overlap.m[ir], self->kinetic.m[ir],
                             self->WWW.m[ir], self->Vnuc.m[ir]));
        mat_assign(&self->x2cRRR.m[ir],
                   X2C_get_R(self->overlap.m[ir], self->kinetic.m[ir],
                             self->x2cXXX.m[ir]));
        mat_assign(&h1e_2c_full.m[ir],
                   X2C_evaluate_h1e_x2c(self->overlap.m[ir],
                                        self->kinetic.m[ir], self->WWW.m[ir],
                                        self->Vnuc.m[ir], self->x2cXXX.m[ir],
                                        self->x2cRRR.m[ir]));
        mat_assign(&overlap_2c_full.m[ir], mat_clone(self->overlap.m[ir]));
    }
    /*
        Calculate 4-c density using approximate PES C_L and C_S
        C_L = R C_{2c}
        C_S = X C_L
    */
    vmat_t coeff_tmp = vmat_new(self->occMax_irrep);
    vmat_t density_tmp = vmat_new(self->occMax_irrep);
    vmat_t coeff_L_tmp = vmat_new(self->occMax_irrep);
    vmat_t coeff_S_tmp = vmat_new(self->occMax_irrep);
    for (int ir = 0; ir < self->occMax_irrep; ir++)
    {
        int size_tmp = self->irrep_list[ir].size;
        mat_assign(&coeff_L_tmp.m[ir],
                   mat_mul(self->x2cRRR.m[ir], self->coeff.m[ir]));
        mat_assign(&coeff_S_tmp.m[ir],
                   mat_mul(self->x2cXXX.m[ir], coeff_L_tmp.m[ir]));
        mat_assign(&coeff_tmp.m[ir], mat_new(2 * size_tmp, 2 * size_tmp));
        for (int ii = 0; ii < size_tmp; ii++)
            for (int jj = 0; jj < size_tmp; jj++)
            {
                M(coeff_tmp.m[ir], ii, size_tmp + jj) =
                    M(coeff_L_tmp.m[ir], ii, jj);
                M(coeff_tmp.m[ir], size_tmp + ii, size_tmp + jj) =
                    M(coeff_S_tmp.m[ir], ii, jj);
            }
        mat_assign(&density_tmp.m[ir],
                   dhf_sph_evaluateDensity_spinor(
                       coeff_tmp.m[ir], self->occNumber.v[ir], false));
    }

    for (int ir = 0; ir < self->Nirrep; ir++)
    {
        int ir_c = self->all2compact[ir];
        int size_tmp = self->irrep_list[ir].size;
        mat_assign(&SO_4c.m[ir], mat_new(2 * size_tmp, 2 * size_tmp));
        /*
            Evaluate SO integrals in 4c basis
        */
        for (int mm = 0; mm < size_tmp; mm++)
            for (int nn = 0; nn <= mm; nn++)
            {
                for (int jr = 0; jr < self->occMax_irrep; jr++)
                {
                    int jr_c = self->all2compact[jr];
                    int size_tmp2 = self->irrep_list[jr].size;
                    for (int ss = 0; ss < size_tmp2; ss++)
                        for (int rr = 0; rr < size_tmp2; rr++)
                        {
                            int emn = mm * size_tmp + nn,
                                esr = ss * size_tmp2 + rr,
                                emr = mm * size_tmp2 + rr,
                                esn = ss * size_tmp + nn;
                            M(SO_4c.m[ir], mm, nn) +=
                                M(density_tmp.m[jr], size_tmp2 + ss,
                                  size_tmp2 + rr) *
                                h2eSSLL_SD->J[jr_c][ir_c][esr][emn];
                            M(SO_4c.m[ir], mm + size_tmp, nn) -=
                                M(density_tmp.m[jr], ss, size_tmp2 + rr) *
                                h2eSSLL_SD->K[ir_c][jr_c][emr][esn];
                            if (mm != nn)
                            {
                                int enr = nn * size_tmp2 + rr,
                                    esm = ss * size_tmp + mm;
                                M(SO_4c.m[ir], nn + size_tmp, mm) -=
                                    M(density_tmp.m[jr], ss, size_tmp2 + rr) *
                                    h2eSSLL_SD->K[ir_c][jr_c][enr][esm];
                            }
                            M(SO_4c.m[ir], mm + size_tmp, nn + size_tmp) +=
                                M(density_tmp.m[jr], size_tmp2 + ss,
                                  size_tmp2 + rr) *
                                    h2eSSSS_SD->J[ir_c][jr_c][emn][esr] +
                                M(density_tmp.m[jr], ss, rr) *
                                    h2eSSLL_SD->J[ir_c][jr_c][emn][esr];
                            if (amfi_with_gaunt)
                            {
                                int enm = nn * size_tmp + mm,
                                    ers = rr * size_tmp2 + ss,
                                    erm = rr * size_tmp + mm,
                                    ens = nn * size_tmp2 + ss;
                                (void)erm;
                                (void)ens;
                                M(SO_4c.m[ir], mm, nn) -=
                                    M(density_tmp.m[jr], size_tmp2 + ss,
                                      size_tmp2 + rr) *
                                    self->gauntLSSL_JK.K[ir_c][jr_c][emr][esn];
                                M(SO_4c.m[ir], mm + size_tmp,
                                  nn + size_tmp) -=
                                    M(density_tmp.m[jr], ss, rr) *
                                    self->gauntLSSL_JK.K[jr_c][ir_c][esn][emr];
                                M(SO_4c.m[ir], mm + size_tmp, nn) +=
                                    M(density_tmp.m[jr], size_tmp2 + ss, rr) *
                                        self->gauntLSLS_JK
                                            .J[ir_c][jr_c][enm][ers] +
                                    M(density_tmp.m[jr], ss, size_tmp2 + rr) *
                                        self->gauntLSSL_JK
                                            .J[jr_c][ir_c][esr][emn];
                                if (mm != nn)
                                {
                                    int ern = rr * size_tmp + nn,
                                        ems = mm * size_tmp2 + ss;
                                    (void)ern;
                                    (void)ems;
                                    M(SO_4c.m[ir], nn + size_tmp, mm) +=
                                        M(density_tmp.m[jr], size_tmp2 + ss,
                                          rr) *
                                            self->gauntLSLS_JK
                                                .J[ir_c][jr_c][emn][ers] +
                                        M(density_tmp.m[jr], ss,
                                          size_tmp2 + rr) *
                                            self->gauntLSSL_JK
                                                .J[jr_c][ir_c][esr][enm];
                                }
                            }
                        }
                }
                M(SO_4c.m[ir], nn, mm) = M(SO_4c.m[ir], mm, nn);
                M(SO_4c.m[ir], nn + size_tmp, mm + size_tmp) =
                    M(SO_4c.m[ir], mm + size_tmp, nn + size_tmp);
                M(SO_4c.m[ir], nn, mm + size_tmp) =
                    M(SO_4c.m[ir], mm + size_tmp, nn);
                M(SO_4c.m[ir], mm, nn + size_tmp) =
                    M(SO_4c.m[ir], nn + size_tmp, mm);
            }

        {
            mat_t b00 = mat_block(SO_4c.m[ir], 0, 0, size_tmp, size_tmp);
            mat_t b0s =
                mat_block(SO_4c.m[ir], 0, size_tmp, size_tmp, size_tmp);
            mat_t bs0 =
                mat_block(SO_4c.m[ir], size_tmp, 0, size_tmp, size_tmp);
            mat_t bss = mat_block(SO_4c.m[ir], size_tmp, size_tmp, size_tmp,
                                  size_tmp);
            mat_t XT = mat_transpose(self->x2cXXX.m[ir]);
            mat_t t1 = mat_mul(b0s, self->x2cXXX.m[ir]);
            mat_t t2 = mat_mul(XT, bs0);
            mat_t t3a = mat_mul(XT, bss);
            mat_t t3 = mat_mul(t3a, self->x2cXXX.m[ir]);
            mat_t sum = mat_add(b00, t1);
            mat_add_inplace(sum, t2, 1.0);
            mat_add_inplace(sum, t3, 1.0);
            mat_t RT = mat_transpose(self->x2cRRR.m[ir]);
            mat_t rs = mat_mul(RT, sum);
            mat_assign(&amfi_unc.m[ir], mat_mul(rs, self->x2cRRR.m[ir]));
            mat_free(&b00);
            mat_free(&b0s);
            mat_free(&bs0);
            mat_free(&bss);
            mat_free(&XT);
            mat_free(&t1);
            mat_free(&t2);
            mat_free(&t3a);
            mat_free(&t3);
            mat_free(&sum);
            mat_free(&RT);
            mat_free(&rs);
        }
    }

    self->X_calculated = true;
    vmat_free(&h1e_2c_full);
    vmat_free(&overlap_2c_full);
    vmat_free(&coeff_tmp);
    vmat_free(&density_tmp);
    vmat_free(&coeff_L_tmp);
    vmat_free(&coeff_S_tmp);
    if (amfi4c)
    {
        vmat_free(&amfi_unc);
        return SO_4c;
    }
    else
    {
        vmat_free(&SO_4c);
        return amfi_unc;
    }
}

/* ----------------------------------------------------------------------
   getters (deep copies)
   ---------------------------------------------------------------------- */
static vmat_t vmat_clone_all(const vmat_t* src)
{
    vmat_t out = vmat_new(src->n);
    for (int ir = 0; ir < src->n; ir++)
        out.m[ir] = mat_clone(src->m[ir]);
    return out;
}

vmat_t dhf_sph_get_fock_4c(DHF_SPH* self)
{
    return vmat_clone_all(&self->fock_4c);
}
vmat_t dhf_sph_get_fock_fw(DHF_SPH* self)
{
    vmat_t fock_fw = vmat_new(self->occMax_irrep);
    for (int ir = 0; ir < self->occMax_irrep; ir++)
    {
        mat_assign(&fock_fw.m[ir],
                   X2C_transform_4c_2c(self->fock_4c.m[ir], self->x2cXXX.m[ir],
                                       self->x2cRRR.m[ir]));
    }
    return fock_fw;
}
vmat_t dhf_sph_get_fock_4c_K(DHF_SPH* self, const vmat_t* den, bool twoC)
{
    vmat_t fock_K = vmat_new(self->occMax_irrep);
    for (int ii = 0; ii < self->occMax_irrep; ii++)
        dhf_sph_evaluateFock_K(self, &fock_K.m[ii], twoC, den,
                               self->irrep_list[ii].size, ii);
    return fock_K;
}
vmat_t dhf_sph_get_fock_4c_2ePart(DHF_SPH* self, const vmat_t* den, bool twoC)
{
    vmat_t fock_2e = vmat_new(self->occMax_irrep);
    for (int ii = 0; ii < self->occMax_irrep; ii++)
        dhf_sph_evaluateFock_2e(self, &fock_2e.m[ii], twoC, den,
                                self->irrep_list[ii].size, ii);
    return fock_2e;
}
vmat_t dhf_sph_get_h1e_4c(DHF_SPH* self)
{
    return vmat_clone_all(&self->h1e_4c);
}
vmat_t dhf_sph_get_overlap_4c(DHF_SPH* self)
{
    return vmat_clone_all(&self->overlap_4c);
}
vmat_t dhf_sph_get_density(DHF_SPH* self)
{
    return vmat_clone_all(&self->density);
}
vmat_t dhf_sph_get_density_fw(DHF_SPH* self)
{
    vmat_t den_fw = vmat_new(self->occMax_irrep);
    for (int ir = 0; ir < self->occMax_irrep; ir++)
    {
        /* R C_2c = C_L */
        mat_t Rinv = mat_inverse(self->x2cRRR.m[ir]);
        mat_t CL = mat_block(self->coeff.m[ir], 0, self->coeff.m[ir].cols / 2,
                             self->coeff.m[ir].rows / 2,
                             self->coeff.m[ir].cols / 2);
        mat_t C2c = mat_mul(Rinv, CL);
        mat_assign(&den_fw.m[ir],
                   dhf_sph_evaluateDensity_spinor(C2c, self->occNumber.v[ir],
                                                  true));
        mat_free(&Rinv);
        mat_free(&CL);
        mat_free(&C2c);
    }
    return den_fw;
}
vvecd_t dhf_sph_get_occNumber(DHF_SPH* self)
{
    vvecd_t out = vvecd_new(self->occNumber.n);
    for (int ir = 0; ir < self->occNumber.n; ir++)
    {
        out.v[ir] = vecd_new(self->occNumber.v[ir].n);
        for (int ii = 0; ii < self->occNumber.v[ir].n; ii++)
            out.v[ir].d[ii] = self->occNumber.v[ir].d[ii];
    }
    return out;
}
vmat_t dhf_sph_get_X(DHF_SPH* self)
{
    if (self->X_calculated)
        return vmat_clone_all(&self->x2cXXX);
    else
    {
        printf("ERROR: get_X was called before X matrices calculated!\n");
        exit(99);
    }
}
vmat_t dhf_sph_get_R(DHF_SPH* self)
{
    if (self->X_calculated)
        return vmat_clone_all(&self->x2cRRR);
    else
    {
        printf("ERROR: get_R was called before X matrices calculated!\n");
        exit(99);
    }
}

/* ----------------------------------------------------------------------
   setters
   ---------------------------------------------------------------------- */
void dhf_sph_set_h1e_4c(DHF_SPH* self, const vmat_t* inputM, bool addto)
{
    printf("VERY DANGEROUS!! You changed h1e_4c!!\n");
    for (int ir = 0; ir < self->h1e_4c.n; ir++)
    {
        if (addto)
            mat_add_inplace(self->h1e_4c.m[ir], inputM->m[ir], 1.0);
        else
            mat_assign(&self->h1e_4c.m[ir], mat_clone(inputM->m[ir]));
    }
    return;
}
