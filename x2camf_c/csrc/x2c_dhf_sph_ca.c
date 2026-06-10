/*
 * x2c_dhf_sph_ca.c -- C translation of src/dhf_sph_ca.cpp (X2CAMF):
 * average-of-configuration open-shell Dirac-Hartree-Fock variant.
 *
 * The DHF_SPH_CA C++ class derives from DHF_SPH.  In this translation a
 * CA handle is an ordinary DHF_SPH* whose ->ca extension block is
 * non-NULL; dhf_sph_run_scf dispatches to dhf_sph_ca_run_scf when so.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "x2c_dhf_sph.h"
#include "x2c_general.h"
#include "x2c_mat.h"

/* ------------------------------------------------------------------ */
/* constructor                                                         */
/* ------------------------------------------------------------------ */
DHF_SPH* dhf_sph_ca_new(INT_SPH* int_sph, const char* filename,
                        int printLevel, bool spinFree, bool twoC,
                        bool with_gaunt, bool with_gauge, bool allInt,
                        bool gaussian_nuc)
{
    /* run the base DHF_SPH constructor first */
    DHF_SPH* self = dhf_sph_new(int_sph, filename, printLevel, spinFree, twoC,
                                with_gaunt, with_gauge, allInt, gaussian_nuc);

    DHF_CA_EXT* ca = (DHF_CA_EXT*)calloc(1, sizeof(DHF_CA_EXT));
    self->ca = ca;

    /* vector<int> openIrreps; */
    int* openIrreps = (int*)malloc((size_t)self->occMax_irrep * sizeof(int));
    int NopenIrreps = 0;

    /* for(int ir = 0; ir < occMax_irrep; ir+=irrep_list(ir).two_j+1) */
    for (int ir = 0; ir < self->occMax_irrep;
         ir += self->irrep_list[ir].two_j + 1)
    {
        for (int ii = 0; ii < self->occNumber.v[ir].n; ii++)
        {
            /* if 1 > occNumber(ir)(ii) > 0 */
            if (fabs(self->occNumber.v[ir].d[ii]) > 1e-4 &&
                self->occNumber.v[ir].d[ii] < 0.9999)
            {
                openIrreps[NopenIrreps] = ir;
                NopenIrreps++;
                break;
            }
        }
    }
    int NOpenShells = NopenIrreps;
    ca->NOpenShells = NOpenShells;
    ca->n_shells = NOpenShells;

    /* occNumberShells.resize(NOpenShells+2); */
    ca->n_occShells = NOpenShells + 2;
    ca->occNumberShells =
        (vvecd_t*)calloc((size_t)ca->n_occShells, sizeof(vvecd_t));

    /* MM_list / NN_list / f_list grow with push_back; allocate to max len */
    ca->MM_list = (double*)malloc((size_t)NOpenShells * sizeof(double));
    ca->NN_list = (double*)malloc((size_t)NOpenShells * sizeof(double));
    ca->f_list = (double*)malloc((size_t)NOpenShells * sizeof(double));
    int MM_size = 0, NN_size = 0, f_size = 0;

    for (int ii = 0; ii < NOpenShells; ii++)
    {
        /* MM_list.push_back(irrep_list(openIrreps[ii]).two_j+1); */
        ca->MM_list[MM_size] = self->irrep_list[openIrreps[ii]].two_j + 1;
        MM_size++;
    }

    /* occNumberShells[0] = occNumber; (deep copy) */
    {
        int nir = self->occNumber.n;
        ca->occNumberShells[0] = vvecd_new(nir);
        for (int ir = 0; ir < nir; ir++)
        {
            int len = self->occNumber.v[ir].n;
            ca->occNumberShells[0].v[ir] = vecd_new(len);
            for (int jj = 0; jj < len; jj++)
                ca->occNumberShells[0].v[ir].d[jj] =
                    self->occNumber.v[ir].d[jj];
        }
    }
    /* for(int ii = 1; ii < occNumberShells.size()-1; ii++)
           occNumberShells[ii].resize(occMax_irrep); */
    for (int ii = 1; ii < ca->n_occShells - 1; ii++)
        ca->occNumberShells[ii] = vvecd_new(self->occMax_irrep);
    /* occNumberShells[NOpenShells+1].resize(irrep_list.rows()); */
    ca->occNumberShells[NOpenShells + 1] = vvecd_new(self->Nirrep);

    for (int ir = 0; ir < self->occMax_irrep; ir++)
    {
        for (int ii = 1; ii < ca->n_occShells - 1; ii++)
        {
            vecd_free(&ca->occNumberShells[ii].v[ir]);
            ca->occNumberShells[ii].v[ir] =
                vecd_new(self->irrep_list[ir].size); /* Zero */
        }
        /* occNumberShells[NOpenShells+1](ir) = Ones(size); */
        vecd_free(&ca->occNumberShells[NOpenShells + 1].v[ir]);
        ca->occNumberShells[NOpenShells + 1].v[ir] =
            vecd_new(self->irrep_list[ir].size);
        for (int jj = 0; jj < self->irrep_list[ir].size; jj++)
            ca->occNumberShells[NOpenShells + 1].v[ir].d[jj] = 1.0;

        for (int ii = 0; ii < self->occNumber.v[ir].n; ii++)
        {
            if (fabs(self->occNumber.v[ir].d[ii]) > 1e-4)
            {
                /* if 1 > occNumber(ir)(ii) > 0 */
                if (self->occNumber.v[ir].d[ii] < 0.9999)
                {
                    if (f_size < NopenIrreps)
                    {
                        if (ir == openIrreps[f_size])
                        {
                            ca->f_list[f_size] = self->occNumber.v[ir].d[ii];
                            f_size++;
                            ca->NN_list[NN_size] =
                                ca->f_list[f_size - 1] *
                                ca->MM_list[f_size - 1];
                            NN_size++;
                        }
                    }
                    ca->occNumberShells[0].v[ir].d[ii] = 0.0;
                    ca->occNumberShells[f_size].v[ir].d[ii] = 1.0;
                }
                ca->occNumberShells[NOpenShells + 1].v[ir].d[ii] = 0.0;
            }
        }
    }
    for (int ir = self->occMax_irrep; ir < self->Nirrep; ir++)
    {
        vecd_free(&ca->occNumberShells[NOpenShells + 1].v[ir]);
        ca->occNumberShells[NOpenShells + 1].v[ir] =
            vecd_new(self->irrep_list[ir].size);
        for (int jj = 0; jj < self->irrep_list[ir].size; jj++)
            ca->occNumberShells[NOpenShells + 1].v[ir].d[jj] = 1.0;
    }

    /* densityShells will be sized in runSCF; pre-create container */
    ca->n_denShells = NOpenShells + 1;
    ca->densityShells = NULL;
    ca->openShell = 1;

    if (printLevel >= 4)
    {
        printf("Open shell occupations:\n");
        for (int ir = 0; ir < self->occMax_irrep;
             ir += self->irrep_list[ir].two_j + 1)
        {
            printf("l = %d\n", self->irrep_list[ir].l);
            for (int ii = 0; ii < ca->n_occShells; ii++)
            {
                printf("%d: ", ii);
                for (int jj = 0; jj < ca->occNumberShells[ii].v[ir].n; jj++)
                    printf("%s%g", jj == 0 ? "" : " ",
                           ca->occNumberShells[ii].v[ir].d[jj]);
                printf("\n");
            }
        }
        for (int ir = self->occMax_irrep; ir < self->Nirrep; ir++)
        {
            printf("%d: ", ca->n_occShells - 1);
            int last = ca->n_occShells - 1;
            for (int jj = 0; jj < ca->occNumberShells[last].v[ir].n; jj++)
                printf("%s%g", jj == 0 ? "" : " ",
                       ca->occNumberShells[last].v[ir].d[jj]);
            printf("\n");
        }
        printf("Configuration-averaged HF initialization.\n");
        printf("Number of open shells: %d\n", NOpenShells);
        printf("No.\tMM\tNN\tf=NN/MM\n");
        for (int ii = 0; ii < NOpenShells; ii++)
        {
            printf("%d\t%g\t%g\t%g\n", ii + 1, ca->MM_list[ii],
                   ca->NN_list[ii], ca->f_list[ii]);
        }
    }

    free(openIrreps);
    return self;
}

/* ------------------------------------------------------------------ */
/* DHF_SPH_CA::evaluateDensity_aoc                                     */
/* ------------------------------------------------------------------ */
static mat_t evaluateDensity_aoc(mat_t coeff_, vecd_t occNumber_, bool twoC)
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
                    if (fabs(occNumber_.d[ii] - 1.0) < 1e-5)
                    {
                        M(den, aa, bb) +=
                            M(coeff_, aa, ii + size) *
                            M(coeff_, bb, ii + size);
                        M(den, size + aa, bb) +=
                            M(coeff_, size + aa, ii + size) *
                            M(coeff_, bb, ii + size);
                        M(den, aa, size + bb) +=
                            M(coeff_, aa, ii + size) *
                            M(coeff_, size + bb, ii + size);
                        M(den, size + aa, size + bb) +=
                            M(coeff_, size + aa, ii + size) *
                            M(coeff_, size + bb, ii + size);
                    }
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
                {
                    if (fabs(occNumber_.d[ii] - 1.0) < 1e-5)
                        M(den, aa, bb) +=
                            M(coeff_, aa, ii) * M(coeff_, bb, ii);
                }
        return den;
    }
}

/* forward declarations for the CA Fock / energy helpers */
static void ca_evaluateFock(DHF_SPH* self, mat_t* fock_c, bool twoC,
                            const vmat_t* densities, int ndens, int size,
                            int Iirrep);
static double ca_evaluateEnergy(DHF_SPH* self, bool twoC);

/* ------------------------------------------------------------------ */
/* DHF_SPH_CA::runSCF                                                  */
/* ------------------------------------------------------------------ */
void dhf_sph_ca_run_scf(DHF_SPH* self, bool twoC, bool renormSmall)
{
    DHF_CA_EXT* ca = self->ca;
    int NOpenShells = ca->NOpenShells;
    int occMax_irrep = self->occMax_irrep;

    if (renormSmall && !twoC)
    {
        dhf_sph_renormalize_small(self);
    }

    /* Matrix<vector<MatrixXd>,-1,-1> error4DIIS(occMax_irrep,NOpenShells+2);
       vector<MatrixXd> fock4DIIS[occMax_irrep];
       translated as fixed-capacity (size_DIIS) mat_t arrays with counts. */
    int size_DIIS = self->size_DIIS;
    int cols_err = NOpenShells + 2;
    /* error4DIIS[ir][kk] is a list (up to size_DIIS+1) of mat_t */
    mat_t* error4DIIS =
        (mat_t*)calloc((size_t)occMax_irrep * (size_t)cols_err *
                           (size_t)(size_DIIS + 1),
                       sizeof(mat_t));
    int* error4DIIS_n =
        (int*)calloc((size_t)occMax_irrep * (size_t)cols_err, sizeof(int));
    mat_t* fock4DIIS = (mat_t*)calloc(
        (size_t)occMax_irrep * (size_t)(size_DIIS + 1), sizeof(mat_t));
    int* fock4DIIS_n = (int*)calloc((size_t)occMax_irrep, sizeof(int));

#define ERR4(ir, kk, idx)                                                     \
    error4DIIS[(((size_t)(ir) * (size_t)cols_err + (size_t)(kk)) *            \
                (size_t)(size_DIIS + 1)) +                                    \
               (size_t)(idx)]
#define ERRN(ir, kk) error4DIIS_n[(size_t)(ir) * (size_t)cols_err + (size_t)(kk)]
#define FOCK4(ir, idx)                                                        \
    fock4DIIS[(size_t)(ir) * (size_t)(size_DIIS + 1) + (size_t)(idx)]

    if (self->printLevel >= 1)
    {
        printf("\n");
        if (twoC)
            printf("Start CA-X2C-1e Hartree-Fock iterations...\n");
        else
            printf("Start CA-Dirac Hartree-Fock iterations...\n");
        printf("\n");
    }

    /* densityShells.resize(NOpenShells+2); */
    if (ca->densityShells != NULL)
    {
        for (int ii = 0; ii < ca->n_denShells + 1; ii++)
            vmat_free(&ca->densityShells[ii]);
        free(ca->densityShells);
    }
    ca->n_denShells = NOpenShells + 1;
    /* container holds NOpenShells+2 vMatrixXd entries */
    ca->densityShells =
        (vmat_t*)calloc((size_t)(NOpenShells + 2), sizeof(vmat_t));
    vmat_t* densityShells = ca->densityShells;
    vmat_t* newDensityShells =
        (vmat_t*)calloc((size_t)(NOpenShells + 2), sizeof(vmat_t));

    dhf_sph_eigensolverG_irrep(self, &self->h1e_4c, &self->overlap_half_i_4c,
                               &self->ene_orb, &self->coeff);
    for (int ii = 0; ii < NOpenShells + 1; ii++)
    {
        densityShells[ii] = vmat_new(occMax_irrep);
        newDensityShells[ii] = vmat_new(occMax_irrep);
    }
    densityShells[NOpenShells + 1] = vmat_new(self->Nirrep);
    newDensityShells[NOpenShells + 1] = vmat_new(self->Nirrep);

    for (int ir = 0; ir < occMax_irrep;
         ir += self->irrep_list[ir].two_j + 1)
    {
        for (int ii = 0; ii < NOpenShells + 2; ii++)
            mat_assign(&densityShells[ii].m[ir],
                       evaluateDensity_aoc(self->coeff.m[ir],
                                           ca->occNumberShells[ii].v[ir],
                                           twoC));
    }
    for (int ir = occMax_irrep; ir < self->Nirrep;
         ir += self->irrep_list[ir].two_j + 1)
    {
        /* WORNG */
        mat_assign(&densityShells[NOpenShells + 1].m[ir],
                   evaluateDensity_aoc(
                       self->coeff.m[ir],
                       ca->occNumberShells[NOpenShells + 1].v[ir], twoC));
    }

    for (int iter = 1; iter <= self->maxIter; iter++)
    {
        if (iter <= 2)
        {
            for (int ir = 0; ir < occMax_irrep;
                 ir += self->irrep_list[ir].two_j + 1)
            {
                int size_tmp = self->irrep_list[ir].size;
                ca_evaluateFock(self, &self->fock_4c.m[ir], twoC,
                                densityShells, NOpenShells + 2, size_tmp, ir);
            }
        }
        else
        {
            int tmp_size = fock4DIIS_n[0];
            mat_t B4DIIS = mat_new(tmp_size + 1, tmp_size + 1);
            double* vec_b = (double*)malloc((size_t)(tmp_size + 1) *
                                            sizeof(double));
            for (int ii = 0; ii < tmp_size; ii++)
            {
                for (int jj = 0; jj <= ii; jj++)
                {
                    M(B4DIIS, ii, jj) = 0.0;
                    for (int ir = 0; ir < occMax_irrep;
                         ir += self->irrep_list[ir].two_j + 1)
                        for (int kk = 0; kk < NOpenShells + 1; kk++)
                        {
                            mat_t e_ii = ERR4(ir, kk, ii);
                            mat_t e_jj = ERR4(ir, kk, jj);
                            mat_t prod =
                                mat_mul(mat_transpose(e_ii), e_jj);
                            M(B4DIIS, ii, jj) += M(prod, 0, 0);
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
            double* Cvec =
                (double*)malloc((size_t)(tmp_size + 1) * sizeof(double));
            mat_solve_vec(B4DIIS, vec_b, Cvec);
            for (int ir = 0; ir < occMax_irrep;
                 ir += self->irrep_list[ir].two_j + 1)
            {
                mat_assign(&self->fock_4c.m[ir],
                           mat_new(self->fock_4c.m[ir].rows,
                                   self->fock_4c.m[ir].cols));
                for (int ii = 0; ii < tmp_size; ii++)
                {
                    mat_add_inplace(self->fock_4c.m[ir], FOCK4(ir, ii),
                                    Cvec[ii]);
                }
            }
            mat_free(&B4DIIS);
            free(vec_b);
            free(Cvec);
        }
        dhf_sph_eigensolverG_irrep(self, &self->fock_4c,
                                   &self->overlap_half_i_4c, &self->ene_orb,
                                   &self->coeff);

        for (int ir = 0; ir < occMax_irrep;
             ir += self->irrep_list[ir].two_j + 1)
        {
            for (int ii = 0; ii < NOpenShells + 2; ii++)
                mat_assign(&newDensityShells[ii].m[ir],
                           evaluateDensity_aoc(
                               self->coeff.m[ir],
                               ca->occNumberShells[ii].v[ir], twoC));
        }
        for (int ir = occMax_irrep; ir < self->Nirrep;
             ir += self->irrep_list[ir].two_j + 1)
        {
            /* WORNG */
            mat_assign(&newDensityShells[NOpenShells + 1].m[ir],
                       evaluateDensity_aoc(
                           self->coeff.m[ir],
                           ca->occNumberShells[NOpenShells + 1].v[ir],
                           twoC));
        }
        self->d_density = 0.0;
        for (int ii = 0; ii < NOpenShells + 1; ii++)
            self->d_density += dhf_sph_evaluateChange_irrep(
                self, &densityShells[ii], &newDensityShells[ii]);
        if (self->printLevel >= 4)
            printf("Iter #%d maximum density difference: %.17g\n", iter,
                   self->d_density);
        for (int ii = 0; ii < NOpenShells + 2; ii++)
        {
            for (int ir = 0; ir < densityShells[ii].n; ir++)
                mat_assign(&densityShells[ii].m[ir],
                           mat_clone(newDensityShells[ii].m[ir]));
        }

        if (self->d_density < self->convControl)
        {
            self->converged = true;
            if (self->printLevel >= 1)
                printf("\nCA-SCF converges after %d iterations.\n", iter);
            if (self->printLevel >= 4)
            {
                printf("\nWARNING: CA-SCF orbital energies are fake!!!\n\n");
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

            self->ene_scf = ca_evaluateEnergy(self, twoC);
            if (twoC)
                printf("Final CA-X2C-1e HF energy is %.15g hartree.\n",
                       self->ene_scf);
            else
                printf("Final CA-DHF energy is %.15g hartree.\n",
                       self->ene_scf);
            break;
        }
        for (int ir = 0; ir < occMax_irrep;
             ir += self->irrep_list[ir].two_j + 1)
        {
            int size_tmp = self->irrep_list[ir].size;
            ca_evaluateFock(self, &self->fock_4c.m[ir], twoC, densityShells,
                            NOpenShells + 2, size_tmp, ir);

            eigensolverG(self->fock_4c.m[ir], self->overlap_half_i_4c.m[ir],
                         self->ene_orb.v[ir].d, &self->coeff.m[ir]);
            for (int ii = 0; ii < NOpenShells + 2; ii++)
            {
                mat_assign(&newDensityShells[ii].m[ir],
                           evaluateDensity_aoc(
                               self->coeff.m[ir],
                               ca->occNumberShells[ii].v[ir], twoC));
                /* error4DIIS(ir,ii).push_back(
                       evaluateErrorDIIS(densityShells(ii)(ir),
                                         newDensityShells(ii)(ir))); */
                ERR4(ir, ii, ERRN(ir, ii)) = dhf_sph_evaluateErrorDIIS_den(
                    densityShells[ii].m[ir], newDensityShells[ii].m[ir]);
                ERRN(ir, ii)++;
            }
            FOCK4(ir, fock4DIIS_n[ir]) = mat_clone(self->fock_4c.m[ir]);
            fock4DIIS_n[ir]++;

            if (ERRN(ir, 0) > size_DIIS)
            {
                for (int ii = 0; ii < NOpenShells + 2; ii++)
                {
                    /* erase begin */
                    mat_free(&ERR4(ir, ii, 0));
                    for (int kk = 1; kk < ERRN(ir, ii); kk++)
                        ERR4(ir, ii, kk - 1) = ERR4(ir, ii, kk);
                    ERRN(ir, ii)--;
                    /* the vacated top slot still aliases the buffer now at
                       index ERRN; clear it so the final cleanup that frees
                       every slot does not double-free. */
                    ERR4(ir, ii, ERRN(ir, ii)) = mat_empty();
                }
                mat_free(&FOCK4(ir, 0));
                for (int kk = 1; kk < fock4DIIS_n[ir]; kk++)
                    FOCK4(ir, kk - 1) = FOCK4(ir, kk);
                fock4DIIS_n[ir]--;
                FOCK4(ir, fock4DIIS_n[ir]) = mat_empty();
            }
        }
    }

    /* density.resize(occMax_irrep); */
    vmat_free(&self->density);
    self->density = vmat_new(occMax_irrep);
    for (int ir = 0; ir < occMax_irrep;
         ir += self->irrep_list[ir].two_j + 1)
    {
        mat_assign(&self->density.m[ir], mat_clone(densityShells[0].m[ir]));
        for (int ii = 1; ii < NOpenShells + 1; ii++)
            mat_add_inplace(self->density.m[ir], densityShells[ii].m[ir],
                            ca->f_list[ii - 1]);
        for (int jj = 1; jj < self->irrep_list[ir].two_j + 1; jj++)
        {
            /* fock_4c is not included here because the Fock matrix of
               AOC-SCF is not well-defined. */
            vecd_free(&self->ene_orb.v[ir + jj]);
            self->ene_orb.v[ir + jj] = vecd_new(self->ene_orb.v[ir].n);
            for (int kk = 0; kk < self->ene_orb.v[ir].n; kk++)
                self->ene_orb.v[ir + jj].d[kk] = self->ene_orb.v[ir].d[kk];
            mat_assign(&self->coeff.m[ir + jj],
                       mat_clone(self->coeff.m[ir]));
            mat_assign(&self->density.m[ir + jj],
                       mat_clone(self->density.m[ir]));
            for (int ii = 0; ii < NOpenShells + 2; ii++)
                mat_assign(&densityShells[ii].m[ir + jj],
                           mat_clone(densityShells[ii].m[ir]));
        }
    }

    /* cleanup of DIIS scratch */
    for (size_t i = 0;
         i < (size_t)occMax_irrep * (size_t)cols_err * (size_t)(size_DIIS + 1);
         i++)
        mat_free(&error4DIIS[i]);
    for (size_t i = 0;
         i < (size_t)occMax_irrep * (size_t)(size_DIIS + 1); i++)
        mat_free(&fock4DIIS[i]);
    free(error4DIIS);
    free(error4DIIS_n);
    free(fock4DIIS);
    free(fock4DIIS_n);
    for (int ii = 0; ii < NOpenShells + 2; ii++)
        vmat_free(&newDensityShells[ii]);
    free(newDensityShells);

#undef ERR4
#undef ERRN
#undef FOCK4
}

/* ------------------------------------------------------------------ */
/* DHF_SPH_CA::evaluateFock                                            */
/* ------------------------------------------------------------------ */
static void ca_evaluateFock(DHF_SPH* self, mat_t* fock_c, bool twoC,
                            const vmat_t* densities, int ndens, int size,
                            int Iirrep)
{
    (void)ndens;
    DHF_CA_EXT* ca = self->ca;
    int NOpenShells = ca->NOpenShells;
    int ir = self->all2compact[Iirrep];
    vmat_t R = vmat_new(NOpenShells + 2);
    vmat_t Q = vmat_new(NOpenShells + 1);
    for (int ii = 0; ii < NOpenShells + 2; ii++)
    {
        R.m[ii] = mat_transpose(densities[ii].m[Iirrep]);
        if (ii < NOpenShells + 1)
        {
            if (twoC)
                Q.m[ii] = mat_new(size, size);
            else
                Q.m[ii] = mat_new(2 * size, 2 * size);
        }
    }
    if (twoC)
    {
#pragma omp parallel for
        for (int mm = 0; mm < size; mm++)
            for (int nn = 0; nn <= mm; nn++)
            {
                for (int ii = 0; ii < NOpenShells + 1; ii++)
                    for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
                    {
                        int Jirrep = self->compact2all[jr];
                        mat_t den_tmp = densities[ii].m[Jirrep];
                        double twojP1 = self->irrep_list[Jirrep].two_j + 1;
                        int size_tmp2 = self->irrep_list[Jirrep].size;
                        for (int aa = 0; aa < size_tmp2; aa++)
                            for (int bb = 0; bb < size_tmp2; bb++)
                            {
                                int emn = mm * size + nn,
                                    eab = aa * size_tmp2 + bb;
                                M(Q.m[ii], mm, nn) +=
                                    twojP1 * M(den_tmp, aa, bb) *
                                    self->h2eLLLL_JK.J[ir][jr][emn][eab];
                            }
                        M(Q.m[ii], nn, mm) = M(Q.m[ii], mm, nn);
                    }
            }
    }
    else
    {
#pragma omp parallel for
        for (int mm = 0; mm < size; mm++)
            for (int nn = 0; nn <= mm; nn++)
            {
                for (int ii = 0; ii < NOpenShells + 1; ii++)
                    for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
                    {
                        int Jirrep = self->compact2all[jr];
                        mat_t den_tmp = densities[ii].m[Jirrep];
                        double twojP1 = self->irrep_list[Jirrep].two_j + 1;
                        int size_tmp2 = self->irrep_list[Jirrep].size;
                        for (int ss = 0; ss < size_tmp2; ss++)
                            for (int rr = 0; rr < size_tmp2; rr++)
                            {
                                int emn = mm * size + nn,
                                    esr = ss * size_tmp2 + rr,
                                    emr = mm * size_tmp2 + rr,
                                    esn = ss * size + nn;
                                M(Q.m[ii], mm, nn) +=
                                    twojP1 * M(den_tmp, ss, rr) *
                                        self->h2eLLLL_JK.J[ir][jr][emn][esr] +
                                    twojP1 *
                                        M(den_tmp, size_tmp2 + ss,
                                          size_tmp2 + rr) *
                                        self->h2eSSLL_JK.J[jr][ir][esr][emn];
                                M(Q.m[ii], mm + size, nn) -=
                                    twojP1 *
                                    M(den_tmp, ss, size_tmp2 + rr) *
                                    self->h2eSSLL_JK.K[ir][jr][emr][esn];
                                M(Q.m[ii], mm + size, nn + size) +=
                                    twojP1 *
                                        M(den_tmp, size_tmp2 + ss,
                                          size_tmp2 + rr) *
                                        self->h2eSSSS_JK.J[ir][jr][emn][esr] +
                                    twojP1 * M(den_tmp, ss, rr) *
                                        self->h2eSSLL_JK.J[ir][jr][emn][esr];
                                if (mm != nn)
                                {
                                    int enr = nn * size_tmp2 + rr,
                                        esm = ss * size + mm;
                                    M(Q.m[ii], nn + size, mm) -=
                                        twojP1 *
                                        M(den_tmp, ss, size_tmp2 + rr) *
                                        self->h2eSSLL_JK.K[ir][jr][enr][esm];
                                }
                                if (self->with_gaunt)
                                {
                                    int enm = nn * size + mm,
                                        ers = rr * size_tmp2 + ss;
                                    M(Q.m[ii], mm, nn) -=
                                        twojP1 *
                                        M(den_tmp, size_tmp2 + ss,
                                          size_tmp2 + rr) *
                                        self->gauntLSSL_JK.K[ir][jr][emr][esn];
                                    M(Q.m[ii], mm + size, nn) +=
                                        twojP1 *
                                            M(den_tmp, ss + size_tmp2, rr) *
                                            self->gauntLSLS_JK
                                                .J[ir][jr][enm][ers] +
                                        twojP1 *
                                            M(den_tmp, ss, size_tmp2 + rr) *
                                            self->gauntLSSL_JK
                                                .J[jr][ir][esr][emn];
                                    M(Q.m[ii], mm + size, nn + size) -=
                                        twojP1 * M(den_tmp, ss, rr) *
                                        self->gauntLSSL_JK.K[jr][ir][esn][emr];
                                    if (mm != nn)
                                    {
                                        M(Q.m[ii], nn + size, mm) +=
                                            twojP1 *
                                                M(den_tmp, size_tmp2 + ss,
                                                  rr) *
                                                self->gauntLSLS_JK
                                                    .J[ir][jr][emn][ers] +
                                            twojP1 *
                                                M(den_tmp, ss,
                                                  size_tmp2 + rr) *
                                                self->gauntLSSL_JK
                                                    .J[jr][ir][esr][enm];
                                    }
                                }
                            }
                        M(Q.m[ii], nn, mm) = M(Q.m[ii], mm, nn);
                        M(Q.m[ii], mm, nn + size) =
                            M(Q.m[ii], nn + size, mm);
                        M(Q.m[ii], nn, mm + size) =
                            M(Q.m[ii], mm + size, nn);
                        M(Q.m[ii], size + nn, size + mm) =
                            M(Q.m[ii], size + mm, size + nn);
                    }
            }
    }

    mat_assign(fock_c, mat_clone(self->h1e_4c.m[Iirrep]));
    for (int ii = 0; ii < NOpenShells + 1; ii++)
    {
        if (ii != 0)
            mat_scale_inplace(Q.m[ii], ca->f_list[ii - 1]);
        mat_add_inplace(*fock_c, Q.m[ii], 1.0);
    }
    mat_t S = self->overlap_4c.m[Iirrep];
    mat_t LM;
    if (twoC)
        LM = mat_new(size, size);
    else
        LM = mat_new(2 * size, 2 * size);

    for (int ii = 1; ii < NOpenShells + 1; ii++)
    {
        double f_u = ca->f_list[ii - 1];
        double a_u = ca->MM_list[ii - 1] * (ca->NN_list[ii - 1] - 1.0) /
                     ca->NN_list[ii - 1] / (ca->MM_list[ii - 1] - 1.0);
        double alpha_u = (1 - a_u) / (1 - f_u);
        /* LM += S*R(ii)*Q(ii)*(alpha_u*f_u*R(0)+(a_u-1.0)*(0.5*R(ii)
                +R(NOpenShells+1)))*S; */
        mat_t inner = mat_scaled(R.m[0], alpha_u * f_u);
        mat_t half_Rii = mat_scaled(R.m[ii], 0.5);
        mat_t bracket = mat_add(half_Rii, R.m[NOpenShells + 1]);
        mat_add_inplace(inner, bracket, (a_u - 1.0));
        mat_t t1 = mat_mul(S, R.m[ii]);
        mat_t t2 = mat_mul(t1, Q.m[ii]);
        mat_t t3 = mat_mul(t2, inner);
        mat_t t4 = mat_mul(t3, S);
        mat_add_inplace(LM, t4, 1.0);
        mat_free(&inner);
        mat_free(&half_Rii);
        mat_free(&bracket);
        mat_free(&t1);
        mat_free(&t2);
        mat_free(&t3);
        mat_free(&t4);
    }
    /* fock_c += LM + LM.adjoint(); */
    mat_t LMt = mat_transpose(LM);
    mat_add_inplace(*fock_c, LM, 1.0);
    mat_add_inplace(*fock_c, LMt, 1.0);
    mat_free(&LM);
    mat_free(&LMt);

    vmat_free(&R);
    vmat_free(&Q);
}

/* ------------------------------------------------------------------ */
/* DHF_SPH_CA::evaluateEnergy                                          */
/* ------------------------------------------------------------------ */
static double ca_evaluateEnergy(DHF_SPH* self, bool twoC)
{
    DHF_CA_EXT* ca = self->ca;
    int NOpenShells = ca->NOpenShells;
    vmat_t* densityShells = ca->densityShells;
    double ene = 0.0;
    for (int ir = 0; ir < self->occMax_irrep_compact; ir++)
    {
        int Iirrep = self->compact2all[ir];
        int size = self->irrep_list[Iirrep].size;
        vmat_t Q = vmat_new(NOpenShells + 1);
        for (int ii = 0; ii < NOpenShells + 1; ii++)
        {
            if (twoC)
                Q.m[ii] = mat_new(size, size);
            else
                Q.m[ii] = mat_new(2 * size, 2 * size);
        }
        if (twoC)
        {
#pragma omp parallel for
            for (int mm = 0; mm < size; mm++)
                for (int nn = 0; nn <= mm; nn++)
                {
                    for (int ii = 0; ii < NOpenShells + 1; ii++)
                        for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
                        {
                            int Jirrep = self->compact2all[jr];
                            mat_t den_tmp = densityShells[ii].m[Jirrep];
                            double twojP1 =
                                self->irrep_list[Jirrep].two_j + 1;
                            int size_tmp2 = self->irrep_list[Jirrep].size;
                            for (int aa = 0; aa < size_tmp2; aa++)
                                for (int bb = 0; bb < size_tmp2; bb++)
                                {
                                    int emn = mm * size + nn,
                                        eab = aa * size_tmp2 + bb;
                                    M(Q.m[ii], mm, nn) +=
                                        twojP1 * M(den_tmp, aa, bb) *
                                        self->h2eLLLL_JK.J[ir][jr][emn][eab];
                                }
                            M(Q.m[ii], nn, mm) = M(Q.m[ii], mm, nn);
                        }
                }
        }
        else
        {
#pragma omp parallel for
            for (int mm = 0; mm < size; mm++)
                for (int nn = 0; nn <= mm; nn++)
                {
                    for (int jr = 0; jr < self->occMax_irrep_compact; jr++)
                        for (int ii = 0; ii < NOpenShells + 1; ii++)
                        {
                            int Jirrep = self->compact2all[jr];
                            mat_t den_tmp = densityShells[ii].m[Jirrep];
                            double twojP1 =
                                self->irrep_list[Jirrep].two_j + 1;
                            int size_tmp2 = self->irrep_list[Jirrep].size;
                            for (int ss = 0; ss < size_tmp2; ss++)
                                for (int rr = 0; rr < size_tmp2; rr++)
                                {
                                    int emn = mm * size + nn,
                                        esr = ss * size_tmp2 + rr,
                                        emr = mm * size_tmp2 + rr,
                                        esn = ss * size + nn;
                                    M(Q.m[ii], mm, nn) +=
                                        twojP1 * M(den_tmp, ss, rr) *
                                            self->h2eLLLL_JK
                                                .J[ir][jr][emn][esr] +
                                        twojP1 *
                                            M(den_tmp, size_tmp2 + ss,
                                              size_tmp2 + rr) *
                                            self->h2eSSLL_JK
                                                .J[jr][ir][esr][emn];
                                    M(Q.m[ii], mm + size, nn) -=
                                        twojP1 *
                                        M(den_tmp, ss, size_tmp2 + rr) *
                                        self->h2eSSLL_JK.K[ir][jr][emr][esn];
                                    M(Q.m[ii], mm + size, nn + size) +=
                                        twojP1 *
                                            M(den_tmp, size_tmp2 + ss,
                                              size_tmp2 + rr) *
                                            self->h2eSSSS_JK
                                                .J[ir][jr][emn][esr] +
                                        twojP1 * M(den_tmp, ss, rr) *
                                            self->h2eSSLL_JK
                                                .J[ir][jr][emn][esr];
                                    if (mm != nn)
                                    {
                                        int enr = nn * size_tmp2 + rr,
                                            esm = ss * size + mm;
                                        M(Q.m[ii], nn + size, mm) -=
                                            twojP1 *
                                            M(den_tmp, ss, size_tmp2 + rr) *
                                            self->h2eSSLL_JK
                                                .K[ir][jr][enr][esm];
                                    }
                                    if (self->with_gaunt)
                                    {
                                        int enm = nn * size + mm,
                                            ers = rr * size_tmp2 + ss;
                                        M(Q.m[ii], mm, nn) -=
                                            twojP1 *
                                            M(den_tmp, size_tmp2 + ss,
                                              size_tmp2 + rr) *
                                            self->gauntLSSL_JK
                                                .K[ir][jr][emr][esn];
                                        M(Q.m[ii], mm + size, nn) +=
                                            twojP1 *
                                                M(den_tmp, ss + size_tmp2,
                                                  rr) *
                                                self->gauntLSLS_JK
                                                    .J[ir][jr][enm][ers] +
                                            twojP1 *
                                                M(den_tmp, ss,
                                                  size_tmp2 + rr) *
                                                self->gauntLSSL_JK
                                                    .J[jr][ir][esr][emn];
                                        M(Q.m[ii], mm + size, nn + size) -=
                                            twojP1 * M(den_tmp, ss, rr) *
                                            self->gauntLSSL_JK
                                                .K[jr][ir][esn][emr];
                                        if (mm != nn)
                                        {
                                            M(Q.m[ii], nn + size, mm) +=
                                                twojP1 *
                                                    M(den_tmp, size_tmp2 + ss,
                                                      rr) *
                                                    self->gauntLSLS_JK
                                                        .J[ir][jr][emn][ers] +
                                                twojP1 *
                                                    M(den_tmp, ss,
                                                      size_tmp2 + rr) *
                                                    self->gauntLSSL_JK
                                                        .J[jr][ir][esr][enm];
                                        }
                                    }
                                }
                            M(Q.m[ii], nn, mm) = M(Q.m[ii], mm, nn);
                            M(Q.m[ii], mm, nn + size) =
                                M(Q.m[ii], nn + size, mm);
                            M(Q.m[ii], nn, mm + size) =
                                M(Q.m[ii], mm + size, nn);
                            M(Q.m[ii], size + nn, size + mm) =
                                M(Q.m[ii], size + mm, size + nn);
                        }
                }
        }

        for (int ii = 0; ii < NOpenShells + 1; ii++)
        {
            double f_i;
            if (ii == 0)
                f_i = 1.0;
            else
                f_i = ca->f_list[ii - 1];
            mat_t fock_e = mat_clone(self->h1e_4c.m[Iirrep]);
            mat_add_inplace(fock_e, Q.m[0], 0.5);
            for (int jj = 1; jj < NOpenShells + 1; jj++)
            {
                double f_j;
                if (jj == ii)
                    f_j = (ca->NN_list[jj - 1] - 1.0) /
                          (ca->MM_list[jj - 1] - 1.0);
                else
                    f_j = ca->f_list[jj - 1];
                mat_add_inplace(fock_e, Q.m[jj], 0.5 * f_j);
            }
            double tmp = 0.0;
            for (int mm = 0; mm < fock_e.rows; mm++)
                for (int nn = 0; nn < fock_e.rows; nn++)
                {
                    tmp += M(fock_e, mm, nn) *
                           M(densityShells[ii].m[Iirrep], mm, nn);
                }
            ene += f_i * tmp * (self->irrep_list[Iirrep].two_j + 1);
            mat_free(&fock_e);
        }
        vmat_free(&Q);
    }

    return ene;
}
