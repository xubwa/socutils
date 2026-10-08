/*
 * x2camf_c.c
 *
 * Public C API of the pure-C X2CAMF implementation.  The entry points and
 * numerics mirror the pybind11 interface of the original C++ code
 * (executables/pyx2camf.cpp); the computation is performed by the C
 * translation in csrc/.
 */

#include "x2camf_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "x2c_dhf_sph.h"
#include "x2c_general.h"
#include "x2c_int_sph.h"
#include "x2c_mat.h"

static int check_args(int atom_number, int nbas, const int* shell,
                      const double* exp_a)
{
    if (atom_number < 1 || atom_number > 118) return X2CAMF_ERR_BAD_ARGUMENT;
    if (nbas <= 0 || shell == NULL || exp_a == NULL)
        return X2CAMF_ERR_BAD_ARGUMENT;
    for (int i = 0; i < nbas; i++)
        if (shell[i] < 0) return X2CAMF_ERR_BAD_ARGUMENT;
    return X2CAMF_SUCCESS;
}

static void set_speed_of_light(double speed_of_light)
{
    speedOfLight = (speed_of_light > 0.0) ? speed_of_light
                                          : X2CAMF_SPEED_OF_LIGHT_DEFAULT;
}

/* copy mat (row major) into out buffer */
static void copy_out(mat_t mat, double* out)
{
    memcpy(out, mat.d, (size_t)mat.rows * mat.cols * sizeof(double));
}

const char* x2camf_error_message(int code)
{
    switch (code)
    {
    case X2CAMF_SUCCESS:          return "success";
    case X2CAMF_ERR_BAD_ARGUMENT: return "invalid argument (atom number must "
                                         "be in [1,118], nbas > 0, shell "
                                         "entries >= 0, no NULL pointers)";
    case X2CAMF_ERR_PCC_WITH_PT:  return "PCC is not implemented for the PT "
                                         "scheme";
    case X2CAMF_ERR_INTERNAL:     return "internal error";
    case X2CAMF_ERR_DIM_MISMATCH: return "internal dimension check failed";
    default:                      return "unknown error code";
    }
}

int x2camf_n2c(int nbas, const int* shell)
{
    if (nbas <= 0 || shell == NULL) return -1;
    int n2c = 0;
    for (int i = 0; i < nbas; i++)
    {
        if (shell[i] < 0) return -1;
        n2c += 4 * shell[i] + 2;
    }
    return n2c;
}

/* decoded soc_int_flavor bits for amfi / pcc_k */
typedef struct
{
    bool Gaunt, gauge, gauNuc, aoc, pt, pcc, int4c, sdGaunt;
    bool spinFree, twoC;
} amfi_flags;

static int decode_flags(int soc_int_flavor, amfi_flags* f)
{
    f->Gaunt = (soc_int_flavor >> 0) & 1;
    f->gauge = (soc_int_flavor >> 1) & 1;
    f->gauNuc = (soc_int_flavor >> 2) & 1;
    f->aoc = (soc_int_flavor >> 3) & 1;
    f->pt = (soc_int_flavor >> 4) & 1;
    f->pcc = (soc_int_flavor >> 5) & 1;
    f->int4c = (soc_int_flavor >> 6) & 1;
    f->sdGaunt = (soc_int_flavor >> 7) & 1;
    if (f->pt && f->pcc) return X2CAMF_ERR_PCC_WITH_PT;
    f->spinFree = f->pt;
    f->twoC = f->pt;
    return X2CAMF_SUCCESS;
}

static DHF_SPH* run_scf(INT_SPH* intor, int print_level, bool spinFree,
                        bool twoC, bool Gaunt, bool gauge, bool gauNuc,
                        bool aoc)
{
    const bool allint = true, renormS = false;
    DHF_SPH* scfer;
    if (aoc)
        scfer = dhf_sph_ca_new(intor, "input", print_level, spinFree, twoC,
                               Gaunt, gauge, allint, gauNuc);
    else
        scfer = dhf_sph_new(intor, "input", print_level, spinFree, twoC,
                            Gaunt, gauge, allint, gauNuc);
    scfer->convControl = 1e-9;
    dhf_sph_run_scf(scfer, twoC, renormS);
    return scfer;
}

int x2camf_amfi(int soc_int_flavor, int atom_number, int nshell, int nbas,
                int print_level, const int* shell, const double* exp_a,
                double speed_of_light, double* amfi_out)
{
    int err = check_args(atom_number, nbas, shell, exp_a);
    if (err != X2CAMF_SUCCESS) return err;
    if (amfi_out == NULL) return X2CAMF_ERR_BAD_ARGUMENT;

    amfi_flags f;
    err = decode_flags(soc_int_flavor, &f);
    if (err != X2CAMF_SUCCESS) return err;

    set_speed_of_light(speed_of_light);
    if (print_level >= 4)
        for (int i = 0; i < nbas; i++)
            printf("%d %g\n", shell[i], exp_a[i]);

    INT_SPH* intor = int_sph_new(atom_number, nshell, nbas, shell, exp_a);
    DHF_SPH* scfer = run_scf(intor, print_level, f.spinFree, f.twoC, f.Gaunt,
                             f.gauge, f.gauNuc, f.aoc);

    vmat_t amfi;
    if (!f.pcc)
        amfi = dhf_sph_get_amfi_unc(scfer, intor, f.twoC, "partialFock",
                                    f.Gaunt, f.gauge, f.int4c, f.sdGaunt);
    else
        amfi = dhf_sph_x2c2ePCC(scfer, f.int4c, NULL);
    mat_t amfi_all;
    if (f.int4c)
        amfi_all = Rotate_unite_irrep_4c(&amfi, intor->irrep_list,
                                         intor->Nirrep);
    else
        amfi_all = Rotate_unite_irrep(&amfi, intor->irrep_list, intor->Nirrep);

    const int n2c = x2camf_n2c(nbas, shell);
    const int dim = f.int4c ? 2 * n2c : n2c;
    int ret = X2CAMF_SUCCESS;
    if (amfi_all.rows != dim || amfi_all.cols != dim)
        ret = X2CAMF_ERR_DIM_MISMATCH;
    else
        copy_out(amfi_all, amfi_out);

    if (print_level >= 4 && ret == X2CAMF_SUCCESS)
        printf("x2camf_amfi finished normally.\n");

    mat_free(&amfi_all);
    vmat_free(&amfi);
    dhf_sph_free(scfer);
    int_sph_free(intor);
    return ret;
}

int x2camf_pcc_k(int soc_int_flavor, int atom_number, int nshell, int nbas,
                 int print_level, const int* shell, const double* exp_a,
                 double speed_of_light, double* pcc_out)
{
    int err = check_args(atom_number, nbas, shell, exp_a);
    if (err != X2CAMF_SUCCESS) return err;
    if (pcc_out == NULL) return X2CAMF_ERR_BAD_ARGUMENT;

    amfi_flags f;
    err = decode_flags(soc_int_flavor, &f);
    if (err != X2CAMF_SUCCESS) return err;

    set_speed_of_light(speed_of_light);
    if (print_level >= 4)
        for (int i = 0; i < nbas; i++)
            printf("%d %g\n", shell[i], exp_a[i]);

    INT_SPH* intor = int_sph_new(atom_number, nshell, nbas, shell, exp_a);
    DHF_SPH* scfer = run_scf(intor, print_level, f.spinFree, f.twoC, f.Gaunt,
                             f.gauge, f.gauNuc, f.aoc);

    vmat_t amfi = dhf_sph_x2c2ePCC_K(scfer, f.int4c, NULL);
    mat_t amfi_all;
    if (f.int4c)
        amfi_all = Rotate_unite_irrep_4c(&amfi, intor->irrep_list,
                                         intor->Nirrep);
    else
        amfi_all = Rotate_unite_irrep(&amfi, intor->irrep_list, intor->Nirrep);

    const int n2c = x2camf_n2c(nbas, shell);
    const int dim = f.int4c ? 2 * n2c : n2c;
    int ret = X2CAMF_SUCCESS;
    if (amfi_all.rows != dim || amfi_all.cols != dim)
        ret = X2CAMF_ERR_DIM_MISMATCH;
    else
        copy_out(amfi_all, pcc_out);

    if (print_level >= 4 && ret == X2CAMF_SUCCESS)
        printf("x2camf_pcc_k finished normally.\n");

    mat_free(&amfi_all);
    vmat_free(&amfi);
    dhf_sph_free(scfer);
    int_sph_free(intor);
    return ret;
}

int x2camf_atm_integrals(int soc_int_flavor, int atom_number, int nshell,
                         int nbas, int print_level, const int* shell,
                         const double* exp_a, double speed_of_light,
                         int spin_free, double* const* outs)
{
    int err = check_args(atom_number, nbas, shell, exp_a);
    if (err != X2CAMF_SUCCESS) return err;
    if (outs == NULL) return X2CAMF_ERR_BAD_ARGUMENT;
    for (int i = 0; i < X2CAMF_N_ATM_INTEGRALS; i++)
        if (outs[i] == NULL) return X2CAMF_ERR_BAD_ARGUMENT;

    /* atm_integrals flavor encoding: bit4 = sdGaunt (cf. pyx2camf.cpp) */
    const bool Gaunt = (soc_int_flavor >> 0) & 1;
    const bool gauge = (soc_int_flavor >> 1) & 1;
    const bool gauNuc = (soc_int_flavor >> 2) & 1;
    const bool aoc = (soc_int_flavor >> 3) & 1;
    const bool sdGaunt = (soc_int_flavor >> 4) & 1;
    const bool twoC = false;
    const bool spinFree = (spin_free != 0);

    set_speed_of_light(speed_of_light);
    if (print_level >= 4)
        for (int i = 0; i < nbas; i++)
            printf("%d %g\n", shell[i], exp_a[i]);

    INT_SPH* intor = int_sph_new(atom_number, nshell, nbas, shell, exp_a);
    DHF_SPH* scfer = run_scf(intor, print_level, spinFree, twoC, Gaunt, gauge,
                             gauNuc, aoc);
    const irrep_jm* irr = intor->irrep_list;
    const int nirr = intor->Nirrep;

    vmat_t tmp;
    mat_t results[X2CAMF_N_ATM_INTEGRALS];

    tmp = dhf_sph_get_amfi_unc(scfer, intor, twoC, "partialFock", Gaunt,
                               gauge, false, sdGaunt);
    mat_t so_2c = Rotate_unite_irrep(&tmp, irr, nirr);
    vmat_free(&tmp);
    /* The den_fw should be called after so_2c, since the X and R matrices
       are not available before that. */
    vmat_t den_fw = dhf_sph_get_density_fw(scfer);
    tmp = dhf_sph_get_fock_fw(scfer);
    mat_t fock_2c = Rotate_unite_irrep(&tmp, irr, nirr);
    vmat_free(&tmp);
    mat_t den_2c = Rotate_unite_irrep(&den_fw, irr, nirr);
    tmp = dhf_sph_get_fock_4c_2ePart(scfer, &den_fw, true);
    mat_t fock_2c_2e = Rotate_unite_irrep(&tmp, irr, nirr);
    vmat_free(&tmp);
    tmp = dhf_sph_get_fock_4c_K(scfer, &den_fw, true);
    mat_t fock_2c_K = Rotate_unite_irrep(&tmp, irr, nirr);
    vmat_free(&tmp);

    tmp = dhf_sph_get_X(scfer);
    mat_t atm_X = Rotate_unite_irrep(&tmp, irr, nirr);
    vmat_free(&tmp);
    tmp = dhf_sph_get_R(scfer);
    mat_t atm_R = Rotate_unite_irrep(&tmp, irr, nirr);
    vmat_free(&tmp);

    tmp = dhf_sph_get_amfi_unc(scfer, intor, twoC, "partialFock", Gaunt,
                               gauge, true, sdGaunt);
    mat_t so_4c = Rotate_unite_irrep_4c(&tmp, irr, nirr);
    vmat_free(&tmp);
    tmp = dhf_sph_get_fock_4c(scfer);
    mat_t fock_4c = Rotate_unite_irrep_4c(&tmp, irr, nirr);
    vmat_free(&tmp);
    vmat_t den_4c_v = dhf_sph_get_density(scfer);
    mat_t den_4c = Rotate_unite_irrep_4c(&den_4c_v, irr, nirr);
    tmp = dhf_sph_get_h1e_4c(scfer);
    mat_t h1e_4c = Rotate_unite_irrep_4c(&tmp, irr, nirr);
    vmat_free(&tmp);
    tmp = dhf_sph_get_fock_4c_2ePart(scfer, &den_4c_v, false);
    mat_t fock_4c_2e = Rotate_unite_irrep_4c(&tmp, irr, nirr);
    vmat_free(&tmp);
    tmp = dhf_sph_get_fock_4c_K(scfer, &den_4c_v, false);
    mat_t fock_4c_K = Rotate_unite_irrep_4c(&tmp, irr, nirr);
    vmat_free(&tmp);
    vmat_free(&den_4c_v);
    vmat_free(&den_fw);

    results[0] = atm_X;
    results[1] = atm_R;
    results[2] = h1e_4c;
    results[3] = fock_4c;
    results[4] = fock_2c;
    results[5] = fock_4c_2e;
    results[6] = fock_2c_2e;
    results[7] = fock_4c_K;
    results[8] = fock_2c_K;
    results[9] = so_4c;
    results[10] = so_2c;
    results[11] = den_4c;
    results[12] = den_2c;

    const int n2c = x2camf_n2c(nbas, shell);
    int ret = X2CAMF_SUCCESS;
    for (int i = 0; i < X2CAMF_N_ATM_INTEGRALS; i++)
    {
        const int dim = results[i].rows;
        if (dim != n2c && dim != 2 * n2c)
        {
            ret = X2CAMF_ERR_DIM_MISMATCH;
            break;
        }
        copy_out(results[i], outs[i]);
    }

    if (print_level >= 4 && ret == X2CAMF_SUCCESS)
        printf("x2camf_atm_integrals finished normally.\n");

    for (int i = 0; i < X2CAMF_N_ATM_INTEGRALS; i++) mat_free(&results[i]);
    dhf_sph_free(scfer);
    int_sph_free(intor);
    return ret;
}

static DHF_SPH* new_scf(INT_SPH* intor, const char* occ_file, int print_level,
                        bool spinFree, bool twoC, bool Gaunt, bool gauge,
                        bool gauNuc, bool aoc)
{
    const bool allint = true;
    if (aoc)
        return dhf_sph_ca_new(intor, occ_file, print_level, spinFree, twoC,
                              Gaunt, gauge, allint, gauNuc);
    return dhf_sph_new(intor, occ_file, print_level, spinFree, twoC, Gaunt,
                       gauge, allint, gauNuc);
}

int x2camf_atom_scf(int atom_number, int nshell, int nbas, int print_level,
                    const int* shell, const double* exp_a,
                    double speed_of_light, int scf_type, int spin_free,
                    int gaunt, int gauge, int gau_nuc, int aoc,
                    const char* occ_file, double conv, int max_iter,
                    int nuc_model, double nuc_p1, double nuc_p2,
                    double* e_scf, int* converged, double* eps, double* occ,
                    double* coeff, double* ovlp, double* fock, double* h1e,
                    double* den)
{
    int err = check_args(atom_number, nbas, shell, exp_a);
    if (err != X2CAMF_SUCCESS) return err;
    if (e_scf == NULL || converged == NULL || eps == NULL || occ == NULL ||
        coeff == NULL || ovlp == NULL || fock == NULL || h1e == NULL ||
        den == NULL)
        return X2CAMF_ERR_BAD_ARGUMENT;
    if (scf_type < 0 || scf_type > 2) return X2CAMF_ERR_BAD_ARGUMENT;
    if (nuc_model < X2C_NUC_POINT || nuc_model > X2C_NUC_SPHERE ||
        (nuc_model != X2C_NUC_POINT && (gau_nuc || nuc_p1 <= 0.0)))
        return X2CAMF_ERR_BAD_ARGUMENT;
    if (occ_file == NULL) occ_file = "";
    if (conv <= 0.0) conv = 1e-9;
    if (max_iter <= 0) max_iter = 100;

    set_speed_of_light(speed_of_light);
    const bool twoC = (scf_type != 0), renormS = false;
    INT_SPH* intor = int_sph_new(atom_number, nshell, nbas, shell, exp_a);
    if (nuc_model != X2C_NUC_POINT)
        int_sph_set_nuclear_model(intor, nuc_model, nuc_p1, nuc_p2);

    DHF_SPH* scfer;
    if (scf_type == 0)
        scfer = new_scf(intor, occ_file, print_level, spin_free != 0, false,
                        gaunt != 0, gauge != 0, gau_nuc != 0, aoc != 0);
    else if (scf_type == 1)
        scfer = new_scf(intor, occ_file, print_level, spin_free != 0, true,
                        false, false, gau_nuc != 0, aoc != 0);
    else
    {
        scfer = new_scf(intor, occ_file, print_level, false, true, false,
                        false, gau_nuc != 0, aoc != 0);
        DHF_SPH* scfer2 = dhf_sph_new(intor, occ_file, print_level, false,
                                      false, gaunt != 0, gauge != 0, true,
                                      gau_nuc != 0);
        scfer2->convControl = conv;
        scfer2->maxIter = max_iter;
        dhf_sph_run_scf(scfer2, false, renormS);
        vmat_t h1 = dhf_sph_get_h1e_4c(scfer);
        vmat_t amfso = dhf_sph_get_amfi_unc(scfer2, intor, false,
                                            "partialFock", gaunt != 0,
                                            gauge != 0, false, false);
        for (int ir = 0; ir < h1.n; ir++)
            mat_add_inplace(h1.m[ir], amfso.m[ir], 1.0);
        dhf_sph_set_h1e_4c(scfer, &h1, false);
        vmat_free(&h1);
        vmat_free(&amfso);
        dhf_sph_free(scfer2);
    }
    scfer->convControl = conv;
    scfer->maxIter = max_iter;
    dhf_sph_run_scf(scfer, twoC, renormS);

    /* ene_scf is only evaluated on convergence */
    *converged = scfer->converged ? 1 : 0;
    *e_scf = scfer->converged ? scfer->ene_scf : NAN;

    const irrep_jm* irr = scfer->irrep_list;
    for (int ir = 0; ir < scfer->Nirrep; ir += irr[ir].two_j + 1)
    {
        const int n = irr[ir].size, dim = twoC ? n : 2 * n;
        const int shift = twoC ? 0 : n; /* electronic solutions */
        const mat_t c = scfer->coeff.m[ir];
        for (int i = 0; i < n; i++)
        {
            eps[i] = scfer->ene_orb.v[ir].d[shift + i];
            occ[i] = i < scfer->occNumber.v[ir].n
                         ? scfer->occNumber.v[ir].d[i] : 0.0;
        }
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < n; j++)
                coeff[(size_t)i * n + j] = M(c, i, shift + j);
        const mat_t* src[4] = {&scfer->overlap_4c.m[ir], &scfer->fock_4c.m[ir],
                               &scfer->h1e_4c.m[ir], &scfer->density.m[ir]};
        double* dst[4] = {ovlp, fock, h1e, den};
        for (int k = 0; k < 4; k++)
        {
            if (src[k]->rows != dim || src[k]->cols != dim)
            {
                dhf_sph_free(scfer);
                int_sph_free(intor);
                return X2CAMF_ERR_DIM_MISMATCH;
            }
            memcpy(dst[k], src[k]->d, (size_t)dim * dim * sizeof(double));
        }
        eps += n;
        occ += n;
        coeff += (size_t)dim * n;
        ovlp += (size_t)dim * dim;
        fock += (size_t)dim * dim;
        h1e += (size_t)dim * dim;
        den += (size_t)dim * dim;
    }

    dhf_sph_free(scfer);
    int_sph_free(intor);
    return X2CAMF_SUCCESS;
}

int x2camf_atom_exp_grad(int atom_number, int nshell, int nbas,
                         int print_level, const int* shell,
                         const double* exp_a, double speed_of_light,
                         int spin_free, int gau_nuc, const char* occ_file,
                         double conv, int max_iter, int nuc_model,
                         double nuc_p1, double nuc_p2, double step,
                         double* e_scf, double* grad)
{
    int err = check_args(atom_number, nbas, shell, exp_a);
    if (err != X2CAMF_SUCCESS) return err;
    if (e_scf == NULL || grad == NULL) return X2CAMF_ERR_BAD_ARGUMENT;
    if (nuc_model < X2C_NUC_POINT || nuc_model > X2C_NUC_SPHERE ||
        (nuc_model != X2C_NUC_POINT && (gau_nuc || nuc_p1 <= 0.0)))
        return X2CAMF_ERR_BAD_ARGUMENT;
    if (occ_file == NULL) occ_file = "";
    if (conv <= 0.0) conv = 1e-9;
    if (max_iter <= 0) max_iter = 100;
    if (step <= 0.0) step = 1e-3;

    set_speed_of_light(speed_of_light);
    INT_SPH* intor = int_sph_new(atom_number, nshell, nbas, shell, exp_a);
    if (nuc_model != X2C_NUC_POINT)
        int_sph_set_nuclear_model(intor, nuc_model, nuc_p1, nuc_p2);
    DHF_SPH* scfer = new_scf(intor, occ_file, print_level, spin_free != 0,
                             false, false, false, gau_nuc != 0, false);
    scfer->convControl = conv;
    scfer->maxIter = max_iter;
    dhf_sph_run_scf(scfer, false, false);
    int ret = X2CAMF_SUCCESS;
    if (!scfer->converged)
    {
        *e_scf = NAN;
        ret = X2CAMF_ERR_INTERNAL;
    }
    else
    {
        *e_scf = scfer->ene_scf;
        if (dhf_sph_exponent_gradient(scfer, intor, step, grad) != 0)
            ret = X2CAMF_ERR_INTERNAL;
    }
    dhf_sph_free(scfer);
    int_sph_free(intor);
    return ret;
}
