/*
 * x2camf_c.c
 *
 * Public C API of the pure-C X2CAMF implementation.  The entry points and
 * numerics mirror the pybind11 interface of the original C++ code
 * (executables/pyx2camf.cpp); the computation is performed by the C
 * translation in csrc/.
 */

#include "x2camf_c.h"

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
