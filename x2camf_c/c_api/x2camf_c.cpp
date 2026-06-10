/*
 * x2camf_c.cpp
 *
 * C ABI wrapper around the X2CAMF C++ core.  The numerical code is
 * identical to the pybind11 interface (executables/pyx2camf.cpp in the
 * upstream repository); only the data marshalling differs: Eigen matrices
 * are copied row-major into caller-allocated plain double buffers.
 */

#include "x2camf_c.h"

#include "dhf_sph.h"
#include "dhf_sph_ca.h"
#include "general.h"
#include "int_sph.h"

#include <Eigen/Core>
#include <bitset>
#include <iostream>
#include <memory>
#include <vector>

using namespace Eigen;
using namespace std;

namespace
{

using RowMajorMap = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>;

/* Copy an Eigen matrix into a caller-provided row-major buffer. */
void copy_out(const MatrixXd& mat, double* out)
{
    RowMajorMap(out, mat.rows(), mat.cols()) = mat;
}

int check_args(int atom_number, int nbas, const int* shell,
               const double* exp_a)
{
    if (atom_number < 1 || atom_number > 118) return X2CAMF_ERR_BAD_ARGUMENT;
    if (nbas <= 0 || shell == nullptr || exp_a == nullptr)
        return X2CAMF_ERR_BAD_ARGUMENT;
    for (int i = 0; i < nbas; i++)
        if (shell[i] < 0) return X2CAMF_ERR_BAD_ARGUMENT;
    return X2CAMF_SUCCESS;
}

void set_speed_of_light(double speed_of_light)
{
    speedOfLight = (speed_of_light > 0.0) ? speed_of_light
                                          : X2CAMF_SPEED_OF_LIGHT_DEFAULT;
}

/* Run the atomic 4c/2c SCF shared by all entry points. */
unique_ptr<DHF_SPH> run_scf(INT_SPH& intor, int print_level, bool spinFree,
                            bool twoC, bool Gaunt, bool gauge, bool gauNuc,
                            bool aoc)
{
    const bool allint = true, renormS = false;
    unique_ptr<DHF_SPH> scfer;
    if (aoc)
        scfer.reset(new DHF_SPH_CA(intor, "input", print_level, spinFree,
                                   twoC, Gaunt, gauge, allint, gauNuc));
    else
        scfer.reset(new DHF_SPH(intor, "input", print_level, spinFree, twoC,
                                Gaunt, gauge, allint, gauNuc));
    scfer->convControl = 1e-9;
    scfer->runSCF(twoC, renormS);
    return scfer;
}

} // namespace

extern "C" {

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
    case X2CAMF_ERR_INTERNAL:     return "internal error (C++ exception)";
    case X2CAMF_ERR_DIM_MISMATCH: return "internal dimension check failed";
    default:                      return "unknown error code";
    }
}

int x2camf_n2c(int nbas, const int* shell)
{
    if (nbas <= 0 || shell == nullptr) return -1;
    int n2c = 0;
    for (int i = 0; i < nbas; i++)
    {
        if (shell[i] < 0) return -1;
        n2c += 4 * shell[i] + 2;
    }
    return n2c;
}

int x2camf_amfi(int soc_int_flavor, int atom_number, int nshell, int nbas,
                int print_level, const int* shell, const double* exp_a,
                double speed_of_light, double* amfi_out)
{
    int err = check_args(atom_number, nbas, shell, exp_a);
    if (err != X2CAMF_SUCCESS) return err;
    if (amfi_out == nullptr) return X2CAMF_ERR_BAD_ARGUMENT;

    const auto cfg = bitset<8>(soc_int_flavor);
    const bool Gaunt = cfg[0], gauge = cfg[1], gauNuc = cfg[2], aoc = cfg[3],
               pt = cfg[4], pcc = cfg[5], int4c = cfg[6], sdGaunt = cfg[7];
    if (pt && pcc) return X2CAMF_ERR_PCC_WITH_PT;
    const bool spinFree = pt, twoC = pt;

    try
    {
        set_speed_of_light(speed_of_light);
        VectorXi shell_vec(nbas);
        VectorXd exp_a_vec(nbas);
        for (int i = 0; i < nbas; i++)
        {
            shell_vec(i) = shell[i];
            exp_a_vec(i) = exp_a[i];
            if (print_level >= 4)
                cout << shell[i] << " " << exp_a[i] << endl;
        }
        INT_SPH intor(atom_number, nshell, nbas, shell_vec, exp_a_vec);
        auto scfer = run_scf(intor, print_level, spinFree, twoC, Gaunt,
                             gauge, gauNuc, aoc);

        vMatrixXd amfi;
        if (!pcc)
            amfi = scfer->get_amfi_unc(intor, twoC, "partialFock", Gaunt,
                                       gauge, int4c, sdGaunt);
        else
            amfi = scfer->x2c2ePCC(int4c);
        MatrixXd amfi_all;
        if (int4c)
            amfi_all = Rotate::unite_irrep_4c(amfi, intor.irrep_list);
        else
            amfi_all = Rotate::unite_irrep(amfi, intor.irrep_list);

        const int n2c = x2camf_n2c(nbas, shell);
        const int dim = int4c ? 2 * n2c : n2c;
        if (amfi_all.rows() != dim || amfi_all.cols() != dim)
            return X2CAMF_ERR_DIM_MISMATCH;
        copy_out(amfi_all, amfi_out);

        if (print_level >= 4)
            cout << "x2camf_amfi finished normally." << endl;
    }
    catch (const std::exception& e)
    {
        cerr << "x2camf_amfi: " << e.what() << endl;
        return X2CAMF_ERR_INTERNAL;
    }
    return X2CAMF_SUCCESS;
}

int x2camf_pcc_k(int soc_int_flavor, int atom_number, int nshell, int nbas,
                 int print_level, const int* shell, const double* exp_a,
                 double speed_of_light, double* pcc_out)
{
    int err = check_args(atom_number, nbas, shell, exp_a);
    if (err != X2CAMF_SUCCESS) return err;
    if (pcc_out == nullptr) return X2CAMF_ERR_BAD_ARGUMENT;

    const auto cfg = bitset<8>(soc_int_flavor);
    const bool Gaunt = cfg[0], gauge = cfg[1], gauNuc = cfg[2], aoc = cfg[3],
               pt = cfg[4], pcc = cfg[5], int4c = cfg[6];
    if (pt && pcc) return X2CAMF_ERR_PCC_WITH_PT;
    const bool spinFree = pt, twoC = pt;

    try
    {
        set_speed_of_light(speed_of_light);
        VectorXi shell_vec(nbas);
        VectorXd exp_a_vec(nbas);
        for (int i = 0; i < nbas; i++)
        {
            shell_vec(i) = shell[i];
            exp_a_vec(i) = exp_a[i];
            if (print_level >= 4)
                cout << shell[i] << " " << exp_a[i] << endl;
        }
        INT_SPH intor(atom_number, nshell, nbas, shell_vec, exp_a_vec);
        auto scfer = run_scf(intor, print_level, spinFree, twoC, Gaunt,
                             gauge, gauNuc, aoc);

        vMatrixXd amfi = scfer->x2c2ePCC_K(int4c);
        MatrixXd amfi_all;
        if (int4c)
            amfi_all = Rotate::unite_irrep_4c(amfi, intor.irrep_list);
        else
            amfi_all = Rotate::unite_irrep(amfi, intor.irrep_list);

        const int n2c = x2camf_n2c(nbas, shell);
        const int dim = int4c ? 2 * n2c : n2c;
        if (amfi_all.rows() != dim || amfi_all.cols() != dim)
            return X2CAMF_ERR_DIM_MISMATCH;
        copy_out(amfi_all, pcc_out);

        if (print_level >= 4)
            cout << "x2camf_pcc_k finished normally." << endl;
    }
    catch (const std::exception& e)
    {
        cerr << "x2camf_pcc_k: " << e.what() << endl;
        return X2CAMF_ERR_INTERNAL;
    }
    return X2CAMF_SUCCESS;
}

int x2camf_atm_integrals(int soc_int_flavor, int atom_number, int nshell,
                         int nbas, int print_level, const int* shell,
                         const double* exp_a, double speed_of_light,
                         int spin_free, double* const* outs)
{
    int err = check_args(atom_number, nbas, shell, exp_a);
    if (err != X2CAMF_SUCCESS) return err;
    if (outs == nullptr) return X2CAMF_ERR_BAD_ARGUMENT;
    for (int i = 0; i < X2CAMF_N_ATM_INTEGRALS; i++)
        if (outs[i] == nullptr) return X2CAMF_ERR_BAD_ARGUMENT;

    const auto cfg = bitset<8>(soc_int_flavor);
    const bool Gaunt = cfg[0], gauge = cfg[1], gauNuc = cfg[2], aoc = cfg[3],
               sdGaunt = cfg[4];
    const bool twoC = false;
    const bool spinFree = (spin_free != 0); // 4c SF-DHF when true; 4c DC otherwise

    try
    {
        set_speed_of_light(speed_of_light);
        VectorXi shell_vec(nbas);
        VectorXd exp_a_vec(nbas);
        for (int i = 0; i < nbas; i++)
        {
            shell_vec(i) = shell[i];
            exp_a_vec(i) = exp_a[i];
            if (print_level >= 4)
                cout << shell[i] << " " << exp_a[i] << endl;
        }
        INT_SPH intor(atom_number, nshell, nbas, shell_vec, exp_a_vec);
        auto scfer = run_scf(intor, print_level, spinFree, twoC, Gaunt,
                             gauge, gauNuc, aoc);

        auto so_2c = Rotate::unite_irrep(
            scfer->get_amfi_unc(intor, twoC, "partialFock", Gaunt, gauge,
                                false, sdGaunt),
            intor.irrep_list);
        // The den_fw should be called after so_2c, since the X and R
        // matrices are not available before that.
        vMatrixXd den_fw = scfer->get_density_fw();
        auto fock_2c = Rotate::unite_irrep(scfer->get_fock_fw(),
                                           intor.irrep_list);
        auto den_2c = Rotate::unite_irrep(den_fw, intor.irrep_list);
        auto fock_2c_2e = Rotate::unite_irrep(
            scfer->get_fock_4c_2ePart(den_fw, true), intor.irrep_list);
        auto fock_2c_K = Rotate::unite_irrep(
            scfer->get_fock_4c_K(den_fw, true), intor.irrep_list);

        auto atm_X = Rotate::unite_irrep(scfer->get_X(), intor.irrep_list);
        auto atm_R = Rotate::unite_irrep(scfer->get_R(), intor.irrep_list);

        auto so_4c = Rotate::unite_irrep_4c(
            scfer->get_amfi_unc(intor, twoC, "partialFock", Gaunt, gauge,
                                true, sdGaunt),
            intor.irrep_list);
        auto fock_4c = Rotate::unite_irrep_4c(scfer->get_fock_4c(),
                                              intor.irrep_list);
        auto den_4c = Rotate::unite_irrep_4c(scfer->get_density(),
                                             intor.irrep_list);
        auto h1e_4c = Rotate::unite_irrep_4c(scfer->get_h1e_4c(),
                                             intor.irrep_list);
        auto fock_4c_2e = Rotate::unite_irrep_4c(
            scfer->get_fock_4c_2ePart(scfer->get_density(), false),
            intor.irrep_list);
        auto fock_4c_K = Rotate::unite_irrep_4c(
            scfer->get_fock_4c_K(scfer->get_density(), false),
            intor.irrep_list);

        const MatrixXd* results[X2CAMF_N_ATM_INTEGRALS] = {
            &atm_X,      &atm_R,      &h1e_4c,    &fock_4c,   &fock_2c,
            &fock_4c_2e, &fock_2c_2e, &fock_4c_K, &fock_2c_K, &so_4c,
            &so_2c,      &den_4c,     &den_2c};

        const int n2c = x2camf_n2c(nbas, shell);
        for (int i = 0; i < X2CAMF_N_ATM_INTEGRALS; i++)
        {
            const int dim = results[i]->rows();
            if (dim != n2c && dim != 2 * n2c)
                return X2CAMF_ERR_DIM_MISMATCH;
            copy_out(*results[i], outs[i]);
        }

        if (print_level >= 4)
            cout << "x2camf_atm_integrals finished normally." << endl;
    }
    catch (const std::exception& e)
    {
        cerr << "x2camf_atm_integrals: " << e.what() << endl;
        return X2CAMF_ERR_INTERNAL;
    }
    return X2CAMF_SUCCESS;
}

} // extern "C"
