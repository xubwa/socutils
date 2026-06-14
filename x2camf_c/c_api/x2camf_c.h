/*
 * x2camf_c.h
 *
 * Plain-C API for the X2CAMF code (https://github.com/Warlocat/x2camf).
 *
 * This header exposes the same three entry points as the pybind11 module
 * (amfi, atm_integrals, pcc_K) through a C ABI so that the library can be
 * loaded from any language with a C FFI (python ctypes, julia, fortran,
 * rust, ...).  The implementation is still C++, but nothing C++-specific
 * leaks through this interface: only int / double scalars and arrays.
 *
 * Conventions
 * -----------
 * - `shell` is an integer array of length `nbas` holding the angular
 *   momentum l of every uncontracted (primitive) basis function, sorted by
 *   shell.  `exp_a` holds the corresponding Gaussian exponents.
 * - `nshell` is shell[nbas-1] + 1 (number of distinct l values).
 * - All output matrices are written ROW-MAJOR into caller-allocated
 *   buffers (directly compatible with C-ordered numpy arrays).
 * - The 2-spinor dimension of the j-adapted basis is
 *       n2c = sum_i (4*shell[i] + 2)
 *   which is also returned by x2camf_n2c().  Two-component matrices are
 *   n2c x n2c, four-component matrices are (2*n2c) x (2*n2c).
 * - `soc_int_flavor` is the same bit-coded integer used by the pybind11
 *   interface.  For x2camf_amfi / x2camf_pcc_k:
 *       bit 0: with_gaunt        bit 1: with_gauge
 *       bit 2: gaussian_nuclear  bit 3: aoc (average-of-configuration)
 *       bit 4: pt                bit 5: pcc
 *       bit 6: int4c             bit 7: with_gaunt_sd
 *   For x2camf_atm_integrals:
 *       bit 0: with_gaunt        bit 1: with_gauge
 *       bit 2: gaussian_nuclear  bit 3: aoc
 *       bit 4: with_gaunt_sd
 * - Every function returns 0 on success and a non-zero error code
 *   otherwise; x2camf_error_message() maps codes to human-readable text.
 */

#ifndef X2CAMF_C_H
#define X2CAMF_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define X2CAMF_SUCCESS            0
#define X2CAMF_ERR_BAD_ARGUMENT   1   /* invalid atom number, nbas, ... */
#define X2CAMF_ERR_PCC_WITH_PT    2   /* pcc not implemented for pt scheme */
#define X2CAMF_ERR_INTERNAL       3   /* unexpected C++ exception */
#define X2CAMF_ERR_DIM_MISMATCH   4   /* internal dimension check failed */

/* Default value used when speed_of_light <= 0 is passed. */
#define X2CAMF_SPEED_OF_LIGHT_DEFAULT 137.0359991

/* Human-readable message for an error code (static storage, do not free). */
const char* x2camf_error_message(int code);

/* Number of 2-spinor basis functions: sum_i (4*shell[i] + 2). */
int x2camf_n2c(int nbas, const int* shell);

/*
 * Spin-orbit integrals in the X2CAMF scheme.
 *
 * amfi_out must hold dim*dim doubles where dim = n2c (or 2*n2c when the
 * int4c bit of soc_int_flavor is set).
 */
int x2camf_amfi(int soc_int_flavor, int atom_number, int nshell, int nbas,
                int print_level, const int* shell, const double* exp_a,
                double speed_of_light, double* amfi_out);

/*
 * Exchange-only two-electron picture-change-correction matrix.
 * Same buffer convention as x2camf_amfi.
 */
int x2camf_pcc_k(int soc_int_flavor, int atom_number, int nshell, int nbas,
                 int print_level, const int* shell, const double* exp_a,
                 double speed_of_light, double* pcc_out);

/*
 * All atomic integrals from the converged atomic 4c-DHF calculation.
 *
 * `outs` is an array of 13 caller-allocated buffers, written in this
 * order (2c = n2c x n2c, 4c = 2*n2c x 2*n2c, all row-major):
 *
 *   idx  name        dim   description
 *   ---  ----------  ----  ------------------------------------------
 *    0   atm_X       2c    X matrix from the atomic 4c Fock matrix
 *    1   atm_R       2c    R matrix from atm_X
 *    2   h1e_4c      4c    4c one-electron Hamiltonian
 *    3   fock_4c     4c    4c Fock matrix (h1e_4c + fock_4c_2e)
 *    4   fock_2c     2c    FW transformed fock_4c
 *    5   fock_4c_2e  4c    4c effective two-electron Veff
 *    6   fock_2c_2e  2c    2c Veff obtained using den_2c
 *    7   fock_4c_K   4c    4c exchange part of Coulomb + entire Breit
 *    8   fock_2c_K   2c    2c exchange obtained using den_2c
 *    9   so_4c       4c    spin-dependent Coulomb + entire Breit term
 *   10   so_2c       2c    FW transformed so_4c
 *   11   den_4c      4c    4c density matrix
 *   12   den_2c      2c    FW transformed den_4c
 */
#define X2CAMF_N_ATM_INTEGRALS 13

int x2camf_atm_integrals(int soc_int_flavor, int atom_number, int nshell,
                         int nbas, int print_level, const int* shell,
                         const double* exp_a, double speed_of_light,
                         int spin_free, double* const* outs);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* X2CAMF_C_H */
