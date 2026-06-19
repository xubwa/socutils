/*
 * x2c_general.h
 *
 * C translation of include/general.h from the X2CAMF code: shared types,
 * factorials, Wigner n-j symbols, generic matrix utilities, the X2C
 * one-electron transformation helpers and the irrep "unite" reordering.
 */

#ifndef X2C_GENERAL_H
#define X2C_GENERAL_H

#include <stdbool.h>

#include "x2c_mat.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPEED_OF_LIGHT_DEFAULT 137.0359991

/* Speed of light in atomic units; overwritten by every API entry point. */
extern double speedOfLight;

/* gtos in form of angular shell (Eigen struct intShell) */
typedef struct
{
    vecd_t exp_a, norm;
    mat_t coeff; /* (n_unc x n_con) contraction coefficients */
    int l;
} intShell;

/* Irreducible representation |j, l, m_j> */
typedef struct
{
    int l, two_j, two_mj, size;
} irrep_jm;

/* Coulomb and exchange integrals in [Iirrep][Jirrep][eij][ekl] layout.
   J[ir][jr] has shape [size_i*size_i][size_j*size_j] and K[ir][jr] has
   shape [size_i*size_j][size_i*size_j], where size_i/size_j are the
   numbers of primitives of compact irreps ir/jr (same as the C++ code). */
typedef struct
{
    double ****J, ****K;
} int2eJK;

/* Free an int2eJK allocated over Ncompact compact irreps whose primitive
   counts are sizes[0..Ncompact-1]. */
void int2e_free(int2eJK* h2e, const int* sizes, int Ncompact);

/* factorial and double factorial */
double factorial(int n);
double double_factorial(int n);

/* max_ij |M1_ij - M2_ij| (named after the C++ helper) */
double evaluateChange(mat_t M1, mat_t M2);
/* M^{-1/2} and M^{1/2} for a symmetric positive definite matrix */
mat_t matrix_half_inverse(mat_t inputM);
mat_t matrix_half(mat_t inputM);
/* Generalized eigensolver for M C = S C E given s_h_i = S^{-1/2}.
   values: caller-allocated, length inputM.rows.
   *vectors is replaced (previous content freed) by a new matrix whose
   columns are the eigenvectors in the original (non-orthogonal) basis. */
void eigensolverG(mat_t inputM, mat_t s_h_i, double* values, mat_t* vectors);

/* Wigner n-j symbols (CG namespace in C++); two_j arguments are 2*j */
bool CG_triangle_fails(int two_j1, int two_j2, int two_j3);
double CG_sqrt_delta(int two_j1, int two_j2, int two_j3);
double CG_wigner_3j_int(int l1, int l2, int l3, int m1, int m2, int m3);
double CG_wigner_3j_zeroM(int l1, int l2, int l3);
double CG_wigner_3j(int tj1, int tj2, int tj3, int tm1, int tm2, int tm3);
double CG_wigner_6j(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6);
double CG_wigner_9j(int tj1, int tj2, int tj3, int tj4, int tj5, int tj6,
                    int tj7, int tj8, int tj9);

/* X2C one-electron helpers (X2C namespace in C++) */
mat_t X2C_get_X(mat_t S, mat_t T, mat_t W, mat_t V);
mat_t X2C_get_X_from_coeff(mat_t coeff);
mat_t X2C_get_R(mat_t S, mat_t T, mat_t X);
mat_t X2C_get_R_4c(mat_t S_4c, mat_t X);
mat_t X2C_evaluate_h1e_x2c(mat_t S, mat_t T, mat_t W, mat_t V, mat_t X,
                           mat_t R);
mat_t X2C_transform_4c_2c(mat_t M_4c, mat_t XXX, mat_t RRR);

/* Rotate::unite_irrep / unite_irrep_4c for double matrices */
mat_t Rotate_unite_irrep(const vmat_t* inputM, const irrep_jm* irrep_list,
                         int Nirrep);
mat_t Rotate_unite_irrep_4c(const vmat_t* inputM, const irrep_jm* irrep_list,
                            int Nirrep);

/* Element symbols, elem_list[1] == "H" ... elem_list[118] == "OG" */
extern const char* const elem_list[119];

#ifdef __cplusplus
}
#endif

#endif /* X2C_GENERAL_H */
