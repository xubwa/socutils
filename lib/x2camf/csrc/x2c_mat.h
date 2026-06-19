/*
 * x2c_mat.h
 *
 * Minimal dense-matrix layer for the C translation of X2CAMF.
 *
 * - mat_t is a value struct {rows, cols, d}; copying the struct aliases the
 *   same buffer.  Functions returning mat_t allocate fresh buffers that the
 *   caller owns and must release with mat_free().
 * - Storage is row major; use the M(A,i,j) accessor macro.
 * - Eigen equivalences used throughout the translation:
 *       MatrixXd A(n,m);            ->  mat_t A = mat_new(n,m);   (zeroed)
 *       A(i,j)                      ->  M(A,i,j)
 *       A * B                       ->  mat_mul(A,B)
 *       A.transpose()               ->  mat_transpose(A)
 *       A.adjoint()  (real)        ->  mat_transpose(A)
 *       A.inverse()                 ->  mat_inverse(A)
 *       A.block(i,j,r,c)            ->  mat_block(A,i,j,r,c)
 *       SelfAdjointEigenSolver      ->  mat_sym_eig (ascending eigenvalues)
 *       partialPivLu().solve(b)     ->  mat_solve_vec
 */

#ifndef X2C_MAT_H
#define X2C_MAT_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int rows, cols;
    double* d;
} mat_t;

#define M(A, i, j) ((A).d[(size_t)(i) * (size_t)(A).cols + (size_t)(j)])

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/* construction / destruction */
mat_t mat_new(int rows, int cols);   /* zero initialized */
mat_t mat_empty(void);               /* {0,0,NULL} */
void mat_free(mat_t* A);
mat_t mat_clone(mat_t A);
/* free *dst (if any) and replace it with src (ownership transfer) */
void mat_assign(mat_t* dst, mat_t src);
mat_t mat_identity(int n);
void mat_zero(mat_t A);

/* arithmetic (results are newly allocated) */
mat_t mat_mul(mat_t A, mat_t B);
mat_t mat_mul3(mat_t A, mat_t B, mat_t C);
mat_t mat_add(mat_t A, mat_t B);
mat_t mat_sub(mat_t A, mat_t B);
mat_t mat_scaled(mat_t A, double s);
mat_t mat_transpose(mat_t A);
/* A += s * B (in place) */
void mat_add_inplace(mat_t A, mat_t B, double s);
void mat_scale_inplace(mat_t A, double s);

/* blocks */
mat_t mat_block(mat_t A, int i0, int j0, int r, int c);
void mat_set_block(mat_t A, int i0, int j0, mat_t B);

/* linear algebra (LAPACK) */
mat_t mat_inverse(mat_t A);                            /* dgetrf + dgetri */
/* symmetric eigenproblem, eigenvalues ascending (dsyev);
   evals: caller-allocated length A.rows; *evecs: overwritten with a new
   matrix whose COLUMNS are the eigenvectors (like Eigen). */
void mat_sym_eig(mat_t A, double* evals, mat_t* evecs);
/* solve A x = b for a single right-hand side (dgesv); A is not modified */
void mat_solve_vec(mat_t A, const double* b, double* x);

/* max_ij |A_ij - B_ij| */
double mat_max_abs_diff(mat_t A, mat_t B);

/* array of matrices (Eigen vMatrixXd) */
typedef struct
{
    int n;
    mat_t* m;
} vmat_t;

vmat_t vmat_new(int n); /* n empty matrices */
void vmat_free(vmat_t* v);

/* fixed-length double vector (Eigen VectorXd) */
typedef struct
{
    int n;
    double* d;
} vecd_t;

vecd_t vecd_new(int n); /* zero initialized */
void vecd_free(vecd_t* v);

/* array of vectors (Eigen vVectorXd) */
typedef struct
{
    int n;
    vecd_t* v;
} vvecd_t;

vvecd_t vvecd_new(int n);
void vvecd_free(vvecd_t* v);

/* growable double array (std::vector<double>) */
typedef struct
{
    int n, cap;
    double* d;
} dvec_t;

void dvec_push(dvec_t* v, double x);
void dvec_free(dvec_t* v);

#ifdef __cplusplus
}
#endif

#endif /* X2C_MAT_H */
