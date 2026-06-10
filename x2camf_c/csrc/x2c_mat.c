#include "x2c_mat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* LAPACK (Fortran) prototypes */
extern void dsyev_(const char* jobz, const char* uplo, const int* n, double* a,
                   const int* lda, double* w, double* work, const int* lwork,
                   int* info);
extern void dgetrf_(const int* m, const int* n, double* a, const int* lda,
                    int* ipiv, int* info);
extern void dgetri_(const int* n, double* a, const int* lda, const int* ipiv,
                    double* work, const int* lwork, int* info);
extern void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
                   int* ipiv, double* b, const int* ldb, int* info);

static void* xmalloc(size_t n)
{
    void* p = malloc(n > 0 ? n : 1);
    if (p == NULL)
    {
        fprintf(stderr, "x2camf: out of memory (%zu bytes)\n", n);
        exit(99);
    }
    return p;
}

mat_t mat_new(int rows, int cols)
{
    mat_t A;
    A.rows = rows;
    A.cols = cols;
    A.d = (double*)xmalloc((size_t)rows * cols * sizeof(double));
    memset(A.d, 0, (size_t)rows * cols * sizeof(double));
    return A;
}

mat_t mat_empty(void)
{
    mat_t A = {0, 0, NULL};
    return A;
}

void mat_free(mat_t* A)
{
    if (A == NULL) return;
    free(A->d);
    A->d = NULL;
    A->rows = A->cols = 0;
}

mat_t mat_clone(mat_t A)
{
    mat_t B;
    B.rows = A.rows;
    B.cols = A.cols;
    B.d = (double*)xmalloc((size_t)A.rows * A.cols * sizeof(double));
    memcpy(B.d, A.d, (size_t)A.rows * A.cols * sizeof(double));
    return B;
}

void mat_assign(mat_t* dst, mat_t src)
{
    mat_free(dst);
    *dst = src;
}

mat_t mat_identity(int n)
{
    mat_t A = mat_new(n, n);
    for (int i = 0; i < n; i++) M(A, i, i) = 1.0;
    return A;
}

void mat_zero(mat_t A)
{
    memset(A.d, 0, (size_t)A.rows * A.cols * sizeof(double));
}

mat_t mat_mul(mat_t A, mat_t B)
{
    if (A.cols != B.rows)
    {
        fprintf(stderr, "x2camf: mat_mul dimension mismatch (%dx%d)*(%dx%d)\n",
                A.rows, A.cols, B.rows, B.cols);
        exit(99);
    }
    mat_t C = mat_new(A.rows, B.cols);
    for (int i = 0; i < A.rows; i++)
        for (int k = 0; k < A.cols; k++)
        {
            const double aik = M(A, i, k);
            if (aik == 0.0) continue;
            const double* brow = &B.d[(size_t)k * B.cols];
            double* crow = &C.d[(size_t)i * C.cols];
            for (int j = 0; j < B.cols; j++) crow[j] += aik * brow[j];
        }
    return C;
}

mat_t mat_mul3(mat_t A, mat_t B, mat_t C)
{
    mat_t T = mat_mul(A, B);
    mat_t R = mat_mul(T, C);
    mat_free(&T);
    return R;
}

mat_t mat_add(mat_t A, mat_t B)
{
    mat_t C = mat_clone(A);
    for (size_t i = 0; i < (size_t)A.rows * A.cols; i++) C.d[i] += B.d[i];
    return C;
}

mat_t mat_sub(mat_t A, mat_t B)
{
    mat_t C = mat_clone(A);
    for (size_t i = 0; i < (size_t)A.rows * A.cols; i++) C.d[i] -= B.d[i];
    return C;
}

mat_t mat_scaled(mat_t A, double s)
{
    mat_t C = mat_clone(A);
    for (size_t i = 0; i < (size_t)A.rows * A.cols; i++) C.d[i] *= s;
    return C;
}

void mat_add_inplace(mat_t A, mat_t B, double s)
{
    for (size_t i = 0; i < (size_t)A.rows * A.cols; i++) A.d[i] += s * B.d[i];
}

void mat_scale_inplace(mat_t A, double s)
{
    for (size_t i = 0; i < (size_t)A.rows * A.cols; i++) A.d[i] *= s;
}

mat_t mat_transpose(mat_t A)
{
    mat_t C = mat_new(A.cols, A.rows);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < A.cols; j++) M(C, j, i) = M(A, i, j);
    return C;
}

mat_t mat_block(mat_t A, int i0, int j0, int r, int c)
{
    mat_t C = mat_new(r, c);
    for (int i = 0; i < r; i++)
        memcpy(&M(C, i, 0), &M(A, i0 + i, j0), (size_t)c * sizeof(double));
    return C;
}

void mat_set_block(mat_t A, int i0, int j0, mat_t B)
{
    for (int i = 0; i < B.rows; i++)
        memcpy(&M(A, i0 + i, j0), &M(B, i, 0), (size_t)B.cols * sizeof(double));
}

mat_t mat_inverse(mat_t A)
{
    if (A.rows != A.cols)
    {
        fprintf(stderr, "x2camf: mat_inverse on non-square matrix\n");
        exit(99);
    }
    const int n = A.rows;
    /* LAPACK is column major; inv(A^T)^T = inv(A), and our row-major buffer
       read as column major is exactly A^T, so no transposition is needed. */
    mat_t C = mat_clone(A);
    int* ipiv = (int*)xmalloc((size_t)n * sizeof(int));
    int info = 0;
    dgetrf_(&n, &n, C.d, &n, ipiv, &info);
    if (info == 0)
    {
        int lwork = -1;
        double wkopt;
        dgetri_(&n, C.d, &n, ipiv, &wkopt, &lwork, &info);
        lwork = (int)wkopt;
        double* work = (double*)xmalloc((size_t)lwork * sizeof(double));
        dgetri_(&n, C.d, &n, ipiv, work, &lwork, &info);
        free(work);
    }
    free(ipiv);
    if (info != 0)
    {
        fprintf(stderr, "x2camf: matrix inversion failed (info=%d)\n", info);
        exit(99);
    }
    return C;
}

void mat_sym_eig(mat_t A, double* evals, mat_t* evecs)
{
    if (A.rows != A.cols)
    {
        fprintf(stderr, "x2camf: mat_sym_eig on non-square matrix\n");
        exit(99);
    }
    const int n = A.rows;
    /* dsyev overwrites the input with eigenvectors stored column major:
       column k of the Fortran matrix is eigenvector k.  Interpreted through
       our row-major M() accessor that buffer is V^T, i.e. eigenvector k
       lives in ROW k, so transpose once at the end to match Eigen's
       convention (eigenvectors in columns). */
    mat_t W = mat_clone(A);
    int info = 0, lwork = -1;
    double wkopt;
    dsyev_("V", "U", &n, W.d, &n, evals, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    double* work = (double*)xmalloc((size_t)lwork * sizeof(double));
    dsyev_("V", "U", &n, W.d, &n, evals, work, &lwork, &info);
    free(work);
    if (info != 0)
    {
        fprintf(stderr, "x2camf: dsyev failed (info=%d)\n", info);
        exit(99);
    }
    mat_t V = mat_transpose(W);
    mat_free(&W);
    mat_assign(evecs, V);
}

void mat_solve_vec(mat_t A, const double* b, double* x)
{
    if (A.rows != A.cols)
    {
        fprintf(stderr, "x2camf: mat_solve_vec on non-square matrix\n");
        exit(99);
    }
    const int n = A.rows, one = 1;
    /* dgesv solves using the column-major interpretation of our buffer,
       which is A^T; pass the transpose so the system solved is A x = b. */
    mat_t T = mat_transpose(A);
    int* ipiv = (int*)xmalloc((size_t)n * sizeof(int));
    memcpy(x, b, (size_t)n * sizeof(double));
    int info = 0;
    dgesv_(&n, &one, T.d, &n, ipiv, x, &n, &info);
    free(ipiv);
    mat_free(&T);
    if (info != 0)
    {
        fprintf(stderr, "x2camf: dgesv failed (info=%d)\n", info);
        exit(99);
    }
}

double mat_max_abs_diff(mat_t A, mat_t B)
{
    double tmp = 0.0;
    for (size_t i = 0; i < (size_t)A.rows * A.cols; i++)
    {
        double diff = A.d[i] - B.d[i];
        if (diff < 0) diff = -diff;
        if (diff > tmp) tmp = diff;
    }
    return tmp;
}

vmat_t vmat_new(int n)
{
    vmat_t v;
    v.n = n;
    v.m = (mat_t*)xmalloc((size_t)n * sizeof(mat_t));
    for (int i = 0; i < n; i++) v.m[i] = mat_empty();
    return v;
}

void vmat_free(vmat_t* v)
{
    if (v == NULL || v->m == NULL) return;
    for (int i = 0; i < v->n; i++) mat_free(&v->m[i]);
    free(v->m);
    v->m = NULL;
    v->n = 0;
}

vecd_t vecd_new(int n)
{
    vecd_t v;
    v.n = n;
    v.d = (double*)xmalloc((size_t)n * sizeof(double));
    memset(v.d, 0, (size_t)n * sizeof(double));
    return v;
}

void vecd_free(vecd_t* v)
{
    if (v == NULL) return;
    free(v->d);
    v->d = NULL;
    v->n = 0;
}

vvecd_t vvecd_new(int n)
{
    vvecd_t v;
    v.n = n;
    v.v = (vecd_t*)xmalloc((size_t)n * sizeof(vecd_t));
    for (int i = 0; i < n; i++)
    {
        v.v[i].n = 0;
        v.v[i].d = NULL;
    }
    return v;
}

void vvecd_free(vvecd_t* v)
{
    if (v == NULL || v->v == NULL) return;
    for (int i = 0; i < v->n; i++) vecd_free(&v->v[i]);
    free(v->v);
    v->v = NULL;
    v->n = 0;
}

void dvec_push(dvec_t* v, double x)
{
    if (v->n == v->cap)
    {
        v->cap = v->cap ? 2 * v->cap : 16;
        double* nd = (double*)realloc(v->d, (size_t)v->cap * sizeof(double));
        if (nd == NULL)
        {
            fprintf(stderr, "x2camf: out of memory in dvec_push\n");
            exit(99);
        }
        v->d = nd;
    }
    v->d[v->n++] = x;
}

void dvec_free(dvec_t* v)
{
    free(v->d);
    v->d = NULL;
    v->n = v->cap = 0;
}
