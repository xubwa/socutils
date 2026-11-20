/* The kernel of gas_dist.h: blas_half of src/zgasci with restrictions
 * (gas_dist_half.inc), compiled for real and complex numbers as zgasci.c
 * compiles its kernels. */
#include "gas_dist.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zgasci/zgasci_plan.h"

#ifndef ZGASCI_NO_BLAS
#define ZG_HAVE_BLAS 1
void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);
void zgemm_(const char *, const char *, const int *, const int *, const int *,
            const void *, const void *, const int *, const void *, const int *,
            const void *, void *, const int *);
void daxpy_(const int *, const double *, const double *, const int *, double *,
            const int *);
void zaxpy_(const int *, const void *, const void *, const int *, void *,
            const int *);
#define ZG_DGEMM dgemm_
#define ZG_ZGEMM zgemm_
#define ZG_DAXPY daxpy_
#define ZG_ZAXPY zaxpy_
#else
#define ZG_HAVE_BLAS 0
#define ZG_NO_GEMM(ta, tb, m, n, k, al, a, lda, b, ldb, be, c, ldc) \
  ((void)(m), (void)(n), (void)(k), (void)(al), (void)(be))
#define ZG_NO_AXPY(n, a, x, incx, y, incy)                         \
  do {                                                             \
    (void)(incx), (void)(incy);                                    \
    for (int i_ = 0; i_ < *(n); ++i_) (y)[i_] += *(a) * (x)[i_];   \
  } while (0)
#define ZG_DGEMM ZG_NO_GEMM
#define ZG_ZGEMM ZG_NO_GEMM
#define ZG_DAXPY ZG_NO_AXPY
#define ZG_ZAXPY ZG_NO_AXPY
#endif
extern int MKL_Set_Num_Threads_Local(int) __attribute__((weak));
extern int openblas_set_num_threads_local(int) __attribute__((weak));
static int zg_blas_single(void) {
  if (MKL_Set_Num_Threads_Local) return MKL_Set_Num_Threads_Local(1);
  if (openblas_set_num_threads_local) return openblas_set_num_threads_local(1);
  return 0;
}
static void zg_blas_restore(int before) {
  if (MKL_Set_Num_Threads_Local) {
    MKL_Set_Num_Threads_Local(before);
  } else if (openblas_set_num_threads_local) {
    openblas_set_num_threads_local(before);
  }
}

int zg_string_occ(const zg_plan *p, int side, int64_t sid, int *occ);

static zgd_matrices *matrices_alloc(void) {
  zgd_matrices *m = calloc(1, sizeof(zgd_matrices));
  m->cols = 256;
  m->direct_ratio = 4.0;
  m->direct_small = 512.0;
  const char *text = getenv("ZGASCI_BLAS_COLS");
  if (text && atol(text) > 0) m->cols = atol(text);
  text = getenv("ZGASCI_BLAS_DIRECT");
  if (text) m->direct_ratio = atof(text);
  text = getenv("ZGASCI_BLAS_SMALL");
  if (text) m->direct_small = atof(text);
  return m;
}

void zgd_matrices_free(zgd_matrices *m) {
  if (!m) return;
  for (int k = 0; k < 9; ++k) free(m->w[k]);
  free(m);
}

#define ZG_SCALAR double
#define ZG_GEMM ZG_DGEMM
#define ZG_AXPY ZG_DAXPY
#define ZG_NAME(x) x##_d
#define ZG_REAL(x) (x)
#include "gas_dist_half.inc"
#include "gas_dist_api.inc"
#undef ZG_SCALAR
#undef ZG_GEMM
#undef ZG_AXPY
#undef ZG_NAME
#undef ZG_REAL

#include <complex.h>
#define ZG_SCALAR double complex
#define ZG_GEMM ZG_ZGEMM
#define ZG_AXPY ZG_ZAXPY
#define ZG_NAME(x) x##_z
#define ZG_REAL(x) creal(x)
#include "gas_dist_half.inc"
#include "gas_dist_api.inc"
