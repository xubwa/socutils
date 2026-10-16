/* The BLAS kernel of src/zgasci restricted to some target rows and some
 * source rows, for a CI vector distributed over processes (fci_mpi.h): the
 * target rows are those a process owns, the source rows a slice of the
 * vector that it holds at the time.  Plain C, as the kernel; gas_dist.c. */
#ifndef ZSHCI_SRC_GAS_DIST_H_
#define ZSHCI_SRC_GAS_DIST_H_

#include <stdint.h>

#include "zgasci/zgasci.h"

/* The matrices W of the nine kinds of terms for one v (zg_sigma_blas builds
 * them at every call), the settings of the kernel from the environment, and
 * the multiply-adds of the calls so far: stat[0..3] in products, dot
 * products, axpy, and the elements scattered. */
typedef struct {
  void *w[9];
  int live[9];
  int64_t cols;
  double direct_ratio, direct_small;
  double stat[4];
} zgd_matrices;

zgd_matrices *zgd_matrices_new_d(const zg_plan *p, const double *v);
zgd_matrices *zgd_matrices_new_z(const zg_plan *p, const double _Complex *v);
void zgd_matrices_free(zgd_matrices *m);

/* sv += (the kinds of half h) cv for the target rows of the row buckets t0
 * .. t1 - 1 of half h and the source rows of its row buckets s0 .. s1 - 1.
 * Half 0 works on the vector, half 1 on the transposed vector (rows the B
 * strings, blockoff_t); sv holds its elements from sbase on, cv from cbase
 * on, and only those are touched. */
void zgd_half_d(const zg_plan *p, zgd_matrices *m, int h, int t0, int t1,
                int s0, int s1, const double *cv, int64_t cbase, double *sv,
                int64_t sbase);
void zgd_half_z(const zg_plan *p, zgd_matrices *m, int h, int t0, int t1,
                int s0, int s1, const double _Complex *cv, int64_t cbase,
                double _Complex *sv, int64_t sbase);

/* The same, adding the seconds spent on the target rows of each row bucket
 * to bucket_seconds[bucket] (if not NULL), and calling progress (if not
 * NULL) from the master thread now and then: one-sided communication may
 * need it to go on. */
void zgd_half_timed_d(const zg_plan *p, zgd_matrices *m, int h, int t0,
                      int t1, int s0, int s1, const double *cv, int64_t cbase,
                      double *sv, int64_t sbase, double *bucket_seconds,
                      void (*progress)(void));
void zgd_half_timed_z(const zg_plan *p, zgd_matrices *m, int h, int t0,
                      int t1, int s0, int s1, const double _Complex *cv,
                      int64_t cbase, double _Complex *sv, int64_t sbase,
                      double *bucket_seconds, void (*progress)(void));

/* out = the diagonal of H for the A buckets b0 .. b1 - 1, out[0] the
 * element at blockoff of the first of their blocks (base). */
void zgd_hdiag_d(const zg_plan *p, const double *v, int b0, int b1,
                 double *out, int64_t base);
void zgd_hdiag_z(const zg_plan *p, const double _Complex *v, int b0, int b1,
                 double *out, int64_t base);

#endif
