/*
 * zgasci: full and generalized-active-space configuration interaction over
 * spinors, the kernels.  Plain C99, no dependencies but OpenMP; arrays come
 * from the caller (socutils.hci.gasci drives them from Python, as PySCF
 * drives its FCI kernels).
 *
 * Spinors are numbered 0 .. n-1 and cut into consecutive sub-groups of at
 * most 16; the first ones form side A, the others side B.  A determinant is
 * a string of each side, a string a sub-string of each sub-group.  The class
 * of a string is the number of electrons in each of its sub-groups; within a
 * class, and within the strings of one symmetry label, strings are numbered
 * and a block of the CI vector is a dense matrix (A string, B string).
 * Which classes pair up is decided by the electron number and by the
 * constraints: each names a set of sub-groups and the least and the largest
 * number of electrons allowed in them together (a GAS space, or any union).
 * Neither the determinants nor the Hamiltonian are stored.
 *
 * The Hamiltonian is taken as
 *     H = sum_{P,R} v[P,R] A+_P A_R ,    A+_P = a+_p a+_q ,  P = (p < q),
 * with pairs numbered P = q (q - 1) / 2 + p and v the antisymmetrized
 * integrals <pq||rs> with the one-electron part folded in,
 *     (h_pr d_qs - h_ps d_qr - h_qr d_ps + h_qs d_pr) / (N - 1).
 * Its action is worked out through the determinants K of N - 2 electrons:
 * for each K, the coefficients of the determinants K + R are gathered,
 * multiplied by v, and scattered to the determinants K + P.  Electrons that
 * pass between the two sides need no special case.
 *
 * Functions ending in _d take real, in _z complex (double _Complex) arrays.
 */
#ifndef ZGASCI_H_
#define ZGASCI_H_

#include <stdint.h>

typedef struct zg_plan zg_plan;

/*
 * sub_size[nsub]: spinors of each sub-group, in order; the first nsub_a
 * sub-groups are side A.  Constraint c holds for the sub-groups
 * gas_sub[gas_ptr[c] .. gas_ptr[c+1]): gas_min[c] <= electrons <= gas_max[c].
 * labels[nsets * nspinor]: symmetry labels, set after set; the labels of the
 * occupied spinors add up to targets[s], modulo mods[s] if that is not 0.
 * Returns NULL on error (zg_error tells why).
 */
zg_plan *zg_plan_new(int nspinor, int nelec, int nsub, int nsub_a,
                     const int *sub_size, int ngas, const int *gas_ptr,
                     const int *gas_sub, const int *gas_min,
                     const int *gas_max, int nsets, const int *labels,
                     const int *mods, const int *targets);
void zg_plan_free(zg_plan *plan);
const char *zg_error(void);

/* Number of determinants, and of 64-bit words that hold one. */
int64_t zg_dim(const zg_plan *plan);
int zg_nwords(const zg_plan *plan);
/* Sizes of the scheme, for reports: out[0..7] = strings of A and of B,
 * classes of A and of B, buckets of A and of B, blocks, tasks. */
void zg_info(const zg_plan *plan, int64_t *out);
/* The determinants in the order of the CI vector: words[dim * nwords], bit
 * p of a determinant set if spinor p is occupied.  For tests and for
 * handing vectors to other programs.  The phase convention is that of the
 * creation operators in increasing order of the spinor applied to the
 * vacuum, a+_p1 a+_p2 ... |0> with p1 < p2 < ... */
void zg_dets(const zg_plan *plan, uint64_t *words);

/* out[dim]: the diagonal of H. */
void zg_hdiag_d(const zg_plan *plan, const double *v, double *out);
void zg_hdiag_z(const zg_plan *plan, const double _Complex *v, double *out);
/* sigma = H c. */
void zg_sigma_d(const zg_plan *plan, const double *v, const double *c,
                double *sigma);
void zg_sigma_z(const zg_plan *plan, const double _Complex *v,
                const double _Complex *c, double _Complex *sigma);
/* The same by the block kernel (each side at most 64 spinors, zg_block_ok):
 * every row of sigma written by one thread, no atomic additions. */
int zg_block_ok(const zg_plan *plan);
void zg_sigma_block_d(const zg_plan *plan, const double *v, const double *c,
                      double *sigma);
void zg_sigma_block_z(const zg_plan *plan, const double _Complex *v,
                      const double _Complex *c, double _Complex *sigma);
/* out[npair * npair] = <bra| A+_P A_R |ket>, P the row. */
void zg_rdm2_d(const zg_plan *plan, const double *bra, const double *ket,
               double *out);
void zg_rdm2_z(const zg_plan *plan, const double _Complex *bra,
               const double _Complex *ket, double _Complex *out);

#endif
