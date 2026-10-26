/*
 * zgasci: full and generalized-active-space configuration interaction over
 * spinors, the kernels.  Plain C99, no dependencies but OpenMP and BLAS
 * (GEMM and AXPY, in zg_sigma_blas only; ZGASCI_NO_BLAS compiles without);
 * arrays come from the caller (socutils.hci.gasci drives them from Python,
 * as PySCF drives its FCI kernels).
 *
 * Spinors are numbered 0 .. n-1 and cut into consecutive sub-groups of at
 * most 16 (at most 32 sub-groups); the first ones form side A, the others
 * side B.  A determinant is
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
 * g = gas_sub[k], k in gas_ptr[c] .. gas_ptr[c+1]), with integer weights
 * w = gas_weight[k] (all 1 if gas_weight is NULL):
 * gas_min[c] <= sum w * (electrons in g) <= gas_max[c].  Weights express
 * e.g. an excitation level, N_active + 2 N_virtual <= k + 2.
 * labels[nsets * nspinor]: symmetry labels, set after set; the labels of the
 * occupied spinors add up to targets[s], modulo mods[s] if that is not 0.
 * Returns NULL on error (zg_error tells why).
 */
zg_plan *zg_plan_new(int nspinor, int nelec, int nsub, int nsub_a,
                     const int *sub_size, int ngas, const int *gas_ptr,
                     const int *gas_sub, const int *gas_weight,
                     const int *gas_min, const int *gas_max, int nsets,
                     const int *labels, const int *mods, const int *targets);
void zg_plan_free(zg_plan *plan);
const char *zg_error(void);

/*
 * A product space (PRODUCT_SPACE.md): the determinants |a, b> of every A
 * string a of str_a and every B string b of str_b with n(a) + n(b) = nelec
 * and the labels of the occupied spinors adding up to targets (labels,
 * mods, targets as in zg_plan_new).  Side A holds the spinors 0 .. nside_a
 * - 1, side B the others.  A string of a side is ceil(nside / 64) 64-bit
 * words, bit j (word j / 64) set if spinor j of the side is occupied.
 * Strings come in any order and must not repeat; those whose electron count
 * and labels pair with no string of the other side are left out.  Within a
 * side, strings are numbered by electron count, then labels, then value
 * (the highest word first); without labels and with one electron count per
 * side, the CI vector is the matrix C[a][b] over the sorted lists.  At most
 * 215 spinors on a side.  The kernels: hdiag, sigma (the BLAS kernel; the
 * reference and block entry points call it), rdm2 (by zg_rdm2_blas).
 */
zg_plan *zg_plan_product(int nspinor, int nelec, int nside_a, int64_t nstr_a,
                         const uint64_t *str_a, int64_t nstr_b,
                         const uint64_t *str_b, int nsets, const int *labels,
                         const int *mods, const int *targets);
/* Whether the plan is a product space. */
int zg_is_product(const zg_plan *plan);
/* Whether a product plan is its own mirror image: the same number of
 * spinors on both sides, the same strings, and spinor j of side B with the
 * labels of spinor j of side A (the spin flip of a spatial product space
 * with equal alpha and beta lists). */
int zg_is_mirror(const zg_plan *plan);
/* The strings of a side of a product plan in their order, words[n * nwords]
 * (nwords = ceil(nside / 64)), if words is not NULL; returns n. */
int64_t zg_product_strings(const zg_plan *plan, int side, uint64_t *words);
/* out[0..3]: words of a string of A and of B, entries of the operator
 * tables of A and of B. */
void zg_product_info(const zg_plan *plan, int64_t *out);

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
/* ia[dim], ib[dim]: the A and the B string (sid) of each determinant; for a
 * product plan, their places in zg_product_strings. */
void zg_det_strings(const zg_plan *plan, int64_t *ia, int64_t *ib);

/* out[dim]: the diagonal of H. */
void zg_hdiag_d(const zg_plan *plan, const double *v, double *out);
void zg_hdiag_z(const zg_plan *plan, const double _Complex *v, double *out);
/* sigma = H c. */
void zg_sigma_d(const zg_plan *plan, const double *v, const double *c,
                double *sigma);
void zg_sigma_z(const zg_plan *plan, const double _Complex *v,
                const double _Complex *c, double _Complex *sigma);
/* The same by the block kernel: every row of sigma written by one thread,
 * no atomic additions.  zg_block_ok: whether it applies, always 1 now (it
 * needed sides of at most 64 spinors before). */
int zg_block_ok(const zg_plan *plan);
void zg_sigma_block_d(const zg_plan *plan, const double *v, const double *c,
                      double *sigma);
void zg_sigma_block_z(const zg_plan *plan, const double _Complex *v,
                      const double _Complex *c, double _Complex *sigma);
/* The same by the BLAS kernel (zgasci_blas.inc; needs zg_block_ok, else the
 * reference kernel runs): the terms as gather - matrix product - scatter,
 * rows written by the thread that owns them.  Workspace: two vectors. */
void zg_sigma_blas_d(const zg_plan *plan, const double *v, const double *c,
                     double *sigma);
void zg_sigma_blas_z(const zg_plan *plan, const double _Complex *v,
                     const double _Complex *c, double _Complex *sigma);
/* sigma = H c for c with C[b][a] = parity C[a][b] (parity +1 or -1), on a
 * mirror plan (zg_is_mirror) with integrals that the exchange of the sides
 * maps to themselves and terms only of the kinds (AA,AA), (AB,AB), (BB,BB)
 * (spatial orbitals): sigma = X + parity X^T + (AB,AB) c with X the (AA,AA)
 * part, no transposed half.  Returns 1, or 0 (sigma untouched) if the plan
 * or the integrals do not qualify. */
int zg_sigma_flip_d(const zg_plan *plan, const double *v, const double *c,
                    double *sigma, int parity);
int zg_sigma_flip_z(const zg_plan *plan, const double _Complex *v,
                    const double _Complex *c, double _Complex *sigma, int parity);
/* out[npair * npair] = <bra| A+_P A_R |ket>, P the row. */
void zg_rdm2_d(const zg_plan *plan, const double *bra, const double *ket,
               double *out);
void zg_rdm2_z(const zg_plan *plan, const double _Complex *bra,
               const double _Complex *ket, double _Complex *out);
/* The same by the loops of the BLAS kernel (dot products, a W-shaped array
 * per kind of term and thread); any plan.  zg_rdm2 calls it for product
 * plans. */
void zg_rdm2_blas_d(const zg_plan *plan, const double *bra, const double *ket,
                    double *out);
void zg_rdm2_blas_z(const zg_plan *plan, const double _Complex *bra,
                    const double _Complex *ket, double _Complex *out);

#endif
