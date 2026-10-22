/*
 * libzmr: OpenMP kernels of zmr.  All arrays are C ordered; complex numbers
 * are C99 double complex (numpy complex128).
 */
#ifndef ZMR_H
#define ZMR_H

#include <complex.h>
#include <stdint.h>

void zmr_set_num_threads(int n);
int zmr_get_max_threads(void);

/* out[v, p, target[p, I]] = sign[p, I] * vecs[v, I] for target[p, I] >= 0;
 * out (nvec, norb, ndst) is overwritten. */
void zmr_apply_op(const int64_t *target, const int64_t *sign, int norb,
                  int64_t nsrc, int64_t ndst, int64_t nvec,
                  const double complex *vecs, double complex *out);

/* out[v, x0 + x, :] = O_x vecs[v, :] for the operator strings
 * O_x = c_1 c_2 ... c_nop (applied right to left), c_k = a^+_{ops[x, k]} if
 * dagger[k] else a_{ops[x, k]}, acting on the nelec-electron strings of norb
 * orbitals (ascending integer order).  vecs (nvec, dim(nelec)),
 * out (nvec, nx_total, dim(nelec + #creators - #annihilators)); only the
 * rows x0 .. x0 + nx - 1 of every vector are written. */
void zmr_apply_strings(int norb, int nelec, int nop, const int *dagger, const int64_t *ops,
                       int64_t nx, int64_t nx_total, int64_t x0, int64_t nvec,
                       const double complex *vecs, double complex *out);

/* a[I, J] = - sum_{k, e} conj(u[I, k, e]) u[J, k, e] / (lam[k] + delta[e] - shifts[J]),
 * a (ns, ns) is overwritten.  col >= 0: only the column J = col is computed,
 * the others are set to zero.  Three forms, each for complex and (suffix _r)
 * real arrays:
 *
 *   zmr_weighted_gram  u (ns, nk, ne) given
 *   zmr_contract_t     u[J, k, e] = sum_X t[J, k, X] v[e, X], t (ns, nk, nx),
 *                      v (ne, nx); u is formed for one k at a time in
 *                      thread-private buffers and never stored, vanishing
 *                      rows t[J, k, :] are skipped
 *   zmr_contract_w     the same u, summed over e first:
 *                      W_k[X, Y] = sum_e conj(v[e, X]) v[e, Y] / (lam[k] + delta[e] - s),
 *                      a[I, J] = - sum_k t[I, k, :]^+ W_k t[J, k, :];
 *                      the cost of W does not grow with the number of states */
/* zmr_sigma: out = H q for q (dim, nvec) without storing H (the operator of
 * zmr_build_hmat, for any complex h1 and eri); out is overwritten.
 * String-driven through the strings with two electrons fewer, one zgemm
 * with the antisymmetrized integrals per string and batch of vectors. */
void zmr_sigma(int norb, int nelec, int64_t dim, const double complex *h1,
               const double complex *eri, int64_t nvec, const double complex *q,
               double complex *out);
/* The same product determinant by determinant, using that H is Hermitian
 * (the kernel before the string-driven one; its reference). */
void zmr_sigma_det(int norb, int nelec, int64_t dim, const double complex *h1,
                   const double complex *eri, int64_t nvec, const double complex *q,
                   double complex *out);
void zmr_w_ccvv(const double complex *bvc, int64_t ldvc, int64_t naux, int64_t nv, int64_t nc,
                const double *ev, const double *ec, const double *d, int64_t nd,
                int mode, double *w);
void zmr_w_aavv(const double complex *bva, int64_t ldva, int64_t naux, int64_t nv, int64_t na,
                const double *ev, const double *d, int64_t nd, double complex *w);
/* the same for spin-restricted orbitals (real, spin-free classes) */
void zmr_wsf_ccvv(const double *bvc, int64_t ldvc, int64_t naux, int64_t nv, int64_t nc,
                  const double *ev, const double *ec, const double *d, int64_t nd,
                  int mode, double *w);
void zmr_wsf_cavv(const double *bvc, int64_t ldvc, const double *bva, int64_t ldva,
                  int64_t naux, int64_t nv, int64_t na, int64_t nc,
                  const double *ev, const double *ec, const double *d, int64_t nd,
                  int mode, double *w);
void zmr_wsf_aavv(const double *bva, int64_t ldva, int64_t naux, int64_t nv, int64_t na,
                  const double *ev, const double *d, int64_t nd, double *w);
void zmr_w_cavv(const double complex *bvc, int64_t ldvc, const double complex *bva,
                int64_t ldva, int64_t naux,
                int64_t nv, int64_t na, int64_t nc, const double *ev, const double *ec,
                const double *d, int64_t nd, int mode, double complex *w);
void zmr_shifted_sums(const double *w, const double *delta, int64_t ne,
                      const double *lam, int64_t nk, const double *shifts, int64_t ns,
                      const unsigned char *mask, double *f);
void zmr_weighted_gram(const double complex *u, int64_t ns, int64_t nk, int64_t ne,
                       const double *lam, const double *delta, const double *shifts,
                       int64_t col, double complex *a);
void zmr_weighted_gram_r(const double *u, int64_t ns, int64_t nk, int64_t ne,
                         const double *lam, const double *delta, const double *shifts,
                         int64_t col, double *a);
void zmr_contract_t(const double complex *t, int64_t ns, int64_t nk, int64_t nx,
                    const double complex *v, int64_t ne,
                    const double *lam, const double *delta, const double *shifts,
                    int64_t col, double complex *a);
void zmr_contract_t_r(const double *t, int64_t ns, int64_t nk, int64_t nx,
                      const double *v, int64_t ne,
                      const double *lam, const double *delta, const double *shifts,
                      int64_t col, double *a);
void zmr_contract_w(const double complex *t, int64_t ns, int64_t nk, int64_t nx,
                    const double complex *v, int64_t ne,
                    const double *lam, const double *delta, const double *shifts,
                    int64_t col, double complex *a);
void zmr_contract_w_r(const double *t, int64_t ns, int64_t nk, int64_t nx,
                      const double *v, int64_t ne,
                      const double *lam, const double *delta, const double *shifts,
                      int64_t col, double *a);

/* Dense matrix of H = sum h_pq p^+ q + 1/2 sum (pq|rs) p^+ r^+ s q in the
 * space of all strings with nelec electrons in norb orbitals (ascending
 * integer order); hmat (dim, dim) is overwritten. */
void zmr_build_hmat(int norb, int nelec, int64_t dim, const double complex *h1,
                    const double complex *eri, double complex *hmat);

/* Spin-restricted active space: determinants with na alpha and nb beta
 * electrons in norb spatial orbitals, as strings (alpha << norb) | beta with
 * the address rank(alpha) * dim(beta) + rank(beta) (order and phases of
 * pyscf.fci).  The spin-orbital of orbital p is p (beta) or norb + p (alpha).
 *
 * zmr_build_hmat_ab: dense real matrix of
 *   sum_sigma h_pq p^+_sigma q_sigma
 *   + 1/2 sum_{sigma tau} (pq|rs) p^+_sigma r^+_tau s_tau q_sigma.
 * zmr_apply_strings_ab: as zmr_apply_strings for real vectors of the sector
 *   (na, nb), with ops given as spin-orbitals; (na2, nb2) is the sector the
 *   strings lead to. */
void zmr_build_hmat_ab(int norb, int na, int nb, const double *h1, const double *eri,
                       double *hmat);
/* out = H q for q (dim, nvec) without storing H (the same operator as
 * zmr_build_hmat_ab, for any real h1 and eri); out (dim, nvec) is
 * overwritten.  String-driven: single excitations of the alpha and beta
 * strings and one dgemm with the integrals per block of determinants. */
void zmr_sigma_ab(int norb, int na, int nb, const double *h1, const double *eri,
                  int64_t nvec, const double *q, double *out);
/* The same product determinant by determinant (Slater-Condon rules); needs
 * (pq|rs) = (rs|pq) and a symmetric matrix.  Reference of zmr_sigma_ab. */
void zmr_sigma_ab_det(int norb, int na, int nb, const double *h1, const double *eri,
                      int64_t nvec, const double *q, double *out);
void zmr_apply_strings_ab(int norb, int na, int nb, int na2, int nb2, int nop,
                          const int *dagger, const int64_t *ops,
                          int64_t nx, int64_t nx_total, int64_t x0, int64_t nvec,
                          const double *vecs, double *out);
/* out[dest[v * nx + x], :] += O_x vecs[v, :] (rows of the dimension of the
 * target sector; a negative destination skips the product).  The destinations
 * of different strings must be different rows. */
void zmr_apply_strings_ab_add(int norb, int na, int nb, int na2, int nb2, int nop,
                              const int *dagger, const int64_t *ops,
                              int64_t nx, int64_t nvec, const double *vecs,
                              const int64_t *dest, double *out);

/* Connected components of the graph of the non-zero elements of the n x n
 * matrix a (either triangle may be empty).  labels[i] is the component of
 * index i, numbered in order of first appearance; returns their number. */
int64_t zmr_components(const double complex *a, int64_t n, int64_t *labels);

/* touch[x, c] = 1 if row x of rows (m, d) has a non-zero element at an index
 * i with labels[i] == c, else 0; touch (m, ncomp). */
void zmr_block_support(const double complex *rows, int64_t m, int64_t d,
                       const int64_t *labels, int64_t ncomp, unsigned char *touch);

/* Folded two-component transition densities over the AO lower triangle:
 * out[(i, j), (m >= n)] = sum_spin conj(c1[m, i]) c2[n, j]
 *                          + [m != n] conj(c1[n, i]) c2[m, j],
 * c*a, c*b: alpha / beta components (nao, n*), out (n1 * n2, nao (nao + 1) / 2). */
void zmr_fold_density(int64_t nao, int64_t n1, int64_t n2,
                      const double complex *c1a, const double complex *c1b,
                      const double complex *c2a, const double complex *c2b,
                      double complex *out);


/* array bookkeeping of the MRCI sigma */
void zmr_transpose_add(int ndim, const int64_t *dshape, const int64_t *sstride,
                       const double complex *src, double complex *dst,
                       int assign, const double *alpha);
void zmr_pair_gather(int64_t nl, int64_t n, int64_t nt,
                     const double complex *src, double complex *dst);
void zmr_pair_scatter(int64_t nl, int64_t n, int64_t nt,
                      const double complex *src, double complex *dst);

#endif
