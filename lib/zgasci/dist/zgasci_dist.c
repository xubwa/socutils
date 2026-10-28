/*
 * zgasci_dist: product-space CI (SQD / QSCI spaces) with the CI vector
 * distributed by rows over MPI processes, driven from C (the algorithm of
 * socutils.hci.gasci_mpi; PRODUCT_SPACE.md, section 6c).
 *
 * The space: every alpha string of a list times every beta string of the
 * same list, nalpha = nbeta, spatial orbitals from an FCIDUMP (no orbital
 * symmetry used), states of even (parity +1) or odd (-1) spin, C[b][a] =
 * parity C[a][b].  Process r holds the rows (alpha strings, places in the
 * sorted list) rows[r] .. rows[r+1]-1 of every vector, each row all beta
 * strings.  sigma = Z + parity Z^T + Y:
 *   Z = (BB,BB) C, row-local (a product plan of the own alpha strings and
 *       all beta strings, integrals of the beta-beta pairs only);
 *   Z^T by pairwise exchanges of blocks;
 *   Y = (AB,AB) C from the own rows and the halo (rows of other processes
 *       reached by alpha single excitations), zg_sigma_ab_rows.
 * Eigensolver: LOBPCG-like iteration of one vector (X, the preconditioned
 * residual T, the previous direction P and their products: six vectors of
 * the own rows); vector operations with OpenMP, dot products summed by
 * MPI_Allreduce, the 3 x 3 Rayleigh-Ritz problem by LAPACK dsygv.
 *
 * Usage (one process per node, all threads):
 *   mpirun -np P -ppn 1 zgasci_dist FCIDUMP STRINGS [options]
 * STRINGS: the alpha (= beta) strings as little-endian int64 (bit p:
 * orbital p occupied, the pyscf cistring convention), any order, no repeats.
 * Environment: ZGASCI_DIST_CHUNK, the largest message in doubles (2^28).
 * Options: --tol E (energy, 1e-8)  --rtol R (residual norm, 1e-3)
 *          --maxit N (40)  --parity +-1 (1)  --bench K (time K products
 *          first, 3)  --shift S (preconditioner floor, 1e-3)
 *          --s2pen L (linear spin penalty H + L S^2, 0)
 *
 * Library (libzgasci_dist.so): zgd_solve, the same solve on a communicator
 * of the caller (a Fortran handle, mpi4py comm.py2f()) from integrals in
 * memory, returning the energies, the orbital occupations and per string
 * the largest |C| of its row and its marginal weight (for the carryover of
 * SQD); socutils.hci.gasci_mpi.make_sci_solver wraps it.
 */
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "../zgasci.h"

extern void dsygv_(const int *itype, const char *jobz, const char *uplo, const int *n,
                   double *a, const int *lda, double *b, const int *ldb, double *w,
                   double *work, const int *lwork, int *info);

static int rank_, size_;
static MPI_Comm comm_;
static int verbose_ = 1;
/* Largest MPI message in doubles; ZGASCI_DIST_CHUNK overrides (tests). */
static int64_t CHUNK = (int64_t)1 << 28;

static void die(const char *msg) {
  fprintf(stderr, "zgasci_dist (rank %d): %s\n", rank_, msg);
  MPI_Abort(comm_, 1);
}

static double now(void) { return MPI_Wtime(); }

static double peak_gb(void) {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss / 1048576.0;
}

/* ---- FCIDUMP ------------------------------------------------------------ */

typedef struct {
  int norb, nelec;
  double ecore;
  double *h1;  /* norb^2 */
  double *eri; /* norb^4, (pq|rs) */
} fcidump;

static void read_fcidump(const char *path, fcidump *f) {
  FILE *fp = fopen(path, "r");
  if (!fp) die("cannot open the FCIDUMP");
  char line[4096];
  f->norb = f->nelec = -1;
  /* header up to &END or / */
  int done = 0;
  while (!done && fgets(line, sizeof line, fp)) {
    char *s;
    if ((s = strstr(line, "NORB"))) sscanf(strchr(s, '=') + 1, "%d", &f->norb);
    if ((s = strstr(line, "NELEC"))) sscanf(strchr(s, '=') + 1, "%d", &f->nelec);
    if (strstr(line, "&END") || strstr(line, "/") || strstr(line, "$END")) done = 1;
  }
  if (f->norb <= 0 || f->nelec <= 0) die("FCIDUMP header without NORB / NELEC");
  const int n = f->norb;
  f->h1 = calloc((size_t)n * n, sizeof(double));
  f->eri = calloc((size_t)n * n * n * n, sizeof(double));
  f->ecore = 0;
  double val;
  int i, j, k, l;
  while (fgets(line, sizeof line, fp)) {
    for (char *c = line; *c; ++c) if (*c == 'D' || *c == 'd') *c = 'E';
    if (sscanf(line, "%lf %d %d %d %d", &val, &i, &j, &k, &l) != 5) continue;
    if (i > 0 && k > 0) {
      --i, --j, --k, --l;
      const int idx[8][4] = {{i, j, k, l}, {j, i, k, l}, {i, j, l, k}, {j, i, l, k},
                             {k, l, i, j}, {l, k, i, j}, {k, l, j, i}, {l, k, j, i}};
      for (int t = 0; t < 8; ++t) {
        const int *x = idx[t];
        f->eri[(((size_t)x[0] * n + x[1]) * n + x[2]) * n + x[3]] = val;
      }
    } else if (i > 0) {
      --i, --j;
      f->h1[(size_t)i * n + j] = f->h1[(size_t)j * n + i] = val;
    } else {
      f->ecore = val;
    }
  }
  fclose(fp);
}

/* v[P,R] = <pq||rs> + the one-electron part folded in over the spin orbitals
 * of a spatial plan (alpha P = P, beta P = norb + P), as
 * socutils.hci.gasci.antisymmetrized; beta_only: only the beta-beta pairs. */
static double *spin_orbital_v(const fcidump *f, int beta_only) {
  const int n = f->norb, ns = 2 * n, nel = f->nelec;
  const int64_t npair = (int64_t)ns * (ns - 1) / 2;
  double *v = calloc((size_t)(npair * npair), sizeof(double));
  int *pp = malloc(npair * sizeof(int)), *qq = malloc(npair * sizeof(int));
  for (int q = 1, P = 0; q < ns; ++q)
    for (int p = 0; p < q; ++p, ++P) pp[P] = p, qq[P] = q;
#define SP(x) ((x) / n)
#define SO(x) ((x) % n)
#define ERI(a, b, c, d) f->eri[(((size_t)SO(a) * n + SO(b)) * n + SO(c)) * n + SO(d)]
  /* <pq|rs> = (pr|qs) if spin(p) = spin(r) and spin(q) = spin(s) */
#define PHYS(p, q, r, s) ((SP(p) == SP(r) && SP(q) == SP(s)) ? ERI(p, r, q, s) : 0.0)
#define H(p, r) (SP(p) == SP(r) ? f->h1[(size_t)SO(p) * n + SO(r)] : 0.0)
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t P = 0; P < npair; ++P) {
    const int p = pp[P], q = qq[P];
    if (beta_only && p < n) continue;
    for (int64_t R = 0; R < npair; ++R) {
      const int r = pp[R], s = qq[R];
      if (beta_only && r < n) continue;
      double w = PHYS(p, q, r, s) - PHYS(p, q, s, r);
      w += ((q == s ? H(p, r) : 0) - (q == r ? H(p, s) : 0) - (p == s ? H(q, r) : 0) +
            (p == r ? H(q, s) : 0)) / (nel - 1);
      v[P * npair + R] = w;
    }
  }
#undef SP
#undef SO
#undef ERI
#undef PHYS
#undef H
  free(pp);
  free(qq);
  return v;
}

/* v += lam S^2 (nalpha = nbeta: S_z = 0) in the format of spin_orbital_v, as
 * socutils.hci.gasci.s2_antisymmetrized: S^2 = N_beta - sum_pq a+_{p alpha}
 * a+_{q beta} a_{p beta} a_{q alpha}; N_beta = N / 2 folded in as the
 * diagonal 2 (1/2) / (N - 1) of every pair.  beta_only: the beta-beta pairs. */
static void add_s2(double *v, int norb, int nelec, double lam, int beta_only) {
  const int ns = 2 * norb;
  const int64_t npair = (int64_t)ns * (ns - 1) / 2;
#define PAIR(a, b) ((int64_t)(b) * ((b) - 1) / 2 + (a))
  for (int q = 1; q < ns; ++q)
    for (int p = 0; p < q; ++p)
      if (!beta_only || p >= norb) v[PAIR(p, q) * (npair + 1)] += lam / (nelec - 1);
  if (!beta_only)
    for (int p = 0; p < norb; ++p)
      for (int q = 0; q < norb; ++q) v[PAIR(p, norb + q) * npair + PAIR(q, norb + p)] -= lam;
#undef PAIR
}

/* ---- MPI helpers -------------------------------------------------------- */

/* Send ns doubles to dest and receive nr from src, both in chunks of CHUNK
 * (counts are C ints): every chunk is its own message, tagged by its index,
 * so that the two sides need not agree on a number of steps (the lengths of
 * a pair differ, e.g. in the halo exchange). */
static void sendrecv(const double *s, int64_t ns, int dest, double *r, int64_t nr, int src) {
  const int64_t ks = (ns + CHUNK - 1) / CHUNK, kr = (nr + CHUNK - 1) / CHUNK;
  MPI_Request *req = malloc((size_t)(ks + kr + 1) * sizeof(MPI_Request));
  int nreq = 0;
  for (int64_t k = 0; k < kr; ++k) {
    const int64_t i = k * CHUNK, b = nr - i < CHUNK ? nr - i : CHUNK;
    MPI_Irecv(r + i, (int)b, MPI_DOUBLE, src, (int)k, comm_, &req[nreq++]);
  }
  for (int64_t k = 0; k < ks; ++k) {
    const int64_t i = k * CHUNK, a = ns - i < CHUNK ? ns - i : CHUNK;
    MPI_Isend(s + i, (int)a, MPI_DOUBLE, dest, (int)k, comm_, &req[nreq++]);
  }
  MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
  free(req);
}

static double gdot(const double *x, const double *y, int64_t len) {
  double s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
  for (int64_t i = 0; i < len; ++i) s += x[i] * y[i];
  MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, comm_);
  return s;
}

static void axpy(double a, const double *x, double *y, int64_t len) {
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < len; ++i) y[i] += a * x[i];
}

static void scal(double a, double *x, int64_t len) {
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < len; ++i) x[i] *= a;
}

static void copy(const double *x, double *y, int64_t len) {
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < len; ++i) y[i] = x[i];
}

/* ---- the distributed product space -------------------------------------- */

typedef struct {
  int norb, nalpha, parity;
  int64_t n, lo, hi, nloc, len; /* strings, own rows, own elements */
  int64_t *rows;                /* size + 1 */
  int64_t *strs;                /* sorted */
  int64_t *perm;                /* strs[k] = the input string perm[k] */
  zg_plan *plan, *lplan;
  double *v, *v_bb;             /* H + s2pen S^2 */
  double *v_h, *v_bb_h;         /* H, if s2pen != 0 */
  double s2pen;
  /* halo */
  int64_t nhalo, *smap;
  int64_t *hsend_cnt, *hsend_off, *hsend_rows; /* rows I send to q (global places) */
  int64_t *hrecv_cnt, *hrecv_off;              /* rows I receive from q, in buf */
  double *hbuf;                                /* own rows then halo rows */
  double *zt;                                  /* transpose buffer, len */
  double *sbuf, *rbuf;                         /* blocks of the transpose / halo */
  int64_t sbuf_len, rbuf_len;
  double t_z, t_tr, t_halo, t_ab, t_sig;
  int nsig;
} dist;

static int cmp64(const void *a, const void *b) {
  const int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
  return x < y ? -1 : x > y;
}

static int64_t find(const int64_t *s, int64_t n, int64_t x) {
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (s[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return (lo < n && s[lo] == x) ? lo : -1;
}

static void halo_setup(dist *d) {
  const int64_t n = d->n;
  unsigned char *reach = calloc((size_t)n, 1);
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t a = d->lo; a < d->hi; ++a) {
    const int64_t s = d->strs[a];
    for (int i = 0; i < d->norb; ++i) {
      if (!((s >> i) & 1)) continue;
      for (int j = 0; j < d->norb; ++j) {
        if ((s >> j) & 1) continue;
        const int64_t t = s ^ (((int64_t)1 << i) | ((int64_t)1 << j));
        const int64_t k = find(d->strs, n, t);
        if (k >= 0) reach[k] = 1;
      }
    }
  }
  for (int64_t a = d->lo; a < d->hi; ++a) reach[a] = 0;
  d->hrecv_cnt = calloc(size_, sizeof(int64_t));
  d->hrecv_off = calloc(size_ + 1, sizeof(int64_t));
  d->smap = malloc((size_t)n * sizeof(int64_t));
  for (int64_t a = 0; a < n; ++a) d->smap[a] = -1;
  for (int64_t a = d->lo; a < d->hi; ++a) d->smap[a] = a - d->lo;
  int64_t nh = 0;
  int owner = 0;
  for (int64_t a = 0; a < n; ++a) {
    if (!reach[a]) continue;
    while (a >= d->rows[owner + 1]) ++owner;
    d->hrecv_cnt[owner]++;
    d->smap[a] = d->nloc + nh++;
  }
  d->nhalo = nh;
  for (int q = 0; q < size_; ++q) d->hrecv_off[q + 1] = d->hrecv_off[q] + d->hrecv_cnt[q];
  /* who wants which of my rows */
  d->hsend_cnt = calloc(size_, sizeof(int64_t));
  MPI_Alltoall(d->hrecv_cnt, 1, MPI_INT64_T, d->hsend_cnt, 1, MPI_INT64_T, comm_);
  d->hsend_off = calloc(size_ + 1, sizeof(int64_t));
  for (int q = 0; q < size_; ++q) d->hsend_off[q + 1] = d->hsend_off[q] + d->hsend_cnt[q];
  d->hsend_rows = malloc((size_t)(d->hsend_off[size_] + 1) * sizeof(int64_t));
  int64_t *want = malloc((size_t)(nh + 1) * sizeof(int64_t));
  for (int64_t a = 0, k = 0; a < n; ++a) if (reach[a]) want[k++] = a;
  int *sc = malloc(size_ * sizeof(int)), *so = malloc(size_ * sizeof(int));
  int *rc = malloc(size_ * sizeof(int)), *ro = malloc(size_ * sizeof(int));
  for (int q = 0; q < size_; ++q) {
    sc[q] = (int)d->hrecv_cnt[q], so[q] = (int)d->hrecv_off[q];
    rc[q] = (int)d->hsend_cnt[q], ro[q] = (int)d->hsend_off[q];
  }
  MPI_Alltoallv(want, sc, so, MPI_INT64_T, d->hsend_rows, rc, ro, MPI_INT64_T, comm_);
  free(sc), free(so), free(rc), free(ro), free(want), free(reach);
}

static int cmp_pair(const void *a, const void *b) { return cmp64(a, b); }

static void dist_init(dist *d, const fcidump *f, const int64_t *strs_in, int64_t n, int parity,
                      double s2pen) {
  memset(d, 0, sizeof *d);
  d->norb = f->norb;
  d->nalpha = f->nelec / 2;
  d->parity = parity;
  d->s2pen = s2pen;
  /* sort a copy, keeping where each string came from */
  int64_t *pairs = malloc((size_t)(2 * n + 1) * sizeof(int64_t));
  for (int64_t k = 0; k < n; ++k) pairs[2 * k] = strs_in[k], pairs[2 * k + 1] = k;
  qsort(pairs, n, 2 * sizeof(int64_t), cmp_pair);
  int64_t *strs = malloc((size_t)(n + 1) * sizeof(int64_t));
  d->perm = malloc((size_t)(n + 1) * sizeof(int64_t));
  for (int64_t k = 0; k < n; ++k) strs[k] = pairs[2 * k], d->perm[k] = pairs[2 * k + 1];
  free(pairs);
  for (int64_t k = 1; k < n; ++k)
    if (strs[k] == strs[k - 1]) die("repeated strings");
  d->strs = strs;
  d->n = n;
  d->rows = malloc((size_ + 1) * sizeof(int64_t));
  for (int r = 0; r <= size_; ++r) d->rows[r] = (n * r) / size_;
  d->lo = d->rows[rank_], d->hi = d->rows[rank_ + 1], d->nloc = d->hi - d->lo;
  d->len = d->nloc * n;
  const int ns = 2 * f->norb;
  d->plan = zg_plan_product(ns, f->nelec, f->norb, n, (const uint64_t *)strs, n,
                            (const uint64_t *)strs, 0, NULL, NULL, NULL);
  if (!d->plan) die(zg_error());
  if (zg_dim(d->plan) != n * n) die("strings left out of the product plan (electron counts?)");
  if (size_ == 1) {
    d->lplan = d->plan;
  } else if (d->nloc > 0) {
    d->lplan = zg_plan_product(ns, f->nelec, f->norb, d->nloc, (const uint64_t *)(strs + d->lo),
                               n, (const uint64_t *)strs, 0, NULL, NULL, NULL);
    if (!d->lplan) die(zg_error());
  }
  d->v = spin_orbital_v(f, 0);
  d->v_bb = spin_orbital_v(f, 1);
  if (s2pen != 0) {
    const int64_t np2 = (int64_t)ns * (ns - 1) / 2 * ((int64_t)ns * (ns - 1) / 2);
    d->v_h = malloc((size_t)np2 * sizeof(double));
    d->v_bb_h = malloc((size_t)np2 * sizeof(double));
    memcpy(d->v_h, d->v, (size_t)np2 * sizeof(double));
    memcpy(d->v_bb_h, d->v_bb, (size_t)np2 * sizeof(double));
    add_s2(d->v, f->norb, f->nelec, s2pen, 0);
    add_s2(d->v_bb, f->norb, f->nelec, s2pen, 1);
  }
  halo_setup(d);
  d->zt = malloc((size_t)(d->len + 1) * sizeof(double));
  if (d->nhalo) d->hbuf = malloc((size_t)((d->nloc + d->nhalo) * n + 1) * sizeof(double));
  /* the largest block of the transpose and of the halo */
  int64_t smax = 1, rmax = 1;
  for (int q = 0; q < size_; ++q) {
    const int64_t w = d->rows[q + 1] - d->rows[q];
    if (d->nloc * w > smax) smax = d->nloc * w;
    if (w * d->nloc > rmax) rmax = w * d->nloc;
    if (d->hsend_cnt[q] * n > smax) smax = d->hsend_cnt[q] * n;
  }
  d->sbuf_len = smax, d->rbuf_len = rmax;
  d->sbuf = malloc((size_t)smax * sizeof(double));
  d->rbuf = malloc((size_t)rmax * sizeof(double));
}

static void dist_free(dist *d) {
  if (d->lplan && d->lplan != d->plan) zg_plan_free(d->lplan);
  if (d->plan) zg_plan_free(d->plan);
  free(d->strs), free(d->perm), free(d->rows);
  free(d->v), free(d->v_bb), free(d->v_h), free(d->v_bb_h);
  free(d->smap), free(d->hsend_cnt), free(d->hsend_off), free(d->hsend_rows);
  free(d->hrecv_cnt), free(d->hrecv_off), free(d->hbuf), free(d->zt), free(d->sbuf), free(d->rbuf);
  memset(d, 0, sizeof *d);
}

/* out = the own rows of X^T (X distributed by rows). */
static void transpose(dist *d, const double *x, double *out) {
  const int64_t n = d->n, lo = d->lo, nl = d->nloc;
  zg_transpose_d(nl, nl, x + lo, n, out + lo, n);
  for (int k = 1; k < size_; ++k) {
    const int dest = (rank_ + k) % size_, src = (rank_ - k + size_) % size_;
    const int64_t c0 = d->rows[dest], wd = d->rows[dest + 1] - c0;
    const int64_t s0 = d->rows[src], ws = d->rows[src + 1] - s0;
    zg_copy_block_d(nl, wd, x + c0, n, d->sbuf, wd);
    sendrecv(d->sbuf, nl * wd, dest, d->rbuf, ws * nl, src);
    zg_transpose_d(ws, nl, d->rbuf, nl, out + s0, n);
  }
}

/* x = (x + parity x^T) / 2 */
static void symmetrize(dist *d, double *x) {
  transpose(d, x, d->zt);
  const double p = d->parity;
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < d->len; ++i) x[i] = 0.5 * (x[i] + p * d->zt[i]);
}

/* hbuf = own rows of c, then the halo rows */
static void halo_exchange(dist *d, const double *c) {
  const int64_t n = d->n;
  copy(c, d->hbuf, d->len);
  for (int k = 1; k < size_; ++k) {
    const int dest = (rank_ + k) % size_, src = (rank_ - k + size_) % size_;
    const int64_t ns = d->hsend_cnt[dest];
    const int64_t *rows = d->hsend_rows + d->hsend_off[dest];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < ns; ++i)
      memcpy(d->sbuf + i * n, c + (rows[i] - d->lo) * n, (size_t)n * sizeof(double));
    double *dst = d->hbuf + (d->nloc + d->hrecv_off[src]) * n;
    sendrecv(d->sbuf, ns * n, dest, dst, d->hrecv_cnt[src] * n, src);
  }
}

/* out = H c for c of the symmetry. */
static void sigma(dist *d, const double *c, double *out) {
  const double t0 = now();
  if (d->nloc) zg_sigma_blas_d(d->lplan, d->v_bb, c, out);
  const double t1 = now();
  transpose(d, out, d->zt);
  axpy((double)d->parity, d->zt, out, d->len);
  const double t2 = now();
  const double *src = c;
  if (d->nhalo) {
    halo_exchange(d, c);
    src = d->hbuf;
  }
  const double t3 = now();
  if (d->nloc && !zg_sigma_ab_rows_d(d->plan, d->v, src, d->smap, out, d->lo, d->hi))
    die("zg_sigma_ab_rows does not apply to this plan");
  const double t4 = now();
  d->t_z += t1 - t0, d->t_tr += t2 - t1, d->t_halo += t3 - t2, d->t_ab += t4 - t3;
  d->t_sig += t4 - t0;
  d->nsig++;
}

static void report_parts(dist *d, const char *what, int per) {
  double t[5] = {d->t_z, d->t_tr, d->t_halo, d->t_ab, d->t_sig};
  MPI_Allreduce(MPI_IN_PLACE, t, 5, MPI_DOUBLE, MPI_MAX, comm_);
  const double k = per ? (d->nsig ? d->nsig : 1) : 1;
  if (rank_ == 0)
    printf("%s: sigma %.2f s (max over procs: Z %.2f  transpose %.2f  halo %.2f  (AB,AB) %.2f)%s\n",
           what, t[4] / k, t[0] / k, t[1] / k, t[2] / k, t[3] / k, per ? " per product" : "");
  fflush(stdout);
}

/* ---- LOBPCG --------------------------------------------------------------- */

/* The lowest state; *xout: its own rows (normalized), for the caller to free. */
static double lobpcg(dist *d, int maxit, double tol, double rtol, double shift, int *cycles,
                     int *converged, double **xout) {
  const int64_t L = d->len;
  double *hd = malloc((size_t)(L + 1) * sizeof(double));
  if (d->nloc) zg_hdiag_d(d->lplan, d->v, hd);
  double *x = calloc((size_t)L + 1, sizeof(double)), *hx = malloc((size_t)(L + 1) * sizeof(double));
  double *t = malloc((size_t)(L + 1) * sizeof(double)), *ht = malloc((size_t)(L + 1) * sizeof(double));
  double *p = NULL, *hp = NULL;
  /* start: the lowest diagonal element, symmetrized */
  struct { double v; int r; } loc = {INFINITY, rank_}, glob;
  int64_t imin = -1;
  /* (odd parity: off the diagonal a = b, where the symmetrized vector vanishes) */
  for (int64_t i = 0; i < L; ++i)
    if (hd[i] < loc.v && (d->parity > 0 || d->lo + i / d->n != i % d->n)) loc.v = hd[i], imin = i;
  MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm_);
  if (rank_ == glob.r) x[imin] = 1;
  symmetrize(d, x);
  scal(1 / sqrt(gdot(x, x, L)), x, L);
  sigma(d, x, hx);
  double e = 0, e_old = NAN, rnorm = 0;
  const double t0 = now();
  *converged = 0;
  int it;
  for (it = 0; it < maxit; ++it) {
    e = gdot(x, hx, L);
    copy(hx, t, L);
    axpy(-e, x, t, L);
    rnorm = sqrt(gdot(t, t, L));
    if (rank_ == 0 && verbose_) {
      printf("  lobpcg %3d  E %.12f  dE %.2e  |r| %.2e  t %.1f s  peak %.1f GB\n", it, e,
             isnan(e_old) ? 0.0 : e - e_old, rnorm, now() - t0, peak_gb());
      fflush(stdout);
    }
    if ((!isnan(e_old) && fabs(e - e_old) < tol && rnorm < rtol) || rnorm < 1e-12) {
      *converged = 1;
      break;
    }
    e_old = e;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < L; ++i) {
      /* max(|hd - e|, shift): positive definite, as LOBPCG needs (hd - e
       * changes sign e.g. with an S^2 penalty and odd parity) */
      const double dd = fmax(fabs(hd[i] - e), shift);
      t[i] /= -dd;
    }
    symmetrize(d, t);
    axpy(-gdot(x, t, L), x, t, L);
    if (p) axpy(-gdot(p, t, L), p, t, L);
    const double nt = sqrt(gdot(t, t, L));
    if (nt < 1e-14) {
      *converged = 1;
      break;
    }
    scal(1 / nt, t, L);
    sigma(d, t, ht);
    const int m = p ? 3 : 2;
    double *b[3] = {x, t, p}, *hb[3] = {hx, ht, hp};
    double hs[9], ss[9], w[3], work[64];
    for (int i = 0; i < m; ++i)
      for (int j = i; j < m; ++j) {
        hs[i * m + j] = hs[j * m + i] = gdot(b[i], hb[j], L);
        ss[i * m + j] = ss[j * m + i] = gdot(b[i], b[j], L);
      }
    const int one = 1, lw = 64;
    int info;
    dsygv_(&one, "V", "U", &m, hs, &m, ss, &m, w, work, &lw, &info);
    if (info) die("dsygv failed");
    const double a = hs[0], bb = hs[1], cc = m == 3 ? hs[2] : 0.0;
    /* X <- a X + b T + c P ;  P <- b T + c P (and the products) */
    scal(a, x, L);
    axpy(bb, t, x, L);
    scal(a, hx, L);
    axpy(bb, ht, hx, L);
    if (!p) {
      p = t, hp = ht;
      scal(bb, p, L);
      scal(bb, hp, L);
      t = malloc((size_t)(L + 1) * sizeof(double));
      ht = malloc((size_t)(L + 1) * sizeof(double));
    } else {
      axpy(cc, p, x, L);
      axpy(cc, hp, hx, L);
      scal(cc, p, L);
      axpy(bb, t, p, L);
      scal(cc, hp, L);
      axpy(bb, ht, hp, L);
    }
    const double nx = sqrt(gdot(x, x, L));
    scal(1 / nx, x, L);
    scal(1 / nx, hx, L);
    const double np_ = sqrt(gdot(p, p, L));
    if (np_ > 0) {
      scal(1 / np_, p, L);
      scal(1 / np_, hp, L);
    }
  }
  *cycles = it + 1;
  if (rank_ == 0 && verbose_) {
    printf("lobpcg: %d iterations, converged %d, |r| %.2e, %.1f s\n", *cycles, *converged, rnorm,
           now() - t0);
    fflush(stdout);
  }
  free(hd), free(hx), free(t), free(ht), free(p), free(hp);
  *xout = x;
  return e;
}

/* After the solve of x (own rows, normalized; e_pen = <x|H + s2pen S^2|x>):
 * out = {e_pen, <H>, <S^2> (NaN without a penalty)}, one more product if
 * penalized; per string in the input order the largest |C| of its row (=
 * of its column) and its weight sum_b C[a][b]^2, and occ[p] = sum_a
 * weight[a] n_p(a), the alpha (= beta) occupations. */
static void finish(dist *d, const double *x, double e_pen, double *out, double *occ,
                   double *maxabs, double *wa) {
  const int64_t n = d->n, L = d->len;
  out[0] = out[1] = e_pen, out[2] = NAN;
  if (d->s2pen != 0) {
    double *hx = malloc((size_t)(L + 1) * sizeof(double));
    double *v = d->v, *v_bb = d->v_bb;
    d->v = d->v_h, d->v_bb = d->v_bb_h;
    sigma(d, x, hx);
    d->v = v, d->v_bb = v_bb;
    out[1] = gdot(x, hx, L);
    out[2] = (e_pen - out[1]) / d->s2pen;
    free(hx);
  }
  double *gw = calloc((size_t)n + 1, sizeof(double)), *gm = calloc((size_t)n + 1, sizeof(double));
#pragma omp parallel for schedule(static)
  for (int64_t a = d->lo; a < d->hi; ++a) {
    const double *row = x + (a - d->lo) * n;
    double w = 0, m = 0;
    for (int64_t b = 0; b < n; ++b) {
      w += row[b] * row[b];
      if (fabs(row[b]) > m) m = fabs(row[b]);
    }
    gw[a] = w, gm[a] = m;
  }
  MPI_Allreduce(MPI_IN_PLACE, gw, (int)n, MPI_DOUBLE, MPI_SUM, comm_);
  MPI_Allreduce(MPI_IN_PLACE, gm, (int)n, MPI_DOUBLE, MPI_MAX, comm_);
  if (occ) {
    for (int p = 0; p < d->norb; ++p) occ[p] = 0;
    for (int64_t k = 0; k < n; ++k)
      for (int p = 0; p < d->norb; ++p)
        if ((d->strs[k] >> p) & 1) occ[p] += gw[k];
  }
  for (int64_t k = 0; k < n; ++k) {
    if (wa) wa[d->perm[k]] = gw[k];
    if (maxabs) maxabs[d->perm[k]] = gm[k];
  }
  free(gw), free(gm);
}

/* The library entry: the lowest state of H (+ s2pen S^2) in the product
 * space strs x strs (nstr int64 strings, any order, no repeats; nalpha =
 * nbeta = nelec / 2) of the states with C[b][a] = parity C[a][b], on the
 * communicator comm_f (Fortran handle; every process calls with the same
 * arguments).  h1: norb^2, eri: norb^4 (pq|rs), no core energy.
 * out[6] = {E of H + s2pen S^2, <H>, <S^2> (NaN if s2pen = 0), iterations,
 * converged, seconds}; occ[norb], maxabs[nstr], wa[nstr] as finish (any may
 * be NULL), the same on every process.  Errors abort the communicator. */
int zgd_solve(int comm_f, int norb, int nelec, const double *h1, const double *eri, int64_t nstr,
              const int64_t *strs, int parity, double s2pen, double tol, double rtol, int maxit,
              int verbose, double *out, double *occ, double *maxabs, double *wa) {
  comm_ = MPI_Comm_f2c((MPI_Fint)comm_f);
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
  verbose_ = verbose;
  if (getenv("ZGASCI_DIST_CHUNK") && atol(getenv("ZGASCI_DIST_CHUNK")) > 0)
    CHUNK = atol(getenv("ZGASCI_DIST_CHUNK"));
  if (nelec % 2) die("nalpha = nbeta is required");
  if (parity != 1 && parity != -1) die("parity must be +1 or -1");
  if (nstr > INT32_MAX) die("more strings than an MPI count");
  const double t0 = now();
  fcidump f = {norb, nelec, 0.0, (double *)h1, (double *)eri};
  dist d;
  dist_init(&d, &f, strs, nstr, parity, s2pen);
  if (rank_ == 0 && verbose_) {
    printf("zgd_solve: %d processes x %d threads; norb %d nelec %d; strings %ld, dim %.4e; "
           "setup %.1f s\n", size_, omp_get_max_threads(), norb, nelec, (long)nstr,
           (double)nstr * nstr, now() - t0);
    fflush(stdout);
  }
  int cycles, conv;
  double *x;
  const double e = lobpcg(&d, maxit, tol, rtol, 1e-3, &cycles, &conv, &x);
  finish(&d, x, e, out, occ, maxabs, wa);
  out[3] = cycles, out[4] = conv, out[5] = now() - t0;
  free(x);
  dist_free(&d);
  return 0;
}

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  comm_ = MPI_COMM_WORLD;
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
  if (getenv("ZGASCI_DIST_CHUNK") && atol(getenv("ZGASCI_DIST_CHUNK")) > 0)
    CHUNK = atol(getenv("ZGASCI_DIST_CHUNK"));
  if (argc < 3) {
    if (!rank_) fprintf(stderr, "usage: zgasci_dist FCIDUMP STRINGS [options]\n");
    MPI_Finalize();
    return 1;
  }
  double tol = 1e-8, rtol = 1e-3, shift = 1e-3, s2pen = 0;
  int maxit = 40, parity = 1, bench = 3;
  for (int i = 3; i + 1 < argc; i += 2) {
    if (!strcmp(argv[i], "--tol")) tol = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--rtol")) rtol = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--maxit")) maxit = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--parity")) parity = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--bench")) bench = atoi(argv[i + 1]);
    else if (!strcmp(argv[i], "--shift")) shift = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "--s2pen")) s2pen = atof(argv[i + 1]);
    else die("unknown option");
  }
  const double t0 = now();
  fcidump f;
  read_fcidump(argv[1], &f);
  if (f.nelec % 2) die("nalpha = nbeta is required");
  FILE *fp = fopen(argv[2], "rb");
  if (!fp) die("cannot open the strings");
  fseek(fp, 0, SEEK_END);
  const int64_t n = ftell(fp) / 8;
  rewind(fp);
  int64_t *strs = malloc((size_t)n * sizeof(int64_t));
  if ((int64_t)fread(strs, 8, n, fp) != n) die("short read of the strings");
  fclose(fp);
  dist d;
  dist_init(&d, &f, strs, n, parity, s2pen);
  free(strs);
  const double t_setup = now() - t0;
  int64_t hmin = d.nhalo, hmax = d.nhalo;
  MPI_Allreduce(MPI_IN_PLACE, &hmin, 1, MPI_INT64_T, MPI_MIN, comm_);
  MPI_Allreduce(MPI_IN_PLACE, &hmax, 1, MPI_INT64_T, MPI_MAX, comm_);
  if (!rank_) {
    printf("zgasci_dist: %d processes x %d threads; norb %d nelec %d; strings %ld, dim %.4e; "
           "own rows %ld..%ld, halo rows %ld..%ld; setup %.1f s\n",
           size_, omp_get_max_threads(), f.norb, f.nelec, (long)n, (double)n * n,
           (long)(n / size_), (long)((n + size_ - 1) / size_), (long)hmin, (long)hmax, t_setup);
    fflush(stdout);
  }
  if (bench > 0) {
    double *x = malloc((size_t)(d.len + 1) * sizeof(double)), *y = malloc((size_t)(d.len + 1) * sizeof(double));
    srand(1 + rank_);
    for (int64_t i = 0; i < d.len; ++i) x[i] = rand() / (double)RAND_MAX - 0.5;
    symmetrize(&d, x);
    sigma(&d, x, y); /* warm-up */
    d.t_z = d.t_tr = d.t_halo = d.t_ab = d.t_sig = 0, d.nsig = 0;
    MPI_Barrier(comm_);
    const double tb = now();
    for (int k = 0; k < bench; ++k) sigma(&d, x, y);
    MPI_Barrier(comm_);
    if (!rank_) printf("bench: %d products, %.2f s each (wall)\n", bench, (now() - tb) / bench);
    report_parts(&d, "bench", 1);
    free(x), free(y);
    d.t_z = d.t_tr = d.t_halo = d.t_ab = d.t_sig = 0, d.nsig = 0;
  }
  int cycles, conv;
  const double tl = now();
  double *x, res[3];
  const double e = lobpcg(&d, maxit, tol, rtol, shift, &cycles, &conv, &x);
  const double t_solve = now() - tl;
  finish(&d, x, e, res, NULL, NULL, NULL);
  double pk = peak_gb();
  MPI_Allreduce(MPI_IN_PLACE, &pk, 1, MPI_DOUBLE, MPI_MAX, comm_);
  report_parts(&d, "solve", 0);
  if (!rank_) {
    printf("E %.12f  (E + ECORE; electronic %.12f)  iterations %d  converged %d  products %d  "
           "solve %.1f s (of which sigma %.1f s)  peak memory %.1f GB per process\n",
           e + f.ecore, e, cycles, conv, d.nsig, t_solve, d.t_sig, pk);
    if (s2pen != 0) printf("<H> + ECORE %.12f  <S^2> %.8f\n", res[1] + f.ecore, res[2]);
    fflush(stdout);
  }
  free(x);
  dist_free(&d);
  MPI_Finalize();
  return 0;
}
