/*
 * zgasci_dist: product-space CI (SQD / QSCI spaces) with the CI vector
 * distributed by rows over MPI processes, driven from C (the algorithm of
 * socutils.hci.gasci_mpi; PRODUCT_SPACE.md, section 6c).
 *
 * The space: every alpha string of a list times every beta string of a
 * list, spatial orbitals (no orbital symmetry used).  Process r holds the
 * rows (alpha strings, places in the sorted list) rows[r] .. rows[r+1]-1 of
 * every vector, each row all beta strings.
 * Same lists, nalpha = nbeta (parity +-1): the states of even (+1) or odd
 * (-1) spin, C[b][a] = parity C[a][b], and sigma = Z + parity Z^T + Y:
 *   Z = (BB,BB) C, row-local (a product plan of the own alpha strings and
 *       all beta strings, integrals of the beta-beta pairs only);
 *   Z^T by pairwise exchanges of blocks;
 *   Y = (AB,AB) C from the own rows and the halo (rows of other processes
 *       reached by alpha single excitations), zg_sigma_ab_rows.
 * Any lists and electron counts (parity 0): Z as above; (AA,AA) C by the
 * same kernel on C^T held by beta rows (a plan of the own beta strings and
 * all alpha strings), transposed there and back; Y as above.
 * Eigensolvers: LOBPCG-like iteration of one vector (X, the preconditioned
 * residual T, the previous direction P and their products: six vectors of
 * the own rows; the 3 x 3 Rayleigh-Ritz problem by LAPACK dsygv), or
 * Davidson (a basis of up to max_space vectors and their products,
 * collapsed to two); vector operations with OpenMP, dot products summed by
 * MPI_Allreduce.
 *
 * Usage (one process per node, all threads):
 *   mpirun -np P -ppn 1 zgasci_dist FCIDUMP STRINGS [options]
 * STRINGS: the alpha (= beta) strings as little-endian int64 (bit p:
 * orbital p occupied, the pyscf cistring convention), any order, no repeats.
 * Environment: ZGASCI_DIST_CHUNK, the largest message in doubles (2^28).
 * Options: --tol E (energy, 1e-8)  --rtol R (residual norm, 1e-3)
 *          --maxit N (40)  --parity +-1 or 0 (1; 0: no symmetry, the
 *          general path on the same lists)  --bench K (time K products
 *          first, 3)  --shift S (preconditioner floor, 1e-3)
 *          --s2pen L (linear spin penalty H + L (S^2 - ss), 0)
 *          --solver lobpcg | davidson (lobpcg)  --max-space M (Davidson, 12)
 *
 * Library (libzgasci_dist.so): zgd_solve_ab (any lists) and zgd_solve
 * (alpha = beta), the same solve on a communicator of the caller (a
 * Fortran handle, mpi4py comm.py2f()) from integrals in memory, returning
 * the energies, the orbital occupations and per string the largest |C| of
 * its row (column) and its marginal weight (for the carryover of SQD), with
 * a warm start between calls (zgd_promote, zgd_reset);
 * socutils.hci.gasci_mpi.make_sci_solver wraps it.
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
  double *h1;  /* norb^2 (alpha if uhf) */
  double *eri; /* norb^4, (pq|rs) (alpha-alpha if uhf) */
  int uhf;     /* different alpha and beta orbitals: */
  double *h1b, *eri_ab, *eri_bb; /* (p_a q_a | r_b s_b), (beta beta | beta beta) */
} fcidump;

static void read_fcidump(const char *path, fcidump *f) {
  FILE *fp = fopen(path, "r");
  if (!fp) die("cannot open the FCIDUMP");
  char line[4096];
  f->norb = f->nelec = -1;
  f->uhf = 0, f->h1b = f->eri_ab = f->eri_bb = NULL;
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
 * socutils.hci.gasci.antisymmetrized; with f->uhf the integrals of each
 * spin block. */
static double *spin_orbital_v(const fcidump *f) {
  const int n = f->norb, ns = 2 * n, nel = f->nelec;
  const int64_t npair = (int64_t)ns * (ns - 1) / 2;
  double *v = calloc((size_t)(npair * npair), sizeof(double));
  int *pp = malloc(npair * sizeof(int)), *qq = malloc(npair * sizeof(int));
  for (int q = 1, P = 0; q < ns; ++q)
    for (int p = 0; p < q; ++p, ++P) pp[P] = p, qq[P] = q;
#define SP(x) ((x) / n)
#define SO(x) ((x) % n)
#define IDX(a, b, c, d) ((((size_t)SO(a) * n + SO(b)) * n + SO(c)) * n + SO(d))
  /* (ab|cd), a, b of one spin and c, d of one spin */
#define ERI(a, b, c, d)                                                          \
  (!f->uhf || (SP(a) == 0 && SP(c) == 0) ? f->eri[IDX(a, b, c, d)]               \
   : SP(a) == 1 && SP(c) == 1            ? f->eri_bb[IDX(a, b, c, d)]            \
   : SP(a) == 0                          ? f->eri_ab[IDX(a, b, c, d)]            \
                                         : f->eri_ab[IDX(c, d, a, b)])
  /* <pq|rs> = (pr|qs) if spin(p) = spin(r) and spin(q) = spin(s) */
#define PHYS(p, q, r, s) ((SP(p) == SP(r) && SP(q) == SP(s)) ? ERI(p, r, q, s) : 0.0)
#define H(p, r)                                                                  \
  (SP(p) != SP(r) ? 0.0 : (f->uhf && SP(p) ? f->h1b : f->h1)[(size_t)SO(p) * n + SO(r)])
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t P = 0; P < npair; ++P) {
    const int p = pp[P], q = qq[P];
    for (int64_t R = 0; R < npair; ++R) {
      const int r = pp[R], s = qq[R];
      double w = PHYS(p, q, r, s) - PHYS(p, q, s, r);
      w += ((q == s ? H(p, r) : 0) - (q == r ? H(p, s) : 0) - (p == s ? H(q, r) : 0) +
            (p == r ? H(q, s) : 0)) / (nel - 1);
      v[P * npair + R] = w;
    }
  }
#undef SP
#undef SO
#undef IDX
#undef ERI
#undef PHYS
#undef H
  free(pp);
  free(qq);
  return v;
}

/* The (BB,BB) block of v (beta: the beta-beta pairs; zero elsewhere), or
 * the (AA,AA) block moved to the beta-beta pairs (alpha pair (p, q) ->
 * (norb + p, norb + q)): the integrals of the (AA,AA) part on vectors held
 * by beta rows, whose plans have the alpha strings on side B. */
static double *spin_block(const double *v, int norb, int beta) {
  const int ns = 2 * norb;
  const int64_t npair = (int64_t)ns * (ns - 1) / 2;
  double *w = calloc((size_t)(npair * npair), sizeof(double));
#define PAIR(a, b) ((int64_t)(b) * ((b) - 1) / 2 + (a))
#pragma omp parallel for schedule(dynamic, 1)
  for (int q = 1; q < norb; ++q)
    for (int p = 0; p < q; ++p) {
      const int64_t P = PAIR(p, q), Pb = PAIR(norb + p, norb + q);
      for (int s = 1; s < norb; ++s)
        for (int r = 0; r < s; ++r) {
          const int64_t R = PAIR(r, s), Rb = PAIR(norb + r, norb + s);
          w[Pb * npair + Rb] = beta ? v[Pb * npair + Rb] : v[P * npair + R];
        }
    }
#undef PAIR
  return w;
}

/* v += lam / (N - 1) times the fold of the one-electron operator h (ns x ns
 * over the spin orbitals) into the pairs, as in spin_orbital_v. */
static void fold_h1(double *v, int ns, const double *h, int nel, double lam) {
  const int64_t npair = (int64_t)ns * (ns - 1) / 2;
  int *pp = malloc(npair * sizeof(int)), *qq = malloc(npair * sizeof(int));
  for (int q = 1, P = 0; q < ns; ++q)
    for (int p = 0; p < q; ++p, ++P) pp[P] = p, qq[P] = q;
#define H(a, b) h[(size_t)(a) * ns + (b)]
#pragma omp parallel for schedule(dynamic, 16)
  for (int64_t P = 0; P < npair; ++P) {
    const int p = pp[P], q = qq[P];
    for (int64_t R = 0; R < npair; ++R) {
      const int r = pp[R], s = qq[R];
      const double w = (q == s ? H(p, r) : 0) - (q == r ? H(p, s) : 0) - (p == s ? H(q, r) : 0) +
                       (p == r ? H(q, s) : 0);
      if (w != 0) v[P * npair + R] += lam * w / (nel - 1);
    }
  }
#undef H
  free(pp);
  free(qq);
}

/* v += lam (S^2 - ss) in the format of spin_orbital_v, ss = |S_z| (|S_z| + 1)
 * the lowest value (the linear penalty of fix_spin_).
 * ovlp NULL (the same alpha and beta orbitals), as
 * socutils.hci.gasci.s2_antisymmetrized: S^2 = N_beta + S_z (S_z + 1) -
 * sum_pq a+_{p alpha} a+_{q beta} a_{p beta} a_{q alpha}; the constant
 * N_beta + S_z (S_z + 1) - ss = min(nalpha, nbeta) is folded in as the
 * diagonal 2 min / (N (N - 1)) of every pair (the same for the alpha-alpha
 * and the beta-beta pairs, as the spin flip needs).
 * ovlp[p * norb + q] = <p alpha | q beta> (different orbitals): S^2 = S- S+
 * + S_z (S_z + 1), S+ = sum_pq O_pq a+_{p alpha} a_{q beta}, S- S+ = sum_rq
 * (O^T O)_rq a+_{r beta} a_{q beta} - sum_pqrs O_sr O_pq a+_{p alpha}
 * a+_{r beta} a_{q beta} a_{s alpha}; S_z (S_z + 1) - ss folded in as a
 * constant. */
static void add_s2(double *v, int norb, int nalpha, int nbeta, double lam, const double *ovlp) {
  const int ns = 2 * norb, nel = nalpha + nbeta;
  const int64_t npair = (int64_t)ns * (ns - 1) / 2;
#define PAIR(a, b) ((int64_t)(b) * ((b) - 1) / 2 + (a))
  if (!ovlp) {
    const double diag = lam * 2.0 * (nalpha < nbeta ? nalpha : nbeta) / ((double)nel * (nel - 1));
    for (int q = 1; q < ns; ++q)
      for (int p = 0; p < q; ++p) v[PAIR(p, q) * (npair + 1)] += diag;
    for (int p = 0; p < norb; ++p)
      for (int q = 0; q < norb; ++q) v[PAIR(p, norb + q) * npair + PAIR(q, norb + p)] -= lam;
  } else {
    const double sz = 0.5 * (nalpha - nbeta), az = fabs(sz);
    const double c = (sz * (sz + 1) - az * (az + 1)) / nel;
    double *h = calloc((size_t)ns * ns, sizeof(double));
    for (int p = 0; p < ns; ++p) h[(size_t)p * ns + p] = c;
    for (int r = 0; r < norb; ++r)
      for (int q = 0; q < norb; ++q) {
        double o = 0;
        for (int p = 0; p < norb; ++p) o += ovlp[p * norb + r] * ovlp[p * norb + q];
        h[(size_t)(norb + r) * ns + norb + q] += o;
      }
    fold_h1(v, ns, h, nel, lam);
    free(h);
#pragma omp parallel for collapse(2) schedule(static)
    for (int p = 0; p < norb; ++p)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          for (int q = 0; q < norb; ++q)
            v[PAIR(p, norb + r) * npair + PAIR(s, norb + q)] -=
                lam * ovlp[s * norb + r] * ovlp[p * norb + q];
  }
#undef PAIR
}

static double s2_lowest(int nalpha, int nbeta) {
  const double sz = 0.5 * fabs((double)(nalpha - nbeta));
  return sz * (sz + 1);
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
  int norb, nalpha, nbeta, parity; /* parity 0: no symmetry (any alpha, beta lists) */
  int64_t n, lo, hi, nloc, len; /* alpha strings, own rows, own elements (nloc * nb) */
  int64_t *rows;                /* size + 1 */
  int64_t *strs;                /* sorted */
  int64_t *perm;                /* strs[k] = the input string perm[k] */
  /* beta strings (the columns); with parity != 0 the same arrays as alpha.
   * For the (AA,AA) part of parity 0, vectors are also held by beta rows
   * (nloc_b rows of n): own beta strings lo_b .. hi_b - 1. */
  int64_t nb, lo_b, hi_b, nloc_b, len_b;
  int64_t *rows_b, *strs_b, *perm_b;
  zg_plan *plan, *lplan, *lplan_b; /* all x all; own alpha x all beta; own beta x all alpha */
  double *xb, *zb;                 /* parity 0: two vectors held by beta rows */
  double *v, *v_bb, *v_aa;     /* H + s2pen S^2; its (BB,BB) block; its (AA,AA) block on the
                                 * beta-beta pairs (parity 0) */
  double *v_h, *v_bb_h, *v_aa_h; /* the same of H, if s2pen != 0 */
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
  const int64_t n = d->n; /* alpha strings (rows) */
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

/* A sorted copy of strs_in (n strings, no repeats), perm[k]: the input place of
 * string k; both newly allocated. */
static int64_t *sorted_copy(const int64_t *strs_in, int64_t n, int64_t **perm) {
  int64_t *pairs = malloc((size_t)(2 * n + 1) * sizeof(int64_t));
  for (int64_t k = 0; k < n; ++k) pairs[2 * k] = strs_in[k], pairs[2 * k + 1] = k;
  qsort(pairs, n, 2 * sizeof(int64_t), cmp_pair);
  int64_t *strs = malloc((size_t)(n + 1) * sizeof(int64_t));
  *perm = malloc((size_t)(n + 1) * sizeof(int64_t));
  for (int64_t k = 0; k < n; ++k) strs[k] = pairs[2 * k], (*perm)[k] = pairs[2 * k + 1];
  free(pairs);
  for (int64_t k = 1; k < n; ++k)
    if (strs[k] == strs[k - 1]) die("repeated strings");
  return strs;
}

static int64_t *even_rows(int64_t n) {
  int64_t *rows = malloc((size_ + 1) * sizeof(int64_t));
  for (int r = 0; r <= size_; ++r) rows[r] = (n * r) / size_;
  return rows;
}

/* The space strs_a x strs_b (nalpha, nbeta electrons; f->nelec their sum).
 * parity +-1: strs_b must be strs_a (nalpha = nbeta), states with C[b][a] =
 * parity C[a][b]; parity 0: any lists, no symmetry. */
static void dist_init(dist *d, const fcidump *f, int nalpha, int nbeta, const int64_t *strs_a,
                      int64_t n, const int64_t *strs_b, int64_t nb, int parity, double s2pen,
                      const double *ovlp) {
  memset(d, 0, sizeof *d);
  d->norb = f->norb;
  d->nalpha = nalpha, d->nbeta = nbeta;
  d->parity = parity;
  d->s2pen = s2pen;
  if (nalpha + nbeta != f->nelec) die("nalpha + nbeta != nelec");
  if (parity && (nalpha != nbeta || nb != n)) die("parity +-1 needs the same alpha and beta lists");
  if (parity && f->uhf) die("parity +-1 needs the same alpha and beta orbitals (not uhf)");
  if (s2pen != 0 && f->uhf && !ovlp) die("the S^2 penalty with uhf integrals needs the overlap");
  d->strs = sorted_copy(strs_a, n, &d->perm);
  d->n = n;
  d->rows = even_rows(n);
  d->lo = d->rows[rank_], d->hi = d->rows[rank_ + 1], d->nloc = d->hi - d->lo;
  if (parity) {
    d->nb = n, d->strs_b = d->strs, d->perm_b = d->perm, d->rows_b = d->rows;
  } else {
    d->strs_b = sorted_copy(strs_b, nb, &d->perm_b);
    d->nb = nb;
    d->rows_b = even_rows(nb);
  }
  d->lo_b = d->rows_b[rank_], d->hi_b = d->rows_b[rank_ + 1], d->nloc_b = d->hi_b - d->lo_b;
  d->len = d->nloc * d->nb;
  d->len_b = d->nloc_b * n;
  const int64_t *sa = d->strs, *sb = d->strs_b;
  const int ns = 2 * f->norb;
  d->plan = zg_plan_product(ns, f->nelec, f->norb, n, (const uint64_t *)sa, d->nb,
                            (const uint64_t *)sb, 0, NULL, NULL, NULL);
  if (!d->plan) die(zg_error());
  if (zg_dim(d->plan) != n * d->nb) die("strings left out of the product plan (electron counts?)");
  if (size_ == 1) {
    d->lplan = d->plan;
  } else if (d->nloc > 0) {
    d->lplan = zg_plan_product(ns, f->nelec, f->norb, d->nloc, (const uint64_t *)(sa + d->lo),
                               d->nb, (const uint64_t *)sb, 0, NULL, NULL, NULL);
    if (!d->lplan) die(zg_error());
  }
  /* (AA,AA) of parity 0: the kernel of (BB,BB) on vectors held by beta rows,
   * a plan with the own beta strings on side A and all alpha strings on B */
  const int same = !parity && nalpha == nbeta && nb == n &&
                   !memcmp(sa, sb, (size_t)n * sizeof(int64_t));
  if (same) { /* the same strings on both sides: the same plan (the excitation tables are
               * those of the strings; only the integrals of the two parts differ) */
    d->lplan_b = d->lplan;
  } else if (!parity && d->nloc_b > 0) {
    d->lplan_b = zg_plan_product(ns, f->nelec, f->norb, d->nloc_b, (const uint64_t *)(sb + d->lo_b),
                                 n, (const uint64_t *)sa, 0, NULL, NULL, NULL);
    if (!d->lplan_b) die(zg_error());
  }
  d->v = spin_orbital_v(f);
  if (s2pen != 0) {
    const int64_t np2 = (int64_t)ns * (ns - 1) / 2 * ((int64_t)ns * (ns - 1) / 2);
    d->v_h = malloc((size_t)np2 * sizeof(double));
    memcpy(d->v_h, d->v, (size_t)np2 * sizeof(double));
    add_s2(d->v, f->norb, nalpha, nbeta, s2pen, f->uhf ? ovlp : NULL);
    d->v_bb_h = spin_block(d->v_h, f->norb, 1);
    if (!parity) d->v_aa_h = spin_block(d->v_h, f->norb, 0);
  }
  d->v_bb = spin_block(d->v, f->norb, 1);
  if (!parity) d->v_aa = spin_block(d->v, f->norb, 0);
  halo_setup(d);
  d->zt = malloc((size_t)(d->len + 1) * sizeof(double));
  if (!parity) {
    d->xb = malloc((size_t)(d->len_b + 1) * sizeof(double));
    d->zb = malloc((size_t)(d->len_b + 1) * sizeof(double));
  }
  if (d->nhalo) d->hbuf = malloc((size_t)((d->nloc + d->nhalo) * d->nb + 1) * sizeof(double));
  /* the largest block of the transposes (both ways) and of the halo */
  int64_t smax = 1, rmax = 1;
  for (int q = 0; q < size_; ++q) {
    const int64_t wa = d->rows[q + 1] - d->rows[q], wb = d->rows_b[q + 1] - d->rows_b[q];
    if (d->nloc * wb > smax) smax = d->nloc * wb;
    if (d->nloc_b * wa > smax) smax = d->nloc_b * wa;
    if (wa * d->nloc_b > rmax) rmax = wa * d->nloc_b;
    if (wb * d->nloc > rmax) rmax = wb * d->nloc;
    if (d->hsend_cnt[q] * d->nb > smax) smax = d->hsend_cnt[q] * d->nb;
  }
  d->sbuf_len = smax, d->rbuf_len = rmax;
  d->sbuf = malloc((size_t)smax * sizeof(double));
  d->rbuf = malloc((size_t)rmax * sizeof(double));
}

static void dist_free(dist *d) {
  if (d->lplan && d->lplan != d->plan) zg_plan_free(d->lplan);
  if (d->lplan_b && d->lplan_b != d->lplan) zg_plan_free(d->lplan_b);
  if (d->plan) zg_plan_free(d->plan);
  if (d->strs_b != d->strs) free(d->strs_b), free(d->perm_b), free(d->rows_b);
  free(d->xb), free(d->zb);
  free(d->strs), free(d->perm), free(d->rows);
  free(d->v), free(d->v_bb), free(d->v_aa), free(d->v_h), free(d->v_bb_h), free(d->v_aa_h);
  free(d->smap), free(d->hsend_cnt), free(d->hsend_off), free(d->hsend_rows);
  free(d->hrecv_cnt), free(d->hrecv_off), free(d->hbuf), free(d->zt), free(d->sbuf), free(d->rbuf);
  memset(d, 0, sizeof *d);
}

/* X (R x C) distributed by rows (rpart; own rows x, nlx of them) -> the own
 * rows y (nly) of X^T distributed by rows by cpart (the columns of X). */
static void transpose_g(dist *d, const double *x, int64_t nlx, const int64_t *rpart, double *y,
                        int64_t nly, const int64_t *cpart) {
  const int64_t C = cpart[size_], R = rpart[size_];
  zg_transpose_d(nlx, nly, x + cpart[rank_], C, y + rpart[rank_], R);
  for (int k = 1; k < size_; ++k) {
    const int dest = (rank_ + k) % size_, src = (rank_ - k + size_) % size_;
    const int64_t c0 = cpart[dest], wd = cpart[dest + 1] - c0;
    const int64_t s0 = rpart[src], ws = rpart[src + 1] - s0;
    zg_copy_block_d(nlx, wd, x + c0, C, d->sbuf, wd);
    sendrecv(d->sbuf, nlx * wd, dest, d->rbuf, ws * nly, src);
    zg_transpose_d(ws, nly, d->rbuf, nly, y + s0, R);
  }
}

/* out = the own rows of X^T (X distributed by rows; parity != 0: square). */
static void transpose(dist *d, const double *x, double *out) {
  transpose_g(d, x, d->nloc, d->rows, out, d->nloc, d->rows);
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
  const int64_t n = d->nb; /* row length */
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

/* out = H c for c of the symmetry.  (BB,BB) row-local; (AA,AA) = parity
 * (BB,BB)^T, or with parity 0 the same kernel on c held by beta rows
 * (transposed there and back); (AB,AB) from the own and the halo rows. */
static void sigma(dist *d, const double *c, double *out) {
  const double t0 = now();
  double tz = 0;
  if (d->nloc) zg_sigma_blas_d(d->lplan, d->v_bb, c, out);
  const double t1 = now();
  if (d->parity) {
    transpose(d, out, d->zt);
    axpy((double)d->parity, d->zt, out, d->len);
  } else {
    transpose_g(d, c, d->nloc, d->rows, d->xb, d->nloc_b, d->rows_b);
    const double ta = now();
    if (d->nloc_b) zg_sigma_blas_d(d->lplan_b, d->v_aa, d->xb, d->zb);
    tz = now() - ta;
    transpose_g(d, d->zb, d->nloc_b, d->rows_b, d->zt, d->nloc, d->rows);
    axpy(1.0, d->zt, out, d->len);
  }
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
  d->t_z += t1 - t0 + tz, d->t_tr += t2 - t1 - tz, d->t_halo += t3 - t2, d->t_ab += t4 - t3;
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

/* ---- eigensolvers ---------------------------------------------------------- */

/* 0: LOBPCG-like (6 vectors), 1: Davidson (max_space_ vectors and their
 * products); zgd_set_eigensolver, or ZGASCI_DIST_SOLVER = lobpcg | davidson
 * and ZGASCI_DIST_MAX_SPACE. */
static int esolver_ = 0, max_space_ = 12, esolver_set_ = 0;

/* (an explicit choice wins over the environment) */
void zgd_set_eigensolver(int kind, int max_space) {
  esolver_set_ = 1;
  esolver_ = kind == 1;
  if (max_space >= 3) max_space_ = max_space;
}

static void eigensolver_env(void) {
  if (esolver_set_) return;
  esolver_set_ = 1;
  const char *s = getenv("ZGASCI_DIST_SOLVER");
  if (s && !strcmp(s, "davidson")) esolver_ = 1;
  if (s && !strcmp(s, "lobpcg")) esolver_ = 0;
  if (getenv("ZGASCI_DIST_MAX_SPACE") && atoi(getenv("ZGASCI_DIST_MAX_SPACE")) >= 3)
    max_space_ = atoi(getenv("ZGASCI_DIST_MAX_SPACE"));
}

/* x = x0, or the lowest diagonal element (symmetrized), normalized. */
static void start_vector(dist *d, const double *hd, const double *x0, double *x) {
  const int64_t L = d->len;
  if (x0) {
    copy(x0, x, L);
    return;
  }
  memset(x, 0, (size_t)L * sizeof(double));
  struct { double v; int r; } loc = {INFINITY, rank_}, glob;
  int64_t imin = -1;
  /* (odd parity: off the diagonal a = b, where the symmetrized vector vanishes) */
  for (int64_t i = 0; i < L; ++i)
    if (hd[i] < loc.v && (d->parity >= 0 || d->lo + i / d->nb != i % d->nb)) loc.v = hd[i], imin = i;
  MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm_);
  if (rank_ == glob.r) x[imin] = 1;
  if (d->parity) symmetrize(d, x);
  scal(1 / sqrt(gdot(x, x, L)), x, L);
}

/* t = -t / max(|hd - e|, shift): positive definite, as LOBPCG needs (hd - e
 * changes sign e.g. with an S^2 penalty and odd parity). */
static void precondition(const double *hd, double e, double shift, double *t, int64_t L) {
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < L; ++i) t[i] /= -fmax(fabs(hd[i] - e), shift);
}

/* The lowest state; *xout: its own rows (normalized), for the caller to free.
 * x0: own rows of a start vector (of the symmetry, normalized), or NULL. */
static double lobpcg(dist *d, const double *x0, int maxit, double tol, double rtol, double shift,
                     int *cycles, int *converged, double **xout) {
  const int64_t L = d->len;
  double *hd = malloc((size_t)(L + 1) * sizeof(double));
  if (d->nloc) zg_hdiag_d(d->lplan, d->v, hd);
  double *x = calloc((size_t)L + 1, sizeof(double)), *hx = malloc((size_t)(L + 1) * sizeof(double));
  double *t = malloc((size_t)(L + 1) * sizeof(double)), *ht = malloc((size_t)(L + 1) * sizeof(double));
  double *p = NULL, *hp = NULL;
  start_vector(d, hd, x0, x);
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
    precondition(hd, e, shift, t, L);
    if (d->parity) symmetrize(d, t);
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


/* out[i] = <v[i], t> for i < m, summed over the processes (one Allreduce). */
static void gdots(double *const *v, int m, const double *t, int64_t L, double *out) {
  for (int i = 0; i < m; ++i) {
    double s = 0;
    const double *vi = v[i];
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (int64_t k = 0; k < L; ++k) s += vi[k] * t[k];
    out[i] = s;
  }
  MPI_Allreduce(MPI_IN_PLACE, out, m, MPI_DOUBLE, MPI_SUM, comm_);
}

/* out = sum_i y[i] v[i], i < m. */
static void combine(double *const *v, const double *y, int m, int64_t L, double *out) {
#pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < L; ++k) {
    double s = 0;
    for (int i = 0; i < m; ++i) s += y[i] * v[i][k];
    out[k] = s;
  }
}

/* Davidson for the lowest state: a basis V of at most msp vectors (own
 * rows) with HV, the projections of H and of the overlap (Rayleigh-Ritz by
 * dsygv, so that a small loss of orthogonality does no harm); the Ritz
 * vector X, the preconditioned residual (as LOBPCG), symmetrized,
 * orthogonalized to V twice and added unless it is linearly dependent on V
 * (norm below lindep of its norm before); a full basis, or a dependent new
 * direction, collapses the basis to X and the previous Ritz vector (no
 * extra products); a dependent direction right after a collapse ends the
 * iteration (stagnation).  2 msp + 5 vectors of the own rows.  As lobpcg
 * otherwise. */
static double davidson(dist *d, const double *x0, int maxit, double tol, double rtol, double shift,
                       int msp, int *cycles, int *converged, double **xout) {
  const double lindep = 1e-8;
  const int64_t L = d->len;
  const size_t vb = (size_t)(L + 1) * sizeof(double);
  double *hd = malloc(vb);
  if (d->nloc) zg_hdiag_d(d->lplan, d->v, hd);
  double **V = calloc(msp, sizeof(double *)), **HV = calloc(msp, sizeof(double *));
  for (int i = 0; i < msp; ++i) V[i] = malloc(vb), HV[i] = malloc(vb);
  double *x = malloc(vb), *hx = malloc(vb), *xo = malloc(vb), *hxo = malloc(vb), *r = malloc(vb);
  double *hs = calloc((size_t)msp * msp, sizeof(double)), *ss = calloc((size_t)msp * msp, sizeof(double));
  double *a = malloc((size_t)msp * msp * sizeof(double)), *b = malloc((size_t)msp * msp * sizeof(double));
  double *w = malloc(msp * sizeof(double)), *c = malloc((msp + 1) * sizeof(double));
  const int lw = 64 * msp + 64;
  double *work = malloc(lw * sizeof(double));
  /* row / column j of the projections (V[0..j] against V[j], HV[j]) */
#define PROJECT(j)                                                                     \
  do {                                                                                 \
    gdots(V, (j) + 1, HV[j], L, c);                                                    \
    for (int i_ = 0; i_ <= (j); ++i_) hs[i_ * msp + (j)] = hs[(j) * msp + i_] = c[i_]; \
    gdots(V, (j) + 1, V[j], L, c);                                                     \
    for (int i_ = 0; i_ <= (j); ++i_) ss[i_ * msp + (j)] = ss[(j) * msp + i_] = c[i_]; \
  } while (0)
  start_vector(d, hd, x0, V[0]);
  sigma(d, V[0], HV[0]);
  PROJECT(0);
  int m = 1, have_old = 0, collapse = 0, just_collapsed = 0;
  double *last = NULL; /* the buffer of the last Ritz vector */
  double e = 0, e_old = NAN, rnorm = 0;
  const double t0 = now();
  *converged = 0;
  int it;
  for (it = 0; it < maxit; ++it) {
    /* the lowest Ritz pair of the basis */
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j) a[i * m + j] = hs[i * msp + j], b[i * m + j] = ss[i * msp + j];
    const int one = 1;
    int info;
    dsygv_(&one, "V", "U", &m, a, &m, b, &m, w, work, &lw, &info);
    if (info) die("davidson: dsygv failed (basis not independent)");
    e = w[0];
    combine(V, a, m, L, x); /* the first eigenvector (y^T S y = 1) */
    last = x;
    combine(HV, a, m, L, hx);
    copy(hx, r, L);
    axpy(-e, x, r, L);
    rnorm = sqrt(gdot(r, r, L));
    if (rank_ == 0 && verbose_) {
      printf("  davidson %3d  E %.12f  dE %.2e  |r| %.2e  space %2d  t %.1f s  peak %.1f GB\n", it,
             e, isnan(e_old) ? 0.0 : e - e_old, rnorm, m, now() - t0, peak_gb());
      fflush(stdout);
    }
    if ((!isnan(e_old) && fabs(e - e_old) < tol && rnorm < rtol) || rnorm < 1e-12) {
      *converged = 1;
      break;
    }
    e_old = e;
    just_collapsed = 0;
    if (m == msp || collapse) { /* X and the part of the previous X orthogonal to it */
      const double nx = sqrt(gdot(x, x, L));
      copy(x, V[0], L), copy(hx, HV[0], L);
      scal(1 / nx, V[0], L), scal(1 / nx, HV[0], L);
      m = 1;
      if (have_old) {
        const double ov = gdot(V[0], xo, L);
        axpy(-ov, V[0], xo, L);
        axpy(-ov, HV[0], hxo, L);
        const double nn = sqrt(gdot(xo, xo, L));
        if (nn > lindep) {
          copy(xo, V[1], L), copy(hxo, HV[1], L);
          scal(1 / nn, V[1], L), scal(1 / nn, HV[1], L);
          m = 2;
        }
      }
      for (int j = 0; j < m; ++j) PROJECT(j);
      collapse = 0, just_collapsed = 1;
    }
    /* the new direction */
    precondition(hd, e, shift, r, L);
    if (d->parity) symmetrize(d, r);
    const double n0 = sqrt(gdot(r, r, L));
    for (int pass = 0; pass < 2; ++pass) {
      gdots(V, m, r, L, c);
      for (int i = 0; i < m; ++i) axpy(-c[i], V[i], r, L);
    }
    const double nt = sqrt(gdot(r, r, L));
    double *tmp = xo;
    xo = x, x = tmp;
    tmp = hxo, hxo = hx, hx = tmp;
    have_old = 1;
    if (!(nt > lindep * n0) || !(nt > 1e-14)) {
      if (just_collapsed) break; /* stagnation: no new direction even in the small basis */
      collapse = 1;              /* retry from the collapsed basis */
      --it;                      /* (the same Ritz pair again: not an iteration) */
      e_old = NAN;
      continue;
    }
    copy(r, V[m], L);
    scal(1 / nt, V[m], L);
    sigma(d, V[m], HV[m]);
    PROJECT(m);
    ++m;
  }
#undef PROJECT
  *cycles = it + 1;
  if (rank_ == 0 && verbose_) {
    printf("davidson: %d iterations, converged %d, |r| %.2e, %.1f s (max space %d)\n", *cycles,
           *converged, rnorm, now() - t0, msp);
    fflush(stdout);
  }
  for (int i = 0; i < msp; ++i) free(V[i]), free(HV[i]);
  if (!last) last = x;
  if (x != last) free(x);
  if (xo != last) free(xo);
  free(V), free(HV), free(hd), free(hx), free(hxo), free(r), free(hs), free(ss), free(a);
  free(b), free(w), free(c), free(work);
  *xout = last;
  return e;
}

/* The chosen eigensolver (esolver_). */
static double eigensolve(dist *d, const double *x0, int maxit, double tol, double rtol,
                         double shift, int *cycles, int *converged, double **xout) {
  if (esolver_ == 1)
    return davidson(d, x0, maxit, tol, rtol, shift, max_space_, cycles, converged, xout);
  return lobpcg(d, x0, maxit, tol, rtol, shift, cycles, converged, xout);
}

/* After the solve of x (own rows, normalized; e_pen = <x|H + s2pen (S^2 -
 * ss)|x>): out = {e_pen, <H>, <S^2> (NaN without a penalty)}, one more
 * product if penalized; per alpha string in the input order the largest |C|
 * of its row and its weight sum_b C[a][b]^2 (maxabs, wa), the same per beta
 * string over its column (maxabs_b, wb; = alpha for parity != 0), and the
 * occupations occ[p] = sum_a wa[a] n_p(a), occ_b likewise. */
static void finish(dist *d, const double *x, double e_pen, double *out, double *occ, double *occ_b,
                   double *maxabs, double *maxabs_b, double *wa, double *wb) {
  const int64_t n = d->n, nb = d->nb, L = d->len;
  out[0] = out[1] = e_pen, out[2] = NAN;
  if (d->s2pen != 0) {
    double *hx = malloc((size_t)(L + 1) * sizeof(double));
    double *v = d->v, *v_bb = d->v_bb, *v_aa = d->v_aa;
    d->v = d->v_h, d->v_bb = d->v_bb_h, d->v_aa = d->v_aa_h;
    sigma(d, x, hx);
    d->v = v, d->v_bb = v_bb, d->v_aa = v_aa;
    out[1] = gdot(x, hx, L);
    out[2] = (e_pen - out[1]) / d->s2pen + s2_lowest(d->nalpha, d->nbeta);
    free(hx);
  }
  double *gw = calloc((size_t)n + 1, sizeof(double)), *gm = calloc((size_t)n + 1, sizeof(double));
#pragma omp parallel for schedule(static)
  for (int64_t a = d->lo; a < d->hi; ++a) {
    const double *row = x + (a - d->lo) * nb;
    double w = 0, m = 0;
    for (int64_t b = 0; b < nb; ++b) {
      w += row[b] * row[b];
      if (fabs(row[b]) > m) m = fabs(row[b]);
    }
    gw[a] = w, gm[a] = m;
  }
  MPI_Allreduce(MPI_IN_PLACE, gw, (int)n, MPI_DOUBLE, MPI_SUM, comm_);
  MPI_Allreduce(MPI_IN_PLACE, gm, (int)n, MPI_DOUBLE, MPI_MAX, comm_);
  double *bw = gw, *bm = gm;
  if (!d->parity) { /* over the columns: own rows, then summed over the processes */
    bw = calloc((size_t)nb + 1, sizeof(double)), bm = calloc((size_t)nb + 1, sizeof(double));
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nb; ++b) {
      double w = 0, m = 0;
      for (int64_t a = 0; a < d->nloc; ++a) {
        const double y = x[a * nb + b];
        w += y * y;
        if (fabs(y) > m) m = fabs(y);
      }
      bw[b] = w, bm[b] = m;
    }
    MPI_Allreduce(MPI_IN_PLACE, bw, (int)nb, MPI_DOUBLE, MPI_SUM, comm_);
    MPI_Allreduce(MPI_IN_PLACE, bm, (int)nb, MPI_DOUBLE, MPI_MAX, comm_);
  }
  if (occ) {
    for (int p = 0; p < d->norb; ++p) occ[p] = 0;
    for (int64_t k = 0; k < n; ++k)
      for (int p = 0; p < d->norb; ++p)
        if ((d->strs[k] >> p) & 1) occ[p] += gw[k];
  }
  if (occ_b) {
    for (int p = 0; p < d->norb; ++p) occ_b[p] = 0;
    for (int64_t k = 0; k < nb; ++k)
      for (int p = 0; p < d->norb; ++p)
        if ((d->strs_b[k] >> p) & 1) occ_b[p] += bw[k];
  }
  for (int64_t k = 0; k < n; ++k) {
    if (wa) wa[d->perm[k]] = gw[k];
    if (maxabs) maxabs[d->perm[k]] = gm[k];
  }
  for (int64_t k = 0; k < nb; ++k) {
    if (wb) wb[d->perm_b[k]] = bw[k];
    if (maxabs_b) maxabs_b[d->perm_b[k]] = bm[k];
  }
  if (bw != gw) free(bw), free(bm);
  free(gw), free(gm);
}

/* ---- warm start between calls ------------------------------------------ */

/* A solution kept for the next calls: its sorted strings, the row ranges,
 * the own rows of the vector (on this process). */
typedef struct {
  int64_t n, *strs, *rows, lo, nloc, nb, *strs_b; /* strs_b: own copy only for parity 0 */
  int size, parity, nalpha, nbeta;
  double e, *x;
} saved;
static saved prev_, cand_; /* the start of this round, the best of this round */

static void saved_free(saved *s) {
  if (s->strs_b != s->strs) free(s->strs_b);
  free(s->strs), free(s->rows), free(s->x);
  memset(s, 0, sizeof *s);
}

/* Keep x (taking it over) as cand_ if cand_ is empty or e is lower. */
static int keep_candidate(dist *d, double *x, double e) {
  if (cand_.x && !(e < cand_.e)) return 0;
  saved_free(&cand_);
  cand_.n = d->n, cand_.lo = d->lo, cand_.nloc = d->nloc, cand_.size = size_;
  cand_.parity = d->parity, cand_.e = e, cand_.x = x;
  cand_.nalpha = d->nalpha, cand_.nbeta = d->nbeta, cand_.nb = d->nb;
  cand_.strs = malloc((size_t)(d->n + 1) * sizeof(int64_t));
  memcpy(cand_.strs, d->strs, (size_t)d->n * sizeof(int64_t));
  cand_.strs_b = cand_.strs;
  if (!d->parity) {
    cand_.strs_b = malloc((size_t)(d->nb + 1) * sizeof(int64_t));
    memcpy(cand_.strs_b, d->strs_b, (size_t)d->nb * sizeof(int64_t));
  }
  cand_.rows = malloc((size_t)(size_ + 1) * sizeof(int64_t));
  memcpy(cand_.rows, d->rows, (size_t)(size_ + 1) * sizeof(int64_t));
  return 1;
}

/* x0 (own rows) = the saved vector s restricted to the space of d (its
 * elements whose alpha and beta strings are both in d), symmetrized and
 * normalized.  Old rows are fetched from their owners in blocks of rows
 * (pairwise exchanges).  Returns the squared norm of the restriction (the
 * weight of the old vector kept), 0 if nothing of it is in the space. */
static double warm_vector(dist *d, const saved *s, double *x0) {
  const int64_t n = d->nb, no = s->nb, nao = s->n; /* row lengths (beta strings); old alpha */
  memset(x0, 0, (size_t)d->len * sizeof(double));
  int64_t *colmap = malloc((size_t)(n + 1) * sizeof(int64_t));
#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < n; ++b) colmap[b] = find(s->strs_b, no, d->strs_b[b]);
  /* the old rows each own row needs, grouped by their owner */
  int64_t *want_cnt = calloc(size_, sizeof(int64_t)), *want_off = calloc(size_ + 1, sizeof(int64_t));
  int64_t *kold = malloc((size_t)(d->nloc + 1) * sizeof(int64_t));
  int *owner = malloc((size_t)(d->nloc + 1) * sizeof(int));
  for (int64_t a = 0; a < d->nloc; ++a) {
    kold[a] = find(s->strs, nao, d->strs[d->lo + a]);
    owner[a] = -1;
    if (kold[a] < 0) continue;
    int q = 0;
    while (kold[a] >= s->rows[q + 1]) ++q;
    owner[a] = q;
    want_cnt[q]++;
  }
  for (int q = 0; q < size_; ++q) want_off[q + 1] = want_off[q] + want_cnt[q];
  int64_t *want = malloc((size_t)(want_off[size_] + 1) * sizeof(int64_t)); /* old row places */
  int64_t *dst = malloc((size_t)(want_off[size_] + 1) * sizeof(int64_t));  /* own new rows */
  {
    int64_t *fill = calloc(size_, sizeof(int64_t));
    for (int64_t a = 0; a < d->nloc; ++a)
      if (owner[a] >= 0) {
        const int64_t i = want_off[owner[a]] + fill[owner[a]]++;
        want[i] = kold[a], dst[i] = a;
      }
    free(fill);
  }
  int64_t *give_cnt = calloc(size_, sizeof(int64_t)), *give_off = calloc(size_ + 1, sizeof(int64_t));
  MPI_Alltoall(want_cnt, 1, MPI_INT64_T, give_cnt, 1, MPI_INT64_T, comm_);
  for (int q = 0; q < size_; ++q) give_off[q + 1] = give_off[q] + give_cnt[q];
  int64_t *give = malloc((size_t)(give_off[size_] + 1) * sizeof(int64_t));
  {
    int *sc = malloc(size_ * sizeof(int)), *so = malloc(size_ * sizeof(int));
    int *rc = malloc(size_ * sizeof(int)), *ro = malloc(size_ * sizeof(int));
    for (int q = 0; q < size_; ++q) {
      sc[q] = (int)want_cnt[q], so[q] = (int)want_off[q];
      rc[q] = (int)give_cnt[q], ro[q] = (int)give_off[q];
    }
    MPI_Alltoallv(want, sc, so, MPI_INT64_T, give, rc, ro, MPI_INT64_T, comm_);
    free(sc), free(so), free(rc), free(ro);
  }
  /* blocks of B rows; every process makes the same number of steps */
  const int64_t B = no > 0 && (CHUNK / no) > 0 ? CHUNK / no : 1;
  int64_t mx = 0;
  for (int q = 0; q < size_; ++q) {
    if (want_cnt[q] > mx) mx = want_cnt[q];
    if (give_cnt[q] > mx) mx = give_cnt[q];
  }
  MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_INT64_T, MPI_MAX, comm_);
  const int64_t nblk = (mx + B - 1) / B;
  const int64_t bl = B < mx ? B : mx;
  double *sb = malloc((size_t)(bl * no + 1) * sizeof(double));
  double *rb = malloc((size_t)(bl * no + 1) * sizeof(double));
  for (int k = 0; k < size_; ++k) {
    const int dest = (rank_ + k) % size_, src = (rank_ - k + size_) % size_;
    for (int64_t j = 0; j < nblk; ++j) {
      const int64_t s0 = j * B, r0 = j * B;
      int64_t ns = give_cnt[dest] - s0, nr = want_cnt[src] - r0;
      ns = ns < 0 ? 0 : (ns > B ? B : ns);
      nr = nr < 0 ? 0 : (nr > B ? B : nr);
      const int64_t *g = give + give_off[dest] + s0;
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < ns; ++i)
        memcpy(sb + i * no, s->x + (g[i] - s->lo) * no, (size_t)no * sizeof(double));
      if (dest == rank_) memcpy(rb, sb, (size_t)(ns * no) * sizeof(double));
      else sendrecv(sb, ns * no, dest, rb, nr * no, src);
      const int64_t *t = dst + want_off[src] + r0;
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < nr; ++i) {
        double *row = x0 + t[i] * n;
        const double *old = rb + i * no;
        for (int64_t b = 0; b < n; ++b)
          if (colmap[b] >= 0) row[b] = old[colmap[b]];
      }
    }
  }
  free(sb), free(rb), free(give), free(give_cnt), free(give_off), free(want), free(dst);
  free(want_cnt), free(want_off), free(kold), free(owner), free(colmap);
  if (d->parity) symmetrize(d, x0);
  const double w = gdot(x0, x0, d->len);
  if (w > 0) scal(1 / sqrt(w), x0, d->len);
  return w;
}

/* Start a new round: the best solution kept in the last one becomes the
 * start of the next calls (zgd_solve with warm & 1). */
void zgd_promote(void) {
  if (!cand_.x) return;
  saved_free(&prev_);
  prev_ = cand_;
  memset(&cand_, 0, sizeof cand_);
}

/* Forget the kept solutions. */
void zgd_reset(void) {
  saved_free(&prev_);
  saved_free(&cand_);
}

/* The library entry: the lowest state of H (+ s2pen (S^2 - ss), ss = |S_z|
 * (|S_z| + 1)) in the product space strs_a x strs_b (n alpha strings of
 * nalpha electrons, nb beta strings of nbeta; int64, bit p: orbital p; any
 * order, no repeats), on the communicator comm_f (Fortran handle; every
 * process calls with the same arguments).  parity +-1: strs_b = strs_a,
 * nalpha = nbeta, the states with C[b][a] = parity C[a][b] (even / odd S);
 * parity 0: no symmetry.  h1: norb^2, eri: norb^4 (pq|rs), no core energy;
 * uhf (parity 0): different alpha and beta orbitals, h1 = (alpha, beta)
 * 2 norb^2, eri = ((aa|aa), (aa|bb), (bb|bb)) 3 norb^4 ((p_a q_a | r_b s_b)
 * in the middle), and ovlp[p * norb + q] = <p alpha | q beta> for the S^2
 * penalty (NULL without one).
 * warm: bit 1, start from the solution of the last round (zgd_promote) if
 * there is one of the same kind (parity, electrons, processes) and it keeps
 * a weight >= 1e-4 in this space (else the lowest diagonal element); bit 2,
 * keep this solution if it is the lowest of the round so far.
 * out[7] = {E of the penalized H, <H>, <S^2> (NaN if s2pen = 0), iterations,
 * converged, seconds, weight of the warm start (-1: none)}; occ_a, occ_b
 * [norb], maxabs_a, wa [n], maxabs_b, wb [nb] as finish (any may be NULL),
 * the same on every process.  Errors abort the communicator. */
int zgd_solve_ab(int comm_f, int norb, int nalpha, int nbeta, int uhf, const double *h1,
                 const double *eri, const double *ovlp, int64_t n, const int64_t *strs_a,
                 int64_t nb, const int64_t *strs_b, int parity, double s2pen, double tol,
                 double rtol, int maxit, int verbose, int warm, double *out, double *occ_a,
                 double *occ_b, double *maxabs_a, double *maxabs_b, double *wa, double *wb) {
  comm_ = MPI_Comm_f2c((MPI_Fint)comm_f);
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
  verbose_ = verbose;
  if (getenv("ZGASCI_DIST_CHUNK") && atol(getenv("ZGASCI_DIST_CHUNK")) > 0)
    CHUNK = atol(getenv("ZGASCI_DIST_CHUNK"));
  if (parity != 1 && parity != -1 && parity != 0) die("parity must be +1, -1 or 0");
  eigensolver_env();
  if (n > INT32_MAX || nb > INT32_MAX) die("more strings than an MPI count");
  const double t0 = now();
  const size_t n2 = (size_t)norb * norb, n4 = n2 * n2;
  fcidump f = {norb, nalpha + nbeta, 0.0, (double *)h1, (double *)eri, uhf != 0,
               uhf ? (double *)h1 + n2 : NULL, uhf ? (double *)eri + n4 : NULL,
               uhf ? (double *)eri + 2 * n4 : NULL};
  dist d;
  dist_init(&d, &f, nalpha, nbeta, strs_a, n, strs_b, nb, parity, s2pen, ovlp);
  if (rank_ == 0 && verbose_) {
    printf("zgd_solve: %d processes x %d threads; norb %d nalpha %d nbeta %d; strings %ld x %ld, "
           "dim %.4e, parity %d; setup %.1f s\n", size_, omp_get_max_threads(), norb, nalpha, nbeta,
           (long)n, (long)nb, (double)n * nb, parity, now() - t0);
    fflush(stdout);
  }
  int cycles, conv;
  double *x, *x0 = NULL, w = -1;
  if ((warm & 1) && prev_.x && prev_.parity == parity && prev_.size == size_ &&
      prev_.nalpha == nalpha && prev_.nbeta == nbeta) {
    x0 = malloc((size_t)(d.len + 1) * sizeof(double));
    w = warm_vector(&d, &prev_, x0);
    if (rank_ == 0 && verbose_) {
      printf("zgd_solve: warm start, weight of the last solution in this space %.6f\n", w);
      fflush(stdout);
    }
    if (w < 1e-4) free(x0), x0 = NULL;
  }
  const double e = eigensolve(&d, x0, maxit, tol, rtol, 1e-3, &cycles, &conv, &x);
  free(x0);
  finish(&d, x, e, out, occ_a, occ_b, maxabs_a, maxabs_b, wa, wb);
  out[3] = cycles, out[4] = conv, out[5] = now() - t0, out[6] = w;
  if (!((warm & 2) && keep_candidate(&d, x, e))) free(x);
  dist_free(&d);
  return 0;
}

/* zgd_solve_ab with strs_b = strs_a, nalpha = nbeta = nelec / 2, parity
 * +-1 (occ, maxabs, wa: of the alpha = beta strings). */
int zgd_solve(int comm_f, int norb, int nelec, const double *h1, const double *eri, int64_t nstr,
              const int64_t *strs, int parity, double s2pen, double tol, double rtol, int maxit,
              int verbose, int warm, double *out, double *occ, double *maxabs, double *wa) {
  if (nelec % 2) {
    fprintf(stderr, "zgd_solve: nalpha = nbeta is required\n");
    MPI_Abort(MPI_Comm_f2c((MPI_Fint)comm_f), 1);
  }
  if (parity == 0) {
    fprintf(stderr, "zgd_solve: parity +-1 (zgd_solve_ab for parity 0)\n");
    MPI_Abort(MPI_Comm_f2c((MPI_Fint)comm_f), 1);
  }
  return zgd_solve_ab(comm_f, norb, nelec / 2, nelec / 2, 0, h1, eri, NULL, nstr, strs, nstr, strs, parity,
                      s2pen, tol, rtol, maxit, verbose, warm, out, occ, NULL, maxabs, NULL, wa,
                      NULL);
}

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  comm_ = MPI_COMM_WORLD;
  eigensolver_env();
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
    else if (!strcmp(argv[i], "--solver")) esolver_ = !strcmp(argv[i + 1], "davidson");
    else if (!strcmp(argv[i], "--max-space")) max_space_ = atoi(argv[i + 1]);
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
  dist_init(&d, &f, f.nelec / 2, f.nelec / 2, strs, n, strs, n, parity, s2pen, NULL);
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
    if (parity) symmetrize(&d, x);
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
  const double e = eigensolve(&d, NULL, maxit, tol, rtol, shift, &cycles, &conv, &x);
  const double t_solve = now() - tl;
  finish(&d, x, e, res, NULL, NULL, NULL, NULL, NULL, NULL);
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
