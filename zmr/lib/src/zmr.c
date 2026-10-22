#include "zmr.h"

#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* MKL service functions (libmkl_rt) */
void MKL_Set_Num_Threads(int n);
/* CBLAS of MKL (row major 101, upper 121, no / transpose / conjugate transpose 111 / 112 / 113);
 * sequential when called inside a parallel region */
void cblas_zherk(int layout, int uplo, int trans, int n, int k, double alpha,
                 const void *a, int lda, double beta, void *c, int ldc);
void cblas_dsyrk(int layout, int uplo, int trans, int n, int k, double alpha,
                 const double *a, int lda, double beta, double *c, int ldc);

void zmr_set_num_threads(int n)
{
        omp_set_num_threads(n);
        MKL_Set_Num_Threads(n);
}

int zmr_get_max_threads(void)
{
        return omp_get_max_threads();
}

/* Number of threads worth starting for a loop of the given size (in
 * elementary operations): at least GRAIN operations per thread.  Small
 * problems run on the calling thread; they are called thousands of times
 * (one external block, one symmetry block at a time) and forking a full
 * team for each of them costs more than the work. */
#define GRAIN 2.0e5

static int threads_for(double work)
{
        int n = omp_get_max_threads();
        double want = work / GRAIN;
        if (want < 1.5) {
                return 1;
        }
        return want < n ? (int)want : n;
}

void zmr_apply_op(const int64_t *target, const int64_t *sign, int norb,
                  int64_t nsrc, int64_t ndst, int64_t nvec,
                  const double complex *vecs, double complex *out)
{
#pragma omp parallel for collapse(2) schedule(static) \
        num_threads(threads_for((double)nvec * norb * (nsrc + ndst)))
        for (int64_t v = 0; v < nvec; v++) {
                for (int p = 0; p < norb; p++) {
                        double complex *o = out + (v * norb + p) * ndst;
                        const double complex *x = vecs + v * nsrc;
                        const int64_t *t = target + (int64_t)p * nsrc;
                        const int64_t *s = sign + (int64_t)p * nsrc;
                        memset(o, 0, sizeof(double complex) * ndst);
                        for (int64_t i = 0; i < nsrc; i++) {
                                if (t[i] >= 0) {
                                        o[t[i]] = (double)s[i] * x[i];
                                }
                        }
                }
        }
}

/* ----------------------------------------------------------------------
 * External sums of the doubly external classes from three-index integrals,
 *
 *     W[m][X, Y] = sum_e conj(v[e, X]) v[e, Y] / (d[m] + delta[e]),
 *
 * for a list of energy arguments d (interpolation nodes).  The external
 * labels are grouped by core orbital (pair); one group is one task: it
 * multiplies its slices of the three-index integrals (one BLAS product),
 * antisymmetrizes, and adds to its own copy of W.  Tasks share nothing but
 * read-only input, as independent processes would; the copies of W are
 * added at the end.  With fewer tasks than threads the tasks run one after
 * the other and the threads go into each task (BLAS, loop over virtuals).
 *
 * Three-index integrals: bvc[L, a, i] (naux, nv, nc) and bva[L, b, t]
 * (naux, nv * na).  They may be slices of
 * larger arrays (cd_pc, cd_pa): row L starts ld elements after row L - 1. */
void cblas_zgemm(int layout, int transa, int transb, int m, int n, int k,
                 const void *alpha, const void *a, int lda, const void *b, int ldb,
                 const void *beta, void *c, int ldc);
int MKL_Set_Num_Threads_Local(int n);

static void core_slice(const double complex *b, int64_t ld, int64_t naux, int64_t n,
                       int64_t nc, int64_t i, double complex *out)
{
        for (int64_t l = 0; l < naux; l++) {
                const double complex *row = b + l * ld + i;
                for (int64_t x = 0; x < n; x++) {
                        out[l * n + x] = row[x * nc];
                }
        }
}

/* m = x^T y for x (naux, nx), y (naux, ny) */
static void product_t(int64_t naux, const double complex *x, int64_t nx,
                      const double complex *y, int64_t ny, int64_t ldy, double complex *m)
{
        const double complex one = 1.0, zero = 0.0;
        cblas_zgemm(101, 112, 111, (int)nx, (int)ny, (int)naux, &one, x, (int)nx, y, (int)ldy,
                    &zero, m, (int)ny);
}

static int outer_tasks(int64_t ntask, int mode)
{
        if (mode == 0 || mode == 1) {
                return mode;
        }
        return 2 * ntask >= zmr_get_max_threads();
}

/* ccvv: tasks (i < j), v[ab] = (ai|bj) - (aj|bi), a < b; w[m] is real */
void zmr_w_ccvv(const double complex *bvc, int64_t ldvc, int64_t naux, int64_t nv, int64_t nc,
                const double *ev, const double *ec, const double *d, int64_t nd,
                int mode, double *w)
{
        memset(w, 0, sizeof(double) * nd);
        int64_t ntask = nc * (nc - 1) / 2;
        if (ntask <= 0 || nv < 2 || nd == 0) {
                return;
        }
        int outer = outer_tasks(ntask, mode);
#pragma omp parallel if(outer)
        {
                double complex *bi = malloc(sizeof(double complex) * naux * nv);
                double complex *bj = malloc(sizeof(double complex) * naux * nv);
                double complex *m = malloc(sizeof(double complex) * nv * nv);
                double *loc = calloc(nd, sizeof(double));
#pragma omp for schedule(dynamic, 1)
                for (int64_t task = 0; task < ntask; task++) {
                        int64_t j = (int64_t)((1.0 + sqrt(1.0 + 8.0 * task)) / 2.0);
                        while (j * (j - 1) / 2 > task) j--;
                        while ((j + 1) * j / 2 <= task) j++;
                        int64_t i = task - j * (j - 1) / 2;
                        core_slice(bvc, ldvc, naux, nv, nc, i, bi);
                        core_slice(bvc, ldvc, naux, nv, nc, j, bj);
                        product_t(naux, bi, nv, bj, nv, nv, m);
                        double e0 = -ec[i] - ec[j];
#pragma omp parallel if(!outer)
                        {
                                double *part = outer ? loc : calloc(nd, sizeof(double));
#pragma omp for schedule(dynamic, 8)
                                for (int64_t a = 0; a < nv; a++) {
                                        for (int64_t b = a + 1; b < nv; b++) {
                                                double complex v = m[a * nv + b] - m[b * nv + a];
                                                double x = creal(v) * creal(v) + cimag(v) * cimag(v);
                                                double del = ev[a] + ev[b] + e0;
#pragma omp simd
                                                for (int64_t k = 0; k < nd; k++) {
                                                        part[k] += x / (d[k] + del);
                                                }
                                        }
                                }
                                if (!outer) {
#pragma omp critical
                                        for (int64_t k = 0; k < nd; k++) loc[k] += part[k];
                                        free(part);
                                }
                        }
                }
#pragma omp critical
                for (int64_t k = 0; k < nd; k++) w[k] += loc[k];
                free(bi);
                free(bj);
                free(m);
                free(loc);
        }
}

/* one external label with the active vector x[t]: w[m][t, u] += conj(x_t) x_u / (d_m + del) */
static inline void add_outer(int64_t na, const double complex *x, double del,
                             const double *d, int64_t nd, double complex *w)
{
        for (int64_t t = 0; t < na; t++) {
                for (int64_t u = 0; u < na; u++) {
                        double complex p = conj(x[t]) * x[u];
                        double complex *dst = w + t * na + u;
                        for (int64_t k = 0; k < nd; k++) {
                                dst[k * na * na] += p / (d[k] + del);
                        }
                }
        }
}

/* aavv: no core index; tasks are segments of the first virtual orbital a.
 * v[ab, tu] = (at|bu) - (au|bt), a < b, t < u.  A task multiplies its
 * segment of bva with all of bva, and for every a collects the rows b > a
 * into a matrix x[b, tu]; the sum over b is a rank-k update per energy
 * argument (rows scaled by the square root of the inverse denominator), so
 * the post-processing runs in BLAS as well.  ev must be ascending in neither
 * sense: any order works. */
void zmr_w_aavv(const double complex *bva, int64_t ldva, int64_t naux, int64_t nv, int64_t na,
                const double *ev, const double *d, int64_t nd, double complex *w)
{
        int64_t np = na * (na - 1) / 2;
        int64_t nw = nd * np * np;
        memset(w, 0, sizeof(double complex) * nw);
        if (nv < 2 || np < 1 || nd == 0) {
                return;
        }
        int nthreads = zmr_get_max_threads();
        int64_t seg = nv / (4 * (int64_t)nthreads);
        if (seg < 1) seg = 1;
        if (seg > 64) seg = 64;
        int64_t ntask = (nv + seg - 1) / seg;
#pragma omp parallel
        {
                double complex *bs = malloc(sizeof(double complex) * naux * seg * na);
                double complex *g = malloc(sizeof(double complex) * seg * na * nv * na);
                double complex *x = malloc(sizeof(double complex) * nv * np);
                double complex *sc = malloc(sizeof(double complex) * nv * np);
                double *del = malloc(sizeof(double) * nv);
                double complex *loc = calloc(nw, sizeof(double complex));
#pragma omp for schedule(dynamic, 1)
                for (int64_t task = 0; task < ntask; task++) {
                        int64_t a0 = task * seg;
                        int64_t a1 = a0 + seg < nv ? a0 + seg : nv;
                        int64_t ns = a1 - a0;
                        int64_t b0 = a0 + 1;               /* first b needed */
                        int64_t nbt = nv - b0;
                        if (nbt <= 0) continue;
                        for (int64_t l = 0; l < naux; l++) {
                                memcpy(bs + l * ns * na, bva + l * ldva + a0 * na,
                                       sizeof(double complex) * ns * na);
                        }
                        /* g[(a, t), (b, u)] = (at|bu), b >= b0 */
                        product_t(naux, bs, ns * na, bva + b0 * na, nbt * na, ldva, g);
                        int64_t ldg = nbt * na;
                        for (int64_t a = a0; a < a1; a++) {
                                int64_t nb = nv - a - 1;
                                if (nb <= 0) continue;
                                const double complex *ga = g + (a - a0) * na * ldg;
                                int positive = 1;
                                for (int64_t k = 0; k < nb; k++) {
                                        int64_t b = a + 1 + k;
                                        del[k] = ev[a] + ev[b];
                                        const double complex *col = ga + (b - b0) * na;
                                        int64_t p = 0;
                                        for (int64_t t = 0; t < na; t++) {
                                                for (int64_t u = t + 1; u < na; u++) {
                                                        x[k * np + p++] = col[t * ldg + u] - col[u * ldg + t];
                                                }
                                        }
                                }
                                for (int64_t m = 0; m < nd; m++) {
                                        for (int64_t k = 0; k < nb; k++) {
                                                positive &= d[m] + del[k] > 0;
                                        }
                                }
                                for (int64_t m = 0; m < nd; m++) {
                                        double complex *wm = loc + m * np * np;
                                        if (positive) {
                                                for (int64_t k = 0; k < nb; k++) {
                                                        double f = sqrt(1.0 / (d[m] + del[k]));
                                                        for (int64_t p = 0; p < np; p++) {
                                                                sc[k * np + p] = f * x[k * np + p];
                                                        }
                                                }
                                                cblas_zherk(101, 121, 113, (int)np, (int)nb, 1.0, sc,
                                                            (int)np, 1.0, wm, (int)np);
                                        } else {
                                                for (int64_t k = 0; k < nb; k++) {
                                                        double f = 1.0 / (d[m] + del[k]);
                                                        for (int64_t p = 0; p < np; p++) {
                                                                double complex c = conj(x[k * np + p]) * f;
                                                                for (int64_t q = p; q < np; q++) {
                                                                        wm[p * np + q] += c * x[k * np + q];
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
#pragma omp critical
                for (int64_t k = 0; k < nw; k++) w[k] += loc[k];
                free(bs); free(g); free(x); free(sc); free(del); free(loc);
        }
        /* the updates filled the upper triangles */
        for (int64_t m = 0; m < nd; m++) {
                double complex *wm = w + m * np * np;
                for (int64_t p = 0; p < np; p++) {
                        wm[p * np + p] = creal(wm[p * np + p]);
                        for (int64_t q = p + 1; q < np; q++) {
                                wm[q * np + p] = conj(wm[p * np + q]);
                        }
                }
        }
}

/* cavv: tasks i, v[ab, t] = -((ai|bt) - (at|bi)), a < b */
void zmr_w_cavv(const double complex *bvc, int64_t ldvc, const double complex *bva,
                int64_t ldva, int64_t naux,
                int64_t nv, int64_t na, int64_t nc, const double *ev, const double *ec,
                const double *d, int64_t nd, int mode, double complex *w)
{
        int64_t nw = nd * na * na;
        memset(w, 0, sizeof(double complex) * nw);
        if (nc < 1 || nv < 2 || na < 1 || nd == 0) {
                return;
        }
        int outer = outer_tasks(nc, mode);
#pragma omp parallel if(outer)
        {
                double complex *bi = malloc(sizeof(double complex) * naux * nv);
                double complex *m = malloc(sizeof(double complex) * nv * nv * na);
                double complex *loc = calloc(nw, sizeof(double complex));
#pragma omp for schedule(dynamic, 1)
                for (int64_t i = 0; i < nc; i++) {
                        core_slice(bvc, ldvc, naux, nv, nc, i, bi);
                        product_t(naux, bi, nv, bva, nv * na, ldva, m);      /* m[a, b, t] = (ai|bt) */
#pragma omp parallel if(!outer)
                        {
                                double complex *part = outer ? loc : calloc(nw, sizeof(double complex));
                                double complex *x = malloc(sizeof(double complex) * na);
#pragma omp for schedule(dynamic, 4)
                                for (int64_t a = 0; a < nv; a++) {
                                        for (int64_t b = a + 1; b < nv; b++) {
                                                for (int64_t t = 0; t < na; t++) {
                                                        x[t] = m[(b * nv + a) * na + t] - m[(a * nv + b) * na + t];
                                                }
                                                add_outer(na, x, ev[a] + ev[b] - ec[i], d, nd, part);
                                        }
                                }
                                free(x);
                                if (!outer) {
#pragma omp critical
                                        for (int64_t k = 0; k < nw; k++) loc[k] += part[k];
                                        free(part);
                                }
                        }
                }
#pragma omp critical
                for (int64_t k = 0; k < nw; k++) w[k] += loc[k];
                free(bi);
                free(m);
                free(loc);
        }
}

/* ----------------------------------------------------------------------
 * The same external sums for spin-restricted orbitals (zmr.nr.classes): real
 * three-index integrals with spatial indices, the labels and parameters of
 * the spin-free classes.  Tasks and threading as above. */
void cblas_dgemm(int layout, int transa, int transb, int m, int n, int k,
                 double alpha, const double *a, int lda, const double *b, int ldb,
                 double beta, double *c, int ldc);

static void core_slice_r(const double *b, int64_t ld, int64_t naux, int64_t n,
                         int64_t nc, int64_t i, double *out)
{
        for (int64_t l = 0; l < naux; l++) {
                const double *row = b + l * ld + i;
                for (int64_t x = 0; x < n; x++) {
                        out[l * n + x] = row[x * nc];
                }
        }
}

static void product_t_r(int64_t naux, const double *x, int64_t nx,
                        const double *y, int64_t ny, int64_t ldy, double *m)
{
        cblas_dgemm(101, 112, 111, (int)nx, (int)ny, (int)naux, 1.0, x, (int)nx, y, (int)ldy,
                    0.0, m, (int)ny);
}

/* ccvv, tasks (i <= j) with m[a, b] = (ai|bj):
 *   i < j, a < b:  2 [(ai|bj) - (aj|bi)]^2     (same-spin pairs)
 *   all i, j, a, b:  (ai|bj)^2                 (opposite spins)
 * over e_a + e_b - e_i - e_j. */
void zmr_wsf_ccvv(const double *bvc, int64_t ldvc, int64_t naux, int64_t nv, int64_t nc,
                  const double *ev, const double *ec, const double *d, int64_t nd,
                  int mode, double *w)
{
        memset(w, 0, sizeof(double) * nd);
        int64_t ntask = nc * (nc + 1) / 2;
        if (ntask <= 0 || nv < 1 || nd == 0) {
                return;
        }
        int outer = outer_tasks(ntask, mode);
#pragma omp parallel if(outer)
        {
                double *bi = malloc(sizeof(double) * naux * nv);
                double *bj = malloc(sizeof(double) * naux * nv);
                double *m = malloc(sizeof(double) * nv * nv);
                double *loc = calloc(nd, sizeof(double));
#pragma omp for schedule(dynamic, 1)
                for (int64_t task = 0; task < ntask; task++) {
                        int64_t j = (int64_t)((sqrt(8.0 * task + 1.0) - 1.0) / 2.0);
                        while (j * (j + 1) / 2 > task) j--;
                        while ((j + 1) * (j + 2) / 2 <= task) j++;
                        int64_t i = task - j * (j + 1) / 2;
                        core_slice_r(bvc, ldvc, naux, nv, nc, i, bi);
                        core_slice_r(bvc, ldvc, naux, nv, nc, j, bj);
                        product_t_r(naux, bi, nv, bj, nv, nv, m);
                        double e0 = -ec[i] - ec[j];
                        const double both = i < j ? 2.0 : 1.0;   /* (i, j) and (j, i) */
                        const double anti = i < j ? 2.0 : 0.0;
#pragma omp parallel if(!outer)
                        {
                                double *part = outer ? loc : calloc(nd, sizeof(double));
#pragma omp for schedule(dynamic, 8)
                                for (int64_t a = 0; a < nv; a++) {
                                        for (int64_t b = a; b < nv; b++) {
                                                double mab = m[a * nv + b], mba = m[b * nv + a];
                                                double x;
                                                if (a == b) {
                                                        x = both * mab * mab;
                                                } else {
                                                        x = both * (mab * mab + mba * mba)
                                                            + anti * (mab - mba) * (mab - mba);
                                                }
                                                double del = ev[a] + ev[b] + e0;
#pragma omp simd
                                                for (int64_t k = 0; k < nd; k++) {
                                                        part[k] += x / (d[k] + del);
                                                }
                                        }
                                }
                                if (!outer) {
#pragma omp critical
                                        for (int64_t k = 0; k < nd; k++) loc[k] += part[k];
                                        free(part);
                                }
                        }
                }
#pragma omp critical
                for (int64_t k = 0; k < nd; k++) w[k] += loc[k];
                free(bi);
                free(bj);
                free(m);
                free(loc);
        }
}

static inline void add_outer_r(int64_t n, const double *x, double del,
                               const double *d, int64_t nd, double *w)
{
        for (int64_t t = 0; t < n; t++) {
                for (int64_t u = 0; u < n; u++) {
                        double p = x[t] * x[u];
                        double *dst = w + t * n + u;
                        for (int64_t k = 0; k < nd; k++) {
                                dst[k * n * n] += p / (d[k] + del);
                        }
                }
        }
}

/* cavv (a < b; i): parameters x = [(ai|bt), (bi|at)] (2 na), over e_a + e_b - e_i */
void zmr_wsf_cavv(const double *bvc, int64_t ldvc, const double *bva, int64_t ldva,
                  int64_t naux, int64_t nv, int64_t na, int64_t nc,
                  const double *ev, const double *ec, const double *d, int64_t nd,
                  int mode, double *w)
{
        int64_t n2 = 2 * na;
        int64_t nw = nd * n2 * n2;
        memset(w, 0, sizeof(double) * nw);
        if (nc < 1 || nv < 2 || na < 1 || nd == 0) {
                return;
        }
        int outer = outer_tasks(nc, mode);
#pragma omp parallel if(outer)
        {
                double *bi = malloc(sizeof(double) * naux * nv);
                double *m = malloc(sizeof(double) * nv * nv * na);
                double *loc = calloc(nw, sizeof(double));
#pragma omp for schedule(dynamic, 1)
                for (int64_t i = 0; i < nc; i++) {
                        core_slice_r(bvc, ldvc, naux, nv, nc, i, bi);
                        product_t_r(naux, bi, nv, bva, nv * na, ldva, m);    /* m[a, b, t] = (ai|bt) */
#pragma omp parallel if(!outer)
                        {
                                double *part = outer ? loc : calloc(nw, sizeof(double));
                                double *x = malloc(sizeof(double) * n2);
#pragma omp for schedule(dynamic, 4)
                                for (int64_t a = 0; a < nv; a++) {
                                        for (int64_t b = a + 1; b < nv; b++) {
                                                for (int64_t t = 0; t < na; t++) {
                                                        x[t] = m[(a * nv + b) * na + t];
                                                        x[na + t] = m[(b * nv + a) * na + t];
                                                }
                                                add_outer_r(n2, x, ev[a] + ev[b] - ec[i], d, nd, part);
                                        }
                                }
                                free(x);
                                if (!outer) {
#pragma omp critical
                                        for (int64_t k = 0; k < nw; k++) loc[k] += part[k];
                                        free(part);
                                }
                        }
                }
#pragma omp critical
                for (int64_t k = 0; k < nw; k++) w[k] += loc[k];
                free(bi);
                free(m);
                free(loc);
        }
}

/* aavv (a < b): parameters x[t na + u] = (at|bu), all (t, u), over e_a + e_b.
 * Tasks are segments of a; the sum over b is a rank-k update per argument. */
void zmr_wsf_aavv(const double *bva, int64_t ldva, int64_t naux, int64_t nv, int64_t na,
                  const double *ev, const double *d, int64_t nd, double *w)
{
        int64_t np = na * na;
        int64_t nw = nd * np * np;
        memset(w, 0, sizeof(double) * nw);
        if (nv < 2 || np < 1 || nd == 0) {
                return;
        }
        int nthreads = zmr_get_max_threads();
        int64_t seg = nv / (4 * (int64_t)nthreads);
        if (seg < 1) seg = 1;
        if (seg > 64) seg = 64;
        int64_t ntask = (nv + seg - 1) / seg;
#pragma omp parallel
        {
                double *bs = malloc(sizeof(double) * naux * seg * na);
                double *g = malloc(sizeof(double) * seg * na * nv * na);
                double *x = malloc(sizeof(double) * nv * np);
                double *sc = malloc(sizeof(double) * nv * np);
                double *del = malloc(sizeof(double) * nv);
                double *loc = calloc(nw, sizeof(double));
#pragma omp for schedule(dynamic, 1)
                for (int64_t task = 0; task < ntask; task++) {
                        int64_t a0 = task * seg;
                        int64_t a1 = a0 + seg < nv ? a0 + seg : nv;
                        int64_t ns = a1 - a0;
                        int64_t b0 = a0 + 1;
                        int64_t nbt = nv - b0;
                        if (nbt <= 0) continue;
                        for (int64_t l = 0; l < naux; l++) {
                                memcpy(bs + l * ns * na, bva + l * ldva + a0 * na,
                                       sizeof(double) * ns * na);
                        }
                        /* g[(a, t), (b, u)] = (at|bu), b >= b0 */
                        product_t_r(naux, bs, ns * na, bva + b0 * na, nbt * na, ldva, g);
                        int64_t ldg = nbt * na;
                        for (int64_t a = a0; a < a1; a++) {
                                int64_t nb = nv - a - 1;
                                if (nb <= 0) continue;
                                const double *ga = g + (a - a0) * na * ldg;
                                int positive = 1;
                                for (int64_t k = 0; k < nb; k++) {
                                        int64_t b = a + 1 + k;
                                        del[k] = ev[a] + ev[b];
                                        const double *col = ga + (b - b0) * na;
                                        for (int64_t t = 0; t < na; t++) {
                                                for (int64_t u = 0; u < na; u++) {
                                                        x[k * np + t * na + u] = col[t * ldg + u];
                                                }
                                        }
                                }
                                for (int64_t m = 0; m < nd; m++) {
                                        for (int64_t k = 0; k < nb; k++) {
                                                positive &= d[m] + del[k] > 0;
                                        }
                                }
                                for (int64_t m = 0; m < nd; m++) {
                                        double *wm = loc + m * np * np;
                                        if (positive) {
                                                for (int64_t k = 0; k < nb; k++) {
                                                        double f = sqrt(1.0 / (d[m] + del[k]));
                                                        for (int64_t p = 0; p < np; p++) {
                                                                sc[k * np + p] = f * x[k * np + p];
                                                        }
                                                }
                                                cblas_dsyrk(101, 121, 112, (int)np, (int)nb, 1.0, sc,
                                                            (int)np, 1.0, wm, (int)np);
                                        } else {
                                                for (int64_t k = 0; k < nb; k++) {
                                                        double f = 1.0 / (d[m] + del[k]);
                                                        for (int64_t p = 0; p < np; p++) {
                                                                double c = x[k * np + p] * f;
                                                                for (int64_t q = p; q < np; q++) {
                                                                        wm[p * np + q] += c * x[k * np + q];
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
#pragma omp critical
                for (int64_t k = 0; k < nw; k++) w[k] += loc[k];
                free(bs); free(g); free(x); free(sc); free(del); free(loc);
        }
        for (int64_t m = 0; m < nd; m++) {
                double *wm = w + m * np * np;
                for (int64_t p = 0; p < np; p++) {
                        for (int64_t q = p + 1; q < np; q++) {
                                wm[q * np + p] = wm[p * np + q];
                        }
                }
        }
}

/* ----------------------------------------------------------------------
 * f[k, j] += sum_e w[e] / (lam[k] + delta[e] - shifts[j]) for the pairs with
 * mask[k, j] != 0: the sum over the external labels of a class with a single
 * active operator (ccvv), needed only where state j has a component k. */
void zmr_shifted_sums(const double *w, const double *delta, int64_t ne,
                      const double *lam, int64_t nk, const double *shifts, int64_t ns,
                      const unsigned char *mask, double *f)
{
#pragma omp parallel for collapse(2) schedule(dynamic, 8) \
        num_threads(threads_for((double)nk * ns * ne))
        for (int64_t k = 0; k < nk; k++) {
                for (int64_t j = 0; j < ns; j++) {
                        if (!mask[k * ns + j]) {
                                continue;
                        }
                        const double d0 = lam[k] - shifts[j];
                        double a = 0;
#pragma omp simd reduction(+:a)
                        for (int64_t e = 0; e < ne; e++) {
                                a += w[e] / (d0 + delta[e]);
                        }
                        f[k * ns + j] += a;
                }
        }
}

/* ----------------------------------------------------------------------
 * a[I, J] = - sum_{k, e} conj(u[I, k, e]) u[J, k, e] / (lam[k] + delta[e] - s_J)
 *
 * Complex numbers are kept as separate real and imaginary arrays, so that
 * the same code serves real input (cplx = 0: the imaginary arrays are not
 * touched) and the loops vectorize. */

/* sr + i si = sum_e conj(x[e]) w[e] */
static inline void conj_dot(int64_t n, const double *xr, const double *xi,
                            const double *wr, const double *wi, double *sr, double *si)
{
        double ar = 0, ai = 0;
#pragma omp simd reduction(+:ar, ai)
        for (int64_t e = 0; e < n; e++) {
                ar += xr[e] * wr[e] + xi[e] * wi[e];
                ai += xr[e] * wi[e] - xi[e] * wr[e];
        }
        *sr = ar;
        *si = ai;
}

static inline double real_dot(int64_t n, const double *x, const double *w)
{
        double a = 0;
#pragma omp simd reduction(+:a)
        for (int64_t e = 0; e < n; e++) {
                a += x[e] * w[e];
        }
        return a;
}

static inline void inverse_denominators(int64_t ne, double d0, const double *delta, double *inv)
{
#pragma omp simd
        for (int64_t e = 0; e < ne; e++) {
                inv[e] = 1.0 / (d0 + delta[e]);
        }
}

static inline void scale(int64_t n, const double *x, const double *f, double *w)
{
#pragma omp simd
        for (int64_t e = 0; e < n; e++) {
                w[e] = x[e] * f[e];
        }
}

/* split a complex array into real and imaginary parts */
static void split_complex(const double complex *z, int64_t n, double **re, double **im)
{
        *re = malloc(sizeof(double) * (n > 0 ? n : 1));
        *im = malloc(sizeof(double) * (n > 0 ? n : 1));
        double *r = *re, *i = *im;
#pragma omp parallel for schedule(static) num_threads(threads_for((double)n))
        for (int64_t x = 0; x < n; x++) {
                r[x] = creal(z[x]);
                i[x] = cimag(z[x]);
        }
}

static void join_complex(const double *re, const double *im, int64_t n, double complex *z)
{
        for (int64_t x = 0; x < n; x++) {
                z[x] = re[x] + im[x] * I;
        }
}

/* fills xr (+ i xi) = u[j, k, :] and returns 0 if the row vanishes */
typedef int (*row_fn)(const void *ctx, int64_t k, int64_t j, double *xr, double *xi);

/* column j of the contribution of one k: out[i] -= sum_e conj(u_i) u_j / D */
static void gram_column(int cplx, int64_t ns, int64_t ne, int64_t j,
                        const double *ur, const double *ui, const int *live, const double *inv,
                        double *wr, double *wi, double *outr, double *outi, int64_t stride)
{
        scale(ne, ur + j * ne, inv, wr);
        if (cplx) {
                scale(ne, ui + j * ne, inv, wi);
        }
        for (int64_t i = 0; i < ns; i++) {
                if (!live[i]) {
                        continue;
                }
                if (cplx) {
                        double sr, si;
                        conj_dot(ne, ur + i * ne, ui + i * ne, wr, wi, &sr, &si);
                        outr[i * stride] -= sr;
                        outi[i * stride] -= si;
                } else {
                        outr[i * stride] -= real_dot(ne, ur + i * ne, wr);
                }
        }
}

/* Rows u[J, k, :] supplied by fill.  Threads share the loop over k if it is
 * long enough, otherwise the loops over the states for every k.  col >= 0:
 * only that column J.  ar, ai (ns, ns) are overwritten (ai only if cplx). */
static void gram_rows(row_fn fill, const void *ctx, int cplx, int64_t ns, int64_t nk, int64_t ne,
                      int64_t row_cost, const double *lam, const double *delta,
                      const double *shifts, int64_t col, double *ar, double *ai)
{
        int64_t nn = ns * ns;
        memset(ar, 0, sizeof(double) * nn);
        if (cplx) {
                memset(ai, 0, sizeof(double) * nn);
        }
        if (nk == 0 || ne == 0 || ns == 0) {
                return;
        }
        int64_t j0 = col < 0 ? 0 : col;
        int64_t j1 = col < 0 ? ns : col + 1;
        /* row_cost: operations per element of a row u[J, k, :] */
        int nthreads = threads_for((double)nk * ns * ne * ((j1 - j0) + row_cost));
        if (nk >= 4 * (int64_t)nthreads) {
#pragma omp parallel num_threads(nthreads)
                {
                        double *locr = calloc(nn, sizeof(double));
                        double *loci = calloc(cplx ? nn : 1, sizeof(double));
                        double *ur = malloc(sizeof(double) * ns * ne);
                        double *ui = malloc(sizeof(double) * (cplx ? ns * ne : 1));
                        double *wr = malloc(sizeof(double) * ne);
                        double *wi = malloc(sizeof(double) * ne);
                        double *inv = malloc(sizeof(double) * ne);
                        int *live = malloc(sizeof(int) * ns);
#pragma omp for schedule(dynamic, 4)
                        for (int64_t k = 0; k < nk; k++) {
                                for (int64_t j = 0; j < ns; j++) {
                                        live[j] = fill(ctx, k, j, ur + j * ne, ui + (cplx ? j * ne : 0));
                                }
                                int have = 0;
                                double last = 0;
                                for (int64_t j = j0; j < j1; j++) {
                                        if (!live[j]) {
                                                continue;
                                        }
                                        /* the denominators depend on J only through its shift */
                                        if (!have || shifts[j] != last) {
                                                inverse_denominators(ne, lam[k] - shifts[j], delta, inv);
                                                last = shifts[j];
                                                have = 1;
                                        }
                                        gram_column(cplx, ns, ne, j, ur, ui, live, inv, wr, wi,
                                                    locr + j, cplx ? loci + j : NULL, ns);
                                }
                        }
#pragma omp critical
                        {
                                for (int64_t x = 0; x < nn; x++) {
                                        ar[x] += locr[x];
                                }
                                if (cplx) {
                                        for (int64_t x = 0; x < nn; x++) {
                                                ai[x] += loci[x];
                                        }
                                }
                        }
                        free(locr);
                        free(loci);
                        free(ur);
                        free(ui);
                        free(wr);
                        free(wi);
                        free(inv);
                        free(live);
                }
                return;
        }
        /* few k: all threads walk through k together and share the states */
        double *ur = malloc(sizeof(double) * ns * ne);
        double *ui = malloc(sizeof(double) * (cplx ? ns * ne : 1));
        int *live = malloc(sizeof(int) * ns);
#pragma omp parallel num_threads(nthreads < ns ? nthreads : (int)ns)
        {
                double *wr = malloc(sizeof(double) * ne);
                double *wi = malloc(sizeof(double) * ne);
                double *inv = malloc(sizeof(double) * ne);
                for (int64_t k = 0; k < nk; k++) {
#pragma omp for schedule(static)
                        for (int64_t j = 0; j < ns; j++) {
                                live[j] = fill(ctx, k, j, ur + j * ne, ui + (cplx ? j * ne : 0));
                        }
                        int have = 0;
                        double last = 0;
#pragma omp for schedule(dynamic, 1)
                        for (int64_t j = j0; j < j1; j++) {
                                if (!live[j]) {
                                        continue;
                                }
                                if (!have || shifts[j] != last) {
                                        inverse_denominators(ne, lam[k] - shifts[j], delta, inv);
                                        last = shifts[j];
                                        have = 1;
                                }
                                /* column j is written by this thread only */
                                gram_column(cplx, ns, ne, j, ur, ui, live, inv, wr, wi,
                                            ar + j, cplx ? ai + j : NULL, ns);
                        }
                }
                free(wr);
                free(wi);
                free(inv);
        }
        free(ur);
        free(ui);
        free(live);
}

/* ---- rows stored: u (ns, nk, ne) */

struct stored_rows {
        const double *ur, *ui;          /* ui == NULL for real u */
        int64_t nk, ne;
};

static int fill_stored(const void *ctx, int64_t k, int64_t j, double *xr, double *xi)
{
        const struct stored_rows *c = ctx;
        size_t off = (size_t)(j * c->nk + k) * c->ne;
        memcpy(xr, c->ur + off, sizeof(double) * c->ne);
        if (c->ui) {
                memcpy(xi, c->ui + off, sizeof(double) * c->ne);
        }
        return 1;
}

void zmr_weighted_gram(const double complex *u, int64_t ns, int64_t nk, int64_t ne,
                       const double *lam, const double *delta, const double *shifts,
                       int64_t col, double complex *a)
{
        double *ur, *ui;
        split_complex(u, ns * nk * ne, &ur, &ui);
        double *ar = malloc(sizeof(double) * ns * ns * 2);
        struct stored_rows ctx = {ur, ui, nk, ne};
        gram_rows(fill_stored, &ctx, 1, ns, nk, ne, 1, lam, delta, shifts, col, ar, ar + ns * ns);
        join_complex(ar, ar + ns * ns, ns * ns, a);
        free(ar);
        free(ur);
        free(ui);
}

void zmr_weighted_gram_r(const double *u, int64_t ns, int64_t nk, int64_t ne,
                         const double *lam, const double *delta, const double *shifts,
                         int64_t col, double *a)
{
        struct stored_rows ctx = {u, NULL, nk, ne};
        gram_rows(fill_stored, &ctx, 0, ns, nk, ne, 1, lam, delta, shifts, col, a, NULL);
}

/* ---- rows formed on the fly: u[J, k, :] = sum_X t[J, k, X] v[:, X] */

struct product_rows {
        const double *tr, *ti;          /* (ns, nk, nx); ti == NULL: real */
        const double *vr, *vi;          /* (nx, ne) */
        int64_t nk, nx, ne;
};

static int fill_product(const void *ctx, int64_t k, int64_t j, double *xr, double *xi)
{
        const struct product_rows *c = ctx;
        size_t off = (size_t)(j * c->nk + k) * c->nx;
        int64_t ne = c->ne;
        int live = 0;
        for (int64_t x = 0; x < c->nx; x++) {
                double tr = c->tr[off + x];
                double ti = c->ti ? c->ti[off + x] : 0.0;
                if (tr == 0 && ti == 0) {
                        continue;
                }
                if (!live) {
                        memset(xr, 0, sizeof(double) * ne);
                        if (c->ti) {
                                memset(xi, 0, sizeof(double) * ne);
                        }
                        live = 1;
                }
                const double *yr = c->vr + x * ne;
                if (c->ti) {
                        const double *yi = c->vi + x * ne;
#pragma omp simd
                        for (int64_t e = 0; e < ne; e++) {
                                xr[e] += tr * yr[e] - ti * yi[e];
                                xi[e] += tr * yi[e] + ti * yr[e];
                        }
                } else {
#pragma omp simd
                        for (int64_t e = 0; e < ne; e++) {
                                xr[e] += tr * yr[e];
                        }
                }
        }
        return live;
}

/* v[e, X] -> (X, e) */
static double *transposed(const double *v, int64_t ne, int64_t nx)
{
        double *out = malloc(sizeof(double) * (nx * ne > 0 ? nx * ne : 1));
#pragma omp parallel for schedule(static) num_threads(threads_for((double)nx * ne))
        for (int64_t x = 0; x < nx; x++) {
                for (int64_t e = 0; e < ne; e++) {
                        out[x * ne + e] = v[e * nx + x];
                }
        }
        return out;
}

void zmr_contract_t(const double complex *t, int64_t ns, int64_t nk, int64_t nx,
                    const double complex *v, int64_t ne,
                    const double *lam, const double *delta, const double *shifts,
                    int64_t col, double complex *a)
{
        double *tr, *ti, *vr0, *vi0;
        split_complex(t, ns * nk * nx, &tr, &ti);
        split_complex(v, ne * nx, &vr0, &vi0);
        double *vr = transposed(vr0, ne, nx);
        double *vi = transposed(vi0, ne, nx);
        double *ar = malloc(sizeof(double) * ns * ns * 2);
        struct product_rows ctx = {tr, ti, vr, vi, nk, nx, ne};
        gram_rows(fill_product, &ctx, 1, ns, nk, ne, nx, lam, delta, shifts, col, ar, ar + ns * ns);
        join_complex(ar, ar + ns * ns, ns * ns, a);
        free(ar);
        free(vr);
        free(vi);
        free(vr0);
        free(vi0);
        free(tr);
        free(ti);
}

void zmr_contract_t_r(const double *t, int64_t ns, int64_t nk, int64_t nx,
                      const double *v, int64_t ne,
                      const double *lam, const double *delta, const double *shifts,
                      int64_t col, double *a)
{
        double *vr = transposed(v, ne, nx);
        struct product_rows ctx = {t, NULL, vr, NULL, nk, nx, ne};
        gram_rows(fill_product, &ctx, 0, ns, nk, ne, nx, lam, delta, shifts, col, a, NULL);
        free(vr);
}

/* ---- the same sum with the external labels contracted first:
 *   a[I, J] = - sum_k sum_XY conj(t[I, k, X]) W_k(s_J)[X, Y] t[J, k, Y],
 *   W_k(s)[X, Y] = sum_e conj(v[e, X]) v[e, Y] / (lam[k] + delta[e] - s).
 * The cost of W does not depend on the number of states. */
static void gram_cols(int cplx, const double *tr, const double *ti, int64_t ns, int64_t nk,
                      int64_t nx, const double *vr, const double *vi, int64_t ne,
                      const double *lam, const double *delta, const double *shifts,
                      int64_t col, double *ar, double *ai)
{
        int64_t nn = ns * ns;
        memset(ar, 0, sizeof(double) * nn);
        if (cplx) {
                memset(ai, 0, sizeof(double) * nn);
        }
        if (nk == 0 || ne == 0 || ns == 0 || nx == 0) {
                return;
        }
        int64_t j0 = col < 0 ? 0 : col;
        int64_t j1 = col < 0 ? ns : col + 1;
        double work = (double)nk * ((double)ne * nx * (nx + 1) / 2 + (double)(j1 - j0) * nx * (nx + ns));
#pragma omp parallel num_threads(threads_for(work))
        {
                double *locr = calloc(nn, sizeof(double));
                double *loci = calloc(cplx ? nn : 1, sizeof(double));
                double *wr = malloc(sizeof(double) * nx * nx);
                double *wi = malloc(sizeof(double) * (cplx ? nx * nx : 1));
                double *sr = malloc(sizeof(double) * ne);
                double *si = malloc(sizeof(double) * ne);
                double *inv = malloc(sizeof(double) * ne);
                double *yr = malloc(sizeof(double) * nx);
                double *yi = malloc(sizeof(double) * nx);
                double *buf = malloc(sizeof(double) * (cplx ? 2 : 1) * ne * nx);
                double *wz = malloc(sizeof(double) * 2 * nx * nx);
#pragma omp for schedule(dynamic, 1)
                for (int64_t k = 0; k < nk; k++) {
                        int have = 0;
                        double last = 0;
                        for (int64_t j = j0; j < j1; j++) {
                                size_t offj = (size_t)(j * nk + k) * nx;
                                if (!have || shifts[j] != last) {
                                        inverse_denominators(ne, lam[k] - shifts[j], delta, inv);
                                        last = shifts[j];
                                        have = 1;
                                        /* W[X, Y] = sum_e conj(v[e, X]) v[e, Y] inv[e], Hermitian */
                                        int positive = 1;
                                        for (int64_t e = 0; e < ne; e++) {
                                                positive &= inv[e] > 0;
                                        }
                                        if (positive) {
                                                /* rank-k update of v scaled by sqrt(inv) */
                                                for (int64_t e = 0; e < ne; e++) {
                                                        sr[e] = sqrt(inv[e]);
                                                }
                                                if (cplx) {
                                                        for (int64_t x = 0; x < nx; x++) {
#pragma omp simd
                                                                for (int64_t e = 0; e < ne; e++) {
                                                                        buf[2 * (e * nx + x)] = vr[x * ne + e] * sr[e];
                                                                        buf[2 * (e * nx + x) + 1] = vi[x * ne + e] * sr[e];
                                                                }
                                                        }
                                                        cblas_zherk(101, 121, 113, (int)nx, (int)ne, 1.0, buf,
                                                                    (int)nx, 0.0, wz, (int)nx);
                                                        for (int64_t x = 0; x < nx; x++) {
                                                                for (int64_t y = x; y < nx; y++) {
                                                                        double a = wz[2 * (x * nx + y)];
                                                                        double b = wz[2 * (x * nx + y) + 1];
                                                                        wr[x * nx + y] = a;
                                                                        wi[x * nx + y] = b;
                                                                        wr[y * nx + x] = a;
                                                                        wi[y * nx + x] = -b;
                                                                }
                                                                wi[x * nx + x] = 0;
                                                        }
                                                } else {
                                                        for (int64_t x = 0; x < nx; x++) {
#pragma omp simd
                                                                for (int64_t e = 0; e < ne; e++) {
                                                                        buf[e * nx + x] = vr[x * ne + e] * sr[e];
                                                                }
                                                        }
                                                        cblas_dsyrk(101, 121, 112, (int)nx, (int)ne, 1.0, buf,
                                                                    (int)nx, 0.0, wz, (int)nx);
                                                        for (int64_t x = 0; x < nx; x++) {
                                                                for (int64_t y = x; y < nx; y++) {
                                                                        wr[x * nx + y] = wz[x * nx + y];
                                                                        wr[y * nx + x] = wz[x * nx + y];
                                                                }
                                                        }
                                                }
                                        } else {
                                        for (int64_t y = 0; y < nx; y++) {
                                                scale(ne, vr + y * ne, inv, sr);
                                                if (cplx) {
                                                        scale(ne, vi + y * ne, inv, si);
                                                }
                                                for (int64_t x = 0; x <= y; x++) {
                                                        if (cplx) {
                                                                double a, b;
                                                                conj_dot(ne, vr + x * ne, vi + x * ne,
                                                                         sr, si, &a, &b);
                                                                wr[x * nx + y] = a;
                                                                wi[x * nx + y] = b;
                                                                wr[y * nx + x] = a;
                                                                wi[y * nx + x] = -b;
                                                        } else {
                                                                double a = real_dot(ne, vr + x * ne, sr);
                                                                wr[x * nx + y] = a;
                                                                wr[y * nx + x] = a;
                                                        }
                                                }
                                        }
                                        }
                                }
                                /* y = W t_J, then a[I, J] -= conj(t_I) . y */
                                for (int64_t x = 0; x < nx; x++) {
                                        double a = 0, b = 0;
                                        for (int64_t y = 0; y < nx; y++) {
                                                double tjr = tr[offj + y];
                                                a += wr[x * nx + y] * tjr;
                                                if (cplx) {
                                                        double tji = ti[offj + y];
                                                        a -= wi[x * nx + y] * tji;
                                                        b += wr[x * nx + y] * tji + wi[x * nx + y] * tjr;
                                                }
                                        }
                                        yr[x] = a;
                                        yi[x] = b;
                                }
                                for (int64_t i = 0; i < ns; i++) {
                                        size_t offi = (size_t)(i * nk + k) * nx;
                                        if (cplx) {
                                                double a, b;
                                                conj_dot(nx, tr + offi, ti + offi, yr, yi, &a, &b);
                                                locr[i * ns + j] -= a;
                                                loci[i * ns + j] -= b;
                                        } else {
                                                locr[i * ns + j] -= real_dot(nx, tr + offi, yr);
                                        }
                                }
                        }
                }
#pragma omp critical
                {
                        for (int64_t x = 0; x < nn; x++) {
                                ar[x] += locr[x];
                        }
                        if (cplx) {
                                for (int64_t x = 0; x < nn; x++) {
                                        ai[x] += loci[x];
                                }
                        }
                }
                free(locr);
                free(loci);
                free(wr);
                free(wi);
                free(sr);
                free(si);
                free(inv);
                free(yr);
                free(yi);
                free(buf);
                free(wz);
        }
}

void zmr_contract_w(const double complex *t, int64_t ns, int64_t nk, int64_t nx,
                    const double complex *v, int64_t ne,
                    const double *lam, const double *delta, const double *shifts,
                    int64_t col, double complex *a)
{
        double *tr, *ti, *vr0, *vi0;
        split_complex(t, ns * nk * nx, &tr, &ti);
        split_complex(v, ne * nx, &vr0, &vi0);
        double *vr = transposed(vr0, ne, nx);
        double *vi = transposed(vi0, ne, nx);
        double *ar = malloc(sizeof(double) * ns * ns * 2);
        gram_cols(1, tr, ti, ns, nk, nx, vr, vi, ne, lam, delta, shifts, col, ar, ar + ns * ns);
        join_complex(ar, ar + ns * ns, ns * ns, a);
        free(ar);
        free(vr);
        free(vi);
        free(vr0);
        free(vi0);
        free(tr);
        free(ti);
}

void zmr_contract_w_r(const double *t, int64_t ns, int64_t nk, int64_t nx,
                      const double *v, int64_t ne,
                      const double *lam, const double *delta, const double *shifts,
                      int64_t col, double *a)
{
        double *vr = transposed(v, ne, nx);
        gram_cols(0, t, NULL, ns, nk, nx, vr, NULL, ne, lam, delta, shifts, col, a, NULL);
        free(vr);
}

/* binom[n * (norb + 1) + k] = C(n, k) */
static int64_t *binomial_table(int norb)
{
        int64_t *binom = calloc((size_t)(norb + 1) * (norb + 1), sizeof(int64_t));
        for (int n = 0; n <= norb; n++) {
                binom[n * (norb + 1)] = 1;
                for (int k = 1; k <= n; k++) {
                        binom[n * (norb + 1) + k] = binom[(n - 1) * (norb + 1) + k - 1]
                                + (k <= n - 1 ? binom[(n - 1) * (norb + 1) + k] : 0);
                }
        }
        return binom;
}

/* the dim strings with nelec set bits in ascending integer order */
static uint64_t *ascending_strings(int nelec, int64_t dim)
{
        uint64_t *strs = malloc(sizeof(uint64_t) * (dim > 0 ? dim : 1));
        uint64_t s = (nelec == 64) ? ~0ULL : ((1ULL << nelec) - 1);
        for (int64_t i = 0; i < dim; i++) {
                strs[i] = s;
                if (s == 0) {
                        break;
                }
                /* next string with the same popcount (Gosper) */
                uint64_t c = s & -s;
                uint64_t r = s + c;
                s = (((r ^ s) >> 2) / c) | r;
        }
        return strs;
}

/* colexicographic rank = position in ascending integer order */
static int64_t string_rank(uint64_t s, const int64_t *binom, int norb)
{
        int64_t r = 0;
        int k = 0;
        while (s) {
                int p = __builtin_ctzll(s);
                k++;
                r += binom[p * (norb + 1) + k];
                s &= s - 1;
        }
        return r;
}

static inline int parity_above(uint64_t s, int p)
{
        return __builtin_popcountll(s >> (p + 1)) & 1;
}

void zmr_apply_strings(int norb, int nelec, int nop, const int *dagger, const int64_t *ops,
                       int64_t nx, int64_t nx_total, int64_t x0, int64_t nvec,
                       const double complex *vecs, double complex *out)
{
        int ndst_elec = nelec;
        for (int k = 0; k < nop; k++) {
                ndst_elec += dagger[k] ? 1 : -1;
        }
        if (nelec < 0 || nelec > norb || ndst_elec < 0 || ndst_elec > norb) {
                return;
        }
        int64_t *binom = binomial_table(norb);
        int64_t dsrc = binom[norb * (norb + 1) + nelec];
        int64_t ddst = binom[norb * (norb + 1) + ndst_elec];
        uint64_t *strs = ascending_strings(nelec, dsrc);
#pragma omp parallel num_threads(threads_for((double)nx * dsrc * (4 * nop + 2 * nvec + 4)))
        {
                /* map of one operator string: +-(address + 1), 0 if annihilated */
                int64_t *map = malloc(sizeof(int64_t) * dsrc);
#pragma omp for schedule(static)
                for (int64_t x = 0; x < nx; x++) {
                        const int64_t *op = ops + x * nop;
                        for (int64_t i = 0; i < dsrc; i++) {
                                uint64_t s = strs[i];
                                int par = 0;
                                int ok = 1;
                                for (int k = nop - 1; k >= 0; k--) {
                                        int p = (int)op[k];
                                        int occ = (int)(s >> p & 1);
                                        if (occ == (dagger[k] != 0)) {
                                                ok = 0;
                                                break;
                                        }
                                        par ^= parity_above(s, p);
                                        s ^= 1ULL << p;
                                }
                                if (ok) {
                                        int64_t addr = string_rank(s, binom, norb) + 1;
                                        map[i] = par ? -addr : addr;
                                } else {
                                        map[i] = 0;
                                }
                        }
                        for (int64_t v = 0; v < nvec; v++) {
                                double complex *o = out + (v * nx_total + x0 + x) * ddst;
                                const double complex *c = vecs + v * dsrc;
                                memset(o, 0, sizeof(double complex) * ddst);
                                for (int64_t i = 0; i < dsrc; i++) {
                                        int64_t m = map[i];
                                        if (m > 0) {
                                                o[m - 1] = c[i];
                                        } else if (m < 0) {
                                                o[-m - 1] = -c[i];
                                        }
                                }
                        }
                }
                free(map);
        }
        free(strs);
        free(binom);
}

void zmr_build_hmat(int norb, int nelec, int64_t dim, const double complex *h1,
                    const double complex *eri, double complex *hmat)
{
        memset(hmat, 0, sizeof(double complex) * dim * dim);
        if (dim == 0 || nelec == 0) {
                return;
        }
        int64_t *binom = binomial_table(norb);
        uint64_t *strs = ascending_strings(nelec, dim);
        size_t n1 = norb, n2 = (size_t)norb * norb, n3 = n2 * norb;
#pragma omp parallel for schedule(dynamic, 8) \
        num_threads(threads_for((double)dim * n2 * n2 / 8))
        for (int64_t jj = 0; jj < dim; jj++) {
                uint64_t sj = strs[jj];
                /* one-body: p^+ q */
                for (int q = 0; q < norb; q++) {
                        if (!(sj >> q & 1)) continue;
                        int sq = parity_above(sj, q);
                        uint64_t s1 = sj ^ (1ULL << q);
                        for (int p = 0; p < norb; p++) {
                                if (s1 >> p & 1) continue;
                                int sp = parity_above(s1, p);
                                uint64_t si = s1 | (1ULL << p);
                                int64_t ii = string_rank(si, binom, norb);
                                double sgn = ((sq + sp) & 1) ? -1.0 : 1.0;
                                hmat[ii * dim + jj] += sgn * h1[p * n1 + q];
                        }
                }
                /* two-body: 1/2 (pq|rs) p^+ r^+ s q, applied right to left */
                for (int q = 0; q < norb; q++) {
                        if (!(sj >> q & 1)) continue;
                        int sq = parity_above(sj, q);
                        uint64_t s1 = sj ^ (1ULL << q);
                        for (int s = 0; s < norb; s++) {
                                if (!(s1 >> s & 1)) continue;
                                int ss = parity_above(s1, s);
                                uint64_t s2 = s1 ^ (1ULL << s);
                                for (int r = 0; r < norb; r++) {
                                        if (s2 >> r & 1) continue;
                                        int sr = parity_above(s2, r);
                                        uint64_t s3 = s2 | (1ULL << r);
                                        for (int p = 0; p < norb; p++) {
                                                if (s3 >> p & 1) continue;
                                                int sp = parity_above(s3, p);
                                                uint64_t si = s3 | (1ULL << p);
                                                int64_t ii = string_rank(si, binom, norb);
                                                double sgn = ((sq + ss + sr + sp) & 1) ? -0.5 : 0.5;
                                                hmat[ii * dim + jj] += sgn * eri[p * n3 + q * n2 + r * n1 + s];
                                        }
                                }
                        }
                }
        }
        free(strs);
        free(binom);
}

/* ----------------------------------------------------------------------
 * Spin-restricted active space: determinants |I_alpha I_beta> with n_alpha,
 * n_beta electrons in norb spatial orbitals.  A determinant is the string
 * (alpha string << norb) | beta string, i.e. the spin-orbital bit of orbital
 * p is p for beta and norb + p for alpha; its address is
 * rank(alpha) * dim_beta + rank(beta), the order of pyscf.fci, and the phases
 * are those of pyscf.fci as well. */

/* Determinant-driven sigma (the kernel before the string-driven zmr_sigma
 * below, kept as its reference): out = H q for nvec vectors stored as
 * q[det * nvec + v], without the matrix.  Row i of H is the conjugate of
 * column i (H must be Hermitian), whose elements the loops of zmr_build_hmat
 * enumerate, so every thread writes its own rows. */
void zmr_sigma_det(int norb, int nelec, int64_t dim, const double complex *h1,
               const double complex *eri, int64_t nvec, const double complex *q,
               double complex *out)
{
        memset(out, 0, sizeof(double complex) * dim * nvec);
        if (dim == 0 || nelec == 0 || nvec == 0) {
                return;
        }
        int64_t *binom = binomial_table(norb);
        uint64_t *strs = ascending_strings(nelec, dim);
        size_t n1 = norb, n2 = (size_t)norb * norb, n3 = n2 * norb;
#pragma omp parallel for schedule(dynamic, 16) \
        num_threads(threads_for((double)dim * n2 * n2 / 8 * nvec))
        for (int64_t jj = 0; jj < dim; jj++) {
                uint64_t sj = strs[jj];
                double complex *o = out + jj * nvec;
                for (int q1 = 0; q1 < norb; q1++) {
                        if (!(sj >> q1 & 1)) continue;
                        int sq = parity_above(sj, q1);
                        uint64_t s1 = sj ^ (1ULL << q1);
                        for (int p = 0; p < norb; p++) {
                                if (s1 >> p & 1) continue;
                                int sp = parity_above(s1, p);
                                int64_t ii = string_rank(s1 | (1ULL << p), binom, norb);
                                double sgn = ((sq + sp) & 1) ? -1.0 : 1.0;
                                double complex h = conj(sgn * h1[p * n1 + q1]);
                                const double complex *x = q + ii * nvec;
                                for (int64_t v = 0; v < nvec; v++) o[v] += h * x[v];
                        }
                        for (int s = 0; s < norb; s++) {
                                if (!(s1 >> s & 1)) continue;
                                int ss = parity_above(s1, s);
                                uint64_t s2 = s1 ^ (1ULL << s);
                                for (int r = 0; r < norb; r++) {
                                        if (s2 >> r & 1) continue;
                                        int sr = parity_above(s2, r);
                                        uint64_t s3 = s2 | (1ULL << r);
                                        for (int p = 0; p < norb; p++) {
                                                if (s3 >> p & 1) continue;
                                                int sp = parity_above(s3, p);
                                                int64_t ii = string_rank(s3 | (1ULL << p), binom, norb);
                                                double sgn = ((sq + ss + sr + sp) & 1) ? -0.5 : 0.5;
                                                double complex h = conj(sgn * eri[p * n3 + q1 * n2 + r * n1 + s]);
                                                const double complex *x = q + ii * nvec;
                                                for (int64_t v = 0; v < nvec; v++) o[v] += h * x[v];
                                        }
                                }
                        }
                }
        }
        free(strs);
        free(binom);
}

/* ----------------------------------------------------------------------
 * String-driven sigma for spinors: resolution through the strings K with
 * two electrons fewer.  In a sector of N >= 2 electrons
 * p^+ q = sum_r p^+ r^+ r q / (N - 1), so with
 * g(pq|rs) = (pq|rs) + 2 h_pq delta_rs / (N - 1)
 *
 *     H = 1/2 sum_pqrs g(pq|rs) p^+ r^+ s q
 *       = sum_{p<r} sum_{q<s} W[pr, qs] p^+ r^+ s q,
 *     W[pr, qs] = [g(pq|rs) + g(rs|pq) - g(ps|rq) - g(rq|ps)] / 2,
 *
 * which holds for any h and (pq|rs): no symmetry of the integrals is used.
 * For every string K of N - 2 electrons and the pairs q < s of its empty
 * orbitals
 *
 *     D[qs] = <K| s q |J> c_J,  J = K + q + s          gather
 *     G[pr] = sum_qs W[pr, qs] D[qs]                   one zgemm
 *     sigma_I += <I| p^+ r^+ |K> G[pr],  I = K + p + r scatter
 *
 * The gather and the scatter use the same addresses and signs; the block
 * of W of the pairs empty in K is collected for every K, which a batch of
 * vectors (the columns of D and G) pays for.  The number of
 * multiplications per determinant, C(N, 2) C(n - N + 2, 2), is about that
 * of its connected determinants.
 *
 * The vectors are taken in batches of ZSIGMA_NV columns and the threads
 * share the strings K of a batch, so that the block of W is collected once
 * for all columns whatever the number of threads.  Different K scatter into
 * the same rows of sigma: a row is updated under the spin lock of its stripe
 * (two K of different threads rarely meet, and no copies of sigma are
 * needed).  The zgemm is called inside the parallel region and is
 * sequential there. */
#define ZSIGMA_NV 128
#define ZSIGMA_NLOCK 4096
/* below this many multiplications a plain loop replaces the zgemm */
#define ZSIGMA_SMALL 256

typedef struct {
        int norb, nelec;
        int npair;              /* pairs of the empty orbitals of one K */
        int64_t npa;            /* pairs of all orbitals */
        const int64_t *binom;
        const double complex *w;        /* (npa, npa) */
} zsigma_ctx;

/* one spin lock per stripe of rows of sigma, a cache line each */
typedef struct {
        volatile char flag;
        char pad[63];
} zsigma_lock;

static inline void zsigma_acquire(zsigma_lock *l)
{
        while (__atomic_test_and_set(&l->flag, __ATOMIC_ACQUIRE)) {
                while (l->flag) {
#if defined(__x86_64__)
                        __builtin_ia32_pause();
#endif
                }
        }
}

typedef struct {
        int64_t *addr;          /* (npair) address of K + q + s */
        double *sign;
        int *gp;                /* index of the pair among all pairs */
        double complex *wk, *d, *g;
} zsigma_work;

static void zsigma_alloc(zsigma_work *w, int npair, int nv)
{
        w->addr = malloc(sizeof(int64_t) * npair);
        w->sign = malloc(sizeof(double) * npair);
        w->gp = malloc(sizeof(int) * npair);
        w->wk = malloc(sizeof(double complex) * npair * npair);
        w->d = malloc(sizeof(double complex) * npair * nv);
        w->g = malloc(sizeof(double complex) * npair * nv);
}

static void zsigma_free(zsigma_work *w)
{
        free(w->addr);
        free(w->sign);
        free(w->gp);
        free(w->wk);
        free(w->d);
        free(w->g);
}

/* the contribution of the string k to sigma for nv vectors c[det * ldc + v],
 * added to s[det * lds + v] */
static void zsigma_string(const zsigma_ctx *x, uint64_t k, int nv,
                          const double complex *c, int64_t ldc,
                          double complex *s, int64_t lds, zsigma_work *w,
                          zsigma_lock *locks)
{
        int n = x->norb, n1 = n + 1, ne2 = x->nelec - 2;
        int np = x->npair;
        const int64_t *binom = x->binom;
        /* rank of K + a + b (a < b): the bits of K below a keep their
         * position among the set bits, those between a and b move up by
         * one, those above b by two; lo[a] + hi[b] with partial sums */
        int64_t s0 = 0, s1 = 0, s2 = 0;
        int64_t lo[64], hi[64];
        int emp[64], cnt[64];
        int j = 0, m = 0;
        for (int p = 0; p < n; p++) {
                if (k >> p & 1) {
                        j++;
                        s0 += binom[p * n1 + j];
                        s1 += binom[p * n1 + j + 1];
                        s2 += binom[p * n1 + j + 2];
                } else {
                        emp[m] = p;
                        cnt[m] = j;
                        lo[m] = s0 + binom[p * n1 + j + 1] - s1;
                        hi[m] = s1 + binom[p * n1 + j + 2] - s2;
                        m++;
                }
        }
        int t = 0;
        for (int ib = 1; ib < m; ib++) {
                int b = emp[ib];
                for (int ia = 0; ia < ib; ia++, t++) {
                        w->addr[t] = lo[ia] + hi[ib] + s2;
                        /* s q |J>: (bits of J above q) + (bits of J - q above s) */
                        w->sign[t] = ((cnt[ia] + cnt[ib]) & 1) ? 1.0 : -1.0;
                        w->gp[t] = b * (b - 1) / 2 + emp[ia];
                }
        }
        double complex *d = w->d, *g = w->g, *wk = w->wk;
        for (t = 0; t < np; t++) {
                const double complex *src = c + w->addr[t] * ldc;
                double complex *dst = d + (int64_t)t * nv;
                double f = w->sign[t];
                for (int v = 0; v < nv; v++) {
                        dst[v] = f * src[v];
                }
        }
        if ((double)np * np * nv < ZSIGMA_SMALL) {
                for (t = 0; t < np; t++) {
                        const double complex *row = x->w + w->gp[t] * x->npa;
                        double complex *gt = g + (int64_t)t * nv;
                        for (int v = 0; v < nv; v++) {
                                gt[v] = 0;
                        }
                        for (int u = 0; u < np; u++) {
                                double complex f = row[w->gp[u]];
                                const double complex *du = d + (int64_t)u * nv;
                                for (int v = 0; v < nv; v++) {
                                        gt[v] += f * du[v];
                                }
                        }
                }
        } else {
                for (t = 0; t < np; t++) {
                        const double complex *row = x->w + w->gp[t] * x->npa;
                        double complex *wt = wk + (int64_t)t * np;
                        for (int u = 0; u < np; u++) {
                                wt[u] = row[w->gp[u]];
                        }
                }
                double complex one = 1, zero = 0;
                cblas_zgemm(101, 111, 111, np, nv, np, &one, wk, np, d, nv, &zero, g, nv);
        }
        for (t = 0; t < np; t++) {
                const double complex *src = g + (int64_t)t * nv;
                double complex *dst = s + w->addr[t] * lds;
                double f = w->sign[t];
                zsigma_lock *l = locks ? locks + (w->addr[t] & (ZSIGMA_NLOCK - 1)) : NULL;
                if (l) {
                        zsigma_acquire(l);
                }
                for (int v = 0; v < nv; v++) {
                        dst[v] += f * src[v];
                }
                if (l) {
                        __atomic_clear(&l->flag, __ATOMIC_RELEASE);
                }
        }
}

void zmr_sigma(int norb, int nelec, int64_t dim, const double complex *h1,
               const double complex *eri, int64_t nvec, const double complex *q,
               double complex *out)
{
        if (dim <= 0 || nvec <= 0 || nelec < 0 || nelec > norb) {
                return;
        }
        if (nelec == 0) {
                memset(out, 0, sizeof(double complex) * dim * nvec);
                return;
        }
        if (nelec == 1) {
                /* the strings are the orbitals: H = h */
                double complex one = 1, zero = 0;
                cblas_zgemm(101, 111, 111, norb, (int)nvec, norb, &one, h1, norb, q, (int)nvec,
                            &zero, out, (int)nvec);
                return;
        }
        size_t n1 = norb, n2 = n1 * n1, n3 = n2 * n1;
        int64_t *binom = binomial_table(norb);
        int64_t npa = (int64_t)norb * (norb - 1) / 2;
        double complex *w = malloc(sizeof(double complex) * npa * npa);
        double fac = 2.0 / (nelec - 1);
#define G2(p, q, r, s) (eri[(p) * n3 + (q) * n2 + (r) * n1 + (s)] \
                        + ((r) == (s) ? fac * h1[(p) * n1 + (q)] : 0.0))
#pragma omp parallel for schedule(static) num_threads(threads_for((double)npa * npa * 8))
        for (int r = 1; r < norb; r++) {
                for (int p = 0; p < r; p++) {
                        double complex *row = w + ((int64_t)r * (r - 1) / 2 + p) * npa;
                        for (int s = 1; s < norb; s++) {
                                for (int q1 = 0; q1 < s; q1++) {
                                        row[s * (s - 1) / 2 + q1] = 0.5 * (G2(p, q1, r, s) + G2(r, s, p, q1)
                                                                           - G2(p, s, r, q1) - G2(r, q1, p, s));
                                }
                        }
                }
        }
#undef G2
        int m = norb - nelec + 2;
        zsigma_ctx x;
        x.norb = norb;
        x.nelec = nelec;
        x.npair = m * (m - 1) / 2;
        x.npa = npa;
        x.binom = binom;
        x.w = w;
        int np = x.npair;
        int64_t dimk = binom[norb * (norb + 1) + nelec - 2];
        uint64_t *kstr = ascending_strings(nelec - 2, dimk);
        double work = (double)dimk * np * ((double)np * (nvec + 1) + 4.0 * nvec + 8.0);
        int nt = threads_for(work);
        zsigma_lock *locks = nt > 1 ? calloc(ZSIGMA_NLOCK, sizeof(zsigma_lock)) : NULL;
        memset(out, 0, sizeof(double complex) * dim * nvec);
        for (int64_t v0 = 0; v0 < nvec; v0 += ZSIGMA_NV) {
                int nv = (int)(nvec - v0 < ZSIGMA_NV ? nvec - v0 : ZSIGMA_NV);
#pragma omp parallel num_threads(nt)
                {
                        zsigma_work wk;
                        zsigma_alloc(&wk, np, nv);
#pragma omp for schedule(dynamic, 8)
                        for (int64_t ik = 0; ik < dimk; ik++) {
                                zsigma_string(&x, kstr[ik], nv, q + v0, nvec, out + v0, nvec, &wk, locks);
                        }
                        zsigma_free(&wk);
                }
        }
        free(locks);
        free(kstr);
        free(w);
        free(binom);
}

static inline int64_t rank_ab(uint64_t s, int norb, int64_t dimb, const int64_t *binom)
{
        uint64_t mask = (1ULL << norb) - 1;
        return string_rank(s >> norb, binom, norb) * dimb + string_rank(s & mask, binom, norb);
}

void zmr_build_hmat_ab(int norb, int na, int nb, const double *h1, const double *eri,
                       double *hmat)
{
        int64_t *binom = binomial_table(norb);
        int64_t dima = binom[norb * (norb + 1) + na];
        int64_t dimb = binom[norb * (norb + 1) + nb];
        int64_t dim = dima * dimb;
        memset(hmat, 0, sizeof(double) * dim * dim);
        uint64_t *stra = ascending_strings(na, dima);
        uint64_t *strb = ascending_strings(nb, dimb);
        size_t n1 = norb, n2 = (size_t)norb * norb, n3 = n2 * norb;
        int nso = 2 * norb;
#pragma omp parallel for schedule(dynamic, 8) \
        num_threads(threads_for((double)dim * n2 * n2 / 2))
        for (int64_t jj = 0; jj < dim; jj++) {
                uint64_t sj = (stra[jj / dimb] << norb) | strb[jj % dimb];
                /* one-body: sum_sigma h_pq p^+_sigma q_sigma */
                for (int q = 0; q < nso; q++) {
                        if (!(sj >> q & 1)) continue;
                        int sq = parity_above(sj, q);
                        uint64_t s1 = sj ^ (1ULL << q);
                        int p0 = q < norb ? 0 : norb;
                        for (int p = p0; p < p0 + norb; p++) {
                                if (s1 >> p & 1) continue;
                                int sp = parity_above(s1, p);
                                uint64_t si = s1 | (1ULL << p);
                                int64_t ii = rank_ab(si, norb, dimb, binom);
                                double sgn = ((sq + sp) & 1) ? -1.0 : 1.0;
                                hmat[ii * dim + jj] += sgn * h1[(p - p0) * n1 + (q - p0)];
                        }
                }
                /* two-body: 1/2 (pq|rs) p^+_sigma r^+_tau s_tau q_sigma */
                for (int q = 0; q < nso; q++) {
                        if (!(sj >> q & 1)) continue;
                        int sq = parity_above(sj, q);
                        uint64_t s1 = sj ^ (1ULL << q);
                        int p0 = q < norb ? 0 : norb;
                        for (int s = 0; s < nso; s++) {
                                if (!(s1 >> s & 1)) continue;
                                int ss = parity_above(s1, s);
                                uint64_t s2 = s1 ^ (1ULL << s);
                                int r0 = s < norb ? 0 : norb;
                                for (int r = r0; r < r0 + norb; r++) {
                                        if (s2 >> r & 1) continue;
                                        int sr = parity_above(s2, r);
                                        uint64_t s3 = s2 | (1ULL << r);
                                        for (int p = p0; p < p0 + norb; p++) {
                                                if (s3 >> p & 1) continue;
                                                int sp = parity_above(s3, p);
                                                uint64_t si = s3 | (1ULL << p);
                                                int64_t ii = rank_ab(si, norb, dimb, binom);
                                                double sgn = ((sq + ss + sr + sp) & 1) ? -0.5 : 0.5;
                                                hmat[ii * dim + jj] += sgn * eri[(p - p0) * n3 + (q - p0) * n2
                                                                               + (r - r0) * n1 + (s - r0)];
                                        }
                                }
                        }
                }
        }
        free(stra);
        free(strb);
        free(binom);
}

static inline __attribute__((always_inline))
void axpy(int64_t n, double f, const double *x, double *y)
{
#pragma omp simd
        for (int64_t v = 0; v < n; v++) {
                y[v] += f * x[v];
        }
}

/* Determinant-driven sigma (the kernel before the string-driven one below,
 * kept as its reference): Slater-Condon rules on the combined string, every
 * connected determinant is visited once.  It assumes (pq|rs) = (rs|pq) and a
 * symmetric matrix.  The vectors are processed in blocks of columns that fit
 * into the cache, because the rows of q are accessed at random. */
#define SIGMA_BLOCK 64

void zmr_sigma_ab_det(int norb, int na, int nb, const double *h1, const double *eri,
                  int64_t nvec, const double *q, double *out)
{
        int64_t *binom = binomial_table(norb);
        int64_t dima = binom[norb * (norb + 1) + na];
        int64_t dimb = binom[norb * (norb + 1) + nb];
        int64_t dim = dima * dimb;
        uint64_t *stra = ascending_strings(na, dima);
        uint64_t *strb = ascending_strings(nb, dimb);
        size_t n1 = norb, n2 = (size_t)norb * norb, n3 = n2 * norb;
        int nso = 2 * norb;
#define SPATIAL(x) ((x) < norb ? (x) : (x) - norb)
#define SAME_SPIN(x, y) (((x) < norb) == ((y) < norb))
#define ERI(p, q, r, s) eri[SPATIAL(p) * n3 + SPATIAL(q) * n2 + SPATIAL(r) * n1 + SPATIAL(s)]
        double work = (double)dim * n2 * n2 / 8 * (1 + nvec / 4.0);
        for (int64_t v0 = 0; v0 < nvec; v0 += SIGMA_BLOCK) {
                int64_t nv = nvec - v0 < SIGMA_BLOCK ? nvec - v0 : SIGMA_BLOCK;
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads_for(work))
                for (int64_t ii = 0; ii < dim; ii++) {
                        uint64_t sj = (stra[ii / dimb] << norb) | strb[ii % dimb];
                        double *o = out + ii * nvec + v0;
                        int occ[64], emp[64];
                        int nocc = 0, nemp = 0;
                        for (int x = 0; x < nso; x++) {
                                if (sj >> x & 1) occ[nocc++] = x; else emp[nemp++] = x;
                        }
                        /* diagonal */
                        double diag = 0;
                        for (int a = 0; a < nocc; a++) {
                                int i = occ[a];
                                diag += h1[SPATIAL(i) * n1 + SPATIAL(i)];
                                for (int b = 0; b < a; b++) {
                                        int j = occ[b];
                                        diag += ERI(i, i, j, j);
                                        if (SAME_SPIN(i, j)) diag -= ERI(i, j, j, i);
                                }
                        }
                        const double *c = q + ii * nvec + v0;
#pragma omp simd
                        for (int64_t v = 0; v < nv; v++) {
                                o[v] = diag * c[v];
                        }
                        /* single excitations q1 -> p */
                        for (int a = 0; a < nocc; a++) {
                                int q1 = occ[a];
                                int sq = parity_above(sj, q1);
                                uint64_t s1 = sj ^ (1ULL << q1);
                                for (int e = 0; e < nemp; e++) {
                                        int p = emp[e];
                                        if (!SAME_SPIN(p, q1)) continue;
                                        double f = h1[SPATIAL(p) * n1 + SPATIAL(q1)];
                                        for (int b = 0; b < nocc; b++) {
                                                int j = occ[b];
                                                if (j == q1) continue;
                                                f += ERI(p, q1, j, j);
                                                if (SAME_SPIN(j, p)) f -= ERI(p, j, j, q1);
                                        }
                                        if (f == 0) continue;
                                        if ((sq + parity_above(s1, p)) & 1) f = -f;
                                        uint64_t si = s1 | (1ULL << p);
                                        axpy(nv, f, q + rank_ab(si, norb, dimb, binom) * nvec + v0, o);
                                }
                        }
                        /* double excitations (q1 < s) -> (p < r):
                         * [(pq|rs) - (ps|rq)] p^+ r^+ s q */
                        for (int a = 0; a < nocc; a++) {
                                int q1 = occ[a];
                                int sq = parity_above(sj, q1);
                                uint64_t s1 = sj ^ (1ULL << q1);
                                for (int b = a + 1; b < nocc; b++) {
                                        int s = occ[b];
                                        int ss = sq + parity_above(s1, s);
                                        uint64_t s2 = s1 ^ (1ULL << s);
                                        for (int e = 0; e < nemp; e++) {
                                                int p = emp[e];
                                                for (int g = e + 1; g < nemp; g++) {
                                                        int r = emp[g];
                                                        double f = 0;
                                                        if (SAME_SPIN(p, q1) && SAME_SPIN(r, s)) f += ERI(p, q1, r, s);
                                                        if (SAME_SPIN(p, s) && SAME_SPIN(r, q1)) f -= ERI(p, s, r, q1);
                                                        if (f == 0) continue;
                                                        uint64_t s3 = s2 | (1ULL << r);
                                                        if ((ss + parity_above(s2, r) + parity_above(s3, p)) & 1) f = -f;
                                                        uint64_t si = s3 | (1ULL << p);
                                                        axpy(nv, f, q + rank_ab(si, norb, dimb, binom) * nvec + v0, o);
                                                }
                                        }
                                }
                        }
                }
        }
#undef SPATIAL
#undef SAME_SPIN
#undef ERI
        free(stra);
        free(strb);
        free(binom);
}

/* ----------------------------------------------------------------------
 * String-driven sigma (Knowles and Handy, Olsen et al.).  With the
 * spin-summed excitations E_pq = sum_sigma p^+_sigma q_sigma,
 *
 *     H = sum_pq k_pq E_pq + 1/2 sum_pqrs (pq|rs) E_pq E_rs,
 *     k_pq = h_pq - 1/2 sum_r (pr|rq),
 *
 * which holds for any h and (pq|rs).  For a block of determinants J
 *
 *     D[rs, J] = (E_rs c)_J                      gather through the links
 *     G[pq, J] = 1/2 sum_rs (pq|rs) D[rs, J] + k_pq c_J      one dgemm
 *     sigma_I += sum_pq <I| E_pq |J> G[pq, J]    scatter through the links
 *
 * (k is a column of the matrix of the product, c a row of D).  E_pq acts on
 * the alpha or on the beta string, so the links are those of the single
 * strings: for every string J the pairs (p, q) with E_pq |J> = sign |I>.
 *
 * If (pq|rs) = (qp|rs) = (pq|sr) and k is symmetric (real orbitals), rs and
 * pq run over the pairs r >= s only: D[rs] + D[sr] is accumulated in one row
 * and G[pq] = G[qp].  Otherwise all norb^2 pairs are used.
 *
 * A block is one alpha string times a range of beta strings times up to
 * SIGMA_NV vectors, so that D and G stay in the cache and every gather or
 * scatter moves SIGMA_NV numbers.  Threads: with enough vectors (half a
 * batch per thread) every thread takes its own batches of vectors (copies of
 * its columns, nothing shared); with fewer the threads share the alpha
 * strings of one batch, and the alpha scatter, which leaves the alpha
 * string, goes into a buffer per thread.  The dgemm is called inside the parallel region
 * and is sequential there. */
#define SIGMA_NV 8
#define SIGMA_NCOL 512

typedef struct {
        int32_t addr;           /* address of the string I */
        int16_t pd;             /* row of D: pair (q, p) */
        int16_t pg;             /* row of G: pair (p, q) */
        double sign;
} sigma_link;

typedef struct {
        int64_t dima, dimb;
        int nla, nlb;           /* links per alpha / beta string */
        int npair;
        const sigma_link *la, *lb;
        const double *v;        /* (npair, npair + 1) */
} sigma_ctx;

static inline int sigma_pair(int p, int q, int norb, int packed)
{
        if (!packed) {
                return p * norb + q;
        }
        return p >= q ? p * (p + 1) / 2 + q : q * (q + 1) / 2 + p;
}

static sigma_link *sigma_links(int norb, int ne, int64_t dim, const int64_t *binom,
                               int packed, int *nlink)
{
        int nl = ne * (norb - ne) + ne;
        uint64_t *strs = ascending_strings(ne, dim);
        sigma_link *links = malloc(sizeof(sigma_link) * (dim * nl > 0 ? dim * nl : 1));
        for (int64_t j = 0; j < dim; j++) {
                uint64_t sj = strs[j];
                sigma_link *l = links + j * nl;
                for (int q = 0; q < norb; q++) {
                        if (!(sj >> q & 1)) continue;
                        int sq = parity_above(sj, q);
                        uint64_t s1 = sj ^ (1ULL << q);
                        for (int p = 0; p < norb; p++) {
                                if (s1 >> p & 1) continue;
                                l->addr = (int32_t)(p == q ? j : string_rank(s1 | (1ULL << p), binom, norb));
                                l->pd = (int16_t)sigma_pair(q, p, norb, packed);
                                l->pg = (int16_t)sigma_pair(p, q, norb, packed);
                                l->sign = ((sq + parity_above(s1, p)) & 1) ? -1.0 : 1.0;
                                l++;
                        }
                }
        }
        free(strs);
        *nlink = nl;
        return links;
}

/* y[j, :nv] += f x[j, :nv] for n rows with leading dimensions ldx, ldy */
static inline __attribute__((always_inline))
void axpy_rows(int64_t n, int nv, double f, const double *x, int64_t ldx,
               double *y, int64_t ldy)
{
        if (ldx == nv && ldy == nv) {
                axpy(n * nv, f, x, y);
                return;
        }
        for (int64_t j = 0; j < n; j++) {
                axpy(nv, f, x + j * ldx, y + j * ldy);
        }
}

/* The determinants (ja, b0 <= jb < b1) of nv vectors c[det * ldc + v]: the
 * part of sigma within the alpha string is added to sb, the part that
 * changes the alpha string to sa.  d: (npair + 1) * ncol, g: npair * ncol
 * with ncol = (b1 - b0) * nv. */
static inline __attribute__((always_inline))
void sigma_block_nv(const sigma_ctx *x, int64_t ja, int64_t b0, int64_t b1, const int nv,
                    const double *c, int64_t ldc, double *sb, int64_t ldb,
                    double *sa, int64_t lda, double *d, double *g)
{
        int64_t dimb = x->dimb;
        int64_t nj = b1 - b0;
        int64_t ncol = nj * nv;
        int npair = x->npair;
        memset(d, 0, sizeof(double) * npair * ncol);
        const double *cj = c + (ja * dimb + b0) * ldc;
        double *last = d + npair * ncol;
        for (int64_t j = 0; j < nj; j++) {
                memcpy(last + j * nv, cj + j * ldc, sizeof(double) * nv);
        }
        const sigma_link *la = x->la + ja * x->nla;
        for (int k = 0; k < x->nla; k++) {
                axpy_rows(nj, nv, la[k].sign, c + (la[k].addr * dimb + b0) * ldc, ldc,
                          d + la[k].pd * ncol, nv);
        }
        const double *ca = c + ja * dimb * ldc;
        for (int64_t j = 0; j < nj; j++) {
                const sigma_link *lb = x->lb + (b0 + j) * x->nlb;
                for (int k = 0; k < x->nlb; k++) {
                        axpy(nv, lb[k].sign, ca + lb[k].addr * ldc, d + lb[k].pd * ncol + j * nv);
                }
        }
        cblas_dgemm(101, 111, 111, npair, (int)ncol, npair + 1, 1.0, x->v, npair + 1,
                    d, (int)ncol, 0.0, g, (int)ncol);
        double *sj = sb + ja * dimb * ldb;
        for (int64_t j = 0; j < nj; j++) {
                const sigma_link *lb = x->lb + (b0 + j) * x->nlb;
                for (int k = 0; k < x->nlb; k++) {
                        axpy(nv, lb[k].sign, g + lb[k].pg * ncol + j * nv, sj + lb[k].addr * ldb);
                }
        }
        for (int k = 0; k < x->nla; k++) {
                axpy_rows(nj, nv, la[k].sign, g + la[k].pg * ncol, nv,
                          sa + (la[k].addr * dimb + b0) * lda, lda);
        }
}

/* the loops over the vectors get a fixed length for the common batch sizes */
static void sigma_block(const sigma_ctx *x, int64_t ja, int64_t b0, int64_t b1, int nv,
                        const double *c, int64_t ldc, double *sb, int64_t ldb,
                        double *sa, int64_t lda, double *d, double *g)
{
        switch (nv) {
        case SIGMA_NV:
                sigma_block_nv(x, ja, b0, b1, SIGMA_NV, c, ldc, sb, ldb, sa, lda, d, g);
                break;
        case 4:
                sigma_block_nv(x, ja, b0, b1, 4, c, ldc, sb, ldb, sa, lda, d, g);
                break;
        case 2:
                sigma_block_nv(x, ja, b0, b1, 2, c, ldc, sb, ldb, sa, lda, d, g);
                break;
        case 1:
                sigma_block_nv(x, ja, b0, b1, 1, c, ldc, sb, ldb, sa, lda, d, g);
                break;
        default:
                sigma_block_nv(x, ja, b0, b1, nv, c, ldc, sb, ldb, sa, lda, d, g);
        }
}

/* the matrix of the product: V[pq, rs] = (pq|rs) / 2, V[pq, npair] = k_pq */
static double *sigma_integrals(int norb, const double *h1, const double *eri, int *packed_out)
{
        size_t n1 = norb, n2 = n1 * n1, n3 = n2 * n1;
        double *k = malloc(sizeof(double) * (n2 > 0 ? n2 : 1));
        double scale = 0, asym = 0;
        for (int p = 0; p < norb; p++) {
                for (int q = 0; q < norb; q++) {
                        double s = h1[p * n1 + q];
                        for (int r = 0; r < norb; r++) {
                                s -= 0.5 * eri[p * n3 + r * n2 + r * n1 + q];
                        }
                        k[p * n1 + q] = s;
                }
        }
        for (int p = 0; p < norb; p++) {
                for (int q = 0; q < norb; q++) {
                        double a = fabs(k[p * n1 + q]);
                        double b = fabs(k[p * n1 + q] - k[q * n1 + p]);
                        if (a > scale) scale = a;
                        if (b > asym) asym = b;
                        for (int r = 0; r < norb; r++) {
                                for (int s = 0; s < norb; s++) {
                                        double e = eri[p * n3 + q * n2 + r * n1 + s];
                                        a = fabs(e);
                                        if (a > scale) scale = a;
                                        b = fabs(e - eri[q * n3 + p * n2 + r * n1 + s]);
                                        if (b > asym) asym = b;
                                        b = fabs(e - eri[p * n3 + q * n2 + s * n1 + r]);
                                        if (b > asym) asym = b;
                                }
                        }
                }
        }
        /* symmetric to rounding: the packed form uses the average */
        int packed = asym <= 1e-14 * scale;
        int npair = packed ? norb * (norb + 1) / 2 : norb * norb;
        size_t ld = npair + 1;
        double *v = malloc(sizeof(double) * (npair > 0 ? npair * ld : 1));
        for (int p = 0; p < norb; p++) {
                for (int q = 0; q <= (packed ? p : norb - 1); q++) {
                        double *row = v + sigma_pair(p, q, norb, packed) * ld;
                        row[npair] = packed ? 0.5 * (k[p * n1 + q] + k[q * n1 + p]) : k[p * n1 + q];
                        for (int r = 0; r < norb; r++) {
                                for (int s = 0; s <= (packed ? r : norb - 1); s++) {
                                        double e = eri[p * n3 + q * n2 + r * n1 + s];
                                        if (packed) {
                                                e = 0.25 * (e + eri[q * n3 + p * n2 + r * n1 + s]
                                                            + eri[p * n3 + q * n2 + s * n1 + r]
                                                            + eri[q * n3 + p * n2 + s * n1 + r]);
                                        }
                                        row[sigma_pair(r, s, norb, packed)] = 0.5 * e;
                                }
                        }
                }
        }
        free(k);
        *packed_out = packed;
        return v;
}

void zmr_sigma_ab(int norb, int na, int nb, const double *h1, const double *eri,
                  int64_t nvec, const double *q, double *out)
{
        if (na < 0 || na > norb || nb < 0 || nb > norb || nvec <= 0) {
                return;
        }
        int64_t *binom = binomial_table(norb);
        sigma_ctx x;
        x.dima = binom[norb * (norb + 1) + na];
        x.dimb = binom[norb * (norb + 1) + nb];
        int64_t dima = x.dima, dimb = x.dimb;
        int64_t dim = dima * dimb;
        int packed;
        double *v = sigma_integrals(norb, h1, eri, &packed);
        x.v = v;
        x.npair = packed ? norb * (norb + 1) / 2 : norb * norb;
        sigma_link *la = sigma_links(norb, na, dima, binom, packed, &x.nla);
        sigma_link *lb = sigma_links(norb, nb, dimb, binom, packed, &x.nlb);
        x.la = la;
        x.lb = lb;
        int npair = x.npair;
        double work = (double)dim * nvec * ((double)npair * (npair + 1) + 4.0 * (x.nla + x.nlb));
        int nt = threads_for(work);
        if (nt == 1 || nvec >= (int64_t)nt * (SIGMA_NV / 2)) {
                /* every thread its own batches of vectors */
                int nv0 = (int)(nvec / nt < SIGMA_NV ? nvec / nt : SIGMA_NV);
                int64_t nbatch = (nvec + nv0 - 1) / nv0;
                int64_t nj0 = SIGMA_NCOL / nv0 < dimb ? SIGMA_NCOL / nv0 : dimb;
#pragma omp parallel num_threads(nt)
                {
                        double *c = malloc(sizeof(double) * dim * nv0);
                        double *s = malloc(sizeof(double) * dim * nv0);
                        double *d = malloc(sizeof(double) * (npair + 1) * nj0 * nv0);
                        double *g = malloc(sizeof(double) * (npair + 1) * nj0 * nv0);
#pragma omp for schedule(dynamic, 1)
                        for (int64_t b = 0; b < nbatch; b++) {
                                int64_t v0 = b * nv0;
                                int nv = (int)(nvec - v0 < nv0 ? nvec - v0 : nv0);
                                for (int64_t i = 0; i < dim; i++) {
                                        memcpy(c + i * nv, q + i * nvec + v0, sizeof(double) * nv);
                                }
                                memset(s, 0, sizeof(double) * dim * nv);
                                for (int64_t ja = 0; ja < dima; ja++) {
                                        for (int64_t b0 = 0; b0 < dimb; b0 += nj0) {
                                                int64_t b1 = b0 + nj0 < dimb ? b0 + nj0 : dimb;
                                                sigma_block(&x, ja, b0, b1, nv, c, nv, s, nv, s, nv, d, g);
                                        }
                                }
                                for (int64_t i = 0; i < dim; i++) {
                                        memcpy(out + i * nvec + v0, s + i * nv, sizeof(double) * nv);
                                }
                        }
                        free(c);
                        free(s);
                        free(d);
                        free(g);
                }
        } else {
                /* few vectors: the threads share the alpha strings */
                memset(out, 0, sizeof(double) * dim * nvec);
                for (int64_t v0 = 0; v0 < nvec; v0 += SIGMA_NV) {
                        int nv = (int)(nvec - v0 < SIGMA_NV ? nvec - v0 : SIGMA_NV);
                        int64_t nj0 = SIGMA_NCOL / nv < dimb ? SIGMA_NCOL / nv : dimb;
#pragma omp parallel num_threads(nt)
                        {
                                double *s = calloc(dim * nv, sizeof(double));
                                double *d = malloc(sizeof(double) * (npair + 1) * nj0 * nv);
                                double *g = malloc(sizeof(double) * (npair + 1) * nj0 * nv);
#pragma omp for schedule(dynamic, 1)
                                for (int64_t ja = 0; ja < dima; ja++) {
                                        for (int64_t b0 = 0; b0 < dimb; b0 += nj0) {
                                                int64_t b1 = b0 + nj0 < dimb ? b0 + nj0 : dimb;
                                                sigma_block(&x, ja, b0, b1, nv, q + v0, nvec,
                                                            out + v0, nvec, s, nv, d, g);
                                        }
                                }
#pragma omp critical(zmr_sigma_ab_reduce)
                                axpy_rows(dim, nv, 1.0, s, nv, out + v0, nvec);
                                free(s);
                                free(d);
                                free(g);
                        }
                }
        }
        free(v);
        free(la);
        free(lb);
        free(binom);
}

void zmr_apply_strings_ab(int norb, int na, int nb, int na2, int nb2, int nop,
                          const int *dagger, const int64_t *ops,
                          int64_t nx, int64_t nx_total, int64_t x0, int64_t nvec,
                          const double *vecs, double *out)
{
        if (na < 0 || na > norb || nb < 0 || nb > norb
            || na2 < 0 || na2 > norb || nb2 < 0 || nb2 > norb) {
                return;
        }
        int64_t *binom = binomial_table(norb);
        int64_t dima = binom[norb * (norb + 1) + na];
        int64_t dimb = binom[norb * (norb + 1) + nb];
        int64_t dimb2 = binom[norb * (norb + 1) + nb2];
        int64_t dsrc = dima * dimb;
        int64_t ddst = binom[norb * (norb + 1) + na2] * dimb2;
        uint64_t *stra = ascending_strings(na, dima);
        uint64_t *strb = ascending_strings(nb, dimb);
#pragma omp parallel num_threads(threads_for((double)nx * dsrc * (4 * nop + 2 * nvec + 4)))
        {
                /* map of one operator string: +-(address + 1), 0 if annihilated */
                int64_t *map = malloc(sizeof(int64_t) * dsrc);
#pragma omp for schedule(static)
                for (int64_t x = 0; x < nx; x++) {
                        const int64_t *op = ops + x * nop;
                        for (int64_t i = 0; i < dsrc; i++) {
                                uint64_t s = (stra[i / dimb] << norb) | strb[i % dimb];
                                int par = 0;
                                int ok = 1;
                                for (int k = nop - 1; k >= 0; k--) {
                                        int p = (int)op[k];
                                        int occ = (int)(s >> p & 1);
                                        if (occ == (dagger[k] != 0)) {
                                                ok = 0;
                                                break;
                                        }
                                        par ^= parity_above(s, p);
                                        s ^= 1ULL << p;
                                }
                                if (ok) {
                                        int64_t addr = rank_ab(s, norb, dimb2, binom) + 1;
                                        map[i] = par ? -addr : addr;
                                } else {
                                        map[i] = 0;
                                }
                        }
                        for (int64_t v = 0; v < nvec; v++) {
                                double *o = out + (v * nx_total + x0 + x) * ddst;
                                const double *c = vecs + v * dsrc;
                                memset(o, 0, sizeof(double) * ddst);
                                for (int64_t i = 0; i < dsrc; i++) {
                                        int64_t m = map[i];
                                        if (m > 0) {
                                                o[m - 1] = c[i];
                                        } else if (m < 0) {
                                                o[-m - 1] = -c[i];
                                        }
                                }
                        }
                }
                free(map);
        }
        free(stra);
        free(strb);
        free(binom);
}

/*
 * out[dest[v * nx + x], :] += O_x vecs[v, :]: as zmr_apply_strings_ab, but the
 * result of string x on vector v is added to the row dest[v * nx + x] of out
 * (rows of ddst numbers; a negative destination skips the product).  The
 * strings are distributed over the threads: the destinations of different
 * strings must be different rows.
 */
void zmr_apply_strings_ab_add(int norb, int na, int nb, int na2, int nb2, int nop,
                              const int *dagger, const int64_t *ops,
                              int64_t nx, int64_t nvec, const double *vecs,
                              const int64_t *dest, double *out)
{
        if (na < 0 || na > norb || nb < 0 || nb > norb
            || na2 < 0 || na2 > norb || nb2 < 0 || nb2 > norb) {
                return;
        }
        int64_t *binom = binomial_table(norb);
        int64_t dima = binom[norb * (norb + 1) + na];
        int64_t dimb = binom[norb * (norb + 1) + nb];
        int64_t dimb2 = binom[norb * (norb + 1) + nb2];
        int64_t dsrc = dima * dimb;
        int64_t ddst = binom[norb * (norb + 1) + na2] * dimb2;
        uint64_t *stra = ascending_strings(na, dima);
        uint64_t *strb = ascending_strings(nb, dimb);
#pragma omp parallel num_threads(threads_for((double)nx * dsrc * (4 * nop + 2 * nvec + 4)))
        {
                /* the non-zero entries of one operator string: source address,
                 * +-(target address + 1) */
                int64_t *src = malloc(sizeof(int64_t) * dsrc);
                int64_t *map = malloc(sizeof(int64_t) * dsrc);
#pragma omp for schedule(static)
                for (int64_t x = 0; x < nx; x++) {
                        const int64_t *op = ops + x * nop;
                        int64_t nnz = 0;
                        for (int64_t i = 0; i < dsrc; i++) {
                                uint64_t s = (stra[i / dimb] << norb) | strb[i % dimb];
                                int par = 0;
                                int ok = 1;
                                for (int k = nop - 1; k >= 0; k--) {
                                        int p = (int)op[k];
                                        int occ = (int)(s >> p & 1);
                                        if (occ == (dagger[k] != 0)) {
                                                ok = 0;
                                                break;
                                        }
                                        par ^= parity_above(s, p);
                                        s ^= 1ULL << p;
                                }
                                if (ok) {
                                        int64_t addr = rank_ab(s, norb, dimb2, binom) + 1;
                                        src[nnz] = i;
                                        map[nnz++] = par ? -addr : addr;
                                }
                        }
                        for (int64_t v = 0; v < nvec; v++) {
                                int64_t row = dest[v * nx + x];
                                if (row < 0) {
                                        continue;
                                }
                                double *o = out + row * ddst;
                                const double *c = vecs + v * dsrc;
                                for (int64_t k = 0; k < nnz; k++) {
                                        int64_t m = map[k];
                                        if (m > 0) {
                                                o[m - 1] += c[src[k]];
                                        } else {
                                                o[-m - 1] -= c[src[k]];
                                        }
                                }
                        }
                }
                free(src);
                free(map);
        }
        free(stra);
        free(strb);
        free(binom);
}

static int64_t find_root(int64_t *parent, int64_t i)
{
        while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
        }
        return i;
}

int64_t zmr_components(const double complex *a, int64_t n, int64_t *labels)
{
        int64_t *parent = malloc(sizeof(int64_t) * (n > 0 ? n : 1));
        for (int64_t i = 0; i < n; i++) {
                parent[i] = i;
        }
        for (int64_t i = 0; i < n; i++) {
                const double complex *row = a + i * n;
                int64_t ri = find_root(parent, i);
                for (int64_t j = 0; j < n; j++) {
                        if (parent[j] == ri || (creal(row[j]) == 0 && cimag(row[j]) == 0)) {
                                continue;
                        }
                        int64_t rj = find_root(parent, j);
                        if (rj != ri) {
                                /* keep the smaller index as the root */
                                if (rj < ri) {
                                        parent[ri] = rj;
                                        ri = rj;
                                } else {
                                        parent[rj] = ri;
                                }
                        }
                }
        }
        int64_t ncomp = 0;
        for (int64_t i = 0; i < n; i++) {
                int64_t r = find_root(parent, i);
                /* the root is the smallest index of its component, so it has
                 * been labelled before i */
                labels[i] = (r == i) ? ncomp++ : labels[r];
        }
        free(parent);
        return ncomp;
}

void zmr_block_support(const double complex *rows, int64_t m, int64_t d,
                       const int64_t *labels, int64_t ncomp, unsigned char *touch)
{
        memset(touch, 0, (size_t)(m * ncomp));
#pragma omp parallel for schedule(static) num_threads(threads_for((double)m * d))
        for (int64_t x = 0; x < m; x++) {
                const double complex *r = rows + x * d;
                unsigned char *t = touch + x * ncomp;
                for (int64_t i = 0; i < d; i++) {
                        if (creal(r[i]) != 0 || cimag(r[i]) != 0) {
                                t[labels[i]] = 1;
                        }
                }
        }
}

void zmr_fold_density(int64_t nao, int64_t n1, int64_t n2,
                      const double complex *c1a, const double complex *c1b,
                      const double complex *c2a, const double complex *c2b,
                      double complex *out)
{
        int64_t npair = nao * (nao + 1) / 2;
#pragma omp parallel for collapse(2) schedule(static) \
        num_threads(threads_for((double)n1 * n2 * npair * 4))
        for (int64_t i = 0; i < n1; i++) {
                for (int64_t j = 0; j < n2; j++) {
                        double complex *o = out + (i * n2 + j) * npair;
                        int64_t mn = 0;
                        for (int64_t m = 0; m < nao; m++) {
                                double complex am = conj(c1a[m * n1 + i]);
                                double complex bm = conj(c1b[m * n1 + i]);
                                double complex xm = c2a[m * n2 + j];
                                double complex ym = c2b[m * n2 + j];
                                for (int64_t n = 0; n < m; n++, mn++) {
                                        o[mn] = am * c2a[n * n2 + j] + bm * c2b[n * n2 + j]
                                                + conj(c1a[n * n1 + i]) * xm + conj(c1b[n * n1 + i]) * ym;
                                }
                                o[mn++] = am * xm + bm * ym;
                        }
                }
        }
}


/* ----------------------------------------------------------------------
 * Array bookkeeping of the MRCI sigma: transposed additions and copies,
 * gathers and scatters over antisymmetric index pairs. */

/* dst (+)= alpha * transpose(src): dst has ndim axes of dimensions dshape,
 * C order; the element dst[i_0, .., i_(n-1)] corresponds to src at the
 * offset sum_k i_k sstride[k] (strides of src in elements, in the order of
 * the axes of dst). */
void zmr_transpose_add(int ndim, const int64_t *dshape, const int64_t *sstride,
                       const double complex *src, double complex *dst,
                       int assign, const double *alpha)
{
        const double complex a = alpha[0] + alpha[1] * I;
        const int unit = (alpha[0] == 1.0 && alpha[1] == 0.0);
        if (ndim == 0) {
                dst[0] = assign ? a * src[0] : dst[0] + a * src[0];
                return;
        }
        /* the two innermost axes of dst are traversed in tiles: along one of
         * them the source is read with a large stride, and a tile keeps both
         * the lines read and the lines written in the cache */
        const int64_t tile = 64;
        const int64_t n2 = dshape[ndim - 1];
        const int64_t s2 = sstride[ndim - 1];
        const int64_t n1 = ndim > 1 ? dshape[ndim - 2] : 1;
        const int64_t s1 = ndim > 1 ? sstride[ndim - 2] : 0;
        int64_t nouter = 1;
        for (int k = 0; k < ndim - 2; k++) nouter *= dshape[k];
        if (nouter * n1 * n2 == 0) return;
        const int64_t nb1 = (n1 + tile - 1) / tile;
#pragma omp parallel for collapse(2) schedule(static) \
        num_threads(threads_for((double)nouter * n1 * n2 * 8))
        for (int64_t o = 0; o < nouter; o++) {
                for (int64_t b1 = 0; b1 < nb1; b1++) {
                        int64_t rem = o, off = 0;
                        for (int k = ndim - 3; k >= 0; k--) {
                                off += (rem % dshape[k]) * sstride[k];
                                rem /= dshape[k];
                        }
                        const int64_t i0 = b1 * tile;
                        const int64_t i1 = i0 + tile < n1 ? i0 + tile : n1;
                        for (int64_t j0 = 0; j0 < n2; j0 += tile) {
                                const int64_t j1 = j0 + tile < n2 ? j0 + tile : n2;
                                for (int64_t i = i0; i < i1; i++) {
                                        double complex *d = dst + (o * n1 + i) * n2;
                                        const double complex *x = src + off + i * s1;
                                        if (assign) {
                                                if (unit) for (int64_t j = j0; j < j1; j++) d[j] = x[j * s2];
                                                else for (int64_t j = j0; j < j1; j++) d[j] = a * x[j * s2];
                                        } else {
                                                if (unit) for (int64_t j = j0; j < j1; j++) d[j] += x[j * s2];
                                                else for (int64_t j = j0; j < j1; j++) d[j] += a * x[j * s2];
                                        }
                                }
                        }
                }
        }
}

/* dst[l, p, t] = src[l, i_p, j_p, t] for the pairs i_p < j_p of range(n):
 * src (nl, n, n, nt), dst (nl, n (n - 1) / 2, nt). */
void zmr_pair_gather(int64_t nl, int64_t n, int64_t nt,
                     const double complex *src, double complex *dst)
{
        const int64_t np = n * (n - 1) / 2;
        if (nl * np * nt == 0) return;
        int64_t *flat = malloc(sizeof(int64_t) * np);
        int64_t p = 0;
        for (int64_t i = 0; i < n; i++)
                for (int64_t j = i + 1; j < n; j++) flat[p++] = i * n + j;
#pragma omp parallel for collapse(2) schedule(static) \
        num_threads(threads_for((double)nl * np * nt * 8))
        for (int64_t l = 0; l < nl; l++) {
                for (int64_t q = 0; q < np; q++) {
                        memcpy(dst + (l * np + q) * nt, src + (l * n * n + flat[q]) * nt,
                               sizeof(double complex) * nt);
                }
        }
        free(flat);
}

/* The antisymmetric array: dst[l, i, j, t] = src[l, p, t], dst[l, j, i, t] =
 * -src[l, p, t], zero on the diagonal; dst (nl, n, n, nt) is overwritten. */
void zmr_pair_scatter(int64_t nl, int64_t n, int64_t nt,
                      const double complex *src, double complex *dst)
{
        const int64_t np = n * (n - 1) / 2;
        if (nl * n * nt == 0) return;
#pragma omp parallel for collapse(2) schedule(static) \
        num_threads(threads_for((double)nl * n * n * nt * 8))
        for (int64_t l = 0; l < nl; l++) {
                for (int64_t i = 0; i < n; i++) {
                        double complex *row = dst + (l * n * n + i * n) * nt;
                        for (int64_t j = 0; j < n; j++) {
                                double complex *d = row + j * nt;
                                if (i == j) {
                                        memset(d, 0, sizeof(double complex) * nt);
                                        continue;
                                }
                                int64_t a = i < j ? i : j, b = i < j ? j : i;
                                int64_t q = a * n - a * (a + 1) / 2 + (b - a - 1);
                                const double complex *x = src + (l * np + q) * nt;
                                if (i < j) {
                                        memcpy(d, x, sizeof(double complex) * nt);
                                } else {
                                        for (int64_t t = 0; t < nt; t++) d[t] = -x[t];
                                }
                        }
                }
        }
}
