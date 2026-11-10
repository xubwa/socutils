/*
 * Direct J / K for many density matrices at once (CPHF, TDDFT responses):
 *
 *     J_ab = sum_cd (ab|cd) D_dc,     K_ad = sum_bc (ab|cd) D_bc,
 *
 * over the unique shell quartets (i >= j, k >= l, ij >= kl), one libcint
 * evaluation per quartet shared by all densities.  The densities are stored
 * as (nao, nao, ndm), density index innermost, so that each integral updates
 * J and K of all densities with contiguous vector operations (the eight
 * permutations of the quartet, any symmetry of D).  Quartets are skipped when
 * q[ij] q[kl] (Schwarz) times the largest density element on the six index
 * pairs is below tol.  Per-thread J / K, reduced at the end.  Spherical AOs.
 */
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cint.h"

int int2e_sph(double *out, int *dims, int *shls, int *atm, int natm,
              int *bas, int nbas, double *env, CINTOpt *opt, double *cache);
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env);

/* jk_ssss.c: s / p quartets with at most two p shells without libcint */
typedef struct SSCtx SSCtx;
SSCtx *SOCVHFss_ctx_build(int *atm, int natm, int *bas, int nbas, double *env);
void SOCVHFss_ctx_free(SSCtx *cx);
size_t SOCVHFss_wksize(int *bas, int nbas);
int SOCVHFss_int2e(double *out, const int *shls, const SSCtx *cx, double dmax, double prim_tol, double *wk);
int SOCVHFss_takes(const int *shls, const int *bas);

/* jk_gen.c: quartets of l <= 2 by the generated Head-Gordon-Pople kernels */
typedef struct GenCtx GenCtx;
GenCtx *SOCVHFgen_ctx_build(int *atm, int natm, int *bas, int nbas, double *env);
void SOCVHFgen_ctx_free(GenCtx *cx);
size_t SOCVHFgen_wksize(int *bas, int nbas);
int SOCVHFgen_int2e(double *out, const int *shls, const GenCtx *cx, double dmax, double prim_tol,
                    double *wk, const int *bas);
int SOCVHFgen_takes(const int *shls, const int *bas);

/*
 * SOCVHFjk_multi_sym's engines by the total l of the quartet (cc-pVDZ
 * per-class timings, scf/test/bench_jk_class.py): L <= 1 by jk_ssss.c, the
 * other quartets of l <= 2 with L <= jk_gen_lmax by the generated kernels of
 * jk_gen.c, the rest by libcint (ahead from L = 7; at L = 6 the lanes are twice as fast, caffeine).
 * Of the generated ones those of few primitives go in batches of GL_NL
 * quartets sharing the bra pair and the ket shell types to the lane kernels
 * (SOCVHFjk_set_lanes, SOCVHFjk_set_lane_np below).
 * SOCVHFjk_set_ssss(0) / SOCVHFjk_set_gen(0) send theirs to libcint.
 */
static int jk_use_ssss = 1, jk_use_gen = 1, jk_gen_lmax = 6, jk_gen_f_lmax = 6;
void SOCVHFjk_set_ssss(int on) { jk_use_ssss = on; }
void SOCVHFjk_set_gen(int on) { jk_use_gen = on; }
void SOCVHFjk_set_gen_lmax(int l) { jk_gen_lmax = l; }
/* quartets with f shells: generated up to total l jk_gen_f_lmax (gen_jk_vrr.py F_LSUM_MAX), libcint above */
void SOCVHFjk_set_gen_f_lmax(int l) { jk_gen_f_lmax = l; }

static const int PERMS[8][4] = {
        {0, 1, 2, 3}, {1, 0, 2, 3}, {0, 1, 3, 2}, {1, 0, 3, 2},
        {2, 3, 0, 1}, {3, 2, 0, 1}, {2, 3, 1, 0}, {3, 2, 1, 0}};

static inline double dmax2(double a, double b) { return a > b ? a : b; }

static inline void shell_pair(int ij, int *i, int *j)
{
        int a = (int)((__builtin_sqrt(1. + 8. * ij) - 1) / 2);
        while (a * (a + 1) / 2 > ij) a--;
        while ((a + 1) * (a + 2) / 2 <= ij) a++;
        *i = a;
        *j = ij - a * (a + 1) / 2;
}

void SOCVHFjk_multi(double *vj, double *vk, double *dms, int ndm,
                    int with_j, int with_k,
                    double *qcond, double *dcond, double tol, CINTOpt *opt,
                    int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        const size_t nv = nao * nao * ndm;
        int dmax = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        size_t cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)dmax * dmax * dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        if (with_j) memset(vj, 0, sizeof(double) * nv);
        if (with_k) memset(vk, 0, sizeof(double) * nv);
#pragma omp parallel
{
        double *jloc = with_j ? calloc(nv, sizeof(double)) : NULL;
        double *kloc = with_k ? calloc(nv, sizeof(double)) : NULL;
        double *buf = malloc(sizeof(double) * (nbuf + cache_size));
        double *cache = buf + nbuf;
        int ij, kl, ish, jsh, ksh, lsh, p, m;
        int shls[4];
#pragma omp for schedule(dynamic, 2)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        double d = dmax2(dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]),
                                         dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                               dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh])));
                        if (q4 * d < tol) {
                                continue;
                        }
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        if (!int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                                continue;
                        }
                        int di = ao_loc[ish + 1] - ao_loc[ish];
                        int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                        int dl = ao_loc[lsh + 1] - ao_loc[lsh];
                        double fac = 1.;
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        int i0, j0, k0, l0;
                        for (l0 = 0; l0 < dl; l0++) {
                        for (k0 = 0; k0 < dk; k0++) {
                        for (j0 = 0; j0 < dj; j0++) {
                        for (i0 = 0; i0 < di; i0++) {
                                double g = buf[((l0 * dk + k0) * dj + j0) * di + i0] * fac;
                                if (g == 0) continue;
                                size_t ao[4] = {ao_loc[ish] + i0, ao_loc[jsh] + j0,
                                                ao_loc[ksh] + k0, ao_loc[lsh] + l0};
                                for (p = 0; p < 8; p++) {
                                        size_t a = ao[PERMS[p][0]], b = ao[PERMS[p][1]];
                                        size_t c = ao[PERMS[p][2]], e = ao[PERMS[p][3]];
                                        if (with_j) {
                                                double *v = jloc + (a * nao + b) * ndm;
                                                double *dd = dms + (e * nao + c) * ndm;
#pragma omp simd
                                                for (m = 0; m < ndm; m++) v[m] += g * dd[m];
                                        }
                                        if (with_k) {
                                                double *v = kloc + (a * nao + e) * ndm;
                                                double *dd = dms + (b * nao + c) * ndm;
#pragma omp simd
                                                for (m = 0; m < ndm; m++) v[m] += g * dd[m];
                                        }
                                }
                        } } } }
                }
        }
#pragma omp critical
        {
                size_t n;
                if (with_j) for (n = 0; n < nv; n++) vj[n] += jloc[n];
                if (with_k) for (n = 0; n < nv; n++) vk[n] += kloc[n];
        }
        free(jloc);
        free(kloc);
        free(buf);
}
}


/* out[0:n] = sum_t bufs[t][0:n], each thread of the enclosing parallel region
 * summing its own slice (instead of the buffers one after another) */
static void reduce_thread_buffers(double *out, double **bufs, int nbuf, size_t n)
{
        const int t = omp_get_thread_num(), nt = omp_get_num_threads();
        const size_t chunk = (n + nt - 1) / nt;
        const size_t s0 = chunk * t < n ? chunk * t : n;
        const size_t s1 = s0 + chunk < n ? s0 + chunk : n;
        size_t i;
        int k;
        for (i = s0; i < s1; i++) out[i] = 0;
        for (k = 0; k < nbuf; k++) {
                const double *b = bufs[k];
                for (i = s0; i < s1; i++) out[i] += b[i];
        }
}

/* onx.c: significant ket pairs of a bra pair from sorted density / Schwarz partner lists */
typedef struct ONXLists ONXLists;
typedef struct ONXScratch ONXScratch;
ONXLists *SOCVHFonx_build(const double *q, const double *dcond, int nbas);
void SOCVHFonx_free(ONXLists *L);
ONXScratch *SOCVHFonx_scratch(int nbas);
void SOCVHFonx_scratch_free(ONXScratch *s);
int SOCVHFonx_kpairs(const ONXLists *L, ONXScratch *S, int ish, int jsh, int ij, double thr,
                     const double *dcond, int *out);
/* K-only builds: the ket pairs of a bra pair from onx.c (0: every kl <= ij tested) */
static int jk_use_onx = 1;
void SOCVHFjk_set_onx(int on) { jk_use_onx = on; }

/* lanes of jk_gen.c: GL_NL quartets sharing the bra pair and the ket shell types per call */
int SOCVHFgen_shell_types(int *type, const int *bas, int nbas);
int SOCVHFgen_lane_ok(int ksh, int lsh, const int *bas);
size_t SOCVHFgen_lane_wksize(const int *bas, int nbas, int nbuf);
int SOCVHFgen_int2e_lanes(double *out, size_t ostride, int ish, int jsh, const int *ksh, const int *lsh, int nl,
                          const GenCtx *cx, double dmax, double prim_tol, double *wk,
                          const int *atm, const int *bas, const double *env);
#ifndef GL_NL
#define GL_NL 8
#endif
#define JK_NL GL_NL             /* = GL_NL of jk_gen.h */
/* J-only builds (with_k 0, one density): the generated quartets through the J forms, the density
 * contracted before the horizontal recursions (0: the full integrals and jk_digest) */
int SOCVHFgen_j2e(double *jb, double *jk, const double *db, const double *dk, const int *shls,
                  const GenCtx *cx, double dmax, double prim_tol, double *wk);
static int jk_use_jonly = 1;
void SOCVHFjk_set_jonly(int on) { jk_use_jonly = on; }
/* the quartets of l <= 2 with jk_lane_lmin <= L <= jk_gen_lmax and at most jk_lane_np primitive
 * quartets (product of the shells' nprim) go to the lanes (on 0: never); the per-quartet kernels
 * vectorise over the bra primitives and screen them, better with many (bench_jk_class.py) */
static int jk_use_lanes = 1, jk_lane_lmin = 2, jk_lane_np = 16;
void SOCVHFjk_set_lanes(int on, int lmin) { jk_use_lanes = on; jk_lane_lmin = lmin; }
void SOCVHFjk_set_lane_np(int np) { jk_lane_np = np; }

/* the integrals buf (int2e_sph layout) of the quartet (ish jsh | ksh lsh) of
 * jk_multi_sym into J_ij (jij), J (jloc), the K rows of i, j (kri, krb) */
static inline void jk_digest(const double *buf, int ish, int jsh, int ksh, int lsh, int same, int do_j, int do_k,
                             int ndm, size_t nao, const int *ao_loc, const double *dms,
                             double *jij, double *jloc, double *kri, double *krb, size_t ncol)
{
        const int di = ao_loc[ish + 1] - ao_loc[ish];
        const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
        const int dk = ao_loc[ksh + 1] - ao_loc[ksh];
        const int dl = ao_loc[lsh + 1] - ao_loc[lsh];
        const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
        double fac = 1.;
        if (ish == jsh) fac *= .5;
        if (ksh == lsh) fac *= .5;
        if (same) fac *= .5;
        int i0, j0, k0, l0, m;
        if (ndm == 1) {
                /* one density: no density loop, row pointers, the j-row K and the
                 * ket J sums kept in registers over the inner loop */
                const size_t c00 = ao_loc[ksh], e00 = ao_loc[lsh];
                for (l0 = 0; l0 < dl; l0++) {
                const size_t e = e00 + l0;
                for (k0 = 0; k0 < dk; k0++) {
                        const size_t c = c00 + k0;
                        const double *gk = buf + (size_t)(l0 * dk + k0) * dj * di;
                        const double dce = 2 * fac * dms[c * nao + e];
                        double jce = 0;
                        /* K buffers transposed (kri[col di + i0], krb[col dj + j0]) and D_ac = D_ca
                         * (symmetric): the i0 loop runs over contiguous memory */
                        double *Kie = kri + e * di, *Kic = kri + c * di;
                        const double *Dc = dms + c * nao + a0, *De = dms + e * nao + a0;
                        for (j0 = 0; j0 < dj; j0++) {
                                const double *g = gk + (size_t)j0 * di;
                                const double *Db = dms + (b0 + j0) * nao;
                                const double dbc = fac * Db[c], dbe = fac * Db[e];
                                double kbe = 0, kbc = 0, jab = 0;
                                if (do_j) {
                                        const double *Da = dms + b0 + j0;
                                        for (i0 = 0; i0 < di; i0++) {
                                                jij[i0 * dj + j0] += g[i0] * dce;
                                                jab += g[i0] * Da[(a0 + i0) * nao];
                                        }
                                }
                                if (do_k) {
                                        /* Kie == Kic for c == e */
                                        for (i0 = 0; i0 < di; i0++) {
                                                Kie[i0] += g[i0] * dbc;
                                                Kic[i0] += g[i0] * dbe;
                                                kbe += g[i0] * Dc[i0];
                                                kbc += g[i0] * De[i0];
                                        }
                                }
                                jce += jab;
                                if (do_k) {
                                        krb[e * dj + j0] += fac * kbe;
                                        krb[c * dj + j0] += fac * kbc;
                                }
                        }
                        if (do_j) jloc[c * nao + e] += 2 * fac * jce;
                } }
                return;
        }
        for (l0 = 0; l0 < dl; l0++) {
        for (k0 = 0; k0 < dk; k0++) {
        for (j0 = 0; j0 < dj; j0++) {
        for (i0 = 0; i0 < di; i0++) {
                double g = buf[((l0 * dk + k0) * dj + j0) * di + i0] * fac;
                if (g == 0) continue;
                size_t a = a0 + i0, b = b0 + j0;
                size_t c = ao_loc[ksh] + k0, e = ao_loc[lsh] + l0;
                if (do_j) {
                        double g2 = 2 * g;
                        double *v0 = jij + ((size_t)i0 * dj + j0) * ndm;
                        const double *d0 = dms + (c * nao + e) * ndm;
                        double *v1 = jloc + (c * nao + e) * ndm;
                        const double *d1 = dms + (a * nao + b) * ndm;
#pragma omp simd
                        for (m = 0; m < ndm; m++) {
                                v0[m] += g2 * d0[m];
                                v1[m] += g2 * d1[m];
                        }
                }
                if (do_k) {
                        /* K_ae += D_bc, K_be += D_ac, K_ac += D_be, K_bc += D_ae */
                        double *v0 = kri + ((size_t)i0 * ncol + e) * ndm;
                        double *v1 = krb + ((size_t)j0 * ncol + e) * ndm;
                        double *v2 = kri + ((size_t)i0 * ncol + c) * ndm;
                        double *v3 = krb + ((size_t)j0 * ncol + c) * ndm;
                        const double *d0 = dms + (b * nao + c) * ndm;
                        const double *d1 = dms + (a * nao + c) * ndm;
                        const double *d2 = dms + (b * nao + e) * ndm;
                        const double *d3 = dms + (a * nao + e) * ndm;
#pragma omp simd
                        for (m = 0; m < ndm; m++) {
                                v0[m] += g * d0[m];
                                v1[m] += g * d1[m];
                                v2[m] += g * d2[m];
                                v3[m] += g * d3[m];
                        }
                }
        } } } }
}

/* a lane batch: nl quartets (ish jsh | ks[l] ls[l]), flags fl[l] = do_j | do_k << 1 | (ij == kl) << 2 */
static int jk_run_lanes(double *lbuf, size_t nbuf, int ish, int jsh, const int *ks, const int *ls, const int *fl,
                        int nl, double dq, const GenCtx *gcx, double prim_tol, double *swk,
                        const int *atm, const int *bas, const double *env, int ndm, size_t nao, const int *ao_loc,
                        const double *dms, double *jij, double *jloc, double *kri, double *krb, size_t ncol)
{
        if (!SOCVHFgen_int2e_lanes(lbuf, nbuf, ish, jsh, ks, ls, nl, gcx, dq, prim_tol, swk, atm, bas, env))
                return 0;
        int l;
        for (l = 0; l < nl; l++)
                jk_digest(lbuf + l * nbuf, ish, jsh, ks[l], ls[l], (fl[l] >> 2) & 1, fl[l] & 1, (fl[l] >> 1) & 1,
                          ndm, nao, ao_loc, dms, jij, jloc, kri, krb, ncol);
        return 1;
}

/*
 * Semi-direct integral store (Haser, Ahlrichs, J. Comput. Chem. 10, 104 (1989)):
 * the shell quartets whose integrals cost the most per stored double kept in
 * memory and reused by the later builds of the same molecule (the SCF cycles,
 * the CP-SCF responses), exact.  The cost per integral of each class of
 * quartets (shell types of the four shells, SOCVHFgen_shell_types) is measured
 * in the builds made while the store is in state 0; SOCVHFjk_store_plan then
 * picks the classes by measured cost per double, most expensive first, within
 * a budget, over the quartets passing q_ij q_kl D >= tol with the density
 * bounds dcond of a full density (state 1); the next build computes them (with
 * the primitive screening of the full density at tol: dq = 2 max dcond,
 * prim_tol = tol 1e-2, as accurate as any later build at a cutoff >= tol
 * needs) and keeps them (state 2); every build from then on reads them.
 */
#if defined(__x86_64__)
#include <x86intrin.h>
#define JK_TICKS() ((double)__rdtsc())
#else
#define JK_TICKS() (omp_get_wtime() * 1e9)
#endif

typedef struct JKStore {
        int state;              /* 0 measuring, 1 planned (the next build fills), 2 filled */
        int nbas, ntype, npair;
        size_t ncls;
        int *stype, *tdim;
        double *ctime, *ccount; /* per class: ticks and quartets of the measuring builds */
        unsigned char *csel;
        size_t *qoff;           /* npair + 1: the stored quartets of bra pair ij, ascending kl */
        int *qkl;
        size_t *eoff;           /* per stored quartet: its integrals at data + eoff */
        double *data;
        size_t nq, ndata;
        double dq, prim_tol, tol;
        int with_j;
} JKStore;

static inline size_t store_cls(const JKStore *st, int i, int j, int k, int l)
{
        const size_t n = st->ntype;
        return ((st->stype[i] * n + st->stype[j]) * n + st->stype[k]) * n + st->stype[l];
}

JKStore *SOCVHFjk_store_new(int *ao_loc, int *bas, int nbas)
{
        JKStore *st = calloc(1, sizeof(JKStore));
        int i;
        st->nbas = nbas;
        st->npair = nbas * (nbas + 1) / 2;
        st->stype = malloc(sizeof(int) * (nbas > 0 ? nbas : 1));
        st->ntype = SOCVHFgen_shell_types(st->stype, bas, nbas);
        st->tdim = calloc(st->ntype + 1, sizeof(int));
        for (i = 0; i < nbas; i++) st->tdim[st->stype[i]] = ao_loc[i + 1] - ao_loc[i];
        st->ncls = (size_t)st->ntype * st->ntype * st->ntype * st->ntype;
        st->ctime = calloc(st->ncls, sizeof(double));
        st->ccount = calloc(st->ncls, sizeof(double));
        st->csel = calloc(st->ncls, 1);
        return st;
}

static void store_drop(JKStore *st)
{
        free(st->qoff);
        free(st->qkl);
        free(st->eoff);
        free(st->data);
        st->qoff = NULL;
        st->qkl = NULL;
        st->eoff = NULL;
        st->data = NULL;
        st->nq = st->ndata = 0;
}

void SOCVHFjk_store_free(JKStore *st)
{
        if (st == NULL) return;
        store_drop(st);
        free(st->stype);
        free(st->tdim);
        free(st->ctime);
        free(st->ccount);
        free(st->csel);
        free(st);
}

int SOCVHFjk_store_state(JKStore *st) { return st->state; }
double SOCVHFjk_store_mb(JKStore *st) { return st->ndata * 8e-6; }
double SOCVHFjk_store_nquartets(JKStore *st) { return (double)st->nq; }

typedef struct { double v; size_t c; } StoreCls;
static int cmp_store_cls(const void *x, const void *y)
{
        const double a = ((const StoreCls *)x)->v, b = ((const StoreCls *)y)->v;
        return a > b ? -1 : a < b ? 1 : 0;
}

static inline int store_pass(const double *qcond, const double *dcond, int nbas, int with_j, double tol,
                             int ish, int jsh, int ksh, int lsh)
{
        const double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
        double d = dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                         dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh]));
        if (with_j) d = dmax2(d, dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]));
        return q4 * d >= tol;
}

/* choose and lay out the stored quartets (state 0 -> 1); returns the MB to be stored */
double SOCVHFjk_store_plan(JKStore *st, double budget_mb, double *qcond, double *dcond, double tol, int with_j)
{
        const int nbas = st->nbas, npair = st->npair;
        const size_t ncls = st->ncls;
        size_t c;
        int ij;
        store_drop(st);
        double *ecls = calloc(ncls, sizeof(double));
        /* the doubles of every class over the quartets passing the screening */
#pragma omp parallel
{
        double *e = calloc(ncls, sizeof(double));
        int ish, jsh, ksh, lsh, kl;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                if (qcond[ish * nbas + jsh] == 0) continue;
                for (kl = 0, ksh = 0, lsh = 0; kl <= ij; kl++, (lsh == ksh ? (ksh++, lsh = 0) : lsh++)) {
                        if (!store_pass(qcond, dcond, nbas, with_j, tol, ish, jsh, ksh, lsh)) continue;
                        const size_t cl = store_cls(st, ish, jsh, ksh, lsh);
                        e[cl] += (double)st->tdim[st->stype[ish]] * st->tdim[st->stype[jsh]]
                               * st->tdim[st->stype[ksh]] * st->tdim[st->stype[lsh]];
                }
        }
#pragma omp critical
        for (c = 0; c < ncls; c++) ecls[c] += e[c];
        free(e);
}
        /* classes by measured cost per double, greedily within the budget */
        StoreCls *order = malloc(sizeof(StoreCls) * (ncls > 0 ? ncls : 1));
        size_t nc = 0;
        for (c = 0; c < ncls; c++) {
                st->csel[c] = 0;
                if (st->ccount[c] > 0 && ecls[c] > 0) {
                        const double nel = ecls[c] / 1.;
                        const size_t i = c / ((size_t)st->ntype * st->ntype * st->ntype);
                        const size_t j = c / ((size_t)st->ntype * st->ntype) % st->ntype;
                        const size_t k = c / st->ntype % st->ntype, l = c % st->ntype;
                        const double nf4 = (double)st->tdim[i] * st->tdim[j] * st->tdim[k] * st->tdim[l];
                        order[nc].v = st->ctime[c] / (st->ccount[c] * nf4);
                        order[nc].c = c;
                        nc++;
                        (void)nel;
                }
        }
        qsort(order, nc, sizeof(StoreCls), cmp_store_cls);
        /* whole classes while they fit, then the next one in part (bra pairs in order) */
        double left = budget_mb * 1e6 / 8;
        for (c = 0; c < nc; c++) {
                const size_t cl = order[c].c;
                if (ecls[cl] <= left) {
                        st->csel[cl] = 1;
                        left -= ecls[cl];
                } else {
                        st->csel[cl] = 2;
                        break;
                }
        }
        free(order);
        free(ecls);
        /* per bra pair: quartets and doubles, then the layout */
        size_t *nqp = calloc(npair + 1, sizeof(size_t)), *nep = calloc(npair + 1, sizeof(size_t));
        size_t *nqx = calloc(npair + 1, sizeof(size_t)), *nex = calloc(npair + 1, sizeof(size_t));
        unsigned char *incl = calloc(npair + 1, 1);
#pragma omp parallel
{
        int ish, jsh, ksh, lsh, kl;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                if (qcond[ish * nbas + jsh] == 0) continue;
                size_t nq = 0, ne = 0, nqp_ = 0, nep_ = 0;
                for (kl = 0, ksh = 0, lsh = 0; kl <= ij; kl++, (lsh == ksh ? (ksh++, lsh = 0) : lsh++)) {
                        const int sel = st->csel[store_cls(st, ish, jsh, ksh, lsh)];
                        if (!sel) continue;
                        if (!store_pass(qcond, dcond, nbas, with_j, tol, ish, jsh, ksh, lsh)) continue;
                        const size_t nf = (size_t)st->tdim[st->stype[ish]] * st->tdim[st->stype[jsh]]
                                        * st->tdim[st->stype[ksh]] * st->tdim[st->stype[lsh]];
                        if (sel == 1) {
                                nq++;
                                ne += nf;
                        } else {
                                nqp_++;
                                nep_ += nf;
                        }
                }
                nqp[ij] = nq;
                nep[ij] = ne;
                nqx[ij] = nqp_;
                nex[ij] = nep_;
        }
}
        /* the part of the last class: bra pairs in order while the budget lasts */
        for (ij = 0; ij < npair; ij++) {
                if (nex[ij] > 0 && nex[ij] <= left) {
                        incl[ij] = 1;
                        left -= nex[ij];
                        nqp[ij] += nqx[ij];
                        nep[ij] += nex[ij];
                }
        }
        free(nqx);
        free(nex);
        st->qoff = malloc(sizeof(size_t) * (npair + 1));
        size_t *doff = malloc(sizeof(size_t) * (npair + 1));
        st->qoff[0] = doff[0] = 0;
        for (ij = 0; ij < npair; ij++) {
                st->qoff[ij + 1] = st->qoff[ij] + nqp[ij];
                doff[ij + 1] = doff[ij] + nep[ij];
        }
        st->nq = st->qoff[npair];
        st->ndata = doff[npair];
        st->qkl = malloc(sizeof(int) * (st->nq + 1));
        st->eoff = malloc(sizeof(size_t) * (st->nq + 1));
        st->data = malloc(sizeof(double) * (st->ndata + 1));
        if (st->data == NULL || st->qkl == NULL || st->eoff == NULL) {
                store_drop(st);
                free(nqp);
                free(nep);
                free(doff);
                free(incl);
                return -1;
        }
#pragma omp parallel
{
        int ish, jsh, ksh, lsh, kl;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < npair; ij++) {
                if (nqp[ij] == 0) continue;
                shell_pair(ij, &ish, &jsh);
                size_t q = st->qoff[ij], e = doff[ij];
                for (kl = 0, ksh = 0, lsh = 0; kl <= ij; kl++, (lsh == ksh ? (ksh++, lsh = 0) : lsh++)) {
                        const int sel = st->csel[store_cls(st, ish, jsh, ksh, lsh)];
                        if (!sel || (sel == 2 && !incl[ij])) continue;
                        if (!store_pass(qcond, dcond, nbas, with_j, tol, ish, jsh, ksh, lsh)) continue;
                        st->qkl[q] = kl;
                        st->eoff[q] = e;
                        q++;
                        e += (size_t)st->tdim[st->stype[ish]] * st->tdim[st->stype[jsh]]
                           * st->tdim[st->stype[ksh]] * st->tdim[st->stype[lsh]];
                }
        }
}
        free(nqp);
        free(nep);
        free(doff);
        free(incl);
        double dm = 0;
        size_t n2 = (size_t)nbas * nbas, i;
        for (i = 0; i < n2; i++) dm = dmax2(dm, dcond[i]);
        st->dq = 2 * dm;
        st->prim_tol = tol * 1e-2;
        st->tol = tol;
        st->with_j = with_j;
        st->state = 1;
        return st->ndata * 8e-6;
}

/* one quartet by the engines of jk_multi_sym (no lanes) into buf; 0 if it vanishes */
static int jk_eval1(double *buf, int *shls, double dq, double prim_tol, int use_ss, const SSCtx *scx,
                    int use_gen, const GenCtx *gcx, double *swk, double *cache, CINTOpt *opt,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        const int l0 = bas[shls[0] * BAS_SLOTS + ANG_OF], l1 = bas[shls[1] * BAS_SLOTS + ANG_OF];
        const int l2 = bas[shls[2] * BAS_SLOTS + ANG_OF], l3 = bas[shls[3] * BAS_SLOTS + ANG_OF];
        const int lq = l0 + l1 + l2 + l3;
        const int hasf = l0 == 3 || l1 == 3 || l2 == 3 || l3 == 3;
        if (use_ss && lq <= 1 && SOCVHFss_takes(shls, bas))
                return SOCVHFss_int2e(buf, shls, scx, dq, prim_tol, swk);
        if (use_gen && (hasf ? lq <= jk_gen_f_lmax : lq <= jk_gen_lmax) && SOCVHFgen_takes(shls, bas))
                return SOCVHFgen_int2e(buf, shls, gcx, dq, prim_tol, swk, bas);
        return int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache);
}

/*
 * The same for symmetric densities (D_cd = D_dc; CPHF): J and K are then
 * symmetric, the eight permutations collapse to two J and four K updates per
 * integral, W += ..., and J = W + W^T at the end.  J and K are screened
 * separately (J by the densities on (ij), (kl), K by those on (ik), (il),
 * (jk), (jl)).  The K rows of the bra shells i, j (columns < ao_loc[i+1],
 * since l <= k <= i) and J_ij are accumulated in local buffers over the ket
 * loop and added to the thread's matrices once per bra pair; the thread
 * matrices are reduced in parallel.
 */
void SOCVHFjk_multi_sym_store(double *vj, double *vk, double *dms, int ndm,
                              int with_j, int with_k,
                              double *qcond, double *dcond, double tol, CINTOpt *opt,
                              int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env, JKStore *st);

void SOCVHFjk_multi_sym(double *vj, double *vk, double *dms, int ndm,
                        int with_j, int with_k,
                        double *qcond, double *dcond, double tol, CINTOpt *opt,
                        int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        SOCVHFjk_multi_sym_store(vj, vk, dms, ndm, with_j, with_k, qcond, dcond, tol, opt,
                                 ao_loc, atm, natm, bas, nbas, env, NULL);
}

/* SOCVHFjk_multi_sym with the semi-direct store st (NULL: none): state 0 measures the class costs,
 * 1 computes and keeps the planned quartets, 2 reads them */
void SOCVHFjk_multi_sym_store(double *vj, double *vk, double *dms, int ndm,
                              int with_j, int with_k,
                              double *qcond, double *dcond, double tol, CINTOpt *opt,
                              int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env, JKStore *st)
{
        const size_t nao = ao_loc[nbas];
        const size_t nv = nao * nao * ndm;
        int dmax = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        size_t cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)dmax * dmax * dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        const int nthmax = omp_get_max_threads();
        double **jbufs = calloc(nthmax, sizeof(double *));
        double **kbufs = calloc(nthmax, sizeof(double *));
        int nth_used = 1;
        const int use_ss = jk_use_ssss;
        SSCtx *scx = use_ss ? SOCVHFss_ctx_build(atm, natm, bas, nbas, env) : NULL;
        const int use_gen = jk_use_gen;
        GenCtx *gcx = use_gen ? SOCVHFgen_ctx_build(atm, natm, bas, nbas, env) : NULL;
        size_t sswk = use_ss ? SOCVHFss_wksize(bas, nbas) : 0;
        if (use_gen && SOCVHFgen_wksize(bas, nbas) > sswk) sswk = SOCVHFgen_wksize(bas, nbas);
        const int use_lanes = use_gen && jk_use_lanes;
        const int jonly = use_gen && jk_use_jonly && with_j && !with_k && ndm == 1;
        if (use_lanes && SOCVHFgen_lane_wksize(bas, nbas, nbuf) > sswk) sswk = SOCVHFgen_lane_wksize(bas, nbas, nbuf);
        int *stype = malloc(sizeof(int) * (nbas > 0 ? nbas : 1));
        const int ntype = SOCVHFgen_shell_types(stype, bas, nbas);
        const int nbk = use_lanes ? ntype * ntype : 1;
        const double prim_tol = tol * 1e-2;
        const int onx = jk_use_onx && with_k && !with_j;
        ONXLists *onxl = onx ? SOCVHFonx_build(qcond, dcond, nbas) : NULL;
        const int measure = st != NULL && st->state == 0;
        const int stored = st != NULL && st->state >= 1 && st->nq > 0;
        const int fill = stored && st->state == 1;
#pragma omp parallel
{
        double *mt = measure ? calloc(st->ncls, sizeof(double)) : NULL;
        double *mc = measure ? calloc(st->ncls, sizeof(double)) : NULL;
        const int tid = omp_get_thread_num();
#pragma omp single
        nth_used = omp_get_num_threads();
        ONXScratch *onxs = onx ? SOCVHFonx_scratch(nbas) : NULL;
        int *cand = onx ? malloc(sizeof(int) * (npair + 1)) : NULL;
        double *jloc = with_j ? calloc(nv, sizeof(double)) : NULL;
        double *kloc = with_k ? calloc(nv, sizeof(double)) : NULL;
        jbufs[tid] = jloc;
        kbufs[tid] = kloc;
        double *buf = malloc(sizeof(double) * (nbuf + cache_size + sswk));
        double *cache = buf + nbuf;
        double *swk = cache + cache_size;
        /* per ket type a lane batch of the row */
        double *lbuf = use_lanes ? malloc(sizeof(double) * nbuf * JK_NL) : NULL;
        int *bkk = malloc(sizeof(int) * nbk * JK_NL), *bkl = malloc(sizeof(int) * nbk * JK_NL);
        int *bkf = malloc(sizeof(int) * nbk * JK_NL), *bkn = calloc(nbk, sizeof(int));
        double *bkd = calloc(nbk, sizeof(double));
        double *kri = with_k ? malloc(sizeof(double) * dmax * nao * ndm) : NULL;
        double *krj = with_k ? malloc(sizeof(double) * dmax * nao * ndm) : NULL;
        double *jij = with_j ? malloc(sizeof(double) * dmax * dmax * ndm) : NULL;
        int it, ij, kl, ish, jsh, ksh, lsh, m;
        int shls[4];
        /* the work of pair ij grows with ij (kl <= ij): largest first */
#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < npair; it++) {
                ij = npair - 1 - it;
                shell_pair(ij, &ish, &jsh);
                const int di = ao_loc[ish + 1] - ao_loc[ish];
                const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
                const size_t ncol = ao_loc[ish + 1];
                double *krb = ish == jsh ? kri : krj;     /* rows of shell j */
                int touched = 0;
                if (with_k) {
                        memset(kri, 0, sizeof(double) * di * ncol * ndm);
                        if (ish != jsh) memset(krj, 0, sizeof(double) * dj * ncol * ndm);
                }
                if (with_j) memset(jij, 0, sizeof(double) * di * dj * ndm);
                const int row_lanes = use_lanes && bas[ish * BAS_SLOTS + ANG_OF] <= 2
                                    && bas[jsh * BAS_SLOTS + ANG_OF] <= 2 && bas[ish * BAS_SLOTS + NCTR_OF] <= 4
                                    && bas[jsh * BAS_SLOTS + NCTR_OF] <= 4;
                const int row_np = bas[ish * BAS_SLOTS + NPRIM_OF] * bas[jsh * BAS_SLOTS + NPRIM_OF];
                int kb, ic, nc = ij + 1;
                size_t sp = 0, s1 = 0;
                if (stored) {
                        /* the stored quartets of the pair: computed and kept (fill), else read */
                        size_t s;
                        for (s = st->qoff[ij], s1 = st->qoff[ij + 1]; s < s1; s++) {
                                kl = st->qkl[s];
                                shell_pair(kl, &ksh, &lsh);
                                const double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                                const double dj_ = dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]);
                                const double dk_ = dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                                         dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh]));
                                const int do_j = with_j && q4 * dj_ >= tol;
                                const int do_k = with_k && q4 * dk_ >= tol;
                                double *g = st->data + st->eoff[s];
                                if (fill) {
                                        const size_t nf = (size_t)di * dj * (ao_loc[ksh + 1] - ao_loc[ksh])
                                                        * (ao_loc[lsh + 1] - ao_loc[lsh]);
                                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                                        if (jk_eval1(buf, shls, st->dq, st->prim_tol, use_ss, scx, use_gen, gcx,
                                                     swk, cache, opt, atm, natm, bas, nbas, env))
                                                memcpy(g, buf, sizeof(double) * nf);
                                        else
                                                memset(g, 0, sizeof(double) * nf);
                                }
                                if (!do_j && !do_k) continue;
                                touched = 1;
                                jk_digest(g, ish, jsh, ksh, lsh, ij == kl, do_j, do_k, ndm, nao, ao_loc, dms,
                                          jij, jloc, kri, krb, ncol);
                        }
                        sp = st->qoff[ij];
                }
                if (onx) {
                        const double qij = qcond[ish * nbas + jsh];
                        nc = qij > 0 ? SOCVHFonx_kpairs(onxl, onxs, ish, jsh, ij, tol / qij, dcond, cand) : 0;
                }
                /* (k, l) of kl = k (k + 1) / 2 + l walked incrementally (onx: from the candidates) */
                for (ic = 0, kl = 0, ksh = 0, lsh = 0; ic < nc; ic++) {
                        if (onx) {
                                kl = cand[ic];
                                shell_pair(kl, &ksh, &lsh);
                        } else if (ic > 0) {
                                kl = ic;
                                if (lsh == ksh) { ksh++; lsh = 0; } else lsh++;
                        }
                        if (sp < s1) {
                                /* done above */
                                while (sp < s1 && st->qkl[sp] < kl) sp++;
                                if (sp < s1 && st->qkl[sp] == kl) continue;
                        }
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        const double dj_ = dmax2(dcond[ish * nbas + jsh], dcond[ksh * nbas + lsh]);
                        const double dk_ = dmax2(dmax2(dcond[ish * nbas + ksh], dcond[ish * nbas + lsh]),
                                                 dmax2(dcond[jsh * nbas + ksh], dcond[jsh * nbas + lsh]));
                        int do_j = with_j && q4 * dj_ >= tol;
                        int do_k = with_k && q4 * dk_ >= tol;
                        if (!do_j && !do_k) {
                                continue;
                        }
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        const int lq = bas[ish * BAS_SLOTS + ANG_OF] + bas[jsh * BAS_SLOTS + ANG_OF]
                                     + bas[ksh * BAS_SLOTS + ANG_OF] + bas[lsh * BAS_SLOTS + ANG_OF];
                        if (jonly && lq >= 2 && lq <= jk_gen_lmax && SOCVHFgen_takes(shls, bas)) {
                                const int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                                const int dl = ao_loc[lsh + 1] - ao_loc[lsh];
                                const size_t c00 = ao_loc[ksh], e00 = ao_loc[lsh];
                                double *db = buf, *dkk = db + di * dj, *jb = dkk + dk * dl, *jkk = jb + di * dj;
                                int i0, j0;
                                for (j0 = 0; j0 < dj; j0++)
                                        for (i0 = 0; i0 < di; i0++) {
                                                db[j0 * di + i0] = dms[(a0 + i0) * nao + b0 + j0];
                                                jb[j0 * di + i0] = 0;
                                        }
                                for (j0 = 0; j0 < dl; j0++)
                                        for (i0 = 0; i0 < dk; i0++) {
                                                dkk[j0 * dk + i0] = dms[(c00 + i0) * nao + e00 + j0];
                                                jkk[j0 * dk + i0] = 0;
                                        }
                                const int r = SOCVHFgen_j2e(jb, jkk, db, dkk, shls, gcx, dj_, prim_tol, swk);
                                if (r == 0) continue;
                                if (r == 1) {
                                        touched = 1;
                                        double fac = 2.;
                                        if (ish == jsh) fac *= .5;
                                        if (ksh == lsh) fac *= .5;
                                        if (ij == kl) fac *= .5;
                                        for (j0 = 0; j0 < dj; j0++)
                                                for (i0 = 0; i0 < di; i0++) jij[i0 * dj + j0] += fac * jb[j0 * di + i0];
                                        for (j0 = 0; j0 < dl; j0++)
                                                for (i0 = 0; i0 < dk; i0++)
                                                        jloc[(c00 + i0) * nao + e00 + j0] += fac * jkk[j0 * dk + i0];
                                        continue;
                                }
                                /* r < 0: no J form for the class, the full integrals below */
                        }
                        const int hasf = bas[ish * BAS_SLOTS + ANG_OF] == 3 || bas[jsh * BAS_SLOTS + ANG_OF] == 3
                                      || bas[ksh * BAS_SLOTS + ANG_OF] == 3 || bas[lsh * BAS_SLOTS + ANG_OF] == 3;
                        if (row_lanes && lq >= jk_lane_lmin && lq <= jk_gen_lmax
                            && row_np * bas[ksh * BAS_SLOTS + NPRIM_OF] * bas[lsh * BAS_SLOTS + NPRIM_OF] <= jk_lane_np
                            && SOCVHFgen_lane_ok(ksh, lsh, bas)) {
                                const int kb = stype[ksh] * ntype + stype[lsh];
                                const int n_ = kb * JK_NL + bkn[kb];
                                bkk[n_] = ksh;
                                bkl[n_] = lsh;
                                bkf[n_] = do_j | do_k << 1 | (ij == kl) << 2;
                                bkd[kb] = dmax2(bkd[kb], dmax2(do_j ? dj_ : 0, do_k ? dk_ : 0));
                                if (++bkn[kb] == JK_NL) {
                                        const double t0 = measure ? JK_TICKS() : 0;
                                        touched |= jk_run_lanes(lbuf, nbuf, ish, jsh, bkk + kb * JK_NL,
                                                                bkl + kb * JK_NL, bkf + kb * JK_NL, JK_NL, bkd[kb],
                                                                gcx, prim_tol, swk, atm, bas, env, ndm, nao, ao_loc,
                                                                dms, jij, jloc, kri, krb, ncol);
                                        if (measure) {
                                                const double dt = (JK_TICKS() - t0) / JK_NL;
                                                int n;
                                                for (n = 0; n < JK_NL; n++) {
                                                        const size_t cl = store_cls(st, ish, jsh, bkk[kb * JK_NL + n],
                                                                                    bkl[kb * JK_NL + n]);
                                                        mt[cl] += dt;
                                                        mc[cl] += 1;
                                                }
                                        }
                                        bkn[kb] = 0;
                                        bkd[kb] = 0;
                                }
                                continue;
                        }
                        const double t0 = measure ? JK_TICKS() : 0;
                        int ok;
                        if (use_ss && lq <= 1 && SOCVHFss_takes(shls, bas)) {
                                const double dq = dmax2(do_j ? dj_ : 0, do_k ? dk_ : 0);
                                ok = SOCVHFss_int2e(buf, shls, scx, dq, prim_tol, swk);
                        } else if (use_gen && (hasf ? lq <= jk_gen_f_lmax : lq <= jk_gen_lmax)
                                   && SOCVHFgen_takes(shls, bas)) {
                                const double dq = dmax2(do_j ? dj_ : 0, do_k ? dk_ : 0);
                                ok = SOCVHFgen_int2e(buf, shls, gcx, dq, prim_tol, swk, bas);
                        } else {
                                ok = int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache);
                        }
                        if (measure) {
                                const size_t cl = store_cls(st, ish, jsh, ksh, lsh);
                                mt[cl] += JK_TICKS() - t0;
                                mc[cl] += 1;
                        }
                        if (!ok) continue;
                        touched = 1;
                        jk_digest(buf, ish, jsh, ksh, lsh, ij == kl, do_j, do_k, ndm, nao, ao_loc, dms,
                                  jij, jloc, kri, krb, ncol);
                }
                /* the lane batches left of the row */
                for (kb = 0; row_lanes && kb < nbk; kb++) {
                        if (bkn[kb] == 0) continue;
                        const double t0 = measure ? JK_TICKS() : 0;
                        touched |= jk_run_lanes(lbuf, nbuf, ish, jsh, bkk + kb * JK_NL, bkl + kb * JK_NL,
                                                bkf + kb * JK_NL, bkn[kb], bkd[kb], gcx, prim_tol, swk,
                                                atm, bas, env, ndm, nao, ao_loc, dms, jij, jloc, kri, krb, ncol);
                        if (measure) {
                                const double dt = (JK_TICKS() - t0) / bkn[kb];
                                int n;
                                for (n = 0; n < bkn[kb]; n++) {
                                        const size_t cl = store_cls(st, ish, jsh, bkk[kb * JK_NL + n],
                                                                    bkl[kb * JK_NL + n]);
                                        mt[cl] += dt;
                                        mc[cl] += 1;
                                }
                        }
                        bkn[kb] = 0;
                        bkd[kb] = 0;
                }
                if (!touched) {
                        continue;
                }
                /* flush the bra-pair buffers */
                int i0, j0;
                size_t q;
                if (with_j) {
                        for (i0 = 0; i0 < di; i0++)
                                for (j0 = 0; j0 < dj; j0++) {
                                        double *o = jloc + ((a0 + i0) * nao + b0 + j0) * ndm;
                                        double *src = jij + ((size_t)i0 * dj + j0) * ndm;
                                        for (m = 0; m < ndm; m++) o[m] += src[m];
                                }
                }
                if (with_k && ndm == 1) {
                        /* the transposed buffers of jk_digest's one-density form */
                        for (q = 0; q < ncol; q++) {
                                for (i0 = 0; i0 < di; i0++) kloc[(a0 + i0) * nao + q] += kri[q * di + i0];
                                if (ish != jsh)
                                        for (j0 = 0; j0 < dj; j0++) kloc[(b0 + j0) * nao + q] += krj[q * dj + j0];
                        }
                } else if (with_k) {
                        for (i0 = 0; i0 < di; i0++) {
                                double *o = kloc + (a0 + i0) * nao * ndm;
                                double *src = kri + (size_t)i0 * ncol * ndm;
                                for (q = 0; q < ncol * ndm; q++) o[q] += src[q];
                        }
                        if (ish != jsh) {
                                for (j0 = 0; j0 < dj; j0++) {
                                        double *o = kloc + (b0 + j0) * nao * ndm;
                                        double *src = krj + (size_t)j0 * ncol * ndm;
                                        for (q = 0; q < ncol * ndm; q++) o[q] += src[q];
                                }
                        }
                }
        }
#pragma omp barrier
        if (with_j) reduce_thread_buffers(vj, jbufs, nth_used, nv);
        if (with_k) reduce_thread_buffers(vk, kbufs, nth_used, nv);
#pragma omp barrier
        free(jloc);
        free(kloc);
        free(buf);
        free(kri);
        free(krj);
        free(jij);
        free(lbuf);
        free(bkk);
        free(bkl);
        free(bkf);
        free(bkn);
        free(bkd);
        free(cand);
        SOCVHFonx_scratch_free(onxs);
        if (measure) {
                size_t c;
#pragma omp critical
                for (c = 0; c < st->ncls; c++) {
                        st->ctime[c] += mt[c];
                        st->ccount[c] += mc[c];
                }
        }
        free(mt);
        free(mc);
}
        if (fill) st->state = 2;
        SOCVHFonx_free(onxl);
        SOCVHFss_ctx_free(scx);
        SOCVHFgen_ctx_free(gcx);
        free(stype);
        free(jbufs);
        free(kbufs);
        /* V = W + W^T over the two AO indices */
        size_t p;
#pragma omp parallel for schedule(dynamic, 4)
        for (p = 0; p < nao; p++) {
                size_t q;
                int m;
                for (q = 0; q <= p; q++) {
                        if (with_j) {
                                double *x = vj + (p * nao + q) * ndm;
                                double *y = vj + (q * nao + p) * ndm;
                                for (m = 0; m < ndm; m++) {
                                        double s = x[m] + y[m];
                                        x[m] = s;
                                        y[m] = s;
                                }
                        }
                        if (with_k) {
                                double *x = vk + (p * nao + q) * ndm;
                                double *y = vk + (q * nao + p) * ndm;
                                for (m = 0; m < ndm; m++) {
                                        double s = x[m] + y[m];
                                        x[m] = s;
                                        y[m] = s;
                                }
                        }
                }
        }
}

/*
 * The s8 ERIs (pyscf's mol.intor('int2e', aosym='s8') layout: eri[ab (ab + 1) / 2 + cd],
 * ab = a (a + 1) / 2 + b, a >= b, cd likewise, ab >= cd) by the engines of
 * SOCVHFjk_multi_sym (jk_ssss.c, jk_gen.c per quartet, libcint for the rest), the
 * quartets with q_ij q_kl < tol left zero (eri zeroed by the caller); for the in-core
 * J / K of small molecules (scf/md_j.py INCORE).  Shells of the unique quartets in
 * parallel over ij: every element is written by one quartet.
 */
void SOCVHFjk_eri_s8(double *eri, double *qcond, double tol, CINTOpt *opt,
                     int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        int dmax = 0, i;
        for (i = 0; i < nbas; i++) {
                int d = ao_loc[i + 1] - ao_loc[i];
                dmax = d > dmax ? d : dmax;
        }
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        const size_t cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4, atm, natm, bas, nbas, env);
        const size_t nbuf = (size_t)dmax * dmax * dmax * dmax;
        const int npair = nbas * (nbas + 1) / 2;
        SSCtx *scx = jk_use_ssss ? SOCVHFss_ctx_build(atm, natm, bas, nbas, env) : NULL;
        GenCtx *gcx = jk_use_gen ? SOCVHFgen_ctx_build(atm, natm, bas, nbas, env) : NULL;
        size_t sswk = scx ? SOCVHFss_wksize(bas, nbas) : 0;
        if (gcx && SOCVHFgen_wksize(bas, nbas) > sswk) sswk = SOCVHFgen_wksize(bas, nbas);
        const double prim_tol = tol * 1e-2;
#pragma omp parallel
{
        double *buf = malloc(sizeof(double) * (nbuf + cache_size + sswk));
        double *cache = buf + nbuf, *swk = cache + cache_size;
        int it, ij, kl, ish, jsh, ksh, lsh;
        int shls[4];
#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < npair; it++) {
                ij = npair - 1 - it;
                shell_pair(ij, &ish, &jsh);
                const int di = ao_loc[ish + 1] - ao_loc[ish], dj = ao_loc[jsh + 1] - ao_loc[jsh];
                const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
                for (kl = 0, ksh = 0, lsh = 0; kl <= ij; kl++, (lsh == ksh ? (ksh++, lsh = 0) : lsh++)) {
                        if (qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh] < tol) continue;
                        shls[0] = ish; shls[1] = jsh; shls[2] = ksh; shls[3] = lsh;
                        const int lq = bas[ish * BAS_SLOTS + ANG_OF] + bas[jsh * BAS_SLOTS + ANG_OF]
                                     + bas[ksh * BAS_SLOTS + ANG_OF] + bas[lsh * BAS_SLOTS + ANG_OF];
                        const int hasf = bas[ish * BAS_SLOTS + ANG_OF] == 3 || bas[jsh * BAS_SLOTS + ANG_OF] == 3
                                      || bas[ksh * BAS_SLOTS + ANG_OF] == 3 || bas[lsh * BAS_SLOTS + ANG_OF] == 3;
                        int ok;
                        if (scx && lq <= 1 && SOCVHFss_takes(shls, bas)) {
                                ok = SOCVHFss_int2e(buf, shls, scx, 1., prim_tol, swk);
                        } else if (gcx && (hasf ? lq <= jk_gen_f_lmax : lq <= jk_gen_lmax)
                                   && SOCVHFgen_takes(shls, bas)) {
                                ok = SOCVHFgen_int2e(buf, shls, gcx, 1., prim_tol, swk, bas);
                        } else {
                                ok = int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache);
                        }
                        if (!ok) continue;
                        const int dk = ao_loc[ksh + 1] - ao_loc[ksh], dl = ao_loc[lsh + 1] - ao_loc[lsh];
                        const size_t c0 = ao_loc[ksh], e0 = ao_loc[lsh];
                        int i0, j0, k0, l0;
                        for (l0 = 0; l0 < dl; l0++)
                        for (k0 = 0; k0 < dk; k0++) {
                                const size_t c = c0 + k0, e = e0 + l0;
                                if (ksh == lsh && e > c) continue;
                                const size_t cd = c * (c + 1) / 2 + e;
                                const double *g = buf + (size_t)(l0 * dk + k0) * dj * di;
                                for (j0 = 0; j0 < dj; j0++)
                                for (i0 = 0; i0 < di; i0++) {
                                        const size_t a = a0 + i0, b = b0 + j0;
                                        if (ish == jsh && b > a) continue;
                                        const size_t ab = a * (a + 1) / 2 + b;
                                        if (ij == kl && cd > ab) continue;
                                        /* ab < cd happens for ish == ksh, jsh > lsh */
                                        const size_t p = ab > cd ? ab : cd, q = ab > cd ? cd : ab;
                                        eri[p * (p + 1) / 2 + q] = g[j0 * di + i0];
                                }
                        }
                }
        }
        free(buf);
}
        SOCVHFss_ctx_free(scx);
        SOCVHFgen_ctx_free(gcx);
}
