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
 * jk_gen.c, the rest by libcint (on par with the generated ones from L = 6).
 * Of the generated ones those of few primitives go in batches of GL_NL
 * quartets sharing the bra pair and the ket shell types to the lane kernels
 * (SOCVHFjk_set_lanes, SOCVHFjk_set_lane_np below).
 * SOCVHFjk_set_ssss(0) / SOCVHFjk_set_gen(0) send theirs to libcint.
 */
static int jk_use_ssss = 1, jk_use_gen = 1, jk_gen_lmax = 5, jk_gen_f_lmax = 6;
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

/* lanes of jk_gen.c: GL_NL quartets sharing the bra pair and the ket shell types per call */
int SOCVHFgen_shell_types(int *type, const int *bas, int nbas);
int SOCVHFgen_lane_ok(int ksh, int lsh, const int *bas);
size_t SOCVHFgen_lane_wksize(const int *bas, int nbas, int nbuf);
int SOCVHFgen_int2e_lanes(double *out, size_t ostride, int ish, int jsh, const int *ksh, const int *lsh, int nl,
                          const GenCtx *cx, double dmax, double prim_tol, double *wk,
                          const int *atm, const int *bas, const double *env);
#define JK_NL 8                 /* = GL_NL of jk_gen.h */
/* J-only builds (with_k 0, one density): the generated quartets through the J forms, the density
 * contracted before the horizontal recursions (0: the full integrals and jk_digest) */
int SOCVHFgen_j2e(double *jb, double *jk, const double *db, const double *dk, const int *shls,
                  const GenCtx *cx, double dmax, double prim_tol, double *wk);
static int jk_use_jonly = 1;
void SOCVHFjk_set_jonly(int on) { jk_use_jonly = on; }
/* the quartets of l <= 2 with jk_lane_lmin <= L <= jk_gen_lmax and at most jk_lane_np primitive
 * quartets (product of the shells' nprim) go to the lanes (on 0: never); the per-quartet kernels
 * vectorise over the bra primitives and screen them, better with many (bench_jk_class.py) */
static int jk_use_lanes = 1, jk_lane_lmin = 2, jk_lane_np = 4;
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
                        for (j0 = 0; j0 < dj; j0++) {
                                const double *g = gk + (size_t)j0 * di;
                                const double *Db = dms + (b0 + j0) * nao;
                                const double dbc = fac * Db[c], dbe = fac * Db[e];
                                double kbe = 0, kbc = 0, jab = 0;
                                for (i0 = 0; i0 < di; i0++) {
                                        const double *Da = dms + (a0 + i0) * nao;
                                        if (do_j) {
                                                jij[i0 * dj + j0] += g[i0] * dce;
                                                jab += g[i0] * Da[b0 + j0];
                                        }
                                        if (do_k) {
                                                double *Ki = kri + (size_t)i0 * ncol;
                                                Ki[e] += g[i0] * dbc;
                                                Ki[c] += g[i0] * dbe;
                                                kbe += g[i0] * Da[c];
                                                kbc += g[i0] * Da[e];
                                        }
                                }
                                jce += jab;
                                if (do_k) {
                                        double *Kb = krb + (size_t)j0 * ncol;
                                        Kb[e] += fac * kbe;
                                        Kb[c] += fac * kbc;
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
 * The same for symmetric densities (D_cd = D_dc; CPHF): J and K are then
 * symmetric, the eight permutations collapse to two J and four K updates per
 * integral, W += ..., and J = W + W^T at the end.  J and K are screened
 * separately (J by the densities on (ij), (kl), K by those on (ik), (il),
 * (jk), (jl)).  The K rows of the bra shells i, j (columns < ao_loc[i+1],
 * since l <= k <= i) and J_ij are accumulated in local buffers over the ket
 * loop and added to the thread's matrices once per bra pair; the thread
 * matrices are reduced in parallel.
 */
void SOCVHFjk_multi_sym(double *vj, double *vk, double *dms, int ndm,
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
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
#pragma omp single
        nth_used = omp_get_num_threads();
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
                int kb;
                /* (k, l) of kl = k (k + 1) / 2 + l walked incrementally */
                for (kl = 0, ksh = 0, lsh = 0; kl <= ij;
                     kl++, (lsh == ksh ? (ksh++, lsh = 0) : lsh++)) {
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
                                        touched |= jk_run_lanes(lbuf, nbuf, ish, jsh, bkk + kb * JK_NL,
                                                                bkl + kb * JK_NL, bkf + kb * JK_NL, JK_NL, bkd[kb],
                                                                gcx, prim_tol, swk, atm, bas, env, ndm, nao, ao_loc,
                                                                dms, jij, jloc, kri, krb, ncol);
                                        bkn[kb] = 0;
                                        bkd[kb] = 0;
                                }
                                continue;
                        }
                        if (use_ss && lq <= 1 && SOCVHFss_takes(shls, bas)) {
                                const double dq = dmax2(do_j ? dj_ : 0, do_k ? dk_ : 0);
                                if (!SOCVHFss_int2e(buf, shls, scx, dq, prim_tol, swk)) {
                                        continue;
                                }
                        } else if (use_gen && (hasf ? lq <= jk_gen_f_lmax : lq <= jk_gen_lmax)
                                   && SOCVHFgen_takes(shls, bas)) {
                                const double dq = dmax2(do_j ? dj_ : 0, do_k ? dk_ : 0);
                                if (!SOCVHFgen_int2e(buf, shls, gcx, dq, prim_tol, swk, bas)) {
                                        continue;
                                }
                        } else if (!int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                                continue;
                        }
                        touched = 1;
                        jk_digest(buf, ish, jsh, ksh, lsh, ij == kl, do_j, do_k, ndm, nao, ao_loc, dms,
                                  jij, jloc, kri, krb, ncol);
                }
                /* the lane batches left of the row */
                for (kb = 0; row_lanes && kb < nbk; kb++) {
                        if (bkn[kb] == 0) continue;
                        touched |= jk_run_lanes(lbuf, nbuf, ish, jsh, bkk + kb * JK_NL, bkl + kb * JK_NL,
                                                bkf + kb * JK_NL, bkn[kb], bkd[kb], gcx, prim_tol, swk,
                                                atm, bas, env, ndm, nao, ao_loc, dms, jij, jloc, kri, krb, ncol);
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
                if (with_k) {
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
}
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
