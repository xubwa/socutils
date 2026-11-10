/*
 * Seminumerical exchange (chain-of-spheres, COSX; Neese et al. 2009) for
 * real densities on a spherical basis of l <= CX_LMAX (socutils.scf.cosx):
 *
 *     K_mu nu = sum_g w_g phi_mu(g) G_nu(g),   G_nu(g) = sum_lam A_nu lam(g) F_lam(g),
 *     F_lam(g) = sum_kap D_kap lam phi_kap(g), A_nu lam(g) = int phi_nu phi_lam / |r - r_g|,
 *
 * (the plain K; the overlap fit and the symmetrisation are applied by the
 * caller).  The grid in blocks of up to blk points (multiples of 56, rows of
 * pyscf's non0tab), one block per thread task; per block
 *
 *   1. the AOs significant on the block (lib/dft/gtoval.c), phi (nl, n);
 *   2. F (nao, n) = D[idx, :]^T phi by one GEMM per density, its max per shell;
 *   3. the potential integrals A of every shell pair whose bound
 *      B_IJ max|F_J| sum_g|w phi| passes tol (and/or the mirror), by
 *      McMurchie-Davidson on the points as SIMD loops:
 *          A_ab(g) = sum_k sum_t E^k_ab,t R_t(p_k, P_k - r_g)
 *      (E^k with the contraction coefficients, the cart -> sph transform and
 *      2 pi / p_k folded in, built once in SOCDFTcosx_build), contracted at
 *      once into G (both a <- b and b <- a);
 *   4. K[idx, :] += (w phi) G^T by one GEMM per density.
 *
 * One-centre shell pairs (both shells on one atom, P_k = the atom for all
 * primitive pairs): primitive pairs with p_k |r - A|^2 >= tcut on the whole
 * block are in their far field, (2 pi / p) R_t(p) = pi^{3/2} p^{-3/2} d^t (1/r)
 * to double precision (tcut = 36: e^-T < 3e-16), so they collapse into one
 * multipole term sum_t M_t d^t(1/r), M the prefix sum of the moments of the
 * primitives (sorted by descending p), and d^t(1/r) is shared by all shell
 * pairs of the atom on the block.  This keeps the 81 primitive pairs of a
 * contracted cc-pVDZ s shell to the few diffuse ones near the atom.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
/* -DCX_PROFILE: per-pair counters and cycle counts in SOCDFTcosx_stats */
#ifdef CX_PROFILE
#include <x86intrin.h>
#define PROF(x) x
#else
#define PROF(x)
#endif
#include "gtoval_c2s.h"

#define ATM_SLOTS 6
#define PTR_COORD 1
#define BAS_SLOTS 8
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define PTR_EXP 5
#define PTR_COEFF 6
#define MASK_BLK 56

#define CX_LMAX 3
#define CX_DMAX (2 * CX_LMAX + 2)                       /* pair degree (+ 2 for derivatives) */
#define NH(d) (((d) + 1) * ((d) + 2) * ((d) + 3) / 6)
#define CX_NHMAX NH(CX_DMAX)
#define CX_GEMM_MIN 200                                /* nab nh from which E R is a GEMM */
#define CX_NCMAX 16                                     /* contracted functions per shell */

typedef void (*dgemm_t)(const char *ta, const char *tb, const int *m, const int *n, const int *k,
                        const double *alpha, const double *a, const int *lda, const double *b,
                        const int *ldb, const double *beta, double *c, const int *ldc);
typedef int (*mkl_local_t)(int nt);

void SOCDFTgtoval_ld(int deriv, int ngrids, int sh0, int sh1, int *ao_loc, double *ao, size_t ld, size_t row0,
                     double *coord, uint8_t *non0table, int *atm, int *bas, int nbas, double *env, double *rowmax);

int SOCDFTcosx_lmax() { return CX_LMAX; }

/* profiling counters (SOCDFTcosx_stats): times of the phases summed over threads, counts */
static double cx_stat[32];
void SOCDFTcosx_stats(double *out, int reset)
{
        memcpy(out, cx_stat, sizeof(cx_stat));
        if (reset) memset(cx_stat, 0, sizeof(cx_stat));
}

/* ---------------------------------------------------------------- Hermite tables, Boys */

static int cx_ready = 0;
static double cx_damp[CX_DMAX + 1];
static int cx_full = 1;     /* the all-in-register primitive kernels for L <= cx_full (c16 / def2-TZVP:
                             * L = 0, 1 2.9 / 5.6 cycles per point against 3.7 / 6.1, L = 2 12.5 against 10.4) */
static int cx_pairdamp = 1;  /* the pair-level distance damping */
static int cx_mfloat = 0;
static int cx_err = 0;       /* set by a kernel that ran out of memory; SOCDFTcosx_error reads and clears it */
int SOCDFTcosx_error(void)
{
        const int e = cx_err;
        cx_err = 0;
        return e;
}    /* the multi-density contraction in single precision (AVX-512 builds) */
int SOCDFTcosx_mfloat(void)
{
#ifdef __AVX512F__
        return cx_mfloat;
#else
        return 0;
#endif
}
void SOCDFTcosx_set_full(int f) { cx_full = f; }
void SOCDFTcosx_set_opt(int which, int v)
{
        if (which == 0) cx_full = v;
        if (which == 1) cx_pairdamp = v;
        if (which == 2) cx_mfloat = v;
}
static int h_t[CX_NHMAX], h_u[CX_NHMAX], h_v[CX_NHMAX], h_deg[CX_NHMAX];
static int h_dir[CX_NHMAX], h_m1[CX_NHMAX], h_m2[CX_NHMAX], h_c[CX_NHMAX];
static int h_idx[CX_DMAX + 1][CX_DMAX + 1][CX_DMAX + 1];

#define BOYS_TMAX 40.
#define BOYS_NT 1281                     /* grid t_k = k / 32 */
#define BOYS_MT (CX_DMAX + 8)
static double boys_tab[BOYS_NT][BOYS_MT];
static double boys_inv[BOYS_MT];

static void boys_exact(double *f, double t, int m)
{
        int n;
        const double et = exp(-t);
        if (t < m + 10) {
                double term = 1. / (2 * m + 1), sum = term;
                int k;
                for (k = 1; k < 400; k++) {
                        term *= 2 * t / (2 * m + 2 * k + 1);
                        sum += term;
                        if (term < 1e-17 * sum) break;
                }
                f[m] = et * sum;
                for (n = m; n > 0; n--) f[n - 1] = (2 * t * f[n] + et) / (2 * n - 1);
        } else {
                const double st = sqrt(t);
                f[0] = .886226925452758014 / st * erf(st);
                const double h = .5 / t;
                for (n = 0; n < m; n++) f[n + 1] = h * ((2 * n + 1) * f[n] - et);
        }
}

static void cx_tables(void)
{
#pragma omp critical(socdft_cosx_tables)
{
        if (!cx_ready) {
                int d, t, u, v, n = 0;
                for (d = 0; d <= CX_DMAX; d++)
                        for (t = d; t >= 0; t--)
                                for (u = d - t; u >= 0; u--) {
                                        v = d - t - u;
                                        h_idx[t][u][v] = n;
                                        h_t[n] = t; h_u[n] = u; h_v[n] = v; h_deg[n] = d;
                                        n++;
                                }
                for (n = 1; n < CX_NHMAX; n++) {
                        int c[3] = {h_t[n], h_u[n], h_v[n]}, k = c[0] ? 0 : (c[1] ? 1 : 2);
                        int m1[3] = {c[0], c[1], c[2]}, m2[3] = {c[0], c[1], c[2]};
                        m1[k] -= 1;
                        m2[k] -= 2;
                        h_dir[n] = k;
                        h_m1[n] = h_idx[m1[0]][m1[1]][m1[2]];
                        h_c[n] = c[k] - 1;
                        h_m2[n] = c[k] >= 2 ? h_idx[m2[0]][m2[1]][m2[2]] : 0;
                }
                for (n = 0; n < BOYS_NT; n++) boys_exact(boys_tab[n], n * (1. / 32), BOYS_MT - 1);
                for (n = 1; n < BOYS_MT; n++) boys_inv[n] = 1. / (2 * n - 1);
                {
                        double df = 1;
                        for (n = 0; n <= CX_DMAX; n++) {
                                cx_damp[n] = .886226925452758014 * df / pow(2., .5 * n);
                                df *= 2 * n + 1;
                        }
                }
                cx_ready = 1;
        }
}
}

#define BOYS_TSW (BOYS_TMAX - 1. / 32)      /* table below, asymptotic form from here */

/* out[g] = 1 / sqrt(x[g]): AVX-512 rsqrt14 and two Newton steps (relative error ~1e-16), else libm */
#if defined(__AVX512F__)
#include <immintrin.h>
static void rsqrt_vec(double *out, const double *x, int n)
{
        int g = 0;
        const __m512d half = _mm512_set1_pd(.5), th = _mm512_set1_pd(1.5);
        for (; g + 8 <= n; g += 8) {
                const __m512d v = _mm512_loadu_pd(x + g), h = _mm512_mul_pd(v, half);
                __m512d y = _mm512_rsqrt14_pd(v);
                y = _mm512_mul_pd(y, _mm512_fnmadd_pd(h, _mm512_mul_pd(y, y), th));
                y = _mm512_mul_pd(y, _mm512_fnmadd_pd(h, _mm512_mul_pd(y, y), th));
                _mm512_storeu_pd(out + g, y);
        }
        for (; g < n; g++) out[g] = 1 / sqrt(x[g]);
}
/* the fused R / E R kernels per Hermite degree */
#define CX_FUSED
#include "cosx_r_gen.h"
#else
static void rsqrt_vec(double *out, const double *x, int n)
{
        int g;
#pragma omp simd
        for (g = 0; g < n; g++) out[g] = 1 / sqrt(x[g]);
}
#endif


/* F[m * ld + g] = F_m(T_g), m = 0..L, g < n, tmin / tmax the range of T.  Below BOYS_TSW: F_L
 * from the table (Taylor to 4th order about the nearest point of the grid k / 32, error < 1e-12,
 * gathered) and downward
 * recursion (exp from libmvec); above, the asymptotic form F_m = (2m - 1)!! / (2t)^m
 * sqrt(pi / t) / 2 (e^-40 = 4e-18).  Branch-free SIMD loops, the two forms blended on mixed blocks. */
static void boys_vec(double *F, int ld, const double *T, int L, int n, double tmin, double tmax)
{
        int g, m;
        double et[n], fa[n];
        if (tmin < BOYS_TSW) {
                const double *tab = boys_tab[0];
                double *fl = F + (size_t)L * ld;
                int kk[n];
                double tt[n];
#pragma omp simd
                for (g = 0; g < n; g++) {
                        const double t = T[g] < BOYS_TSW ? T[g] : BOYS_TSW;
                        const int k = (int)(t * 32 + .5);
                        kk[g] = k * BOYS_MT + L;
                        tt[g] = k * (1. / 32) - t;
                }
#pragma omp simd
                for (g = 0; g < n; g++) {
                        const int k = kk[g];
                        const double dl = tt[g];
                        fl[g] = tab[k] + dl * (tab[k + 1] + dl * .5 * (tab[k + 2] + dl * (1. / 3) * (tab[k + 3]
                              + dl * .25 * tab[k + 4])));
                }
                if (L > 0) {
#pragma omp simd
                        for (g = 0; g < n; g++) et[g] = exp(-(T[g] < BOYS_TSW ? T[g] : BOYS_TSW));
                        for (m = L; m > 0; m--) {
                                const double inv = boys_inv[m];
                                double *f0 = F + (size_t)(m - 1) * ld, *f1 = F + (size_t)m * ld;
#pragma omp simd
                                for (g = 0; g < n; g++) f0[g] = (2 * T[g] * f1[g] + et[g]) * inv;
                        }
                }
        }
        if (tmax < BOYS_TSW) return;
        {
                double tc[n];
#pragma omp simd
                for (g = 0; g < n; g++) tc[g] = T[g] > 1 ? T[g] : 1;
                rsqrt_vec(fa, tc, n);
#pragma omp simd
                for (g = 0; g < n; g++) {
                        et[g] = .5 * fa[g] * fa[g];
                        fa[g] *= .886226925452758014;
                }
        }
        if (tmin >= BOYS_TSW) {
                for (m = 0; m <= L; m++) {
                        double *f = F + (size_t)m * ld;
                        const double c = 2 * m + 1;
#pragma omp simd
                        for (g = 0; g < n; g++) {
                                f[g] = fa[g];
                                fa[g] *= c * et[g];
                        }
                }
                return;
        }
        for (m = 0; m <= L; m++) {
                double *f = F + (size_t)m * ld;
                const double c = 2 * m + 1;
#pragma omp simd
                for (g = 0; g < n; g++) {
                        f[g] = T[g] >= BOYS_TSW ? fa[g] : f[g];
                        fa[g] *= c * et[g];
                }
        }
}

/* R^(0)_i (i < NH(L)) on n points from the seeds S[m * n + g] = R^(m)_000 (m = 0..L) and
 * X, Y, Z (n each): out[i * n + g]; wk 2 NH(L) n */
static void hermite_R_vec(double *out, const double *S, const double *X, const double *Y, const double *Z,
                          int L, int n, double *wk)
{
        const double *XYZ[3] = {X, Y, Z};
        double *cur = wk, *nxt = wk + (size_t)NH(L) * n;
        int m, i, g;
        if (L == 0) {
                memcpy(out, S, sizeof(double) * n);
                return;
        }
        memcpy(cur, S + (size_t)L * n, sizeof(double) * n);
        for (m = L - 1; m >= 0; m--) {
                const int nh = NH(L - m);
                double *dst = m == 0 ? out : nxt;
                memcpy(dst, S + (size_t)m * n, sizeof(double) * n);
                for (i = 1; i < nh; i++) {
                        const double *x = XYZ[h_dir[i]], *a = cur + (size_t)h_m1[i] * n;
                        double *o = dst + (size_t)i * n;
                        const int c = h_c[i];
                        if (c > 0) {
                                const double *b = cur + (size_t)h_m2[i] * n, cc = c;
#pragma omp simd
                                for (g = 0; g < n; g++) o[g] = x[g] * a[g] + cc * b[g];
                        } else {
#pragma omp simd
                                for (g = 0; g < n; g++) o[g] = x[g] * a[g];
                        }
                }
                if (m > 0) {
                        double *t = cur; cur = nxt; nxt = t;
                }
        }
}

/* ---------------------------------------------------------------- pair data */

typedef struct {
        int I, J, nI, nJ, L, onec, atom, np;
        double bound;
        double pc[3], ext, pmin;  /* centroid of the primitive centres, their max distance to it, min p */
        size_t po;      /* first primitive pair in pp_* */
        size_t eo;      /* E of primitive pair k at edata + eo + k nI nJ NH(L) */
        size_t mo;      /* one-centre: prefix moments at mdata + mo + k nI nJ NH(L) (after k + 1 pairs) */
} CXPair;

typedef struct {
        int nbas, nao, npair, natm, lmax;
        CXPair *pairs;
        double *pp_p, *pp_P, *pp_b;
        int *pp_id;             /* ip * nprim_J + jp of every primitive pair */
        double *eddata;         /* gradient: d E / dA of the two-centre pairs (SOCDFTcosx_build_grad) */
        size_t *edo;
        double *edata, *mdata;
        double *shl_bound;      /* max over the pairs of a shell of the pair bound */
        int *atm_lmax;          /* 2 lmax of the one-centre pairs of each atom (-1: none) */
} CXOpt;

static double common_fac_sp(int l)
{
        return l == 0 ? 0.282094791773878143 : (l == 1 ? 0.488602511902919921 : 1.);
}

static void hermite_E(double *E, int li, int lj, double ai, double aj, double xa, double xb, double e0)
{
        const int lij = li + lj, s2 = lij + 1, s1 = (lj + 1) * s2;
        const double a = ai + aj, xp = (ai * xa + aj * xb) / a, xpa = xp - xa, xpb = xp - xb, h = .5 / a;
        int i, j, t;
        memset(E, 0, sizeof(double) * (li + 1) * s1);
        E[0] = e0;
        for (i = 0; i <= li; i++) {
                for (j = 0; j <= lj; j++) {
                        if (i == 0 && j == 0) continue;
                        double *e = E + i * s1 + j * s2;
                        const int raise_j = j > 0;
                        const double *p = raise_j ? E + i * s1 + (j - 1) * s2 : E + (i - 1) * s1;
                        const double xr = raise_j ? xpb : xpa;
                        for (t = 0; t <= i + j; t++) {
                                double v = xr * p[t];
                                if (t > 0) v += h * p[t - 1];
                                if (t + 1 <= i + j - 1) v += (t + 1) * p[t + 1];
                                e[t] = v;
                        }
                }
        }
}

static const double *c2s_row(int l, int m, int *ncart)
{
        *ncart = (l + 1) * (l + 2) / 2;
        return l == 2 ? C2S_L2[m] : C2S_L3[m];
}

/* the spherical E (degI nI... rows (ci deg_i + mi), cols (cj deg_j + mj), NH(L)) of primitive pair (ip, jp) of
 * shells I, J, coefficients and 2 pi / p folded in; returns the bound max_ab sum_t |E| (2p)^{deg/2} */
static double prim_E(double *out, int I, int J, int ip, int jp, int *atm, int *bas, double *env, double *wk)
{
        const int *bi = bas + I * BAS_SLOTS, *bj = bas + J * BAS_SLOTS;
        const int li = bi[ANG_OF], lj = bj[ANG_OF], L = li + lj, nh = NH(L);
        const int npi = bi[NPRIM_OF], npj = bj[NPRIM_OF], nci = bi[NCTR_OF], ncj = bj[NCTR_OF];
        const int di = 2 * li + 1, dj = 2 * lj + 1, nI = nci * di, nJ = ncj * dj;
        const int nfi = (li + 1) * (li + 2) / 2, nfj = (lj + 1) * (lj + 2) / 2;
        const double ai = env[bi[PTR_EXP] + ip], aj = env[bj[PTR_EXP] + jp];
        const double *Ra = env + atm[bi[ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double *Rb = env + atm[bj[ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double p = ai + aj;
        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
        const double e0 = 2 * M_PI / p * exp(-ai * aj / p * r2) * common_fac_sp(li) * common_fac_sp(lj);
        const int s2 = L + 1, s1 = (lj + 1) * s2, sz = (li + 1) * s1;
        double *Ex = wk, *Ey = Ex + sz, *Ez = Ey + sz;
        double *Ec = Ez + sz;                           /* (nfi, nfj, nh) */
        double *Eh = Ec + (size_t)nfi * nfj * nh;       /* (di, nfj, nh) */
        double *Es = Eh + (size_t)di * nfj * nh;        /* (di, dj, nh) */
        hermite_E(Ex, li, lj, ai, aj, Ra[0], Rb[0], 1.);
        hermite_E(Ey, li, lj, ai, aj, Ra[1], Rb[1], 1.);
        hermite_E(Ez, li, lj, ai, aj, Ra[2], Rb[2], e0);
        int ix, iy, jx, jy, n, fa = 0;
        for (ix = li; ix >= 0; ix--)
        for (iy = li - ix; iy >= 0; iy--, fa++) {
                const int iz = li - ix - iy;
                int fb = 0;
                for (jx = lj; jx >= 0; jx--)
                for (jy = lj - jx; jy >= 0; jy--, fb++) {
                        const int jz = lj - jx - jy;
                        double *e = Ec + (size_t)(fa * nfj + fb) * nh;
                        const double *ex = Ex + ix * s1 + jx * s2, *ey = Ey + iy * s1 + jy * s2;
                        const double *ez = Ez + iz * s1 + jz * s2;
                        for (n = 0; n < nh; n++) e[n] = ex[h_t[n]] * ey[h_u[n]] * ez[h_v[n]];
                }
        }
        /* cart -> sph on a, then on b */
        int m, f, k, nc;
        if (li <= 1) {
                memcpy(Eh, Ec, sizeof(double) * nfi * nfj * nh);
        } else {
                for (m = 0; m < di; m++) {
                        const double *c = c2s_row(li, m, &nc);
                        double *o = Eh + (size_t)m * nfj * nh;
                        memset(o, 0, sizeof(double) * nfj * nh);
                        for (f = 0; f < nfi; f++) {
                                if (c[f] == 0) continue;
                                const double *s = Ec + (size_t)f * nfj * nh;
                                for (k = 0; k < nfj * nh; k++) o[k] += c[f] * s[k];
                        }
                }
        }
        if (lj <= 1) {
                memcpy(Es, Eh, sizeof(double) * di * nfj * nh);
        } else {
                for (fa = 0; fa < di; fa++)
                        for (m = 0; m < dj; m++) {
                                const double *c = c2s_row(lj, m, &nc);
                                double *o = Es + ((size_t)fa * dj + m) * nh;
                                memset(o, 0, sizeof(double) * nh);
                                for (f = 0; f < nfj; f++) {
                                        if (c[f] == 0) continue;
                                        const double *s = Eh + ((size_t)fa * nfj + f) * nh;
                                        for (k = 0; k < nh; k++) o[k] += c[f] * s[k];
                                }
                        }
        }
        /* contraction coefficients */
        const double *cfi = env + bi[PTR_COEFF], *cfj = env + bj[PTR_COEFF];
        double bnd = 0, sc[CX_DMAX + 1];
        {
                const double s = sqrt(2 * p);
                sc[0] = 1;
                for (k = 1; k <= CX_DMAX; k++) sc[k] = sc[k - 1] * s;
        }
        int ci, cj, mi, mj;
        for (ci = 0; ci < nci; ci++)
        for (mi = 0; mi < di; mi++)
        for (cj = 0; cj < ncj; cj++)
        for (mj = 0; mj < dj; mj++) {
                const double cc = cfi[ci * npi + ip] * cfj[cj * npj + jp];
                const double *s = Es + ((size_t)mi * dj + mj) * nh;
                double *o = out + ((size_t)(ci * di + mi) * nJ + cj * dj + mj) * nh;
                double b = 0;
                for (k = 0; k < nh; k++) {
                        o[k] = cc * s[k];
                        b += fabs(o[k]) * sc[h_deg[k]];
                }
                if (b > bnd) bnd = b;
        }
        (void)nI;
        return bnd;
}

static size_t prim_wk_size(void)
{
        const int sE = (CX_LMAX + 1) * (CX_LMAX + 1) * (2 * CX_LMAX + 1);
        const int nc = (CX_LMAX + 1) * (CX_LMAX + 2) / 2, nh = NH(2 * CX_LMAX);
        return 3 * sE + 3 * (size_t)nc * nc * nh;
}

static int cmp_desc2(const void *x, const void *y)
{
        const double a = ((const double *)x)[0], b = ((const double *)y)[0];
        return a < b ? 1 : (a > b ? -1 : 0);
}

/*
 * The pair table of shells (atm, bas, env): primitive pairs with the bound
 * c_i c_j e^{-mu R^2} 2 pi / p above prim_tol, shell pairs with sum_k bound_k above
 * pair_tol.  Returns an opaque pointer (SOCDFTcosx_free), NULL if l > CX_LMAX.
 */
void *SOCDFTcosx_build(double prim_tol, double pair_tol, int *ao_loc, int *atm, int natm, int *bas, int nbas,
                       double *env)
{
        cx_tables();
        int I, J;
        for (I = 0; I < nbas; I++)
                if (bas[I * BAS_SLOTS + ANG_OF] > CX_LMAX || bas[I * BAS_SLOTS + NCTR_OF] * (2 * bas[I * BAS_SLOTS + ANG_OF] + 1) > 64)
                        return NULL;
        CXOpt *opt = calloc(1, sizeof(CXOpt));
        opt->nbas = nbas;
        opt->nao = ao_loc[nbas];
        opt->natm = natm;
        const size_t npmax = (size_t)nbas * (nbas + 1) / 2;
        CXPair *pairs = malloc(sizeof(CXPair) * npmax);
        /* pass 1: the kept primitive pairs by the cheap bound, sizes */
        int *npk = malloc(sizeof(int) * npmax);
        size_t n = 0;
        for (I = 0; I < nbas; I++)
                for (J = 0; J <= I; J++, n++) {
                        const int *bi = bas + I * BAS_SLOTS, *bj = bas + J * BAS_SLOTS;
                        const double *Ra = env + atm[bi[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                        const double *Rb = env + atm[bj[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
                        const int npi = bi[NPRIM_OF], npj = bj[NPRIM_OF], nci = bi[NCTR_OF], ncj = bj[NCTR_OF];
                        int ip, jp, c, cnt = 0;
                        for (ip = 0; ip < npi; ip++) {
                                double ci = 0;
                                for (c = 0; c < nci; c++) ci = fmax(ci, fabs(env[bi[PTR_COEFF] + c * npi + ip]));
                                const double ai = env[bi[PTR_EXP] + ip];
                                for (jp = 0; jp < npj; jp++) {
                                        double cj = 0;
                                        for (c = 0; c < ncj; c++) cj = fmax(cj, fabs(env[bj[PTR_COEFF] + c * npj + jp]));
                                        const double aj = env[bj[PTR_EXP] + jp], p = ai + aj;
                                        const double b = ci * cj * 2 * M_PI / p * exp(-ai * aj / p * r2)
                                                       * (1 + ai * aj / p * r2);
                                        if (b > prim_tol) cnt++;
                                }
                        }
                        npk[n] = cnt;
                }
        /* offsets */
        size_t po = 0, eo = 0, mo = 0;
        int np_ = 0;
        n = 0;
        for (I = 0; I < nbas; I++)
                for (J = 0; J <= I; J++, n++) {
                        if (npk[n] == 0) continue;
                        const int *bi = bas + I * BAS_SLOTS, *bj = bas + J * BAS_SLOTS;
                        CXPair *pp = pairs + np_;
                        pp->I = I; pp->J = J;
                        pp->nI = bi[NCTR_OF] * (2 * bi[ANG_OF] + 1);
                        pp->nJ = bj[NCTR_OF] * (2 * bj[ANG_OF] + 1);
                        pp->L = bi[ANG_OF] + bj[ANG_OF];
                        pp->atom = bi[ATOM_OF];
                        pp->onec = bi[ATOM_OF] == bj[ATOM_OF];
                        pp->np = npk[n];
                        pp->po = po;
                        pp->eo = eo;
                        pp->mo = mo;
                        const size_t sz = (size_t)pp->nI * pp->nJ * NH(pp->L);
                        po += pp->np;
                        eo += sz * pp->np;
                        if (pp->onec) mo += sz * pp->np;
                        np_++;
                }
        free(npk);
        opt->npair = np_;
        opt->pairs = pairs;
        opt->pp_p = malloc(sizeof(double) * (po + 1));
        opt->pp_P = malloc(sizeof(double) * 3 * (po + 1));
        opt->pp_b = malloc(sizeof(double) * (po + 1));
        opt->pp_id = malloc(sizeof(int) * (po + 1));
        opt->eddata = NULL;
        opt->edo = NULL;
        opt->edata = malloc(sizeof(double) * (eo + 1));
        opt->mdata = malloc(sizeof(double) * (mo + 1));
        /* pass 2: E, bounds, the one-centre ordering and prefix moments */
        int ip_;
#pragma omp parallel
{
        double *wk = malloc(sizeof(double) * prim_wk_size());
        double *tmpE = malloc(sizeof(double) * 64 * 64 * NH(2 * CX_LMAX));
        double *key = malloc(sizeof(double) * 2 * 4096);
#pragma omp for schedule(dynamic, 8)
        for (ip_ = 0; ip_ < np_; ip_++) {
                CXPair *pp = pairs + ip_;
                const int I = pp->I, J = pp->J;
                const int *bi = bas + I * BAS_SLOTS, *bj = bas + J * BAS_SLOTS;
                const double *Ra = env + atm[bi[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double *Rb = env + atm[bj[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                                + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
                const int npi = bi[NPRIM_OF], npj = bj[NPRIM_OF], nci = bi[NCTR_OF], ncj = bj[NCTR_OF];
                const size_t sz = (size_t)pp->nI * pp->nJ * NH(pp->L);
                int ip, jp, c, k = 0;
                /* the kept primitive pairs (same test as pass 1), keyed by p (descending) */
                for (ip = 0; ip < npi; ip++) {
                        double ci = 0;
                        for (c = 0; c < nci; c++) ci = fmax(ci, fabs(env[bi[PTR_COEFF] + c * npi + ip]));
                        const double ai = env[bi[PTR_EXP] + ip];
                        for (jp = 0; jp < npj; jp++) {
                                double cj = 0;
                                for (c = 0; c < ncj; c++) cj = fmax(cj, fabs(env[bj[PTR_COEFF] + c * npj + jp]));
                                const double aj = env[bj[PTR_EXP] + jp], p = ai + aj;
                                const double b = ci * cj * 2 * M_PI / p * exp(-ai * aj / p * r2)
                                               * (1 + ai * aj / p * r2);
                                if (b > prim_tol) {
                                        key[2 * k] = p;
                                        key[2 * k + 1] = ip * npj + jp;
                                        k++;
                                }
                        }
                }
                if (pp->onec) qsort(key, k, 2 * sizeof(double), cmp_desc2);
                double btot = 0;
                for (k = 0; k < pp->np; k++) {
                        const int id = (int)key[2 * k + 1];
                        ip = id / npj; jp = id % npj;
                        const double ai = env[bi[PTR_EXP] + ip], aj = env[bj[PTR_EXP] + jp], p = ai + aj;
                        double *E = opt->edata + pp->eo + sz * k;
                        const double b = prim_E(E, I, J, ip, jp, atm, bas, env, wk);
                        opt->pp_p[pp->po + k] = p;
                        opt->pp_id[pp->po + k] = id;
                        opt->pp_b[pp->po + k] = b;
                        for (c = 0; c < 3; c++) opt->pp_P[3 * (pp->po + k) + c] = (ai * Ra[c] + aj * Rb[c]) / p;
                        btot += b;
                        if (pp->onec) {
                                /* M_k = M_{k-1} + E_k sqrt(pi) / (2 sqrt(p)) (E carries 2 pi / p) */
                                double *M = opt->mdata + pp->mo + sz * k;
                                const double f = .886226925452758014 / sqrt(p);
                                size_t q;
                                for (q = 0; q < sz; q++) M[q] = (k ? M[q - sz] : 0) + f * E[q];
                        }
                }
                pp->bound = btot;
                {
                        double c[3] = {0, 0, 0}, e = 0, pm = 1e300;
                        for (k = 0; k < pp->np; k++) {
                                const double *P = opt->pp_P + 3 * (pp->po + k);
                                c[0] += P[0]; c[1] += P[1]; c[2] += P[2];
                                pm = fmin(pm, opt->pp_p[pp->po + k]);
                        }
                        for (c[0] /= pp->np, c[1] /= pp->np, c[2] /= pp->np, k = 0; k < pp->np; k++) {
                                const double *P = opt->pp_P + 3 * (pp->po + k);
                                e = fmax(e, sqrt((P[0] - c[0]) * (P[0] - c[0]) + (P[1] - c[1]) * (P[1] - c[1])
                                                 + (P[2] - c[2]) * (P[2] - c[2])));
                        }
                        memcpy(pp->pc, c, sizeof(c));
                        pp->ext = e;
                        pp->pmin = pm;
                }
        }
        free(wk);
        free(tmpE);
        free(key);
}
        opt->shl_bound = calloc(nbas, sizeof(double));
        opt->atm_lmax = malloc(sizeof(int) * natm);
        int a;
        for (a = 0; a < natm; a++) opt->atm_lmax[a] = -1;
        opt->lmax = 0;
        for (ip_ = 0; ip_ < np_; ip_++) {
                CXPair *pp = pairs + ip_;
                if (pp->bound < pair_tol) {
                        pp->bound = 0;          /* never passes the screening (tol > 0) */
                        continue;
                }
                opt->shl_bound[pp->I] = fmax(opt->shl_bound[pp->I], pp->bound);
                opt->shl_bound[pp->J] = fmax(opt->shl_bound[pp->J], pp->bound);
                if (pp->onec && pp->L > opt->atm_lmax[pp->atom]) opt->atm_lmax[pp->atom] = pp->L;
                if (pp->L > opt->lmax) opt->lmax = pp->L;
        }
        return opt;
}

void SOCDFTcosx_free(void *p)
{
        CXOpt *opt = p;
        if (!opt) return;
        free(opt->pairs); free(opt->pp_p); free(opt->pp_P); free(opt->pp_b);
        free(opt->edata); free(opt->mdata); free(opt->pp_id); free(opt->eddata); free(opt->edo); free(opt->shl_bound); free(opt->atm_lmax);
        free(opt);
}

/* sizes: npair, primitive pairs, doubles of E */
void SOCDFTcosx_info(void *p, double *out)
{
        CXOpt *opt = p;
        size_t np = 0, ne = 0;
        int i;
        for (i = 0; i < opt->npair; i++) {
                np += opt->pairs[i].np;
                ne += (size_t)opt->pairs[i].np * opt->pairs[i].nI * opt->pairs[i].nJ * NH(opt->pairs[i].L);
        }
        out[0] = opt->npair;
        out[1] = np;
        out[2] = ne;
}

/* ---------------------------------------------------------------- the K kernel */

/* the AOs of the shells flagged on the block's non0tab rows with max_g |phi| > cutoff: ix (nl), a (nl, n) */
static int eval_select(double *a, int *ix, int *runs, int n, double *xyz, uint8_t *msk, double cutoff,
                       int *ao_loc, int *atm, int *bas, int nbas, double *env, double *rmx)
{
        const int nrow = (n + MASK_BLK - 1) / MASK_BLK;
        int sh = 0, nrun = 0, r, i;
        size_t ntot = 0;
        while (sh < nbas) {
                int on = 0;
                for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                if (!on) { sh++; continue; }
                const int s0 = sh;
                for (sh++; sh < nbas; sh++) {
                        on = 0;
                        for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                        if (!on) break;
                }
                runs[2 * nrun] = s0;
                runs[2 * nrun + 1] = sh;
                nrun++;
                ntot += ao_loc[sh] - ao_loc[s0];
        }
        if (ntot == 0) return 0;
        size_t row = 0;
        memset(rmx, 0, sizeof(double) * ntot);
        for (i = 0; i < nrun; i++) {
                SOCDFTgtoval_ld(0, n, runs[2 * i], runs[2 * i + 1], ao_loc, a, ntot, row, xyz, msk, atm, bas,
                                nbas, env, rmx);
                row += ao_loc[runs[2 * i + 1]] - ao_loc[runs[2 * i]];
        }
        int nl = 0;
        row = 0;
        for (i = 0; i < nrun; i++) {
                int mu;
                for (mu = ao_loc[runs[2 * i]]; mu < ao_loc[runs[2 * i + 1]]; mu++, row++) {
                        if (rmx[row] > cutoff) {
                                if ((size_t)nl != row)
                                        memmove(a + (size_t)nl * n, a + row * n, sizeof(double) * n);
                                ix[nl++] = mu;
                        }
                }
        }
        return nl;
}

/* A (nab, n) += E (nab, nh) R (nh, n) */
static void e_times_r(double *A, const double *E, const double *R, int nab, int nh, int n)
{
        int ab, i, g;
        for (ab = 0; ab < nab; ab++) {
                double *a = A + (size_t)ab * n;
                const double *e = E + (size_t)ab * nh;
                i = 0;
                for (; i + 3 < nh; i += 4) {
                        const double e0 = e[i], e1 = e[i + 1], e2 = e[i + 2], e3 = e[i + 3];
                        const double *r0 = R + (size_t)i * n, *r1 = r0 + n, *r2 = r1 + n, *r3 = r2 + n;
#pragma omp simd
                        for (g = 0; g < n; g++) a[g] += e0 * r0[g] + e1 * r1[g] + e2 * r2[g] + e3 * r3[g];
                }
                for (; i < nh; i++) {
                        const double e0 = e[i], *r0 = R + (size_t)i * n;
#pragma omp simd
                        for (g = 0; g < n; g++) a[g] += e0 * r0[g];
                }
        }
}

typedef struct { int I0, J0, nI, nJ, cI, cJ; size_t off; } CXRec;

#ifdef CX_FUSED
#define V8 __m512d
#define V8LOAD(p) _mm512_loadu_pd(p)
#define V8STORE(p, v) _mm512_storeu_pd(p, v)
#define V8FMA(a, b, c) _mm512_fmadd_pd(a, b, c)
#define V8SET1(x) _mm512_set1_pd(x)
#else
typedef double V8 __attribute__((vector_size(64)));
static inline V8 v8load(const double *p) { V8 v; memcpy(&v, p, sizeof(v)); return v; }
static inline void v8store(double *p, V8 v) { memcpy(p, &v, sizeof(v)); }
#define V8LOAD(p) v8load(p)
#define V8STORE(p, v) v8store(p, v)
#define V8FMA(a, b, c) ((a) * (b) + (c))
#define V8SET1(x) ((V8){x, x, x, x, x, x, x, x})
#endif

/* multi-density contraction of the stored pair integrals (At[off + g nab + ab], point-major so that
 * the integrals of one point are contiguous): G[g][a][d] += A_ab(g) F[g][b][d] (and the mirror),
 * point by point so that F[g], G[g] (NQ = nao nset doubles each) stay in cache; the densities
 * (nset a multiple of 8) are the vector dimension, 8 at a time.  The records come in runs of one
 * shell I (the pair order): a row of G_I is accumulated in a register over the whole run. */
static void multi_contract(double *Ga, const double *Fa, const double *Abuf, const CXRec *rec, int nrec,
                           int nao, int n, int nset)
{
        const size_t NQ = (size_t)nao * nset;
        int g, r, r1, rr;
        for (g = 0; g < n; g++) {
                const double *Fg = Fa + g * NQ;
                double *Gg = Ga + g * NQ;
                for (r = 0; r < nrec; r = r1) {
                        const int I0 = rec[r].I0, nI = rec[r].nI;
                        int anyI = 0;
                        for (r1 = r; r1 < nrec && rec[r1].I0 == I0; r1++) anyI |= rec[r1].cI;
                        const double *FI = Fg + (size_t)I0 * nset;
                        double *GI = Gg + (size_t)I0 * nset;
                        int ia, jb, c;
                        if (anyI) {
                                for (c = 0; c < nset; c += 8) {
                                        for (ia = 0; ia < nI; ia++) {
                                                double *ga = GI + (size_t)ia * nset + c;
                                                V8 acc = V8LOAD(ga);
                                                for (rr = r; rr < r1; rr++) {
                                                        const CXRec *rc = rec + rr;
                                                        if (!rc->cI) continue;
                                                        const int nJ = rc->nJ;
                                                        const double *ar = Abuf + rc->off + (size_t)g * nI * nJ + ia * nJ;
                                                        const double *FJ = Fg + (size_t)rc->J0 * nset + c;
                                                        for (jb = 0; jb < nJ; jb++)
                                                                acc = V8FMA(V8SET1(ar[jb]), V8LOAD(FJ + (size_t)jb * nset), acc);
                                                }
                                                V8STORE(ga, acc);
                                        }
                                }
                        }
                        for (rr = r; rr < r1; rr++) {
                                const CXRec *rc = rec + rr;
                                if (!rc->cJ) continue;
                                const int nJ = rc->nJ;
                                const double *Ar = Abuf + rc->off + (size_t)g * nI * nJ;
                                double *GJ = Gg + (size_t)rc->J0 * nset;
                                for (c = 0; c < nset; c += 8) {
                                        for (jb = 0; jb < nJ; jb++) {
                                                double *gb = GJ + (size_t)jb * nset + c;
                                                V8 acc = V8LOAD(gb);
                                                for (ia = 0; ia < nI; ia++)
                                                        acc = V8FMA(V8SET1(Ar[ia * nJ + jb]), V8LOAD(FI + (size_t)ia * nset + c), acc);
                                                V8STORE(gb, acc);
                                        }
                                }
                        }
                }
        }
}

#ifdef CX_FUSED
/* the same in single precision, 16 densities per vector (SOCDFTcosx_set_opt(2, 1)): the CPHF K only
 * needs the COSX accuracy (~1e-3), the float sums keep ~1e-6 */
static void multi_contract_f(float *Ga, const float *Fa, const float *Abuf, const CXRec *rec, int nrec,
                             int nao, int n, int nset)
{
        const size_t NQ = (size_t)nao * nset;
        int g, r, r1, rr;
        for (g = 0; g < n; g++) {
                const float *Fg = Fa + g * NQ;
                float *Gg = Ga + g * NQ;
                for (r = 0; r < nrec; r = r1) {
                        const int I0 = rec[r].I0, nI = rec[r].nI;
                        int anyI = 0;
                        for (r1 = r; r1 < nrec && rec[r1].I0 == I0; r1++) anyI |= rec[r1].cI;
                        const float *FI = Fg + (size_t)I0 * nset;
                        float *GI = Gg + (size_t)I0 * nset;
                        int ia, jb, c;
                        if (anyI) {
                                for (c = 0; c < nset; c += 16) {
                                        for (ia = 0; ia < nI; ia++) {
                                                float *ga = GI + (size_t)ia * nset + c;
                                                __m512 acc = _mm512_loadu_ps(ga);
                                                for (rr = r; rr < r1; rr++) {
                                                        const CXRec *rc = rec + rr;
                                                        if (!rc->cI) continue;
                                                        const int nJ = rc->nJ;
                                                        const float *ar = Abuf + rc->off + (size_t)g * nI * nJ + ia * nJ;
                                                        const float *FJ = Fg + (size_t)rc->J0 * nset + c;
                                                        for (jb = 0; jb < nJ; jb++)
                                                                acc = _mm512_fmadd_ps(_mm512_set1_ps(ar[jb]),
                                                                                      _mm512_loadu_ps(FJ + (size_t)jb * nset), acc);
                                                }
                                                _mm512_storeu_ps(ga, acc);
                                        }
                                }
                        }
                        for (rr = r; rr < r1; rr++) {
                                const CXRec *rc = rec + rr;
                                if (!rc->cJ) continue;
                                const int nJ = rc->nJ;
                                const float *Ar = Abuf + rc->off + (size_t)g * nI * nJ;
                                float *GJ = Gg + (size_t)rc->J0 * nset;
                                for (c = 0; c < nset; c += 16) {
                                        for (jb = 0; jb < nJ; jb++) {
                                                float *gb = GJ + (size_t)jb * nset + c;
                                                __m512 acc = _mm512_loadu_ps(gb);
                                                for (ia = 0; ia < nI; ia++)
                                                        acc = _mm512_fmadd_ps(_mm512_set1_ps(Ar[ia * nJ + jb]),
                                                                              _mm512_loadu_ps(FI + (size_t)ia * nset + c), acc);
                                                _mm512_storeu_ps(gb, acc);
                                        }
                                }
                        }
                }
        }
}
#endif

#define CX_ABUF (2 << 20)
#define CX_KROWS 64             /* rows of K per GEMM in the multi-density mode */       /* doubles of stored pair integrals per thread in the multi-density mode */

/*
 * vk (nset, nao, nao) = the plain K of dms (nset, nao, nao), K_mu nu =
 * sum_g w_g phi_mu(g) sum_lam A_nu lam(g) F_lam(g), F_lam = sum_kap D_kap lam phi_kap.
 * coords (3, ngrid), weights, non0tab (ngrid / 56, nbas); blk points per task (multiple of 56);
 * cutoff: AO values; tol: the screening of the pair contributions; tcut: the far-field T.
 */
void SOCDFTcosx_k(double *vk, double *dms, int nset, void *optp, double *coords, double *weights,
                  uint8_t *non0tab, int ngrid, int blk, double cutoff, double tol, double tcut,
                  int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                  dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local, double *dmh, double *eout,
                  double *dtm)
{
        /* multi-density mode (dtm != NULL): dtm (nao, nao, nset) = the densities, dtm[l][n][d] = D_d[l][n],
         * vk (nao, nao, nset); the integrals of a block's pairs stored, then contracted point by point
         * with the densities as the vector dimension (multi_contract), F and K by one GEMM each */
        const int multi = dtm != NULL;
        const int mfl = multi && SOCDFTcosx_mfloat();
        /* energy mode (dmh != NULL): eout[d] = sum_g w sum_ab H_a A_ab F_b, H = Dh_d^T phi, vk untouched;
         * the screening then weighs each pair by the densities on both sides */
        const int eonly = dmh != NULL;
        CXOpt *opt = optp;
        cx_tables();
        const int nao = ao_loc[nbas];
        const size_t nn = (size_t)nao * nao;
        const int nblk = (ngrid + blk - 1) / blk;
        const int nthreads = omp_get_max_threads();
        double **kbufs = malloc(sizeof(double *) * nthreads);
        const int Lmax = opt->lmax, nhmax = NH(Lmax);
        int nab_max = 1, i0;
        for (i0 = 0; i0 < opt->npair; i0++)
                if (opt->pairs[i0].nI * opt->pairs[i0].nJ > nab_max) nab_max = opt->pairs[i0].nI * opt->pairs[i0].nJ;
        int *shl_of = malloc(sizeof(int) * nao);
        for (i0 = 0; i0 < nbas; i0++) {
                int mu;
                for (mu = ao_loc[i0]; mu < ao_loc[i0 + 1]; mu++) shl_of[mu] = i0;
        }
        if (eonly) memset(eout, 0, sizeof(double) * nset);
        else memset(vk, 0, sizeof(double) * nn * nset);
#pragma omp parallel
{
        const int tid = omp_get_thread_num();
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        const double one = 1, zero = 0;
        double *Kt = kbufs[tid] = eonly ? NULL : calloc(nn * nset, sizeof(double));
        const size_t NQ = (size_t)nao * nset;
        double *Bm = multi ? malloc(sizeof(double) * NQ * (nao > CX_KROWS ? nao : CX_KROWS)) : NULL;
        double *Abuf = multi ? malloc(sizeof(double) * CX_ABUF) : NULL;
        CXRec *rec = multi ? malloc(sizeof(CXRec) * (opt->npair + 1)) : NULL;
        float *Ff = mfl ? malloc(sizeof(float) * NQ * blk) : NULL;
        float *Gf = mfl ? malloc(sizeof(float) * NQ * blk) : NULL;
        float *Abf = (float *)Abuf;
        const size_t abcap = mfl ? 2 * (size_t)CX_ABUF : (size_t)CX_ABUF;
        int nrec = 0;
        size_t aoff = 0;
        double *H = eonly ? malloc(sizeof(double) * (size_t)nset * nao * blk) : NULL;
        double *xs = malloc(sizeof(double) * nbas);
        double esum[nset > 0 ? nset : 1];
        memset(esum, 0, sizeof(esum));
        double *a = malloc(sizeof(double) * (size_t)nao * blk);
        double *wx = malloc(sizeof(double) * (size_t)nao * blk);
        double *rmx = malloc(sizeof(double) * nao);
        int *ix = malloc(sizeof(int) * nao);
        int *runs = malloc(sizeof(int) * 2 * (nbas + 1));
        double *dr = malloc(sizeof(double) * (size_t)nao * nao);
        double *F = malloc(sizeof(double) * (size_t)nset * nao * blk);
        double *G = malloc(sizeof(double) * (size_t)nset * nao * blk);
        double *ktmp = malloc(sizeof(double) * (size_t)nao * nao);
        double *fmax = malloc(sizeof(double) * nbas);
        double *A0 = malloc(sizeof(double) * (size_t)nab_max * blk), *A = A0;
        double *bk = malloc(sizeof(double) * 4096);
        double *R = malloc(sizeof(double) * (size_t)nhmax * blk);
        double *Rwk = malloc(sizeof(double) * 2 * (size_t)nhmax * blk);
        double *S = malloc(sizeof(double) * (size_t)(Lmax + 1) * blk);
        double *X = malloc(sizeof(double) * 4 * blk), *Y = X + blk, *Z = Y + blk, *T = Z + blk;
        double *xyz = malloc(sizeof(double) * 3 * blk);
        /* per atom on the block: d^t (1/r) up to its one-centre degree, min r^2 (stamp = block) */
        double **farR = calloc(natm, sizeof(double *));
        int *stamp = malloc(sizeof(int) * natm);
        double *armin = malloc(sizeof(double) * natm);
        int at, i;
        double st[32] = {0};
        for (at = 0; at < natm; at++) {
                stamp[at] = -1;
                if (opt->atm_lmax[at] >= 0) farR[at] = malloc(sizeof(double) * NH(opt->atm_lmax[at]) * blk);
        }
        int b;
        /* out of memory: flagged (SOCDFTcosx_error), the blocks skipped */
        const int oom = (!eonly && !Kt) || !a || !wx || !F || !G || !dr || !ktmp
                        || (multi && (!Bm || !Abuf || !rec)) || (mfl && (!Ff || !Gf)) || (eonly && !H);
        if (oom) {
#pragma omp atomic write
                cx_err = 1;
        }
#pragma omp for schedule(dynamic, 1)
        for (b = 0; b < nblk; b++) {
                if (oom) continue;
                const int g0 = b * blk, n = g0 + blk <= ngrid ? blk : ngrid - g0;
                uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int k, g, d, ip;
                double tt0 = omp_get_wtime(), tt1;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select(a, ix, runs, n, xyz, msk, cutoff, ao_loc, atm, bas, nbas, env, rmx);
                if (nl == 0) continue;
                const double *w = weights + g0;
                double xsum = 0;
                for (i = 0; i < nl; i++) {
                        double s = 0;
                        for (g = 0; g < n; g++) {
                                wx[(size_t)i * n + g] = w[g] * a[(size_t)i * n + g];
                                s += fabs(wx[(size_t)i * n + g]);
                        }
                        if (s > xsum) xsum = s;
                }
                /* F_d (nao, n) = D_d[ix, :]^T a */
                for (i = 0; i < nbas; i++) fmax[i] = 0;
                if (multi) {
                        /* F[g][nu][d] = sum_i phi_i(g) D_d[ix_i][nu]: the density rows gathered, one GEMM
                         * (a GEMM per run of consecutive AOs would stream F once per run) */
                        const int nq = (int)NQ;
                        for (i = 0; i < nl; i++) memcpy(Bm + (size_t)i * NQ, dtm + (size_t)ix[i] * NQ, sizeof(double) * NQ);
                        dgemm_("N", "T", &nq, &n, &nl, &one, Bm, &nq, a, &n, &zero, F, &nq);
                        if (mfl) {
                                size_t q;
                                for (q = 0; q < NQ * n; q++) Ff[q] = (float)F[q];
                                memset(Gf, 0, sizeof(float) * NQ * n);
                        }
                        int mu;
                        for (g = 0; g < n; g++)
                                for (mu = 0; mu < nao; mu++) {
                                        const double *f = F + (size_t)g * NQ + (size_t)mu * nset;
                                        double m = fmax[shl_of[mu]];
                                        for (d = 0; d < nset; d++) m = fabs(f[d]) > m ? fabs(f[d]) : m;
                                        fmax[shl_of[mu]] = m;
                                }
                        nrec = 0;
                        aoff = 0;
                }
                for (d = 0; d < nset && !multi; d++) {
                        const double *D = dms + nn * d;
                        double *Fd = F + (size_t)d * nao * n;
                        for (i = 0; i < nl; i++) memcpy(dr + (size_t)i * nao, D + (size_t)ix[i] * nao, sizeof(double) * nao);
                        dgemm_("N", "T", &n, &nao, &nl, &one, a, &n, dr, &nao, &zero, Fd, &n);
                        int mu;
                        for (mu = 0; mu < nao; mu++) {
                                double m = 0;
                                const double *f = Fd + (size_t)mu * n;
                                for (g = 0; g < n; g++) m = fabs(f[g]) > m ? fabs(f[g]) : m;
                                if (m > fmax[shl_of[mu]]) fmax[shl_of[mu]] = m;
                        }
                }
                if (!eonly) memset(G, 0, sizeof(double) * nset * nao * n);
                /* the weights of the shells on the grid side: max_mu sum_g |w phi| (K), max_a sum_g |H_a| (energy) */
                for (i = 0; i < nbas; i++) xs[i] = xsum;
                if (eonly) {
                        for (i = 0; i < nbas; i++) xs[i] = 0;
                        for (d = 0; d < nset; d++) {
                                const double *D = dmh + nn * d;
                                double *Hd = H + (size_t)d * nao * n;
                                for (i = 0; i < nl; i++)
                                        memcpy(dr + (size_t)i * nao, D + (size_t)ix[i] * nao, sizeof(double) * nao);
                                dgemm_("N", "T", &n, &nao, &nl, &one, wx, &n, dr, &nao, &zero, Hd, &n);
                                int mu;
                                for (mu = 0; mu < nao; mu++) {
                                        double m = 0;
                                        const double *h = Hd + (size_t)mu * n;
                                        for (g = 0; g < n; g++) m += fabs(h[g]);
                                        if (m > xs[shl_of[mu]]) xs[shl_of[mu]] = m;
                                }
                        }
                }
                /* the block's bounding sphere, for the distance-damped primitive bounds */
                double cb[3] = {0, 0, 0}, rb = 0;
                for (g = 0; g < n; g++) {
                        cb[0] += xyz[g]; cb[1] += xyz[n + g]; cb[2] += xyz[2 * n + g];
                }
                cb[0] /= n; cb[1] /= n; cb[2] /= n;
                for (g = 0; g < n; g++) {
                        const double dx = xyz[g] - cb[0], dy = xyz[n + g] - cb[1], dz = xyz[2 * n + g] - cb[2];
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 > rb) rb = r2;
                }
                rb = sqrt(rb);
                tt1 = omp_get_wtime(); st[0] += tt1 - tt0; tt0 = tt1;
                for (ip = 0; ip < opt->npair; ip++) {
                        const CXPair *pp = opt->pairs + ip;
                        /* contributions to G_I (A F_J, weighed by the grid side of I) and to G_J */
                        const double wI = fmax[pp->J] * xs[pp->I], wJ = pp->I != pp->J ? fmax[pp->I] * xs[pp->J] : 0;
                        if (pp->bound == 0) continue;           /* below PAIR_TOL */
                        const int cI = pp->bound * wI >= tol;
                        const int cJ = pp->I != pp->J && pp->bound * wJ >= tol;
                        if (!cI && !cJ) continue;
                        const int L = pp->L, nh = NH(L), nI = pp->nI, nJ = pp->nJ, nab = nI * nJ;
                        {
                                /* the pair's bound damped over the distance of its primitive centres to the
                                 * block (the smallest p): one sqrt before the per-primitive bounds */
                                const double dx = pp->pc[0] - cb[0], dy = pp->pc[1] - cb[1], dz = pp->pc[2] - cb[2];
                                const double dmin = sqrt(dx * dx + dy * dy + dz * dz) - rb - pp->ext;
                                if (cx_pairdamp && dmin > 0) {
                                        const double T0 = pp->pmin * dmin * dmin;
                                        if (T0 > 1) {
                                                const double ir = 1 / sqrt(T0);
                                                double rp = ir, mx = 0;
                                                int dd;
                                                for (dd = 0; dd <= L; dd++) {
                                                        const double v = cx_damp[dd] * rp;
                                                        if (v > mx) mx = v;
                                                        rp *= ir;
                                                }
                                                if (mx < 1 && !((pp->bound * mx * wI >= tol && cI)
                                                                || (pp->bound * mx * wJ >= tol && cJ)))
                                                        continue;
                                        }
                                }
                        }
                        /* the bounds of the primitive pairs damped by their distance to the block:
                         * |(2 pi / p) R_t| <= (2p)^{d/2} min(1, c_d T^{-(d+1)/2}) (the far-field form) */
                        double bsum = 0;
                        for (k = 0; k < pp->np; k++) {
                                const double *P = opt->pp_P + 3 * (pp->po + k);
                                const double dx = P[0] - cb[0], dy = P[1] - cb[1], dz = P[2] - cb[2];
                                double dmin = sqrt(dx * dx + dy * dy + dz * dz) - rb;
                                double f = 1;
                                if (dmin > 0) {
                                        const double T0 = opt->pp_p[pp->po + k] * dmin * dmin;
                                        if (T0 > 1) {
                                                const double ir = 1 / sqrt(T0);
                                                double rp = ir, mx = 0;
                                                int dd;
                                                for (dd = 0; dd <= L; dd++) {
                                                        const double v = cx_damp[dd] * rp;
                                                        if (v > mx) mx = v;
                                                        rp *= ir;
                                                }
                                                f = mx < 1 ? mx : 1;
                                        }
                                }
                                bk[k] = f * opt->pp_b[pp->po + k];
                                bsum += bk[k];
                        }
                        const int cI2 = cI && bsum * wI >= tol;
                        const int cJ2 = cJ && bsum * wJ >= tol;
                        if (!cI2 && !cJ2) continue;
                        const double fm = cI2 && cJ2 ? (wI > wJ ? wI : wJ) : (cI2 ? wI : wJ);
                        PROF(st[4] += 1);
                        const size_t sz = (size_t)nab * nh;
                        int k0 = 0;
                        PROF(const unsigned long long cp0 = __rdtsc());
                        if (multi && aoff + (size_t)nab * n > abcap) {
#ifdef CX_FUSED
                                if (mfl) multi_contract_f(Gf, Ff, Abf, rec, nrec, nao, n, nset);
                                else
#endif
                                multi_contract(G, F, Abuf, rec, nrec, nao, n, nset);
                                nrec = 0;
                                aoff = 0;
                        }
                        memset(A, 0, sizeof(double) * nab * n);
                        if (pp->onec) {
                                at = pp->atom;
                                const double *Ra = env + atm[at * ATM_SLOTS + PTR_COORD];
                                if (stamp[at] != b) {
                                        double rmin = 1e300;
                                        for (g = 0; g < n; g++) {
                                                X[g] = Ra[0] - xyz[g];
                                                Y[g] = Ra[1] - xyz[n + g];
                                                Z[g] = Ra[2] - xyz[2 * n + g];
                                                T[g] = X[g] * X[g] + Y[g] * Y[g] + Z[g] * Z[g];
                                                if (T[g] < rmin) rmin = T[g];
                                        }
                                        armin[at] = rmin;
                                        stamp[at] = b;
                                        /* the far-field R only if some pair is far on the block (rmin > 0) */
                                        if (rmin > 1e-12) {
                                                const int La = opt->atm_lmax[at];
                                                int m;
                                                rsqrt_vec(S + (size_t)La * n, T, n);
                                                for (g = 0; g < n; g++) {
                                                        const double ir = S[(size_t)La * n + g], ir2 = ir * ir;
                                                        double s = ir;
                                                        for (m = 0; m <= La; m++) {
                                                                S[m * n + g] = s;
                                                                s *= -(2 * m + 1) * ir2;
                                                        }
                                                }
                                                hermite_R_vec(farR[at], S, X, Y, Z, La, n, Rwk);
                                        } else {
                                                armin[at] = 0;
                                        }
                                }
                                const double rmin = armin[at];
                                if (rmin > 0) {
                                        const double *pv = opt->pp_p + pp->po;
                                        while (k0 < pp->np && pv[k0] * rmin >= tcut) k0++;
                                }
                                PROF(st[6] += k0 > 0);
                                if (k0 > 0)
                                        e_times_r(A, opt->mdata + pp->mo + sz * (k0 - 1), farR[at], nab, nh, n);
                        }
                        PROF(st[24] += __rdtsc() - cp0);
                        for (k = k0; k < pp->np; k++) {
                                if (bk[k] * fm < tol) continue;
                                PROF(st[5] += 1; st[8 + L] += n; const unsigned long long c0 = __rdtsc());
                                const double p = opt->pp_p[pp->po + k];
                                const double *P = opt->pp_P + 3 * (pp->po + k);
#ifdef CX_FUSED
                                if (L <= cx_full) {
                                        cxp_table[L](A, nab, opt->edata + pp->eo + sz * k, p, P, xyz, n);
                                        PROF(st[16 + L] += __rdtsc() - c0);
                                        continue;
                                }
#endif
                                double tmin = 1e300, tmax = 0;
#pragma omp simd reduction(min:tmin) reduction(max:tmax)
                                for (g = 0; g < n; g++) {
                                        X[g] = P[0] - xyz[g];
                                        Y[g] = P[1] - xyz[n + g];
                                        Z[g] = P[2] - xyz[2 * n + g];
                                        T[g] = p * (X[g] * X[g] + Y[g] * Y[g] + Z[g] * Z[g]);
                                        tmin = T[g] < tmin ? T[g] : tmin;
                                        tmax = T[g] > tmax ? T[g] : tmax;
                                }
                                boys_vec(S, n, T, L, n, tmin, tmax);
                                double sc[CX_DMAX + 1];
                                int m;
                                sc[0] = 1;
                                for (m = 1; m <= L; m++) sc[m] = sc[m - 1] * (-2 * p);
#ifdef CX_FUSED
                                cxr_table[L](A, nab, opt->edata + pp->eo + sz * k, S, sc, X, Y, Z, n);
#else
                                for (m = 1; m <= L; m++) {
                                        double *sm = S + (size_t)m * n;
                                        for (g = 0; g < n; g++) sm[g] *= sc[m];
                                }
                                hermite_R_vec(R, S, X, Y, Z, L, n, Rwk);
                                if (nab * nh >= CX_GEMM_MIN)
                                        dgemm_("N", "N", &n, &nab, &nh, &one, R, &n, opt->edata + pp->eo + sz * k, &nh,
                                               &one, A, &n);
                                else
                                        e_times_r(A, opt->edata + pp->eo + sz * k, R, nab, nh, n);
#endif
                                PROF(st[16 + L] += __rdtsc() - c0);
                        }
                        PROF(st[25] += __rdtsc() - cp0; const unsigned long long cc0 = __rdtsc());
                        /* G_I += A F_J, G_J += A^T F_I */
                        const int I0 = ao_loc[pp->I], J0 = ao_loc[pp->J];
                        if (multi) {
                                /* point-major copy */
                                int ab;
                                if (mfl) {
                                        float *at = Abf + aoff;
                                        for (ab = 0; ab < nab; ab++) {
                                                const double *ar = A + (size_t)ab * n;
                                                for (g = 0; g < n; g++) at[(size_t)g * nab + ab] = (float)ar[g];
                                        }
                                } else {
                                        double *at = Abuf + aoff;
                                        for (ab = 0; ab < nab; ab++) {
                                                const double *ar = A + (size_t)ab * n;
                                                for (g = 0; g < n; g++) at[(size_t)g * nab + ab] = ar[g];
                                        }
                                }
                                rec[nrec].I0 = I0;
                                rec[nrec].J0 = J0;
                                rec[nrec].nI = nI;
                                rec[nrec].nJ = nJ;
                                rec[nrec].cI = cI2;
                                rec[nrec].cJ = cJ2;
                                rec[nrec].off = aoff;
                                nrec++;
                                aoff += (size_t)nab * n;
                                continue;
                        }
                        if (eonly) {
                                for (d = 0; d < nset; d++) {
                                        const double *Fd = F + (size_t)d * nao * n, *Hd = H + (size_t)d * nao * n;
                                        const int both = pp->I != pp->J;
                                        int ia, jb;
                                        double e = 0;
                                        for (ia = 0; ia < nI; ia++) {
                                                const double *hi = Hd + (size_t)(I0 + ia) * n, *fi = Fd + (size_t)(I0 + ia) * n;
                                                for (jb = 0; jb < nJ; jb++) {
                                                        const double *ar = A + (size_t)(ia * nJ + jb) * n;
                                                        const double *fj = Fd + (size_t)(J0 + jb) * n;
                                                        const double *hj = Hd + (size_t)(J0 + jb) * n;
                                                        if (both) {
#pragma omp simd reduction(+:e)
                                                                for (g = 0; g < n; g++) e += ar[g] * (hi[g] * fj[g] + hj[g] * fi[g]);
                                                        } else {
#pragma omp simd reduction(+:e)
                                                                for (g = 0; g < n; g++) e += ar[g] * hi[g] * fj[g];
                                                        }
                                                }
                                        }
                                        esum[d] += e;
                                }
                                PROF(st[26] += __rdtsc() - cc0);
                                continue;
                        }
                        for (d = 0; d < nset; d++) {
                                const double *Fd = F + (size_t)d * nao * n;
                                double *Gd = G + (size_t)d * nao * n;
                                int ia, jb;
                                if (cI2) {
                                        for (ia = 0; ia < nI; ia++) {
                                                double *gi = Gd + (size_t)(I0 + ia) * n;
                                                for (jb = 0; jb < nJ; jb++) {
                                                        const double *ar = A + (size_t)(ia * nJ + jb) * n;
                                                        const double *fj = Fd + (size_t)(J0 + jb) * n;
#pragma omp simd
                                                        for (g = 0; g < n; g++) gi[g] += ar[g] * fj[g];
                                                }
                                        }
                                }
                                if (cJ2) {
                                        for (jb = 0; jb < nJ; jb++) {
                                                double *gj = Gd + (size_t)(J0 + jb) * n;
                                                for (ia = 0; ia < nI; ia++) {
                                                        const double *ar = A + (size_t)(ia * nJ + jb) * n;
                                                        const double *fi = Fd + (size_t)(I0 + ia) * n;
#pragma omp simd
                                                        for (g = 0; g < n; g++) gj[g] += ar[g] * fi[g];
                                                }
                                        }
                                }
                        }
                        PROF(st[26] += __rdtsc() - cc0);
                }
                if (multi && nrec) {
#ifdef CX_FUSED
                        if (mfl) multi_contract_f(Gf, Ff, Abf, rec, nrec, nao, n, nset);
                        else
#endif
                        multi_contract(G, F, Abuf, rec, nrec, nao, n, nset);
                }
                if (mfl) {
                        size_t q;
                        for (q = 0; q < NQ * n; q++) G[q] = Gf[q];
                }
                tt1 = omp_get_wtime(); st[1] += tt1 - tt0; tt0 = tt1;
                if (eonly) continue;
                if (multi) {
                        /* K[ix_i][nu][d] += sum_g wx_i(g) G[g][nu][d]: GEMMs of CX_KROWS rows into Bm, then
                         * the rows added to the thread's K */
                        const int nq = (int)NQ;
                        int r0;
                        for (r0 = 0; r0 < nl; r0 += CX_KROWS) {
                                const int nr = nl - r0 < CX_KROWS ? nl - r0 : CX_KROWS;
                                dgemm_("N", "N", &nq, &nr, &n, &one, G, &nq, wx + (size_t)r0 * n, &n, &zero, Bm, &nq);
                                for (i = 0; i < nr; i++) {
                                        double *row = Kt + (size_t)ix[r0 + i] * NQ;
                                        const double *src = Bm + (size_t)i * NQ;
                                        size_t q;
                                        for (q = 0; q < NQ; q++) row[q] += src[q];
                                }
                        }
                        st[3] += omp_get_wtime() - tt0;
                        continue;
                }
                /* K[ix, :] += wx G^T */
                for (d = 0; d < nset; d++) {
                        const double *Gd = G + (size_t)d * nao * n;
                        double *Kd = Kt + nn * d;
                        dgemm_("T", "N", &nao, &nl, &n, &one, Gd, &n, wx, &n, &zero, ktmp, &nao);
                        for (i = 0; i < nl; i++) {
                                double *row = Kd + (size_t)ix[i] * nao;
                                const double *src = ktmp + (size_t)i * nao;
                                int nu;
                                for (nu = 0; nu < nao; nu++) row[nu] += src[nu];
                        }
                }
                st[3] += omp_get_wtime() - tt0;
                st[7] += nl;
        }
#pragma omp critical(socdft_cosx_stat)
{
        if (eonly)
                for (i = 0; i < nset; i++) eout[i] += esum[i];
        for (i = 0; i < 32; i++) cx_stat[i] += st[i];
}
#pragma omp barrier
        /* reduce the thread buffers, each thread a slice */
        if (!eonly) {
                const size_t tot = nn * nset, chunk = (tot + nthreads - 1) / nthreads;
                const size_t s0 = chunk * tid < tot ? chunk * tid : tot, s1 = s0 + chunk < tot ? s0 + chunk : tot;
                size_t q;
                int t;
                for (t = 0; t < nthreads; t++) {
                        const double *kb = kbufs[t];
                        if (!kb) continue;
                        for (q = s0; q < s1; q++) vk[q] += kb[q];
                }
        }
#pragma omp barrier
        free(Kt); free(H); free(xs); free(a); free(wx); free(rmx); free(ix); free(runs); free(dr); free(F); free(G); free(ktmp);
        free(fmax); free(A0); free(bk); free(Bm); free(Abuf); free(rec); free(Ff); free(Gf); free(R); free(Rwk); free(S); free(X); free(xyz);
        for (at = 0; at < natm; at++) free(farR[at]);
        free(farR); free(stamp); free(armin);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
        free(kbufs);
        free(shl_of);
}

/* ---------------------------------------------------------------- nuclear gradient */

/* the derivative of the spherical E of primitive pair (ip, jp) of shells I, J with respect to the
 * centre of I: out[x][ab][t] (3, nI nJ, NH(L + 1)), d phi_a / dA_x = 2 alpha phi_{a+1x} - a_x phi_{a-1x},
 * coefficients and 2 pi / p folded in as prim_E */
static void prim_Ed(double *out, int I, int J, int ip, int jp, int *atm, int *bas, double *env, double *wk)
{
        const int *bi = bas + I * BAS_SLOTS, *bj = bas + J * BAS_SLOTS;
        const int li = bi[ANG_OF], lj = bj[ANG_OF], L = li + lj, nh1 = NH(L + 1);
        const int npi = bi[NPRIM_OF], npj = bj[NPRIM_OF], nci = bi[NCTR_OF], ncj = bj[NCTR_OF];
        const int di = 2 * li + 1, dj = 2 * lj + 1, nJ = ncj * dj, nI = nci * di;
        const int nfi = (li + 1) * (li + 2) / 2, nfj = (lj + 1) * (lj + 2) / 2;
        const double ai = env[bi[PTR_EXP] + ip], aj = env[bj[PTR_EXP] + jp];
        const double *Ra = env + atm[bi[ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double *Rb = env + atm[bj[ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double p = ai + aj;
        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
        const double e0 = 2 * M_PI / p * exp(-ai * aj / p * r2) * common_fac_sp(li) * common_fac_sp(lj);
        const int Li = li + 1, s2 = Li + lj + 1, s1 = (lj + 1) * s2, sz = (Li + 1) * s1;
        double *Ex = wk, *Ey = Ex + sz, *Ez = Ey + sz;
        double *Ec = Ez + sz;                                   /* (3, nfi, nfj, nh1) */
        double *Eh = Ec + (size_t)3 * nfi * nfj * nh1;          /* (3, di, nfj, nh1) */
        double *Es = Eh + (size_t)3 * di * nfj * nh1;           /* (3, di, dj, nh1) */
        hermite_E(Ex, Li, lj, ai, aj, Ra[0], Rb[0], 1.);
        hermite_E(Ey, Li, lj, ai, aj, Ra[1], Rb[1], 1.);
        hermite_E(Ez, Li, lj, ai, aj, Ra[2], Rb[2], e0);
        int ix, iy, jx, jy, n, fa = 0, x;
        for (ix = li; ix >= 0; ix--)
        for (iy = li - ix; iy >= 0; iy--, fa++) {
                const int iz = li - ix - iy, ia[3] = {ix, iy, iz};
                int fb = 0;
                for (jx = lj; jx >= 0; jx--)
                for (jy = lj - jx; jy >= 0; jy--, fb++) {
                        const int jz = lj - jx - jy;
                        const double *ex = Ex + ix * s1 + jx * s2, *ey = Ey + iy * s1 + jy * s2;
                        const double *ez = Ez + iz * s1 + jz * s2;
                        const double *exp_[3] = {Ex + (ix + 1) * s1 + jx * s2, Ey + (iy + 1) * s1 + jy * s2,
                                                 Ez + (iz + 1) * s1 + jz * s2};
                        const double *exm[3] = {ix ? Ex + (ix - 1) * s1 + jx * s2 : NULL,
                                                iy ? Ey + (iy - 1) * s1 + jy * s2 : NULL,
                                                iz ? Ez + (iz - 1) * s1 + jz * s2 : NULL};
                        for (x = 0; x < 3; x++) {
                                double *e = Ec + (((size_t)x * nfi + fa) * nfj + fb) * nh1;
                                for (n = 0; n < nh1; n++) {
                                        const int t = h_t[n], u = h_u[n], v = h_v[n];
                                        const int tt[3] = {t, u, v};
                                        const double base[3] = {ex[t], ey[u], ez[v]};
                                        /* the factor of direction x differentiated */
                                        double dfx = 2 * ai * exp_[x][tt[x]];
                                        if (ia[x]) dfx -= ia[x] * exm[x][tt[x]];
                                        double val = dfx;
                                        int y;
                                        for (y = 0; y < 3; y++)
                                                if (y != x) val *= base[y];
                                        e[n] = val;
                                }
                        }
                }
        }
        /* note: the E^{ij}_t of the undifferentiated directions vanish for t > i + j; the tables
         * from hermite_E (memset) hold zeros there up to the raised length s2 */
        int m, f, k, nc;
        for (x = 0; x < 3; x++) {
                const double *Ecx = Ec + (size_t)x * nfi * nfj * nh1;
                double *Ehx = Eh + (size_t)x * di * nfj * nh1, *Esx = Es + (size_t)x * di * dj * nh1;
                if (li <= 1) {
                        memcpy(Ehx, Ecx, sizeof(double) * nfi * nfj * nh1);
                } else {
                        for (m = 0; m < di; m++) {
                                const double *c = c2s_row(li, m, &nc);
                                double *o = Ehx + (size_t)m * nfj * nh1;
                                memset(o, 0, sizeof(double) * nfj * nh1);
                                for (f = 0; f < nfi; f++) {
                                        if (c[f] == 0) continue;
                                        const double *s_ = Ecx + (size_t)f * nfj * nh1;
                                        for (k = 0; k < nfj * nh1; k++) o[k] += c[f] * s_[k];
                                }
                        }
                }
                if (lj <= 1) {
                        memcpy(Esx, Ehx, sizeof(double) * di * nfj * nh1);
                } else {
                        for (fa = 0; fa < di; fa++)
                                for (m = 0; m < dj; m++) {
                                        const double *c = c2s_row(lj, m, &nc);
                                        double *o = Esx + ((size_t)fa * dj + m) * nh1;
                                        memset(o, 0, sizeof(double) * nh1);
                                        for (f = 0; f < nfj; f++) {
                                                if (c[f] == 0) continue;
                                                const double *s_ = Ehx + ((size_t)fa * nfj + f) * nh1;
                                                for (k = 0; k < nh1; k++) o[k] += c[f] * s_[k];
                                        }
                                }
                }
        }
        const double *cfi = env + bi[PTR_COEFF], *cfj = env + bj[PTR_COEFF];
        int ci, cj, mi, mj;
        for (x = 0; x < 3; x++)
        for (ci = 0; ci < nci; ci++)
        for (mi = 0; mi < di; mi++)
        for (cj = 0; cj < ncj; cj++)
        for (mj = 0; mj < dj; mj++) {
                const double cc = cfi[ci * npi + ip] * cfj[cj * npj + jp];
                const double *s_ = Es + (((size_t)x * di + mi) * dj + mj) * nh1;
                double *o = out + (((size_t)x * nI + ci * di + mi) * nJ + cj * dj + mj) * nh1;
                for (k = 0; k < nh1; k++) o[k] = cc * s_[k];
        }
}

static size_t prim_wk_size_d(void)
{
        const int sE = (CX_LMAX + 2) * (CX_LMAX + 1) * (2 * CX_LMAX + 2);
        const int nc = (CX_LMAX + 1) * (CX_LMAX + 2) / 2, nh = NH(2 * CX_LMAX + 1);
        return 3 * sE + 9 * (size_t)nc * nc * nh;
}

/* the derivative tables of the two-centre pairs (one-centre pairs need none: the sum of the two
 * centre derivatives is minus the point derivative); returns 0 on failure */
int SOCDFTcosx_build_grad(void *p, int *atm, int *bas, double *env)
{
        CXOpt *opt = p;
        if (opt->eddata) return 1;
        cx_tables();
        size_t eo = 0;
        int ip_;
        opt->edo = malloc(sizeof(size_t) * (opt->npair + 1));
        for (ip_ = 0; ip_ < opt->npair; ip_++) {
                const CXPair *pp = opt->pairs + ip_;
                opt->edo[ip_] = eo;
                if (!pp->onec && pp->bound > 0) eo += (size_t)3 * pp->nI * pp->nJ * NH(pp->L + 1) * pp->np;
        }
        opt->eddata = malloc(sizeof(double) * (eo + 1));
        if (!opt->eddata) return 0;
#pragma omp parallel
{
        double *wk = malloc(sizeof(double) * prim_wk_size_d());
#pragma omp for schedule(dynamic, 8)
        for (ip_ = 0; ip_ < opt->npair; ip_++) {
                const CXPair *pp = opt->pairs + ip_;
                if (pp->onec || pp->bound == 0) continue;
                const int *bi = bas + pp->I * BAS_SLOTS, *bj = bas + pp->J * BAS_SLOTS;
                const size_t sz = (size_t)3 * pp->nI * pp->nJ * NH(pp->L + 1);
                const int npj = bj[NPRIM_OF];
                int k;
                for (k = 0; k < pp->np; k++) {
                        const int id = opt->pp_id[pp->po + k];
                        prim_Ed(opt->eddata + opt->edo[ip_] + sz * k, pp->I, pp->J, id / npj, id % npj, atm, bas, env, wk);
                }
                (void)bi;
        }
        free(wk);
}
        return 1;
}

/* Mm (nab, nh) = W (nab, n) R (nh, n)^T: dot products for the small cases (a GEMM call costs more
 * there), GEMM beyond */
static void w_dot_r(double *Mm, const double *W, const double *R, int nab, int nh, int n, dgemm_t dgemm_)
{
        if ((size_t)nab * nh >= 256) {
                const double one = 1, zero = 0;
                dgemm_("T", "N", &nh, &nab, &n, &one, R, &n, W, &n, &zero, Mm, &nh);
                return;
        }
        int ab, t, g;
        for (ab = 0; ab < nab; ab++) {
                const double *w = W + (size_t)ab * n;
                for (t = 0; t < nh; t++) {
                        const double *r = R + (size_t)t * n;
                        double s = 0;
#pragma omp simd reduction(+:s)
                        for (g = 0; g < n; g++) s += w[g] * r[g];
                        Mm[(size_t)ab * nh + t] = s;
                }
        }
}

/* AO values and gradients (4, nl, n) of the significant AOs of a block (max_g |phi| > cutoff) */
static int eval_select1(double *a, int *ix, int *runs, int n, double *xyz, uint8_t *msk, double cutoff,
                        int *ao_loc, int *atm, int *bas, int nbas, double *env, double *rmx, double *full)
{
        const int nrow = (n + MASK_BLK - 1) / MASK_BLK;
        int sh = 0, nrun = 0, r, i, c;
        size_t ntot = 0;
        while (sh < nbas) {
                int on = 0;
                for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                if (!on) { sh++; continue; }
                const int s0 = sh;
                for (sh++; sh < nbas; sh++) {
                        on = 0;
                        for (r = 0; r < nrow && !on; r++) on = msk[(size_t)r * nbas + sh] != 0;
                        if (!on) break;
                }
                runs[2 * nrun] = s0;
                runs[2 * nrun + 1] = sh;
                nrun++;
                ntot += ao_loc[sh] - ao_loc[s0];
        }
        if (ntot == 0) return 0;
        size_t row = 0;
        memset(rmx, 0, sizeof(double) * ntot);
        for (i = 0; i < nrun; i++) {
                SOCDFTgtoval_ld(1, n, runs[2 * i], runs[2 * i + 1], ao_loc, full, ntot, row, xyz, msk, atm, bas,
                                nbas, env, rmx);
                row += ao_loc[runs[2 * i + 1]] - ao_loc[runs[2 * i]];
        }
        int nl = 0;
        row = 0;
        int *src = runs + 2 * nbas;
        for (i = 0; i < nrun; i++) {
                int mu;
                for (mu = ao_loc[runs[2 * i]]; mu < ao_loc[runs[2 * i + 1]]; mu++, row++) {
                        if (rmx[row] > cutoff) {
                                src[nl] = (int)row;
                                ix[nl++] = mu;
                        }
                }
        }
        for (c = 0; c < 4; c++)
                for (i = 0; i < nl; i++)
                        memcpy(a + ((size_t)c * nl + i) * n, full + ((size_t)c * ntot + src[i]) * n, sizeof(double) * n);
        return nl;
}

/*
 * grad (natm, 3) = d/dR of E = sum_g w_g sum_ab H_a(g) A_ab(g) F_b(g) at fixed densities, on the
 * fixed grid: H = M^T phi (grid side), F = D^T phi (D symmetric), plus, with Yg != NULL, the
 * derivative of sum_mu nu Yg_mu nu S_g,mu nu (S_g = sum_g w phi phi^T):
 *   (i)   -sum_{rho in X} sum_g w grad phi_rho (sum_nu M_rho nu (A F)_nu + D_rho nu (A H)_nu + Yg_rho nu phi_nu)
 *   (ii)  sum_pairs (d A_ab / dR) W_ab, W_ab = w (H_a F_b + H_b F_a), the centre-of-I derivative from
 *         the derivative E tables, the point derivative from R_{t+1x}, the centre of J by
 *         translational invariance (one-centre pairs: minus the point derivative).
 */
void SOCDFTcosx_grad(double *grad, double *dm, double *dmm, double *yg, void *optp, double *coords,
                     double *weights, uint8_t *non0tab, int ngrid, int blk, double cutoff, double tol,
                     double tcut, int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env,
                     dgemm_t dgemm_, mkl_local_t mkl_set_num_threads_local)
{
        CXOpt *opt = optp;
        cx_tables();
        const int nao = ao_loc[nbas];
        const int nblk = (ngrid + blk - 1) / blk;
        const int Lmax = opt->lmax + 1, nhmax = NH(Lmax);
        int nab_max = 1, i0;
        for (i0 = 0; i0 < opt->npair; i0++)
                if (opt->pairs[i0].nI * opt->pairs[i0].nJ > nab_max) nab_max = opt->pairs[i0].nI * opt->pairs[i0].nJ;
        int *shl_of = malloc(sizeof(int) * nao);
        for (i0 = 0; i0 < nbas; i0++) {
                int mu;
                for (mu = ao_loc[i0]; mu < ao_loc[i0 + 1]; mu++) shl_of[mu] = i0;
        }
        int *atm_of = malloc(sizeof(int) * nbas);
        for (i0 = 0; i0 < nbas; i0++) atm_of[i0] = bas[i0 * BAS_SLOTS + ATOM_OF];
        memset(grad, 0, sizeof(double) * natm * 3);
#pragma omp parallel
{
        const int mkl0 = mkl_set_num_threads_local ? mkl_set_num_threads_local(1) : 0;
        const double one = 1, zero = 0;
        double *gr = calloc(natm * 3, sizeof(double));
        double *a = malloc(sizeof(double) * 4 * (size_t)nao * blk);
        double *full = malloc(sizeof(double) * 4 * (size_t)nao * blk);
        double *rmx = malloc(sizeof(double) * nao);
        int *ix = malloc(sizeof(int) * nao);
        int *runs = malloc(sizeof(int) * (2 * (nbas + 1) + nao));
        double *dr = malloc(sizeof(double) * (size_t)nao * nao);
        double *F = malloc(sizeof(double) * (size_t)nao * blk);
        double *H = malloc(sizeof(double) * (size_t)nao * blk);
        double *GF = malloc(sizeof(double) * (size_t)nao * blk);
        double *GH = malloc(sizeof(double) * (size_t)nao * blk);
        double *Q = malloc(sizeof(double) * (size_t)nao * blk);
        double *fm = malloc(sizeof(double) * nbas), *ws = malloc(sizeof(double) * nbas);
        double *A = malloc(sizeof(double) * (size_t)nab_max * blk);
        double *W = malloc(sizeof(double) * (size_t)nab_max * blk);
        double *Mm = malloc(sizeof(double) * (size_t)nab_max * nhmax);
        double *bk = malloc(sizeof(double) * 4096);
        double *R = malloc(sizeof(double) * (size_t)nhmax * blk);
        double *Rwk = malloc(sizeof(double) * 2 * (size_t)nhmax * blk);
        double *S = malloc(sizeof(double) * (size_t)(Lmax + 1) * blk);
        double *X = malloc(sizeof(double) * 4 * blk), *Y = X + blk, *Z = Y + blk, *T = Z + blk;
        double *xyz = malloc(sizeof(double) * 3 * blk);
        double **farR = calloc(natm, sizeof(double *));
        int *stamp = malloc(sizeof(int) * natm);
        double *armin = malloc(sizeof(double) * natm);
        int at, i;
        for (at = 0; at < natm; at++) {
                stamp[at] = -1;
                if (opt->atm_lmax[at] >= 0) farR[at] = malloc(sizeof(double) * NH(opt->atm_lmax[at] + 1) * blk);
        }
        int b;
#pragma omp for schedule(dynamic, 1)
        for (b = 0; b < nblk; b++) {
                const int g0 = b * blk, n = g0 + blk <= ngrid ? blk : ngrid - g0;
                uint8_t *msk = non0tab + (size_t)(g0 / MASK_BLK) * nbas;
                int k, g, ip, x;
                for (k = 0; k < 3; k++) memcpy(xyz + k * n, coords + (size_t)k * ngrid + g0, sizeof(double) * n);
                const int nl = eval_select1(a, ix, runs, n, xyz, msk, cutoff, ao_loc, atm, bas, nbas, env, rmx, full);
                if (nl == 0) continue;
                const double *w = weights + g0;
                /* F = D[ix, :]^T phi, H = M[ix, :]^T phi (nao, n) */
                for (i = 0; i < nl; i++) memcpy(dr + (size_t)i * nao, dm + (size_t)ix[i] * nao, sizeof(double) * nao);
                dgemm_("N", "T", &n, &nao, &nl, &one, a, &n, dr, &nao, &zero, F, &n);
                for (i = 0; i < nl; i++) memcpy(dr + (size_t)i * nao, dmm + (size_t)ix[i] * nao, sizeof(double) * nao);
                dgemm_("N", "T", &n, &nao, &nl, &one, a, &n, dr, &nao, &zero, H, &n);
                for (i = 0; i < nbas; i++) fm[i] = ws[i] = 0;
                {
                        int mu;
                        for (mu = 0; mu < nao; mu++) {
                                const double *f = F + (size_t)mu * n, *h = H + (size_t)mu * n;
                                double m = 0, s = 0;
                                for (g = 0; g < n; g++) {
                                        const double v = fabs(f[g]) + fabs(h[g]);
                                        m = v > m ? v : m;
                                        s += w[g] * v;
                                }
                                const int sh = shl_of[mu];
                                if (m > fm[sh]) fm[sh] = m;
                                if (s > ws[sh]) ws[sh] = s;
                        }
                }
                memset(GF, 0, sizeof(double) * nao * n);
                memset(GH, 0, sizeof(double) * nao * n);
                double cb[3] = {0, 0, 0}, rb = 0;
                for (g = 0; g < n; g++) {
                        cb[0] += xyz[g]; cb[1] += xyz[n + g]; cb[2] += xyz[2 * n + g];
                }
                cb[0] /= n; cb[1] /= n; cb[2] /= n;
                for (g = 0; g < n; g++) {
                        const double dx = xyz[g] - cb[0], dy = xyz[n + g] - cb[1], dz = xyz[2 * n + g] - cb[2];
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 > rb) rb = r2;
                }
                rb = sqrt(rb);
                for (ip = 0; ip < opt->npair; ip++) {
                        const CXPair *pp = opt->pairs + ip;
                        const int I = pp->I, J = pp->J, both = I != J;
                        const double wgt = fm[J] * ws[I] + (both ? fm[I] * ws[J] : 0);
                        if (pp->bound == 0 || pp->bound * wgt < tol) continue;      /* bound 0: below PAIR_TOL */
                        const int L = pp->L, nh = NH(L), nh1 = NH(L + 1), nI = pp->nI, nJ = pp->nJ, nab = nI * nJ;
                        double bsum = 0;
                        for (k = 0; k < pp->np; k++) {
                                const double *P = opt->pp_P + 3 * (pp->po + k);
                                const double dx = P[0] - cb[0], dy = P[1] - cb[1], dz = P[2] - cb[2];
                                const double dmin = sqrt(dx * dx + dy * dy + dz * dz) - rb;
                                double f = 1;
                                if (dmin > 0) {
                                        const double T0 = opt->pp_p[pp->po + k] * dmin * dmin;
                                        if (T0 > 1) {
                                                const double ir = 1 / sqrt(T0);
                                                double rp = ir, mx = 0;
                                                int dd;
                                                for (dd = 0; dd <= L + 1; dd++) {
                                                        const double v = cx_damp[dd] * rp;
                                                        if (v > mx) mx = v;
                                                        rp *= ir;
                                                }
                                                f = mx < 1 ? mx : 1;
                                        }
                                }
                                /* the derivative brings sqrt(2p) */
                                bk[k] = f * opt->pp_b[pp->po + k] * sqrt(2 * opt->pp_p[pp->po + k]);
                                bsum += bk[k];
                        }
                        if (bsum * wgt < tol) continue;
                        const size_t sz = (size_t)nab * nh, szd = (size_t)3 * nab * nh1;
                        const int I0 = ao_loc[I], J0 = ao_loc[J];
                        /* W_ab = w (H_a F_b + H_b F_a) */
                        {
                                int ia, jb;
                                for (ia = 0; ia < nI; ia++)
                                        for (jb = 0; jb < nJ; jb++) {
                                                double *wr = W + (size_t)(ia * nJ + jb) * n;
                                                const double *ha = H + (size_t)(I0 + ia) * n, *fb = F + (size_t)(J0 + jb) * n;
                                                const double *hb = H + (size_t)(J0 + jb) * n, *fa = F + (size_t)(I0 + ia) * n;
                                                if (both) {
#pragma omp simd
                                                        for (g = 0; g < n; g++) wr[g] = w[g] * (ha[g] * fb[g] + hb[g] * fa[g]);
                                                } else {
#pragma omp simd
                                                        for (g = 0; g < n; g++) wr[g] = w[g] * ha[g] * fb[g];
                                                }
                                        }
                        }
                        double gA[3] = {0, 0, 0}, gC[3] = {0, 0, 0};
                        memset(A, 0, sizeof(double) * nab * n);
                        int k0 = 0;
                        if (pp->onec) {
                                at = pp->atom;
                                const double *Ra = env + atm[at * ATM_SLOTS + PTR_COORD];
                                const int La = opt->atm_lmax[at] + 1;
                                if (stamp[at] != b) {
                                        double rmin = 1e300;
                                        for (g = 0; g < n; g++) {
                                                X[g] = Ra[0] - xyz[g];
                                                Y[g] = Ra[1] - xyz[n + g];
                                                Z[g] = Ra[2] - xyz[2 * n + g];
                                                T[g] = X[g] * X[g] + Y[g] * Y[g] + Z[g] * Z[g];
                                                if (T[g] < rmin) rmin = T[g];
                                        }
                                        armin[at] = rmin;
                                        stamp[at] = b;
                                        if (rmin > 1e-12) {
                                                int m;
                                                rsqrt_vec(S + (size_t)La * n, T, n);
                                                for (g = 0; g < n; g++) {
                                                        const double ir = S[(size_t)La * n + g], ir2 = ir * ir;
                                                        double s = ir;
                                                        for (m = 0; m <= La; m++) {
                                                                S[m * n + g] = s;
                                                                s *= -(2 * m + 1) * ir2;
                                                        }
                                                }
                                                hermite_R_vec(farR[at], S, X, Y, Z, La, n, Rwk);
                                        } else {
                                                armin[at] = 0;
                                        }
                                }
                                const double rmin = armin[at];
                                if (rmin > 0) {
                                        const double *pv = opt->pp_p + pp->po;
                                        while (k0 < pp->np && pv[k0] * rmin >= tcut) k0++;
                                }
                                if (k0 > 0) {
                                        const double *Mo = opt->mdata + pp->mo + sz * (k0 - 1);
                                        e_times_r(A, Mo, farR[at], nab, nh, n);
                                        /* Mm (nab, nh1) = W R^T */
                                        w_dot_r(Mm, W, farR[at], nab, nh1, n, dgemm_);
                                        int ab, t;
                                        for (ab = 0; ab < nab; ab++)
                                                for (t = 0; t < nh; t++) {
                                                        const double e = Mo[(size_t)ab * nh + t];
                                                        const int tx = h_t[t], ty = h_u[t], tz = h_v[t];
                                                        gC[0] -= e * Mm[(size_t)ab * nh1 + h_idx[tx + 1][ty][tz]];
                                                        gC[1] -= e * Mm[(size_t)ab * nh1 + h_idx[tx][ty + 1][tz]];
                                                        gC[2] -= e * Mm[(size_t)ab * nh1 + h_idx[tx][ty][tz + 1]];
                                                }
                                }
                        }
                        for (k = k0; k < pp->np; k++) {
                                if (bk[k] * wgt < tol) continue;
                                const double p = opt->pp_p[pp->po + k];
                                const double *P = opt->pp_P + 3 * (pp->po + k);
                                const double *E = opt->edata + pp->eo + sz * k;
#ifdef CX_FUSED
                                /* R to L + 1 stored, A += E R, in one register pass */
                                cxs_table[L](A, nab, E, p, P, xyz, n, R);
#else
                                double tmin = 1e300, tmax = 0;
                                for (g = 0; g < n; g++) {
                                        X[g] = P[0] - xyz[g];
                                        Y[g] = P[1] - xyz[n + g];
                                        Z[g] = P[2] - xyz[2 * n + g];
                                        T[g] = p * (X[g] * X[g] + Y[g] * Y[g] + Z[g] * Z[g]);
                                        tmin = T[g] < tmin ? T[g] : tmin;
                                        tmax = T[g] > tmax ? T[g] : tmax;
                                }
                                boys_vec(S, n, T, L + 1, n, tmin, tmax);
                                double sc = 1;
                                int m;
                                for (m = 1; m <= L + 1; m++) {
                                        sc *= -2 * p;
                                        double *sm = S + (size_t)m * n;
                                        for (g = 0; g < n; g++) sm[g] *= sc;
                                }
                                hermite_R_vec(R, S, X, Y, Z, L + 1, n, Rwk);
                                e_times_r(A, E, R, nab, nh, n);
#endif
                                w_dot_r(Mm, W, R, nab, nh1, n, dgemm_);
                                int ab, t;
                                for (ab = 0; ab < nab; ab++)
                                        for (t = 0; t < nh; t++) {
                                                const double e = E[(size_t)ab * nh + t];
                                                const int tx = h_t[t], ty = h_u[t], tz = h_v[t];
                                                gC[0] -= e * Mm[(size_t)ab * nh1 + h_idx[tx + 1][ty][tz]];
                                                gC[1] -= e * Mm[(size_t)ab * nh1 + h_idx[tx][ty + 1][tz]];
                                                gC[2] -= e * Mm[(size_t)ab * nh1 + h_idx[tx][ty][tz + 1]];
                                        }
                                if (!pp->onec) {
                                        const double *Ed = opt->eddata + opt->edo[ip] + szd * k;
                                        for (x = 0; x < 3; x++) {
                                                const double *e = Ed + (size_t)x * nab * nh1;
                                                double s = 0;
                                                size_t q;
                                                for (q = 0; q < (size_t)nab * nh1; q++) s += e[q] * Mm[q];
                                                gA[x] += s;
                                        }
                                }
                        }
                        const int aI = atm_of[I], aJ = atm_of[J];
                        for (x = 0; x < 3; x++) {
                                if (pp->onec) {
                                        gr[aI * 3 + x] -= gC[x];
                                } else {
                                        gr[aI * 3 + x] += gA[x];
                                        gr[aJ * 3 + x] += -gA[x] - gC[x];
                                }
                        }
                        /* G^F = A F, G^H = A H (both directions) */
                        {
                                int ia, jb;
                                for (ia = 0; ia < nI; ia++) {
                                        double *gfi = GF + (size_t)(I0 + ia) * n, *ghi = GH + (size_t)(I0 + ia) * n;
                                        for (jb = 0; jb < nJ; jb++) {
                                                const double *ar = A + (size_t)(ia * nJ + jb) * n;
                                                const double *fj = F + (size_t)(J0 + jb) * n, *hj = H + (size_t)(J0 + jb) * n;
#pragma omp simd
                                                for (g = 0; g < n; g++) {
                                                        gfi[g] += ar[g] * fj[g];
                                                        ghi[g] += ar[g] * hj[g];
                                                }
                                        }
                                }
                                if (both) {
                                        for (jb = 0; jb < nJ; jb++) {
                                                double *gfj = GF + (size_t)(J0 + jb) * n, *ghj = GH + (size_t)(J0 + jb) * n;
                                                for (ia = 0; ia < nI; ia++) {
                                                        const double *ar = A + (size_t)(ia * nJ + jb) * n;
                                                        const double *fi = F + (size_t)(I0 + ia) * n, *hi = H + (size_t)(I0 + ia) * n;
#pragma omp simd
                                                        for (g = 0; g < n; g++) {
                                                                gfj[g] += ar[g] * fi[g];
                                                                ghj[g] += ar[g] * hi[g];
                                                        }
                                                }
                                        }
                                }
                        }
                }
                /* (i): Q (nl, n) = M[ix, :] G^F + D[ix, :] G^H (+ Yg[ix, ix] phi) */
                for (i = 0; i < nl; i++) memcpy(dr + (size_t)i * nao, dmm + (size_t)ix[i] * nao, sizeof(double) * nao);
                dgemm_("N", "N", &n, &nl, &nao, &one, GF, &n, dr, &nao, &zero, Q, &n);
                for (i = 0; i < nl; i++) memcpy(dr + (size_t)i * nao, dm + (size_t)ix[i] * nao, sizeof(double) * nao);
                dgemm_("N", "N", &n, &nl, &nao, &one, GH, &n, dr, &nao, &one, Q, &n);
                if (yg) {
                        int j;
                        for (i = 0; i < nl; i++)
                                for (j = 0; j < nl; j++) dr[(size_t)i * nl + j] = yg[(size_t)ix[i] * nao + ix[j]];
                        dgemm_("N", "N", &n, &nl, &nl, &one, a, &n, dr, &nl, &one, Q, &n);
                }
                for (i = 0; i < nl; i++) {
                        const double *q = Q + (size_t)i * n;
                        const int aa = atm_of[shl_of[ix[i]]];
                        for (x = 0; x < 3; x++) {
                                const double *d = a + ((size_t)(1 + x) * nl + i) * n;
                                double s = 0;
                                for (g = 0; g < n; g++) s += w[g] * d[g] * q[g];
                                gr[aa * 3 + x] -= s;
                        }
                }
        }
#pragma omp critical(socdft_cosx_grad)
        for (i = 0; i < natm * 3; i++) grad[i] += gr[i];
        free(gr); free(a); free(full); free(rmx); free(ix); free(runs); free(dr); free(F); free(H); free(GF); free(GH);
        free(Q); free(fm); free(ws); free(A); free(W); free(Mm); free(bk); free(R); free(Rwk); free(S); free(X);
        free(xyz);
        for (at = 0; at < natm; at++) free(farR[at]);
        free(farR); free(stamp); free(armin);
        if (mkl_set_num_threads_local) mkl_set_num_threads_local(mkl0);
}
        free(shl_of);
        free(atm_of);
}

#include "cosx_hess.c"
