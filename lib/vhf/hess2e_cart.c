/*
 * Second nuclear derivatives of the Coulomb / exchange energies at a fixed
 * symmetric density, all derivative components of a shell quartet from one
 * Rys quadrature, contracted with the density on the fly (nothing is
 * written out), in the Cartesian basis (the caller transforms the density,
 * D_cart = C D C^T, so no Cartesian-to-spherical transformation of the 54
 * derivative components is needed):
 *
 *     E_g = jfac_g E_J + kfac_g E_K,
 *     E_J = 1/2 sum (ab|cd) D_ab D_cd,   E_K = 1/4 sum (ab|cd) D_ad D_bc.
 *
 * Per unique shell quartet (i >= j, k >= l, ij >= kl) the density weight
 * symmetrized over the eight permutations is dd = 4 jfac D_ab D_cd
 * + kfac (D_ac D_bd + D_ad D_bc).  Per primitive quartet: one set of Rys
 * roots for L + 2, one VRR table per root with the bra (i + j) and the ket
 * (k + l) raised by two, HRR to I[i][j][k][l] with i, j, k raised by up to
 * two, and per Cartesian direction a table of the 1D values (all of it for
 * VL = 8 (primitive quartet, root) lanes at once, the lane index innermost,
 * runtime-dispatched to AVX-512 / AVX2 / baseline code).  The lanes of a
 * flush may come from different shell quartets of the same angular class
 * (l_i, l_j, l_k, l_l): each thread keeps one lane buffer per class with the
 * geometry and density weights per lane, adds the lane sums of a flush to
 * the shell quartets they belong to, and finalizes a shell quartet when its
 * last lane has been flushed (an s/p shell quartet has only a few lanes, so
 * per-quartet flushes ran mostly empty)
 *
 *     I, d_A I, d_B I, d_C I, d_AA I, d_BB I, d_CC I, d_AB I, d_AC I, d_BC I
 *
 * (d_A phi_i = 2 a phi_{i+1} - i phi_{i-1}, the nuclear derivative).  The
 * products over the three directions give G[s][t][x][y] = sum dd
 * d^2(ab|cd)/dS_x dT_y for S, T in {A, B, C}; the fourth centre follows by
 * translational invariance, d_D = -(d_A + d_B + d_C).
 *
 * Primitive layout and normalisation as libcint's int2e_cart (radial
 * normalisation in env's coefficients, CINTcommon_fac_sp for s and p).
 * Roots from pyscf's libcint (CINTrys_roots, which returns u = t^2/(1 - t^2)).
 *
 * Shell quartets are skipped when q[ij] q[kl] times the density bound is
 * below tol, primitive quartets when the Gaussian prefactor times the largest
 * primitive density weight is below prim_tol (both set by the caller).
 *
 * mode 0: hess (ngroup, natm, natm, 3, 3) += d^2 E_g;
 * mode 1: hess[g] += E_g (the energies, a check of the normalisation).
 *
 * SOCVHFgrad2e_jk_cart: the same driver at derivative order 1 (VRR / HRR
 * raised by one, the four values I, d_A I, d_B I, d_C I per direction, nine
 * terms), grad (ngroup, natm, 3) += d E_g, the 2e part of SCF gradients.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"

#define MAXGROUP 4
#define MAXPRIM 64
#define LMAX 4                          /* up to g functions */
#define NVRR (2 * LMAX + 3)            /* VRR orders 0 .. li + lj + 2 */
#define MAXROOTS 16

void CINTrys_roots(int nroots, double x, double *u, double *w);

static inline double dmax2(double a, double b) { return a > b ? a : b; }

static inline void shell_pair(int ij, int *i, int *j)
{
        int a = (int)((sqrt(1. + 8. * ij) - 1) / 2);
        while (a * (a + 1) / 2 > ij) a--;
        while ((a + 1) * (a + 2) / 2 <= ij) a++;
        *i = a;
        *j = ij - a * (a + 1) / 2;
}

static double common_fac_sp(int l)
{
        if (l == 0) return 0.282094791773878143;
        if (l == 1) return 0.488602511902919921;
        return 1.;
}

/* Cartesian powers of shell l in libcint order */
static int cart_powers(int l, int *px, int *py, int *pz)
{
        int n = 0, lx, ly;
        for (lx = l; lx >= 0; lx--) {
                for (ly = l - lx; ly >= 0; ly--) {
                        px[n] = lx;
                        py[n] = ly;
                        pz[n] = l - lx - ly;
                        n++;
                }
        }
        return n;
}

/* 10 1D derivative values */
enum { V0, VA, VB, VC, VAA, VBB, VCC, VAB, VAC, VBC, NV };

typedef struct {
        int l[4];
        int nf[4];
        int px[4][(LMAX + 1) * (LMAX + 2) / 2];
        int py[4][(LMAX + 1) * (LMAX + 2) / 2];
        int pz[4][(LMAX + 1) * (LMAX + 2) / 2];
} QuartetShape;

/* index of the symmetric second-derivative value for centres s, t in {A,B,C} */
static const int D2[3][3] = {{VAA, VAB, VAC}, {VAB, VBB, VBC}, {VAC, VBC, VCC}};

/* ---- lane-vectorized core: VL (primitive quartet, Rys root) pairs at once ---- */

#define VL 8
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#define MULTIVERSION __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define MULTIVERSION
#endif

typedef struct {
        double c00[3][VL], c00p[3][VL];
        double b00[VL], b10[VL], b01[VL];
        double gz[VL], one[VL];
        double ai2[VL], aj2[VL], ak2[VL];
} Lanes;

/* per-thread workspace of tables_dir (doubles) */
#define WSV_G (NVRR * NVRR * VL)
#define WSV_HB (NVRR * (LMAX + 3) * NVRR * VL)
#define WSV_I4 ((LMAX + 3) * (LMAX + 3) * (LMAX + 3) * (LMAX + 1) * VL)
#define WSV_E (NVRR * (LMAX + 1) * VL)
#define WSV_SIZE (WSV_G + WSV_HB + WSV_I4 + WSV_E)
#define MAXT ((LMAX + 1) * (LMAX + 1) * (LMAX + 1) * (LMAX + 1))
#define TSZV (MAXT * NV * VL)            /* one direction of T */
#define NTERM 46                         /* 18 diagonal + 27 off-diagonal + energy */
#define NCLASS ((LMAX + 1) * (LMAX + 1) * (LMAX + 1) * (LMAX + 1))
/* primitive-pair table entries */
enum { PP_A, PP_IA, PP_E, PP_EC, PP_ECS, PP_P, PP_PA = PP_P + 3, PPN = PP_PA + 3 };

/*
 * One Cartesian direction for VL lanes: VRR, HRR to I[i][j][k][l] and the NV
 * derivative values of every power tuple, T[(t * NV + v) * VL + lane].
 */
MULTIVERSION
static void tables_dir(double *T, double *ws, const int *l, const int order,
                       const double *c00, const double *c00p, const double *b10,
                       const double *b01, const double *b00, const double *ab, const double *cd,
                       const double *g00, const double *ai2, const double *aj2, const double *ak2)
{
        const int li = l[0], lj = l[1], lk = l[2], ll = l[3];
        const int nmax = li + lj + order, mmax = lk + ll + order;
        const int Dm = mmax + 1, Dj = lj + 1 + order, Dk = lk + 1 + order, Dl = ll + 1;
        double *g = ws, *hb = ws + WSV_G, *I4 = hb + WSV_HB, *e = I4 + WSV_I4;
        int n, m, i, j, k, q, v;
#define GV(n, m) (g + ((n) * Dm + (m)) * VL)
        for (v = 0; v < VL; v++) GV(0, 0)[v] = g00[v];
        for (v = 0; v < VL; v++) GV(1, 0)[v] = c00[v] * g00[v];
        for (n = 1; n < nmax; n++) {
                double *g2 = GV(n + 1, 0), *g1 = GV(n, 0), *g0 = GV(n - 1, 0);
#pragma omp simd
                for (v = 0; v < VL; v++) g2[v] = c00[v] * g1[v] + n * b10[v] * g0[v];
        }
        for (n = 0; n <= nmax; n++) {
                double *gn1 = GV(n, 1), *gn0 = GV(n, 0);
                if (n == 0) {
#pragma omp simd
                        for (v = 0; v < VL; v++) gn1[v] = c00p[v] * gn0[v];
                } else {
                        double *gp0 = GV(n - 1, 0);
#pragma omp simd
                        for (v = 0; v < VL; v++) gn1[v] = c00p[v] * gn0[v] + n * b00[v] * gp0[v];
                }
                for (m = 1; m < mmax; m++) {
                        double *g2 = GV(n, m + 1), *g1 = GV(n, m), *g0 = GV(n, m - 1);
                        if (n == 0) {
#pragma omp simd
                                for (v = 0; v < VL; v++) g2[v] = c00p[v] * g1[v] + m * b01[v] * g0[v];
                        } else {
                                double *gp = GV(n - 1, m);
#pragma omp simd
                                for (v = 0; v < VL; v++)
                                        g2[v] = c00p[v] * g1[v] + m * b01[v] * g0[v] + n * b00[v] * gp[v];
                        }
                }
        }
#define HB(i, j) (hb + ((i) * Dj + (j)) * Dm * VL)
        for (i = 0; i <= nmax; i++) memcpy(HB(i, 0), GV(i, 0), sizeof(double) * Dm * VL);
        for (j = 1; j <= lj + order && j <= nmax; j++) {
                for (i = 0; i + j <= nmax; i++) {
                        double *h2 = HB(i, j), *h1 = HB(i + 1, j - 1), *h0 = HB(i, j - 1);
                        for (m = 0; m < Dm; m++) {
#pragma omp simd
                                for (v = 0; v < VL; v++)
                                        h2[m * VL + v] = h1[m * VL + v] + ab[v] * h0[m * VL + v];
                        }
                }
        }
#undef GV
#define IDX(i, j, k, q) (((((i) * Dj + (j)) * Dk + (k)) * Dl + (q)) * VL)
#define EV(k, q) (e + ((k) * Dl + (q)) * VL)
        for (i = 0; i <= li + order; i++) {
                for (j = 0; j <= lj + order; j++) {
                        if (i + j > nmax) continue;
                        double *h = HB(i, j);
                        for (k = 0; k <= mmax; k++) memcpy(EV(k, 0), h + k * VL, sizeof(double) * VL);
                        for (q = 1; q <= ll; q++) {
                                for (k = 0; k + q <= mmax; k++) {
                                        double *e2 = EV(k, q), *e1 = EV(k + 1, q - 1), *e0 = EV(k, q - 1);
#pragma omp simd
                                        for (v = 0; v < VL; v++) e2[v] = e1[v] + cd[v] * e0[v];
                                }
                        }
                        for (k = 0; k <= lk + order; k++)
                                for (q = 0; q <= ll; q++)
                                        if (k + q <= mmax)
                                                memcpy(I4 + IDX(i, j, k, q), EV(k, q), sizeof(double) * VL);
                }
        }
#undef HB
#undef EV
        int t = 0;
        for (i = 0; i <= li; i++) {
        for (j = 0; j <= lj; j++) {
        for (k = 0; k <= lk; k++) {
        for (q = 0; q <= ll; q++, t++) {
                double *val = T + (size_t)t * NV * VL;
#define G(a, b, c) (I4 + IDX(a, b, c, q))
#define VAL(x) (val + (x) * VL)
                const double *g0 = G(i, j, k);
                const double *gi1 = G(i + 1, j, k), *gj1 = G(i, j + 1, k), *gk1 = G(i, j, k + 1);
                if (order == 1) {
#pragma omp simd
                        for (v = 0; v < VL; v++) {
                                VAL(V0)[v] = g0[v];
                                VAL(VA)[v] = ai2[v] * gi1[v];
                                VAL(VB)[v] = aj2[v] * gj1[v];
                                VAL(VC)[v] = ak2[v] * gk1[v];
                        }
                        if (i) {
                                const double *gim = G(i - 1, j, k);
#pragma omp simd
                                for (v = 0; v < VL; v++) VAL(VA)[v] -= i * gim[v];
                        }
                        if (j) {
                                const double *gjm = G(i, j - 1, k);
#pragma omp simd
                                for (v = 0; v < VL; v++) VAL(VB)[v] -= j * gjm[v];
                        }
                        if (k) {
                                const double *gkm = G(i, j, k - 1);
#pragma omp simd
                                for (v = 0; v < VL; v++) VAL(VC)[v] -= k * gkm[v];
                        }
                        continue;
                }
                const double *gi2 = G(i + 2, j, k), *gj2 = G(i, j + 2, k), *gk2 = G(i, j, k + 2);
                const double *gij = G(i + 1, j + 1, k), *gik = G(i + 1, j, k + 1), *gjk = G(i, j + 1, k + 1);
#pragma omp simd
                for (v = 0; v < VL; v++) {
                        VAL(V0)[v] = g0[v];
                        VAL(VA)[v] = ai2[v] * gi1[v];
                        VAL(VB)[v] = aj2[v] * gj1[v];
                        VAL(VC)[v] = ak2[v] * gk1[v];
                        VAL(VAA)[v] = ai2[v] * ai2[v] * gi2[v] - ai2[v] * (2 * i + 1) * g0[v];
                        VAL(VBB)[v] = aj2[v] * aj2[v] * gj2[v] - aj2[v] * (2 * j + 1) * g0[v];
                        VAL(VCC)[v] = ak2[v] * ak2[v] * gk2[v] - ak2[v] * (2 * k + 1) * g0[v];
                        VAL(VAB)[v] = ai2[v] * aj2[v] * gij[v];
                        VAL(VAC)[v] = ai2[v] * ak2[v] * gik[v];
                        VAL(VBC)[v] = aj2[v] * ak2[v] * gjk[v];
                }
                if (i) {
                        const double *gim = G(i - 1, j, k), *gimj = G(i - 1, j + 1, k), *gimk = G(i - 1, j, k + 1);
#pragma omp simd
                        for (v = 0; v < VL; v++) {
                                VAL(VA)[v] -= i * gim[v];
                                VAL(VAB)[v] -= aj2[v] * i * gimj[v];
                                VAL(VAC)[v] -= ak2[v] * i * gimk[v];
                        }
                        if (i > 1) {
                                const double *gi2m = G(i - 2, j, k);
#pragma omp simd
                                for (v = 0; v < VL; v++) VAL(VAA)[v] += i * (i - 1) * gi2m[v];
                        }
                }
                if (j) {
                        const double *gjm = G(i, j - 1, k), *gijm = G(i + 1, j - 1, k), *gjmk = G(i, j - 1, k + 1);
#pragma omp simd
                        for (v = 0; v < VL; v++) {
                                VAL(VB)[v] -= j * gjm[v];
                                VAL(VAB)[v] -= ai2[v] * j * gijm[v];
                                VAL(VBC)[v] -= ak2[v] * j * gjmk[v];
                        }
                        if (j > 1) {
                                const double *gj2m = G(i, j - 2, k);
#pragma omp simd
                                for (v = 0; v < VL; v++) VAL(VBB)[v] += j * (j - 1) * gj2m[v];
                        }
                }
                if (k) {
                        const double *gkm = G(i, j, k - 1), *gikm = G(i + 1, j, k - 1), *gjkm = G(i, j + 1, k - 1);
#pragma omp simd
                        for (v = 0; v < VL; v++) {
                                VAL(VC)[v] -= k * gkm[v];
                                VAL(VAC)[v] -= ai2[v] * k * gikm[v];
                                VAL(VBC)[v] -= aj2[v] * k * gjkm[v];
                        }
                        if (k > 1) {
                                const double *gk2m = G(i, j, k - 2);
#pragma omp simd
                                for (v = 0; v < VL; v++) VAL(VCC)[v] += k * (k - 1) * gk2m[v];
                        }
                }
                if (i && j) {
                        const double *gm = G(i - 1, j - 1, k);
#pragma omp simd
                        for (v = 0; v < VL; v++) VAL(VAB)[v] += i * j * gm[v];
                }
                if (i && k) {
                        const double *gm = G(i - 1, j, k - 1);
#pragma omp simd
                        for (v = 0; v < VL; v++) VAL(VAC)[v] += i * k * gm[v];
                }
                if (j && k) {
                        const double *gm = G(i, j - 1, k - 1);
#pragma omp simd
                        for (v = 0; v < VL; v++) VAL(VBC)[v] += j * k * gm[v];
                }
#undef G
#undef VAL
        } } } }
#undef IDX
}

/* pair index of s <= t in {A, B, C} */
static const int PAIR[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

/*
 * acc[g][term][lane] += w[g][f][lane] (uniform: w[g][f] for all lanes) * products of the three directions for
 * every Cartesian component f: terms 0..17 the diagonal-direction blocks
 * (x * 6 + pair(s, t)), 18..44 the off-diagonal ones (18 + c * 9 + s * 3 + t,
 * c = xy, xz, yz), 45 the energy.
 */
MULTIVERSION
static void assemble(double *acc, const double *T, const QuartetShape *qs, const double *wl,
                     int ngroup, int nf4, int mode, int uniform, int order)
{
        const int l1 = qs->l[1] + 1, l2 = qs->l[2] + 1, l3 = qs->l[3] + 1;
        const int nfi = qs->nf[0], nfj = qs->nf[1], nfk = qs->nf[2], nfl = qs->nf[3];
        int fi, fj, fk, fl, f = 0, g, v, s, t;
        double P[NTERM][VL];
        for (fi = 0; fi < nfi; fi++)
        for (fj = 0; fj < nfj; fj++)
        for (fk = 0; fk < nfk; fk++)
        for (fl = 0; fl < nfl; fl++, f++) {
                const double *X = T + (size_t)(((qs->px[0][fi] * l1 + qs->px[1][fj]) * l2
                                  + qs->px[2][fk]) * l3 + qs->px[3][fl]) * NV * VL;
                const double *Y = T + TSZV + (size_t)(((qs->py[0][fi] * l1 + qs->py[1][fj]) * l2
                                  + qs->py[2][fk]) * l3 + qs->py[3][fl]) * NV * VL;
                const double *Z = T + 2 * TSZV + (size_t)(((qs->pz[0][fi] * l1 + qs->pz[1][fj]) * l2
                                  + qs->pz[2][fk]) * l3 + qs->pz[3][fl]) * NV * VL;
                if (mode == 1) {
#pragma omp simd
                        for (v = 0; v < VL; v++) P[45][v] = X[v] * Y[v] * Z[v];
                        for (g = 0; g < ngroup; g++) {
                                double *a = acc + ((size_t)g * NTERM + 45) * VL;
                                if (uniform) {
                                        const double w1 = wl[(size_t)g * nf4 + f];
#pragma omp simd
                                        for (v = 0; v < VL; v++) a[v] += w1 * P[45][v];
                                } else {
                                        const double *w = wl + ((size_t)g * nf4 + f) * VL;
#pragma omp simd
                                        for (v = 0; v < VL; v++) a[v] += w[v] * P[45][v];
                                }
                        }
                        continue;
                }
                double yz[VL], xz[VL], xy[VL];
#pragma omp simd
                for (v = 0; v < VL; v++) {
                        yz[v] = Y[v] * Z[v];
                        xz[v] = X[v] * Z[v];
                        xy[v] = X[v] * Y[v];
                }
                if (order == 1) {
                        /* term x * 3 + s: d/dS_x, S = A, B, C */
                        for (s = 0; s < 3; s++) {
                                const double *Xs = X + (VA + s) * VL, *Ys = Y + (VA + s) * VL, *Zs = Z + (VA + s) * VL;
#pragma omp simd
                                for (v = 0; v < VL; v++) {
                                        P[s][v] = Xs[v] * yz[v];
                                        P[3 + s][v] = Ys[v] * xz[v];
                                        P[6 + s][v] = Zs[v] * xy[v];
                                }
                        }
                        for (g = 0; g < ngroup; g++) {
                                double *a = acc + (size_t)g * NTERM * VL;
                                const double *w = wl + ((size_t)g * nf4 + f) * VL;
                                int k;
                                for (k = 0; k < 9; k++) {
#pragma omp simd
                                        for (v = 0; v < VL; v++) a[k * VL + v] += w[v] * P[k][v];
                                }
                        }
                        continue;
                }
                for (s = 0; s < 3; s++) {
                        for (t = s; t < 3; t++) {
                                const int k2 = D2[s][t], pr = PAIR[s][t];
                                const double *Xk = X + k2 * VL, *Yk = Y + k2 * VL, *Zk = Z + k2 * VL;
#pragma omp simd
                                for (v = 0; v < VL; v++) {
                                        P[pr][v] = Xk[v] * yz[v];
                                        P[6 + pr][v] = Yk[v] * xz[v];
                                        P[12 + pr][v] = Zk[v] * xy[v];
                                }
                        }
                }
                for (s = 0; s < 3; s++) {
                        const double *Xs = X + (1 + s) * VL, *Ys = Y + (1 + s) * VL;
                        for (t = 0; t < 3; t++) {
                                const double *Yt = Y + (1 + t) * VL, *Zt = Z + (1 + t) * VL;
                                const int c = s * 3 + t;
#pragma omp simd
                                for (v = 0; v < VL; v++) {
                                        P[18 + c][v] = Xs[v] * Yt[v] * Z[v];
                                        P[27 + c][v] = Xs[v] * Y[v] * Zt[v];
                                        P[36 + c][v] = X[v] * Ys[v] * Zt[v];
                                }
                        }
                }
                for (g = 0; g < ngroup; g++) {
                        double *a = acc + (size_t)g * NTERM * VL;
                        int k;
                        if (uniform) {
                                const double w1 = wl[(size_t)g * nf4 + f];
                                for (k = 0; k < 45; k++) {
#pragma omp simd
                                        for (v = 0; v < VL; v++) a[k * VL + v] += w1 * P[k][v];
                                }
                        } else {
                                const double *w = wl + ((size_t)g * nf4 + f) * VL;
                                for (k = 0; k < 45; k++) {
#pragma omp simd
                                        for (v = 0; v < VL; v++) a[k * VL + v] += w[v] * P[k][v];
                                }
                        }
                }
        }
}


/* lanes of one angular class waiting for a flush, and the (at most VL) shell
 * quartets they belong to */
typedef struct {
        QuartetShape qs;
        int nf4;
        int nlane;
        Lanes lanes;
        double ab[3][VL], cd[3][VL];
        int slot[VL];
        double *wl;                             /* (ngroup, nf4, VL) */
        int busy[VL], open[VL], nleft[VL], used[VL];
        int at[VL][4];
        double sum[VL][MAXGROUP * NTERM];
} ClassBuf;

static ClassBuf *class_new(const int *l, int ngroup)
{
        ClassBuf *cb = calloc(1, sizeof(ClassBuf));
        int s, v;
        cb->nf4 = 1;
        for (s = 0; s < 4; s++) {
                cb->qs.l[s] = l[s];
                cb->qs.nf[s] = cart_powers(l[s], cb->qs.px[s], cb->qs.py[s], cb->qs.pz[s]);
                cb->nf4 *= cb->qs.nf[s];
        }
        for (v = 0; v < VL; v++) cb->lanes.one[v] = 1.;
        cb->wl = malloc(sizeof(double) * cb->nf4 * ngroup * VL);
        return cb;
}

/* G[g][s][t][x][y] from the term sums of a shell quartet, the fourth centre
 * by translational invariance, added to the atom blocks (mode 1: energies) */
static void finalize(const double *sum, const int *at, int ngroup, int mode, int natm, size_t nh1,
                     double *hloc, int order)
{
        int g, s1, t1, x, y;
        if (mode == 1) {
                for (g = 0; g < ngroup; g++) hloc[g] += sum[g * NTERM + 45];
                return;
        }
        if (order == 1) {
                /* d/dS_x for S = A, B, C; the fourth centre d_D = -(d_A + d_B + d_C) */
                for (g = 0; g < ngroup; g++) {
                        const double *sm = sum + g * NTERM;
                        double *hg = hloc + g * nh1;
                        for (x = 0; x < 3; x++) {
                                double dsum = 0;
                                for (s1 = 0; s1 < 3; s1++) {
                                        hg[at[s1] * 3 + x] += sm[x * 3 + s1];
                                        dsum += sm[x * 3 + s1];
                                }
                                hg[at[3] * 3 + x] -= dsum;
                        }
                }
                return;
        }
        for (g = 0; g < ngroup; g++) {
                const double *sm = sum + g * NTERM;
                double Gf[4][4][3][3];
                for (s1 = 0; s1 < 3; s1++)
                for (t1 = 0; t1 < 3; t1++)
                for (x = 0; x < 3; x++)
                for (y = 0; y < 3; y++) {
                        double v;
                        if (x == y) {
                                v = sm[x * 6 + (s1 <= t1 ? PAIR[s1][t1] : PAIR[t1][s1])];
                        } else if (x < y) {
                                v = sm[18 + (x + y - 1) * 9 + s1 * 3 + t1];
                        } else {
                                v = sm[18 + (x + y - 1) * 9 + t1 * 3 + s1];
                        }
                        Gf[s1][t1][x][y] = v;
                }
                /* fourth centre: d_D = -(d_A + d_B + d_C) */
                for (x = 0; x < 3; x++)
                for (y = 0; y < 3; y++) {
                        double dd3 = 0;
                        for (s1 = 0; s1 < 3; s1++) {
                                double r3 = 0, c3 = 0;
                                for (t1 = 0; t1 < 3; t1++) {
                                        r3 += Gf[s1][t1][x][y];
                                        c3 += Gf[t1][s1][x][y];
                                }
                                Gf[s1][3][x][y] = -r3;
                                Gf[3][s1][x][y] = -c3;
                                dd3 += r3;
                        }
                        Gf[3][3][x][y] = dd3;
                }
                double *hg = hloc + g * nh1;
                for (s1 = 0; s1 < 4; s1++)
                for (t1 = 0; t1 < 4; t1++) {
                        double *hp = hg + ((size_t)at[s1] * natm + at[t1]) * 9;
                        for (x = 0; x < 3; x++)
                                for (y = 0; y < 3; y++) hp[x * 3 + y] += Gf[s1][t1][x][y];
                }
        }
}

/* Rys core for the buffered lanes of a class (unused lanes padded with zero
 * weight), lane sums to their shell quartets, finished quartets finalized */
static void flush_class(ClassBuf *cb, double *T, double *ws, double *acc, int ngroup, int mode,
                        int natm, size_t nh1, double *hloc, int order)
{
        const int nl = cb->nlane;
        int s, v, g, k, a;
        if (nl == 0) return;
        for (v = nl; v < VL; v++) {
                for (s = 0; s < 3; s++) {
                        cb->lanes.c00[s][v] = cb->lanes.c00p[s][v] = 0;
                        cb->ab[s][v] = cb->cd[s][v] = 0;
                }
                cb->lanes.b00[v] = cb->lanes.b10[v] = cb->lanes.b01[v] = cb->lanes.gz[v] = 0;
                cb->lanes.ai2[v] = cb->lanes.aj2[v] = cb->lanes.ak2[v] = 0;
                for (a = 0; a < ngroup * cb->nf4; a++) cb->wl[(size_t)a * VL + v] = 0;
        }
        for (s = 0; s < 3; s++)
                tables_dir(T + (size_t)s * TSZV, ws, cb->qs.l, order, cb->lanes.c00[s], cb->lanes.c00p[s],
                           cb->lanes.b10, cb->lanes.b01, cb->lanes.b00, cb->ab[s], cb->cd[s],
                           s == 2 ? cb->lanes.gz : cb->lanes.one, cb->lanes.ai2, cb->lanes.aj2, cb->lanes.ak2);
        memset(acc, 0, sizeof(double) * ngroup * NTERM * VL);
        assemble(acc, T, &cb->qs, cb->wl, ngroup, cb->nf4, mode, 0, order);
        for (v = 0; v < nl; v++) {
                const int q = cb->slot[v];
                double *sm = cb->sum[q];
                for (g = 0; g < ngroup; g++)
                        for (k = 0; k < NTERM; k++) sm[g * NTERM + k] += acc[((size_t)g * NTERM + k) * VL + v];
                if (--cb->nleft[q] == 0 && !cb->open[q]) {
                        finalize(sm, cb->at[q], ngroup, mode, natm, nh1, hloc, order);
                        cb->busy[q] = 0;
                }
        }
        cb->nlane = 0;
}

static void deriv2e_drv(int order, double *hess, int mode, int ngroup, double *jfac, double *kfac,
                        double *dm, double *qcond, double *dcond, double tol, double prim_tol,
                        int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        const size_t nh1 = mode == 1 ? 1 : (order == 2 ? (size_t)natm * natm * 9 : (size_t)natm * 3);
        const size_t nh = nh1 * ngroup;
        const int npair = nbas * (nbas + 1) / 2;
        int maxf = 0, ib;
        for (ib = 0; ib < nbas; ib++) {
                int nf = ao_loc[ib + 1] - ao_loc[ib];
                maxf = nf > maxf ? nf : maxf;
        }
        const size_t maxf4 = (size_t)maxf * maxf * maxf * maxf;
        /* |dd| <= jw |D_ab D_cd| + kw max(|D_ac D_bd|, |D_ad D_bc|) */
        double jw = 0, kw = 0;
        int g;
        for (g = 0; g < ngroup; g++) {
                jw = dmax2(jw, fabs(jfac[g]) * 4);
                kw = dmax2(kw, fabs(kfac[g]) * 2);
        }
        /* max_c |c| of every primitive of every shell */
        double *cmax_all = malloc(sizeof(double) * nbas * MAXPRIM);
        for (ib = 0; ib < nbas; ib++) {
                const int np = bas[ib * BAS_SLOTS + NPRIM_OF], nc = bas[ib * BAS_SLOTS + NCTR_OF];
                const double *c = env + bas[ib * BAS_SLOTS + PTR_COEFF];
                int ip, ic;
                for (ip = 0; ip < np; ip++) {
                        double m = 0;
                        for (ic = 0; ic < nc; ic++) m = dmax2(m, fabs(c[ic * np + ip]));
                        cmax_all[ib * MAXPRIM + ip] = m;
                }
        }
        /* primitive pairs (p_i, p_j) of every shell pair i >= j, index p_j np_i + p_i:
         * a = a_i + a_j, 1 / a, e = exp(-a_i a_j / a |AB|^2), e max|c_i| max|c_j|,
         * that / (a sqrt(a)) for the ket-level bound, P, P - A; and per pair the
         * largest e max|c_i| max|c_j| / a for the bra side of that bound */
        size_t *ppoff = malloc(sizeof(size_t) * (npair + 1));
        double *brmax = malloc(sizeof(double) * npair);
        {
                size_t tot = 0;
                int pq_, i_, j_;
                for (pq_ = 0; pq_ < npair; pq_++) {
                        shell_pair(pq_, &i_, &j_);
                        ppoff[pq_] = tot;
                        tot += (size_t)bas[i_ * BAS_SLOTS + NPRIM_OF] * bas[j_ * BAS_SLOTS + NPRIM_OF];
                }
                ppoff[npair] = tot;
        }
        double *ppd = malloc(sizeof(double) * ppoff[npair] * PPN);
        {
                int pq_;
#pragma omp parallel for schedule(dynamic, 16)
                for (pq_ = 0; pq_ < npair; pq_++) {
                        int i_, j_, ip, jp, x;
                        shell_pair(pq_, &i_, &j_);
                        const int npi = bas[i_ * BAS_SLOTS + NPRIM_OF], npj = bas[j_ * BAS_SLOTS + NPRIM_OF];
                        const double *exi = env + bas[i_ * BAS_SLOTS + PTR_EXP], *exj = env + bas[j_ * BAS_SLOTS + PTR_EXP];
                        const double *Ri = env + atm[bas[i_ * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
                        const double *Rj = env + atm[bas[j_ * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
                        const double r2 = (Ri[0] - Rj[0]) * (Ri[0] - Rj[0]) + (Ri[1] - Rj[1]) * (Ri[1] - Rj[1])
                                        + (Ri[2] - Rj[2]) * (Ri[2] - Rj[2]);
                        double bm = 0;
                        for (jp = 0; jp < npj; jp++) {
                                for (ip = 0; ip < npi; ip++) {
                                        double *d = ppd + (ppoff[pq_] + (size_t)jp * npi + ip) * PPN;
                                        const double ai = exi[ip], aj = exj[jp], a = ai + aj, ia = 1. / a;
                                        const double e = exp(-ai * aj * ia * r2);
                                        const double ec = e * cmax_all[i_ * MAXPRIM + ip] * cmax_all[j_ * MAXPRIM + jp];
                                        d[PP_A] = a;
                                        d[PP_IA] = ia;
                                        d[PP_E] = e;
                                        d[PP_EC] = ec;
                                        d[PP_ECS] = ec * ia / sqrt(a);
                                        for (x = 0; x < 3; x++) {
                                                d[PP_P + x] = (ai * Ri[x] + aj * Rj[x]) * ia;
                                                d[PP_PA + x] = d[PP_P + x] - Ri[x];
                                        }
                                        bm = dmax2(bm, ec * ia);
                                }
                        }
                        brmax[pq_] = bm;
                }
        }
#pragma omp parallel
{
        double *hloc = calloc(nh, sizeof(double));
        double *dd = malloc(sizeof(double) * maxf4 * ngroup);     /* contracted weights */
        double *x1 = malloc(sizeof(double) * maxf4 * ngroup);     /* kl-primitive weights */
        double *w = malloc(sizeof(double) * maxf4 * ngroup);      /* primitive weights */
        double *T = malloc(sizeof(double) * 3 * TSZV);
        double *ws = malloc(sizeof(double) * WSV_SIZE);
        double *acc = malloc(sizeof(double) * MAXGROUP * NTERM * VL);
        int *plist = malloc(sizeof(int) * MAXPRIM * MAXPRIM);
        double *prefl = malloc(sizeof(double) * MAXPRIM * MAXPRIM);
        double *iasl = malloc(sizeof(double) * MAXPRIM * MAXPRIM);
        ClassBuf **cbs = calloc(NCLASS, sizeof(ClassBuf *));
        int ij, kl, ish, jsh, ksh, lsh, v;
#pragma omp for schedule(dynamic, 2)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                const double bra_max = brmax[ij];
                const double *ppij = ppd + ppoff[ij] * PPN;
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        /* J and K density bounds apart: a K-only call skips the
                         * quartets with only a large Coulomb weight */
                        double dbound = jw * dcond[ish * nbas + jsh] * dcond[ksh * nbas + lsh]
                                      + kw * dmax2(dcond[ish * nbas + ksh] * dcond[jsh * nbas + lsh],
                                                   dcond[ish * nbas + lsh] * dcond[jsh * nbas + ksh]);
                        if (q4 * dbound < tol) {
                                continue;
                        }
                        int sh[4] = {ish, jsh, ksh, lsh};
                        int l[4], nf[4], nprim[4], nctr[4], at[4], s;
                        double *ex[4], *cf[4], *R[4];
                        double fac = 1;
                        for (s = 0; s < 4; s++) {
                                int *b = bas + sh[s] * BAS_SLOTS;
                                l[s] = b[ANG_OF];
                                nf[s] = (l[s] + 1) * (l[s] + 2) / 2;
                                nprim[s] = b[NPRIM_OF];
                                nctr[s] = b[NCTR_OF];
                                at[s] = b[ATOM_OF];
                                ex[s] = env + b[PTR_EXP];
                                cf[s] = env + b[PTR_COEFF];
                                R[s] = env + atm[at[s] * ATM_SLOTS + PTR_COORD];
                                fac *= common_fac_sp(l[s]);
                        }
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        const int ckey = ((l[0] * (LMAX + 1) + l[1]) * (LMAX + 1) + l[2]) * (LMAX + 1) + l[3];
                        ClassBuf *cb = cbs[ckey];
                        if (cb == NULL) cb = cbs[ckey] = class_new(l, ngroup);
                        const int nfi = nf[0], nfj = nf[1], nfk = nf[2], nfl = nf[3];
                        const int nci = nctr[0], ncj = nctr[1], nck = nctr[2], ncl = nctr[3];
                        /* contracted functions: index (c, f) -> c * nf + f; dd[g][I][J][K][L] */
                        const int NI = nci * nfi, NJ = ncj * nfj, NK = nck * nfk, NL = ncl * nfl;
                        const size_t nd = (size_t)NI * NJ * NK * NL;
                        size_t a0 = ao_loc[ish], b0 = ao_loc[jsh], c0 = ao_loc[ksh], e0 = ao_loc[lsh];
                        int a, b, c, d;
                        for (g = 0; g < ngroup; g++) {
                                double *ddg = dd + g * nd;
                                double jf = 4 * jfac[g] * fac, kf = kfac[g] * fac;
                                for (a = 0; a < NI; a++)
                                for (b = 0; b < NJ; b++) {
                                        const double dab = jf * dm[(a0 + a) * nao + b0 + b];
                                        const double *dA = dm + (a0 + a) * nao, *dB = dm + (b0 + b) * nao;
                                        for (c = 0; c < NK; c++) {
                                                const double dbc = dB[c0 + c], dac = dA[c0 + c];
                                                const double *dC = dm + (c0 + c) * nao;
                                                double *o = ddg + ((a * NJ + b) * NK + c) * NL;
                                                for (d = 0; d < NL; d++)
                                                        o[d] = dab * dC[e0 + d] + kf * (dac * dB[e0 + d] + dA[e0 + d] * dbc);
                                        }
                                }
                        }
                        /* a free slot of the class buffer: every pending quartet
                         * has a lane in the buffer, which holds fewer than VL */
                        int qslot = 0;
                        while (cb->busy[qslot]) qslot++;
                        cb->busy[qslot] = cb->open[qslot] = 1;
                        cb->nleft[qslot] = cb->used[qslot] = 0;
                        for (s = 0; s < 4; s++) cb->at[qslot][s] = at[s];
                        memset(cb->sum[qslot], 0, sizeof(double) * ngroup * NTERM);
                        double ab[3], cdv[3];
                        for (s = 0; s < 3; s++) {
                                ab[s] = R[0][s] - R[1][s];
                                cdv[s] = R[2][s] - R[3][s];
                        }
                        const int nroots = (l[0] + l[1] + l[2] + l[3] + order) / 2 + 1;
                        const int nf4 = nfi * nfj * nfk * nfl;
                        int pi, pj, pk, pl;
                        /* bounds for the primitive screening (no thresholds of their own:
                         * |w| <= ddmax prod_s nctr_s max_c |c_s|) */
                        double ddmax = 0, *cmax[4];
                        for (a = 0; a < (int)(nd * ngroup); a++) ddmax = dmax2(ddmax, fabs(dd[a]));
                        ddmax *= nci * ncj * nck * ncl;
                        for (s = 0; s < 4; s++) cmax[s] = cmax_all + sh[s] * MAXPRIM;
                        /* all four shells singly contracted: the primitive weights are
                         * c_i c_j c_k c_l dd, the coefficients go into the z-direction
                         * prefactor and dd is the lane weight */
                        const int single = nci == 1 && ncj == 1 && nck == 1 && ncl == 1;
                        const double *ppkl = ppd + ppoff[kl] * PPN;
                        const int npij = nprim[0] * nprim[1];
                        for (pl = 0; pl < nprim[3]; pl++) {
                        for (pk = 0; pk < nprim[2]; pk++) {
                                const double *dk = ppkl + ((size_t)pl * nprim[2] + pk) * PPN;
                                const double akl = dk[PP_A], iakl = dk[PP_IA];
                                /* 1 / (aij akl sqrt(aij + akl)) <= 1 / (aij akl sqrt(akl)) */
                                if (34.986836655249725 * bra_max * dk[PP_ECS] * ddmax < prim_tol) continue;
                                /* the bra primitive pairs passing the coefficient bound */
                                const double eckl = dk[PP_EC] * ddmax, ekl = dk[PP_E];
                                int nsurv = 0, pij, isv;
                                for (pij = 0; pij < npij; pij++) {
                                        const double *di = ppij + (size_t)pij * PPN;
                                        const double ias = 1. / (di[PP_A] + akl);
                                        /* 2 pi^(5/2) / (aij akl sqrt(aij + akl)) */
                                        const double pr = 34.986836655249725 * di[PP_IA] * iakl * sqrt(ias);
                                        if (pr * di[PP_EC] * eckl < prim_tol) continue;
                                        plist[nsurv] = pij;
                                        prefl[nsurv] = pr * di[PP_E] * ekl;
                                        iasl[nsurv] = ias;
                                        nsurv++;
                                }
                                if (nsurv == 0) continue;
                                /* x1[g][a][b][fk][fl] = sum_{ck,cl} c_k c_l dd[g][a][b][ck fk][cl fl] */
                                size_t nx1 = (size_t)NI * NJ * nfk * nfl;
                                int ck, cl, fk, fl, ci, cj, fi, fj;
                                for (g = 0; g < (single ? 0 : ngroup); g++) {
                                        double *xg = x1 + g * nx1, *ddg = dd + g * nd;
                                        for (a = 0; a < NI * NJ; a++) {
                                                for (fk = 0; fk < nfk; fk++)
                                                for (fl = 0; fl < nfl; fl++) {
                                                        double v1 = 0;
                                                        for (ck = 0; ck < nck; ck++)
                                                        for (cl = 0; cl < ncl; cl++)
                                                                v1 += cf[2][ck * nprim[2] + pk] * cf[3][cl * nprim[3] + pl]
                                                                    * ddg[(size_t)a * NK * NL + (ck * nfk + fk) * NL
                                                                          + cl * nfl + fl];
                                                        xg[((size_t)a * nfk + fk) * nfl + fl] = v1;
                                                }
                                        }
                                }
                                const double *Q = dk + PP_P, *QC = dk + PP_PA;
                                const double ak2 = 2 * ex[2][pk];
                                for (isv = 0; isv < nsurv; isv++) {
                                        pij = plist[isv];
                                        pi = pij % nprim[0];
                                        pj = pij / nprim[0];
                                        const double *di = ppij + (size_t)pij * PPN;
                                        const double aij = di[PP_A], iaij = di[PP_IA];
                                        const double ias = iasl[isv], pref = prefl[isv];
                                        /* w[g][fi][fj][fk][fl] = sum_{ci,cj} c_i c_j x1[g][ci fi][cj fj][fk][fl] */
                                        double wmax = 0, cc4 = 1;
                                        if (single) {
                                                cc4 = cf[0][pi] * cf[1][pj] * cf[2][pk] * cf[3][pl];
                                                wmax = fabs(cc4) * ddmax;
                                        }
                                        for (g = 0; g < (single ? 0 : ngroup); g++) {
                                                double *xg = x1 + g * nx1, *wg = w + g * nf4;
                                                for (fi = 0; fi < nfi; fi++)
                                                for (fj = 0; fj < nfj; fj++) {
                                                        double *wo = wg + (size_t)(fi * nfj + fj) * nfk * nfl;
                                                        for (a = 0; a < nfk * nfl; a++) wo[a] = 0;
                                                        for (ci = 0; ci < nci; ci++)
                                                        for (cj = 0; cj < ncj; cj++) {
                                                                double cc = cf[0][ci * nprim[0] + pi]
                                                                          * cf[1][cj * nprim[1] + pj];
                                                                double *xi = xg + ((size_t)(ci * nfi + fi) * NJ
                                                                                   + cj * nfj + fj) * nfk * nfl;
                                                                for (a = 0; a < nfk * nfl; a++) wo[a] += cc * xi[a];
                                                        }
                                                        for (a = 0; a < nfk * nfl; a++) wmax = dmax2(wmax, fabs(wo[a]));
                                                }
                                        }
                                        if (pref * wmax < prim_tol) continue;
                                        double PQ[3];
                                        for (s = 0; s < 3; s++) PQ[s] = di[PP_P + s] - Q[s];
                                        const double x = aij * akl * ias * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);
                                        double u[MAXROOTS], rw[MAXROOTS];
                                        CINTrys_roots(nroots, x, u, rw);
                                        const double *wsrc = single ? dd : w;
                                        const double ai2 = 2 * ex[0][pi], aj2 = 2 * ex[1][pj], pc = pref * cc4;
                                        int r;
                                        for (r = 0; r < nroots; r++) {
                                                const int n = cb->nlane;
                                                /* t^2 / (aij + akl) */
                                                const double ta = u[r] / (1 + u[r]) * ias;
                                                cb->lanes.b00[n] = .5 * ta;
                                                cb->lanes.b10[n] = .5 * iaij * (1 - akl * ta);
                                                cb->lanes.b01[n] = .5 * iakl * (1 - aij * ta);
                                                for (s = 0; s < 3; s++) {
                                                        cb->lanes.c00[s][n] = di[PP_PA + s] - akl * ta * PQ[s];
                                                        cb->lanes.c00p[s][n] = QC[s] + aij * ta * PQ[s];
                                                        cb->ab[s][n] = ab[s];
                                                        cb->cd[s][n] = cdv[s];
                                                }
                                                cb->lanes.gz[n] = pc * rw[r];
                                                cb->lanes.ai2[n] = ai2;
                                                cb->lanes.aj2[n] = aj2;
                                                cb->lanes.ak2[n] = ak2;
                                                for (a = 0; a < ngroup * nf4; a++) cb->wl[(size_t)a * VL + n] = wsrc[a];
                                                cb->slot[n] = qslot;
                                                cb->nleft[qslot]++;
                                                cb->used[qslot] = 1;
                                                if (++cb->nlane == VL)
                                                        flush_class(cb, T, ws, acc, ngroup, mode, natm, nh1, hloc, order);
                                        }
                                }
                        } }
                        /* the quartet is complete once its buffered lanes are flushed */
                        cb->open[qslot] = 0;
                        if (cb->nleft[qslot] == 0) {
                                if (cb->used[qslot])
                                        finalize(cb->sum[qslot], cb->at[qslot], ngroup, mode, natm, nh1, hloc, order);
                                cb->busy[qslot] = 0;
                        }
                }
        }
        /* the partly filled buffers of this thread */
        for (v = 0; v < NCLASS; v++) {
                if (cbs[v] != NULL) {
                        flush_class(cbs[v], T, ws, acc, ngroup, mode, natm, nh1, hloc, order);
                        free(cbs[v]->wl);
                        free(cbs[v]);
                }
        }
        free(cbs);
#pragma omp critical
        {
                size_t n;
                for (n = 0; n < nh; n++) hess[n] += hloc[n];
        }
        free(hloc);
        free(dd);
        free(x1);
        free(w);
        free(T);
        free(ws);
        free(acc);
        free(plist);
        free(prefl);
        free(iasl);
}
        free(cmax_all);
        free(ppoff);
        free(brmax);
        free(ppd);
}

void SOCVHFhess2e_jk_cart(double *hess, int mode, int ngroup, double *jfac, double *kfac,
                          double *dm, double *qcond, double *dcond, double tol, double prim_tol,
                          int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        deriv2e_drv(2, hess, mode, ngroup, jfac, kfac, dm, qcond, dcond, tol, prim_tol,
                    ao_loc, atm, natm, bas, nbas, env);
}

void SOCVHFgrad2e_jk_cart(double *grad, int mode, int ngroup, double *jfac, double *kfac,
                          double *dm, double *qcond, double *dcond, double tol, double prim_tol,
                          int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        deriv2e_drv(1, grad, mode, ngroup, jfac, kfac, dm, qcond, dcond, tol, prim_tol,
                    ao_loc, atm, natm, bas, nbas, env);
}
