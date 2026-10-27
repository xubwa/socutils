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
 * runtime-dispatched to AVX-512 / AVX2 / baseline code; the lanes are summed
 * once per shell quartet)
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

/*
 * One Cartesian direction for VL lanes: VRR, HRR to I[i][j][k][l] and the NV
 * derivative values of every power tuple, T[(t * NV + v) * VL + lane].
 */
MULTIVERSION
static void tables_dir(double *T, double *ws, const int *l,
                       const double *c00, const double *c00p, const double *b10,
                       const double *b01, const double *b00, const double ab, const double cd,
                       const double *g00, const double *ai2, const double *aj2, const double *ak2)
{
        const int li = l[0], lj = l[1], lk = l[2], ll = l[3];
        const int nmax = li + lj + 2, mmax = lk + ll + 2;
        const int Dm = mmax + 1, Dj = lj + 3, Dk = lk + 3, Dl = ll + 1;
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
        for (j = 1; j <= lj + 2 && j <= nmax; j++) {
                for (i = 0; i + j <= nmax; i++) {
                        double *h2 = HB(i, j), *h1 = HB(i + 1, j - 1), *h0 = HB(i, j - 1);
#pragma omp simd
                        for (v = 0; v < Dm * VL; v++) h2[v] = h1[v] + ab * h0[v];
                }
        }
#undef GV
#define IDX(i, j, k, q) (((((i) * Dj + (j)) * Dk + (k)) * Dl + (q)) * VL)
#define EV(k, q) (e + ((k) * Dl + (q)) * VL)
        for (i = 0; i <= li + 2; i++) {
                for (j = 0; j <= lj + 2; j++) {
                        if (i + j > nmax) continue;
                        double *h = HB(i, j);
                        for (k = 0; k <= mmax; k++) memcpy(EV(k, 0), h + k * VL, sizeof(double) * VL);
                        for (q = 1; q <= ll; q++) {
                                for (k = 0; k + q <= mmax; k++) {
                                        double *e2 = EV(k, q), *e1 = EV(k + 1, q - 1), *e0 = EV(k, q - 1);
#pragma omp simd
                                        for (v = 0; v < VL; v++) e2[v] = e1[v] + cd * e0[v];
                                }
                        }
                        for (k = 0; k <= lk + 2; k++)
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
 * acc[g][term][lane] += w[g][f][lane] * products of the three directions for
 * every Cartesian component f: terms 0..17 the diagonal-direction blocks
 * (x * 6 + pair(s, t)), 18..44 the off-diagonal ones (18 + c * 9 + s * 3 + t,
 * c = xy, xz, yz), 45 the energy.
 */
MULTIVERSION
static void assemble(double *acc, const double *T, const QuartetShape *qs, const double *wl,
                     int ngroup, int nf4, int mode)
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
                                const double *w = wl + ((size_t)g * nf4 + f) * VL;
                                double *a = acc + ((size_t)g * NTERM + 45) * VL;
#pragma omp simd
                                for (v = 0; v < VL; v++) a[v] += w[v] * P[45][v];
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
                        const double *w = wl + ((size_t)g * nf4 + f) * VL;
                        double *a = acc + (size_t)g * NTERM * VL;
                        int k;
                        for (k = 0; k < 45; k++) {
#pragma omp simd
                                for (v = 0; v < VL; v++) a[k * VL + v] += w[v] * P[k][v];
                        }
                }
        }
}


void SOCVHFhess2e_jk_cart(double *hess, int mode, int ngroup, double *jfac, double *kfac,
                          double *dm, double *qcond, double *dcond, double tol, double prim_tol,
                          int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        const size_t nh1 = mode == 0 ? (size_t)natm * natm * 9 : 1;
        const size_t nh = nh1 * ngroup;
        const int npair = nbas * (nbas + 1) / 2;
        int maxf = 0, ib;
        for (ib = 0; ib < nbas; ib++) {
                int nf = ao_loc[ib + 1] - ao_loc[ib];
                maxf = nf > maxf ? nf : maxf;
        }
        const size_t maxf4 = (size_t)maxf * maxf * maxf * maxf;
        double ksum = 0;
        int g;
        for (g = 0; g < ngroup; g++) ksum = dmax2(ksum, fabs(jfac[g]) * 4 + fabs(kfac[g]) * 2);
#pragma omp parallel
{
        double *hloc = calloc(nh, sizeof(double));
        double *dd = malloc(sizeof(double) * maxf4 * ngroup);     /* contracted weights */
        double *x1 = malloc(sizeof(double) * maxf4 * ngroup);     /* kl-primitive weights */
        double *w = malloc(sizeof(double) * maxf4 * ngroup);      /* primitive weights */
        double *T = malloc(sizeof(double) * 3 * TSZV);
        double *ws = malloc(sizeof(double) * WSV_SIZE);
        double *wl = malloc(sizeof(double) * maxf4 * ngroup * VL);     /* lane weights */
        double *acc = malloc(sizeof(double) * MAXGROUP * NTERM * VL);
        Lanes lanes;
        int nlane = 0, v;
        for (v = 0; v < VL; v++) lanes.one[v] = 1.;
        double *cm = malloc(sizeof(double) * 4 * MAXPRIM);
        double *eijs = malloc(sizeof(double) * MAXPRIM * MAXPRIM);
        QuartetShape qs;
        int ij, kl, ish, jsh, ksh, lsh;
#pragma omp for schedule(dynamic, 2)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        double dbound = ksum * dmax2(dmax2(dcond[ish * nbas + jsh] * dcond[ksh * nbas + lsh],
                                                           dcond[ish * nbas + ksh] * dcond[jsh * nbas + lsh]),
                                                     dcond[ish * nbas + lsh] * dcond[jsh * nbas + ksh]);
                        if (q4 * dbound < tol) {
                                continue;
                        }
                        int sh[4] = {ish, jsh, ksh, lsh};
                        int nprim[4], nctr[4], at[4], s;
                        double *ex[4], *cf[4], *R[4];
                        double fac = 1;
                        for (s = 0; s < 4; s++) {
                                int *b = bas + sh[s] * BAS_SLOTS;
                                qs.l[s] = b[ANG_OF];
                                nprim[s] = b[NPRIM_OF];
                                nctr[s] = b[NCTR_OF];
                                at[s] = b[ATOM_OF];
                                ex[s] = env + b[PTR_EXP];
                                cf[s] = env + b[PTR_COEFF];
                                R[s] = env + atm[at[s] * ATM_SLOTS + PTR_COORD];
                                qs.nf[s] = cart_powers(qs.l[s], qs.px[s], qs.py[s], qs.pz[s]);
                                fac *= common_fac_sp(qs.l[s]);
                        }
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        const int nfi = qs.nf[0], nfj = qs.nf[1], nfk = qs.nf[2], nfl = qs.nf[3];
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
                                for (b = 0; b < NJ; b++)
                                for (c = 0; c < NK; c++)
                                for (d = 0; d < NL; d++) {
                                        size_t A = a0 + a, B = b0 + b, C = c0 + c, D = e0 + d;
                                        ddg[((a * NJ + b) * NK + c) * NL + d] =
                                                jf * dm[A * nao + B] * dm[C * nao + D]
                                              + kf * (dm[A * nao + C] * dm[B * nao + D]
                                                      + dm[A * nao + D] * dm[B * nao + C]);
                                }
                        }
                        double ab[3], cdv[3];
                        for (s = 0; s < 3; s++) {
                                ab[s] = R[0][s] - R[1][s];
                                cdv[s] = R[2][s] - R[3][s];
                        }
                        double rab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
                        double rcd2 = cdv[0] * cdv[0] + cdv[1] * cdv[1] + cdv[2] * cdv[2];
                        const int nroots = (qs.l[0] + qs.l[1] + qs.l[2] + qs.l[3] + 2) / 2 + 1;
                        const int nf4 = nfi * nfj * nfk * nfl;
                        /* G[g][s][t][x][y] */
                        double G[MAXGROUP][3][3][3][3];
                        memset(G, 0, sizeof(G));
                        double E[MAXGROUP] = {0};
                        int pi, pj, pk, pl;
                        memset(acc, 0, sizeof(double) * ngroup * NTERM * VL);
                        nlane = 0;
#define FLUSH_LANES() do { \
                                for (s = 0; s < 3; s++) \
                                        tables_dir(T + (size_t)s * TSZV, ws, qs.l, lanes.c00[s], lanes.c00p[s], \
                                                   lanes.b10, lanes.b01, lanes.b00, ab[s], cdv[s], \
                                                   s == 2 ? lanes.gz : lanes.one, lanes.ai2, lanes.aj2, lanes.ak2); \
                                assemble(acc, T, &qs, wl, ngroup, nf4, mode); \
                                nlane = 0; } while (0)
                        /* bounds for the primitive screening (no thresholds of their own:
                         * |w| <= ddmax prod_s nctr_s max_c |c_s|) */
                        double ddmax = 0, *cmax[4];
                        for (a = 0; a < (int)(nd * ngroup); a++) ddmax = dmax2(ddmax, fabs(dd[a]));
                        ddmax *= nci * ncj * nck * ncl;
                        for (s = 0; s < 4; s++) {
                                cmax[s] = cm + s * MAXPRIM;
                                for (a = 0; a < nprim[s]; a++) {
                                        double m = 0;
                                        for (b = 0; b < nctr[s]; b++) m = dmax2(m, fabs(cf[s][b * nprim[s] + a]));
                                        cmax[s][a] = m;
                                }
                        }
                        /* bra primitive pairs: exp(-ai aj / aij |AB|^2) and the largest
                         * eij |ci cj| / aij for the ket-level bound */
                        double bra_max = 0;
                        for (pj = 0; pj < nprim[1]; pj++) {
                                for (pi = 0; pi < nprim[0]; pi++) {
                                        double ai = ex[0][pi], aj = ex[1][pj];
                                        double e = exp(-ai * aj / (ai + aj) * rab2);
                                        eijs[pj * nprim[0] + pi] = e;
                                        bra_max = dmax2(bra_max, e * cmax[0][pi] * cmax[1][pj] / (ai + aj));
                                }
                        }
                        for (pl = 0; pl < nprim[3]; pl++) {
                        for (pk = 0; pk < nprim[2]; pk++) {
                                double ak = ex[2][pk], al = ex[3][pl];
                                double akl = ak + al;
                                double ekl = exp(-ak * al / akl * rcd2);
                                /* 1 / (aij akl sqrt(aij + akl)) <= 1 / (aij akl sqrt(akl)) */
                                if (34.986836655249725 * bra_max * ekl * cmax[2][pk] * cmax[3][pl] * ddmax
                                    / (akl * sqrt(akl)) < prim_tol) continue;
                                /* x1[g][a][b][fk][fl] = sum_{ck,cl} c_k c_l dd[g][a][b][ck fk][cl fl] */
                                size_t nx1 = (size_t)NI * NJ * nfk * nfl;
                                int ck, cl, fk, fl, ci, cj, fi, fj;
                                for (g = 0; g < ngroup; g++) {
                                        double *xg = x1 + g * nx1, *ddg = dd + g * nd;
                                        for (a = 0; a < NI * NJ; a++) {
                                                for (fk = 0; fk < nfk; fk++)
                                                for (fl = 0; fl < nfl; fl++) {
                                                        double v = 0;
                                                        for (ck = 0; ck < nck; ck++)
                                                        for (cl = 0; cl < ncl; cl++)
                                                                v += cf[2][ck * nprim[2] + pk] * cf[3][cl * nprim[3] + pl]
                                                                   * ddg[(size_t)a * NK * NL + (ck * nfk + fk) * NL
                                                                         + cl * nfl + fl];
                                                        xg[((size_t)a * nfk + fk) * nfl + fl] = v;
                                                }
                                        }
                                }
                                double Q[3];
                                for (s = 0; s < 3; s++) Q[s] = (ak * R[2][s] + al * R[3][s]) / akl;
                                for (pj = 0; pj < nprim[1]; pj++) {
                                for (pi = 0; pi < nprim[0]; pi++) {
                                        double ai = ex[0][pi], aj = ex[1][pj];
                                        double aij = ai + aj;
                                        double eij = eijs[pj * nprim[0] + pi];
                                        double asum = aij + akl;
                                        double pref = 34.986836655249725 /* 2 pi^(5/2) */ / (aij * akl * sqrt(asum)) * eij * ekl;
                                        if (pref * ddmax * cmax[0][pi] * cmax[1][pj] * cmax[2][pk] * cmax[3][pl]
                                            < prim_tol) continue;
                                        /* w[g][fi][fj][fk][fl] = sum_{ci,cj} c_i c_j x1[g][ci fi][cj fj][fk][fl] */
                                        double wmax = 0;
                                        for (g = 0; g < ngroup; g++) {
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
                                        double P[3], PQ[3];
                                        for (s = 0; s < 3; s++) {
                                                P[s] = (ai * R[0][s] + aj * R[1][s]) / aij;
                                                PQ[s] = P[s] - Q[s];
                                        }
                                        double rho = aij * akl / asum;
                                        double x = rho * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);
                                        if (pref * wmax < prim_tol) continue;
                                        double u[MAXROOTS], rw[MAXROOTS];
                                        CINTrys_roots(nroots, x, u, rw);
                                        int r;
                                        for (r = 0; r < nroots; r++) {
                                                double t2 = u[r] / (1 + u[r]);
                                                lanes.b00[nlane] = .5 * t2 / asum;
                                                lanes.b10[nlane] = .5 / aij - .5 * akl * t2 / (aij * asum);
                                                lanes.b01[nlane] = .5 / akl - .5 * aij * t2 / (akl * asum);
                                                for (s = 0; s < 3; s++) {
                                                        lanes.c00[s][nlane] = P[s] - R[0][s] - akl * t2 / asum * PQ[s];
                                                        lanes.c00p[s][nlane] = Q[s] - R[2][s] + aij * t2 / asum * PQ[s];
                                                }
                                                lanes.gz[nlane] = pref * rw[r];
                                                lanes.ai2[nlane] = 2 * ai;
                                                lanes.aj2[nlane] = 2 * aj;
                                                lanes.ak2[nlane] = 2 * ak;
                                                for (a = 0; a < ngroup * nf4; a++) wl[(size_t)a * VL + nlane] = w[a];
                                                if (++nlane == VL) FLUSH_LANES();
                                        }
                                } }
                        } }
                        if (nlane > 0) {
                                for (v = nlane; v < VL; v++) {
                                        for (s = 0; s < 3; s++) lanes.c00[s][v] = lanes.c00p[s][v] = 0;
                                        lanes.b00[v] = lanes.b10[v] = lanes.b01[v] = lanes.gz[v] = 0;
                                        lanes.ai2[v] = lanes.aj2[v] = lanes.ak2[v] = 0;
                                        for (a = 0; a < ngroup * nf4; a++) wl[(size_t)a * VL + v] = 0;
                                }
                                FLUSH_LANES();
                        }
                        {
                                int s1, t1, x;
                                for (g = 0; g < ngroup; g++) {
                                        const double *ag = acc + (size_t)g * NTERM * VL;
                                        double sum[NTERM];
                                        int k;
                                        for (k = 0; k < NTERM; k++) {
                                                double t3 = 0;
                                                for (v = 0; v < VL; v++) t3 += ag[k * VL + v];
                                                sum[k] = t3;
                                        }
                                        for (s1 = 0; s1 < 3; s1++)
                                                for (t1 = s1; t1 < 3; t1++)
                                                        for (x = 0; x < 3; x++)
                                                                G[g][s1][t1][x][x] = sum[x * 6 + PAIR[s1][t1]];
                                        for (s1 = 0; s1 < 3; s1++)
                                                for (t1 = 0; t1 < 3; t1++) {
                                                        G[g][s1][t1][0][1] = sum[18 + s1 * 3 + t1];
                                                        G[g][s1][t1][0][2] = sum[27 + s1 * 3 + t1];
                                                        G[g][s1][t1][1][2] = sum[36 + s1 * 3 + t1];
                                                }
                                        E[g] = sum[45];
                                }
                        }
#undef FLUSH_LANES
                        if (mode == 1) {
                                for (g = 0; g < ngroup; g++) hloc[g] += E[g];
                                continue;
                        }
                        /* complete G: symmetric diagonal blocks, G[s][t][y][x] = G[t][s][x][y] */
                        for (g = 0; g < ngroup; g++) {
                                double Gf[4][4][3][3];
                                int s1, t1, x, y;
                                for (s1 = 0; s1 < 3; s1++)
                                for (t1 = 0; t1 < 3; t1++)
                                for (x = 0; x < 3; x++)
                                for (y = 0; y < 3; y++) {
                                        double v;
                                        if (x == y) {
                                                v = s1 <= t1 ? G[g][s1][t1][x][x] : G[g][t1][s1][x][x];
                                        } else if (x < y) {
                                                v = G[g][s1][t1][x][y];
                                        } else {
                                                v = G[g][t1][s1][y][x];
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
        }
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
        free(wl);
        free(acc);
        free(cm);
        free(eijs);
}
}
