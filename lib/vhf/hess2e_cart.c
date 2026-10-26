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
 * two, and per Cartesian direction a table of the 1D values
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
#define LMAX 6
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

/*
 * 1D integrals I[i][j][k][l] (i <= li+2, j <= lj+2, k <= lk+2, l <= ll) of one
 * direction for one root, and from them the NV derivative values for every
 * power tuple (i <= li, j <= lj, k <= lk, l <= ll), stored at
 * T[((i * (lj+1) + j) * (lk+1) + k) * (ll+1) + l][NV].
 */
static void tables_1d(double *T, const int *l, double c00, double c00p,
                      double b10, double b01, double b00, double ab, double cd,
                      double g00, double ai, double aj, double ak)
{
        const int li = l[0], lj = l[1], lk = l[2], ll = l[3];
        const int nmax = li + lj + 2, mmax = lk + ll + 2;
        double g[NVRR][NVRR];
        int n, m, i, j, k, q;
        /* VRR */
        g[0][0] = g00;
        if (nmax > 0) g[1][0] = c00 * g00;
        for (n = 1; n < nmax; n++) g[n + 1][0] = c00 * g[n][0] + n * b10 * g[n - 1][0];
        for (n = 0; n <= nmax; n++) {
                if (mmax > 0) g[n][1] = c00p * g[n][0] + (n > 0 ? n * b00 * g[n - 1][0] : 0);
                for (m = 1; m < mmax; m++) {
                        g[n][m + 1] = c00p * g[n][m] + m * b01 * g[n][m - 1]
                                    + (n > 0 ? n * b00 * g[n - 1][m] : 0);
                }
        }
        /* bra HRR: hb[i][j][m] for i + j <= nmax, j <= lj + 2 */
        double hb[NVRR][LMAX + 3][NVRR];
        for (i = 0; i <= nmax; i++)
                for (m = 0; m <= mmax; m++) hb[i][0][m] = g[i][m];
        for (j = 1; j <= lj + 2 && j <= nmax; j++) {
                for (i = 0; i + j <= nmax; i++) {
                        for (m = 0; m <= mmax; m++)
                                hb[i][j][m] = hb[i + 1][j - 1][m] + ab * hb[i][j - 1][m];
                }
        }
        /* ket HRR: I[i][j][k][l] for k <= lk + 2, l <= ll, k + l <= mmax */
        const int nk = lk + 3, nl = ll + 1;
#define IDX(i, j, k, q) ((((i) * (lj + 3) + (j)) * nk + (k)) * nl + (q))
        double I4[(LMAX + 3) * (LMAX + 3) * (LMAX + 3) * (LMAX + 1)];
        double e[NVRR][LMAX + 1];
        for (i = 0; i <= li + 2; i++) {
                for (j = 0; j <= lj + 2; j++) {
                        if (i + j > nmax) continue;
                        for (m = 0; m <= mmax; m++) e[m][0] = hb[i][j][m];
                        for (q = 1; q <= ll; q++)
                                for (k = 0; k + q <= mmax; k++)
                                        e[k][q] = e[k + 1][q - 1] + cd * e[k][q - 1];
                        for (k = 0; k <= lk + 2; k++)
                                for (q = 0; q <= ll; q++)
                                        if (k + q <= mmax) I4[IDX(i, j, k, q)] = e[k][q];
                }
        }
        /* derivative values */
        const double ai2 = 2 * ai, aj2 = 2 * aj, ak2 = 2 * ak;
        int t = 0;
        for (i = 0; i <= li; i++) {
        for (j = 0; j <= lj; j++) {
        for (k = 0; k <= lk; k++) {
        for (q = 0; q <= ll; q++, t++) {
                double *v = T + (size_t)t * NV;
#define G(a, b, c) I4[IDX(a, b, c, q)]
                v[V0] = G(i, j, k);
                v[VA] = ai2 * G(i + 1, j, k) - (i ? i * G(i - 1, j, k) : 0);
                v[VB] = aj2 * G(i, j + 1, k) - (j ? j * G(i, j - 1, k) : 0);
                v[VC] = ak2 * G(i, j, k + 1) - (k ? k * G(i, j, k - 1) : 0);
                v[VAA] = ai2 * ai2 * G(i + 2, j, k) - ai2 * (2 * i + 1) * G(i, j, k)
                       + (i > 1 ? i * (i - 1) * G(i - 2, j, k) : 0);
                v[VBB] = aj2 * aj2 * G(i, j + 2, k) - aj2 * (2 * j + 1) * G(i, j, k)
                       + (j > 1 ? j * (j - 1) * G(i, j - 2, k) : 0);
                v[VCC] = ak2 * ak2 * G(i, j, k + 2) - ak2 * (2 * k + 1) * G(i, j, k)
                       + (k > 1 ? k * (k - 1) * G(i, j, k - 2) : 0);
                {
                        double s = ai2 * aj2 * G(i + 1, j + 1, k);
                        if (j) s -= ai2 * j * G(i + 1, j - 1, k);
                        if (i) s -= aj2 * i * G(i - 1, j + 1, k);
                        if (i && j) s += i * j * G(i - 1, j - 1, k);
                        v[VAB] = s;
                        s = ai2 * ak2 * G(i + 1, j, k + 1);
                        if (k) s -= ai2 * k * G(i + 1, j, k - 1);
                        if (i) s -= ak2 * i * G(i - 1, j, k + 1);
                        if (i && k) s += i * k * G(i - 1, j, k - 1);
                        v[VAC] = s;
                        s = aj2 * ak2 * G(i, j + 1, k + 1);
                        if (k) s -= aj2 * k * G(i, j + 1, k - 1);
                        if (j) s -= ak2 * j * G(i, j - 1, k + 1);
                        if (j && k) s += j * k * G(i, j - 1, k - 1);
                        v[VBC] = s;
                }
#undef G
        } } } }
#undef IDX
}

/* index of the symmetric second-derivative value for centres s, t in {A,B,C} */
static const int D2[3][3] = {{VAA, VAB, VAC}, {VAB, VBB, VBC}, {VAC, VBC, VCC}};

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
        const int maxt = (LMAX + 1) * (LMAX + 1) * (LMAX + 1) * (LMAX + 1);
        double *T = malloc(sizeof(double) * 3 * maxt * NV);
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
                                                double b00 = .5 * t2 / asum;
                                                double b10 = .5 / aij - .5 * akl * t2 / (aij * asum);
                                                double b01 = .5 / akl - .5 * aij * t2 / (akl * asum);
                                                for (s = 0; s < 3; s++) {
                                                        double c00 = P[s] - R[0][s] - akl * t2 / asum * PQ[s];
                                                        double c00p = Q[s] - R[2][s] + aij * t2 / asum * PQ[s];
                                                        tables_1d(T + (size_t)s * maxt * NV, qs.l, c00, c00p,
                                                                  b10, b01, b00, ab[s], cdv[s],
                                                                  s == 2 ? pref * rw[r] : 1., ai, aj, ak);
                                                }
                                                const double *TX = T, *TY = T + (size_t)maxt * NV,
                                                             *TZ = T + (size_t)2 * maxt * NV;
                                                const int l1 = qs.l[1] + 1, l2 = qs.l[2] + 1, l3 = qs.l[3] + 1;
                                                int f = 0;
                                                for (fi = 0; fi < nfi; fi++)
                                                for (fj = 0; fj < nfj; fj++)
                                                for (fk = 0; fk < nfk; fk++)
                                                for (fl = 0; fl < nfl; fl++, f++) {
                                                        const double *X = TX + (size_t)(((qs.px[0][fi] * l1 + qs.px[1][fj]) * l2
                                                                         + qs.px[2][fk]) * l3 + qs.px[3][fl]) * NV;
                                                        const double *Y = TY + (size_t)(((qs.py[0][fi] * l1 + qs.py[1][fj]) * l2
                                                                         + qs.py[2][fk]) * l3 + qs.py[3][fl]) * NV;
                                                        const double *Z = TZ + (size_t)(((qs.pz[0][fi] * l1 + qs.pz[1][fj]) * l2
                                                                         + qs.pz[2][fk]) * l3 + qs.pz[3][fl]) * NV;
                                                        for (g = 0; g < ngroup; g++) {
                                                                double wv = w[g * nf4 + f];
                                                                if (mode == 1) {
                                                                        E[g] += wv * X[V0] * Y[V0] * Z[V0];
                                                                        continue;
                                                                }
                                                                double yz = wv * Y[V0] * Z[V0];
                                                                double xz = wv * X[V0] * Z[V0];
                                                                double xy = wv * X[V0] * Y[V0];
                                                                int s1, t1;
                                                                for (s1 = 0; s1 < 3; s1++) {
                                                                        for (t1 = s1; t1 < 3; t1++) {
                                                                                int k2 = D2[s1][t1];
                                                                                G[g][s1][t1][0][0] += X[k2] * yz;
                                                                                G[g][s1][t1][1][1] += Y[k2] * xz;
                                                                                G[g][s1][t1][2][2] += Z[k2] * xy;
                                                                        }
                                                                        double xs = wv * X[1 + s1], ys = wv * Y[1 + s1];
                                                                        for (t1 = 0; t1 < 3; t1++) {
                                                                                G[g][s1][t1][0][1] += xs * Y[1 + t1] * Z[V0];
                                                                                G[g][s1][t1][0][2] += xs * Y[V0] * Z[1 + t1];
                                                                                G[g][s1][t1][1][2] += ys * X[V0] * Z[1 + t1];
                                                                        }
                                                                }
                                                        }
                                                }
                                        }
                                } }
                        } }
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
        free(cm);
        free(eijs);
}
}
