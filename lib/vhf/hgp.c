/*
 * Head-Gordon-Pople (Obara-Saika VRR, contraction, HRR) first derivatives of
 * E_g = jfac_g E_J + kfac_g E_K at a fixed symmetric density (the conventions
 * of hess2e_cart.c), for contracted Cartesian shells (general contraction
 * natively, up to MAXCTR contractions per shell):
 *
 *   per primitive quartet: Boys functions F_m(T), m <= L + 1 (Taylor table),
 *     the vertical recursion for [e0|f0]^(m) (deg e + deg f <= L + 1), and the
 *     accumulation into four contracted streams per contraction combination
 *     (c_i c_j | c_k c_l) with the weights c, 2 a_A c, 2 a_B c, 2 a_C c (c the
 *     coefficient product);
 *   per shell quartet: the horizontal recursion of each stream to the shapes
 *     the derivatives need, d/dA (ab|cd) = 2 a_A (a+1 b|cd) - a_x (a-1 b|cd)
 *     (and B, C; D by translational invariance), contracted with the density
 *     weights dd.
 *
 * Only the Boys functions and a short recursion are done per primitive
 * quartet, so the cost of a contracted low-l quartet follows its primitive
 * count with a small constant (the Rys kernel does the whole recursion and the
 * derivative assembly per primitive quartet and root).
 *
 * Screening as hess2e_cart.c: shell quartets by q_ij q_kl times the J / K
 * density bounds, primitive quartets by the Gaussian prefactor times the
 * largest weight.  grad (ngroup, natm, 3) += d E_g; mode 1: grad[g] += E_g.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"

#define LMAX 4
#define MAXPRIM 64
#define MAXGROUP 4
#define MAXCTR 4
#define EMAX (2 * LMAX + 1)                     /* bra / ket degree of the VRR */
#define MMAX (4 * LMAX + 1)                     /* Boys order */
#define N3(d) (((d) + 1) * ((d) + 2) * ((d) + 3) / 6)   /* components of degree <= d */
#define N2(d) (((d) + 1) * ((d) + 2) / 2)               /* components of degree d */
#define NE3 N3(EMAX)

static inline double dmax2(double a, double b) { return a > b ? a : b; }

/* quartets with l_i + l_j + l_k + l_l in [hgp_lmin, hgp_lmax] only (default all) */
static int hgp_lmin = 0, hgp_lmax = 1000;
void SOCVHFhgp_set_lrange(int lo, int hi) { hgp_lmin = lo; hgp_lmax = hi; }

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

/* ---- Cartesian index tables: degree-ordered, libcint order within a degree ---- */
static int c_ready = 0;
static int c_deg[NE3 + N2(EMAX + 1)], c_pow[NE3 + N2(EMAX + 1)][3];
static int c_dir[NE3], c_dec1[NE3], c_dec2[NE3], c_n1[NE3];  /* VRR along the first nonzero component */
static int c_dec[NE3][3];                                    /* e - 1_k (-1 if e_k = 0) */
static int c_inc[NE3][3];                                    /* e + 1_k */
static int c_idx[EMAX + 2][EMAX + 2][EMAX + 2];

/* ---- Boys functions: Taylor table ---- */
#define BT_STEP 0.05
#define BT_TMAX 40.
#define BT_N ((int)(BT_TMAX / BT_STEP) + 2)
#define BT_ORD 8                                   /* Taylor terms */
#define BT_M (MMAX + BT_ORD + 1)
static double *bt_tab = NULL;                      /* [k][n] = F_n(k * step) */

static void boys_series(double *f, double t, int m)
{
        /* F_m by its series, downward recursion (exact to rounding for t < ~60) */
        const double et = exp(-t);
        double term = 1. / (2 * m + 1), sum = term;
        int k, n;
        for (k = 1; k < 400; k++) {
                term *= 2 * t / (2 * m + 2 * k + 1);
                sum += term;
                if (term < 1e-18 * sum) break;
        }
        f[m] = et * sum;
        for (n = m; n > 0; n--) f[n - 1] = (2 * t * f[n] + et) / (2 * n - 1);
}

static void tables_init(void)
{
        int d, x, y, n = 0, k;
        for (d = 0; d <= EMAX + 1; d++)
                for (x = d; x >= 0; x--)
                        for (y = d - x; y >= 0; y--) {
                                const int z = d - x - y;
                                c_idx[x][y][z] = n;
                                c_deg[n] = d;
                                c_pow[n][0] = x; c_pow[n][1] = y; c_pow[n][2] = z;
                                n++;
                        }
        for (n = 0; n < NE3; n++) {
                for (k = 0; k < 3; k++) {
                        int p[3] = {c_pow[n][0], c_pow[n][1], c_pow[n][2]};
                        p[k] += 1;
                        c_inc[n][k] = c_idx[p[0]][p[1]][p[2]];
                        p[k] -= 2;
                        c_dec[n][k] = p[k] >= 0 ? c_idx[p[0]][p[1]][p[2]] : -1;
                }
                if (n == 0) continue;
                k = c_pow[n][0] ? 0 : (c_pow[n][1] ? 1 : 2);
                c_dir[n] = k;
                c_dec1[n] = c_dec[n][k];
                c_n1[n] = c_pow[n][k] - 1;
                c_dec2[n] = c_n1[n] > 0 ? c_dec[c_dec1[n]][k] : 0;
        }
        bt_tab = malloc(sizeof(double) * BT_N * BT_M);
        for (k = 0; k < BT_N; k++) boys_series(bt_tab + (size_t)k * BT_M, k * BT_STEP, BT_M - 1);
        c_ready = 1;
}

/* f[n] = F_n(t), n = 0..m */
static inline void boys_fn(double *f, double t, int m)
{
        int n;
        if (t < BT_TMAX) {
                const int k = (int)(t / BT_STEP + .5);
                const double dt = k * BT_STEP - t;     /* F_n(t) = sum_j F_{n+j}(t_k) dt^j / j! */
                const double *tk = bt_tab + (size_t)k * BT_M;
                double s = 0, c = 1;
                int j;
                for (j = 0; j < BT_ORD; j++) {
                        s += c * tk[m + j];
                        c *= dt / (j + 1);
                }
                f[m] = s;
                if (m > 0) {
                        const double et = exp(-t);
                        for (n = m; n > 0; n--) f[n - 1] = (2 * t * f[n] + et) / (2 * n - 1);
                }
        } else {
                const double et = exp(-t), h = .5 / t;
                f[0] = .886226925452758014 / sqrt(t);
                for (n = 0; n < m; n++) f[n + 1] = h * ((2 * n + 1) * f[n] - et);
        }
}

/* ---- primitive-pair tables (as hess2e_cart.c) ---- */
enum { PP_A, PP_IA, PP_E, PP_EC, PP_ECS, PP_P, PP_PA = PP_P + 3, PPN = PP_PA + 3 };

/*
 * The horizontal recursion of one side: src[e * ncol + col] for e of degree
 * l1 .. l1 + l2 (global component indices), dst[(a * nb + b) * ncol + col] for
 * a of degree l1, b of degree l2 (local indices), (a, b + 1_k) = (a + 1_k, b)
 * + R_k (a, b), R = A - B.  wk: scratch.
 */
static void hrr(double *dst, const double *src, int ncol, int l1, int l2, const double *R, double *wk)
{
        const int nb2 = N2(l2);
        int j, a, b, col;
        if (l2 == 0) {
                memcpy(dst, src + (size_t)N3(l1 - 1) * ncol, sizeof(double) * N2(l1) * ncol);
                return;
        }
        /* level j: w[(a_global - a0) * nbj + b_local][col], a degree l1 .. l1 + l2 - j */
        const int a0 = l1 > 0 ? N3(l1 - 1) : 0;
        const int na = N3(l1 + l2) - a0;
        double *prev = wk, *cur = wk + (size_t)na * nb2 * ncol;
        memcpy(prev, src + (size_t)a0 * ncol, sizeof(double) * na * ncol);   /* nbj = 1 at j = 0 */
        int nbp = 1;
        for (j = 1; j <= l2; j++) {
                const int nbj = N2(j), amax = N3(l1 + l2 - j) - a0;
                const int b0 = N3(j - 1), bp0 = j > 1 ? N3(j - 2) : 0;
                for (a = 0; a < amax; a++) {
                        const int ag = a0 + a;
                        for (b = 0; b < nbj; b++) {
                                const int bg = b0 + b, k = c_dir[bg], b1 = c_dec1[bg] - bp0;
                                const int a1 = c_inc[ag][k] - a0;
                                const double *u = prev + ((size_t)a1 * nbp + b1) * ncol;
                                const double *v = prev + ((size_t)a * nbp + b1) * ncol;
                                double *o = cur + ((size_t)a * nbj + b) * ncol;
                                const double r = R[k];
                                for (col = 0; col < ncol; col++) o[col] = u[col] + r * v[col];
                        }
                }
                double *t = prev; prev = cur; cur = t;
                nbp = nbj;
        }
        /* a of degree l1 only */
        memcpy(dst, prev, sizeof(double) * N2(l1) * nb2 * ncol);
}

/*
 * (la' lb' | lc' ld') of one contracted stream S[e * NF + f] (e, f global,
 * NF = N3(F)): bra HRR, transpose, ket HRR; out[((c * nd + d) * na + a) * nb + b].
 */
static void stream_integrals(double *out, const double *S, int NF, int la, int lb, int lc, int ld,
                             const double *AB, const double *CD, double *wk)
{
        const int f0 = lc > 0 ? N3(lc - 1) : 0, f1 = N3(lc + ld);
        const int nf = f1 - f0, nab = N2(la) * N2(lb);
        double *bra = wk, *t = bra + (size_t)nab * NF, *w2 = t + (size_t)NF * nab;
        int e, f, ab;
        /* bra HRR over the columns f0 .. f1 */
        double *src = w2;
        const int e0 = la > 0 ? N3(la - 1) : 0, e1 = N3(la + lb);
        for (e = 0; e < e1; e++) {
                if (e < e0) continue;
                for (f = 0; f < nf; f++) src[(size_t)e * nf + f] = S[(size_t)e * NF + f0 + f];
        }
        hrr(bra, src, nf, la, lb, AB, w2 + (size_t)e1 * nf);
        /* t[f][ab] with f global from 0 (only f0 .. f1 filled) */
        for (ab = 0; ab < nab; ab++)
                for (f = 0; f < nf; f++) t[(size_t)(f0 + f) * nab + ab] = bra[(size_t)ab * nf + f];
        hrr(out, t, nab, lc, ld, CD, w2);
}

void SOCVHFhgp_grad2e(double *grad, int mode, int ngroup, double *jfac, double *kfac,
                      double *dm, double *qcond, double *dcond, double tol, double prim_tol,
                      int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        if (!c_ready) tables_init();
        const size_t nao = ao_loc[nbas];
        const size_t nh1 = mode == 1 ? 1 : (size_t)natm * 3;
        const size_t nh = nh1 * ngroup;
        const int npair = nbas * (nbas + 1) / 2;
        int ib, g, mxctr = 1;
        for (ib = 0; ib < nbas; ib++) {
                if (bas[ib * BAS_SLOTS + ANG_OF] > LMAX || bas[ib * BAS_SLOTS + NPRIM_OF] > MAXPRIM
                    || bas[ib * BAS_SLOTS + NCTR_OF] > MAXCTR) return;          /* the caller checks */
                if (bas[ib * BAS_SLOTS + NCTR_OF] > mxctr) mxctr = bas[ib * BAS_SLOTS + NCTR_OF];
        }
        double jw = 0, kw = 0;
        for (g = 0; g < ngroup; g++) {
                jw = dmax2(jw, fabs(jfac[g]) * 4);
                kw = dmax2(kw, fabs(kfac[g]) * 2);
        }
        /* max_c |c| of every primitive */
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
        /* primitive pairs of every shell pair (index p_j np_i + p_i) */
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
        int maxf = 0, lmx = 0;
        for (ib = 0; ib < nbas; ib++) {
                int nf = ao_loc[ib + 1] - ao_loc[ib];
                maxf = nf > maxf ? nf : maxf;
                lmx = bas[ib * BAS_SLOTS + ANG_OF] > lmx ? bas[ib * BAS_SLOTS + ANG_OF] : lmx;
        }
        const size_t maxf4 = (size_t)maxf * maxf * maxf * maxf;
        const int mxcomb = mxctr * mxctr * mxctr * mxctr;
        /* workspace for the largest l of the basis: VRR (4 l + 3) x NE x NE,
         * streams 4 ncomb NE x NE, the six integral shapes, HRR */
        const size_t NE = N3(2 * lmx + 1), nab = (size_t)N2(lmx + 1) * N2(lmx);
        const size_t nvrr = (size_t)(4 * lmx + 3) * NE * NE;
        const size_t nstr = NE * NE;
        const size_t ni = nab * N2(lmx) * N2(lmx);
        const size_t nhrr = 2 * nab * NE + NE * NE + 2 * NE * N2(lmx + 1) * (NE > nab ? NE : nab);

#pragma omp parallel
{
        double *hloc = calloc(nh, sizeof(double));
        double *dd = malloc(sizeof(double) * maxf4 * ngroup);
        double *V = malloc(sizeof(double) * nvrr);
        double *S = malloc(sizeof(double) * 4 * mxcomb * nstr);
        double *Ibuf = malloc(sizeof(double) * 6 * ni);
        double *wk = malloc(sizeof(double) * nhrr);
        double F[MMAX + 2], cc[MAXCTR * MAXCTR * MAXCTR * MAXCTR];
        int ij, kl, ish, jsh, ksh, lsh;
#pragma omp for schedule(dynamic, 2)
        for (ij = 0; ij < npair; ij++) {
                shell_pair(ij, &ish, &jsh);
                const double bra_max = brmax[ij];
                const double *ppij = ppd + ppoff[ij] * PPN;
                for (kl = 0; kl <= ij; kl++) {
                        shell_pair(kl, &ksh, &lsh);
                        const double q4 = qcond[ish * nbas + jsh] * qcond[ksh * nbas + lsh];
                        const double dbound = jw * dcond[ish * nbas + jsh] * dcond[ksh * nbas + lsh]
                                            + kw * dmax2(dcond[ish * nbas + ksh] * dcond[jsh * nbas + lsh],
                                                         dcond[ish * nbas + lsh] * dcond[jsh * nbas + ksh]);
                        if (q4 * dbound < tol) continue;
                        const int sh[4] = {ish, jsh, ksh, lsh};
                        int l[4], nf[4], nprim[4], nctr[4], at[4], s;
                        double *ex[4], *cf[4], *R[4];
                        double fac = 1;
                        for (s = 0; s < 4; s++) {
                                const int *b = bas + sh[s] * BAS_SLOTS;
                                l[s] = b[ANG_OF];
                                nf[s] = N2(l[s]);
                                nprim[s] = b[NPRIM_OF];
                                nctr[s] = b[NCTR_OF];
                                at[s] = b[ATOM_OF];
                                ex[s] = env + b[PTR_EXP];
                                cf[s] = env + b[PTR_COEFF];
                                R[s] = env + atm[at[s] * ATM_SLOTS + PTR_COORD];
                                fac *= common_fac_sp(l[s]);
                        }
                        const int L = l[0] + l[1] + l[2] + l[3];
                        if (L < hgp_lmin || L > hgp_lmax) continue;
                        if (ish == jsh) fac *= .5;
                        if (ksh == lsh) fac *= .5;
                        if (ij == kl) fac *= .5;
                        const int nfi = nf[0], nfj = nf[1], nfk = nf[2], nfl = nf[3];
                        const int nci = nctr[0], ncj = nctr[1], nck = nctr[2], ncl = nctr[3];
                        const int ncomb = nci * ncj * nck * ncl;
                        /* contracted functions (c, f) -> c * nf + f; dd[g][I][J][K][L] */
                        const int NI = nci * nfi, NJ = ncj * nfj, NK = nck * nfk, NL = ncl * nfl;
                        const size_t nd = (size_t)NI * NJ * NK * NL;
                        const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh], c0 = ao_loc[ksh], e0 = ao_loc[lsh];
                        int a, b, c, d;
                        double ddmax = 0;
                        for (g = 0; g < ngroup; g++) {
                                double *ddg = dd + g * nd;
                                const double jf = 4 * jfac[g] * fac, kf = kfac[g] * fac;
                                for (a = 0; a < NI; a++)
                                for (b = 0; b < NJ; b++) {
                                        const double dab = jf * dm[(a0 + a) * nao + b0 + b];
                                        const double *dA = dm + (a0 + a) * nao, *dB = dm + (b0 + b) * nao;
                                        for (c = 0; c < NK; c++) {
                                                const double dbc = dB[c0 + c], dac = dA[c0 + c];
                                                const double *dC = dm + (c0 + c) * nao;
                                                double *o = ddg + ((a * NJ + b) * NK + c) * NL;
                                                for (d = 0; d < NL; d++) {
                                                        o[d] = dab * dC[e0 + d] + kf * (dac * dB[e0 + d] + dA[e0 + d] * dbc);
                                                        ddmax = dmax2(ddmax, fabs(o[d]));
                                                }
                                        }
                                }
                        }
                        if (ddmax == 0) continue;
                        ddmax *= ncomb;             /* |sum_comb c..c dd| <= ncomb max|c|^4 max|dd| */
                        const int dord = mode == 1 ? 0 : 1;
                        const int E = l[0] + l[1] + dord, Fk = l[2] + l[3] + dord, M = L + dord;
                        const int NEb = N3(E), NFk = N3(Fk);
                        const size_t nsf = (size_t)NEb * NFk;
                        double AB[3], CD[3];
                        for (s = 0; s < 3; s++) {
                                AB[s] = R[0][s] - R[1][s];
                                CD[s] = R[2][s] - R[3][s];
                        }
                        /* streams S[(st * ncomb + comb) * nsf + e * NFk + f], st: weights 1, 2 a_A, 2 a_B, 2 a_C */
                        const int nst = mode == 1 ? 1 : 4;
                        memset(S, 0, sizeof(double) * nst * ncomb * nsf);
                        const int la_ = l[0], lb_ = l[1], lc_ = l[2], ld_ = l[3];
                        const int eb[4][2] = {{la_ - 1, la_ + lb_}, {la_ + 1, la_ + lb_ + 1},
                                              {la_, la_ + lb_ + 1}, {la_, la_ + lb_}};
                        const int fb[4][2] = {{lc_ - 1, lc_ + ld_}, {lc_, lc_ + ld_},
                                              {lc_, lc_ + ld_}, {lc_ + 1, lc_ + ld_ + 1}};
                        int ebnd[4][2], fbnd[4][2], st;
                        for (st = 0; st < 4; st++) {
                                ebnd[st][0] = eb[st][0] > 0 ? N3(eb[st][0] - 1) : 0;
                                ebnd[st][1] = N3(eb[st][1] < E ? eb[st][1] : E);
                                fbnd[st][0] = fb[st][0] > 0 ? N3(fb[st][0] - 1) : 0;
                                fbnd[st][1] = N3(fb[st][1] < Fk ? fb[st][1] : Fk);
                        }
                        if (mode == 1) {        /* the energy: (la lb | lc ld) from S0 */
                                ebnd[0][0] = la_ > 0 ? N3(la_ - 1) : 0;
                                fbnd[0][0] = lc_ > 0 ? N3(lc_ - 1) : 0;
                        }
                        const double *ppkl = ppd + ppoff[kl] * PPN;
                        const int npij = nprim[0] * nprim[1];
                        const double *cmx[4];
                        for (s = 0; s < 4; s++) cmx[s] = cmax_all + sh[s] * MAXPRIM;
                        int pk, pl;
                        for (pl = 0; pl < nprim[3]; pl++)
                        for (pk = 0; pk < nprim[2]; pk++) {
                                const double *dk = ppkl + ((size_t)pl * nprim[2] + pk) * PPN;
                                const double akl = dk[PP_A], iakl = dk[PP_IA];
                                if (34.986836655249725 * bra_max * dk[PP_ECS] * ddmax < prim_tol) continue;
                                const double eckl = dk[PP_EC] * ddmax, ekl = dk[PP_E];
                                const double *Q = dk + PP_P, *QC = dk + PP_PA;
                                const double a2c = 2 * ex[2][pk];
                                int pij;
                                for (pij = 0; pij < npij; pij++) {
                                        const double *di = ppij + (size_t)pij * PPN;
                                        const double aij = di[PP_A], iaij = di[PP_IA];
                                        const double ias = 1. / (aij + akl);
                                        const double pr = 34.986836655249725 * iaij * iakl * sqrt(ias);
                                        if (pr * di[PP_EC] * eckl < prim_tol) continue;
                                        const int pi = pij % nprim[0], pj = pij / nprim[0];
                                        const double pref = pr * di[PP_E] * ekl;
                                        const double rho = aij * akl * ias;
                                        double PQ[3], WP[3], WQ[3];
                                        for (s = 0; s < 3; s++) {
                                                PQ[s] = di[PP_P + s] - Q[s];
                                                WP[s] = -akl * ias * PQ[s];
                                                WQ[s] = aij * ias * PQ[s];
                                        }
                                        boys_fn(F, rho * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]), M);
                                        /* V[(e * NFk + f) * MP + m] = [e0|f0]^(m), m innermost */
                                        int m, e, f;
                                        const int MP = M + 1;
                                        for (m = 0; m <= M; m++) V[m] = pref * F[m];
                                        const double h2p = .5 * iaij, rp = rho * iaij;
                                        for (e = 1; e < NEb; e++) {
                                                const int k = c_dir[e], n1 = c_n1[e], de = c_deg[e];
                                                const double pa = di[PP_PA + k], wp = WP[k];
                                                const double *v1 = V + (size_t)c_dec1[e] * NFk * MP;
                                                double *o = V + (size_t)e * NFk * MP;
                                                if (n1 > 0) {
                                                        const double *v2 = V + (size_t)c_dec2[e] * NFk * MP;
                                                        const double c2 = n1 * h2p, c3 = n1 * h2p * rp;
#pragma omp simd
                                                        for (m = 0; m <= M - de; m++)
                                                                o[m] = pa * v1[m] + wp * v1[m + 1] + c2 * v2[m] - c3 * v2[m + 1];
                                                } else {
#pragma omp simd
                                                        for (m = 0; m <= M - de; m++) o[m] = pa * v1[m] + wp * v1[m + 1];
                                                }
                                        }
                                        const double h2q = .5 * iakl, rq = rho * iakl, h2pq = .5 * ias;
                                        for (f = 1; f < NFk; f++) {
                                                const int k = c_dir[f], f1 = c_dec1[f], n1 = c_n1[f], df = c_deg[f];
                                                const int f2 = c_dec2[f];
                                                const double qc = QC[k], wq = WQ[k];
                                                const double c2 = n1 * h2q, c3 = n1 * h2q * rq;
                                                const int emax = N3(M - df < E ? M - df : E);
                                                for (e = 0; e < emax; e++) {
                                                        const int ek = c_pow[e][k], de = c_deg[e];
                                                        double *bse = V + (size_t)e * NFk * MP;
                                                        const double *v1 = bse + (size_t)f1 * MP, *v2 = bse + (size_t)f2 * MP;
                                                        double *o = bse + (size_t)f * MP;
                                                        const int mm = M - de - df;
                                                        if (n1 > 0) {
#pragma omp simd
                                                                for (m = 0; m <= mm; m++)
                                                                        o[m] = qc * v1[m] + wq * v1[m + 1] + c2 * v2[m] - c3 * v2[m + 1];
                                                        } else {
#pragma omp simd
                                                                for (m = 0; m <= mm; m++) o[m] = qc * v1[m] + wq * v1[m + 1];
                                                        }
                                                        if (ek > 0) {
                                                                const double *u = V + ((size_t)c_dec[e][k] * NFk + f1) * MP + 1;
                                                                const double c4 = ek * h2pq;
#pragma omp simd
                                                                for (m = 0; m <= mm; m++) o[m] += c4 * u[m];
                                                        }
                                                }
                                        }
                                        /* coefficient products of the contraction combinations */
                                        int ci, cj, ck, cl, cb = 0;
                                        for (ci = 0; ci < nci; ci++)
                                        for (cj = 0; cj < ncj; cj++) {
                                                const double cij = cf[0][ci * nprim[0] + pi] * cf[1][cj * nprim[1] + pj];
                                                for (ck = 0; ck < nck; ck++)
                                                for (cl = 0; cl < ncl; cl++, cb++)
                                                        cc[cb] = cij * cf[2][ck * nprim[2] + pk] * cf[3][cl * nprim[3] + pl];
                                        }
                                        /* m = 0 into the streams, each over the degree band its HRR reads */
                                        const double ws[4] = {1., 2 * ex[0][pi], 2 * ex[1][pj], a2c};
                                        for (st = 0; st < nst; st++) {
                                                for (e = ebnd[st][0]; e < ebnd[st][1]; e++) {
                                                        const double *v = V + (size_t)e * NFk * MP;
                                                        for (cb = 0; cb < ncomb; cb++) {
                                                                const double w = ws[st] * cc[cb];
                                                                double *so = S + (size_t)(st * ncomb + cb) * nsf + (size_t)e * NFk;
                                                                for (f = fbnd[st][0]; f < fbnd[st][1]; f++)
                                                                        so[f] += w * v[(size_t)f * MP];
                                                        }
                                                }
                                        }
                                }
                        }
                        /* per contraction combination: the integrals of the needed shapes,
                         * the derivative combination with dd */
                        const int la = la_, lb = lb_, lc = lc_, ld = ld_;
                        double sum[MAXGROUP][10];
                        memset(sum, 0, sizeof(sum));
                        double *IA = Ibuf, *IB = Ibuf + ni, *IC = IB + ni, *Ia = IC + ni, *Ib = Ia + ni, *Ic = Ib + ni;
                        const int na1 = N2(la + 1), nb1 = N2(lb + 1);
                        const int nam = la > 0 ? N2(la - 1) : 1, nbm = lb > 0 ? N2(lb - 1) : 1;
                        const int ga0 = la > 0 ? N3(la - 1) : 0, gb0 = lb > 0 ? N3(lb - 1) : 0;
                        const int gc0 = lc > 0 ? N3(lc - 1) : 0;
                        int ci, cj, ck, cl, cb = 0;
                        for (ci = 0; ci < nci; ci++)
                        for (cj = 0; cj < ncj; cj++)
                        for (ck = 0; ck < nck; ck++)
                        for (cl = 0; cl < ncl; cl++, cb++) {
                                const double *S0 = S + (size_t)cb * nsf;
                                if (mode == 1) {
                                        stream_integrals(Ibuf, S0, NFk, la, lb, lc, ld, AB, CD, wk);
                                        for (g = 0; g < ngroup; g++) {
                                                const double *ddg = dd + g * nd;
                                                for (a = 0; a < nfi; a++)
                                                for (b = 0; b < nfj; b++)
                                                for (c = 0; c < nfk; c++)
                                                for (d = 0; d < nfl; d++) {
                                                        const size_t n = (((size_t)(ci * nfi + a) * NJ + cj * nfj + b) * NK
                                                                          + ck * nfk + c) * NL + cl * nfl + d;
                                                        sum[g][9] += ddg[n] * Ibuf[((c * nfl + d) * nfi + a) * nfj + b];
                                                }
                                        }
                                        continue;
                                }
                                const double *SA = S + (size_t)(ncomb + cb) * nsf;
                                const double *SB = S + (size_t)(2 * ncomb + cb) * nsf;
                                const double *SC = S + (size_t)(3 * ncomb + cb) * nsf;
                                stream_integrals(IA, SA, NFk, la + 1, lb, lc, ld, AB, CD, wk);
                                stream_integrals(IB, SB, NFk, la, lb + 1, lc, ld, AB, CD, wk);
                                stream_integrals(IC, SC, NFk, la, lb, lc + 1, ld, AB, CD, wk);
                                if (la > 0) stream_integrals(Ia, S0, NFk, la - 1, lb, lc, ld, AB, CD, wk);
                                if (lb > 0) stream_integrals(Ib, S0, NFk, la, lb - 1, lc, ld, AB, CD, wk);
                                if (lc > 0) stream_integrals(Ic, S0, NFk, la, lb, lc - 1, ld, AB, CD, wk);
                                int x;
                                for (a = 0; a < nfi; a++)
                                for (b = 0; b < nfj; b++)
                                for (c = 0; c < nfk; c++)
                                for (d = 0; d < nfl; d++) {
                                        const size_t n = (((size_t)(ci * nfi + a) * NJ + cj * nfj + b) * NK
                                                          + ck * nfk + c) * NL + cl * nfl + d;
                                        const int ag = ga0 + a, bg = gb0 + b, cg = gc0 + c;
                                        double vA[3], vB[3], vC[3];
                                        for (x = 0; x < 3; x++) {
                                                /* local indices of a +- 1_x etc. within their degrees */
                                                const int ap = c_inc[ag][x] - N3(la), bp = c_inc[bg][x] - N3(lb);
                                                const int cp = c_inc[cg][x] - N3(lc);
                                                vA[x] = IA[((c * nfl + d) * na1 + ap) * nfj + b];
                                                vB[x] = IB[((c * nfl + d) * nfi + a) * nb1 + bp];
                                                vC[x] = IC[((cp * nfl + d) * nfi + a) * nfj + b];
                                                const int ax = c_pow[ag][x], bx = c_pow[bg][x], cx = c_pow[cg][x];
                                                if (ax > 0) {
                                                        const int aml = c_dec[ag][x] - (la > 1 ? N3(la - 2) : 0);
                                                        vA[x] -= ax * Ia[((c * nfl + d) * nam + aml) * nfj + b];
                                                }
                                                if (bx > 0) {
                                                        const int bml = c_dec[bg][x] - (lb > 1 ? N3(lb - 2) : 0);
                                                        vB[x] -= bx * Ib[((c * nfl + d) * nfi + a) * nbm + bml];
                                                }
                                                if (cx > 0) {
                                                        const int cml = c_dec[cg][x] - (lc > 1 ? N3(lc - 2) : 0);
                                                        vC[x] -= cx * Ic[((cml * nfl + d) * nfi + a) * nfj + b];
                                                }
                                        }
                                        for (g = 0; g < ngroup; g++) {
                                                const double w = dd[g * nd + n];
                                                for (x = 0; x < 3; x++) {
                                                        sum[g][x * 3 + 0] += w * vA[x];
                                                        sum[g][x * 3 + 1] += w * vB[x];
                                                        sum[g][x * 3 + 2] += w * vC[x];
                                                }
                                        }
                                }
                        }
                        if (mode == 1) {
                                for (g = 0; g < ngroup; g++) hloc[g] += sum[g][9];
                                continue;
                        }
                        for (g = 0; g < ngroup; g++) {
                                double *hg = hloc + g * nh1;
                                int x;
                                for (x = 0; x < 3; x++) {
                                        const double gA = sum[g][x * 3], gB = sum[g][x * 3 + 1], gC = sum[g][x * 3 + 2];
                                        hg[at[0] * 3 + x] += gA;
                                        hg[at[1] * 3 + x] += gB;
                                        hg[at[2] * 3 + x] += gC;
                                        hg[at[3] * 3 + x] -= gA + gB + gC;
                                }
                        }
                }
        }
#pragma omp critical
        {
                size_t n;
                for (n = 0; n < nh; n++) grad[n] += hloc[n];
        }
        free(hloc);
        free(dd);
        free(V);
        free(S);
        free(Ibuf);
        free(wk);
}
        free(cmax_all);
        free(ppoff);
        free(brmax);
        free(ppd);
}
