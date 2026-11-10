/*
 * Direct RI-J (density-fitted Coulomb without the three-centre tensor) by
 * McMurchie-Davidson on primitive Hermite Gaussians, for socutils.scf.rij_direct.
 *
 * Bra: the orbital basis decontracted to Cartesian primitives (one primitive
 * per shell, pyscf mol.decontract_basis(to_cart=True)), shell pairs a >= b with
 * Hermite densities rho^P_t = w_P sum_ab D_ab E^ab_t (w = 2 for a != b).  Ket:
 * the auxiliary primitives in spherical form, each a pure degree-l Hermite
 * Gaussian (Hobson: S_lm(r) e^{-q r^2} = (2q)^-l S_lm(d/dA) e^{-q r_A^2}), so a
 * primitive carries only its (l+1)(l+2)/2 degree-l Hermite components; the
 * contracted spherical functions, normalisation and the sign (-1)^l are the
 * caller's sparse map A (rho_aux = A^T X, h = A d).
 *
 *     pass 1 (rho):  X^Q_s = sum_P pref_PQ sum_t R_{t+s}(theta, P - Q) rho^P_t
 *     pass 2 (J):    V^P_t = sum_Q pref_PQ sum_s R_{t+s} h^Q_s,  J_ab = sum_t E^ab_t V^P_t
 *
 * with pref = 2 pi^(5/2) / (p q sqrt(p + q)), theta = p q / (p + q), |s| = l_Q.
 *
 * Far field: once theta |P - Q|^2 >= T_far (the Boys functions equal their
 * asymptotic forms to the relative accuracy far_eps), pref R_{t+s} equals
 * (pi/p)^(3/2) (pi/q)^(3/2) d_{t+s}(1/R) exactly to that accuracy, the
 * classical interaction of two point Hermite multipoles.  The auxiliary
 * primitives of one centre, sorted by exponent descending, are far from P for a
 * prefix of that order (theta grows with q), so pass 2 uses prefix sums of the
 * centre's Hermite multipoles M^C_k = sum_{i < k} (pi/q_i)^(3/2) h^i (one
 * multipole tensor per bra pair and centre, not per primitive), and pass 1
 * accumulates the bra side d(1/R) rho^P into buckets by the prefix length,
 * resolved by suffix sums at the end.  The near primitives go one by one.
 *
 * Vectorisation: per bra pair the near primitives (by l) and the far centres (by the largest l of
 * their far prefix) are collected into batches of RJ_NL lanes, each batch one call of the
 * generated lane kernels (Boys functions from the table, R recursion, contraction; omp simd
 * over the lanes).
 *
 * Screening: bra pairs by sqrt((ab|ab)) (qcond of the primitive shell pairs)
 * times the caller's bound of the other side; pass 1 also by the density-
 * weighted sqrt((Omega_P|Omega_P)).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "cint.h"

#define LP 7                    /* bra pair degree (f f) + 1 of the gradient */
#define LQ 6                    /* auxiliary angular momentum */
#define LT (LP + LQ)
#define NH(d) ((d) < 0 ? 0 : ((d) + 1) * ((d) + 2) * ((d) + 3) / 6)
#define NHQ NH(LQ)              /* the multipole slots of a centre, degrees 0..LQ */
#define PI_FAC 34.986836655249725               /* 2 pi^(5/2) */
#define PI32 5.568327996831708                  /* pi^(3/2) */

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

/* Boys functions from a table (F_m by a 6th-order Taylor step on t_k = k / 20,
 * downward recursion); t >= BOYS_TMAX: the asymptotic F_0 and upward recursion */
#define BOYS_TMAX 64.           /* > every T_far: the near field always from the table */
#define BOYS_NT 1281
#define BOYS_MT (LT + 8)
static double boys_tab[BOYS_NT][BOYS_MT];
static double boys_inv[BOYS_MT];
static double boys_exp[BOYS_NT];
static double t_far[LT + 1];            /* T_far by degree, for far_eps */
static double t_far_eps = -1;
static int rj_ready = 0;
long SOCVHFrij_direct_stats[4];          /* near, far evaluations of the last rho, vj passes */

#include "rij_direct_gen.c"

static inline void boys_fn(double *f, double t, int m)
{
        int n;
        if (t < BOYS_TMAX) {
                const int k = (int)(t * 20 + .5);
                const double dl = k * .05 - t;
                const double *b = boys_tab[k] + m;
                /* e^-t = e^-t_k e^dl, |dl| <= 1/40 */
                const double et = boys_exp[k] * (1 + dl * (1 + dl * .5 * (1 + dl * (1. / 3) * (1 + dl * .25
                                * (1 + dl * .2 * (1 + dl * (1. / 6)))))));
                f[m] = b[0] + dl * (b[1] + dl * .5 * (b[2] + dl * (1. / 3) * (b[3] + dl * .25 * (b[4]
                     + dl * .2 * (b[5] + dl * (1. / 6) * b[6])))));
                for (n = m; n > 0; n--) f[n - 1] = (2 * t * f[n] + et) * boys_inv[n];
        } else {
                const double et = exp(-t);
                f[0] = .886226925452758014 / sqrt(t);
                const double h = .5 / t;
                for (n = 0; n < m; n++) f[n + 1] = h * ((2 * n + 1) * f[n] - et);
        }
}

static void rj_init(double far_eps)
{
#pragma omp critical(rij_direct_init)
{
        if (!rj_ready) {
                int n;
                for (n = 0; n < BOYS_NT; n++) boys_exact(boys_tab[n], n * .05, BOYS_MT - 1);
                for (n = 1; n < BOYS_MT; n++) boys_inv[n] = 1. / (2 * n - 1);
                for (n = 0; n < BOYS_NT; n++) boys_exp[n] = exp(-n * .05);
                rj_ready = 1;
        }
        if (far_eps != t_far_eps) {
                int n;
                /* smallest T on a 0.25 grid from which max_n<=L |F_n / F_n^asym - 1| < far_eps */
                int L;
                for (L = 0; L <= LT; L++) {
                        double T = 200;
                        double t;
                        for (t = 200; t >= 1; t -= .25) {
                                double f[LT + 2];
                                boys_exact(f, t, L);
                                double a = .886226925452758014 / sqrt(t), e = 0;
                                for (n = 0; n <= L; n++) {
                                        const double r = fabs(f[n] / a - 1);
                                        if (r > e) e = r;
                                        a *= (2 * n + 1) * .5 / t;
                                }
                                if (e >= far_eps) break;
                                T = t;
                        }
                        t_far[L] = T < BOYS_TMAX - 1 ? T : BOYS_TMAX - 1;
                }
                t_far_eps = far_eps;
        }
}
}

static double common_fac_sp(int l)
{
        if (l == 0) return 0.282094791773878143;
        if (l == 1) return 0.488602511902919921;
        return 1.;
}

/* E^{ij}_t of one direction for i <= li, j <= lj: E[(i * (lj + 1) + j) * (li + lj + 1) + t] */
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

typedef struct {
        int ish, jsh;
        int lp;
        double p, P[3];
} RJPair;

/* E^{ab}_tuv of all Cartesian pairs of a primitive pair, coefficients included:
 * Et[(fa * nfb + fb) * NH(lij) + n] */
static void pair_Et(double *Et, const RJPair *pp, int *atm, int *bas, double *env, double *wk)
{
        const int ish = pp->ish, jsh = pp->jsh;
        const int li = bas[ish * BAS_SLOTS + ANG_OF], lj = bas[jsh * BAS_SLOTS + ANG_OF];
        const int lij = li + lj, nh = NH(lij);
        const double *Ra = env + atm[bas[ish * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double *Rb = env + atm[bas[jsh * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double ai = env[bas[ish * BAS_SLOTS + PTR_EXP]], aj = env[bas[jsh * BAS_SLOTS + PTR_EXP]];
        const double cc = env[bas[ish * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(li)
                        * env[bas[jsh * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(lj);
        const int s2 = lij + 1, s1 = (lj + 1) * s2, sz = (li + 1) * s1;
        double *Ex = wk, *Ey = Ex + sz, *Ez = Ey + sz;
        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
        hermite_E(Ex, li, lj, ai, aj, Ra[0], Rb[0], 1.);
        hermite_E(Ey, li, lj, ai, aj, Ra[1], Rb[1], 1.);
        hermite_E(Ez, li, lj, ai, aj, Ra[2], Rb[2], cc * exp(-ai * aj / (ai + aj) * r2));
        int ix, iy, jx, jy, n, fa = 0;
        const int nfb = (lj + 1) * (lj + 2) / 2;
        for (ix = li; ix >= 0; ix--)
        for (iy = li - ix; iy >= 0; iy--, fa++) {
                const int iz = li - ix - iy;
                int fb = 0;
                for (jx = lj; jx >= 0; jx--)
                for (jy = lj - jx; jy >= 0; jy--, fb++) {
                        const int jz = lj - jx - jy;
                        double *e = Et + (size_t)(fa * nfb + fb) * nh;
                        const double *ex = Ex + ix * s1 + jx * s2, *ey = Ey + iy * s1 + jy * s2;
                        const double *ez = Ez + iz * s1 + jz * s2;
                        for (n = 0; n < nh; n++) e[n] = ex[rj_t[n]] * ey[rj_u[n]] * ez[rj_v[n]];
                }
        }
}

/* the bra pairs a >= b with qcond[a, b] * qother >= tol */
static RJPair *make_pairs(int *npair_out, const double *qcond, double qother, double tol,
                          int *atm, int *bas, int nbas, double *env)
{
        size_t cap = 1024, n = 0;
        RJPair *pairs = malloc(sizeof(RJPair) * cap);
        int a, b;
        for (a = 0; a < nbas; a++) {
                const double ai = env[bas[a * BAS_SLOTS + PTR_EXP]];
                const double *Ra = env + atm[bas[a * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
                for (b = 0; b <= a; b++) {
                        if (qcond[(size_t)a * nbas + b] * qother < tol) continue;
                        if (n == cap) {
                                cap *= 2;
                                pairs = realloc(pairs, sizeof(RJPair) * cap);
                        }
                        RJPair *pp = pairs + n++;
                        const double aj = env[bas[b * BAS_SLOTS + PTR_EXP]];
                        const double *Rb = env + atm[bas[b * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
                        pp->ish = a;
                        pp->jsh = b;
                        pp->lp = bas[a * BAS_SLOTS + ANG_OF] + bas[b * BAS_SLOTS + ANG_OF];
                        pp->p = ai + aj;
                        int x;
                        for (x = 0; x < 3; x++) pp->P[x] = (ai * Ra[x] + aj * Rb[x]) / pp->p;
                }
        }
        *npair_out = (int)n;
        return pairs;
}

/* auxiliary data: naux primitives sorted by centre (cen_off[c] .. cen_off[c + 1]), within a centre
 * by exponent descending; aexp, al, hoff[k] (offset of the degree-l components of primitive k);
 * cxyz (ncen, 3) */
typedef struct {
        int naux, ncen;
        const double *aexp, *cxyz;
        const int *al, *hoff, *cen_off;
        int *lpre;              /* lpre[cen_off[c] + c + k]: max l of the first k primitives of c */
} RJAux;

static void aux_prefix_l(RJAux *ax)
{
        int c, k;
        ax->lpre = malloc(sizeof(int) * (ax->naux + ax->ncen));
        for (c = 0; c < ax->ncen; c++) {
                int *lp = ax->lpre + ax->cen_off[c] + c;
                lp[0] = 0;
                for (k = ax->cen_off[c]; k < ax->cen_off[c + 1]; k++) {
                        const int i = k - ax->cen_off[c];
                        lp[i + 1] = ax->al[k] > lp[i] ? ax->al[k] : lp[i];
                }
        }
}

/* the number of leading primitives of centre c far from a pair of exponent p at distance^2 r2 */
static inline int far_count(const RJAux *ax, int c, double p, double r2, int lp, double *tf_out)
{
        const int k0 = ax->cen_off[c], k1 = ax->cen_off[c + 1];
        const double tf = t_far[lp + ax->lpre[k1 + c]];
        *tf_out = tf;
        /* theta r2 >= tf  <=>  q >= qmin = tf p / (p r2 - tf) (p r2 > tf; else none) */
        const double den = p * r2 - tf;
        if (den <= 0) return 0;
        const double qmin = tf * p / den;
        const double *q = ax->aexp;
        if (q[k1 - 1] >= qmin) return k1 - k0;
        if (q[k0] < qmin) return 0;
        int lo = k0, hi = k1 - 1;       /* q[lo] >= qmin > q[hi] */
        while (hi - lo > 1) {
                const int mid = (lo + hi) / 2;
                if (q[mid] >= qmin) lo = mid; else hi = mid;
        }
        return hi - k0;
}

/* r[n] = pref (-2 theta)^n F_n(theta r2) */
static inline void near_r(double *r, int L, double p, double q, double r2)
{
        const double pq = p * q, s = p + q, theta = pq / s;
        double F[LT + 1];
        boys_fn(F, theta * r2, L);
        double f = PI_FAC / (pq * sqrt(s));
        const double m2t = -2 * theta;
        int n;
        for (n = 0; n <= L; n++) {
                r[n] = f * F[n];
                f *= m2t;
        }
}

/* sqrt((Omega|Omega)) of a Hermite density of degree lp and exponent p */
static double self_bound(const double *rho, int lp, double p)
{
        double r[LT + 1], R0[NH(LT)], v[NH(LP)];
        near_r(r, 2 * lp, p, p, 0.);
        rjr_table[2 * lp](R0, r, 0., 0., 0.);
        /* (Omega|Omega) = sum_ts rho_t (-1)^|s| R_{t+s} rho_s: by degree blocks of s */
        double s = 0;
        int l, k;
        for (l = 0; l <= lp; l++) {
                memset(v, 0, sizeof(double) * NH(lp));
                rjy_table[lp][l](R0, rho + NH(l - 1), v);
                double t = 0;
                for (k = 0; k < NH(lp); k++) t += v[k] * rho[k];
                s += (l % 2 ? -t : t);
        }
        return sqrt(fabs(s));
}

/* the lane batches of one bra pair: near-field primitives by l, far-field centres by the
 * largest l of their far prefix */
typedef struct {
        int n;
        double X[RJ_NL], Y[RJ_NL], Z[RJ_NL], r2[RJ_NL], q[RJ_NL];
        int k[RJ_NL];           /* near: the primitive; far: the bucket / prefix index */
        int c[RJ_NL];           /* the auxiliary centre (gradient) */
} RJBatch;

static inline void batch_pad(RJBatch *b)
{
        int i;
        for (i = b->n; i < RJ_NL; i++) {
                b->X[i] = b->Y[i] = b->Z[i] = 0;
                b->r2[i] = 1.;
                b->q[i] = 1.;
                b->k[i] = -1;
        }
}

/* r[n * NL + lane] = fp (-1)^n (2n-1)!! / R^(2n+1) */
static inline void farv(double *restrict r, int L, double fp, const double *restrict r2)
{
        int l, n;
#pragma omp simd
        for (l = 0; l < RJ_NL; l++) {
                const double ir2 = 1. / r2[l];
                double f = fp * sqrt(ir2);
                for (n = 0; n <= L; n++) {
                        r[n * RJ_NL + l] = f;
                        f *= -(2 * n + 1) * ir2;
                }
        }
}

/* pass 1 flushes: Xl[hoff[k] + s] += (P|k)_s rho; Yl[k * NHQ + s] += the far tensor contracted with rho */
static void flush_near_rho(RJBatch *b, int lp, int lq, double p, const double *rho, const int *hoff, double *Xl,
                           double *R0, double *r, double *tmp)
{
        const int L = lp + lq, nt = (lq + 1) * (lq + 2) / 2;
        batch_pad(b);
        nearv_table[L](r, p, b->q, b->r2);
        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
        rjxv_table[lp][lq](R0, rho, tmp);
        int i, s;
        for (i = 0; i < b->n; i++) {
                double *x = Xl + hoff[b->k[i]];
                for (s = 0; s < nt; s++) x[s] += tmp[s * RJ_NL + i];
        }
        b->n = 0;
}

static void flush_far_rho(RJBatch *b, int lp, int lm, double fp, const double *rho, double *Yl,
                          double *R0, double *r, double *tmp)
{
        const int L = lp + lm;
        batch_pad(b);
        farv(r, L, fp, b->r2);
        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
        int l, i, s;
        for (l = 0; l <= lm; l++) rjxv_table[lp][l](R0, rho, tmp + NH(l - 1) * RJ_NL);
        const int ns = NH(lm);
        for (i = 0; i < b->n; i++) {
                double *y = Yl + (size_t)b->k[i] * NHQ;
                for (s = 0; s < ns; s++) y[s] += tmp[s * RJ_NL + i];
        }
        b->n = 0;
}

/* pass 2 flushes: Vl[t * NL + lane] += sum_s R_{t+s} h_s */
static void flush_near_vj(RJBatch *b, int lp, int lq, double p, const double *h, const int *hoff, double *Vl,
                          double *R0, double *r, double *hl)
{
        const int L = lp + lq, nt = (lq + 1) * (lq + 2) / 2;
        int i, s;
        for (i = 0; i < b->n; i++) {
                const double *hk = h + hoff[b->k[i]];
                for (s = 0; s < nt; s++) hl[s * RJ_NL + i] = hk[s];
        }
        for (i = b->n; i < RJ_NL; i++)
                for (s = 0; s < nt; s++) hl[s * RJ_NL + i] = 0;
        batch_pad(b);
        nearv_table[L](r, p, b->q, b->r2);
        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
        rjyv_table[lp][lq](R0, hl, Vl);
        b->n = 0;
}

static void flush_far_vj(RJBatch *b, int lp, int lm, double fp, const double *M, double *Vl,
                         double *R0, double *r, double *hl)
{
        const int L = lp + lm, ns = NH(lm);
        int i, s, l;
        for (i = 0; i < b->n; i++) {
                const double *m = M + (size_t)b->k[i] * NHQ;
                for (s = 0; s < ns; s++) hl[s * RJ_NL + i] = m[s];
        }
        for (i = b->n; i < RJ_NL; i++)
                for (s = 0; s < ns; s++) hl[s * RJ_NL + i] = 0;
        batch_pad(b);
        farv(r, L, fp, b->r2);
        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
        for (l = 0; l <= lm; l++) rjyv_table[lp][l](R0, hl + NH(l - 1) * RJ_NL, Vl);
        b->n = 0;
}

/*
 * Pass 1: X (hoff[naux]) = the degree-l Hermite projections of the bra density dm
 * (symmetric, decontracted Cartesian basis of bas).  qaux: max sqrt((mu|mu)) of the
 * auxiliary functions (the bra screening).  Returns the number of bra pairs kept.
 */
int SOCVHFrij_direct_rho(double *X, double *dm, double *qcond, double tol, double qaux, double far_eps,
                         int naux, int ncen, double *aexp, int *al, int *hoff, int *cen_off, double *cxyz,
                         int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        rj_init(far_eps);
        RJAux ax = {naux, ncen, aexp, cxyz, al, hoff, cen_off, NULL};
        aux_prefix_l(&ax);
        const size_t nao = ao_loc[nbas];
        const size_t nx = hoff[naux];
        int npair;
        double dmax = 0;
        {
                size_t k;
                for (k = 0; k < nao * nao; k++) if (fabs(dm[k]) > dmax) dmax = fabs(dm[k]);
        }
        RJPair *pairs = make_pairs(&npair, qcond, qaux * dmax, tol, atm, bas, nbas, env);
        const int nthreads = omp_get_max_threads();
        /* far buckets: Yb[(cen_off[c] + c + k) * NHQ + s], k = prefix length 1..n_c */
        const size_t nyb = (size_t)(naux + ncen) * NHQ;
        double **xbufs = malloc(sizeof(double *) * nthreads);
        double **ybufs = malloc(sizeof(double *) * nthreads);
        int nkept = 0;
        long cnt_near = 0, cnt_far = 0;
#pragma omp parallel num_threads(nthreads) reduction(+:nkept, cnt_near, cnt_far)
{
        const int tid = omp_get_thread_num();
        double *Xl = xbufs[tid] = calloc(nx, sizeof(double));
        double *Yl = ybufs[tid] = calloc(nyb, sizeof(double));
        const int ncmax = (LP / 2 + 1) * (LP / 2 + 2) / 2;
        double *Et = malloc(sizeof(double) * ncmax * ncmax * NH(LP));
        double wk[3 * (LP + 1) * (LP + 1) * (LP + 1)];
        double rho[NH(LP)];
        double *R0 = malloc(sizeof(double) * (NH(LT) + (LT + 1) + NHQ) * RJ_NL);
        double *r = R0 + NH(LT) * RJ_NL, *tmp = r + (LT + 1) * RJ_NL;
        RJBatch bn[LQ + 1], bf[LQ + 1];
        int ip;
#pragma omp for schedule(dynamic, 32)
        for (ip = 0; ip < npair; ip++) {
                const RJPair *pp = pairs + ip;
                const int lp = pp->lp, nh = NH(lp);
                const int ish = pp->ish, jsh = pp->jsh;
                const int li = bas[ish * BAS_SLOTS + ANG_OF], lj = bas[jsh * BAS_SLOTS + ANG_OF];
                const int nfa = (li + 1) * (li + 2) / 2, nfb = (lj + 1) * (lj + 2) / 2;
                const double w = ish == jsh ? 1. : 2.;
                pair_Et(Et, pp, atm, bas, env, wk);
                const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
                int fa, fb, k, c, l;
                for (k = 0; k < nh; k++) rho[k] = 0;
                for (fa = 0; fa < nfa; fa++)
                        for (fb = 0; fb < nfb; fb++) {
                                const double v = w * dm[(a0 + fa) * nao + b0 + fb];
                                const double *e = Et + (size_t)(fa * nfb + fb) * nh;
                                for (k = 0; k < nh; k++) rho[k] += v * e[k];
                        }
                if (self_bound(rho, lp, pp->p) * qaux < tol) continue;
                nkept++;
                const double p = pp->p, fp = PI32 / (p * sqrt(p));
                for (l = 0; l <= LQ; l++) bn[l].n = bf[l].n = 0;
                for (c = 0; c < ncen; c++) {
                        const double X0 = pp->P[0] - cxyz[3 * c], Y0 = pp->P[1] - cxyz[3 * c + 1];
                        const double Z0 = pp->P[2] - cxyz[3 * c + 2];
                        const double r2 = X0 * X0 + Y0 * Y0 + Z0 * Z0;
                        double tf;
                        const int nfar = far_count(&ax, c, p, r2, lp, &tf);
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        cnt_near += k1 - k0 - nfar;
                        if (nfar > 0) {
                                cnt_far++;
                                const int lm = ax.lpre[k0 + c + nfar];
                                RJBatch *b = bf + lm;
                                const int i = b->n++;
                                b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                b->k[i] = k0 + c + nfar;
                                if (b->n == RJ_NL) flush_far_rho(b, lp, lm, fp, rho, Yl, R0, r, tmp);
                        }
                        for (k = k0 + nfar; k < k1; k++) {
                                const int lq = al[k];
                                RJBatch *b = bn + lq;
                                const int i = b->n++;
                                b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                b->q[i] = aexp[k];
                                b->k[i] = k;
                                if (b->n == RJ_NL) flush_near_rho(b, lp, lq, p, rho, hoff, Xl, R0, r, tmp);
                        }
                }
                for (l = 0; l <= LQ; l++) {
                        if (bn[l].n) flush_near_rho(bn + l, lp, l, p, rho, hoff, Xl, R0, r, tmp);
                        if (bf[l].n) flush_far_rho(bf + l, lp, l, fp, rho, Yl, R0, r, tmp);
                }
        }
        free(Et);
        free(R0);
#pragma omp barrier
        /* sum the thread buffers, each thread a slice */
        {
                size_t chunk = (nx + nthreads - 1) / nthreads, s0 = chunk * tid < nx ? chunk * tid : nx;
                size_t s1 = s0 + chunk < nx ? s0 + chunk : nx, k;
                int t;
                for (k = s0; k < s1; k++) {
                        double s = 0;
                        for (t = 0; t < nthreads; t++) s += xbufs[t][k];
                        X[k] = s;
                }
                chunk = (nyb + nthreads - 1) / nthreads;
                s0 = chunk * tid < nyb ? chunk * tid : nyb;
                s1 = s0 + chunk < nyb ? s0 + chunk : nyb;
                for (k = s0; k < s1; k++) {
                        double s = 0;
                        for (t = 1; t < nthreads; t++) s += ybufs[t][k];
                        ybufs[0][k] += s;
                }
        }
#pragma omp barrier
#pragma omp single
{
        /* far buckets -> primitives: X^i += (pi/q_i)^(3/2) sum_{k > i} Yb[k] (degree l_i block) */
        const double *Y0 = ybufs[0];
        int c;
        for (c = 0; c < ncen; c++) {
                const int k0 = cen_off[c], k1 = cen_off[c + 1], n = k1 - k0;
                double acc[NHQ];
                int i, s;
                memset(acc, 0, sizeof(acc));
                for (i = n - 1; i >= 0; i--) {
                        const double *y = Y0 + (size_t)(k0 + c + i + 1) * NHQ;
                        for (s = 0; s < NHQ; s++) acc[s] += y[s];
                        const int k = k0 + i, l = al[k];
                        const double q = aexp[k], fq = PI32 / (q * sqrt(q));
                        const int nt = (l + 1) * (l + 2) / 2;
                        for (s = 0; s < nt; s++) X[hoff[k] + s] += fq * acc[NH(l - 1) + s];
                }
        }
}
        free(Xl);
        free(Yl);
}
        free(xbufs);
        free(ybufs);
        SOCVHFrij_direct_stats[0] = cnt_near;
        SOCVHFrij_direct_stats[1] = cnt_far;
        free(pairs);
        free(ax.lpre);
        return nkept;
}

/*
 * Pass 2: vj (nao, nao) = sum_ab J of the auxiliary Hermite coefficients h (hoff[naux]),
 * J_ab = sum_Q (ab|Q) h_Q, in the decontracted Cartesian basis.  hmax: the bound of the
 * auxiliary side (max_mu sqrt((mu|mu)) |d_mu|) for the bra screening.  Returns the pairs kept.
 */
int SOCVHFrij_direct_vj(double *vj, double *h, double *qcond, double tol, double hmax, double far_eps,
                        int naux, int ncen, double *aexp, int *al, int *hoff, int *cen_off, double *cxyz,
                        int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        rj_init(far_eps);
        RJAux ax = {naux, ncen, aexp, cxyz, al, hoff, cen_off, NULL};
        aux_prefix_l(&ax);
        const size_t nao = ao_loc[nbas];
        int npair;
        RJPair *pairs = make_pairs(&npair, qcond, hmax, tol, atm, bas, nbas, env);
        /* prefix multipoles M[(cen_off[c] + c + k) * NHQ + s] = sum_{i < k} (pi/q_i)^(3/2) h^i */
        double *M = calloc((size_t)(naux + ncen) * NHQ, sizeof(double));
        {
                int c;
                for (c = 0; c < ncen; c++) {
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        int k, s;
                        for (k = k0; k < k1; k++) {
                                const double *m0 = M + (size_t)(k + c) * NHQ;
                                double *m1 = M + (size_t)(k + c + 1) * NHQ;
                                for (s = 0; s < NHQ; s++) m1[s] = m0[s];
                                const int l = al[k], nt = (l + 1) * (l + 2) / 2;
                                const double q = aexp[k], fq = PI32 / (q * sqrt(q));
                                for (s = 0; s < nt; s++) m1[NH(l - 1) + s] += fq * h[hoff[k] + s];
                        }
                }
        }
        memset(vj, 0, sizeof(double) * nao * nao);
        long cnt_near = 0, cnt_far = 0;
#pragma omp parallel reduction(+:cnt_near, cnt_far)
{
        const int ncmax = (LP / 2 + 1) * (LP / 2 + 2) / 2;
        double *Et = malloc(sizeof(double) * ncmax * ncmax * NH(LP));
        double wk[3 * (LP + 1) * (LP + 1) * (LP + 1)];
        double V[NH(LP)];
        double *R0 = malloc(sizeof(double) * (NH(LT) + (LT + 1) + NHQ + NH(LP)) * RJ_NL);
        double *r = R0 + NH(LT) * RJ_NL, *hl = r + (LT + 1) * RJ_NL, *Vl = hl + NHQ * RJ_NL;
        RJBatch bn[LQ + 1], bf[LQ + 1];
        int ip;
#pragma omp for schedule(dynamic, 32)
        for (ip = 0; ip < npair; ip++) {
                const RJPair *pp = pairs + ip;
                const int lp = pp->lp, nh = NH(lp);
                const double p = pp->p, fp = PI32 / (p * sqrt(p));
                int c, k, l;
                for (k = 0; k < nh * RJ_NL; k++) Vl[k] = 0;
                for (l = 0; l <= LQ; l++) bn[l].n = bf[l].n = 0;
                for (c = 0; c < ncen; c++) {
                        const double X0 = pp->P[0] - cxyz[3 * c], Y0 = pp->P[1] - cxyz[3 * c + 1];
                        const double Z0 = pp->P[2] - cxyz[3 * c + 2];
                        const double r2 = X0 * X0 + Y0 * Y0 + Z0 * Z0;
                        double tf;
                        const int nfar = far_count(&ax, c, p, r2, lp, &tf);
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        cnt_near += k1 - k0 - nfar;
                        if (nfar > 0) {
                                cnt_far++;
                                const int lm = ax.lpre[k0 + c + nfar];
                                RJBatch *b = bf + lm;
                                const int i = b->n++;
                                b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                b->k[i] = k0 + c + nfar;
                                if (b->n == RJ_NL) flush_far_vj(b, lp, lm, fp, M, Vl, R0, r, hl);
                        }
                        for (k = k0 + nfar; k < k1; k++) {
                                const int lq = al[k];
                                RJBatch *b = bn + lq;
                                const int i = b->n++;
                                b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                b->q[i] = aexp[k];
                                b->k[i] = k;
                                if (b->n == RJ_NL) flush_near_vj(b, lp, lq, p, h, hoff, Vl, R0, r, hl);
                        }
                }
                for (l = 0; l <= LQ; l++) {
                        if (bn[l].n) flush_near_vj(bn + l, lp, l, p, h, hoff, Vl, R0, r, hl);
                        if (bf[l].n) flush_far_vj(bf + l, lp, l, fp, M, Vl, R0, r, hl);
                }
                for (k = 0; k < nh; k++) {
                        double s = 0;
                        int i;
                        for (i = 0; i < RJ_NL; i++) s += Vl[k * RJ_NL + i];
                        V[k] = s;
                }
                const int ish = pp->ish, jsh = pp->jsh;
                const int li = bas[ish * BAS_SLOTS + ANG_OF], lj = bas[jsh * BAS_SLOTS + ANG_OF];
                const int nfa = (li + 1) * (li + 2) / 2, nfb = (lj + 1) * (lj + 2) / 2;
                pair_Et(Et, pp, atm, bas, env, wk);
                const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
                int fa, fb;
                for (fa = 0; fa < nfa; fa++)
                        for (fb = 0; fb < nfb; fb++) {
                                const double *e = Et + (size_t)(fa * nfb + fb) * nh;
                                double s = 0;
                                for (k = 0; k < nh; k++) s += e[k] * V[k];
                                vj[(a0 + fa) * nao + b0 + fb] = s;
                                vj[(b0 + fb) * nao + a0 + fa] = s;
                        }
        }
        free(Et);
        free(R0);
}
        SOCVHFrij_direct_stats[2] = cnt_near;
        SOCVHFrij_direct_stats[3] = cnt_far;
        free(M);
        free(pairs);
        free(ax.lpre);
        return npair;
}

/* Hermite density rho (NH(lp)) of a pair and its derivative densities d/dA_x (dA[x * NH(lp + 1)])
 * and d/dB_x, from d phi_a / dA_x = 2 alpha_a phi_{a+1x} - a_x phi_{a-1x} (md_j.c pair_densities,
 * in the order of rij_direct_gen.c) */
static void pair_rho_der(double *rho, double *dA, double *dB, const RJPair *pp, const double *dm, size_t nao,
                         const int *ao_loc, int *atm, int *bas, double *env, double *wk)
{
        const int ish = pp->ish, jsh = pp->jsh;
        const int li = bas[ish * BAS_SLOTS + ANG_OF], lj = bas[jsh * BAS_SLOTS + ANG_OF], lij = li + lj;
        const double *Ra = env + atm[bas[ish * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double *Rb = env + atm[bas[jsh * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double ai = env[bas[ish * BAS_SLOTS + PTR_EXP]], aj = env[bas[jsh * BAS_SLOTS + PTR_EXP]];
        const double w = (ish == jsh ? 1. : 2.) * env[bas[ish * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(li)
                       * env[bas[jsh * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(lj);
        const int Li = li + 1, Lj = lj + 1, s2 = Li + Lj + 1, s1 = (Lj + 1) * s2, sz = (Li + 1) * s1;
        double *Ex = wk, *Ey = Ex + sz, *Ez = Ey + sz;
        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
        hermite_E(Ex, Li, Lj, ai, aj, Ra[0], Rb[0], 1.);
        hermite_E(Ey, Li, Lj, ai, aj, Ra[1], Rb[1], 1.);
        hermite_E(Ez, Li, Lj, ai, aj, Ra[2], Rb[2], exp(-ai * aj / (ai + aj) * r2));
        const int nh0 = NH(lij), nh1 = NH(lij + 1);
        memset(rho, 0, sizeof(double) * nh0);
        memset(dA, 0, sizeof(double) * 3 * nh1);
        memset(dB, 0, sizeof(double) * 3 * nh1);
        const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
        int ix, iy, jx, jy, n, x, fa = 0;
        for (ix = li; ix >= 0; ix--)
        for (iy = li - ix; iy >= 0; iy--, fa++) {
                const int iz = li - ix - iy;
                int fb = 0;
                for (jx = lj; jx >= 0; jx--)
                for (jy = lj - jx; jy >= 0; jy--, fb++) {
                        const int jz = lj - jx - jy;
                        const double d = w * dm[(a0 + fa) * nao + b0 + fb];
                        if (d == 0) continue;
                        const int ia[3] = {ix, iy, iz}, ja[3] = {jx, jy, jz};
                        const double *Es[3] = {Ex, Ey, Ez};
                        double e[3][3][2 * LP + 4];
                        int k, t;
                        for (k = 0; k < 3; k++) {
                                const double *E = Es[k];
                                const int i = ia[k], j = ja[k];
                                for (t = 0; t < s2; t++) {
                                        e[k][0][t] = E[i * s1 + j * s2 + t];
                                        e[k][1][t] = 2 * ai * E[(i + 1) * s1 + j * s2 + t]
                                                   - (i ? i * E[(i - 1) * s1 + j * s2 + t] : 0);
                                        e[k][2][t] = 2 * aj * E[i * s1 + (j + 1) * s2 + t]
                                                   - (j ? j * E[i * s1 + (j - 1) * s2 + t] : 0);
                                }
                        }
                        for (n = 0; n < nh1; n++) {
                                const int t = rj_t[n], u = rj_u[n], v = rj_v[n];
                                const double ex = e[0][0][t], ey = e[1][0][u], ez = e[2][0][v];
                                if (n < nh0) rho[n] += d * ex * ey * ez;
                                for (x = 0; x < 3; x++) {
                                        const double fA = (x == 0 ? e[0][1][t] : ex) * (x == 1 ? e[1][1][u] : ey)
                                                        * (x == 2 ? e[2][1][v] : ez);
                                        const double fB = (x == 0 ? e[0][2][t] : ex) * (x == 1 ? e[1][2][u] : ey)
                                                        * (x == 2 ? e[2][2][v] : ez);
                                        dA[x * nh1 + n] += d * fA;
                                        dB[x * nh1 + n] += d * fB;
                                }
                        }
                }
        }
}

/* rj_up[x][n]: the index of the Hermite term n raised by one along x */
static int rj_up[3][NH(LP)];
static int rj_up_ready = 0;

static void up_tables(void)
{
#pragma omp critical(rij_direct_up)
{
        if (!rj_up_ready) {
                int n, m, x;
                for (n = 0; n < NH(LP - 1); n++)
                        for (x = 0; x < 3; x++) {
                                const int t = rj_t[n] + (x == 0), u = rj_u[n] + (x == 1), v = rj_v[n] + (x == 2);
                                for (m = 0; m < NH(LP); m++)
                                        if (rj_t[m] == t && rj_u[m] == u && rj_v[m] == v) break;
                                rj_up[x][n] = m;
                        }
                rj_up_ready = 1;
        }
}
}

/* the potential of one batch (degree lp + 1) into Vl and the forces on the auxiliary centres,
 * g_C -= sum_t rho_t V^{PC}_{t+x} (translational invariance of each pair-centre interaction) */
static void grad_flush_tail(RJBatch *b, int lp, const double *rho, const double *tmp, double *Vl, double *gc)
{
        const int nh1 = NH(lp + 1), nh0 = NH(lp);
        int i, n;
        for (i = 0; i < b->n; i++) {
                double f0 = 0, f1 = 0, f2 = 0;
                for (n = 0; n < nh0; n++) {
                        f0 += rho[n] * tmp[rj_up[0][n] * RJ_NL + i];
                        f1 += rho[n] * tmp[rj_up[1][n] * RJ_NL + i];
                        f2 += rho[n] * tmp[rj_up[2][n] * RJ_NL + i];
                }
                double *g = gc + 3 * b->c[i];
                g[0] -= f0;
                g[1] -= f1;
                g[2] -= f2;
        }
        for (n = 0; n < nh1 * RJ_NL; n++) Vl[n] += tmp[n];
        b->n = 0;
}

static void flush_near_grad(RJBatch *b, int lp, int lq, double p, const double *rho, const double *h,
                            const int *hoff, double *Vl, double *gc, double *R0, double *r, double *hl, double *tmp)
{
        const int L = lp + 1 + lq, nt = (lq + 1) * (lq + 2) / 2;
        int i, s;
        for (i = 0; i < b->n; i++) {
                const double *hk = h + hoff[b->k[i]];
                for (s = 0; s < nt; s++) hl[s * RJ_NL + i] = hk[s];
        }
        for (i = b->n; i < RJ_NL; i++)
                for (s = 0; s < nt; s++) hl[s * RJ_NL + i] = 0;
        batch_pad(b);
        nearv_table[L](r, p, b->q, b->r2);
        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
        memset(tmp, 0, sizeof(double) * NH(lp + 1) * RJ_NL);
        rjyv_table[lp + 1][lq](R0, hl, tmp);
        grad_flush_tail(b, lp, rho, tmp, Vl, gc);
}

static void flush_far_grad(RJBatch *b, int lp, int lm, double fp, const double *rho, const double *M,
                           double *Vl, double *gc, double *R0, double *r, double *hl, double *tmp)
{
        const int L = lp + 1 + lm, ns = NH(lm);
        int i, s, l;
        for (i = 0; i < b->n; i++) {
                const double *m = M + (size_t)b->k[i] * NHQ;
                for (s = 0; s < ns; s++) hl[s * RJ_NL + i] = m[s];
        }
        for (i = b->n; i < RJ_NL; i++)
                for (s = 0; s < ns; s++) hl[s * RJ_NL + i] = 0;
        batch_pad(b);
        farv(r, L, fp, b->r2);
        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
        memset(tmp, 0, sizeof(double) * NH(lp + 1) * RJ_NL);
        for (l = 0; l <= lm; l++) rjyv_table[lp + 1][l](R0, hl + NH(l - 1) * RJ_NL, tmp);
        grad_flush_tail(b, lp, rho, tmp, Vl, gc);
}

/*
 * The three-centre part of the RIJ gradient, sum_P h_P sum_ab D_ab d(ab|P)/dR, without the
 * three-centre integrals: the derivative densities of the bra pairs (degree + 1) against the
 * potential of the auxiliary Hermite coefficients h (pass 2 at degree + 1, far field included),
 * the auxiliary centres by translational invariance of each pair-centre interaction.
 * gbra (natm, 3) += the part of the bra atoms (atoms of bas), gaux (ncen, 3) += the auxiliary
 * centres'.  dm: symmetric, decontracted Cartesian basis.  bound: max |D| x max sqrt((mu|mu)) |d_mu|
 * (the bra-pair screening).  Returns the number of pairs.
 */
int SOCVHFrij_direct_grad(double *gbra, double *gaux, double *dm, double *h, double *qcond, double tol,
                          double bound, double far_eps,
                          int naux, int ncen, double *aexp, int *al, int *hoff, int *cen_off, double *cxyz,
                          int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        rj_init(far_eps);
        up_tables();
        RJAux ax = {naux, ncen, aexp, cxyz, al, hoff, cen_off, NULL};
        aux_prefix_l(&ax);
        const size_t nao = ao_loc[nbas];
        int npair;
        RJPair *pairs = make_pairs(&npair, qcond, bound, tol, atm, bas, nbas, env);
        double *M = calloc((size_t)(naux + ncen) * NHQ, sizeof(double));
        {
                int c;
                for (c = 0; c < ncen; c++) {
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        int k, s;
                        for (k = k0; k < k1; k++) {
                                const double *m0 = M + (size_t)(k + c) * NHQ;
                                double *m1 = M + (size_t)(k + c + 1) * NHQ;
                                for (s = 0; s < NHQ; s++) m1[s] = m0[s];
                                const int l = al[k], nt = (l + 1) * (l + 2) / 2;
                                const double q = aexp[k], fq = PI32 / (q * sqrt(q));
                                for (s = 0; s < nt; s++) m1[NH(l - 1) + s] += fq * h[hoff[k] + s];
                        }
                }
        }
#pragma omp parallel
{
        double *gb = calloc((size_t)natm * 3, sizeof(double));
        double *gc = calloc((size_t)ncen * 3, sizeof(double));
        double *wk = malloc(sizeof(double) * 3 * (LP + 2) * (LP + 2) * (2 * LP + 4));
        double rho[NH(LP)], dA[3 * NH(LP)], dB[3 * NH(LP)], V[NH(LP)];
        double *R0 = malloc(sizeof(double) * (NH(LT) + (LT + 1) + NHQ + 2 * NH(LP)) * RJ_NL);
        double *r = R0 + NH(LT) * RJ_NL, *hl = r + (LT + 1) * RJ_NL, *Vl = hl + NHQ * RJ_NL;
        double *tmp = Vl + NH(LP) * RJ_NL;
        RJBatch bn[LQ + 1], bf[LQ + 1];
        int ip;
#pragma omp for schedule(dynamic, 32)
        for (ip = 0; ip < npair; ip++) {
                const RJPair *pp = pairs + ip;
                const int lp = pp->lp, nh1 = NH(lp + 1);
                const double p = pp->p, fp = PI32 / (p * sqrt(p));
                int c, k, l, x;
                pair_rho_der(rho, dA, dB, pp, dm, nao, ao_loc, atm, bas, env, wk);
                for (k = 0; k < nh1 * RJ_NL; k++) Vl[k] = 0;
                for (l = 0; l <= LQ; l++) bn[l].n = bf[l].n = 0;
                for (c = 0; c < ncen; c++) {
                        const double X0 = pp->P[0] - cxyz[3 * c], Y0 = pp->P[1] - cxyz[3 * c + 1];
                        const double Z0 = pp->P[2] - cxyz[3 * c + 2];
                        const double r2 = X0 * X0 + Y0 * Y0 + Z0 * Z0;
                        double tf;
                        const int nfar = far_count(&ax, c, p, r2, lp + 1, &tf);
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        if (nfar > 0) {
                                const int lm = ax.lpre[k0 + c + nfar];
                                RJBatch *b = bf + lm;
                                const int i = b->n++;
                                b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                b->k[i] = k0 + c + nfar;
                                b->c[i] = c;
                                if (b->n == RJ_NL) flush_far_grad(b, lp, lm, fp, rho, M, Vl, gc, R0, r, hl, tmp);
                        }
                        for (k = k0 + nfar; k < k1; k++) {
                                const int lq = al[k];
                                RJBatch *b = bn + lq;
                                const int i = b->n++;
                                b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                b->q[i] = aexp[k];
                                b->k[i] = k;
                                b->c[i] = c;
                                if (b->n == RJ_NL)
                                        flush_near_grad(b, lp, lq, p, rho, h, hoff, Vl, gc, R0, r, hl, tmp);
                        }
                }
                for (l = 0; l <= LQ; l++) {
                        if (bn[l].n) flush_near_grad(bn + l, lp, l, p, rho, h, hoff, Vl, gc, R0, r, hl, tmp);
                        if (bf[l].n) flush_far_grad(bf + l, lp, l, fp, rho, M, Vl, gc, R0, r, hl, tmp);
                }
                for (k = 0; k < nh1; k++) {
                        double s = 0;
                        int i;
                        for (i = 0; i < RJ_NL; i++) s += Vl[k * RJ_NL + i];
                        V[k] = s;
                }
                const int ia = bas[pp->ish * BAS_SLOTS + ATOM_OF], ja = bas[pp->jsh * BAS_SLOTS + ATOM_OF];
                for (x = 0; x < 3; x++) {
                        double sA = 0, sB = 0;
                        for (k = 0; k < nh1; k++) {
                                sA += dA[x * nh1 + k] * V[k];
                                sB += dB[x * nh1 + k] * V[k];
                        }
                        gb[ia * 3 + x] += sA;
                        gb[ja * 3 + x] += sB;
                }
        }
#pragma omp critical(rij_direct_grad_sum)
        {
                int n;
                for (n = 0; n < natm * 3; n++) gbra[n] += gb[n];
                for (n = 0; n < ncen * 3; n++) gaux[n] += gc[n];
        }
        free(gb);
        free(gc);
        free(wk);
        free(R0);
}
        free(M);
        free(pairs);
        free(ax.lpre);
        return npair;
}

/*
 * ------------------------------------------------------------------------------------------------
 * First-order terms of the RIJ Hessian (socutils.hessian.rhf_fast.ri_j_deriv), no three-centre
 * integrals:
 *
 *   Y^{A,x}_Q = sum_ab D_ab d(ab|Q)/dA_x of the bra atoms (the projections of the derivative
 *               densities onto the auxiliary primitives, pass 1 at degree + 1; the auxiliary
 *               centres' part follows by translational invariance, -sum_A Y^{A,x}_Q);
 *   Wrows^x_ab = sum_Q (d_x a b|Q) d_Q (the derivative on the first function), and
 *   Waux^{C,x}_ab = sum_{Q in C} d(ab|Q)/dC_x d_Q of a range of auxiliary centres.
 * ------------------------------------------------------------------------------------------------
 */

/* pass-1 flushes of three densities dx (x * nd + t, degree lp1) into X3 (x * nx + hoff) / Y3 (buckets) */
static void flush_near_y(RJBatch *b, int lp1, int lq, double p, const double *dx, int nd, const int *hoff,
                         double *X3, size_t nx, double *R0, double *r, double *tmp)
{
        const int L = lp1 + lq, nt = (lq + 1) * (lq + 2) / 2;
        batch_pad(b);
        nearv_table[L](r, p, b->q, b->r2);
        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
        int x, i, s;
        for (x = 0; x < 3; x++) {
                rjxv_table[lp1][lq](R0, dx + x * nd, tmp);
                double *X = X3 + x * nx;
                for (i = 0; i < b->n; i++) {
                        double *o = X + hoff[b->k[i]];
                        for (s = 0; s < nt; s++) o[s] += tmp[s * RJ_NL + i];
                }
        }
        b->n = 0;
}

static void flush_far_y(RJBatch *b, int lp1, int lm, double fp, const double *dx, int nd, double *Y3, size_t ny,
                        double *R0, double *r, double *tmp)
{
        const int L = lp1 + lm, ns = NH(lm);
        batch_pad(b);
        farv(r, L, fp, b->r2);
        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
        int x, l, i, s;
        for (x = 0; x < 3; x++) {
                for (l = 0; l <= lm; l++) rjxv_table[lp1][l](R0, dx + x * nd, tmp + NH(l - 1) * RJ_NL);
                double *Y = Y3 + x * ny;
                for (i = 0; i < b->n; i++) {
                        double *o = Y + (size_t)b->k[i] * NHQ;
                        for (s = 0; s < ns; s++) o[s] += tmp[s * RJ_NL + i];
                }
        }
        b->n = 0;
}

/* CSR of the pairs by the atom of their ish (side 0) or jsh (side 1) */
static void pairs_by_atom(int *off, int *list, const RJPair *pairs, int npair, int side, int *bas, int natm)
{
        int i;
        memset(off, 0, sizeof(int) * (natm + 1));
        for (i = 0; i < npair; i++) off[bas[(side ? pairs[i].jsh : pairs[i].ish) * BAS_SLOTS + ATOM_OF] + 1]++;
        for (i = 0; i < natm; i++) off[i + 1] += off[i];
        int *pos = malloc(sizeof(int) * natm);
        memcpy(pos, off, sizeof(int) * natm);
        for (i = 0; i < npair; i++) list[pos[bas[(side ? pairs[i].jsh : pairs[i].ish) * BAS_SLOTS + ATOM_OF]]++] = i;
        free(pos);
}

/*
 * Xh (natm * 3, hoff[naux]): the degree-l Hermite projections of the derivative densities
 * d D / dA_x (bra atoms A), Xh[A,x] = sum_ab D_ab (d(ab)/dA_x | Q) before the map A^T (the caller's
 * rho_aux = A^T X).  Parallel over the atoms, each thread its atom's rows (the far buckets of one
 * atom per thread); each pair twice (once per side).  Returns the number of pairs.
 */
int SOCVHFrij_direct_deriv_y(double *Xh, double *dm, double *qcond, double tol, double qaux, double far_eps,
                             int naux, int ncen, double *aexp, int *al, int *hoff, int *cen_off, double *cxyz,
                             int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        rj_init(far_eps);
        RJAux ax = {naux, ncen, aexp, cxyz, al, hoff, cen_off, NULL};
        aux_prefix_l(&ax);
        const size_t nao = ao_loc[nbas], nx = hoff[naux];
        const size_t ny = (size_t)(naux + ncen) * NHQ;
        double dmax = 0;
        {
                size_t k;
                for (k = 0; k < nao * nao; k++) if (fabs(dm[k]) > dmax) dmax = fabs(dm[k]);
        }
        int npair;
        RJPair *pairs = make_pairs(&npair, qcond, qaux * dmax, tol, atm, bas, nbas, env);
        int *off[2], *list[2];
        int s;
        for (s = 0; s < 2; s++) {
                off[s] = malloc(sizeof(int) * (natm + 1));
                list[s] = malloc(sizeof(int) * (npair + 1));
                pairs_by_atom(off[s], list[s], pairs, npair, s, bas, natm);
        }
        memset(Xh, 0, sizeof(double) * natm * 3 * nx);
#pragma omp parallel
{
        double *Y3 = malloc(sizeof(double) * 3 * ny);
        double *wk = malloc(sizeof(double) * 3 * (LP + 2) * (LP + 2) * (2 * LP + 4));
        double rho[NH(LP)], dA[3 * NH(LP)], dB[3 * NH(LP)];
        double *R0 = malloc(sizeof(double) * (NH(LT) + (LT + 1) + NHQ) * RJ_NL);
        double *r = R0 + NH(LT) * RJ_NL, *tmp = r + (LT + 1) * RJ_NL;
        RJBatch bn[LQ + 1], bf[LQ + 1];
        int A;
#pragma omp for schedule(dynamic, 1)
        for (A = 0; A < natm; A++) {
                if (off[0][A + 1] == off[0][A] && off[1][A + 1] == off[1][A]) continue;
                double *X3 = Xh + (size_t)A * 3 * nx;
                memset(Y3, 0, sizeof(double) * 3 * ny);
                int side, n;
                for (side = 0; side < 2; side++)
                for (n = off[side][A]; n < off[side][A + 1]; n++) {
                        const RJPair *pp = pairs + list[side][n];
                        const int lp = pp->lp, lp1 = lp + 1, nd = NH(lp1);
                        const double p = pp->p, fp = PI32 / (p * sqrt(p));
                        pair_rho_der(rho, dA, dB, pp, dm, nao, ao_loc, atm, bas, env, wk);
                        if (self_bound(rho, lp, p) * qaux < tol * 1e-2) continue;
                        const double *dx = side ? dB : dA;
                        int c, k, l;
                        for (l = 0; l <= LQ; l++) bn[l].n = bf[l].n = 0;
                        for (c = 0; c < ncen; c++) {
                                const double X0 = pp->P[0] - cxyz[3 * c], Y0 = pp->P[1] - cxyz[3 * c + 1];
                                const double Z0 = pp->P[2] - cxyz[3 * c + 2];
                                const double r2 = X0 * X0 + Y0 * Y0 + Z0 * Z0;
                                double tf;
                                const int nfar = far_count(&ax, c, p, r2, lp1, &tf);
                                const int k0 = cen_off[c], k1 = cen_off[c + 1];
                                if (nfar > 0) {
                                        const int lm = ax.lpre[k0 + c + nfar];
                                        RJBatch *b = bf + lm;
                                        const int i = b->n++;
                                        b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                        b->k[i] = k0 + c + nfar;
                                        if (b->n == RJ_NL) flush_far_y(b, lp1, lm, fp, dx, nd, Y3, ny, R0, r, tmp);
                                }
                                for (k = k0 + nfar; k < k1; k++) {
                                        const int lq = al[k];
                                        RJBatch *b = bn + lq;
                                        const int i = b->n++;
                                        b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                        b->q[i] = aexp[k];
                                        b->k[i] = k;
                                        if (b->n == RJ_NL)
                                                flush_near_y(b, lp1, lq, p, dx, nd, hoff, X3, nx, R0, r, tmp);
                                }
                        }
                        for (l = 0; l <= LQ; l++) {
                                if (bn[l].n) flush_near_y(bn + l, lp1, l, p, dx, nd, hoff, X3, nx, R0, r, tmp);
                                if (bf[l].n) flush_far_y(bf + l, lp1, l, fp, dx, nd, Y3, ny, R0, r, tmp);
                        }
                }
                /* far buckets -> primitives (suffix sums, as pass 1) */
                int x, c;
                for (x = 0; x < 3; x++) {
                        const double *Yx = Y3 + x * ny;
                        double *Xx = X3 + x * nx;
                        for (c = 0; c < ncen; c++) {
                                const int k0 = cen_off[c], k1 = cen_off[c + 1], nc = k1 - k0;
                                double acc[NHQ];
                                int i, t;
                                memset(acc, 0, sizeof(acc));
                                for (i = nc - 1; i >= 0; i--) {
                                        const double *y = Yx + (size_t)(k0 + c + i + 1) * NHQ;
                                        for (t = 0; t < NHQ; t++) acc[t] += y[t];
                                        const int k = k0 + i, l = al[k];
                                        const double q = aexp[k], fq = PI32 / (q * sqrt(q));
                                        const int nt = (l + 1) * (l + 2) / 2;
                                        for (t = 0; t < nt; t++) Xx[hoff[k] + t] += fq * acc[NH(l - 1) + t];
                                }
                        }
                }
        }
        free(Y3);
        free(wk);
        free(R0);
}
        for (s = 0; s < 2; s++) {
                free(off[s]);
                free(list[s]);
        }
        free(pairs);
        free(ax.lpre);
        return npair;
}

/* E^{ab}_t (E0) and the first-function derivative coefficients d/dA_x (EA) and d/dB_x (EB) of all
 * Cartesian pairs of a primitive pair: E0[(fa * nfb + fb) * nh1 + n] (degree <= lp, padded to nh1),
 * EA[((fa * nfb + fb) * 3 + x) * nh1 + n] (degree <= lp + 1) */
static void pair_dEt(double *E0, double *EA, double *EB, const RJPair *pp, int *atm, int *bas, double *env,
                     double *wk)
{
        const int ish = pp->ish, jsh = pp->jsh;
        const int li = bas[ish * BAS_SLOTS + ANG_OF], lj = bas[jsh * BAS_SLOTS + ANG_OF], lij = li + lj;
        const double *Ra = env + atm[bas[ish * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double *Rb = env + atm[bas[jsh * BAS_SLOTS + ATOM_OF] * ATM_SLOTS + PTR_COORD];
        const double ai = env[bas[ish * BAS_SLOTS + PTR_EXP]], aj = env[bas[jsh * BAS_SLOTS + PTR_EXP]];
        const double w = env[bas[ish * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(li)
                       * env[bas[jsh * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(lj);
        const int Li = li + 1, Lj = lj + 1, s2 = Li + Lj + 1, s1 = (Lj + 1) * s2, sz = (Li + 1) * s1;
        double *Ex = wk, *Ey = Ex + sz, *Ez = Ey + sz;
        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
        hermite_E(Ex, Li, Lj, ai, aj, Ra[0], Rb[0], 1.);
        hermite_E(Ey, Li, Lj, ai, aj, Ra[1], Rb[1], 1.);
        hermite_E(Ez, Li, Lj, ai, aj, Ra[2], Rb[2], w * exp(-ai * aj / (ai + aj) * r2));
        const int nh0 = NH(lij), nh1 = NH(lij + 1), nfb = (lj + 1) * (lj + 2) / 2;
        int ix, iy, jx, jy, n, x, fa = 0;
        for (ix = li; ix >= 0; ix--)
        for (iy = li - ix; iy >= 0; iy--, fa++) {
                const int iz = li - ix - iy;
                int fb = 0;
                for (jx = lj; jx >= 0; jx--)
                for (jy = lj - jx; jy >= 0; jy--, fb++) {
                        const int jz = lj - jx - jy;
                        const int ia[3] = {ix, iy, iz}, ja[3] = {jx, jy, jz};
                        const double *Es[3] = {Ex, Ey, Ez};
                        double e[3][3][2 * LP + 4];
                        int k, t;
                        for (k = 0; k < 3; k++) {
                                const double *E = Es[k];
                                const int i = ia[k], j = ja[k];
                                for (t = 0; t < s2; t++) {
                                        e[k][0][t] = E[i * s1 + j * s2 + t];
                                        e[k][1][t] = 2 * ai * E[(i + 1) * s1 + j * s2 + t]
                                                   - (i ? i * E[(i - 1) * s1 + j * s2 + t] : 0);
                                        e[k][2][t] = 2 * aj * E[i * s1 + (j + 1) * s2 + t]
                                                   - (j ? j * E[i * s1 + (j - 1) * s2 + t] : 0);
                                }
                        }
                        double *e0 = E0 + (size_t)(fa * nfb + fb) * nh1;
                        double *eA = EA + (size_t)(fa * nfb + fb) * 3 * nh1;
                        double *eB = EB + (size_t)(fa * nfb + fb) * 3 * nh1;
                        for (n = 0; n < nh1; n++) {
                                const int t = rj_t[n], u = rj_u[n], v = rj_v[n];
                                const double ex = e[0][0][t], ey = e[1][0][u], ez = e[2][0][v];
                                e0[n] = n < nh0 ? ex * ey * ez : 0;
                                for (x = 0; x < 3; x++) {
                                        eA[x * nh1 + n] = (x == 0 ? e[0][1][t] : ex) * (x == 1 ? e[1][1][u] : ey)
                                                        * (x == 2 ? e[2][1][v] : ez);
                                        eB[x * nh1 + n] = (x == 0 ? e[0][2][t] : ex) * (x == 1 ? e[1][2][u] : ey)
                                                        * (x == 2 ? e[2][2][v] : ez);
                                }
                        }
                }
        }
}

/* the per-lane potential of a batch (degree lp + 1, tmp) into Vl, and for the lanes of centres in
 * [c0, c1): Waux[c - c0][x][a, b] (and [b, a] of two shells) -= 1/2 sum_t E^ab_t V^{PC}_{t+x} */
static void w_flush_tail(RJBatch *b, const RJPair *pp, int lp, const double *E0, int nfa, int nfb, size_t a0,
                         size_t b0, const double *tmp, double *Vl, double *Waux, int c0, int c1, size_t nao)
{
        const int nh1 = NH(lp + 1), nh0 = NH(lp);
        int i, n, x, fa, fb;
        if (Waux != NULL) {
                const size_t nn = nao * nao;
                for (i = 0; i < b->n; i++) {
                        const int c = b->c[i];
                        if (c < c0 || c >= c1) continue;
                        double *W = Waux + (size_t)(c - c0) * 3 * nn;
                        for (x = 0; x < 3; x++)
                        for (fa = 0; fa < nfa; fa++)
                        for (fb = 0; fb < nfb; fb++) {
                                const double *e = E0 + (size_t)(fa * nfb + fb) * nh1;
                                double s = 0;
                                for (n = 0; n < nh0; n++) s += e[n] * tmp[rj_up[x][n] * RJ_NL + i];
                                /* accumulated over the batches of the pair (near by l, far) */
                                W[x * nn + (a0 + fa) * nao + b0 + fb] -= .5 * s;
                                if (pp->ish != pp->jsh) W[x * nn + (b0 + fb) * nao + a0 + fa] -= .5 * s;
                        }
                }
        }
        for (n = 0; n < nh1 * RJ_NL; n++) Vl[n] += tmp[n];
        b->n = 0;
}

/*
 * Wrows (3, nao, nao) = sum_Q (d_x a b|Q) h_Q (derivative on the first function, every ordered pair),
 * and if Waux != NULL, Waux (c1 - c0, 3, nao, nao) = 1/2 sum_{Q in C} d(ab|Q)/dC_x h_Q of the
 * auxiliary centres c0 <= C < c1 (symmetric; the caller adds W + W^T).  Decontracted Cartesian
 * basis; h the auxiliary Hermite coefficients (A d).  hmax: the bra screening bound.  Every
 * element of the significant pairs is written (the caller zeroes the arrays).
 */
int SOCVHFrij_direct_deriv_w(double *Wrows, double *Waux, int c0, int c1, double *h, double *qcond, double tol,
                             double hmax, double far_eps,
                             int naux, int ncen, double *aexp, int *al, int *hoff, int *cen_off, double *cxyz,
                             int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        rj_init(far_eps);
        up_tables();
        RJAux ax = {naux, ncen, aexp, cxyz, al, hoff, cen_off, NULL};
        aux_prefix_l(&ax);
        const size_t nao = ao_loc[nbas], nn = nao * nao;
        int npair;
        RJPair *pairs = make_pairs(&npair, qcond, hmax, tol, atm, bas, nbas, env);
        double *M = calloc((size_t)(naux + ncen) * NHQ, sizeof(double));
        {
                int c;
                for (c = 0; c < ncen; c++) {
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        int k, s;
                        for (k = k0; k < k1; k++) {
                                const double *m0 = M + (size_t)(k + c) * NHQ;
                                double *m1 = M + (size_t)(k + c + 1) * NHQ;
                                for (s = 0; s < NHQ; s++) m1[s] = m0[s];
                                const int l = al[k], nt = (l + 1) * (l + 2) / 2;
                                const double q = aexp[k], fq = PI32 / (q * sqrt(q));
                                for (s = 0; s < nt; s++) m1[NH(l - 1) + s] += fq * h[hoff[k] + s];
                        }
                }
        }
#pragma omp parallel
{
        const int ncmax = (LP / 2 + 1) * (LP / 2 + 2) / 2;
        double *E0 = malloc(sizeof(double) * ncmax * ncmax * 7 * NH(LP));
        double *EA = E0 + ncmax * ncmax * NH(LP), *EB = EA + ncmax * ncmax * 3 * NH(LP);
        double *wk = malloc(sizeof(double) * 3 * (LP + 2) * (LP + 2) * (2 * LP + 4));
        double V[NH(LP)];
        double *R0 = malloc(sizeof(double) * (NH(LT) + (LT + 1) + NHQ + 2 * NH(LP)) * RJ_NL);
        double *r = R0 + NH(LT) * RJ_NL, *hl = r + (LT + 1) * RJ_NL, *Vl = hl + NHQ * RJ_NL;
        double *tmp = Vl + NH(LP) * RJ_NL;
        RJBatch bn[LQ + 1], bf[LQ + 1];
        int ip;
#pragma omp for schedule(dynamic, 32)
        for (ip = 0; ip < npair; ip++) {
                const RJPair *pp = pairs + ip;
                const int lp = pp->lp, lp1 = lp + 1, nh1 = NH(lp1);
                const int ish = pp->ish, jsh = pp->jsh;
                const int li = bas[ish * BAS_SLOTS + ANG_OF], lj = bas[jsh * BAS_SLOTS + ANG_OF];
                const int nfa = (li + 1) * (li + 2) / 2, nfb = (lj + 1) * (lj + 2) / 2;
                const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
                const double p = pp->p, fp = PI32 / (p * sqrt(p));
                int c, k, l, x, fa, fb, n;
                pair_dEt(E0, EA, EB, pp, atm, bas, env, wk);
                for (k = 0; k < nh1 * RJ_NL; k++) Vl[k] = 0;
                for (l = 0; l <= LQ; l++) bn[l].n = bf[l].n = 0;
                for (c = 0; c < ncen; c++) {
                        const double X0 = pp->P[0] - cxyz[3 * c], Y0 = pp->P[1] - cxyz[3 * c + 1];
                        const double Z0 = pp->P[2] - cxyz[3 * c + 2];
                        const double r2 = X0 * X0 + Y0 * Y0 + Z0 * Z0;
                        double tf;
                        const int nfar = far_count(&ax, c, p, r2, lp1, &tf);
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        if (nfar > 0) {
                                const int lm = ax.lpre[k0 + c + nfar];
                                RJBatch *b = bf + lm;
                                const int i = b->n++;
                                b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                b->k[i] = k0 + c + nfar;
                                b->c[i] = c;
                                if (b->n == RJ_NL) {
                                        const int L = lp1 + lm, ns = NH(lm);
                                        int i2, s2;
                                        for (i2 = 0; i2 < RJ_NL; i2++) {
                                                const double *m = M + (size_t)b->k[i2] * NHQ;
                                                for (s2 = 0; s2 < ns; s2++) hl[s2 * RJ_NL + i2] = m[s2];
                                        }
                                        farv(r, L, fp, b->r2);
                                        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
                                        memset(tmp, 0, sizeof(double) * nh1 * RJ_NL);
                                        int l2;
                                        for (l2 = 0; l2 <= lm; l2++)
                                                rjyv_table[lp1][l2](R0, hl + NH(l2 - 1) * RJ_NL, tmp);
                                        w_flush_tail(b, pp, lp, E0, nfa, nfb, a0, b0, tmp, Vl, Waux, c0, c1, nao);
                                }
                        }
                        for (k = k0 + nfar; k < k1; k++) {
                                const int lq = al[k];
                                RJBatch *b = bn + lq;
                                const int i = b->n++;
                                b->X[i] = X0; b->Y[i] = Y0; b->Z[i] = Z0; b->r2[i] = r2;
                                b->q[i] = aexp[k];
                                b->k[i] = k;
                                b->c[i] = c;
                                if (b->n == RJ_NL) {
                                        const int L = lp1 + lq, nt = (lq + 1) * (lq + 2) / 2;
                                        int i2, s2;
                                        for (i2 = 0; i2 < RJ_NL; i2++) {
                                                const double *hk = h + hoff[b->k[i2]];
                                                for (s2 = 0; s2 < nt; s2++) hl[s2 * RJ_NL + i2] = hk[s2];
                                        }
                                        nearv_table[L](r, p, b->q, b->r2);
                                        rjrv_table[L](R0, r, b->X, b->Y, b->Z);
                                        memset(tmp, 0, sizeof(double) * nh1 * RJ_NL);
                                        rjyv_table[lp1][lq](R0, hl, tmp);
                                        w_flush_tail(b, pp, lp, E0, nfa, nfb, a0, b0, tmp, Vl, Waux, c0, c1, nao);
                                }
                        }
                }
                for (l = 0; l <= LQ; l++) {
                        RJBatch *b = bn + l;
                        if (b->n) {
                                const int L = lp1 + l, nt = (l + 1) * (l + 2) / 2;
                                int i2, s2;
                                for (i2 = 0; i2 < RJ_NL; i2++)
                                        for (s2 = 0; s2 < nt; s2++)
                                                hl[s2 * RJ_NL + i2] = i2 < b->n ? h[hoff[b->k[i2]] + s2] : 0;
                                const int nb = b->n;
                                batch_pad(b);
                                b->n = nb;
                                nearv_table[L](r, p, b->q, b->r2);
                                rjrv_table[L](R0, r, b->X, b->Y, b->Z);
                                memset(tmp, 0, sizeof(double) * nh1 * RJ_NL);
                                rjyv_table[lp1][l](R0, hl, tmp);
                                w_flush_tail(b, pp, lp, E0, nfa, nfb, a0, b0, tmp, Vl, Waux, c0, c1, nao);
                        }
                        b = bf + l;
                        if (b->n) {
                                const int L = lp1 + l, ns = NH(l);
                                int i2, s2;
                                for (i2 = 0; i2 < RJ_NL; i2++)
                                        for (s2 = 0; s2 < ns; s2++)
                                                hl[s2 * RJ_NL + i2] = i2 < b->n ? M[(size_t)b->k[i2] * NHQ + s2] : 0;
                                const int nb = b->n;
                                batch_pad(b);
                                b->n = nb;
                                farv(r, L, fp, b->r2);
                                rjrv_table[L](R0, r, b->X, b->Y, b->Z);
                                memset(tmp, 0, sizeof(double) * nh1 * RJ_NL);
                                int l2;
                                for (l2 = 0; l2 <= l; l2++) rjyv_table[lp1][l2](R0, hl + NH(l2 - 1) * RJ_NL, tmp);
                                w_flush_tail(b, pp, lp, E0, nfa, nfb, a0, b0, tmp, Vl, Waux, c0, c1, nao);
                        }
                }
                for (k = 0; k < nh1; k++) {
                        double s = 0;
                        int i;
                        for (i = 0; i < RJ_NL; i++) s += Vl[k * RJ_NL + i];
                        V[k] = s;
                }
                for (x = 0; x < 3; x++)
                for (fa = 0; fa < nfa; fa++)
                for (fb = 0; fb < nfb; fb++) {
                        const double *eA = EA + ((size_t)(fa * nfb + fb) * 3 + x) * nh1;
                        const double *eB = EB + ((size_t)(fa * nfb + fb) * 3 + x) * nh1;
                        double sA = 0, sB = 0;
                        for (n = 0; n < nh1; n++) {
                                sA += eA[n] * V[n];
                                sB += eB[n] * V[n];
                        }
                        Wrows[x * nn + (a0 + fa) * nao + b0 + fb] = sA;
                        Wrows[x * nn + (b0 + fb) * nao + a0 + fa] = sB;
                }
        }
        free(E0);
        free(wk);
        free(R0);
}
        free(M);
        free(pairs);
        free(ax.lpre);
        return npair;
}

/* pass 2 of nset coefficient vectors: h (nset, nh), M (nset, naux + ncen, NHQ), Vl (nset, NH(lp), NL);

/* rj_add[t][s]: the index of the Hermite term t + s (t of degree <= LP, s of degree <= LQ) */
static int rj_add[NH(LP)][NH(LQ)];
static int rj_add_ready = 0;

static void add_tables(void)
{
#pragma omp critical(rij_direct_add)
{
        if (!rj_add_ready) {
                int t, s, m;
                for (t = 0; t < NH(LP); t++)
                        for (s = 0; s < NH(LQ); s++) {
                                const int a = rj_t[t] + rj_t[s], b = rj_u[t] + rj_u[s], c = rj_v[t] + rj_v[s];
                                for (m = 0; m < RJ_NH; m++)
                                        if (rj_t[m] == a && rj_u[m] == b && rj_v[m] == c) break;
                                rj_add[t][s] = m;
                        }
                rj_add_ready = 1;
        }
}
}

/* Vs[t * nset + d] += sum_lanes sum_s R_{t+s}[lane] H[(row_lane + s') * nset + d], s over degrees
 * l0..l1 of the source (sets contiguous: one axpy of length nset per (lane, t, s)) */
static void contract_sets(const double *R0, int n, int lp, int l0, int l1, const double *H, const size_t *row,
                          int nset, double *Vs)
{
        const int nh0 = NH(lp);
        int i, t, s, d;
        for (i = 0; i < n; i++) {
                for (s = NH(l0 - 1); s < NH(l1); s++) {
                        const double *hs = H + (row[i] + s - NH(l0 - 1)) * nset;
                        for (t = 0; t < nh0; t++) {
                                const double rr = R0[rj_add[t][s] * RJ_NL + i];
                                double *v = Vs + t * nset;
#pragma omp simd
                                for (d = 0; d < nset; d++) v[d] += rr * hs[d];
                        }
                }
        }
}

static void flush_far_m(RJBatch *b, int lp, int l, double fp, const double *MT, int nset, double *Vs,
                        double *R0, double *r)
{
        size_t row[RJ_NL];
        const int n = b->n;
        int i;
        batch_pad(b);
        farv(r, lp + l, fp, b->r2);
        rjrv_table[lp + l](R0, r, b->X, b->Y, b->Z);
        for (i = 0; i < n; i++) row[i] = (size_t)b->k[i] * NHQ;
        contract_sets(R0, n, lp, 0, l, MT, row, nset, Vs);
        b->n = 0;
}

static void flush_near_m(RJBatch *b, int lp, int l, double p, const double *HT, const int *hoff, int nset,
                         double *Vs, double *R0, double *r)
{
        size_t row[RJ_NL];
        const int n = b->n;
        int i;
        batch_pad(b);
        nearv_table[lp + l](r, p, b->q, b->r2);
        rjrv_table[lp + l](R0, r, b->X, b->Y, b->Z);
        for (i = 0; i < n; i++) row[i] = hoff[b->k[i]];
        contract_sets(R0, n, lp, l, l, HT, row, nset, Vs);
        b->n = 0;
}

/*
 * vj (nset, nao, nao) of nset auxiliary coefficient vectors h (nset, hoff[naux]) in one pass (pass 2
 * with the R tensors shared, the contraction vectorised over the sets), decontracted Cartesian basis.
 * hmax: the largest bound of the sets.
 */
int SOCVHFrij_direct_vj_multi(double *vj, double *h, int nset, double *qcond, double tol, double hmax,
                              double far_eps,
                              int naux, int ncen, double *aexp, int *al, int *hoff, int *cen_off, double *cxyz,
                              int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        rj_init(far_eps);
        add_tables();
        RJAux ax = {naux, ncen, aexp, cxyz, al, hoff, cen_off, NULL};
        aux_prefix_l(&ax);
        const size_t nao = ao_loc[nbas], nn = nao * nao, nh = hoff[naux];
        const size_t nm = (size_t)(naux + ncen) * NHQ;
        int npair;
        RJPair *pairs = make_pairs(&npair, qcond, hmax, tol, atm, bas, nbas, env);
        /* sets innermost: HT[(hoff[k] + s) * nset + d], MT[(prefix * NHQ + s) * nset + d] */
        double *HT = malloc(sizeof(double) * nh * nset);
        double *MT = calloc(nm * nset, sizeof(double));
        {
                size_t k;
                int d, c;
                for (k = 0; k < nh; k++)
                        for (d = 0; d < nset; d++) HT[k * nset + d] = h[d * nh + k];
                for (c = 0; c < ncen; c++) {
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        int kk, s;
                        for (kk = k0; kk < k1; kk++) {
                                const double *m0 = MT + (size_t)(kk + c) * NHQ * nset;
                                double *m1 = MT + (size_t)(kk + c + 1) * NHQ * nset;
                                memcpy(m1, m0, sizeof(double) * NHQ * nset);
                                const int l = al[kk], nt = (l + 1) * (l + 2) / 2;
                                const double q = aexp[kk], fq = PI32 / (q * sqrt(q));
                                for (s = 0; s < nt; s++)
                                        for (d = 0; d < nset; d++)
                                                m1[(NH(l - 1) + s) * nset + d] += fq * HT[(hoff[kk] + s) * nset + d];
                        }
                }
        }
        memset(vj, 0, sizeof(double) * nn * nset);
#pragma omp parallel
{
        const int ncmax = (LP / 2 + 1) * (LP / 2 + 2) / 2;
        double *Et = malloc(sizeof(double) * ncmax * ncmax * NH(LP));
        double wk[3 * (LP + 1) * (LP + 1) * (LP + 1)];
        double *R0 = malloc(sizeof(double) * (NH(LT) + (LT + 1)) * RJ_NL);
        double *r = R0 + NH(LT) * RJ_NL;
        double *Vs = malloc(sizeof(double) * NH(LP) * nset);
        RJBatch bn[LQ + 1], bf[LQ + 1];
        int ip;
#pragma omp for schedule(dynamic, 8)
        for (ip = 0; ip < npair; ip++) {
                const RJPair *pp = pairs + ip;
                const int lp = pp->lp, nh0 = NH(lp);
                const double p = pp->p, fp = PI32 / (p * sqrt(p));
                int c, k, l, d;
                memset(Vs, 0, sizeof(double) * nh0 * nset);
                for (l = 0; l <= LQ; l++) bn[l].n = bf[l].n = 0;
                for (c = 0; c < ncen; c++) {
                        const double X0 = pp->P[0] - cxyz[3 * c], Y0 = pp->P[1] - cxyz[3 * c + 1];
                        const double Z0 = pp->P[2] - cxyz[3 * c + 2];
                        const double r2 = X0 * X0 + Y0 * Y0 + Z0 * Z0;
                        double tf;
                        const int nfar = far_count(&ax, c, p, r2, lp, &tf);
                        const int k0 = cen_off[c], k1 = cen_off[c + 1];
                        if (nfar > 0) {
                                const int lm = ax.lpre[k0 + c + nfar];
                                RJBatch *b = bf + lm;
                                const int j = b->n++;
                                b->X[j] = X0; b->Y[j] = Y0; b->Z[j] = Z0; b->r2[j] = r2;
                                b->k[j] = k0 + c + nfar;
                                if (b->n == RJ_NL) flush_far_m(b, lp, lm, fp, MT, nset, Vs, R0, r);
                        }
                        for (k = k0 + nfar; k < k1; k++) {
                                const int lq = al[k];
                                RJBatch *b = bn + lq;
                                const int j = b->n++;
                                b->X[j] = X0; b->Y[j] = Y0; b->Z[j] = Z0; b->r2[j] = r2;
                                b->q[j] = aexp[k];
                                b->k[j] = k;
                                if (b->n == RJ_NL) flush_near_m(b, lp, lq, p, HT, hoff, nset, Vs, R0, r);
                        }
                }
                for (l = 0; l <= LQ; l++) {
                        if (bf[l].n) flush_far_m(bf + l, lp, l, fp, MT, nset, Vs, R0, r);
                        if (bn[l].n) flush_near_m(bn + l, lp, l, p, HT, hoff, nset, Vs, R0, r);
                }
                const int ish = pp->ish, jsh = pp->jsh;
                const int li = bas[ish * BAS_SLOTS + ANG_OF], lj = bas[jsh * BAS_SLOTS + ANG_OF];
                const int nfa = (li + 1) * (li + 2) / 2, nfb = (lj + 1) * (lj + 2) / 2;
                pair_Et(Et, pp, atm, bas, env, wk);
                const size_t a0 = ao_loc[ish], b0 = ao_loc[jsh];
                int fa, fb;
                for (fa = 0; fa < nfa; fa++)
                        for (fb = 0; fb < nfb; fb++) {
                                const double *e = Et + (size_t)(fa * nfb + fb) * nh0;
                                for (d = 0; d < nset; d++) {
                                        double s = 0;
                                        for (k = 0; k < nh0; k++) s += e[k] * Vs[k * nset + d];
                                        vj[d * nn + (a0 + fa) * nao + b0 + fb] = s;
                                        vj[d * nn + (b0 + fb) * nao + a0 + fa] = s;
                                }
                        }
        }
        free(Et);
        free(R0);
        free(Vs);
}
        free(HT);
        free(MT);
        free(pairs);
        free(ax.lpre);
        return npair;
}
