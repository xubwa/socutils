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

#define LP 4                    /* bra pair degree (d d) */
#define LQ 4                    /* auxiliary angular momentum */
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
