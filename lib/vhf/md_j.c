/*
 * McMurchie-Davidson J engine (Coulomb energy and its nuclear gradient) on a
 * decontracted Cartesian basis, one primitive per shell (pyscf
 * mol.decontract_basis(to_cart=True), the density transformed to it).  The
 * density is contracted into Hermite form once per shell pair P = (a >= b),
 *
 *     rho^P_tuv = w_P sum_{ab} D_ab E^{ab}_t E^{ab}_u E^{ab}_v        (w_P = 2 for a != b),
 *
 * and, for the gradient, the derivative densities of d phi_a / dA_x =
 * 2 alpha_a phi_{a+1x} - a_x phi_{a-1x} (and of b), one Hermite order higher.
 * With T_PQ{t,s} = 2 pi^(5/2) / (p q sqrt(p + q)) (-1)^|s| R_{t+s}(pq/(p+q), P - Q)
 * (symmetric, T_QP = T_PQ^T),
 *
 *     E_J = 1/2 sum_PQ rho^P T_PQ rho^Q,    V^P = sum_Q T_PQ rho^Q,
 *     dE_J / dR = sum_P <d rho^P / dR, V^P>,
 *
 * one pass over the pairs of pairs P <= Q (in the screening order) for V of
 * both.  No Cartesian integrals are formed (gpu4pyscf's md_j does the same
 * for the J matrix on GPUs; this is the CPU gradient).
 *
 * Screening: |(Omega_P|Omega_Q)| <= |Omega_P| |Omega_Q| (Schwarz for the
 * density-weighted charge distributions Omega_P = sum D_ab phi_a phi_b and
 * their derivatives), the norms from one diagonal pass; pairs sorted by
 * b_P = max(|Omega_P|, |d Omega_P|), so the inner loop stops at
 * 2 b_P b_Q < tol.
 *
 * Normalisation as libcint's int2e_cart (radial normalisation in env,
 * CINTcommon_fac_sp for s and p).  Boys functions of our own (boys_fn):
 * libcint's gamma_inc_like loses all accuracy for small m at tiny arguments
 * (m = 1, t = 1e-30: F_1 wrong by 100%), which pairs of coincident pair
 * centres in symmetric molecules hit.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "cint.h"

/* f[n] = F_n(t), n = 0..m, to machine precision: for t < m + 10 the series of
 * F_m, F_m(t) = e^-t sum_k (2t)^k / ((2m+1)(2m+3)...(2m+2k+1)), and downward
 * recursion; otherwise F_0 from erf and upward recursion (stable for t > m).
 * Generates the table of boys_fn. */
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
                f[0] = .886226925452758014 / st * erf(st);          /* sqrt(pi) / 2 */
                const double h = .5 / t;
                for (n = 0; n < m; n++) f[n + 1] = h * ((2 * n + 1) * f[n] - et);
        }
}

#define LMAX 4
#define DOP (2 * LMAX + 1)              /* largest degree of a (derivative) pair density */
#define DMAX (2 * DOP)                  /* largest degree of R */
#define NH(d) (((d) + 1) * ((d) + 2) * ((d) + 3) / 6)
#define NHOP NH(DOP)
#define NHMAX NH(DMAX)

/* Hermite index tables, ordered by degree (an array truncated at degree d is a prefix) */
static int h_ready = 0;
static int h_idx[DMAX + 1][DMAX + 1][DMAX + 1];
static int h_t[NHMAX], h_u[NHMAX], h_v[NHMAX], h_deg[NHMAX];
static int h_dir[NHMAX], h_m1[NHMAX], h_m2[NHMAX], h_c[NHMAX];
static int h_add[NHOP * NHOP];
static double h_sgn[NHMAX];

/* Boys functions from a table: F_m(t) = sum_{j<=6} F_{m+j}(t_k) (t_k - t)^j / j!
 * on the grid t_k = k / 20 (|t_k - t| <= 1/40, truncation ~1e-15), downward
 * recursion to F_0; t >= 40: F_0 = sqrt(pi / t) / 2 and upward recursion */
#define BOYS_TMAX 40.
#define BOYS_NT 801
#define BOYS_MT (DMAX + 8)
static double boys_tab[BOYS_NT][BOYS_MT];
static double boys_inv[BOYS_MT];

static void boys_fn(double *f, double t, int m)
{
        int n;
        const double et = exp(-t);
        if (t < BOYS_TMAX) {
                const int k = (int)(t * 20 + .5);
                const double dl = k * .05 - t;
                const double *b = boys_tab[k] + m;
                f[m] = b[0] + dl * (b[1] + dl * .5 * (b[2] + dl * (1. / 3) * (b[3] + dl * .25 * (b[4]
                     + dl * .2 * (b[5] + dl * (1. / 6) * b[6])))));
                for (n = m; n > 0; n--) f[n - 1] = (2 * t * f[n] + et) * boys_inv[n];
        } else {
                f[0] = .886226925452758014 / sqrt(t);
                const double h = .5 / t;
                for (n = 0; n < m; n++) f[n + 1] = h * ((2 * n + 1) * f[n] - et);
        }
}

#include "md_j_gen.c"

static void hermite_tables(void)
{
        int d, t, u, v, n = 0, a, b;
        for (d = 0; d <= DMAX; d++)
                for (t = 0; t <= d; t++)
                        for (u = 0; u <= d - t; u++) {
                                v = d - t - u;
                                h_idx[t][u][v] = n;
                                h_t[n] = t; h_u[n] = u; h_v[n] = v; h_deg[n] = d;
                                h_sgn[n] = d % 2 ? -1. : 1.;
                                n++;
                        }
        /* R^(n)_{t+1} = t R^(n+1)_{t-1} + X R^(n+1)_t along the first nonzero component */
        for (n = 1; n < NHMAX; n++) {
                int c[3] = {h_t[n], h_u[n], h_v[n]}, k = c[0] ? 0 : (c[1] ? 1 : 2);
                int m1[3] = {c[0], c[1], c[2]}, m2[3] = {c[0], c[1], c[2]};
                m1[k] -= 1;
                m2[k] -= 2;
                h_dir[n] = k;
                h_m1[n] = h_idx[m1[0]][m1[1]][m1[2]];
                h_c[n] = c[k] - 1;
                h_m2[n] = c[k] >= 2 ? h_idx[m2[0]][m2[1]][m2[2]] : 0;
        }
        for (a = 0; a < NHOP; a++)
                for (b = 0; b < NHOP; b++)
                        h_add[a * NHOP + b] = h_idx[h_t[a] + h_t[b]][h_u[a] + h_u[b]][h_v[a] + h_v[b]];
        for (n = 0; n < BOYS_NT; n++) boys_exact(boys_tab[n], n * .05, BOYS_MT - 1);
        for (n = 1; n < BOYS_MT; n++) boys_inv[n] = 1. / (2 * n - 1);
        h_ready = 1;
}

static double common_fac_sp(int l)
{
        if (l == 0) return 0.282094791773878143;
        if (l == 1) return 0.488602511902919921;
        return 1.;
}

/* R0[i] = R^(0)_i up to degree L, for exponent theta and separation (X, Y, Z) */
static void hermite_R(double *R0, double *wk, int L, double pref, double theta, double X, double Y, double Z)
{
        const int nh = NH(L);
        double F[DMAX + 2];
        int n, i;
        boys_fn(F, theta * (X * X + Y * Y + Z * Z), L);
        /* wk[n * nh + i] = R^(n)_i */
        double f = pref;
        for (n = 0; n <= L; n++) {
                wk[n * nh] = f * F[n];
                f *= -2 * theta;
        }
        const double XYZ[3] = {X, Y, Z};
        for (i = 1; i < nh; i++) {
                const int d = h_deg[i], m1 = h_m1[i], m2 = h_m2[i], c = h_c[i];
                const double x = XYZ[h_dir[i]];
                for (n = 0; n <= L - d; n++) {
                        double r = x * wk[(n + 1) * nh + m1];
                        if (c > 0) r += c * wk[(n + 1) * nh + m2];
                        wk[n * nh + i] = r;
                }
        }
        for (i = 0; i < nh; i++) R0[i] = wk[i];
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
                        /* raise j from (i, j - 1), or i from (i - 1, 0) */
                        const int raise_j = j > 0;
                        const double *p = raise_j ? E + i * s1 + (j - 1) * s2 : E + (i - 1) * s1;
                        const double xr = raise_j ? xpb : xpa;
                        for (t = 0; t <= i + j; t++) {
                                /* E^{i,j+1}_t = h E^{ij}_{t-1} + X_PB E^{ij}_t + (t + 1) E^{ij}_{t+1} */
                                double v = xr * p[t];
                                if (t > 0) v += h * p[t - 1];
                                if (t + 1 <= i + j - 1) v += (t + 1) * p[t + 1];
                                e[t] = v;
                        }
                }
        }
}

typedef struct {
        int ish, jsh, li, lj, ia, ja;
        double p, P[3];
        size_t orho, oder, ov;          /* offsets of rho (NH(lij)), the six derivative
                                         * densities (6 NH(lij + 1)) and V (NH(lij + 1)) */
} MDPair;

/* rho^P and the derivative densities d/dA_x, d/dB_x (x = 0..2) of one pair */
static void pair_densities(double *rho, double *der, const MDPair *pp, const double *Ra, const double *Rb,
                           double *dm, size_t nao, int *ao_loc, int *bas, double *env, double *wk)
{
        const int li = pp->li, lj = pp->lj, lij = li + lj;
        const double ai = env[bas[pp->ish * BAS_SLOTS + PTR_EXP]], aj = env[bas[pp->jsh * BAS_SLOTS + PTR_EXP]];
        const double ci = env[bas[pp->ish * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(li);
        const double cj = env[bas[pp->jsh * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(lj);
        const double w = (pp->ish == pp->jsh ? 1. : 2.) * ci * cj;
        /* E tables with both indices raised by one, t up to lij + 2 */
        const int Li = li + 1, Lj = lj + 1, s2 = Li + Lj + 1, s1 = (Lj + 1) * s2, sz = (Li + 1) * s1;
        double *Ex = wk, *Ey = Ex + sz, *Ez = Ey + sz;
        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
        const double kab = exp(-ai * aj / (ai + aj) * r2);
        hermite_E(Ex, Li, Lj, ai, aj, Ra[0], Rb[0], 1.);
        hermite_E(Ey, Li, Lj, ai, aj, Ra[1], Rb[1], 1.);
        hermite_E(Ez, Li, Lj, ai, aj, Ra[2], Rb[2], kab);
        const int nh0 = NH(lij), nh1 = NH(lij + 1);
        memset(rho, 0, sizeof(double) * nh0);
        memset(der, 0, sizeof(double) * 6 * nh1);
        int ix, iy, iz, jx, jy, jz, n, x;
        const size_t a0 = ao_loc[pp->ish], b0 = ao_loc[pp->jsh];
        int fa = 0;
        for (ix = li; ix >= 0; ix--)
        for (iy = li - ix; iy >= 0; iy--, fa++) {
                iz = li - ix - iy;
                int fb = 0;
                for (jx = lj; jx >= 0; jx--)
                for (jy = lj - jx; jy >= 0; jy--, fb++) {
                        jz = lj - jx - jy;
                        const double d = w * dm[(a0 + fa) * nao + b0 + fb];
                        if (d == 0) continue;
                        const int ia[3] = {ix, iy, iz}, ja[3] = {jx, jy, jz};
                        const double *Es[3] = {Ex, Ey, Ez};
                        /* the 1D factors of the pair and of its derivatives:
                         * e[k][0] = E^{ij}, e[k][1] = d/dA, e[k][2] = d/dB, length s2 */
                        double e[3][3][2 * LMAX + 4];
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
                                const int t = h_t[n], u = h_u[n], v = h_v[n];
                                const double ex = e[0][0][t], ey = e[1][0][u], ez = e[2][0][v];
                                if (n < nh0) rho[n] += d * ex * ey * ez;
                                for (x = 0; x < 3; x++) {
                                        const double fA = (x == 0 ? e[0][1][t] : ex) * (x == 1 ? e[1][1][u] : ey)
                                                        * (x == 2 ? e[2][1][v] : ez);
                                        const double fB = (x == 0 ? e[0][2][t] : ex) * (x == 1 ? e[1][2][u] : ey)
                                                        * (x == 2 ? e[2][2][v] : ez);
                                        der[x * nh1 + n] += d * fA;
                                        der[(3 + x) * nh1 + n] += d * fB;
                                }
                        }
                }
        }
}

#define PI_FAC 34.986836655249725               /* 2 pi^(5/2) */

/* a^T T_PQ b with T over degrees (na, nb), R0 the Hermite R of the pair of pairs */
static double hermite_quad(const double *a, int na, const double *b, int nb, const double *R0)
{
        double s = 0;
        int i, j;
        for (i = 0; i < NH(na); i++) {
                double t = 0;
                for (j = 0; j < NH(nb); j++) t += h_sgn[j] * R0[h_add[i * NHOP + j]] * b[j];
                s += a[i] * t;
        }
        return s;
}

static int cmp_desc(const void *x, const void *y)
{
        const double a = ((const double *)x)[0], b = ((const double *)y)[0];
        return a < b ? 1 : (a > b ? -1 : 0);
}

/*
 * grad (natm, 3) += dE_J / dR; *energy = E_J (if not NULL).  dm: the
 * symmetric density in the decontracted Cartesian basis of (bas, nbas);
 * all shell pairs a >= b enter (density screening inside).
 */
void SOCVHFmd_j_grad(double *grad, double *energy, double *dm, double tol,
                     int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        if (!h_ready) hermite_tables();
        const size_t nao = ao_loc[nbas];
        const int npair = nbas * (nbas + 1) / 2;
        int i;
        for (i = 0; i < nbas; i++) {
                if (bas[i * BAS_SLOTS + ANG_OF] > LMAX || bas[i * BAS_SLOTS + NPRIM_OF] != 1
                    || bas[i * BAS_SLOTS + NCTR_OF] != 1) {
                        return;         /* the caller checks the shells */
                }
        }
        MDPair *pairs = malloc(sizeof(MDPair) * npair);
        size_t orho = 0, oder = 0, ov = 0;
        {
                int a, b, n = 0;
                for (a = 0; a < nbas; a++)
                        for (b = 0; b <= a; b++, n++) {
                                MDPair *pp = pairs + n;
                                pp->ish = a; pp->jsh = b;
                                pp->li = bas[a * BAS_SLOTS + ANG_OF];
                                pp->lj = bas[b * BAS_SLOTS + ANG_OF];
                                pp->ia = bas[a * BAS_SLOTS + ATOM_OF];
                                pp->ja = bas[b * BAS_SLOTS + ATOM_OF];
                                const double ai = env[bas[a * BAS_SLOTS + PTR_EXP]], aj = env[bas[b * BAS_SLOTS + PTR_EXP]];
                                const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                                const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                                int x;
                                pp->p = ai + aj;
                                for (x = 0; x < 3; x++) pp->P[x] = (ai * Ra[x] + aj * Rb[x]) / pp->p;
                                const int lij = pp->li + pp->lj;
                                pp->orho = orho; orho += NH(lij);
                                pp->oder = oder; oder += 6 * NH(lij + 1);
                                pp->ov = ov; ov += NH(lij + 1);
                        }
        }
        double *rho = malloc(sizeof(double) * orho);
        double *der = malloc(sizeof(double) * oder);
        double *bound = malloc(sizeof(double) * 2 * npair);     /* (b_P, index) for the sort */
        double *snorm = malloc(sizeof(double) * 2 * npair);     /* |Omega_P|, |d Omega_P| */
        const int nthreads = omp_get_max_threads();
        double **vbufs = malloc(sizeof(double *) * nthreads);
        double **gbufs = malloc(sizeof(double *) * nthreads);
        double esum = 0;
        int *order = malloc(sizeof(int) * npair);
        int nkeep = 0;

#pragma omp parallel num_threads(nthreads)
{
        const int tid = omp_get_thread_num();
        const int sE = (LMAX + 2) * (LMAX + 2) * (2 * LMAX + 5);
        double *wk = malloc(sizeof(double) * (3 * sE + (DMAX + 2) * NHMAX));
        double *R0 = malloc(sizeof(double) * NHMAX);
        double *wR = wk + 3 * sE;
        int n;
        /* densities and the Schwarz norms of every pair */
#pragma omp for schedule(dynamic, 16)
        for (n = 0; n < npair; n++) {
                MDPair *pp = pairs + n;
                const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                pair_densities(rho + pp->orho, der + pp->oder, pp, Ra, Rb, dm, nao, ao_loc, bas, env, wk);
                const int lij = pp->li + pp->lj;
                const double pref = PI_FAC / (pp->p * pp->p * sqrt(2 * pp->p));
                hermite_R(R0, wR, 2 * lij + 2, pref, .5 * pp->p, 0, 0, 0);
                double s0 = hermite_quad(rho + pp->orho, lij, rho + pp->orho, lij, R0);
                double s1 = 0;
                int x;
                for (x = 0; x < 6; x++) {
                        const double *dx = der + pp->oder + (size_t)x * NH(lij + 1);
                        double v = hermite_quad(dx, lij + 1, dx, lij + 1, R0);
                        if (v > s1) s1 = v;
                }
                snorm[2 * n] = sqrt(fabs(s0));
                snorm[2 * n + 1] = sqrt(fabs(s1));
                bound[2 * n] = snorm[2 * n] > snorm[2 * n + 1] ? snorm[2 * n] : snorm[2 * n + 1];
                bound[2 * n + 1] = n;
        }
#pragma omp single
{
        qsort(bound, npair, 2 * sizeof(double), cmp_desc);
        const double bmax = npair ? bound[0] : 0;
        for (n = 0; n < npair; n++) {
                if (2 * bound[2 * n] * bmax < tol) break;
                order[n] = (int)bound[2 * n + 1];
        }
        nkeep = n;
}
        double *V = vbufs[tid] = calloc(ov, sizeof(double));
        double *gl = gbufs[tid] = calloc((size_t)natm * 3, sizeof(double));
        /* V of every pair: pairs of pairs P <= Q in the screening order */
        int ip;
#pragma omp for schedule(dynamic, 1)
        for (ip = 0; ip < nkeep; ip++) {
                const MDPair *P = pairs + order[ip];
                const int lp = P->li + P->lj;
                const double bp = bound[2 * ip];
                const double *rp = rho + P->orho;
                double *vp = V + P->ov;
                int iq;
                for (iq = ip; iq < nkeep; iq++) {
                        if (2 * bp * bound[2 * iq] < tol) break;
                        const MDPair *Q = pairs + order[iq];
                        const int lq = Q->li + Q->lj;
                        const double *rq = rho + Q->orho;
                        double *vq = V + Q->ov;
                        const double pq = P->p * Q->p, psum = P->p + Q->p;
                        if (lp + 1 <= MDC_LMAX && lq + 1 <= MDC_LMAX) {
                                /* md_j_gen.c: V^P (degree lp + 1) by the up part of mdc_(lp+1)_lq,
                                 * V^Q (degree lq + 1) by the uq part of mdc_lp_(lq+1) */
                                mdr_table[lp + lq + 1](R0, PI_FAC / (pq * sqrt(psum)), pq / psum,
                                                       P->P[0] - Q->P[0], P->P[1] - Q->P[1], P->P[2] - Q->P[2]);
                                mdc_table[lp + 1][lq](R0, NULL, rq, vp, NULL, 1, 0);
                                if (iq != ip) mdc_table[lp][lq + 1](R0, rp, NULL, NULL, vq, 0, 1);
                                continue;
                        }
                        hermite_R(R0, wR, lp + lq + 1, PI_FAC / (pq * sqrt(psum)), pq / psum,
                                  P->P[0] - Q->P[0], P->P[1] - Q->P[1], P->P[2] - Q->P[2]);
                        int a, b;
                        /* V^P_a += sum_b (-1)^|b| R_{a+b} rho^Q_b, a up to lp + 1 */
                        for (a = 0; a < NH(lp + 1); a++) {
                                double s = 0;
                                const int *ad = h_add + a * NHOP;
                                for (b = 0; b < NH(lq); b++) s += h_sgn[b] * R0[ad[b]] * rq[b];
                                vp[a] += s;
                        }
                        if (iq == ip) continue;
                        /* V^Q_b += sum_a (-1)^|b| R_{a+b} rho^P_a, b up to lq + 1 */
                        for (b = 0; b < NH(lq + 1); b++) {
                                double s = 0;
                                for (a = 0; a < NH(lp); a++) s += R0[h_add[a * NHOP + b]] * rp[a];
                                vq[b] += h_sgn[b] * s;
                        }
                }
        }
#pragma omp barrier
        /* reduce V (each thread a slice) */
        {
                const size_t chunk = (ov + nthreads - 1) / nthreads;
                const size_t s0 = chunk * tid < ov ? chunk * tid : ov;
                const size_t s1 = s0 + chunk < ov ? s0 + chunk : ov;
                size_t k;
                int t;
                for (t = 0; t < nthreads; t++) {
                        if (t == tid) continue;
                        const double *vt = vbufs[t];
                        for (k = s0; k < s1; k++) V[k] += vt[k];
                }
                for (t = 0; t < nthreads; t++) {
                        if (t == tid) continue;
                        double *vt = vbufs[t];
                        for (k = s0; k < s1; k++) vt[k] = V[k];
                }
        }
#pragma omp barrier
        /* gradient and energy from every thread's (now complete) V */
        double el = 0;
#pragma omp for schedule(static)
        for (ip = 0; ip < nkeep; ip++) {
                const MDPair *P = pairs + order[ip];
                const int lp = P->li + P->lj, nh1 = NH(lp + 1);
                const double *vp = V + P->ov, *rp = rho + P->orho, *dp = der + P->oder;
                int a, x;
                for (a = 0; a < NH(lp); a++) el += rp[a] * vp[a];
                for (x = 0; x < 3; x++) {
                        double gA = 0, gB = 0;
                        for (a = 0; a < nh1; a++) {
                                gA += dp[x * nh1 + a] * vp[a];
                                gB += dp[(3 + x) * nh1 + a] * vp[a];
                        }
                        gl[P->ia * 3 + x] += gA;
                        gl[P->ja * 3 + x] += gB;
                }
        }
#pragma omp critical
        {
                esum += .5 * el;
                for (n = 0; n < natm * 3; n++) grad[n] += gl[n];
        }
        free(wk);
        free(R0);
#pragma omp barrier
        free(V);
        free(gl);
}
        if (energy != NULL) *energy = esum;
        free(pairs);
        free(rho);
        free(der);
        free(bound);
        free(snorm);
        free(vbufs);
        free(gbufs);
        free(order);
}

/* E^{ab}_tuv of all Cartesian pairs of a primitive pair, the libcint coefficients
 * included: Et[(fa * nfb + fb) * NH(lij) + n] */
static void pair_Et(double *Et, const MDPair *pp, const double *Ra, const double *Rb, int *bas, double *env,
                    double *wk)
{
        const int li = pp->li, lj = pp->lj, lij = li + lj, nh = NH(lij);
        const double ai = env[bas[pp->ish * BAS_SLOTS + PTR_EXP]], aj = env[bas[pp->jsh * BAS_SLOTS + PTR_EXP]];
        const double cc = env[bas[pp->ish * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(li)
                        * env[bas[pp->jsh * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(lj);
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
                        for (n = 0; n < nh; n++) e[n] = ex[h_t[n]] * ey[h_u[n]] * ez[h_v[n]];
                }
        }
}

/*
 * J matrices of ndm symmetric densities in the decontracted Cartesian basis,
 * vj[d] = sum_cd (ab|cd) D[d]_cd, from the Hermite potentials:
 *
 *     rho^Q_s = w_Q sum_cd D_cd E^cd_s,   V^P_t = sum_Q T_PQ{t,s} rho^Q_s,
 *     J_ab = sum_t E^ab_t V^P_t           (both a b and b a),
 *
 * one pass over the pairs of pairs P <= Q.  Screening: the contributions to
 * J^P and J^Q are bounded by q_P |Omega_Q| and |Omega_P| q_Q, with q_P the
 * Schwarz factor sqrt((ab|ab)) of the shell pair (qcond, nbas x nbas) and
 * |Omega_P| = max_d sqrt((Omega^d_P|Omega^d_P)); pairs sorted by
 * b_P = max(q_P, |Omega_P|), the inner loop stops at b_P b_Q < tol.
 */
void SOCVHFmd_j_matrix(double *vj, double *dm, int ndm, double *qcond, double tol,
                       int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        if (!h_ready) hermite_tables();
        const size_t nao = ao_loc[nbas], nn = nao * nao;
        const int npair = nbas * (nbas + 1) / 2;
        int i;
        for (i = 0; i < nbas; i++) {
                if (bas[i * BAS_SLOTS + ANG_OF] > LMAX || bas[i * BAS_SLOTS + NPRIM_OF] != 1
                    || bas[i * BAS_SLOTS + NCTR_OF] != 1) {
                        return;
                }
        }
        MDPair *pairs = malloc(sizeof(MDPair) * npair);
        size_t oh = 0;          /* Hermite length of all pairs, per density */
        {
                int a, b, n = 0;
                for (a = 0; a < nbas; a++)
                        for (b = 0; b <= a; b++, n++) {
                                MDPair *pp = pairs + n;
                                pp->ish = a; pp->jsh = b;
                                pp->li = bas[a * BAS_SLOTS + ANG_OF];
                                pp->lj = bas[b * BAS_SLOTS + ANG_OF];
                                pp->ia = bas[a * BAS_SLOTS + ATOM_OF];
                                pp->ja = bas[b * BAS_SLOTS + ATOM_OF];
                                const double ai = env[bas[a * BAS_SLOTS + PTR_EXP]], aj = env[bas[b * BAS_SLOTS + PTR_EXP]];
                                const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                                const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                                int x;
                                pp->p = ai + aj;
                                for (x = 0; x < 3; x++) pp->P[x] = (ai * Ra[x] + aj * Rb[x]) / pp->p;
                                pp->orho = oh;
                                oh += NH(pp->li + pp->lj);
                        }
        }
        /* rho[d * oh + orho + n] */
        double *rho = malloc(sizeof(double) * oh * ndm);
        double *bound = malloc(sizeof(double) * 2 * npair);
        double *qv = malloc(sizeof(double) * npair);    /* q_P, |Omega_P| by pair */
        double *sv = malloc(sizeof(double) * npair);
        const int nthreads = omp_get_max_threads();
        double **vbufs = malloc(sizeof(double *) * nthreads);
        int *order = malloc(sizeof(int) * npair);
        int nkeep = 0;
        memset(vj, 0, sizeof(double) * nn * ndm);

#pragma omp parallel num_threads(nthreads)
{
        const int tid = omp_get_thread_num();
        const int sE = (LMAX + 1) * (LMAX + 1) * (2 * LMAX + 1);
        const int ncmax = (LMAX + 1) * (LMAX + 2) / 2;
        double *wk = malloc(sizeof(double) * (3 * sE + (DMAX + 2) * NHMAX));
        double *wR = wk + 3 * sE;
        double *Et = malloc(sizeof(double) * ncmax * ncmax * NH(2 * LMAX));
        double *R0 = malloc(sizeof(double) * NHMAX);
        int n, d;
        /* Hermite densities and bounds */
#pragma omp for schedule(dynamic, 16)
        for (n = 0; n < npair; n++) {
                const MDPair *pp = pairs + n;
                const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                const int lij = pp->li + pp->lj, nh = NH(lij);
                const int nfa = (pp->li + 1) * (pp->li + 2) / 2, nfb = (pp->lj + 1) * (pp->lj + 2) / 2;
                const double w = pp->ish == pp->jsh ? 1. : 2.;
                pair_Et(Et, pp, Ra, Rb, bas, env, wk);
                const size_t a0 = ao_loc[pp->ish], b0 = ao_loc[pp->jsh];
                for (d = 0; d < ndm; d++) {
                        double *r = rho + (size_t)d * oh + pp->orho;
                        const double *D = dm + (size_t)d * nn;
                        int fa, fb, k;
                        for (k = 0; k < nh; k++) r[k] = 0;
                        for (fa = 0; fa < nfa; fa++)
                                for (fb = 0; fb < nfb; fb++) {
                                        const double v = w * D[(a0 + fa) * nao + b0 + fb];
                                        const double *e = Et + (size_t)(fa * nfb + fb) * nh;
                                        for (k = 0; k < nh; k++) r[k] += v * e[k];
                                }
                }
                const double pref = PI_FAC / (pp->p * pp->p * sqrt(2 * pp->p));
                hermite_R(R0, wR, 2 * lij, pref, .5 * pp->p, 0, 0, 0);
                double s = 0;
                for (d = 0; d < ndm; d++) {
                        const double *r = rho + (size_t)d * oh + pp->orho;
                        const double v = hermite_quad(r, lij, r, lij, R0);
                        if (v > s) s = v;
                }
                s = sqrt(fabs(s));
                const double q = qcond[pp->ish * nbas + pp->jsh];
                qv[n] = q;
                sv[n] = s;
                bound[2 * n] = q > s ? q : s;
                bound[2 * n + 1] = n;
        }
#pragma omp single
{
        qsort(bound, npair, 2 * sizeof(double), cmp_desc);
        const double bmax = npair ? bound[0] : 0;
        for (n = 0; n < npair; n++) {
                if (bound[2 * n] * bmax < tol) break;
                order[n] = (int)bound[2 * n + 1];
        }
        nkeep = n;
}
        double *V = vbufs[tid] = calloc(oh * ndm, sizeof(double));
        int ip;
#pragma omp for schedule(dynamic, 1)
        for (ip = 0; ip < nkeep; ip++) {
                const MDPair *P = pairs + order[ip];
                const int lp = P->li + P->lj, np = NH(lp);
                const double bp = bound[2 * ip], qp = qv[order[ip]], sp = sv[order[ip]];
                int iq;
                for (iq = ip; iq < nkeep; iq++) {
                        if (bp * bound[2 * iq] < tol) break;
                        /* V^P needs rho^Q (bound q_P |Omega_Q|), V^Q needs rho^P */
                        const int up = qp * sv[order[iq]] >= tol;
                        const int uq = iq != ip && sp * qv[order[iq]] >= tol;
                        if (!up && !uq) continue;
                        const MDPair *Q = pairs + order[iq];
                        const int lq = Q->li + Q->lj, nq = NH(lq);
                        const double pq = P->p * Q->p, psum = P->p + Q->p;
                        if (lp <= MDC_LMAX && lq <= MDC_LMAX) {
                                /* the unrolled R and contractions of md_j_gen.c */
                                mdr_table[lp + lq](R0, PI_FAC / (pq * sqrt(psum)), pq / psum,
                                                   P->P[0] - Q->P[0], P->P[1] - Q->P[1], P->P[2] - Q->P[2]);
                                for (d = 0; d < ndm; d++)
                                        mdc_table[lp][lq](R0, rho + (size_t)d * oh + P->orho,
                                                          rho + (size_t)d * oh + Q->orho, V + (size_t)d * oh + P->orho,
                                                          V + (size_t)d * oh + Q->orho, up, uq);
                                continue;
                        }
                        hermite_R(R0, wR, lp + lq, PI_FAC / (pq * sqrt(psum)), pq / psum,
                                  P->P[0] - Q->P[0], P->P[1] - Q->P[1], P->P[2] - Q->P[2]);
                        int a, b;
                        for (d = 0; d < ndm; d++) {
                                const double *rp = rho + (size_t)d * oh + P->orho;
                                const double *rq = rho + (size_t)d * oh + Q->orho;
                                double *vp = V + (size_t)d * oh + P->orho;
                                double *vq = V + (size_t)d * oh + Q->orho;
                                if (up) {
                                        for (a = 0; a < np; a++) {
                                                const int *ad = h_add + a * NHOP;
                                                double s = 0;
                                                for (b = 0; b < nq; b++) s += h_sgn[b] * R0[ad[b]] * rq[b];
                                                vp[a] += s;
                                        }
                                }
                                if (!uq) continue;
                                for (b = 0; b < nq; b++) {
                                        double s = 0;
                                        for (a = 0; a < np; a++) s += R0[h_add[a * NHOP + b]] * rp[a];
                                        vq[b] += h_sgn[b] * s;
                                }
                        }
                }
        }
#pragma omp barrier
        /* sum the thread buffers into vbufs[0], each thread a slice */
        {
                const size_t tot = oh * ndm;
                const size_t chunk = (tot + nthreads - 1) / nthreads;
                const size_t s0 = chunk * tid < tot ? chunk * tid : tot;
                const size_t s1 = s0 + chunk < tot ? s0 + chunk : tot;
                double *V0 = vbufs[0];
                size_t k;
                int t;
                for (t = 1; t < nthreads; t++) {
                        const double *vt = vbufs[t];
                        for (k = s0; k < s1; k++) V0[k] += vt[k];
                }
        }
#pragma omp barrier
        /* J_ab = sum_t E^ab_t V^P_t for every kept pair (blocks of different pairs are disjoint) */
        const double *V0 = vbufs[0];
#pragma omp for schedule(dynamic, 16)
        for (ip = 0; ip < nkeep; ip++) {
                const MDPair *pp = pairs + order[ip];
                const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                const int lij = pp->li + pp->lj, nh = NH(lij);
                const int nfa = (pp->li + 1) * (pp->li + 2) / 2, nfb = (pp->lj + 1) * (pp->lj + 2) / 2;
                pair_Et(Et, pp, Ra, Rb, bas, env, wk);
                const size_t a0 = ao_loc[pp->ish], b0 = ao_loc[pp->jsh];
                for (d = 0; d < ndm; d++) {
                        const double *v = V0 + (size_t)d * oh + pp->orho;
                        double *J = vj + (size_t)d * nn;
                        int fa, fb, k;
                        for (fa = 0; fa < nfa; fa++)
                                for (fb = 0; fb < nfb; fb++) {
                                        const double *e = Et + (size_t)(fa * nfb + fb) * nh;
                                        double s = 0;
                                        for (k = 0; k < nh; k++) s += e[k] * v[k];
                                        J[(a0 + fa) * nao + b0 + fb] = s;
                                        J[(b0 + fb) * nao + a0 + fa] = s;
                                }
                }
        }
        free(wk);
        free(Et);
        free(R0);
#pragma omp barrier
        free(V);
}
        free(pairs);
        free(rho);
        free(bound);
        free(qv);
        free(sv);
        free(vbufs);
        free(order);
}

/*
 * The three-centre part of the RIJ gradient, g^xi = sum_P d_P sum_ij D_ij d(ij|P)/dR_xi = the
 * derivative of the cross energy E_x = sum_ij D_ij (ij|d~), d~ = sum_P d_P phi_P, at fixed D and d,
 * by the J engine above: the bra pairs (a >= b of shells [0, nbra)) carry rho^P of D, every
 * auxiliary primitive c (shells [nbra, nbra + naux)) is a "pair" (c, s0) with s0 = s0sh[c - nbra] a
 * zero-exponent s shell on its centre (coefficient 1 / CINTcommon_fac_sp(0): the constant 1), its
 * density the fit coefficient in dm[c, s0] (times 2 by the pair weight, so dm holds d / 2).  Only the
 * cross pairs of pairs (bra, aux) enter; V^P and V^Q in one pass, then
 *     g = sum_P <d rho^P, V^P> + sum_Q <d rho^Q, V^Q>
 * (the s0 derivative densities vanish).  grad (natm, 3) +=, *energy = E_x.  All shells one primitive,
 * Cartesian (decontracted), l <= LMAX.
 */
void SOCVHFmd_rij_grad(double *grad, double *energy, double *dm, double tol, int nbra, int naux, int *s0sh,
                       int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        if (!h_ready) hermite_tables();
        const size_t nao = ao_loc[nbas];
        const int nbp = nbra * (nbra + 1) / 2, npair = nbp + naux;
        int i;
        for (i = 0; i < nbas; i++)
                if (bas[i * BAS_SLOTS + ANG_OF] > LMAX || bas[i * BAS_SLOTS + NPRIM_OF] != 1
                    || bas[i * BAS_SLOTS + NCTR_OF] != 1) return;
        MDPair *pairs = malloc(sizeof(MDPair) * npair);
        size_t orho = 0, oder = 0, ov = 0;
        {
                int a, b, n = 0;
                for (n = 0; n < npair; n++) {
                        if (n < nbp) {
                                a = (int)((sqrt(1. + 8. * n) - 1) / 2);
                                while (a * (a + 1) / 2 > n) a--;
                                while ((a + 1) * (a + 2) / 2 <= n) a++;
                                b = n - a * (a + 1) / 2;
                        } else {
                                a = nbra + (n - nbp);
                                b = s0sh[n - nbp];
                        }
                        MDPair *pp = pairs + n;
                        pp->ish = a; pp->jsh = b;
                        pp->li = bas[a * BAS_SLOTS + ANG_OF];
                        pp->lj = bas[b * BAS_SLOTS + ANG_OF];
                        pp->ia = bas[a * BAS_SLOTS + ATOM_OF];
                        pp->ja = bas[b * BAS_SLOTS + ATOM_OF];
                        const double ai = env[bas[a * BAS_SLOTS + PTR_EXP]], aj = env[bas[b * BAS_SLOTS + PTR_EXP]];
                        const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                        const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                        int x;
                        pp->p = ai + aj;
                        for (x = 0; x < 3; x++) pp->P[x] = (ai * Ra[x] + aj * Rb[x]) / pp->p;
                        const int lij = pp->li + pp->lj;
                        pp->orho = orho; orho += NH(lij);
                        pp->oder = oder; oder += 6 * NH(lij + 1);
                        pp->ov = ov; ov += NH(lij + 1);
                }
        }
        double *rho = malloc(sizeof(double) * orho);
        double *der = malloc(sizeof(double) * oder);
        double *bound = malloc(sizeof(double) * 2 * npair);
        const int nthreads = omp_get_max_threads();
        double **vbufs = malloc(sizeof(double *) * nthreads);
        double esum = 0;
        int *order = malloc(sizeof(int) * npair);
        int nkb = 0, nka = 0;
#pragma omp parallel num_threads(nthreads)
{
        const int tid = omp_get_thread_num();
        const int sE = (LMAX + 2) * (LMAX + 2) * (2 * LMAX + 5);
        double *wk = malloc(sizeof(double) * (3 * sE + (DMAX + 2) * NHMAX));
        double *R0 = malloc(sizeof(double) * NHMAX);
        double *wR = wk + 3 * sE;
        int n;
#pragma omp for schedule(dynamic, 16)
        for (n = 0; n < npair; n++) {
                MDPair *pp = pairs + n;
                const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                pair_densities(rho + pp->orho, der + pp->oder, pp, Ra, Rb, dm, nao, ao_loc, bas, env, wk);
                const int lij = pp->li + pp->lj;
                const double pref = PI_FAC / (pp->p * pp->p * sqrt(2 * pp->p));
                hermite_R(R0, wR, 2 * lij + 2, pref, .5 * pp->p, 0, 0, 0);
                double s0 = hermite_quad(rho + pp->orho, lij, rho + pp->orho, lij, R0);
                double s1 = 0;
                int x;
                for (x = 0; x < 6; x++) {
                        const double *dx = der + pp->oder + (size_t)x * NH(lij + 1);
                        double v = hermite_quad(dx, lij + 1, dx, lij + 1, R0);
                        if (v > s1) s1 = v;
                }
                s0 = sqrt(fabs(s0));
                s1 = sqrt(fabs(s1));
                bound[2 * n] = s0 > s1 ? s0 : s1;
                bound[2 * n + 1] = n;
        }
#pragma omp single
{
        /* bra and auxiliary pairs sorted separately, each by its bound */
        qsort(bound, nbp, 2 * sizeof(double), cmp_desc);
        qsort(bound + 2 * nbp, naux, 2 * sizeof(double), cmp_desc);
        const double bbmax = nbp ? bound[0] : 0, bamax = naux ? bound[2 * nbp] : 0;
        for (n = 0; n < nbp; n++) {
                if (2 * bound[2 * n] * bamax < tol) break;
                order[n] = (int)bound[2 * n + 1];
        }
        nkb = n;
        for (n = 0; n < naux; n++) {
                if (2 * bound[2 * (nbp + n)] * bbmax < tol) break;
                order[nbp + n] = (int)bound[2 * (nbp + n) + 1];
        }
        nka = n;
}
        double *V = vbufs[tid] = calloc(ov, sizeof(double));
        int ip;
#pragma omp for schedule(dynamic, 1)
        for (ip = 0; ip < nkb; ip++) {
                const MDPair *P = pairs + order[ip];
                const int lp = P->li + P->lj;
                const double bp = bound[2 * ip];
                const double *rp = rho + P->orho;
                double *vp = V + P->ov;
                int iq;
                for (iq = 0; iq < nka; iq++) {
                        if (2 * bp * bound[2 * (nbp + iq)] < tol) break;
                        const MDPair *Q = pairs + order[nbp + iq];
                        const int lq = Q->li + Q->lj;
                        const double *rq = rho + Q->orho;
                        double *vq = V + Q->ov;
                        const double pq = P->p * Q->p, psum = P->p + Q->p;
                        if (lp + 1 <= MDC_LMAX && lq + 1 <= MDC_LMAX) {
                                mdr_table[lp + lq + 1](R0, PI_FAC / (pq * sqrt(psum)), pq / psum,
                                                       P->P[0] - Q->P[0], P->P[1] - Q->P[1], P->P[2] - Q->P[2]);
                                mdc_table[lp + 1][lq](R0, NULL, rq, vp, NULL, 1, 0);
                                mdc_table[lp][lq + 1](R0, rp, NULL, NULL, vq, 0, 1);
                                continue;
                        }
                        hermite_R(R0, wR, lp + lq + 1, PI_FAC / (pq * sqrt(psum)), pq / psum,
                                  P->P[0] - Q->P[0], P->P[1] - Q->P[1], P->P[2] - Q->P[2]);
                        int a, b;
                        for (a = 0; a < NH(lp + 1); a++) {
                                double s = 0;
                                const int *ad = h_add + a * NHOP;
                                for (b = 0; b < NH(lq); b++) s += h_sgn[b] * R0[ad[b]] * rq[b];
                                vp[a] += s;
                        }
                        for (b = 0; b < NH(lq + 1); b++) {
                                double s = 0;
                                for (a = 0; a < NH(lp); a++) s += R0[h_add[a * NHOP + b]] * rp[a];
                                vq[b] += h_sgn[b] * s;
                        }
                }
        }
#pragma omp barrier
        {
                const size_t chunk = (ov + nthreads - 1) / nthreads;
                const size_t s0 = chunk * tid < ov ? chunk * tid : ov;
                const size_t s1 = s0 + chunk < ov ? s0 + chunk : ov;
                size_t k;
                int t;
                for (t = 0; t < nthreads; t++) {
                        if (t == tid) continue;
                        const double *vt = vbufs[t];
                        for (k = s0; k < s1; k++) V[k] += vt[k];
                }
                for (t = 0; t < nthreads; t++) {
                        if (t == tid) continue;
                        double *vt = vbufs[t];
                        for (k = s0; k < s1; k++) vt[k] = V[k];
                }
        }
#pragma omp barrier
        double el = 0;
        double *gl = calloc((size_t)natm * 3, sizeof(double));
        const int ntot = nkb + nka;
#pragma omp for schedule(static)
        for (ip = 0; ip < ntot; ip++) {
                const MDPair *P = pairs + (ip < nkb ? order[ip] : order[nbp + ip - nkb]);
                const int lp = P->li + P->lj, nh1 = NH(lp + 1);
                const double *vp = V + P->ov, *rp = rho + P->orho, *dp = der + P->oder;
                int a, x;
                if (ip < nkb)
                        for (a = 0; a < NH(lp); a++) el += rp[a] * vp[a];
                for (x = 0; x < 3; x++) {
                        double gA = 0, gB = 0;
                        for (a = 0; a < nh1; a++) {
                                gA += dp[x * nh1 + a] * vp[a];
                                gB += dp[(3 + x) * nh1 + a] * vp[a];
                        }
                        gl[P->ia * 3 + x] += gA;
                        gl[P->ja * 3 + x] += gB;
                }
        }
#pragma omp critical
        {
                esum += el;
                for (n = 0; n < natm * 3; n++) grad[n] += gl[n];
        }
        free(gl);
        free(wk);
        free(R0);
#pragma omp barrier
        free(V);
}
        if (energy != NULL) *energy = esum;
        free(pairs);
        free(rho);
        free(der);
        free(bound);
        free(vbufs);
        free(order);
}

/* <d2 rho^P / dX_x dY_y, V> of one pair for (X, Y) = (A, A), (A, B), (B, B): out[(blk * 3 + x) * 3 + y],
 * blk 0 AA, 1 AB, 2 BB; V of degree lij + 2.  The second-derivative densities from E tables with both
 * indices raised by two, formed on the fly. */
static void pair_d2_dot(double *out, const MDPair *pp, const double *Ra, const double *Rb, const double *V,
                        double *dm, size_t nao, int *ao_loc, int *bas, double *env, double *wk)
{
        const int li = pp->li, lj = pp->lj, lij = li + lj;
        const double ai = env[bas[pp->ish * BAS_SLOTS + PTR_EXP]], aj = env[bas[pp->jsh * BAS_SLOTS + PTR_EXP]];
        const double ci = env[bas[pp->ish * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(li);
        const double cj = env[bas[pp->jsh * BAS_SLOTS + PTR_COEFF]] * common_fac_sp(lj);
        const double w = (pp->ish == pp->jsh ? 1. : 2.) * ci * cj;
        const int Li = li + 2, Lj = lj + 2, s2 = Li + Lj + 1, s1 = (Lj + 1) * s2, sz = (Li + 1) * s1;
        double *Ex = wk, *Ey = Ex + sz, *Ez = Ey + sz;
        const double r2 = (Ra[0] - Rb[0]) * (Ra[0] - Rb[0]) + (Ra[1] - Rb[1]) * (Ra[1] - Rb[1])
                        + (Ra[2] - Rb[2]) * (Ra[2] - Rb[2]);
        const double kab = exp(-ai * aj / (ai + aj) * r2);
        hermite_E(Ex, Li, Lj, ai, aj, Ra[0], Rb[0], 1.);
        hermite_E(Ey, Li, Lj, ai, aj, Ra[1], Rb[1], 1.);
        hermite_E(Ez, Li, Lj, ai, aj, Ra[2], Rb[2], kab);
        const int nh2 = NH(lij + 2);
        memset(out, 0, sizeof(double) * 27);
        int ix, iy, iz, jx, jy, jz, n;
        const size_t a0 = ao_loc[pp->ish], b0 = ao_loc[pp->jsh];
        int fa = 0;
        for (ix = li; ix >= 0; ix--)
        for (iy = li - ix; iy >= 0; iy--, fa++) {
                iz = li - ix - iy;
                int fb = 0;
                for (jx = lj; jx >= 0; jx--)
                for (jy = lj - jx; jy >= 0; jy--, fb++) {
                        jz = lj - jx - jy;
                        const double d = w * dm[(a0 + fa) * nao + b0 + fb];
                        if (d == 0) continue;
                        const int ia[3] = {ix, iy, iz}, ja[3] = {jx, jy, jz};
                        const double *Es[3] = {Ex, Ey, Ez};
                        /* e[k][type][t]: 0 E, 1 d/dA, 2 d/dB, 3 d2/dA2, 4 d2/dAdB, 5 d2/dB2 */
                        double e[3][6][2 * LMAX + 7];
                        int k, t;
                        for (k = 0; k < 3; k++) {
                                const double *E = Es[k];
                                const int i = ia[k], j = ja[k];
#define EE(p, q) ((p) < 0 || (q) < 0 ? 0. : E[(p) * s1 + (q) * s2 + t])
                                for (t = 0; t <= lij + 2; t++) {
                                        e[k][0][t] = EE(i, j);
                                        e[k][1][t] = 2 * ai * EE(i + 1, j) - i * EE(i - 1, j);
                                        e[k][2][t] = 2 * aj * EE(i, j + 1) - j * EE(i, j - 1);
                                        e[k][3][t] = 4 * ai * ai * EE(i + 2, j) - 2 * ai * (2 * i + 1) * EE(i, j)
                                                   + i * (i - 1) * EE(i - 2, j);
                                        e[k][4][t] = 4 * ai * aj * EE(i + 1, j + 1) - 2 * ai * j * EE(i + 1, j - 1)
                                                   - 2 * aj * i * EE(i - 1, j + 1) + i * j * EE(i - 1, j - 1);
                                        e[k][5][t] = 4 * aj * aj * EE(i, j + 2) - 2 * aj * (2 * j + 1) * EE(i, j)
                                                   + j * (j - 1) * EE(i, j - 2);
                                }
#undef EE
                        }
                        for (n = 0; n < nh2; n++) {
                                const int tt[3] = {h_t[n], h_u[n], h_v[n]};
                                const double vn = d * V[n];
                                if (vn == 0) continue;
                                int blk, x, y;
                                for (blk = 0; blk < 3; blk++) {
                                        /* the derivative types of X and Y */
                                        const int tX = blk == 2 ? 2 : 1, tY = blk == 0 ? 1 : 2;
                                        const int tXY = blk == 0 ? 3 : (blk == 1 ? 4 : 5);
                                        for (x = 0; x < 3; x++)
                                        for (y = 0; y < 3; y++) {
                                                double pr = 1;
                                                for (k = 0; k < 3; k++) {
                                                        int ty;
                                                        if (x == y) ty = k == x ? tXY : 0;
                                                        else ty = k == x ? tX : (k == y ? tY : 0);
                                                        pr *= e[k][ty][tt[k]];
                                                }
                                                out[(blk * 3 + x) * 3 + y] += vn * pr;
                                        }
                                }
                        }
                }
        }
}

/*
 * The second derivatives of the cross energy E_x = sum_ij D_ij (ij|d~) (the three-centre part of the
 * RIJ Hessian, d . rho^{xi eta}), the pairs as SOCVHFmd_rij_grad:
 *   d2 E_x = sum_P <d2 rho^P, V^P> + sum_Q <d2 rho^Q, V^Q> + sum_PQ (<d rho^P, T d rho^Q> + c.c.)
 * with V^P (V^Q) to degree l_P + 2 (l_Q + 2).  hess (natm, natm, 3, 3) +=.  Returns 0 if a pair's
 * degree exceeds the generated contractions (the caller keeps the libcint pass).
 */
int SOCVHFmd_rij_hess(double *hess, double *dm, double tol, int nbra, int naux, int *s0sh,
                      int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        if (!h_ready) hermite_tables();
        const size_t nao = ao_loc[nbas];
        const int nbp = nbra * (nbra + 1) / 2, npair = nbp + naux;
        int i;
        for (i = 0; i < nbas; i++)
                if (bas[i * BAS_SLOTS + ANG_OF] > LMAX || bas[i * BAS_SLOTS + NPRIM_OF] != 1
                    || bas[i * BAS_SLOTS + NCTR_OF] != 1) return 0;
        MDPair *pairs = malloc(sizeof(MDPair) * npair);
        size_t orho = 0, oder = 0, ov = 0;
        int lmb = 0, lma = 0;
        {
                int a, b, n;
                for (n = 0; n < npair; n++) {
                        if (n < nbp) {
                                a = (int)((sqrt(1. + 8. * n) - 1) / 2);
                                while (a * (a + 1) / 2 > n) a--;
                                while ((a + 1) * (a + 2) / 2 <= n) a++;
                                b = n - a * (a + 1) / 2;
                        } else {
                                a = nbra + (n - nbp);
                                b = s0sh[n - nbp];
                        }
                        MDPair *pp = pairs + n;
                        pp->ish = a; pp->jsh = b;
                        pp->li = bas[a * BAS_SLOTS + ANG_OF];
                        pp->lj = bas[b * BAS_SLOTS + ANG_OF];
                        pp->ia = bas[a * BAS_SLOTS + ATOM_OF];
                        pp->ja = bas[b * BAS_SLOTS + ATOM_OF];
                        const double ai = env[bas[a * BAS_SLOTS + PTR_EXP]], aj = env[bas[b * BAS_SLOTS + PTR_EXP]];
                        const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                        const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                        int x;
                        pp->p = ai + aj;
                        for (x = 0; x < 3; x++) pp->P[x] = (ai * Ra[x] + aj * Rb[x]) / pp->p;
                        const int lij = pp->li + pp->lj;
                        if (n < nbp) lmb = lij > lmb ? lij : lmb; else lma = lij > lma ? lij : lma;
                        pp->orho = orho; orho += NH(lij);
                        pp->oder = oder; oder += 6 * NH(lij + 1);
                        pp->ov = ov; ov += NH(lij + 2);
                }
        }
        if (lmb + 2 > MDC_LMAX || lma + 2 > MDC_LMAX || lmb + lma + 2 > 12) {
                free(pairs);
                return 0;
        }
        double *rho = malloc(sizeof(double) * orho);
        double *der = malloc(sizeof(double) * oder);
        double *bound = malloc(sizeof(double) * 2 * npair);
        const int nthreads = omp_get_max_threads();
        double **vbufs = malloc(sizeof(double *) * nthreads);
        int *order = malloc(sizeof(int) * npair);
        const size_t nh9 = (size_t)natm * natm * 9;
        int nkb = 0, nka = 0;
#pragma omp parallel num_threads(nthreads)
{
        const int tid = omp_get_thread_num();
        const int sE = (LMAX + 3) * (LMAX + 3) * (2 * LMAX + 7);
        double *wk = malloc(sizeof(double) * (3 * sE + (DMAX + 2) * NHMAX));
        double *R0 = malloc(sizeof(double) * NHMAX);
        double *wR = wk + 3 * sE;
        double *hl = calloc(nh9, sizeof(double));
        int n;
#pragma omp for schedule(dynamic, 16)
        for (n = 0; n < npair; n++) {
                MDPair *pp = pairs + n;
                const double *Ra = env + atm[pp->ia * ATM_SLOTS + PTR_COORD];
                const double *Rb = env + atm[pp->ja * ATM_SLOTS + PTR_COORD];
                pair_densities(rho + pp->orho, der + pp->oder, pp, Ra, Rb, dm, nao, ao_loc, bas, env, wk);
                const int lij = pp->li + pp->lj;
                const double pref = PI_FAC / (pp->p * pp->p * sqrt(2 * pp->p));
                hermite_R(R0, wR, 2 * lij + 2, pref, .5 * pp->p, 0, 0, 0);
                double s0 = hermite_quad(rho + pp->orho, lij, rho + pp->orho, lij, R0);
                double s1 = 0;
                int x;
                for (x = 0; x < 6; x++) {
                        const double *dx = der + pp->oder + (size_t)x * NH(lij + 1);
                        double v = hermite_quad(dx, lij + 1, dx, lij + 1, R0);
                        if (v > s1) s1 = v;
                }
                s0 = sqrt(fabs(s0));
                s1 = sqrt(fabs(s1));
                bound[2 * n] = s0 > s1 ? s0 : s1;
                bound[2 * n + 1] = n;
        }
#pragma omp single
{
        qsort(bound, nbp, 2 * sizeof(double), cmp_desc);
        qsort(bound + 2 * nbp, naux, 2 * sizeof(double), cmp_desc);
        const double bbmax = nbp ? bound[0] : 0, bamax = naux ? bound[2 * nbp] : 0;
        for (n = 0; n < nbp; n++) {
                if (2 * bound[2 * n] * bamax < tol) break;
                order[n] = (int)bound[2 * n + 1];
        }
        nkb = n;
        for (n = 0; n < naux; n++) {
                if (2 * bound[2 * (nbp + n)] * bbmax < tol) break;
                order[nbp + n] = (int)bound[2 * (nbp + n) + 1];
        }
        nka = n;
}
        double *V = vbufs[tid] = calloc(ov, sizeof(double));
        int ip;
#pragma omp for schedule(dynamic, 1)
        for (ip = 0; ip < nkb; ip++) {
                const MDPair *P = pairs + order[ip];
                const int lp = P->li + P->lj, nhp1 = NH(lp + 1);
                const double bp = bound[2 * ip];
                const double *rp = rho + P->orho, *dp = der + P->oder;
                double *vp = V + P->ov;
                int iq;
                for (iq = 0; iq < nka; iq++) {
                        if (2 * bp * bound[2 * (nbp + iq)] < tol) break;
                        const MDPair *Q = pairs + order[nbp + iq];
                        const int lq = Q->li + Q->lj, nhq1 = NH(lq + 1);
                        const double *rq = rho + Q->orho, *dq = der + Q->oder;
                        double *vq = V + Q->ov;
                        const double pq = P->p * Q->p, psum = P->p + Q->p;
                        mdr_table[lp + lq + 2](R0, PI_FAC / (pq * sqrt(psum)), pq / psum,
                                               P->P[0] - Q->P[0], P->P[1] - Q->P[1], P->P[2] - Q->P[2]);
                        mdc_table[lp + 2][lq](R0, NULL, rq, vp, NULL, 1, 0);
                        mdc_table[lp][lq + 2](R0, rp, NULL, NULL, vq, 0, 1);
                        /* <d rho^P_X, T d rho^Q_C>, X = A (P's i centre), B (its j centre); C = Q's centre */
                        int y, x, a;
                        for (y = 0; y < 3; y++) {
                                double G[NHOP];
                                memset(G, 0, sizeof(double) * nhp1);
                                mdc_table[lp + 1][lq + 1](R0, NULL, dq + (size_t)y * nhq1, G, NULL, 1, 0);
                                for (x = 0; x < 3; x++) {
                                        double hA = 0, hB = 0;
                                        const double *dA = dp + (size_t)x * nhp1, *dB = dp + (size_t)(3 + x) * nhp1;
                                        for (a = 0; a < nhp1; a++) {
                                                hA += dA[a] * G[a];
                                                hB += dB[a] * G[a];
                                        }
                                        hl[(((size_t)P->ia * natm + Q->ia) * 3 + x) * 3 + y] += hA;
                                        hl[(((size_t)Q->ia * natm + P->ia) * 3 + y) * 3 + x] += hA;
                                        hl[(((size_t)P->ja * natm + Q->ia) * 3 + x) * 3 + y] += hB;
                                        hl[(((size_t)Q->ia * natm + P->ja) * 3 + y) * 3 + x] += hB;
                                }
                        }
                }
        }
#pragma omp barrier
        {
                const size_t chunk = (ov + nthreads - 1) / nthreads;
                const size_t c0 = chunk * tid < ov ? chunk * tid : ov;
                const size_t c1 = c0 + chunk < ov ? c0 + chunk : ov;
                size_t k;
                int t;
                for (t = 0; t < nthreads; t++) {
                        if (t == tid) continue;
                        const double *vt = vbufs[t];
                        for (k = c0; k < c1; k++) V[k] += vt[k];
                }
                for (t = 0; t < nthreads; t++) {
                        if (t == tid) continue;
                        double *vt = vbufs[t];
                        for (k = c0; k < c1; k++) vt[k] = V[k];
                }
        }
#pragma omp barrier
        /* <d2 rho, V> of the kept pairs: bra AA / AB / BB, auxiliary CC (the s0 derivatives vanish) */
        const int ntot = nkb + nka;
#pragma omp for schedule(dynamic, 8)
        for (ip = 0; ip < ntot; ip++) {
                const MDPair *P = pairs + (ip < nkb ? order[ip] : order[nbp + ip - nkb]);
                const double *Ra = env + atm[P->ia * ATM_SLOTS + PTR_COORD];
                const double *Rb = env + atm[P->ja * ATM_SLOTS + PTR_COORD];
                double o[27];
                pair_d2_dot(o, P, Ra, Rb, V + P->ov, dm, nao, ao_loc, bas, env, wk);
                int x, y;
                for (x = 0; x < 3; x++)
                for (y = 0; y < 3; y++) {
                        hl[(((size_t)P->ia * natm + P->ia) * 3 + x) * 3 + y] += o[x * 3 + y];
                        if (ip < nkb) {
                                hl[(((size_t)P->ia * natm + P->ja) * 3 + x) * 3 + y] += o[9 + x * 3 + y];
                                hl[(((size_t)P->ja * natm + P->ia) * 3 + y) * 3 + x] += o[9 + x * 3 + y];
                                hl[(((size_t)P->ja * natm + P->ja) * 3 + x) * 3 + y] += o[18 + x * 3 + y];
                        }
                }
        }
#pragma omp critical
        {
                size_t k;
                for (k = 0; k < nh9; k++) hess[k] += hl[k];
        }
        free(hl);
        free(wk);
        free(R0);
#pragma omp barrier
        free(V);
}
        free(pairs);
        free(rho);
        free(der);
        free(bound);
        free(vbufs);
        free(order);
        return 1;
}
