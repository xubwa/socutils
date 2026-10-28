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

/* f[n] = F_n(t), n = 0..m: for t < m + 10 the series of F_m,
 * F_m(t) = e^-t sum_k (2t)^k / ((2m+1)(2m+3)...(2m+2k+1)), and downward
 * recursion; otherwise F_0 from erf and upward recursion (stable for t > m) */
static void boys_fn(double *f, double t, int m)
{
        int n;
        const double et = exp(-t);
        if (t < m + 10) {
                double term = 1. / (2 * m + 1), sum = term;
                int k;
                for (k = 1; k < 200; k++) {
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
