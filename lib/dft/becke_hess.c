/*
 * First derivatives per point and the second derivatives (contracted with a per-point scalar) of the
 * partition weights of a molecular grid whose points move with their owner atoms, for the cell functions
 * of lib/dft/becke.c / becke_deriv.c (Becke, SSF; size adjustments a_ij, antisymmetric):
 *
 *     w_g = vol_g F,  F = P_o / Z,  Z = sum_i P_i,  P_i = prod_{j != i} s(nu_ij),
 *     nu_ij = mu_ij + a_ij (1 - mu_ij^2),  mu_ij = (|p| - |q|) / |d|,  p = r - R_i, q = r - R_j, d = R_i - R_j,
 *     r = R_o + u_g (o the owner).
 *
 * With g_i = d ln P_i = sum_j lambda_ij d nu_ij (lambda = s'/s), pi_i = P_i / Z, gbar = sum_i pi_i g_i and
 * H_i = d^2 ln P_i = sum_j [(s''/s - lambda^2) d nu d nu^T + lambda d^2 nu]:
 *
 *     d w / w = g_o - gbar,
 *     d^2 w / w = (g_o - gbar)(g_o - gbar)^T + H_o - sum_i pi_i H_i - sum_i pi_i g_i g_i^T + gbar gbar^T,
 *     d nu = (1 - 2 a mu) d mu,  d^2 nu = (1 - 2 a mu) d^2 mu - 2 a d mu d mu^T,
 *
 * and d mu, d^2 mu from the gradient / Hessian of mu in (p, q, d) mapped to the atoms (o, i, j) by
 * p = R_o - R_i + u, q = R_o - R_j + u, d = R_i - R_j (p constant when o = i, q when o = j):
 *
 *     f_p = u_i / |d|, f_q = -u_j / |d|, f_d = -mu d / |d|^2,
 *     f_pp = (1 - u_i u_i^T) / (r_i |d|), f_qq = -(1 - u_j u_j^T) / (r_j |d|),
 *     f_dd = -(mu / |d|^2)(1 - 3 d^ d^T), f_pd = -u_i d^T / |d|^3, f_qd = u_j d^T / |d|^3, f_pq = 0.
 *
 * Only the atoms i with pi_i > PI_TINY (and the owner) enter; their g_i are kept on the atoms they touch.
 * SOCDFTpartition_deriv1: w1 (n, natm, 3) = d w_g / d R (dense per point).
 * SOCDFTpartition_hess: hess (natm, natm, 3, 3) += sum_g e_g d^2 w_g / d R d R.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define SSF_A .64
#define S_TINY 1e-250
#define PI_TINY 1e-16

static inline void becke_s2(double nu, double *s, double *ds, double *d2s)
{
        double f = nu, df = 1, d2f = 0;
        int it;
        for (it = 0; it < 3; it++) {
                const double t = 1 - f * f;
                d2f = 1.5 * (t * d2f - 2 * f * df * df);
                df *= 1.5 * t;
                f = (3 - f * f) * f * .5;
        }
        *s = .5 - .5 * f;
        *ds = -.5 * df;
        *d2s = -.5 * d2f;
}

static inline void ssf_s2(double nu, double *s, double *ds, double *d2s)
{
        if (nu <= -SSF_A) { *s = 1; *ds = 0; *d2s = 0; return; }
        if (nu >= SSF_A) { *s = 0; *ds = 0; *d2s = 0; return; }
        const double z = nu / SSF_A, z2 = z * z;
        const double g = (1. / 16.) * (z * (35 + z2 * (-35 + z2 * (21 - 5 * z2))));
        const double dg = (1. / 16.) * (35 + z2 * (-105 + z2 * (105 - 35 * z2)));
        const double d2g = (1. / 16.) * (z * (-210 + z2 * (420 - 210 * z2)));
        *s = .5 * (1 - g);
        *ds = -.5 * dg / SSF_A;
        *d2s = -.5 * d2g / (SSF_A * SSF_A);
}

static inline void cell_s(int scheme, double nu, double *s, double *ds, double *d2s)
{
        if (scheme == 0) becke_s2(nu, s, ds, d2s);
        else ssf_s2(nu, s, ds, d2s);
}

/* the gradient (3 slots x 3: o, i, j) and Hessian (9 x 9) of mu_ij at the point */
static void mu_derivs(double *G, double *H, int o, int i, int j, double mu, const double *ui, const double *uj,
                      double ri, double rj, const double *dvec, double dn, int with_hess)
{
        const double idn = 1 / dn, idn2 = idn * idn, idn3 = idn2 * idn;
        double fp[3], fq[3], fd[3];
        int a, b;
        const int usep = o != i, useq = o != j;
        for (a = 0; a < 3; a++) {
                fp[a] = usep ? ui[a] * idn : 0;
                fq[a] = useq ? -uj[a] * idn : 0;
                fd[a] = -mu * dvec[a] * idn2;
        }
        /* slots: 0 = o, 1 = i, 2 = j;  p = +o -i, q = +o -j, d = +i -j */
        for (a = 0; a < 3; a++) {
                G[a] = fp[a] + fq[a];
                G[3 + a] = -fp[a] + fd[a];
                G[6 + a] = -fq[a] - fd[a];
        }
        if (!with_hess) return;
        double Hpp[9], Hqq[9], Hdd[9], Hpd[9], Hqd[9];
        for (a = 0; a < 3; a++)
                for (b = 0; b < 3; b++) {
                        const double dab = a == b;
                        Hpp[a * 3 + b] = usep && ri > 1e-14 ? (dab - ui[a] * ui[b]) / (ri * dn) : 0;
                        Hqq[a * 3 + b] = useq && rj > 1e-14 ? -(dab - uj[a] * uj[b]) / (rj * dn) : 0;
                        Hdd[a * 3 + b] = -mu * idn2 * (dab - 3 * dvec[a] * dvec[b] * idn2);
                        Hpd[a * 3 + b] = usep ? -ui[a] * dvec[b] * idn3 : 0;
                        Hqd[a * 3 + b] = useq ? uj[a] * dvec[b] * idn3 : 0;
                }
        /* J (3 vars x 3 slots), coefficient of each variable block on each slot */
        static const double Jp[3] = {1, -1, 0}, Jq[3] = {1, 0, -1}, Jd[3] = {0, 1, -1};
        int s1, s2;
        for (s1 = 0; s1 < 3; s1++)
                for (s2 = 0; s2 < 3; s2++)
                        for (a = 0; a < 3; a++)
                                for (b = 0; b < 3; b++) {
                                        const int ab = a * 3 + b, ba = b * 3 + a;
                                        double h = Jp[s1] * Jp[s2] * Hpp[ab] + Jq[s1] * Jq[s2] * Hqq[ab]
                                                 + Jd[s1] * Jd[s2] * Hdd[ab]
                                                 + Jp[s1] * Jd[s2] * Hpd[ab] + Jd[s1] * Jp[s2] * Hpd[ba]
                                                 + Jq[s1] * Jd[s2] * Hqd[ab] + Jd[s1] * Jq[s2] * Hqd[ba];
                                        H[(s1 * 3 + a) * 9 + s2 * 3 + b] = h;
                                }
}

typedef struct {
        double *r, *u, *P, *gv, *gbar, *v;      /* gv: (nI, natm, 3) */
        int *I, *touched, *mark, *U;
} work_t;

/* per point: P, the active atoms I, their g_i (dense rows, on the atoms touched); with hess the H terms
 * accumulated into hess with ew (delta_io - pi_i); returns the number of active atoms (0: skip the point) */
static int point_terms(work_t *W, double *hess, double ew, const double *xyz, int o, const double *atm,
                       const double *inv, const double *radii, int natm, int scheme, int *nU)
{
        double *r = W->r, *u = W->u, *P = W->P;
        int ii, jj, k = 0;
        for (ii = 0; ii < natm; ii++) {
                const double dx = xyz[0] - atm[ii * 3], dy = xyz[1] - atm[ii * 3 + 1], dz = xyz[2] - atm[ii * 3 + 2];
                r[ii] = sqrt(dx * dx + dy * dy + dz * dz);
                const double ir = r[ii] > 1e-14 ? 1 / r[ii] : 0;
                u[ii * 3] = dx * ir; u[ii * 3 + 1] = dy * ir; u[ii * 3 + 2] = dz * ir;
                if (r[ii] < r[k]) k = ii;
        }
        double Z = 0;
        for (ii = 0; ii < natm; ii++) {
                if (scheme == 1 && ii != k) {
                        const double mu = (r[ii] - r[k]) * inv[ii * natm + k];
                        const double nu = radii != NULL ? mu + radii[ii * natm + k] * (1 - mu * mu) : mu;
                        if (nu >= SSF_A) { P[ii] = 0; continue; }
                }
                double p = 1;
                for (jj = 0; jj < natm && p != 0; jj++) {
                        if (jj == ii) continue;
                        const double mu = (r[ii] - r[jj]) * inv[ii * natm + jj];
                        const double nu = radii != NULL ? mu + radii[ii * natm + jj] * (1 - mu * mu) : mu;
                        double sv, dv, d2v;
                        cell_s(scheme, nu, &sv, &dv, &d2v);
                        p *= sv;
                }
                P[ii] = p;
                Z += p;
        }
        if (Z <= 0 || P[o] <= 0) return 0;
        const double iZ = 1 / Z;
        int nI = 0;
        W->I[nI++] = o;
        for (ii = 0; ii < natm; ii++)
                if (ii != o && P[ii] * iZ > PI_TINY) W->I[nI++] = ii;
        int nu_ = 0;
        double G[9], H[81];
        int q;
        for (q = 0; q < nI; q++) {
                ii = W->I[q];
                double *g = W->gv + (size_t)q * natm * 3;
                memset(g, 0, sizeof(double) * natm * 3);
                const double ci = ((ii == o) - P[ii] * iZ) * ew;
                for (jj = 0; jj < natm; jj++) {
                        if (jj == ii) continue;
                        const double R = 1 / inv[ii * natm + jj];
                        const double mu = (r[ii] - r[jj]) / R;
                        const double a = radii != NULL ? radii[ii * natm + jj] : 0;
                        const double nu = mu + a * (1 - mu * mu);
                        double sv, dv, d2v;
                        cell_s(scheme, nu, &sv, &dv, &d2v);
                        if (sv <= S_TINY || (dv == 0 && d2v == 0)) continue;
                        const double lam = dv / sv, hh = d2v / sv - lam * lam;
                        const double dnu = 1 - 2 * a * mu;
                        double dvec[3] = {atm[ii * 3] - atm[jj * 3], atm[ii * 3 + 1] - atm[jj * 3 + 1],
                                          atm[ii * 3 + 2] - atm[jj * 3 + 2]};
                        const int wh = hess != NULL && ci != 0;
                        mu_derivs(G, H, o, ii, jj, mu, u + ii * 3, u + jj * 3, r[ii], r[jj], dvec, R, wh);
                        const int sl[3] = {o, ii, jj};
                        int s1, s2, x, y;
                        for (s1 = 0; s1 < 3; s1++) {
                                for (x = 0; x < 3; x++) g[sl[s1] * 3 + x] += lam * dnu * G[s1 * 3 + x];
                                if (!W->mark[sl[s1]]) { W->mark[sl[s1]] = 1; W->U[nu_++] = sl[s1]; }
                        }
                        if (!wh) continue;
                        /* ci [hh dnu dnu^T + lam d^2 nu],  d^2 nu = dnu d^2 mu - 2 a dmu dmu^T */
                        for (s1 = 0; s1 < 3; s1++)
                                for (s2 = 0; s2 < 3; s2++) {
                                        double *hb = hess + ((size_t)sl[s1] * natm + sl[s2]) * 9;
                                        for (x = 0; x < 3; x++)
                                                for (y = 0; y < 3; y++) {
                                                        const double gg = G[s1 * 3 + x] * G[s2 * 3 + y];
                                                        hb[x * 3 + y] += ci * ((hh * dnu * dnu - 2 * a * lam) * gg
                                                                + lam * dnu * H[(s1 * 3 + x) * 9 + s2 * 3 + y]);
                                                }
                                }
                }
        }
        /* gbar = sum_i pi_i g_i on the union of the atoms touched */
        int t;
        for (t = 0; t < nu_; t++) {
                const int A = W->U[t];
                W->mark[A] = 0;
                double s0 = 0, s1 = 0, s2 = 0;
                for (q = 0; q < nI; q++) {
                        const double pi = P[W->I[q]] * iZ;
                        const double *g = W->gv + (size_t)q * natm * 3 + A * 3;
                        s0 += pi * g[0]; s1 += pi * g[1]; s2 += pi * g[2];
                }
                W->gbar[A * 3] = s0; W->gbar[A * 3 + 1] = s1; W->gbar[A * 3 + 2] = s2;
        }
        *nU = nu_;
        return nI;
}

static void init_inv(double *inv, const double *atm, int natm)
{
        int i, j;
        for (i = 0; i < natm; i++) {
                inv[i * natm + i] = 0;
                for (j = 0; j < i; j++) {
                        const double dx = atm[i * 3] - atm[j * 3], dy = atm[i * 3 + 1] - atm[j * 3 + 1],
                                     dz = atm[i * 3 + 2] - atm[j * 3 + 2];
                        inv[i * natm + j] = inv[j * natm + i] = 1 / sqrt(dx * dx + dy * dy + dz * dz);
                }
        }
}

static void work_alloc(work_t *W, int natm)
{
        W->r = malloc(sizeof(double) * natm);
        W->u = malloc(sizeof(double) * natm * 3);
        W->P = malloc(sizeof(double) * natm);
        W->gv = malloc(sizeof(double) * (size_t)natm * natm * 3);
        W->gbar = calloc(natm * 3, sizeof(double));
        W->v = calloc(natm * 3, sizeof(double));
        W->I = malloc(sizeof(int) * natm);
        W->U = malloc(sizeof(int) * natm);
        W->mark = calloc(natm, sizeof(int));
        W->touched = NULL;
}

static void work_free(work_t *W)
{
        free(W->r); free(W->u); free(W->P); free(W->gv); free(W->gbar); free(W->v);
        free(W->I); free(W->U); free(W->mark);
}

void SOCDFTpartition_deriv1(double *w1, const double *coords, const double *weights, const int *owner, int n,
                            const double *atm, const double *radii, int natm, int scheme)
{
        double *inv = malloc(sizeof(double) * natm * natm);
        init_inv(inv, atm, natm);
#pragma omp parallel
{
        work_t W;
        work_alloc(&W, natm);
        int g;
#pragma omp for schedule(dynamic, 32)
        for (g = 0; g < n; g++) {
                double *out = w1 + (size_t)g * natm * 3;
                memset(out, 0, sizeof(double) * natm * 3);
                const int o = owner[g];
                if (o < 0 || weights[g] == 0) continue;
                int nU;
                const int nI = point_terms(&W, NULL, 0, coords + g * 3, o, atm, inv, radii, natm, scheme, &nU);
                if (nI == 0) continue;
                int t;
                for (t = 0; t < nU; t++) {
                        const int A = W.U[t];
                        int x;
                        for (x = 0; x < 3; x++)
                                out[A * 3 + x] = weights[g] * (W.gv[A * 3 + x] - W.gbar[A * 3 + x]);
                }
        }
        work_free(&W);
}
        free(inv);
}

void SOCDFTpartition_hess(double *hess, const double *coords, const double *weights, const int *owner,
                          const double *eg, int n, const double *atm, const double *radii, int natm, int scheme)
{
        double *inv = malloc(sizeof(double) * natm * natm);
        init_inv(inv, atm, natm);
        const size_t nh = (size_t)natm * natm * 9;
#pragma omp parallel
{
        work_t W;
        work_alloc(&W, natm);
        double *hl = calloc(nh, sizeof(double));
        double *dg = malloc(sizeof(double) * natm * 3);
        int g;
#pragma omp for schedule(dynamic, 32)
        for (g = 0; g < n; g++) {
                const int o = owner[g];
                if (o < 0 || weights[g] == 0 || eg[g] == 0) continue;
                const double ew = eg[g] * weights[g];
                int nU;
                const int nI = point_terms(&W, hl, ew, coords + g * 3, o, atm, inv, radii, natm, scheme, &nU);
                if (nI == 0) continue;
                double Z = 0;
                int q, t1, t2, x, y;
                for (q = 0; q < natm; q++) Z += W.P[q];
                /* (g_o - gbar)(g_o - gbar)^T - sum_i pi_i g_i g_i^T + gbar gbar^T on the touched atoms */
                for (t1 = 0; t1 < nU; t1++) {
                        const int A = W.U[t1];
                        for (x = 0; x < 3; x++) dg[A * 3 + x] = W.gv[A * 3 + x] - W.gbar[A * 3 + x];
                }
                for (t1 = 0; t1 < nU; t1++) {
                        const int A = W.U[t1];
                        for (t2 = 0; t2 < nU; t2++) {
                                const int B = W.U[t2];
                                double *hb = hl + ((size_t)A * natm + B) * 9;
                                for (x = 0; x < 3; x++)
                                        for (y = 0; y < 3; y++) {
                                                double h = dg[A * 3 + x] * dg[B * 3 + y]
                                                         + W.gbar[A * 3 + x] * W.gbar[B * 3 + y];
                                                for (q = 0; q < nI; q++) {
                                                        const double pi = W.P[W.I[q]] / Z;
                                                        const double *gq = W.gv + (size_t)q * natm * 3;
                                                        h -= pi * gq[A * 3 + x] * gq[B * 3 + y];
                                                }
                                                hb[x * 3 + y] += ew * h;
                                        }
                        }
                }
        }
#pragma omp critical(socdft_partition_hess)
        {
                size_t i;
                for (i = 0; i < nh; i++) hess[i] += hl[i];
        }
        free(hl); free(dg);
        work_free(&W);
}
        free(inv);
}
