/*
 * Nuclear derivatives of the partition weights of a molecular grid (the weight part of the grid
 * response of DFT / COSX gradients), contracted on the fly with a per-point scalar:
 *
 *     grad[B] += sum_g e_g d w_g / d R_B,     w_g = vol_g P_A(r_g) / Z(r_g),  Z = sum_i P_i,
 *
 * A the atom owning point g, r_g = R_A + u_g (the point moves with its atom), for exactly the cell
 * functions of lib/dft/becke.c (SOCDFTbecke_owner, SOCDFTssf_owner) and pyscf's VXCgen_grid:
 *
 *     P_i = prod_{j != i} s(nu_ij),  nu_ij = mu_ij + a_ij (1 - mu_ij^2),  mu_ij = (r_i - r_j) / R_ij,
 *
 * r_i = |r - R_i|, a_ij the (geometry-independent) size adjustment (radii[i natm + j], NULL: none);
 * Becke: s_ij = (1 - f3(nu_ij)) / 2, f3 three iterations of f(x) = (3 - x^2) x / 2, s_ji = 1 - s_ij;
 * SSF (Stratmann, Scuseria, Frisch, CPL 257, 213 (1996)): s = (1 - g(nu / a)) / 2, a = 0.64,
 * g(z) = (35 z - 35 z^3 + 21 z^5 - 5 z^7) / 16 inside |nu| < a, 1 / 0 outside.  The derivatives
 * (Johnson, Gill, Pople, JCP 98, 5612 (1993), eqs. B6-B10, for any cell function):
 *
 *     d w_g / d R_B (point fixed) = w_g [ d ln P_A / d R_B - sum_i (P_i / Z) d ln P_i / d R_B ],
 *     d ln P_i / d nu_ij = s'(nu_ij) / s(nu_ij),   d nu / d mu = 1 - 2 a_ij mu,
 *     d mu_ij / d R_i = -u_i / R_ij - mu_ij (R_i - R_j) / R_ij^2,
 *     d mu_ij / d R_j =  u_j / R_ij + mu_ij (R_i - R_j) / R_ij^2      (u_i = (r - R_i) / r_i),
 *
 * and the motion of the point with A by translational invariance of w_g:
 * d w_g / d R_A (total) = -sum_{B != A} d w_g / d R_B (point fixed).  With c_i = e_g w_g (delta_iA - P_i / Z)
 * each factor s(nu_ij) of P_i contributes c_i s'/s (d nu / d mu) d mu_ij / d R_{i,j}; no natm x ngrid
 * array is formed.  Becke costs natm^2 / 2 per point (as the weights); SSF only the candidate atoms
 * with P_i > 0 times natm (its compact support used exactly, as SOCDFTssf_owner).
 *
 * grad (natm, 3) accumulated (+=); coords (n, 3) C order; weights, eg (n); owner (n, int; < 0 skipped);
 * atm (natm, 3); scheme 0 Becke, 1 SSF.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define SSF_A .64
#define S_TINY 1e-250

/* Becke: s_ij(nu) and ds_ij / dnu */
static inline void becke_s(double nu, double *s, double *ds)
{
        double f = nu, df = 1;
        int it;
        for (it = 0; it < 3; it++) {
                df *= 1.5 * (1 - f * f);
                f = (3 - f * f) * f * .5;
        }
        *s = .5 - .5 * f;
        *ds = -.5 * df;
}

static inline void ssf_sd(double nu, double *s, double *ds)
{
        if (nu <= -SSF_A) { *s = 1; *ds = 0; return; }
        if (nu >= SSF_A) { *s = 0; *ds = 0; return; }
        const double z = nu / SSF_A, z2 = z * z;
        const double g = (1. / 16.) * (z * (35 + z2 * (-35 + z2 * (21 - 5 * z2))));
        const double dg = (1. / 16.) * (35 + z2 * (-105 + z2 * (105 - 35 * z2)));
        *s = .5 * (1 - g);
        *ds = -.5 * dg / SSF_A;
}

/* coefficient k on d mu_ij: gr[i] += k dmu_ij/dR_i, gr[j] += k dmu_ij/dR_j; returns the sum of both
 * (into tot, for the point motion of the owner) */
static inline void add_dmu(double *gr, double *tot, double k, int i, int j, double mu, const double *u,
                           const double *atm, double invR)
{
        const double c1 = k * invR, c2 = k * mu * invR * invR;
        const double dx = atm[i * 3] - atm[j * 3], dy = atm[i * 3 + 1] - atm[j * 3 + 1], dz = atm[i * 3 + 2] - atm[j * 3 + 2];
        const double gi0 = -c1 * u[i * 3] - c2 * dx, gi1 = -c1 * u[i * 3 + 1] - c2 * dy, gi2 = -c1 * u[i * 3 + 2] - c2 * dz;
        const double gj0 = c1 * u[j * 3] + c2 * dx, gj1 = c1 * u[j * 3 + 1] + c2 * dy, gj2 = c1 * u[j * 3 + 2] + c2 * dz;
        gr[i * 3] += gi0; gr[i * 3 + 1] += gi1; gr[i * 3 + 2] += gi2;
        gr[j * 3] += gj0; gr[j * 3 + 1] += gj1; gr[j * 3 + 2] += gj2;
        tot[0] += gi0 + gj0; tot[1] += gi1 + gj1; tot[2] += gi2 + gj2;
}

void SOCDFTpartition_grad(double *grad, const double *coords, const double *weights, const int *owner,
                          const double *eg, int n, const double *atm, const double *radii, int natm, int scheme)
{
        double *inv = malloc(sizeof(double) * natm * natm);
        int i, j;
        for (i = 0; i < natm; i++) {
                inv[i * natm + i] = 0;
                for (j = 0; j < i; j++) {
                        const double dx = atm[i * 3] - atm[j * 3], dy = atm[i * 3 + 1] - atm[j * 3 + 1],
                                     dz = atm[i * 3 + 2] - atm[j * 3 + 2];
                        inv[i * natm + j] = inv[j * natm + i] = 1 / sqrt(dx * dx + dy * dy + dz * dz);
                }
        }
#pragma omp parallel
{
        double *gr = calloc(natm * 3, sizeof(double));
        double *r = malloc(sizeof(double) * natm);
        double *u = malloc(sizeof(double) * natm * 3);
        double *P = malloc(sizeof(double) * natm);
        double *s = scheme == 0 ? malloc(sizeof(double) * natm * natm) : NULL;
        double *ds = scheme == 0 ? malloc(sizeof(double) * natm * natm) : NULL;
        int *cand = malloc(sizeof(int) * natm);
        int g;
#pragma omp for schedule(dynamic, 64)
        for (g = 0; g < n; g++) {
                const int A = owner[g];
                if (A < 0 || weights[g] == 0 || eg[g] == 0) continue;
                const double x = coords[g * 3], y = coords[g * 3 + 1], z = coords[g * 3 + 2];
                const double ew = eg[g] * weights[g];
                int ii, jj, k = 0;
                for (ii = 0; ii < natm; ii++) {
                        const double dx = x - atm[ii * 3], dy = y - atm[ii * 3 + 1], dz = z - atm[ii * 3 + 2];
                        r[ii] = sqrt(dx * dx + dy * dy + dz * dz);
                        const double ir = r[ii] > 1e-14 ? 1 / r[ii] : 0;
                        u[ii * 3] = dx * ir; u[ii * 3 + 1] = dy * ir; u[ii * 3 + 2] = dz * ir;
                        if (r[ii] < r[k]) k = ii;
                }
                double tot[3] = {0, 0, 0};
                if (scheme == 0) {
                        for (ii = 0; ii < natm; ii++) P[ii] = 1;
                        for (ii = 0; ii < natm; ii++)
                                for (jj = 0; jj < ii; jj++) {
                                        double mu = (r[ii] - r[jj]) * inv[ii * natm + jj];
                                        const double a = radii != NULL ? radii[ii * natm + jj] : 0;
                                        const double nu = mu + a * (1 - mu * mu);
                                        double sv, dv;
                                        becke_s(nu, &sv, &dv);
                                        s[ii * natm + jj] = sv;
                                        ds[ii * natm + jj] = dv * (1 - 2 * a * mu);     /* d s_ij / d mu_ij */
                                        P[ii] *= sv;
                                        P[jj] *= 1 - sv;
                                }
                        double Z = 0;
                        for (ii = 0; ii < natm; ii++) Z += P[ii];
                        const double iZ = 1 / Z;
                        for (ii = 0; ii < natm; ii++) {
                                const double ci = (ii == A) - P[ii] * iZ;      /* c_i P_i / (e w) */
                                for (jj = 0; jj < ii; jj++) {
                                        const double cj = (jj == A) - P[jj] * iZ;
                                        const double sv = s[ii * natm + jj], sj = 1 - sv;
                                        /* P_i holds s_ij, P_j holds s_ji = 1 - s_ij (d s_ji / d mu_ij = -ds) */
                                        double kk = 0;
                                        if (sv > S_TINY) kk += ci / sv;
                                        if (sj > S_TINY) kk -= cj / sj;
                                        kk *= ew * ds[ii * natm + jj];
                                        if (kk == 0) continue;
                                        const double mu = (r[ii] - r[jj]) * inv[ii * natm + jj];
                                        add_dmu(gr, tot, kk, ii, jj, mu, u, atm, inv[ii * natm + jj]);
                                }
                        }
                } else {
                        /* the candidates: P_i > 0 (as SOCDFTssf_owner) */
                        int nc = 0;
                        double Z = 0;
                        for (ii = 0; ii < natm; ii++) {
                                if (ii != k) {
                                        const double mu = (r[ii] - r[k]) * inv[ii * natm + k];
                                        const double nu = radii != NULL ? mu + radii[ii * natm + k] * (1 - mu * mu) : mu;
                                        if (nu >= SSF_A) { P[ii] = 0; continue; }
                                }
                                double p = 1;
                                for (jj = 0; jj < natm && p != 0; jj++) {
                                        if (jj == ii) continue;
                                        const double mu = (r[ii] - r[jj]) * inv[ii * natm + jj];
                                        const double nu = radii != NULL ? mu + radii[ii * natm + jj] * (1 - mu * mu) : mu;
                                        if (nu <= -SSF_A) continue;
                                        double sv, dv;
                                        ssf_sd(nu, &sv, &dv);
                                        p *= sv;
                                }
                                P[ii] = p;
                                if (p != 0) cand[nc++] = ii;
                                Z += p;
                        }
                        if (Z <= 0) continue;
                        const double iZ = 1 / Z;
                        int q;
                        for (q = 0; q < nc; q++) {
                                ii = cand[q];
                                const double ci = ((ii == A) - P[ii] * iZ) * ew;
                                if (ci == 0) continue;
                                for (jj = 0; jj < natm; jj++) {
                                        if (jj == ii) continue;
                                        const double mu = (r[ii] - r[jj]) * inv[ii * natm + jj];
                                        const double a = radii != NULL ? radii[ii * natm + jj] : 0;
                                        const double nu = mu + a * (1 - mu * mu);
                                        if (nu <= -SSF_A || nu >= SSF_A) continue;
                                        double sv, dv;
                                        ssf_sd(nu, &sv, &dv);
                                        if (sv <= S_TINY) continue;
                                        const double kk = ci * dv * (1 - 2 * a * mu) / sv;
                                        add_dmu(gr, tot, kk, ii, jj, mu, u, atm, inv[ii * natm + jj]);
                                }
                        }
                }
                /* the point moves with its owner */
                gr[A * 3] -= tot[0]; gr[A * 3 + 1] -= tot[1]; gr[A * 3 + 2] -= tot[2];
        }
#pragma omp critical(socdft_partition_grad)
        for (i = 0; i < natm * 3; i++) grad[i] += gr[i];
        free(gr); free(r); free(u); free(P); free(s); free(ds); free(cand);
}
        free(inv);
}

/* sum_{g owned by A} v[x][g] into out (natm, 3) (+=); v (3, n) */
void SOCDFTowner_sum(double *out, const double *v, const int *owner, int n, int natm)
{
        int g;
        for (g = 0; g < n; g++) {
                const int A = owner[g];
                if (A < 0 || A >= natm) continue;
                out[A * 3] += v[g];
                out[A * 3 + 1] += v[(size_t)n + g];
                out[A * 3 + 2] += v[2 * (size_t)n + g];
        }
}
