/*
 * (ss|ss) shell quartets for the J / K builds of jk_multi.c: no recursions,
 *
 *   (ab|cd) = sum_prim c_a c_b c_c c_d  2 pi^(5/2) K_ab K_cd / (p q sqrt(p + q)) F_0(T),
 *   T = p q / (p + q) |P - Q|^2,
 *
 * with the primitive pairs of every pair of s shells tabulated once per build
 * (p, P, K_ab / p, the coefficient products of the contraction pairs; pairs
 * with K_ab below SS_KCUT dropped).  Per shell quartet: the ket primitive
 * pairs outside, a vector over the bra primitive pairs inside, the bra
 * contraction per ket pair, the ket contraction into the result.  General
 * contraction as libcint; out in int2e_sph's layout (i fastest).  F_0 from a
 * Taylor table (asymptotic beyond SS_TMAX).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"

#define SS_KCUT 1e-24                   /* primitive pairs with exp(-ab/p R^2) below are dropped */
#define SS_STEP 0.05
#define SS_TMAX 40.
#define SS_N ((int)(SS_TMAX / SS_STEP) + 2)
#define SS_ORD 7                        /* Taylor terms: |dt| <= 0.025, error < 1e-14 F_0 */
#define TWO_PI52 34.986836655249725     /* 2 pi^(5/2) */
#define S_FAC4 0.006332573977646111     /* (1 / (2 sqrt(pi)))^4, libcint's s normalisation */

typedef struct {
        int n, nc, ld;                  /* kept primitive pairs, contraction pairs, stride of c */
        double *p, *px, *py, *pz, *kp;  /* p, P, K_ab / p */
        double *c;                      /* c[cb * ld + t], cb = c_j nc_i + c_i */
        double *bnd;                    /* |K_ab / p| max_c |c c| / sqrt(p) */
        double bmax;                    /* max_t |K_ab / p| max_c |c c| */
} SSPair;

typedef struct SSCtx {
        int nbas;
        SSPair *pair;                   /* per shell pair i >= j of two s shells (n = 0 otherwise) */
        double *tab;                    /* F_n(k step), n < SS_ORD */
} SSCtx;

static void f_series(double *f, double t, int m)
{
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

static inline double boys0(const double *tab, double t)
{
        if (t < SS_TMAX) {
                const int k = (int)(t / SS_STEP + .5);
                const double dt = k * SS_STEP - t;
                const double *f = tab + k * SS_ORD;
                /* sum_j F_j(t_k) dt^j / j! (Horner) */
                double s = f[6] * (dt / 6);
                s = (f[5] + s) * (dt / 5);
                s = (f[4] + s) * (dt / 4);
                s = (f[3] + s) * (dt / 3);
                s = (f[2] + s) * (dt / 2);
                s = (f[1] + s) * dt;
                return f[0] + s;
        }
        return .886226925452758014 / sqrt(t);
}

static inline void shell_pair(int ij, int *i, int *j)
{
        int a = (int)((sqrt(1. + 8. * ij) - 1) / 2);
        while (a * (a + 1) / 2 > ij) a--;
        while ((a + 1) * (a + 2) / 2 <= ij) a++;
        *i = a;
        *j = ij - a * (a + 1) / 2;
}

SSCtx *SOCVHFss_ctx_build(int *atm, int natm, int *bas, int nbas, double *env)
{
        SSCtx *cx = malloc(sizeof(SSCtx));
        const int npair = nbas * (nbas + 1) / 2;
        int k, pq;
        cx->nbas = nbas;
        cx->tab = malloc(sizeof(double) * SS_N * SS_ORD);
        for (k = 0; k < SS_N; k++) f_series(cx->tab + k * SS_ORD, k * SS_STEP, SS_ORD - 1);
        cx->pair = calloc(npair, sizeof(SSPair));
#pragma omp parallel for schedule(dynamic, 16)
        for (pq = 0; pq < npair; pq++) {
                int i, j;
                shell_pair(pq, &i, &j);
                const int *bi = bas + i * BAS_SLOTS, *bj = bas + j * BAS_SLOTS;
                if (bi[ANG_OF] != 0 || bj[ANG_OF] != 0) continue;
                const int npi = bi[NPRIM_OF], npj = bj[NPRIM_OF];
                const int nci = bi[NCTR_OF], ncj = bj[NCTR_OF];
                const double *ei = env + bi[PTR_EXP], *ej = env + bj[PTR_EXP];
                const double *ci = env + bi[PTR_COEFF], *cj = env + bj[PTR_COEFF];
                const double *Ri = env + atm[bi[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double *Rj = env + atm[bj[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double r2 = (Ri[0] - Rj[0]) * (Ri[0] - Rj[0]) + (Ri[1] - Rj[1]) * (Ri[1] - Rj[1])
                                + (Ri[2] - Rj[2]) * (Ri[2] - Rj[2]);
                SSPair *P = cx->pair + pq;
                const int nmax = npi * npj, nc = nci * ncj;
                P->nc = nc;
                P->ld = nmax;
                P->p = malloc(sizeof(double) * nmax * (6 + nc));
                P->px = P->p + nmax; P->py = P->px + nmax; P->pz = P->py + nmax;
                P->kp = P->pz + nmax; P->bnd = P->kp + nmax; P->c = P->bnd + nmax;
                int pi, pj, a, b, t = 0;
                double bmax = 0;
                for (pj = 0; pj < npj; pj++)
                for (pi = 0; pi < npi; pi++) {
                        const double ai = ei[pi], aj = ej[pj], p = ai + aj, ip = 1. / p;
                        const double e = exp(-ai * aj * ip * r2);
                        if (e < SS_KCUT) continue;
                        double cm = 0;
                        for (b = 0; b < ncj; b++)
                                for (a = 0; a < nci; a++)
                                        cm = fmax(cm, fabs(ci[a * npi + pi] * cj[b * npj + pj]));
                        P->p[t] = p;
                        P->px[t] = (ai * Ri[0] + aj * Rj[0]) * ip;
                        P->py[t] = (ai * Ri[1] + aj * Rj[1]) * ip;
                        P->pz[t] = (ai * Ri[2] + aj * Rj[2]) * ip;
                        P->kp[t] = e * ip;
                        P->bnd[t] = e * ip * cm / sqrt(p);
                        bmax = fmax(bmax, e * ip * cm);
                        for (b = 0; b < ncj; b++)
                                for (a = 0; a < nci; a++)
                                        P->c[(b * nci + a) * nmax + t] = ci[a * npi + pi] * cj[b * npj + pj];
                        t++;
                }
                P->n = t;
                P->bmax = bmax;
        }
        return cx;
}

void SOCVHFss_ctx_free(SSCtx *cx)
{
        if (!cx) return;
        const int npair = cx->nbas * (cx->nbas + 1) / 2;
        int pq;
        for (pq = 0; pq < npair; pq++) free(cx->pair[pq].p);
        free(cx->pair);
        free(cx->tab);
        free(cx);
}

/* doubles of workspace for SOCVHFss_int2e: the bra vector and the bra contraction */
size_t SOCVHFss_wksize(int *bas, int nbas)
{
        int i, np = 1, nc = 1;
        for (i = 0; i < nbas; i++) {
                if (bas[i * BAS_SLOTS + ANG_OF] != 0) continue;
                if (bas[i * BAS_SLOTS + NPRIM_OF] > np) np = bas[i * BAS_SLOTS + NPRIM_OF];
                if (bas[i * BAS_SLOTS + NCTR_OF] > nc) nc = bas[i * BAS_SLOTS + NCTR_OF];
        }
        return (size_t)np * np + (size_t)nc * nc;
}

/*
 * out (nc_l, nc_k, nc_j, nc_i) of the s shell quartet shls (i >= j, k >= l);
 * ket primitive pairs whose bound times dmax is below prim_tol are skipped;
 * returns 0 when all were skipped (out not set).
 */
int SOCVHFss_int2e(double *out, const int *shls, const SSCtx *cx, double dmax, double prim_tol, double *wk)
{
        const int ij = shls[0] * (shls[0] + 1) / 2 + shls[1];
        const int kl = shls[2] * (shls[2] + 1) / 2 + shls[3];
        const SSPair *B = cx->pair + ij, *K = cx->pair + kl;
        const int nb = B->n, nk = K->n, ncb = B->nc, nck = K->nc;
        if (nb == 0 || nk == 0) return 0;
        double *v = wk, *w = wk + nb;
        const double *tab = cx->tab;
        const double cut = prim_tol / (TWO_PI52 * S_FAC4 * B->bmax * dmax * ncb * nck);
        int t, b, cb, kc, any = 0;
        memset(out, 0, sizeof(double) * ncb * nck);
        for (t = 0; t < nk; t++) {
                if (K->bnd[t] < cut) continue;
                any = 1;
                const double q = K->p[t], qx = K->px[t], qy = K->py[t], qz = K->pz[t];
                for (b = 0; b < nb; b++) {
                        const double p = B->p[b], s = 1. / (p + q);
                        const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;
                        const double T = p * q * s * (dx * dx + dy * dy + dz * dz);
                        v[b] = B->kp[b] * sqrt(s) * boys0(tab, T);
                }
                for (cb = 0; cb < ncb; cb++) {
                        const double *c = B->c + (size_t)cb * B->ld;
                        double x = 0;
                        for (b = 0; b < nb; b++) x += c[b] * v[b];
                        w[cb] = x;
                }
                const double kq = K->kp[t] * TWO_PI52 * S_FAC4;
                for (kc = 0; kc < nck; kc++) {
                        const double f = K->c[(size_t)kc * K->ld + t] * kq;
                        double *o = out + kc * ncb;
                        for (cb = 0; cb < ncb; cb++) o[cb] += f * w[cb];
                }
        }
        return any;
}
