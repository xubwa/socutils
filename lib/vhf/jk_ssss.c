/*
 * Low-l shell quartets for the J / K builds of jk_multi.c without libcint:
 * (ss|ss), the quartets with one p shell, (ps|ss), (sp|ss), (ss|ps), (ss|sp),
 * and with two, (pp|ss), (ss|pp), (ps|ps) and its orientations (two_p below),
 * by Obara-Saika in closed form,
 *
 *   (ss|ss)    = sum_prim c c c c  G F_0(T),   G = 2 pi^(5/2) K_ab K_cd / (p q sqrt(p + q)),
 *   (p_x s|ss) = sum_prim c c c c  G (PA_x F_0(T) + WP_x F_1(T)),   WP = q (Q - P) / (p + q),
 *   (ss|p_x s) = sum_prim c c c c  G (QC_x F_0(T) + WQ_x F_1(T)),   WQ = p (P - Q) / (p + q),
 *
 * T = p q / (p + q) |P - Q|^2, PA = P - A with A the centre of the p shell (B
 * for (sp|ss), likewise for the ket).  The primitive pairs of every shell pair
 * of s / p shells are tabulated once per build (p, P, K_ab / p, P - R_i, P - R_j,
 * the coefficient products of the contraction pairs; pairs with K_ab below
 * SS_KCUT dropped).  Per shell quartet: the ket primitive pairs outside
 * (screened by their bounds), a vector over the bra primitive pairs inside,
 * the bra contraction per ket pair, the ket contraction into the result.
 * General contraction as libcint; out in int2e_sph's layout (i fastest; s / p
 * normalisation as libcint's common factors).  F_0 .. F_2 from Taylor tables
 * (asymptotic beyond SS_TMAX).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"

/* primitive pairs with exp(-ab/p R^2) below are dropped (0: none, for tests) */
static double SS_KCUT = 1e-24;
void SOCVHFss_set_kcut(double c) { SS_KCUT = c; }
#define SS_STEP 0.05
#define SS_TMAX 40.
#define SS_N ((int)(SS_TMAX / SS_STEP) + 2)
#define SS_ORD 7                        /* Taylor terms: |dt| <= 0.025, error < 1e-14 F_m */
#define SS_TN (SS_ORD + 2)              /* F_0 .. F_(SS_ORD+1) per grid point (Taylor of F_0 .. F_2) */
#define TWO_PI52 34.986836655249725     /* 2 pi^(5/2) */
#define FAC_S 0.282094791773878143      /* libcint's common factors of s, p */
#define FAC_P 0.488602511902919921

typedef struct {
        int n, nc, ld;                  /* kept primitive pairs, contraction pairs, stride of c */
        int nci, ncj;                   /* contractions of shells i, j */
        int lsum, pside;                /* l_i + l_j; the p shell: 0 (i) or 1 (j) */
        double *p, *px, *py, *pz, *kp;  /* p, P, K_ab / p */
        double *ax, *ay, *az;           /* P - R_p (one p shell), P - R_i (two) */
        double *bx, *by, *bz;           /* P - R_j */
        double *c;                      /* c[cb * ld + t], cb = c_j nc_i + c_i */
        double *bnd;                    /* bound of the pair's primitive t (with the ket / bra factor) */
        double bmax;                    /* max_t of the bra bound */
} SSPair;

typedef struct SSCtx {
        int nbas;
        SSPair *pair;                   /* per shell pair i >= j with l_i + l_j <= 1 (n = 0 otherwise) */
        double *tab;                    /* F_n(k step), n < SS_TN */
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

/* sum_j f[j] dt^j / j!, j < SS_ORD (Horner) */
static inline double taylor7(const double *f, double dt)
{
        double s = f[6] * (dt / 6);
        s = (f[5] + s) * (dt / 5);
        s = (f[4] + s) * (dt / 4);
        s = (f[3] + s) * (dt / 3);
        s = (f[2] + s) * (dt / 2);
        s = (f[1] + s) * dt;
        return f[0] + s;
}

static inline double boys0(const double *tab, double t)
{
        if (t < SS_TMAX) {
                const int k = (int)(t / SS_STEP + .5);
                return taylor7(tab + k * SS_TN, k * SS_STEP - t);
        }
        return .886226925452758014 / sqrt(t);
}

static inline void boys01(const double *tab, double t, double *f0, double *f1)
{
        if (t < SS_TMAX) {
                const int k = (int)(t / SS_STEP + .5);
                const double dt = k * SS_STEP - t, *f = tab + k * SS_TN;
                *f0 = taylor7(f, dt);
                *f1 = taylor7(f + 1, dt);
        } else {
                *f0 = .886226925452758014 / sqrt(t);
                *f1 = *f0 * .5 / t;
        }
}

static inline void boys012(const double *tab, double t, double *f0, double *f1, double *f2)
{
        if (t < SS_TMAX) {
                const int k = (int)(t / SS_STEP + .5);
                const double dt = k * SS_STEP - t, *f = tab + k * SS_TN;
                *f0 = taylor7(f, dt);
                *f1 = taylor7(f + 1, dt);
                *f2 = taylor7(f + 2, dt);
        } else {
                const double h = .5 / t;
                *f0 = .886226925452758014 / sqrt(t);
                *f1 = *f0 * h;
                *f2 = *f1 * 3 * h;
        }
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
        cx->tab = malloc(sizeof(double) * SS_N * SS_TN);
        for (k = 0; k < SS_N; k++) f_series(cx->tab + k * SS_TN, k * SS_STEP, SS_TN - 1);
        cx->pair = calloc(npair, sizeof(SSPair));
#pragma omp parallel for schedule(dynamic, 16)
        for (pq = 0; pq < npair; pq++) {
                int i, j;
                shell_pair(pq, &i, &j);
                const int *bi = bas + i * BAS_SLOTS, *bj = bas + j * BAS_SLOTS;
                const int lsum = bi[ANG_OF] + bj[ANG_OF];
                if (bi[ANG_OF] > 1 || bj[ANG_OF] > 1) continue;
                const int npi = bi[NPRIM_OF], npj = bj[NPRIM_OF];
                const int nci = bi[NCTR_OF], ncj = bj[NCTR_OF];
                const double *ei = env + bi[PTR_EXP], *ej = env + bj[PTR_EXP];
                const double *ci = env + bi[PTR_COEFF], *cj = env + bj[PTR_COEFF];
                const double *Ri = env + atm[bi[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                const double *Rj = env + atm[bj[ATOM_OF] * ATM_SLOTS + PTR_COORD];
                /* a = P - R_p of the p shell (one p), P - R_i (two p); b = P - R_j (two p) */
                const double *Rp = (bj[ANG_OF] == 1 && bi[ANG_OF] == 0) ? Rj : Ri;
                const double r2 = (Ri[0] - Rj[0]) * (Ri[0] - Rj[0]) + (Ri[1] - Rj[1]) * (Ri[1] - Rj[1])
                                + (Ri[2] - Rj[2]) * (Ri[2] - Rj[2]);
                SSPair *P = cx->pair + pq;
                const int nmax = npi * npj, nc = nci * ncj;
                P->nc = nc;
                P->nci = nci;
                P->ncj = ncj;
                P->ld = nmax;
                P->lsum = lsum;
                P->pside = bj[ANG_OF] == 1 && bi[ANG_OF] == 0;
                P->p = malloc(sizeof(double) * nmax * (12 + nc));
                P->px = P->p + nmax; P->py = P->px + nmax; P->pz = P->py + nmax;
                P->kp = P->pz + nmax; P->bnd = P->kp + nmax;
                P->ax = P->bnd + nmax; P->ay = P->ax + nmax; P->az = P->ay + nmax;
                P->bx = P->az + nmax; P->by = P->bx + nmax; P->bz = P->by + nmax;
                P->c = P->bz + nmax;
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
                        P->ax[t] = P->px[t] - Rp[0];
                        P->ay[t] = P->py[t] - Rp[1];
                        P->az[t] = P->pz[t] - Rp[2];
                        P->bx[t] = P->px[t] - Rj[0];
                        P->by[t] = P->py[t] - Rj[1];
                        P->bz[t] = P->pz[t] - Rj[2];
                        P->kp[t] = e * ip;
                        /* |PA F_0 + WP F_1| <= |PA| + sqrt(T) F_1 / sqrt(p) <= |PA| + 1 / sqrt(p);
                         * two p: (|PA| + 1 / sqrt(p)) (|PB| + 1 / sqrt(p)) + 1 / p */
                        const double na = sqrt(P->ax[t] * P->ax[t] + P->ay[t] * P->ay[t] + P->az[t] * P->az[t]);
                        const double nb_ = sqrt(P->bx[t] * P->bx[t] + P->by[t] * P->by[t] + P->bz[t] * P->bz[t]);
                        const double pf = lsum == 0 ? 1. : (lsum == 1 ? na + sqrt(ip) :
                                          (na + sqrt(ip)) * (nb_ + sqrt(ip)) + ip);
                        P->bnd[t] = e * ip * cm * pf / sqrt(p);
                        bmax = fmax(bmax, e * ip * cm * pf);
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

/* doubles of workspace for SOCVHFss_int2e */
size_t SOCVHFss_wksize(int *bas, int nbas)
{
        int i, np = 1, nc = 1;
        for (i = 0; i < nbas; i++) {
                if (bas[i * BAS_SLOTS + ANG_OF] > 1) continue;
                if (bas[i * BAS_SLOTS + NPRIM_OF] > np) np = bas[i * BAS_SLOTS + NPRIM_OF];
                if (bas[i * BAS_SLOTS + NCTR_OF] > nc) nc = bas[i * BAS_SLOTS + NCTR_OF];
        }
        const size_t n2 = (size_t)np * np, c2 = (size_t)nc * nc;
        return 9 * n2 + 9 * c2 + 9 * c2 * c2;             /* v[9][nb], w[ncb][9], acc[nck][ncb][9] */
}

/* 1 if SOCVHFss_int2e takes the quartet: s / p shells, at most two p */
int SOCVHFss_takes(const int *shls, const int *bas)
{
        const int l0 = bas[shls[0] * BAS_SLOTS + ANG_OF], l1 = bas[shls[1] * BAS_SLOTS + ANG_OF];
        const int l2 = bas[shls[2] * BAS_SLOTS + ANG_OF], l3 = bas[shls[3] * BAS_SLOTS + ANG_OF];
        return l0 <= 1 && l1 <= 1 && l2 <= 1 && l3 <= 1 && l0 + l1 + l2 + l3 <= 2;
}

/*
 * Two p shells, per bra primitive b at the ket primitive t, the nine values
 * v[(u * 3 + w) * nb + b] (u on the first p shell in i, j, k, l order, w on
 * the second) of G / (2 pi^(5/2) K_cd / q c..c fac):
 *   (pp|ss): PB_w e0_u + WP_w e1_u + d_uw / (2p) (F_0 - rho / p F_1)
 *   (ss|pp): QD_w e0_u + WQ_w e1_u + d_uw / (2q) (F_0 - rho / q F_1)
 *   (ps|ps): QC_w e0_u + WQ_w e1_u + d_uw / (2 (p + q)) F_1
 * with e_m = PA F_m + WP F_(m+1) (pp, ps-ps) or QC F_m + WQ F_(m+1) (ss|pp).
 */
static void two_p(double *v, const SSPair *B, const SSPair *K, int t, const double *tab)
{
        const int nb = B->n, kind = B->lsum == 2 ? 0 : (K->lsum == 2 ? 1 : 2);
        const double q = K->p[t], qx = K->px[t], qy = K->py[t], qz = K->pz[t];
        const double kax = K->ax[t], kay = K->ay[t], kaz = K->az[t];
        const double kbx = K->bx[t], kby = K->by[t], kbz = K->bz[t];
        int b;
        for (b = 0; b < nb; b++) {
                const double p = B->p[b], s = 1. / (p + q);
                const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;
                const double rho = p * q * s;
                const double T = rho * (dx * dx + dy * dy + dz * dz);
                double f0, f1, f2;
                boys012(tab, T, &f0, &f1, &f2);
                const double g = B->kp[b] * sqrt(s);
                double A[3], W[3], C[3], dd;
                if (kind == 1) {                /* (ss|pp): QC, QD, WQ = p (P - Q) / (p + q) */
                        A[0] = kax; A[1] = kay; A[2] = kaz;
                        C[0] = kbx; C[1] = kby; C[2] = kbz;
                        W[0] = p * s * dx; W[1] = p * s * dy; W[2] = p * s * dz;
                        dd = .5 / q * (f0 - rho / q * f1);
                } else {                        /* PA, WP = q (Q - P) / (p + q) */
                        A[0] = B->ax[b]; A[1] = B->ay[b]; A[2] = B->az[b];
                        W[0] = -q * s * dx; W[1] = -q * s * dy; W[2] = -q * s * dz;
                        if (kind == 0) {        /* (pp|ss): PB */
                                C[0] = B->bx[b]; C[1] = B->by[b]; C[2] = B->bz[b];
                                dd = .5 / p * (f0 - rho / p * f1);
                        } else {                /* (ps|ps): QC of the ket p, WQ */
                                C[0] = kax; C[1] = kay; C[2] = kaz;
                                dd = .5 * s * f1;
                        }
                }
                double Wc[3];                   /* the second index's W: WP (pp), WQ (ps-ps, ss|pp) */
                if (kind == 0) {
                        Wc[0] = W[0]; Wc[1] = W[1]; Wc[2] = W[2];
                } else if (kind == 2) {
                        Wc[0] = p * s * dx; Wc[1] = p * s * dy; Wc[2] = p * s * dz;
                } else {
                        Wc[0] = W[0]; Wc[1] = W[1]; Wc[2] = W[2];
                }
                int u, w;
                for (u = 0; u < 3; u++) {
                        const double e0 = A[u] * f0 + W[u] * f1, e1 = A[u] * f1 + W[u] * f2;
                        for (w = 0; w < 3; w++)
                                v[(u * 3 + w) * nb + b] = g * (C[w] * e0 + Wc[w] * e1 + (u == w ? dd : 0));
                }
        }
}

/*
 * out of the shell quartet shls (i >= j, k >= l; SOCVHFss_takes true) in
 * int2e_sph's layout; ket primitive pairs whose bound times dmax is below
 * prim_tol are skipped; returns 0 when all were skipped (out not set).
 */
int SOCVHFss_int2e(double *out, const int *shls, const SSCtx *cx, double dmax, double prim_tol, double *wk)
{
        const int ij = shls[0] * (shls[0] + 1) / 2 + shls[1];
        const int kl = shls[2] * (shls[2] + 1) / 2 + shls[3];
        const SSPair *B = cx->pair + ij, *K = cx->pair + kl;
        const int nb = B->n, nk = K->n, ncb = B->nc, nck = K->nc;
        if (nb == 0 || nk == 0) return 0;
        const double *tab = cx->tab;
        const int L = B->lsum + K->lsum;
        const double fac = L == 0 ? FAC_S * FAC_S * FAC_S * FAC_S :
                           (L == 1 ? FAC_S * FAC_S * FAC_S * FAC_P : FAC_S * FAC_S * FAC_P * FAC_P);
        const double cut = prim_tol / (TWO_PI52 * fac * B->bmax * dmax * ncb * nck);
        int t, b, cb, kc, x, any = 0;
        if (L == 0) {
                double *v = wk, *w = wk + nb;
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
                                double y = 0;
                                for (b = 0; b < nb; b++) y += c[b] * v[b];
                                w[cb] = y;
                        }
                        const double kq = K->kp[t] * TWO_PI52 * fac;
                        for (kc = 0; kc < nck; kc++) {
                                const double f = K->c[(size_t)kc * K->ld + t] * kq;
                                double *o = out + kc * ncb;
                                for (cb = 0; cb < ncb; cb++) o[cb] += f * w[cb];
                        }
                }
                return any;
        }
        if (L == 2) {
                /* two p shells: acc[(kc * ncb + cb) * 9 + u * 3 + w] */
                double *v = wk, *w9 = v + 9 * nb, *acc = w9 + 9 * ncb;
                memset(acc, 0, sizeof(double) * nck * ncb * 9);
                for (t = 0; t < nk; t++) {
                        if (K->bnd[t] < cut) continue;
                        any = 1;
                        two_p(v, B, K, t, tab);
                        for (cb = 0; cb < ncb; cb++) {
                                const double *c = B->c + (size_t)cb * B->ld;
                                for (x = 0; x < 9; x++) {
                                        const double *vx = v + (size_t)x * nb;
                                        double y = 0;
                                        for (b = 0; b < nb; b++) y += c[b] * vx[b];
                                        w9[cb * 9 + x] = y;
                                }
                        }
                        const double kq = K->kp[t] * TWO_PI52 * fac;
                        for (kc = 0; kc < nck; kc++) {
                                const double f = K->c[(size_t)kc * K->ld + t] * kq;
                                double *o = acc + (size_t)kc * ncb * 9;
                                for (cb = 0; cb < 9 * ncb; cb++) o[cb] += f * w9[cb];
                        }
                }
                if (!any) return 0;
                /* the p shells among i, j, k, l; u on the first, w on the second */
                int isp[4];
                isp[0] = B->lsum == 2 || (B->lsum == 1 && B->pside == 0);
                isp[1] = B->lsum == 2 || (B->lsum == 1 && B->pside == 1);
                isp[2] = K->lsum == 2 || (K->lsum == 1 && K->pside == 0);
                isp[3] = K->lsum == 2 || (K->lsum == 1 && K->pside == 1);
                int slot[4], s_, np_ = 0;
                for (s_ = 0; s_ < 4; s_++) slot[s_] = isp[s_] ? np_++ : -1;
                const int nci = B->nci, ncj = B->ncj, nck_ = K->nci, ncl = K->ncj;
                const size_t DI = nci * (isp[0] ? 3 : 1), DJ = ncj * (isp[1] ? 3 : 1);
                const size_t DK = nck_ * (isp[2] ? 3 : 1);
                int ci, cj, ck, cl, u, w;
                for (cl = 0; cl < ncl; cl++)
                for (ck = 0; ck < nck_; ck++) {
                        kc = cl * nck_ + ck;
                        for (cj = 0; cj < ncj; cj++)
                        for (ci = 0; ci < nci; ci++) {
                                cb = cj * nci + ci;
                                const double *a = acc + ((size_t)kc * ncb + cb) * 9;
                                for (u = 0; u < 3; u++)
                                for (w = 0; w < 3; w++) {
                                        const int comp[2] = {u, w};
                                        const size_t i = isp[0] ? ci * 3 + comp[slot[0]] : ci;
                                        const size_t j = isp[1] ? cj * 3 + comp[slot[1]] : cj;
                                        const size_t k = isp[2] ? ck * 3 + comp[slot[2]] : ck;
                                        const size_t l = isp[3] ? cl * 3 + comp[slot[3]] : cl;
                                        out[((l * DK + k) * DJ + j) * DI + i] = a[u * 3 + w];
                                }
                        }
                }
                return 1;
        }
        /* one p shell: acc[(kc * ncb + cb) * 3 + x] */
        const int pbra = B->lsum == 1;
        double *vx = wk, *vy = vx + nb, *vz = vy + nb, *v0 = vz + nb;
        double *w = v0 + nb, *acc = w + 3 * ncb;
        memset(acc, 0, sizeof(double) * nck * ncb * 3);
        for (t = 0; t < nk; t++) {
                if (K->bnd[t] < cut) continue;
                any = 1;
                const double q = K->p[t], qx = K->px[t], qy = K->py[t], qz = K->pz[t];
                if (pbra) {
                        for (b = 0; b < nb; b++) {
                                const double p = B->p[b], s = 1. / (p + q);
                                const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;
                                const double T = p * q * s * (dx * dx + dy * dy + dz * dz);
                                double f0, f1;
                                boys01(tab, T, &f0, &f1);
                                const double g = B->kp[b] * sqrt(s);
                                const double a0 = g * f0, a1 = g * f1 * q * s;
                                /* PA F_0 + q (Q - P) / (p + q) F_1 */
                                vx[b] = a0 * B->ax[b] - a1 * dx;
                                vy[b] = a0 * B->ay[b] - a1 * dy;
                                vz[b] = a0 * B->az[b] - a1 * dz;
                        }
                        for (cb = 0; cb < ncb; cb++) {
                                const double *c = B->c + (size_t)cb * B->ld;
                                double sx = 0, sy = 0, sz = 0;
                                for (b = 0; b < nb; b++) {
                                        sx += c[b] * vx[b];
                                        sy += c[b] * vy[b];
                                        sz += c[b] * vz[b];
                                }
                                w[cb * 3] = sx; w[cb * 3 + 1] = sy; w[cb * 3 + 2] = sz;
                        }
                } else {
                        const double cx_ = K->ax[t], cy_ = K->ay[t], cz_ = K->az[t];
                        for (b = 0; b < nb; b++) {
                                const double p = B->p[b], s = 1. / (p + q);
                                const double dx = B->px[b] - qx, dy = B->py[b] - qy, dz = B->pz[b] - qz;
                                const double T = p * q * s * (dx * dx + dy * dy + dz * dz);
                                double f0, f1;
                                boys01(tab, T, &f0, &f1);
                                const double g = B->kp[b] * sqrt(s);
                                const double a1 = g * f1 * p * s;
                                /* QC F_0 + p (P - Q) / (p + q) F_1 */
                                v0[b] = g * f0;
                                vx[b] = a1 * dx;
                                vy[b] = a1 * dy;
                                vz[b] = a1 * dz;
                        }
                        for (cb = 0; cb < ncb; cb++) {
                                const double *c = B->c + (size_t)cb * B->ld;
                                double s0 = 0, sx = 0, sy = 0, sz = 0;
                                for (b = 0; b < nb; b++) {
                                        s0 += c[b] * v0[b];
                                        sx += c[b] * vx[b];
                                        sy += c[b] * vy[b];
                                        sz += c[b] * vz[b];
                                }
                                w[cb * 3] = cx_ * s0 + sx;
                                w[cb * 3 + 1] = cy_ * s0 + sy;
                                w[cb * 3 + 2] = cz_ * s0 + sz;
                        }
                }
                const double kq = K->kp[t] * TWO_PI52 * fac;
                for (kc = 0; kc < nck; kc++) {
                        const double f = K->c[(size_t)kc * K->ld + t] * kq;
                        double *o = acc + (size_t)kc * ncb * 3;
                        for (cb = 0; cb < 3 * ncb; cb++) o[cb] += f * w[cb];
                }
        }
        if (!any) return 0;
        /* into int2e_sph's layout: index c * 3 + x on the p shell, c elsewhere */
        const int pi_ = pbra && B->pside == 0, pj_ = pbra && B->pside == 1;
        const int pk_ = !pbra && K->pside == 0, pl_ = !pbra && K->pside == 1;
        const int nci = B->nci, ncj = B->ncj, nck_ = K->nci, ncl = K->ncj;
        const size_t DI = nci * (pi_ ? 3 : 1), DJ = ncj * (pj_ ? 3 : 1), DK = nck_ * (pk_ ? 3 : 1);
        int ci, cj, ck, cl;
        for (cl = 0; cl < ncl; cl++)
        for (ck = 0; ck < nck_; ck++) {
                kc = cl * nck_ + ck;
                for (cj = 0; cj < ncj; cj++)
                for (ci = 0; ci < nci; ci++) {
                        cb = cj * nci + ci;
                        const double *a = acc + ((size_t)kc * ncb + cb) * 3;
                        for (x = 0; x < 3; x++) {
                                const size_t i = pi_ ? ci * 3 + x : ci, j = pj_ ? cj * 3 + x : cj;
                                const size_t k = pk_ ? ck * 3 + x : ck, l = pl_ ? cl * 3 + x : cl;
                                out[((l * DK + k) * DJ + j) * DI + i] = a[x];
                        }
                }
        }
        return 1;
}
